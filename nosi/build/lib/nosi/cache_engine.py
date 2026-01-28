import torch
from queue import Queue
import time
from .flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
from .flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
from torch.utils.cpp_extension import load
from torch.cuda import nvtx
import argparse
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import CacheLayerMixin, DynamicLayer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import torch.nn.functional as F
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

diff = load(
    name="diff_offload",
    sources=[os.path.join(this_dir, "flash_cache_engine/diff_offload.cpp"), os.path.join(this_dir, "flash_cache_engine/diff_offload_kernel.cu")],
    verbose=False,
)

class CacheEngine:
    def __init__(self,
        gpu_mem_usage: int = 60,
        num_layers: int = 28,
        block_size: int = 64,
        head_num: int = 2,
        head_dim: int = 128,
        cpu_gpu_mem_size_fraction: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 1024,
        max_seq_len: int = 131072,
        topk: int = 64,
        has_kv_bias: bool = False,
    ):
        single_block_size = block_size * head_num * head_dim * (2 if dtype in [torch.float16, torch.bfloat16] else 4)
        self.block_num = gpu_mem_usage * 1024**3 // (single_block_size * num_layers * 2)
        self.block_num_cpu = self.block_num * cpu_gpu_mem_size_fraction
        self.block_size = block_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.max_batch_size = max_batch_size

        self._k_cpu = None
        self._v_cpu = None
        self._k_gpu = None
        self._v_gpu = None


        self.seq_length = 0

        self.total_load_all = 0
        self.total_offload_all = 0
        self.total_update_time = 0
        self.is_first_decode = True
        self.max_seq_len = max_seq_len
        self.topk = topk
        self.max_gen_len = 8192

        self.decode_update = self.decode_update_has_kv_bias if has_kv_bias else self.decode_update_no_kv_bias
        
        
    def prefill_update(self, key_states, value_states, kv_bias, current_batch_pos, total_bsz):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # return: None
        B, S, H, D = key_states.shape

        assert H == self.head_num and D == self.head_dim

        num_full_blocks = S // self.block_size
        tail_len = S % self.block_size
        self.seq_length = S


        if current_batch_pos == 0:
            # A. 开空间: CPU, GPU, block_map
            # 1. 在cpu上开空间，直接开到self.max_seq_len
            self._k_cpu = torch.empty((total_bsz, S+self.max_gen_len, H, D), dtype=self.dtype, device='cpu').pin_memory()
            self._v_cpu = torch.empty((total_bsz, S+self.max_gen_len, H, D), dtype=self.dtype, device='cpu').pin_memory()

            # 2. 在gpu上开空间，开topk * block_size大小
            self._k_gpu = torch.empty((total_bsz, self.topk * self.block_size, H, D), dtype=self.dtype, device=self.device)
            self._v_gpu = torch.empty((total_bsz, self.topk * self.block_size, H, D), dtype=self.dtype, device=self.device)
            self._kv_bias_gpu = torch.empty((total_bsz, self.topk * self.block_size, H), dtype=self.dtype, device=self.device)

            # 3. block_map
            self._block_map = torch.full((H, total_bsz, self.topk), -1, dtype=torch.int64, device=self.device)
            # 开好中间buffer方便外面用cuda graph包起来
            self._new_block_map_buf = torch.empty_like(self._block_map) 
            self._load_mask = torch.empty_like(self._new_block_map_buf)

            # 4. 直接把尾块在GPU上设置好
            self._tail_block_len_on_gpu = S % self.block_size

        if self._tail_block_len_on_gpu > 0: # 如果有尾块，搬到gpu上
            self._tail_block_idx_on_gpu = self.topk - 1 # 放在最后一个位置，方便用seq_len来
            _tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size

            self._k_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :, :].copy_(key_states[:, -self._tail_block_len_on_gpu:, :, :], non_blocking=True)

            self._v_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :, :].copy_(value_states[:, -self._tail_block_len_on_gpu:, :, :], non_blocking=True)

            self._kv_bias_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :].copy_(kv_bias[:, -self._tail_block_len_on_gpu:, :], non_blocking=True)

            self._block_map[:, current_batch_pos:current_batch_pos+B, self._tail_block_idx_on_gpu] = S // self.block_size
        else: # 如果没有尾块，在gpu上开一个块作为尾块，预备下一次decode写入
            self._tail_block_idx_on_gpu = self.topk - 1 # 放在最后一个位置，方便用seq_len来
            self._block_map[:, current_batch_pos:current_batch_pos+B, self._tail_block_idx_on_gpu] = S // self.block_size

        if current_batch_pos == 0:
            self._cache_lens = torch.full((total_bsz,), (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu, dtype=torch.int32, device=self.device)        

        # B. 把kv cache写到cpu
        self._k_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(key_states, non_blocking=True)
        self._v_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(value_states, non_blocking=True)

        return
    
    def decode_update_has_kv_bias(self, key_states, value_states, kv_bias, topk_idx):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # kv_bias: (head_num, seq_len, 1)
        # topk_idx: list, [H, B, M]
        # return: (self._k_gpu, self._v_gpu, mapped_topk_idx)
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)


        self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回 TODO: 最后弄写回的逻辑
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # B. 计算load mask和新的映射 尾块的映射不会动
        diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)
        self._block_map.copy_(self._new_block_map_buf, non_blocking=True)

        # C. 从disk取需要load进来的块
        flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
        flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, self._load_mask, self.block_size)

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu
            

        return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens 
    
    def decode_update_no_kv_bias(self, key_states, value_states, kv_bias, topk_idx):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # kv_bias: (head_num, seq_len, 1)
        # topk_idx: list, [H, B, M]
        # return: (self._k_gpu, self._v_gpu, mapped_topk_idx)
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)

        # 注意！用python写分支会非常慢
        if kv_bias != None:
            self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回 TODO: 最后弄写回的逻辑
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # B. 计算load mask和新的映射 尾块的映射不会动
        diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)
        self._block_map.copy_(self._new_block_map_buf, non_blocking=True)

        # C. 从disk取需要load进来的块
        flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
        flash_h2d_from_mask(self._v_gpu, self._v_cpu, self._load_mask, self.block_size)

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu
            

        return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens 
        


class InfLLMv2CacheLayer(DynamicLayer):
    def __init__(self, config, has_kv_bias):
        super().__init__()
        # Initialize any additional attributes specific to InfLLMv2CacheLayer
        self.no_rope_keys = torch.tensor([], dtype=torch.float32)
        self.compress_k_cache = []
        self.no_compress_k_cache = []
        self.cached_compressed_cu_seqlens = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.tail_cis = None
        self.compressed_cis = None
        self.tail_cis_len = 0
        self.total_cis = None

        self.cache_engine = CacheEngine(
            gpu_mem_usage=12,
            num_layers=config.num_hidden_layers,
            block_size=64,
            head_num=config.num_key_value_heads,
            head_dim=config.head_dim,
            has_kv_bias=has_kv_bias
        )
        self.seq_length = 0

    def update_no_rope_key(self, key_states):
        if self.no_rope_keys.numel() == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = torch.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        if len(self.compress_k_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens = cu_seqlens.clone()
            self.compress_k_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k_cache = list(torch.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k_cache[index] = torch.cat([self.compress_k_cache[index], k], dim=0)
            new_seq_lens = torch.tensor([tensor.shape[0] for tensor in self.compress_k_cache], dtype=torch.int32)
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)
            
            self.compress_k_cache_varlen = torch.cat(self.compress_k_cache, dim=0)
            self.cached_compressed_cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), new_cumsum]).to(self.compress_k_cache_varlen.device)
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens
    

    def update_compress_k_prefill(self, key_states, cu_seqlens, max_seqlen, current_batch_pos, total_bsz):
        key_shape = key_states.size()
        key_states_pad = key_states.view(-1, max_seqlen, key_shape[1], key_shape[2]).contiguous()
        self.compress_k_cache_varlen = torch.empty((total_bsz, max_seqlen, key_shape[1], key_shape[2]), dtype=key_states.dtype, device=key_states.device)
        self.compress_k_cache_varlen[current_batch_pos:current_batch_pos+key_states_pad.shape[0]].copy_(key_states_pad, non_blocking=True)
        if current_batch_pos == 0:
            self.cached_compressed_cu_seqlens = torch.empty((total_bsz+1,), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            self.cached_compressed_cu_seqlens[:cu_seqlens.shape[-1]].copy_(cu_seqlens)
            self.cached_compressed_max_seqlen = max_seqlen
            self.cached_compressed_cu_seqlens_adder = torch.arange(total_bsz, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        else:
            self.cached_compressed_cu_seqlens[current_batch_pos:current_batch_pos+cu_seqlens.shape[-1]].copy_(cu_seqlens + self.cached_compressed_cu_seqlens[current_batch_pos])


    def update_compress_k_decode(self, key_states, cu_seqlens):
        if key_states is None:
            return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens, self.cached_compressed_max_seqlen
        
        self.compress_k_cache_varlen = torch.cat((self.compress_k_cache_varlen, key_states), dim=1)
        self.cached_compressed_cu_seqlens += self.cached_compressed_cu_seqlens_adder
        self.cached_compressed_max_seqlen += 1
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens, self.cached_compressed_max_seqlen

    def update_no_compress_k_prefill(self, key_states, kernel_size=32, kernel_stride=16, current_batch_pos=None, total_bsz=None):
        # key_states: (B, N, H, D)
        B, N, H, D = key_states.shape
        self.no_compress_k_cache = torch.empty((total_bsz, kernel_size, H, D), dtype=key_states.dtype, device=key_states.device)
        self.no_compress_k_cache[current_batch_pos:current_batch_pos+B, :N, :, :].copy_(key_states)
        self.no_compress_k_len = N
        return

    def update_no_compress_k_decode(self, key_states, kernel_size=32, kernel_stride=16):
        # key_states: (B, N, H, D)
        if self.no_compress_k_len >= kernel_size:
            to_compress = self.no_compress_k_cache.clone()
            self.no_compress_k_cache[:, :kernel_stride].copy_(self.no_compress_k_cache[:, kernel_stride:])
            self.no_compress_k_cache[:, kernel_stride:kernel_stride+1].copy_(key_states)
            self.no_compress_k_len = kernel_stride + 1
            return to_compress
        else:
            self.no_compress_k_cache[:, self.no_compress_k_len:self.no_compress_k_len+1, :, :].copy_(key_states)
            self.no_compress_k_len += 1
            return None
    

    # 从update函数拆出来的
    def prefill_update_kv(self, key_states, value_states, kv_bias, current_batch_pos, total_bsz):
        self.seq_length = key_states.shape[1]
        self.cache_engine.prefill_update(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
    
    def decode_update_kv(self, key_states, value_states, kv_bias, topk_idx):
        self.seq_length += 1
        return self.cache_engine.decode_update(key_states.unsqueeze(1), value_states.unsqueeze(1), kv_bias, topk_idx)


    def update(self, key_states, value_states, cache_kwargs=None):
        is_prefill = cache_kwargs.get("is_prefill", True)
        if is_prefill: # prefill
            self.seq_length = key_states.shape[2]
            self.cache_engine.prefill_update(key_states.transpose(1, 2), value_states.transpose(1, 2))
            return key_states, value_states
        else: # decode
            topk_idx = cache_kwargs.get("topk_idx", None)
            self.seq_length += 1
            output = self.cache_engine.decode_update(key_states, value_states, topk_idx)
            return output
    
    def update_cis(self, cis, current_batch_pos, total_bsz, kernel_size=32, kernel_stride=16):
        # cis: (H, B, N)
        H, B, N = cis.shape
        if N > 1: # prefill
            if self.compressed_cis == None:
                full_block_len = N // kernel_stride
                tail_block_len = N % kernel_stride
                full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
                comp_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)
                M = comp_cis.shape[-1]
                self.comp_cis_len = M
                self.compressed_cis =  torch.empty((H, total_bsz, M), dtype=cis.dtype, device=cis.device)
                self.compressed_cis[:, :B, :].copy_(comp_cis, non_blocking=True)
                self.tail_cis = torch.empty((H, total_bsz, kernel_size), dtype=cis.dtype, device=cis.device)
                self.tail_cis_len = kernel_stride + tail_block_len
                self.tail_cis[:, :B, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * kernel_stride:], non_blocking=True)
                return comp_cis
            else:
                full_block_len = N // kernel_stride
                tail_block_len = N % kernel_stride
                full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
                comp_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)
                M = comp_cis.shape[-1]
                self.compressed_cis[:, current_batch_pos:current_batch_pos+B, :].copy_(comp_cis, non_blocking=True)
                self.tail_cis[:, current_batch_pos:current_batch_pos+B, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * kernel_stride:], non_blocking=True)
                return comp_cis
            # full_block_len = N // kernel_stride
            # tail_block_len = N % kernel_stride
            # full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
            # self.compressed_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)

            # self.tail_cis = [cis[:, :, (full_block_len - 1) * kernel_stride:]]
            # self.tail_cis_len = kernel_stride + tail_block_len
        else:
            # self.tail_cis.append(cis)
            self.tail_cis[:, :, self.tail_cis_len:self.tail_cis_len+1].copy_(cis, non_blocking=True)
            self.tail_cis_len += 1
            if self.tail_cis_len == kernel_size:
                # tail = torch.cat(self.tail_cis, dim=-1)
                tail = self.tail_cis
                tail_comp = tail.mean(dim=-1, keepdim=True)
                # self.compressed_cis = torch.cat((self.compressed_cis, tail_comp), dim=-1)
                self.compressed_cis[:, :, self.comp_cis_len:self.comp_cis_len+1].copy_(tail_comp, non_blocking=True)
                # self.tail_cis = [tail[:, :, kernel_stride:]]
                self.tail_cis[:, :, :kernel_stride].copy_(self.tail_cis[:, :, kernel_stride:])
                self.tail_cis_len = kernel_stride
            return self.compressed_cis
    
    def update_uncompressed_cis(self, cis, current_batch_pos, total_bsz):
        if self.total_cis is None:
            B, S, H = cis.shape
            self.total_cis = torch.empty((total_bsz, S+1024, H), dtype=cis.dtype, device=cis.device)
            self.total_cis[:B, :S, :].copy_(cis, non_blocking=True)
            self.cis_len = S
            self.cis_prefilled_batch = B
            return cis # prefill用的是fa接口
        elif self.cis_prefilled_batch < total_bsz:
            B, S, H = cis.shape
            assert self.cis_len == S
            self.total_cis[current_batch_pos:current_batch_pos+B, :S, :].copy_(cis, non_blocking=True)
            self.cis_prefilled_batch += B
            return cis
        else:
            self.total_cis[:, self.cis_len:self.cis_len+1, :].copy_(cis, non_blocking=True)
            self.cis_len += 1
            return self.total_cis # 用的是flash_attn_nosa的接口，为了用cuda graph保证地址不变
        
    def get_paged_kv(self):
        return self.cache_engine._k, self.cache_engine._v
    
    def get_seq_length(self):
        return self.seq_length


class InfLLMv2Cache(DynamicCache):
    def __init__(self,
                 config,num_hidden_layers: Optional[int] = None, has_kv_bias=False) -> None:
        super().__init__(config=config)
        self.layers = [InfLLMv2CacheLayer(config, has_kv_bias) for _ in range(num_hidden_layers)] if num_hidden_layers else []
        self._seen_tokens = 0

    def prefill_update_kv(self, key_states, value_states, kv_bias, layer_idx, current_batch_pos, total_bsz):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].prefill_update_kv(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
    
    def decode_update_kv(self, key_states, value_states, kv_bias, layer_idx, topk_idx):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].decode_update_kv(key_states, value_states, kv_bias, topk_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
    
    def update_cis(self, cis, layer_idx, current_batch_pos, total_bsz):
        return self.layers[layer_idx].update_cis(cis, current_batch_pos, total_bsz)

    def update_uncompressed_cis(self, cis, layer_idx, current_batch_pos, total_bsz):
        return self.layers[layer_idx].update_uncompressed_cis(cis, current_batch_pos, total_bsz)
    
    def get_paged_kv(self, layer_idx):
        return self.layers[layer_idx].get_paged_kv()

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k_prefill(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None, max_seqlen=None, current_batch_pos=None, total_bsz=None):
        return self.layers[layer_idx].update_compress_k_prefill(key_states, cu_seqlens, max_seqlen, current_batch_pos, total_bsz)

    def update_compress_k_decode(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k_decode(key_states, cu_seqlens)

    def update_no_compress_k_prefill(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None, current_batch_pos=None, total_bsz=None):
        return self.layers[layer_idx].update_no_compress_k_prefill(key_states, kernel_size, kernel_stride, current_batch_pos=current_batch_pos, total_bsz=total_bsz)

    def update_no_compress_k_decode(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k_decode(key_states, kernel_size, kernel_stride)

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)
    
    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()
    
    def get_max_length(self):
        raise NotImplementedError