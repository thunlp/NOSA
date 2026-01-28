import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from flashinfer.norm import rmsnorm
from flashinfer.activation import silu_and_mul
from tqdm import tqdm
import torch.nn.functional as F
from functools import lru_cache
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from infllm_v2 import (
    infllmv2_attn_stage1,
    infllmv2_attn_stage1_fast,
    infllmv2_attn_varlen_func,
    infllmv2_attn_with_kvcache,
    max_pooling_1d,
    max_pooling_1d_varlen
)
import time
import torch.cuda.nvtx as nvtx

from .cache_engine import InfLLMv2Cache
from .cache_engine_gpu import InfLLMv2Cache as InfLLMv2CacheNoOffload
from .max_pooling_fused import nosa_pooling
from .nosa_linear import nosa_linear
from flash_attn import flash_attn_with_kvcache
from flash_attn_nosa import flash_attn_with_kvcache as flash_attn_nosa_with_kvcache



stream_offload = torch.cuda.Stream()

def layer_norm(
    hidden_states: torch.Tensor,
    eps: float,
    w: torch.Tensor,
):
    return rmsnorm(hidden_states.view(-1, hidden_states.size(-1)), w, eps).view_as(hidden_states)


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (torch.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (torch.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (torch.Tensor): Cumulative sequence lengths after compression.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device)
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq]

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. Generate filtered_indices
    chunk_indices = torch.arange(
        0, chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size]
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices
    return filtered_indices, cu_seqlens_compressed

def compress_k(k: torch.Tensor, cu_seqlens, head_num_k, head_dim, kernel_size=32, kernel_stride=16):
    filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
        cu_seqlens, kernel_size, kernel_stride
    )

    # Extract filtered key vectors
    filtered_k = k.index_select(0, filtered_k_indices.view(-1))

    # split
    filtered_k = filtered_k.view(filtered_k.shape[0] // kernel_size, kernel_size, head_num_k, head_dim)  # [l, block_size,h,d]

    compressed_k = filtered_k.mean(dim=1)
    return compressed_k, cu_seqlens_compressed


class LlamaLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wqkv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_up_proj :torch.Tensor = None 
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.has_buffers = False
    
    def init_parameters(self, hf_layer, num_heads, num_key_value_heads, head_dim):
        self.wqkv :torch.Tensor= torch.cat((hf_layer.self_attn.q_proj.weight.detach(), hf_layer.self_attn.k_proj.weight.detach(), hf_layer.self_attn.v_proj.weight.detach()), dim=0)
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.q_size = hf_layer.self_attn.q_proj.weight.shape[0]
        self.kv_size = hf_layer.self_attn.k_proj.weight.shape[0]

        self.gate_up_proj = torch.cat((hf_layer.mlp.gate_proj.weight.detach(), hf_layer.mlp.up_proj.weight.detach()), dim=0)
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
    
    def init_gpu(self, device:str = 'cuda:0'):
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

    @torch.inference_mode()
    def prefill_forward(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, total_bsz, current_batch_pos):
        # [ATTN] prepare
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()

        # [ATTN] prenorm
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        # [ATTN] calculate q k v (bsz, seq_len, *)
        qkv = F.linear(hidden_states, self.wqkv)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # [ATTN] apply rope
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True)


        # [ATTN] unpad data
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)

        # [ATTN] compress k
        compressed_k, cu_seqlens_comp = compress_k(key_states, cu_seqlens, self.num_key_value_heads, self.head_dim)
        compressed_k = compressed_k.contiguous()
        max_seqlen_comp = (cu_seqlens_comp[1:] - cu_seqlens_comp[:-1]).max().item()

        # [ATTN] update kv
        cache_engine.prefill_update_kv(
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim),
            value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim),
            torch.empty((bsz, q_len, self.num_key_value_heads), dtype=key_states.dtype, device=key_states.device),
            self.layer_idx, current_batch_pos, total_bsz
        )

        # [ATTN] update compressed k
        cache_engine.update_compress_k_prefill(compressed_k, self.layer_idx, cu_seqlens_comp, max_seqlen=max_seqlen_comp, current_batch_pos=current_batch_pos, total_bsz=total_bsz)
        pad_key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        kernel_stride = 16
        no_compress_k_start = max_seqlen_comp * kernel_stride
        no_compress_k = pad_key_states[:, no_compress_k_start:]
        cache_engine.update_no_compress_k_prefill(no_compress_k, self.layer_idx, 32, 16, current_batch_pos=current_batch_pos, total_bsz=total_bsz)

        # [ATTN] stage 1
        score = infllmv2_attn_stage1(
            query_states,
            compressed_k,
            compressed_k,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_comp,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_comp,
            causal=True
        )

        # [ATTN] max pooling
        cache_lens = torch.zeros(bsz, dtype=torch.int32, device=query_states.device) 
        q_idx = torch.cat([
            (torch.arange(cu_seqlens[i + 1] - cu_seqlens[i], device=query_states.device) + 
                max_seqlen - (cu_seqlens[i + 1] - cu_seqlens[i])) // 64
            for i in range(bsz)
        ], dim=0)
        score = score[:, :q_idx.shape[0], :]
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens,
            cu_seqlens_comp,
            cache_lens,
            max_seqlen,
            max_seqlen_comp,
            local_blocks=16,
            init_blocks=1,
            block_size=64,
            stride=16)
        del score

        # generate topk index
        topk = 64
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)
        del block_score
        # [ATTN] sparse attention
        topk_attn_output = infllmv2_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
            deterministic=False,
            softmax_scale=None,
            causal=True,
            return_attn_probs=False,
            topk_idx=topk_idx
        )
        attn_output = topk_attn_output
        attn_output = attn_output.view(bsz, q_len, -1)

        del query_states, key_states

        # [ATTN] wo
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states

        # [FFN] 
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states


    @torch.inference_mode()
    def decode_forward_warmup(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, pooling_buf, topk_buf_val, topk_buf_indices, topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf):
        # [ATTN] prepare
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        max_pooling_buf = pooling_buf[:self.num_key_value_heads]
        max_pooling_buf_cis = pooling_buf[self.num_key_value_heads:]

        # [ATTN] prenorm
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        # [ATTN] calculate q k v (bsz, seq_len, *)
        nvtx.range_push("qkv_proj")
        qkv = F.linear(hidden_states, self.wqkv)
        nvtx.range_pop()


        nvtx.range_push("split and cis") 
        qkv = F.linear(hidden_states, self.wqkv)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        nvtx.range_pop()


        # [ATTN] apply rope
        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True)
        nvtx.range_pop()

        # [ATTN] unpad data
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        
        # [ATTN] compress k
        nvtx.range_push("compress k")
        no_compress_k = cache_engine.update_no_compress_k_decode(key_states.unsqueeze(1), self.layer_idx, 32, 16)
        if no_compress_k is not None:
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
        else:
            new_compressed_k = None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(new_compressed_k, self.layer_idx)
        compressed_k = compressed_k.contiguous().flatten(0, 1)
        nvtx.range_pop()


        # [ATTN] stage 1
        nvtx.range_push("stage 1")
        nvtx.range_push("real stage 1")

        score = infllmv2_attn_stage1_fast(
            query_states,
            compressed_k,
            compressed_k,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_comp,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_comp,
            causal=False,
        )
        nvtx.range_pop()


        nvtx.range_push("max pooling")
        nvtx.range_push("kernel: max pooling")



        nvtx.range_pop()
        nvtx.range_pop()
        nvtx.range_pop()

        self.score_buf = score
        self.compressed_cis_buf = torch.empty_like(score) # make a fake input

        nosa_pooling(
            self.score_buf,
            self.compressed_cis_buf,
            max_seqlen_comp,
            max_pooling_buf,
            max_pooling_buf_cis,
            stride=16,
            block_size=64,
            local_blocks=16,
            init_blocks=1
        )

        self.after_pooling_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.after_pooling_graph):

            nosa_pooling(
                self.score_buf,
                self.compressed_cis_buf,
                max_seqlen_comp,
                max_pooling_buf,
                max_pooling_buf_cis,
                stride=16,
                block_size=64,
                local_blocks=16,
                init_blocks=1
            )
            topk = 64
            # qk_select = 33

            # torch.topk(
            #     max_pooling_buf, qk_select, dim=-1, sorted=False, out=(topk_val_buf_q, topk_idx_buf_q)
            # )
            # scatter_mask = mask_buf
            # scatter_mask.zero_()
            # scatter_mask.scatter_(2, topk_idx_buf_q, True)
            # max_pooling_buf_cis.masked_fill_(scatter_mask, float('inf'))

            torch.topk(
                max_pooling_buf, topk, dim=-1, sorted=False, out=(topk_val_buf, topk_idx_buf)
            )
        self.after_pooling_graph.replay()
    
        topk_idx = topk_idx_buf

        # [ATTN] offloading-update
        nvtx.range_push("offloading update")
        key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(key_states, value_states, None, self.layer_idx, topk_idx)
        nvtx.range_pop()
        # [ATTN] stage 2
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states,
            value_states,
            cache_seqlens=_cache_lens,
        )
        nvtx.range_pop()

        attn_output = attn_output.view(bsz, q_len, -1)
        # [ATTN] wo
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states

        # [FFN] 
        nvtx.range_push("ffn forward norm")
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        nvtx.range_pop()
        nvtx.range_push("ffn forward actual")
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states
        nvtx.range_pop()
        return hidden_states


    @torch.inference_mode()
    def decode_forward(self, hidden_states, position_ids, cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, pooling_buf, topk_buf_val, topk_buf_indices, topk_val_buf_q, topk_idx_buf_q, topk_val_buf, topk_idx_buf, mask_buf):
        # [ATTN] prepare
        nvtx.range_push("linear")
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        max_pooling_buf = pooling_buf[:self.num_key_value_heads]
        max_pooling_buf_cis = pooling_buf[self.num_key_value_heads:]

        # [ATTN] prenorm
        hidden_states = layer_norm(hidden_states, self.input_layernorm_variance_epsilon, self.input_layernorm_weight)

        # [ATTN] calculate q k v (bsz, seq_len, *)
        qkv = F.linear(hidden_states, self.wqkv)

        qkv = F.linear(hidden_states, self.wqkv)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        nvtx.range_pop()

        # [ATTN] apply rope
        nvtx.range_push("rope")
        query_states = query_states.view(bsz * q_len, -1)
        key_states = key_states.view(bsz * q_len, -1)
        apply_rope_with_cos_sin_cache_inplace(position_ids.flatten(), query_states, key_states, self.head_dim, cos_sin_cache, True)
        nvtx.range_pop()

        # [ATTN] unpad data
        query_states = query_states.reshape(bsz * q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        
        # [ATTN] compress k
        nvtx.range_push("compress k")
        no_compress_k = cache_engine.update_no_compress_k_decode(key_states.unsqueeze(1), self.layer_idx, 32, 16)
        if no_compress_k is not None:
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
        else:
            new_compressed_k = None
        compressed_k, cu_seqlens_comp, max_seqlen_comp = cache_engine.update_compress_k_decode(new_compressed_k, self.layer_idx)
        compressed_k = compressed_k.contiguous().flatten(0, 1)
        nvtx.range_pop()


        # [ATTN] stage 1
        nvtx.range_push("stage 1")

        score = infllmv2_attn_stage1_fast(
            query_states,
            compressed_k,
            compressed_k,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_comp,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_comp,
            causal=False,
        )
        nvtx.range_pop()


        nvtx.range_push("max pooling")

        self.score_buf.copy_(score)

        nvtx.range_pop()

        self.after_pooling_graph.replay()
        topk_idx = topk_idx_buf
        # [ATTN] offloading-update
        nvtx.range_push("offloading update")
        key_states, value_states, kv_bias, _cache_lens = cache_engine.decode_update_kv(key_states, value_states, None, self.layer_idx, topk_idx)
        nvtx.range_pop()

        # [ATTN] stage 2
        nvtx.range_push("stage 2")
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            key_states,
            value_states,
            cache_seqlens=_cache_lens,
        )
        nvtx.range_pop()

        attn_output = attn_output.view(bsz, q_len, -1)
        # [ATTN] wo
        hidden_states = F.linear(attn_output, self.wo)
        hidden_states = residual + hidden_states

        # [FFN] 
        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, self.post_attention_layernorm_variance_epsilon, self.post_attention_layernorm_weight)
        hidden_states = F.linear(hidden_states, self.gate_up_proj)
        dd = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (dd, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, self.down_proj)
        hidden_states = residual + hidden_states
        nvtx.range_pop()
        return hidden_states


class Llama:
    def __init__(self, 
        model_name: str = "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        offload: bool = True,
        max_length :int = 128*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full') -> None:
        

        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.vocab_size = self.config.vocab_size
        self.past_len = 0
        self.has_buffers = False
        self.offload = offload

        self.init_parameters()

    def init_parameters(self):
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype, trust_remote_code=True)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        cos_cache, sin_cache = self._set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.to(self.device))

        # _v = torch.tensor([0], dtype=self.dtype, device=self.device)
        # position_ids = torch.arange(0, self.max_length + 1024, device=self.device, dtype=self.dtype).unsqueeze(0)
        # hf_model.model.rotary_emb = hf_model.model.rotary_emb.to(self.device)
        # cos_cache, sin_cache = hf_model.model.rotary_emb(_v, position_ids)
        self.cos_sin_cache = torch.cat((cos_cache[:, :64], sin_cache[:, :64]), dim=-1).to(torch.float32)
        
        del cos_cache, sin_cache

        self.layers :list[LlamaLayer] = []

        for idx, hf_layer in enumerate(tqdm(hf_model.model.layers, desc="coverting model")):
            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer, head_dim=self.head_dim, num_heads=self.num_heads, num_key_value_heads=self.num_key_value_heads)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            # gc.collect()

        self.num_layers = len(self.layers)
    
    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length + 1024, device=self.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        if input_len > 1:
            self.past_len = 0
        position_ids = torch.arange(self.past_len, self.past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        if input_len > 1:
            self.past_len = input_len
        else:
            self.past_len += 1
        return position_ids


    @torch.inference_mode()
    def prefill_inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            cache_engine = None):
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        max_seqlen = seq_len

        output_hs = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=self.device)
        T = 1
        for t in tqdm(range(0, bsz, T), desc="prefill batch"):
            hs = F.embedding(input_ids[t:t+T], self.embed_tokens)
            cu_seqlens = torch.arange(0, hs.shape[0] * seq_len + 1, seq_len, dtype=torch.int, device=input_ids.device)
            for idx in tqdm(range(self.num_layers), desc="prefill layer"):
                hs = self.layers[idx].prefill_forward(hs, position_ids[t:t+T], self.cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, total_bsz=bsz, current_batch_pos=t)
            output_hs[t:t+T].copy_(hs[:, -1:, :], non_blocking=True)
        
        hidden_states = layer_norm(output_hs, w=self.norm_weight, eps=self.norm_variance_epsilon)

        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits
    

    @torch.inference_mode()
    def decode_inference(self,
            input_ids: torch.LongTensor,
            cu_seqlens: torch.Tensor,
            position_ids: torch.LongTensor,
            cache_engine = None, warmup=False):
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        max_seqlen = 1

        cache_lens = cache_engine.get_seq_length(0)
        cache_lens = torch.tensor([cache_lens] * bsz, dtype=torch.int, device=hidden_states.device)
        q_idx = cache_lens // 64

        warmup = not self.has_buffers

        if warmup:

            
            total_len = cache_engine.get_seq_length(0) + 1
            out_len = (total_len + 64 - 1) // 64
            self.pooling_buf_all = torch.empty(2 * self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=hidden_states.dtype)
            self.max_pooling_buf = torch.empty(self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=hidden_states.dtype)
            self.max_pooling_buf_cis = torch.empty(self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_buf_val = torch.empty(self.num_key_value_heads, bsz, 64, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_buf_indices = torch.empty(self.num_key_value_heads, bsz, 64, device=hidden_states.device, dtype=torch.int64)
            self.topk_val_buf_q = torch.empty(self.num_key_value_heads, bsz, 33, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_idx_buf_q = torch.empty(self.num_key_value_heads, bsz, 33, device=hidden_states.device, dtype=torch.int64)
            self.topk_val_buf = torch.empty(self.num_key_value_heads, bsz, 64, device=hidden_states.device, dtype=hidden_states.dtype)
            self.topk_idx_buf = torch.empty(self.num_key_value_heads, bsz, 64, device=hidden_states.device, dtype=torch.int64)
            self.mask_buf = torch.empty(self.num_key_value_heads, bsz, out_len, device=hidden_states.device, dtype=torch.bool)

            self.has_buffers = True


            for idx in range(self.num_layers):
                hidden_states = self.layers[idx].decode_forward_warmup(hidden_states, position_ids, self.cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices, self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf)
        else:
            for idx in range(self.num_layers):
                hidden_states = self.layers[idx].decode_forward(hidden_states, position_ids, self.cos_sin_cache, cu_seqlens, max_seqlen, cache_engine, cache_lens, q_idx, self.pooling_buf_all, self.topk_buf_val, self.topk_buf_indices, self.topk_val_buf_q, self.topk_idx_buf_q, self.topk_val_buf, self.topk_idx_buf, self.mask_buf)
        
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, cache_engine=None):
        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)
        position_ids = torch.zeros(batch_size, input_ids.shape[-1], device=self.device, dtype=torch.int)
        T = 10000 
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            pos_ids = self.get_ctx(req_input_ids)
            logits[bsz:bsz+T].copy_(self.prefill_inference(input_ids=req_input_ids, position_ids=pos_ids, cache_engine=cache_engine))
            position_ids[bsz:bsz+T].copy_(pos_ids)
        return logits, position_ids


    def batch_generate(self, input_ids, max_new_tokens=4):
        cache_cls = InfLLMv2Cache if self.offload else InfLLMv2CacheNoOffload
        cache_engine = cache_cls(config=self.config, num_hidden_layers=self.config.num_hidden_layers, has_kv_bias=False)
        logits, position_ids = self.batch_prefill(input_ids, cache_engine)

        next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = [next_ids]
        position_ids = position_ids[:, -1:] + 1

        cu_seqlens_de = torch.arange(0, next_ids.shape[0] + 1, dtype=torch.int, device=input_ids.device)

        for idx in range(max_new_tokens - 1):
            warmup = idx == 0
            logits = self.decode_inference(next_ids, cu_seqlens_de, position_ids, cache_engine, warmup=warmup)

            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_ids)
            position_ids = position_ids[:, -1:] + 1
        gen_ids = torch.cat(gen_ids, dim=-1)
        return gen_ids

    def batch_generate_benchmark(self, input_ids, max_new_tokens=4):
        cache_cls = InfLLMv2Cache if self.offload else InfLLMv2CacheNoOffload
        cache_engine = cache_cls(config=self.config, num_hidden_layers=self.config.num_hidden_layers, has_kv_bias=False)
        logits, position_ids = self.batch_prefill(input_ids, cache_engine)

        next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = [next_ids]
        position_ids = position_ids[:, -1:] + 1

        cu_seqlens_de = torch.arange(0, next_ids.shape[0] + 1, dtype=torch.int, device=input_ids.device)

        for it in range(max_new_tokens - 1):
            warmup = it == 0
            logits = self.decode_inference(next_ids, cu_seqlens_de, position_ids, cache_engine, warmup=warmup)

            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_ids)
            position_ids = position_ids[:, -1:] + 1
            if it == 0:
                torch.cuda.synchronize()
                beg = time.time()
        torch.cuda.synchronize()
        end = time.time()
        gen_ids = torch.cat(gen_ids, dim=-1)
        return gen_ids, gen_ids.shape[0] * (max_new_tokens - 2) / (end - beg)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

if __name__ == "__main__":


    path = "/somepath"
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, legacy=False)
    input_ids = tokenizer("<|im_start|>user\nEfficient Sparse/Dense Attention Kernels: Efficient single/batch attention for sparse(paged)/dense KV-storage on CUDA Cores and Tensor Cores (both FA2 & FA3) templates. The vector-sparse attention can achieve 90% of the bandwidth of dense kernels with same problem size. Write a brief introduction of flash attention <|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda").input_ids.repeat(4, 1)

    std_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cuda')
    output = std_model.generate(input_ids, max_new_tokens=16)
    print(tokenizer.batch_decode(output[:, input_ids.shape[0]:])[0])

    model = Llama(
        model_name=path,
        device="cuda",
    )
    output = model.batch_generate(input_ids, max_new_tokens=16)
    print(tokenizer.batch_decode(output)[0])

