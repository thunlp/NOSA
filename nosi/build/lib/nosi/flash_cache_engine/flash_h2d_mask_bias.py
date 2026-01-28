import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
import random
import time



@triton.jit
def flash_h2d_from_mask_bias_kernel(
    gpu_ptr, # (B, S_GPU, H, D)
    cpu_ptr, # (B, S_CPU, H, D)
    bias_buf_ptr, # (B, S_GPU, H)
    bias_ptr, # (B, *, H)
    load_ids_ptr, # (H, B, M)
    stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu,
    stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu,
    stride_b_bias_buf, stride_m_bias_buf, stride_h_bias_buf,
    stride_b_bias, stride_m_bias, stride_h_bias,
    S_GPU, S_CPU,
    B, H, M,
    block_size: tl.constexpr,
    D: tl.constexpr,
):

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # load offset
    cpu_block_id = tl.load(
        load_ids_ptr + pid_h * B * M + pid_b * M + pid_m
    )

    if cpu_block_id < 0:
        return

    offset_s = tl.arange(0, block_size)[:, None]  # (bs,1)
    offset_d = tl.arange(0, D)[None, :]           # (1,D)

    mask_src = (cpu_block_id * block_size + offset_s) < S_CPU
    mask_dst = (pid_m * block_size + offset_s) < S_GPU

    bias_src_ptrs = (
        bias_ptr
        + pid_b * stride_b_bias
        + (cpu_block_id * block_size + offset_s) * stride_m_bias
        + pid_h * stride_h_bias
    )
    bias_dst_ptrs = (
        bias_buf_ptr
        + pid_b * stride_b_bias_buf
        + (pid_m * block_size + offset_s) * stride_m_bias_buf
        + pid_h * stride_h_bias_buf
    )
    bias_vals = tl.load(bias_src_ptrs, mask=mask_src)
    tl.store(bias_dst_ptrs, bias_vals, mask=mask_dst)

    # src: CPU
    src_ptrs = (
        cpu_ptr
        + pid_b * stride_b_cpu
        + (cpu_block_id * block_size + offset_s) * stride_m_cpu
        + pid_h * stride_h_cpu
        + offset_d * stride_d_cpu
    )

    # dst: GPU
    dst_ptrs = (
        gpu_ptr
        + pid_b * stride_b_gpu
        + (pid_m * block_size + offset_s) * stride_m_gpu
        + pid_h * stride_h_gpu
        + offset_d * stride_d_gpu
    )


    vals = tl.load(src_ptrs, mask=mask_src)
    tl.store(dst_ptrs, vals, mask=mask_dst)


def flash_h2d_from_mask_bias(
    gpu_data, # (B, S_GPU, H, D)
    cpu_data, # (B, S_CPU, H, D)
    kv_bias_buf, # (B, S_GPU, H)
    kv_bias,  #(B, *, H)
    load_ids, # (H,B,M)
    block_size: int = 64,
):
    B, S_GPU, H, D = gpu_data.shape
    _, S_CPU, _, _ = cpu_data.shape
    H2, B2, M = load_ids.shape
    assert H == H2 and B == B2

    stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu = cpu_data.stride()
    stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu = gpu_data.stride()
    stride_b_bias_buf, stride_m_bias_buf, stride_h_bias_buf = kv_bias_buf.stride()
    stride_b_bias, stride_m_bias, stride_h_bias = kv_bias.stride()

    grid = (B, H, M)

    flash_h2d_from_mask_bias_kernel[grid](
        gpu_data, cpu_data, kv_bias_buf, kv_bias, load_ids,
        stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu,
        stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu,
        stride_b_bias_buf, stride_m_bias_buf, stride_h_bias_buf,
        stride_b_bias, stride_m_bias, stride_h_bias,
        S_GPU, S_CPU,
        B, H, M,
        block_size, D,
    )
