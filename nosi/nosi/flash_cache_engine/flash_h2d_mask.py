import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
import random
import time



@triton.jit
def flash_h2d_from_mask_kernel(
    gpu_ptr, # (B, S_GPU, H, D)
    cpu_ptr, # (B, S_CPU, H, D)
    load_ids_ptr, # (H, B, M)
    stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu,
    stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu,
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

    mask_src = (cpu_block_id * block_size + offset_s) < S_CPU
    mask_dst = (pid_m * block_size + offset_s) < S_GPU

    vals = tl.load(src_ptrs, mask=mask_src)
    tl.store(dst_ptrs, vals, mask=mask_dst)


def flash_h2d_from_mask(
    gpu_data, # (B, S_GPU, H, D)
    cpu_data, # (B, S_CPU, H, D)
    load_ids, # (H,B,M)
    block_size: int = 64,
):
    B, S_GPU, H, D = gpu_data.shape
    _, S_CPU, _, _ = cpu_data.shape
    H2, B2, M = load_ids.shape
    assert H == H2 and B == B2

    stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu = cpu_data.stride()
    stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu = gpu_data.stride()

    grid = (B, H, M)

    flash_h2d_from_mask_kernel[grid](
        gpu_data, cpu_data, load_ids,
        stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu,
        stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu,
        S_GPU, S_CPU,
        B, H, M,
        block_size, D,
    )

def make_block_index_map(H, B, M, max_M):
    """
    old_map / new_act 合法 block index = [0..M-1]
    每个 (h,b) 行内是一个排列（无重复）
    """
    out = torch.empty((H, B, M), dtype=torch.int32)
    for h in range(H):
        for b in range(B):
            out[h, b] = torch.randperm(max_M)[:M]
    return out

def test_flash_h2d():
    H, B, M = 2, 16, 64
    block_size = 64
    D = 128
    S_CPU = 4 * M * block_size
    S_GPU = M * block_size

    # --- simulate cpu_data ---
    cpu_data = torch.randn(B, S_CPU, H, D, device="cuda")

    # --- simulate gpu_data ---
    gpu_data = torch.zeros(B, S_GPU, H, D, device="cuda")

    # --- generate unique old_map / new_act ---
    old_map = make_block_index_map(H, B, M, 4*M).cuda().long()
    new_act = make_block_index_map(H, B, M, 4*M).cuda().long()

    old_map[...] = -1 # 测试一下如果old_map是空的时候，第一次decode/prefill之后调整block时

    # --- run diff_offload ---
    def load_extension():
        diff = load(
            name="diff_offload",
            sources=["diff_offload.cpp", "diff_offload_kernel.cu"],
            verbose=True,
        )
        return diff
    diff = load_extension()
    new_map = torch.empty_like(old_map)
    load_map = torch.empty_like(old_map)
    diff.diff_offload(old_map, new_act, new_map, load_map)

    print("load_map =\n", load_map)

    # --- Run flash operator ---
    flash_h2d_from_mask(gpu_data, cpu_data, load_map, block_size)

    # --- CPU reference copy ---
    cpu_ref = torch.zeros_like(gpu_data)
    for h in range(H):
        for b in range(B):
            for m in range(M):
                blk = load_map[h,b,m].item()
                if blk < 0:
                    continue
                cpu_start = blk * block_size
                gpu_start = m * block_size
                cpu_ref[b, gpu_start:gpu_start+block_size, h, :] = \
                    cpu_data[b, cpu_start:cpu_start+block_size, h, :]

    # verify
    if torch.allclose(cpu_ref, gpu_data):
        print("✅ flash_h2d_from_mask works!")
    else:
        print("❌ mismatch detected!")
        print("Expected:\n", cpu_ref)
        print("Got:\n", gpu_data)

def benchmark_diff_h2d(
    H=2,
    B=16,
    M=64,
    block_size=64,
    D=128,
    miss_ratio=0.1,
    warmup=20,
    repeat=2000,
):
    print("\n=== Benchmark diff_offload + flash_h2d (CUDA Graph) ===")
    print(f" H={H}, B={B}, M={M}, D={D}, miss_ratio={miss_ratio}")

    def load_extension():
        diff = load(
            name="diff_offload",
            sources=["diff_offload.cpp", "diff_offload_kernel.cu"],
            verbose=False,
        )
        return diff

    diff = load_extension()

    # CPU has 4M blocks
    S_CPU = 4 * M * block_size
    # GPU cache only M blocks
    S_GPU = M * block_size

    # cpu_data should be host-side (pinned)
    cpu_data = torch.randn(B, S_CPU, H, D, pin_memory=True, dtype=torch.bfloat16)
    gpu_data = torch.zeros(B, S_GPU, H, D, device="cuda", dtype=torch.bfloat16)

    # -------------------------------
    # Construct controlled-miss new_act
    # -------------------------------
    old_map = make_block_index_map(H, B, M, 4 * M).cuda().long()
    new_act = old_map.clone()

    num_miss = int(M * miss_ratio)
    if num_miss > 0:
        miss_vals = torch.randperm(4 * M)[M: M + num_miss].long().cuda()

        for h in range(H):
            for b in range(B):
                miss_positions = torch.randperm(M)[:num_miss]
                new_act[h, b, miss_positions] = miss_vals[torch.randperm(num_miss)]

    new_map = torch.empty_like(old_map)
    load_ids = torch.empty_like(old_map)

    # -------------------------------
    # Warmup
    # -------------------------------
    for _ in range(warmup):
        diff.diff_offload(old_map, new_act, new_map, load_ids)
        flash_h2d_from_mask(gpu_data, cpu_data, load_ids, block_size)
    torch.cuda.synchronize()

    # -------------------------------
    # CUDA Graph capture
    # -------------------------------
    g = torch.cuda.CUDAGraph()

    old_s = old_map.clone()
    new_s = new_act.clone()
    newmap_s = new_map.clone()
    load_s = load_ids.clone()
    gpu_s = gpu_data.clone()

    torch.cuda.synchronize()
    print("Capturing graph...")

    # First invocation outside graph (establish memory pools)

    with torch.cuda.graph(g):
        diff.diff_offload(old_s, new_s, newmap_s, load_s)
        flash_h2d_from_mask(gpu_s, cpu_data, load_s, block_size)
        flash_h2d_from_mask(gpu_s, cpu_data, load_s, block_size)

    torch.cuda.synchronize()

    # -------------------------------
    # Benchmark using CUDA events
    # -------------------------------
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        g.replay()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / repeat

    # -------------------------------
    # Bandwidth Calculation
    # -------------------------------
    # Count blocks that require loading     (blk >= 0)
    load_count = (load_s >= 0).sum().item()

    # Each block: block_size × D × 4 bytes
    bytes_per_block = block_size * D * 2
    total_bytes = load_count * bytes_per_block * 2   # run flash twice inside graph

    # bandwidth in GB/s
    bandwidth_gbs = (total_bytes / (avg_ms * 1e-3)) / 1e9

    print(f"Avg kernel time = {avg_ms:.6f} ms")
    print(f"Load blocks: {load_count}")
    print(f"Effective bandwidth = {bandwidth_gbs:.2f} GB/s\n")

    return avg_ms, bandwidth_gbs


if __name__ == "__main__":
    # test_flash_h2d()
    benchmark_diff_h2d()
