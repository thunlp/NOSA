import torch
import triton
import triton.language as tl


@triton.jit
def nosa_linear_kernel(
    qkv_ptr,
    delta_ptr,
    A_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cis_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    R: tl.constexpr, # q_kv_ratio
    kv_heads: tl.constexpr,
    D_DIM: tl.constexpr,
    D_DIM_Q: tl.constexpr,
):
    tidx_bs, tidx_d, tidx_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tidx_b = tidx_bs // S
    tidx_s = tidx_bs % S

    d_offsets = tl.arange(0, D_DIM)
    d_offsets_q = tl.arange(0, D_DIM_Q)
    
    # copy q
    row_q_from = (
        qkv_ptr 
        + tidx_b * S * (q_size + 2 * kv_size) # B
        + tidx_s * (q_size + 2 * kv_size) # S
        + (d_offsets_q + D_DIM_Q * tidx_d) # D
    )
    q_val = tl.load(row_q_from, mask=(d_offsets_q + D_DIM_Q * tidx_d) < q_size)
    row_q_to = (
        q_ptr
        + tidx_b * S * q_size # B
        + tidx_s * q_size # S
        + (d_offsets_q + D_DIM_Q * tidx_d) # D
    )
    tl.store(row_q_to, q_val, mask=(d_offsets_q + D_DIM_Q * tidx_d) < q_size)

    # copy k
    row_k_from = (
        qkv_ptr
        + tidx_b * S * (q_size + 2 * kv_size) # B
        + tidx_s * (q_size + 2 * kv_size) # S
        + (q_size + d_offsets + D_DIM * tidx_d) # D
    )
    k_val = tl.load(row_k_from, mask=(d_offsets + D_DIM * tidx_d) < kv_size)
    row_k_to = (
        k_ptr
        + tidx_b * S * kv_size # B
        + tidx_s * kv_size # S
        + (d_offsets + D_DIM * tidx_d) # D
    )
    tl.store(row_k_to, k_val, mask=(d_offsets + D_DIM * tidx_d) < kv_size)

    # copy v
    row_v_from = (
        qkv_ptr
        + tidx_b * S * (q_size + 2 * kv_size) # B
        + tidx_s * (q_size + 2 * kv_size) # S
        + (q_size + kv_size + d_offsets + D_DIM * tidx_d) # D
    )
    v_val = tl.load(row_v_from, mask=(d_offsets + D_DIM * tidx_d) < kv_size)
    row_v_to = (
        v_ptr
        + tidx_b * S * kv_size # B
        + tidx_s * kv_size # S
        + (d_offsets + D_DIM * tidx_d) # D
    )
    tl.store(row_v_to, v_val, mask=(d_offsets + D_DIM * tidx_d) < kv_size)

    delta_from = (
        delta_ptr + (tidx_h * kv_size) + (d_offsets + D_DIM * tidx_d)
    )
    delta_val = tl.load(delta_from, mask=(d_offsets + D_DIM * tidx_d) < kv_size)

    act = tl.sum(v_val * delta_val, axis=0)

    # 这里不对，没有做reduce
    a_from = A_ptr + tidx_h
    a_val = tl.load(a_from, mask=tidx_h < kv_heads)
    act = tl.log(1 + tl.exp(act))
    act = act * a_val

    cis_to = cis_ptr + tidx_b * S * kv_heads + tidx_s * kv_heads + tidx_h
    tl.store(cis_to, act, mask=tidx_h < kv_heads)
    




def nosa_linear(qkv, delta_weight, A, q_size=2048, kv_size=256, kv_heads=2):
    # 假定全是contiguous的
    B, S, _ = qkv.shape
    q   = torch.empty((B, S, q_size),  device=qkv.device, dtype=qkv.dtype)
    k   = torch.empty((B, S, kv_size), device=qkv.device, dtype=qkv.dtype)
    v   = torch.empty((B, S, kv_size), device=qkv.device, dtype=qkv.dtype)
    cis = torch.empty((B, S, kv_heads), device=qkv.device, dtype=qkv.dtype)


    D_DIM = 256 # 这里需要和kv_size一致，已经足够快了，不需要tile

    grid = (B * S, (kv_size + kv_size - 1) // kv_size, kv_heads)

    nosa_linear_kernel[grid](
        qkv, delta_weight, A, q, k, v, cis,
        B,
        S,
        q_size,
        kv_size,
        q_size // kv_size, # q_kv_ratio
        kv_heads,
        D_DIM,
        D_DIM * q_size // kv_size, # D_DIM_Q
    )

    # reshape back
    q   = q.view(B, S, q_size)
    k   = k.view(B, S, kv_size)
    v   = v.view(B, S, kv_size)
    cis = cis.view(B, S, kv_heads)
    return q, k, v, cis


import torch
import torch.nn.functional as F

torch.manual_seed(0)


def reference_nosa_linear(qkv, delta_weight, A, q_size=2048, kv_size=256, kv_heads=2):
    """
    PyTorch 原版实现，用来和 fused kernel 对比。
    """
    B, S, D = qkv.shape
    assert D == q_size + 2 * kv_size

    # split
    query_states, key_states, value_states = torch.split(
        qkv, [q_size, kv_size, kv_size], dim=-1
    )

    # contiguous (PyTorch 默认是不一定 contiguous 的)
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    # delta = Linear(kv_size -> kv_heads)，但我们只有 weight，没有 bias
    # 所以手动 matmul
    # [B,S,256] @ [256,2]  → [B,S,2]
    dt_states = value_states @ delta_weight.t()

    # softplus + scale
    cis = A * F.softplus(dt_states)

    return query_states, key_states, value_states, cis


def test_nosa_linear():
    B, S = 16, 1
    q_size = 2048
    kv_heads = 2
    head_dim = 128
    kv_size = kv_heads * head_dim  # 256

    # 构造输入
    qkv = torch.randn(B, S, q_size + 2 * kv_size, device="cuda", dtype=torch.float32)

    # delta: weight-only linear
    delta_weight = torch.randn(kv_heads, kv_size, device="cuda")

    # A: scale
    A = torch.randn(kv_heads, device="cuda")

    # Triton fused 结果
    q_f, k_f, v_f, cis_f = nosa_linear(
        qkv, delta_weight, A,
        q_size=q_size, kv_size=kv_size, kv_heads=kv_heads
    )

    # PyTorch 参考实现
    q_r, k_r, v_r, cis_r = reference_nosa_linear(
        qkv, delta_weight, A,
        q_size=q_size, kv_size=kv_size, kv_heads=kv_heads
    )
    # 比较
    assert torch.allclose(q_f, q_r, atol=1e-5), "Q mismatch!"
    assert torch.allclose(k_f, k_r, atol=1e-5), "K mismatch!"
    assert torch.allclose(v_f, v_r, atol=1e-5), "V mismatch!"
    assert torch.allclose(cis_f, cis_r, atol=1e-5), "CIS mismatch!"

    print("✓ Triton nosa_linear 单测通过！输出与 PyTorch 完全一致。")


import time

def benchmark_nosa():
    B, S = 16, 1
    q_size = 2048
    kv_heads = 2
    head_dim = 128
    kv_size = kv_heads * head_dim  # 256

    qkv = torch.randn(B, S, q_size + 2 * kv_size, device="cuda", dtype=torch.float32)
    delta_weight = torch.randn(kv_heads, kv_size, device="cuda")
    A = torch.randn(kv_heads, device="cuda")

    # ===== Warmup =====
    for _ in range(10):
        _ = reference_nosa_linear(qkv, delta_weight, A, q_size, kv_size, kv_heads)
        _ = nosa_linear(qkv, delta_weight, A, q_size=q_size, kv_size=kv_size, kv_heads=kv_heads)
    torch.cuda.synchronize()

    # ===== Benchmark Reference =====
    iters = 200
    start = time.perf_counter()
    for _ in range(iters):
        _ = reference_nosa_linear(qkv, delta_weight, A, q_size, kv_size, kv_heads)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - start) / iters

    # ===== Benchmark Triton Fused =====
    start = time.perf_counter()
    for _ in range(iters):
        _ = nosa_linear(qkv, delta_weight, A, q_size=q_size, kv_size=kv_size, kv_heads=kv_heads)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - start) / iters

    print("===== Benchmark Results =====")
    print(f"PyTorch reference: {t_ref*1e6:.2f} us / iteration")
    print(f"Triton fused:      {t_fused*1e6:.2f} us / iteration")
    print(f"Speedup:           {t_ref/t_fused:.2f}x")



if __name__ == "__main__":
    test_nosa_linear()
    benchmark_nosa()
