import triton
import triton.language as tl
import torch

MAX_M = 512

@triton.jit
def nosa_pool_kernel(
    q_ptr,              # [H, B, N]
    cis_ptr,            # [H, B, N]
    max_seq_len,        # int32
    pooled_q_ptr,       # [H, B, M]
    pooled_cis_ptr,     # [H, B, M]
    q_stride_h,         # int32
    q_stride_b,         # int32
    q_stride_n,         # int32
    cis_stride_h,         # int32
    cis_stride_b,         # int32
    cis_stride_n,         # int32
    pooled_q_stride_h,         # int32
    pooled_q_stride_b,         # int32
    pooled_q_stride_m,         # int32
    pooled_cis_stride_h,         # int32
    pooled_cis_stride_b,         # int32
    pooled_cis_stride_m,         # int32
    H,                          # int32
    B,                          # int32
    M,                          # int32
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    set_size: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
):
    # grid: (H, B, M)
    tidx_h = tl.program_id(0)  # head
    tidx_b = tl.program_id(1)  # batch idx
    tidx_m = tl.program_id(2)  # window idx

    block_idx = tl.arange(0, 8)
    n_offset = tidx_m * (set_size - 1) - padding
    q_beg_pos = q_ptr + tidx_h * q_stride_h + tidx_b * q_stride_b + n_offset * q_stride_n
    q_block_ptrs = q_beg_pos + block_idx * q_stride_n
    cis_beg_pos = cis_ptr + tidx_h * cis_stride_h + tidx_b * cis_stride_b + n_offset * cis_stride_n
    cis_block_ptrs = cis_beg_pos + block_idx * cis_stride_n
    mask = (block_idx + n_offset < max_seq_len) & (block_idx + n_offset >= 0) & (block_idx < set_size)
    q_block_scores = tl.load(
        q_block_ptrs,
        mask=mask,
        other=-float("inf"),
    )
    acc_q = tl.max(q_block_scores, axis=0)
    cis_block_scores = tl.load(
        cis_block_ptrs,
        mask=mask,
        other=-float("inf"),
    )
    acc_cis = tl.max(cis_block_scores, axis=0)

    boundary_mask = (tidx_m < init_blocks) | (M - tidx_m <= local_blocks)
    out_q = tl.where(boundary_mask, float("inf"), acc_q)
    out_cis = tl.where(boundary_mask, float("inf"), acc_cis)

    tl.store(
        pooled_q_ptr + tidx_b * pooled_q_stride_b + tidx_h * pooled_q_stride_h + tidx_m * pooled_q_stride_m,
        out_q,
        mask=tidx_m < M,
    )

    tl.store(
        pooled_cis_ptr + tidx_b * pooled_cis_stride_b + tidx_h * pooled_cis_stride_h + tidx_m * pooled_cis_stride_m,
        out_cis,
        mask=tidx_m < M,
    )


# 这里假定：1) 必须是decode，2) 格式是齐头pad转unpad，所有batch一样长
def nosa_pooling(q_score, cis_score, max_seqlen, q_score_pooled, cis_score_pooled, block_size=32, stride=16, padding=1, set_size=5, init_blocks=1, local_blocks=16):
    # input shape: (H, B, N)
    # pooled shape: (H, B, M)
    H, B = q_score.shape[:2]
    M = q_score_pooled.shape[-1]
    # assert M < MAX_M

    grid = (H, B, M)
    nosa_pool_kernel[grid](
        q_score,
        cis_score,
        max_seqlen,
        q_score_pooled,
        cis_score_pooled,
        q_score.stride(0),
        q_score.stride(1),
        q_score.stride(2),
        cis_score.stride(0),
        cis_score.stride(1),
        cis_score.stride(2),
        q_score_pooled.stride(0),
        q_score_pooled.stride(1),
        q_score_pooled.stride(2),
        cis_score_pooled.stride(0),
        cis_score_pooled.stride(1),
        cis_score_pooled.stride(2),
        H, B, M, block_size, stride, padding, set_size, init_blocks, local_blocks+1,
    )
    return


def main():
    import torch
    torch.manual_seed(0)
    device = "cuda"

    # ---------------------------
    # 模拟输入数据
    # ---------------------------
    H = 4
    B = 2
    N = 1432          # 所有 batch 等长
    kernel_size = 32
    stride = 16

    # Triton 输入: [H, B, N]
    q_score  = torch.randn(H, B, N, device=device, dtype=torch.bfloat16)
    cis_score = torch.randn(H, B, N, device=device, dtype=torch.bfloat16)

    # 输出 M 计算方式与 Triton kernel 一致
    M = 88

    q_score_pooled = torch.empty(H, B, M, device=device, dtype=torch.bfloat16)
    cis_score_pooled = torch.empty(H, B, M, device=device, dtype=torch.bfloat16)

    # ---------------------------
    # 调 Triton kernel
    # ---------------------------
    nosa_pooling(
        q_score,
        cis_score,
        N,                      # max_seqlen
        q_score_pooled,
        cis_score_pooled,
        kernel_size,
        stride,
    )

    # ---------------------------
    # PyTorch baseline
    # ---------------------------
    # baseline: shape [H, B, M]
    baseline_q = torch.zeros(H, B, M, device=device, dtype=torch.float32)
    baseline_cis = torch.zeros(H, B, M, device=device, dtype=torch.float32)

    for h in range(H):
        for b in range(B):
            seq_q = q_score[h, b].unsqueeze(0).unsqueeze(0)     # [1,1,N]
            seq_c = cis_score[h, b].unsqueeze(0).unsqueeze(0)

            pooled_q = torch.nn.functional.max_pool1d(seq_q.float(), kernel_size, stride)  # Triton 用 max
            pooled_cis = torch.nn.functional.max_pool1d(seq_c.float(), kernel_size, stride)

            baseline_q[h, b, :pooled_q.size(-1)] = pooled_q.squeeze(0).squeeze(0)
            baseline_cis[h, b, :pooled_cis.size(-1)] = pooled_cis.squeeze(0).squeeze(0)

    # Triton 是 bfloat16 → 转 float32 对比
    diff_q = (q_score_pooled.float() - baseline_q).abs().max()
    diff_cis = (cis_score_pooled.float() - baseline_cis).abs().max()
    print("Max diff q   :", diff_q.item())
    print("Max diff cis :", diff_cis.item())

if __name__ == "__main__":
    main()
