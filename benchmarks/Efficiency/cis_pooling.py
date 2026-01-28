import triton
import triton.language as tl
import torch


@triton.jit
def nosa_mean_pool_kernel(
    cis_ptr,            # [N, H]
    cu_seqlens_ptr,     # int32 [B]
    result_ptr,         # [H, N, M]
    cis_stride_n,        # int32
    cis_stride_h,        # int32
    result_stride_h,     # int32
    result_stride_n,     # int32
    result_stride_m,     # int32
    max_seqlen: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
):
    # grid: (H, B, M)
    tidx_h = tl.program_id(0)  # head
    tidx_b = tl.program_id(1)  # batch idx
    tidx_m = tl.program_id(2)  # window idx


    batch_start = tl.load(cu_seqlens_ptr + tidx_b)
    batch_end = tl.load(cu_seqlens_ptr + tidx_b + 1)

    block_idx = tl.arange(0, kernel_size)

    beg_pos = cis_ptr + tidx_h * cis_stride_h + (batch_start + tidx_m * stride) * cis_stride_n

    block_cis_ptrs = beg_pos + block_idx * cis_stride_n
    mask = (block_idx + tidx_m * stride) < (batch_end - batch_start)
    block_scores = tl.load(
        block_cis_ptrs, 
        mask=mask,
        other=0.0,
    )

    # 对block_scores做平均值，注意mask要对, 分母上是mask的有效元素数
    val_cnt = tl.sum(mask.to(tl.int32), axis=0)
    # acc = tl.sum(block_scores, axis=0) / tl.sum(block_idx + tidx_m * stride < batch_end - batch_start, axis=0)
    acc = tl.sum(block_scores, axis=0) / val_cnt
    
    if tidx_m * stride + kernel_size <= batch_end - batch_start:
        write_pos = result_ptr + tidx_h * result_stride_h + batch_start * result_stride_n + tidx_m * result_stride_m
        write_idx = tl.arange(0, max_seqlen)
        write_ptrs = write_pos + write_idx * result_stride_n
        tl.store(write_ptrs, acc, mask=write_idx < batch_end - batch_start)

def nosa_mean_pooling(cis_score, cu_seqlens, max_seqlen, kernel_size=32, stride=16):
    """
    cis_score: [N, H] (torch.Tensor, float32/bfloat16/float16都行，但triton里先用float32)
    cu_seqlens: [B+1] (torch.int32)
    """
    assert kernel_size == 32 and stride == 16

    N, H = cis_score.shape
    B = cu_seqlens.numel() - 1
    M = max_seqlen // stride - 1  # 每个batch最大窗口数

    result = torch.zeros((H, N, M), dtype=cis_score.dtype, device=cis_score.device)

    grid = (H, B, M)
    nosa_mean_pool_kernel[grid](
        cis_score,
        cu_seqlens,
        result,
        cis_score.stride(0),
        cis_score.stride(1),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        triton.next_power_of_2(max_seqlen),
        N, H, M, kernel_size, stride
    )
    return result


def main():
    torch.manual_seed(0)
    device = "cuda"

    # 模拟数据
    B = 2
    H = 4
    lens = [67, 1432]  # 每个 batch 的长度
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), dim=0)), dtype=torch.int32, device=device)
    N = cu_seqlens[-1].item()
    max_seqlen = max(lens)

    cis_score = torch.randn(N, H, device=device, dtype=torch.bfloat16)

    # Triton 版本
    result = nosa_mean_pooling(cis_score, cu_seqlens, max_seqlen, kernel_size=32, stride=16)

    # PyTorch baseline: 对每个 batch 做 pooling 然后广播
    M = max_seqlen // 16
    baseline = torch.zeros((H, N, M), device=device, dtype=torch.bfloat16)
    for b in range(B):
        start, end = cu_seqlens[b].item(), cu_seqlens[b+1].item()
        seq = cis_score[start:end].T.unsqueeze(0)  # [1, H, L]
        pooled = torch.nn.functional.avg_pool1d(seq, kernel_size=32, stride=16)  # [1, H, m]
        pooled = pooled.squeeze(0)  # [H, m]
        baseline[:, start:end, :pooled.size(-1)] = pooled.unsqueeze(1).expand(H, end-start, pooled.size(-1))

    # 检查差异
    max_diff = (result - baseline).abs().max()
    print("Triton vs PyTorch max diff:", max_diff.item())

if __name__ == "__main__":
    main()
