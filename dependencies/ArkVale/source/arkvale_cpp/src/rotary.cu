#include "pytorch_extension_utils.h"
#include <cstdint>
#include <flashinfer/pos_enc.cuh>

using namespace flashinfer;

void qk_apply_rotary_in_place(
    torch::Tensor q, // [bsz, len, n_qo_heads, head_dim]
    torch::Tensor k, // [bsz, len, n_kv_heads, head_dim]
    unsigned int past_kv_len, float rope_scale, float rope_theta) {

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(1), k.size(1));
  CHECK_EQ(q.size(3), k.size(3));

  size_t batch_size = q.size(0);
  size_t seq_len = q.size(1);
  size_t head_dim = q.size(3);
  size_t num_qo_heads = q.size(2);
  size_t num_kv_heads = k.size(2);

  torch::Tensor indptr = torch::arange(0, (batch_size + 1) * seq_len, seq_len,
                                       q.options().dtype(torch::kInt32));
  torch::Tensor offsets =
      torch::full({(int)batch_size}, past_kv_len, indptr.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotaryInPlace<c_type, int32_t>(
        static_cast<c_type *>(q.data_ptr()),
        static_cast<c_type *>(k.data_ptr()),
        static_cast<int32_t *>(indptr.data_ptr()),
        static_cast<int32_t *>(offsets.data_ptr()), batch_size, num_qo_heads,
        num_kv_heads, head_dim, rope_scale, rope_theta);

    TORCH_CHECK(status == cudaSuccess,
                "qk_apply_rotary_in_place failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}

template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
__global__ void BatchQKQApplyRotaryInPlaceKernel(DType* __restrict__ q, DType* __restrict__ k, DType* __restrict__ q1,
                                                IdType* __restrict__ indptr,
                                                IdType* __restrict__ offsets, uint32_t batch_size,
                                                uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    freq[i] =
        rope_rcp_scale *
        __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr =
            q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                    indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0, seq_len, num_qo_heads);
        q_vec = vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty);
        q_vec.cast_store(q_ptr + tx * vec_size);
      }
    }
  } else if (bx < batch_size * (num_qo_heads + num_kv_heads)) {
    // apply rotary to k
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr =
            k + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                    indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0, seq_len, num_kv_heads);
        k_vec = vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty);
        k_vec.cast_store(k_ptr + tx * vec_size);
      }
    }
  } else {
    // apply rotary to q1
    const uint32_t batch_idx = (bx - batch_size * (num_qo_heads + num_kv_heads)) / num_qo_heads;
    const uint32_t qo_head_idx = (bx - batch_size * (num_qo_heads + num_kv_heads)) % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr =
            q1 + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                    indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0, seq_len, num_qo_heads);
        q_vec = vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty);
        q_vec.cast_store(q_ptr + tx * vec_size);
      }
    }
  }
}

template <typename DType, typename IdType>
cudaError_t BatchQKQApplyRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k, DType* __restrict__ q1,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t head_dim,
                                      float rope_scale = 1.f, float rope_theta = 1e4,
                                      cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = num_threads / bdx;
    dim3 nblks(batch_size * (num_qo_heads + num_kv_heads + num_qo_heads));
    dim3 nthrs(bdx, bdy);
    auto kernel = BatchQKQApplyRotaryInPlaceKernel<HEAD_DIM, vec_size, bdx, DType, IdType>;
    void* args[] = {(void*)&q,
                    (void*)&k,
                    (void*)&q1,
                    (void*)&indptr,
                    (void*)&offsets,
                    (void*)&batch_size,
                    (void*)&num_qo_heads,
                    (void*)&num_kv_heads,
                    (void*)&rope_rcp_scale,
                    (void*)&rope_rcp_theta};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });

  return cudaSuccess;
}

void qkq_apply_rotary_in_place(
    torch::Tensor q, // [bsz, len, n_qo_heads, head_dim]
    torch::Tensor k, // [bsz, len, n_kv_heads, head_dim]
    torch::Tensor q1, // [bsz, len, n_qo_heads, head_dim]
    unsigned int past_kv_len, float rope_scale, float rope_theta) {

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(q1);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, q1);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(1), k.size(1));
  CHECK_EQ(q.size(3), k.size(3));
  CHECK_EQ(q1.size(0), k.size(0));
  CHECK_EQ(q1.size(1), k.size(1));
  CHECK_EQ(q1.size(3), k.size(3));

  size_t batch_size = q.size(0);
  size_t seq_len = q.size(1);
  size_t head_dim = q.size(3);
  size_t num_qo_heads = q.size(2);
  size_t num_kv_heads = k.size(2);

  torch::Tensor indptr = torch::arange(0, (batch_size + 1) * seq_len, seq_len,
                                       q.options().dtype(torch::kInt32));
  torch::Tensor offsets =
      torch::full({(int)batch_size}, past_kv_len, indptr.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKQApplyRotaryInPlace<c_type, int32_t>(
        static_cast<c_type *>(q.data_ptr()),
        static_cast<c_type *>(k.data_ptr()),
        static_cast<c_type *>(q1.data_ptr()),
        static_cast<int32_t *>(indptr.data_ptr()),
        static_cast<int32_t *>(offsets.data_ptr()), batch_size, num_qo_heads,
        num_kv_heads, head_dim, rope_scale, rope_theta);

    TORCH_CHECK(status == cudaSuccess,
                "qkq_apply_rotary_in_place failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}