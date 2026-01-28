#include <cstdint>
#include <flashinfer/page.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

void append_paged_kv_cache_prefill(
    torch::Tensor k, // [bsz, kv_len, n_kv_heads, head_dim]
    torch::Tensor v, // [bsz, kv_len, n_kv_heads, head_dim]
    torch::Tensor
        kv_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    torch::Tensor kv_indices,       // [bsz, num_pages]
    torch::Tensor kv_indptr,        // [bsz+1]
    torch::Tensor kv_last_page_len, // [bsz]
    unsigned int layout) {

  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);

  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(5, kv_data);
  CHECK_DIM(2, kv_indices);
  CHECK_DIM(1, kv_indptr);

  CHECK_GE(k.size(1), 2);
  CHECK_GE(v.size(1), 2);
  CHECK_EQ(k.size(0), v.size(0));
  CHECK_EQ(k.size(1), v.size(1));
  CHECK_EQ(k.size(2), v.size(2));
  CHECK_EQ(k.size(3), v.size(3));
  CHECK_EQ(k.size(0), kv_indices.size(0));
  CHECK_EQ(k.size(0) + 1, kv_indptr.size(0));
  CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);

  int32_t batch_size = k.size(0);
  int32_t seq_len = k.size(1);
  int32_t num_kv_heads = k.size(2);
  int32_t head_dim = k.size(3);
  int32_t page_size;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kHND) {
    page_size = kv_data.size(3);
    CHECK_EQ(kv_data.size(2), num_kv_heads);
    CHECK_EQ(kv_data.size(4), head_dim);
  } else {
    page_size = kv_data.size(2);
    CHECK_EQ(kv_data.size(3), num_kv_heads);
    CHECK_EQ(kv_data.size(4), head_dim);
  }

  torch::Tensor append_indptr = torch::arange(0, (batch_size + 1) * seq_len,
                                              seq_len, kv_indices.options());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(k.scalar_type(), c_type, [&] {
    return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type *>(kv_data.data_ptr()),
          static_cast<int32_t *>(kv_indices.data_ptr()),
          static_cast<int32_t *>(kv_indptr.data_ptr()),
          static_cast<int32_t *>(kv_last_page_len.data_ptr()));

      cudaError_t status =
          AppendPagedKVCache<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t>(
              paged_kv, static_cast<c_type *>(k.data_ptr()),
              static_cast<c_type *>(v.data_ptr()),
              static_cast<int32_t *>(append_indptr.data_ptr()));

      TORCH_CHECK(status == cudaSuccess,
                  "append_paged_kv_cache_prefill failed with error code ",
                  cudaGetErrorString(status));
      return true;
    });
  });
}

void append_paged_kv_cache_decode(
    torch::Tensor k, // [bsz, 1, n_kv_heads, head_dim]
    torch::Tensor v, // [bsz, 1, n_kv_heads, head_dim]
    torch::Tensor
        kv_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    torch::Tensor kv_indices,       // [bsz, num_pages]
    torch::Tensor kv_indptr,        // [bsz+1]
    torch::Tensor kv_last_page_len, // [bsz]
    unsigned int layout) {

  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);

  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(5, kv_data);
  CHECK_DIM(2, kv_indices);
  CHECK_DIM(1, kv_indptr);

  CHECK_EQ(k.size(1), 1);
  CHECK_EQ(v.size(1), 1);
  CHECK_EQ(k.size(0), v.size(0));
  CHECK_EQ(k.size(1), v.size(1));
  CHECK_EQ(k.size(2), v.size(2));
  CHECK_EQ(k.size(3), v.size(3));
  CHECK_EQ(k.size(0), kv_indices.size(0));
  CHECK_EQ(k.size(0) + 1, kv_indptr.size(0));
  CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);

  size_t batch_size = k.size(0);
  size_t num_kv_heads = k.size(2);
  size_t head_dim = k.size(3);
  size_t page_size;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kHND) {
    page_size = kv_data.size(3);
    CHECK_EQ(kv_data.size(2), num_kv_heads);
    CHECK_EQ(kv_data.size(4), head_dim);
  } else {
    page_size = kv_data.size(2);
    CHECK_EQ(kv_data.size(3), num_kv_heads);
    CHECK_EQ(kv_data.size(4), head_dim);
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(k.scalar_type(), c_type, [&] {
    return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type *>(kv_data.data_ptr()),
          static_cast<int32_t *>(kv_indices.data_ptr()),
          static_cast<int32_t *>(kv_indptr.data_ptr()),
          static_cast<int32_t *>(kv_last_page_len.data_ptr()));

      cudaError_t status = AppendPagedKVCacheDecode(
          paged_kv, static_cast<c_type *>(k.data_ptr()),
          static_cast<c_type *>(v.data_ptr()));

      TORCH_CHECK(status == cudaSuccess,
                  "Append_kv_cache_decode failed with error code ",
                  cudaGetErrorString(status));
      return true;
    });
  });
}
