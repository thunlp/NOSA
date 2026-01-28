#include <flashinfer/attention/prefill.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward(
    torch::Tensor workspace_buffer, torch::Tensor qo_indptr,
    unsigned int batch_size, unsigned int num_qo_heads,
    unsigned int num_kv_heads, unsigned int head_dim) {
  // NOTE(Zihao): not necessary to be a CUDA tensor
  CHECK_CONTIGUOUS(qo_indptr);
  CHECK_CONTIGUOUS(workspace_buffer);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, workspace_buffer);

  // TODO(Zihao): support dispatching to different index data types.
  CHECK_EQ(qo_indptr.scalar_type(), torch::kInt32);
  size_t workspace_size_in_bytes =
      workspace_buffer.size(0) * workspace_buffer.element_size();
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  handler_.SetCUDAStream(torch_current_stream);

  cudaError_t status = handler_.BeginForward(
      static_cast<void *>(workspace_buffer.data_ptr()), workspace_size_in_bytes,
      static_cast<int32_t *>(qo_indptr.data_ptr()), batch_size, num_qo_heads,
      num_kv_heads, head_dim);
  TORCH_CHECK(status == cudaSuccess,
              "BatchPrefillWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}

void BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward() {
  handler_.EndForward();
}

template <PageStorage page_storage, QKVLayout kv_layout, uint32_t PAGE_SIZE, uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM, PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION,
          bool CAUSAL, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream) {
  float* tmp = nullptr;
  IdType* request_indices = nullptr;
  IdType* tile_indices = nullptr;
  uint32_t num_frags_x = 0U;
  uint32_t num_qo_tiles = 0U;
  if (handler->IsForwardStarted()) {
    request_indices = handler->GetRequestIndices<IdType>();
    tile_indices = handler->GetTileIndices<IdType>();
    num_frags_x = handler->GetNumFragsX();
    num_qo_tiles = handler->GetNumQOTiles();
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call BatchPrefillHandler's BeginForward() before calling "
               "BatchPrefillWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }

  DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, {
    return BatchPrefillWithPagedKVCacheDispatched<
        page_storage, kv_layout, NUM_FRAGS_X, PAGE_SIZE, GROUP_SIZE, HEAD_DIM, pos_encoding_mode,
        ALLOW_FP16_QK_REDUCTION, CAUSAL, DTypeIn, DTypeOut, IdType>(
        q, request_indices, tile_indices, qo_indptr, q_offset, paged_kv, o, tmp, lse, num_qo_tiles,
        sm_scale, rope_scale, rope_theta, stream);
  });
  return cudaSuccess;
}

std::vector<torch::Tensor> BatchPrefillWithPagedKVCachePyTorchWrapper::Forward(
    torch::Tensor q, torch::Tensor qo_indptr, torch::Tensor paged_kv_data,
    torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len, bool causal,
    unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
    float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
  CHECK_INPUT(q);
  CHECK_INPUT(qo_indptr);
  CHECK_INPUT(paged_kv_data);
  CHECK_INPUT(paged_kv_indptr);
  CHECK_INPUT(paged_kv_indices);
  CHECK_INPUT(paged_kv_last_page_len);
  CHECK_DIM(3, q);         // (nnz_qo, H_qo, D)
  CHECK_DIM(1, qo_indptr); // (B + 1,)
  // [max_num_pages, 2, num_kv_heads, page_size, head_dim] for HND
  // [max_num_pages, 2, page_size, num_kv_heads, head_dim] for NHD
  CHECK_DIM(5, paged_kv_data);
  CHECK_DIM(1, paged_kv_indptr);        // (B + 1,)
  CHECK_DIM(1, paged_kv_indices);       // (nnz_kv,)
  CHECK_DIM(1, paged_kv_last_page_len); // (B,)
  int64_t batch_size = qo_indptr.size(0) - 1;
  int64_t nnz_qo = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t num_kv_heads, page_size;
  if (kv_layout_ == QKVLayout::kHND) {
    num_kv_heads = paged_kv_data.size(2);
    page_size = paged_kv_data.size(3);
  } else {
    page_size = paged_kv_data.size(2);
    num_kv_heads = paged_kv_data.size(3);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  CHECK_EQ(qo_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(paged_kv_last_page_len.size(0), batch_size);
  CHECK_EQ(paged_kv_data.size(1), 2);
  CHECK_EQ(paged_kv_data.size(4), head_dim);
  // TODO(Zihao): support dispatching to different index data types.
  CHECK_EQ(qo_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_last_page_len.scalar_type(), torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();
  torch::Tensor o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({nnz_qo, num_qo_heads}, q.options()).to(torch::kFloat32);
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    return DISPATCH_kv_layout(kv_layout_, KV_LAYOUT, [&] {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type *>(paged_kv_data.data_ptr()),
          static_cast<int32_t *>(paged_kv_indices.data_ptr()),
          static_cast<int32_t *>(paged_kv_indptr.data_ptr()),
          static_cast<int32_t *>(paged_kv_last_page_len.data_ptr()));
      return DISPATCH_group_size(num_qo_heads / num_kv_heads, GROUP_SIZE, [&] {
        return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
          return DISPATCH_causal(causal, CAUSAL, [&] {
            return DISPATCH_allow_fp16_qk_reduction(
                allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                  return DISPATCH_pos_encoding_mode(
                      PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE,
                      [&] {
                        return DISPATCH_page_size(page_size, PAGE_SIZE, [&] {
                          cudaError_t status =
                              BatchPrefillWithPagedKVCacheWrapperDispatched<
                                  PageStorage::kIndices, KV_LAYOUT, PAGE_SIZE,
                                  GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE,
                                  ALLOW_FP16_QK_REDUCTION, CAUSAL, c_type,
                                  c_type, int32_t>(
                                  &handler_,
                                  static_cast<c_type *>(q.data_ptr()),
                                  static_cast<int32_t *>(qo_indptr.data_ptr()),
                                  /*q_offset=*/nullptr, paged_kv,
                                  static_cast<c_type *>(o.data_ptr()),
                                  /*lse=*/
                                      return_lse
                                      ? static_cast<float *>(lse.data_ptr())
                                      : nullptr,
                                  sm_scale, rope_scale, rope_theta,
                                  /*stream=*/torch_current_stream);
                          TORCH_CHECK(status == cudaSuccess,
                                      "BatchPrefillWithPagedKVCache failed "
                                      "with error code ",
                                      cudaGetErrorString(status));
                          return true;
                        });
                      });
                });
          });
        });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
