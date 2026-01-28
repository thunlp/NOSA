#include "flashinfer/pos_enc.cuh"
#include <flashinfer/attention/decode.cuh>

using namespace flashinfer;

template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx,
          uint32_t tile_size, typename T, typename U>
__device__ __forceinline__ void
compute_cuboid_scores(const T *k_smem, const T *v_smem,
                      // uint32_t compute_stage_idx,
                      const vec_t<float, vec_size> &q_vec,
                      // const vec_t<float, vec_size>& freq,
                      uint32_t kv_idx_base, uint32_t iter_base,
                      uint32_t iter_bound, U *o
                      // const int32_t q_offset,
                      // float alibi_slope,
                      // float* s,
                      // state_t<vec_size>& st
) {
  static_assert(pos_encoding_mode == PosEncodingMode::kNone);
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  // float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    vec_t<float, vec_size> v_vec;
    // if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    //   // apply rotary embedding for all rows in k matrix of kv-cache
    //   k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size,
    //   freq,
    //                                               kv_idx_base + tz *
    //                                               tile_size + j);
    // } else {
    // do not apply rotary embedding
    k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);
    v_vec.cast_load(v_smem + (j * bdx + tx) * vec_size);
    // }
    // s[j] = 0.f;
    float o_tmp = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      // s[j] += q_vec[i] * k_vec[i];
      o_tmp += max(q_vec[i] * k_vec[i], q_vec[i] * v_vec[i]);
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      // s[j] += math::shfl_xor_sync(s[j], offset);
      o_tmp += math::shfl_xor_sync(o_tmp, offset);
    }
    // s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -5e4;
    if (iter_base + tz * tile_size + j < iter_bound && tx == 0)
      o[kv_idx_base + tz * tile_size + j] = static_cast<U>(o_tmp);
    // if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
    //   s[j] += alibi_slope * float(int(kv_idx_base + tz * tile_size + j) -
    //   q_offset);
    // }
    // st.m = max(st.m, s[j]);
  }

  //   float o_scale = math::ptx_exp2(m_prev - st.m);
  //   st.d *= o_scale;
  // #pragma unroll
  //   for (uint32_t j = 0; j < tile_size; ++j) {
  //     s[j] = math::ptx_exp2(s[j] - st.m);
  //     st.d += s[j];
  //   }
  // #pragma unroll
  //   for (uint32_t i = 0; i < vec_size; ++i) {
  //     st.o[i] = st.o[i] * o_scale;
  //   }
}

template <bool partition_kv, PosEncodingMode pos_encoding_mode,
          uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn,
          typename DTypeOut, typename IdType>
__global__ void EstimateScoresKernel(
    DTypeIn *__restrict__ q,
    // IdType* __restrict__ q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    // kv_partition_info_t<IdType> kv_partition_info,
    DTypeOut *__restrict__ o
    // ,DTypeOut* __restrict__ tmp, float* __restrict__ lse, float sm_scale,
    // float rope_rcp_scale, float rope_rcp_theta
) {
  static_assert(!partition_kv);
  static_assert(pos_encoding_mode == PosEncodingMode::kNone);
  auto block = cg::this_thread_block();
  // sm_scale *= math::log2e;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t kv_head_idx = blockIdx.y;
  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  const uint32_t num_qo_heads = gridDim.y * bdy;
  const float alibi_slope =
      get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
  // const uint32_t cur_chunk_start = partition_kv ?
  // kv_partition_info.chunk_start_pos[batch_idx] : 0U;
  const uint32_t cur_chunk_start = 0U;
  const uint32_t cur_page_indptr_begin = paged_kv.indptr[batch_idx],
                 cur_page_indptr_end = paged_kv.indptr[batch_idx + 1];
  const uint32_t cur_last_page_len = paged_kv.last_page_len[batch_idx];
  const uint32_t kv_chunk_len =
      cur_page_indptr_begin != cur_page_indptr_end
          ? (cur_page_indptr_end - cur_page_indptr_begin - 1) *
                    paged_kv.page_size +
                cur_last_page_len
          : 0;

  // uint32_t acc_last_page_rest = 0U;
  // for (uint32_t i = 0; i + 1 < batch_idx; ++i)
  //   acc_last_page_rest += paged_kv.page_size - paged_kv.last_page_len[i];
  // const uint32_t acc_kv_chunk_len =
  //     paged_kv.indptr[batch_idx] * paged_kv.page_size - acc_last_page_rest;

  // const uint32_t seq_len =
  //     partition_kv ? kv_partition_info.seq_lens_before_partition[batch_idx] :
  //     kv_chunk_len;
  // const uint32_t seq_len = kv_chunk_len;
  // const uint32_t mapped_batch_idx =
  //     partition_kv ? kv_partition_info.batch_idx_map[batch_idx] : batch_idx;
  const uint32_t mapped_batch_idx = batch_idx;

  extern __shared__ uint8_t smem[];
  DTypeIn *k_smem = (DTypeIn *)smem;
  DTypeIn *v_smem =
      (DTypeIn *)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz *
                             head_dim * sizeof(DTypeIn));
  DTypeIn **k_ptrs_smem =
      (DTypeIn **)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                              head_dim * sizeof(DTypeIn));
  // float *smem_md = (float *)(smem + 2 * num_stages_smem * tile_size_per_bdx *
  //                                       bdy * bdz * head_dim *
  //                                       sizeof(DTypeIn));

  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  // vec_t<float, vec_size> freq;
  // int32_t q_offset_val = q_offset == nullptr ? (seq_len - 1) :
  // q_offset[mapped_batch_idx];
  //   if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
  // #pragma unroll
  //     for (uint32_t i = 0; i < vec_size; ++i) {
  //       freq[i] = rope_rcp_scale *
  //                 __powf(rope_rcp_theta,
  //                        float(2 * ((tx * vec_size + i) % (head_dim / 2))) /
  //                        float(head_dim));
  //     }
  //     // apply rotary embedding to q matrix
  //     q_vec = vec_apply_llama_rope<vec_size, bdx>(
  //         q + (mapped_batch_idx * num_qo_heads + qo_head_idx) * head_dim,
  //         freq, q_offset_val);
  //   } else {
  // do not apply rotary embedding to q matrix
  q_vec.cast_load(q +
                  (mapped_batch_idx * num_qo_heads + qo_head_idx) * head_dim +
                  tx * vec_size);
  // }
  // #pragma unroll
  //   for (uint32_t i = 0; i < vec_size; ++i) {
  //     q_vec[i] *= sm_scale;
  //   }
  block.sync();

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_k_ptr(
            cur_page_indptr_begin +
                (((j * bdz + tz) * bdy + ty) * bdx + tx) / paged_kv.page_size,
            kv_head_idx,
            (((j * bdz + tz) * bdy + ty) * bdx + tx) % paged_kv.page_size, 0,
            last_indptr);
  }
  block.sync();

  DTypeIn *k_ptrs[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] +
          tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kNoFill>(
          k_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          k_ptrs[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    // cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeIn *v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kFillZero>(
          v_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          v_ptr,
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  // state_t<vec_size> st;
  // float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0;
       iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_k_ptr(
                cur_page_indptr_begin +
                    ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                     ((j * bdz + tz) * bdy + ty) * bdx + tx) /
                        paged_kv.page_size,
                kv_head_idx,
                ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                 ((j * bdz + tz) * bdy + ty) * bdx + tx) %
                    paged_kv.page_size,
                0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    // compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
    //     k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
    //     stage_idx, q_vec, freq, (paged_kv.rope_pos_offset == nullptr ? 0 :
    //     paged_kv.rope_pos_offset[mapped_batch_idx]) +
    //         cur_chunk_start + iter * tile_size_per_bdx * bdy * bdz,
    //     iter * tile_size_per_bdx * bdy * bdz, kv_chunk_len, q_offset_val,
    //     alibi_slope, s, st);
    compute_cuboid_scores<pos_encoding_mode, vec_size, bdx,
                          bdy * tile_size_per_bdx>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        q_vec,
        (paged_kv.rope_pos_offset == nullptr
             ? 0
             : paged_kv.rope_pos_offset[mapped_batch_idx]) +
            cur_chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, kv_chunk_len,
        // o + acc_kv_chunk_len * num_qo_heads + qo_head_idx * kv_chunk_len
        o + (batch_idx * num_qo_heads + qo_head_idx) * kv_chunk_len);

    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy +
                       ty) *
                          tile_size_per_bdx +
                      j] +
          tx * vec_size;
    }
    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kNoFill>(
          k_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          k_ptrs[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) *
                      tile_size_per_bdx +
                  j <
              kv_chunk_len);
    }
    // cp_async::commit_group();

    // update m/d/o states
    // cp_async::wait_group<2 * num_stages_smem - 1>();
    // block.sync();
    // update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
    //     v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
    //     s, stage_idx, st);
    // block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeIn *v_ptr = k_ptrs[j] + paged_kv.kv_offset_delta();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kFillZero>(
          v_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          v_ptr,
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) *
                      tile_size_per_bdx +
                  j <
              kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  // block.sync();

  // sync local state of all warps inside a threadblock
  // sync_state<vec_size, bdx, bdy, bdz>(st, reinterpret_cast<float*>(smem),
  // smem_md); st.normalize();

  // if constexpr (partition_kv) {
  //   st.o.cast_store(tmp + (batch_idx * num_qo_heads + qo_head_idx) * head_dim
  //   + tx * vec_size); float* tmp_lse = (float*)(tmp + paged_kv.batch_size *
  //   num_qo_heads * head_dim); tmp_lse[batch_idx * num_qo_heads + qo_head_idx]
  //   = st.get_lse();
  // } else {
  //   st.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim +
  //   tx * vec_size);
  //   // write lse
  //   if (lse != nullptr) {
  //     lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
  //   }
  // }
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PageStorage page_storage,
          QKVLayout kv_layout, PosEncodingMode POS_ENCODING_MODE,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t EstimateScoresDispatched(
    DTypeIn *q,
    // IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    // kv_partition_info_t<IdType> kv_partition_info,
    DTypeOut *o,
    // DTypeOut* tmp, float* lse, float sm_scale, float rope_scale, float
    // rope_theta,
    cudaStream_t stream) {
  // const float rope_rcp_scale = 1.f / rope_scale;
  // const float rope_rcp_theta = 1.f / rope_theta;
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t batch_size = paged_kv.batch_size;
  // const uint32_t num_qo_heads = num_kv_heads * GROUP_SIZE;

  constexpr uint32_t vec_size =
      std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
  constexpr uint32_t num_stages_smem = 2U;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  constexpr uint32_t tile_size_per_bdx =
      GROUP_SIZE == 1 ? (sizeof(DTypeIn) == 1 ? 2U : 4U) : 1U;
  const uint32_t smem_size =
      2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * HEAD_DIM *
          sizeof(DTypeIn) +
      std::max(tile_size_per_bdx * num_threads * sizeof(DTypeIn *),
               2 * bdy * bdz * sizeof(float));

  // if (tmp == nullptr) {
  // do not use partition-kv kernel
  dim3 nblks(batch_size, num_kv_heads);
  dim3 nthrs(bdx, bdy, bdz);
  auto kernel = EstimateScoresKernel<
      /*partition_kv=*/false, POS_ENCODING_MODE, num_stages_smem,
      tile_size_per_bdx, vec_size, bdx, bdy, bdz, page_storage, kv_layout,
      DTypeIn, DTypeOut, IdType>;
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  void *args[] = {
      (void *)&q,
      // (void*)&q_offset,
      (void *)&paged_kv,
      // (void*)&kv_partition_info,
      (void *)&o,
      // (void*)&tmp,
      // (void*)&lse,
      // (void*)&sm_scale,
      // (void*)&rope_rcp_scale,
      // (void*)&rope_rcp_theta
  };
  FLASHINFER_CUDA_CALL(
      cudaLaunchKernel((void *)kernel, nblks, nthrs, args, smem_size, stream));
  // } else {
  //   // use partition-kv kernel
  //   auto partition_kv_kernel =
  //       EstimateScoresKernel</*partition_kv=*/true, POS_ENCODING_MODE,
  //       num_stages_smem,
  //                                         tile_size_per_bdx, vec_size, bdx,
  //                                         bdy, bdz, page_storage, kv_layout,
  //                                         DTypeIn, DTypeOut, IdType>;
  //   FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
  //       partition_kv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
  //       smem_size));
  //   void* args[] = {(void*)&q,
  //                   (void*)&q_offset,
  //                   (void*)&paged_kv,
  //                   (void*)&kv_partition_info,
  //                   (void*)&o,
  //                   (void*)&tmp,
  //                   (void*)&lse,
  //                   (void*)&sm_scale,
  //                   (void*)&rope_rcp_scale,
  //                   (void*)&rope_rcp_theta};
  //   dim3 nblks(batch_size, num_kv_heads);
  //   dim3 nthrs(bdx, bdy, bdz);
  //   FLASHINFER_CUDA_CALL(
  //       cudaLaunchKernel((void*)partition_kv_kernel, nblks, nthrs, args,
  //       smem_size, stream));
  //   FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
  //       tmp, (float*)(tmp + batch_size * num_qo_heads * HEAD_DIM),
  //       kv_partition_info.chunk_indptr, o, lse,
  //       kv_partition_info.batch_size_before_partition, num_qo_heads,
  //       HEAD_DIM, stream));
  // }

  return cudaSuccess;
}

#include "pytorch_extension_utils.h"

template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t
EstimateScores(DTypeIn *q,
               // IdType* q_offset,
               paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
               //  kv_partition_info_t<IdType> kv_partition_info,
               DTypeOut *o,
               // DTypeOut* tmp, float* lse,
               uint32_t num_qo_heads,
               PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
               // std::optional<float> maybe_sm_scale = std::nullopt, float
               // rope_scale = 1.f, float rope_theta = 1e4,
               cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  // const float sm_scale = maybe_sm_scale.value_or(1.f /
  // std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads
            << " is not a multiple of num_kv_heads " << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_group_size(num_qo_heads / num_kv_heads, GROUP_SIZE, [&] {
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_pos_encoding_mode(
          pos_encoding_mode, POS_ENCODING_MODE, [&] {
            cudaError_t status =
                EstimateScoresDispatched<GROUP_SIZE, HEAD_DIM, page_storage,
                                         kv_layout, POS_ENCODING_MODE, DTypeIn,
                                         DTypeOut, IdType>(
                    q,
                    // q_offset,
                    paged_kv,
                    // kv_partition_info,
                    o,
                    // tmp, lse, sm_scale, rope_scale, rope_theta,
                    stream);
            TORCH_CHECK(status == cudaSuccess,
                        "EstimateScores failed with error code ",
                        cudaGetErrorString(status));
            return true;
          });
    });
  });

  return cudaSuccess;
}

#include <torch/extension.h>

torch::Tensor estimate_scores(torch::Tensor q, // [bsz, 1, num_heads, head_dim]
                              torch::Tensor dg_data,
                              torch::Tensor dg_indices, // [bsz, num_pages]
                              torch::Tensor dg_indptr,  // [bsz+1]
                              torch::Tensor dg_last_page_len, // [bsz]
                              unsigned int dg_seq_len, unsigned int layout,
                              unsigned int n_groups) {
  CHECK_INPUT(q);
  CHECK_INPUT(dg_data);
  CHECK_INPUT(dg_indices);
  CHECK_INPUT(dg_indptr);

  CHECK_DIM(4, q);
  CHECK_DIM(5, dg_data);
  CHECK_DIM(2, dg_indices);
  CHECK_DIM(1, dg_indptr);

  CHECK_EQ(q.size(1), 1);
  CHECK_EQ(dg_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(dg_indptr.scalar_type(), torch::kInt32);

  int32_t batch_size = q.size(0);
  int32_t num_qo_heads = q.size(2);
  int32_t head_dim = q.size(3);
  int32_t page_size, num_kv_heads;

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kHND) {
    page_size = dg_data.size(3);
    num_kv_heads = dg_data.size(2);
    CHECK_EQ(dg_data.size(4), head_dim);
  } else {
    page_size = dg_data.size(2);
    num_kv_heads = dg_data.size(3);
    CHECK_EQ(dg_data.size(4), head_dim);
  }

  torch::Tensor o =
      torch::empty({batch_size, num_qo_heads, dg_seq_len}, q.options());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    return DISPATCH_kv_layout(kv_layout, KV_LAYOUT, [&] {
      paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
          num_kv_heads, page_size, head_dim, batch_size,
          static_cast<c_type *>(dg_data.data_ptr()),
          static_cast<int32_t *>(dg_indices.data_ptr()),
          static_cast<int32_t *>(dg_indptr.data_ptr()),
          static_cast<int32_t *>(dg_last_page_len.data_ptr()));
      cudaError_t status =
          EstimateScores<PageStorage::kIndices, KV_LAYOUT, c_type, c_type,
                         int32_t>(static_cast<c_type *>(q.data_ptr()), paged_kv,
                                  static_cast<c_type *>(o.data_ptr()),
                                  num_qo_heads, PosEncodingMode::kNone);
      TORCH_CHECK(status == cudaSuccess,
                  "estimate_scores failed with error code ",
                  cudaGetErrorString(status));
      return true;
    });
  });

  // if (n_groups == 1)
  //   return o.mean(1);
  CHECK_EQ(num_qo_heads % n_groups, 0);
  return o.reshape({batch_size, n_groups, num_qo_heads / n_groups, dg_seq_len})
      .mean(2);
}
