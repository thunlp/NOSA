#include <cuda_runtime.h>
#include <raft/matrix/detail/select_k-inl.cuh>

#include "pytorch_extension_utils.h"

using namespace raft::matrix::detail::select::radix::impl;

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
__device__ inline void radix_topk_one_block_kernel_helper(
    const T *in, const IdxT *in_idx, const IdxT len, const IdxT k, T *out,
    IdxT *out_idx, const bool select_min, char *bufs, Counter<T, IdxT> &counter,
    IdxT *histogram) {

  if (threadIdx.x == 0) {
    counter.k = k;
    counter.len = len;
    counter.previous_len = len;
    counter.kth_value_bits = 0;
    counter.out_cnt = 0;
    counter.out_back_cnt = 0;
  }
  __syncthreads();

  const size_t batch_id = blockIdx.x; // size_t to avoid multiplication overflow
  const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
  bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    const T *in_buf;
    const IdxT *in_idx_buf;
    T *out_buf;
    IdxT *out_idx_buf;
    set_buf_pointers(in, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf,
                     out_buf, out_idx_buf);

    const IdxT current_len = counter.len;
    const IdxT current_k = counter.k;
    IdxT previous_len = counter.previous_len;
    if (previous_len > buf_len) {
      in_buf = in;
      in_idx_buf = in_idx;
      previous_len = len;
    }
    if (current_len > buf_len) {
      // so "out_buf==nullptr" denotes skipping writing buffer in current pass
      out_buf = nullptr;
      out_idx_buf = nullptr;
    }

    filter_and_histogram_for_one_block<T, IdxT, BitsPerPass>(
        in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, previous_len,
        &counter, histogram, select_min, pass);
    __syncthreads();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) {
      counter.previous_len = current_len;
    }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      last_filter<T, IdxT, BitsPerPass>(
          out_buf ? out_buf : in, out_buf ? out_idx_buf : in_idx, out, out_idx,
          out_buf ? current_len : len, k, &counter, select_min, pass);
      break;
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
RAFT_KERNEL SelectTopkKernel(const T *in,      // [bsz, len]
                             T *out,           // [bsz, k]
                             IdxT *out_idx,    // [bsz, k + ns + nw]
                             T *new_in,        // [bsz, cap]
                             int32_t *incache, // [bsz, len + 1]
                             IdxT *pos_ids,    // [bsz, cap]
                             IdxT *recall_ids, // [bsz, k + 1]
                             char *bufs, const IdxT len, const IdxT cap,
                             const IdxT k, const IdxT ns, const IdxT nw,
                             const bool select_min) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  const size_t batch_id = blockIdx.x;
  in += batch_id * len;
  out += batch_id * k;
  out_idx += batch_id * (k + ns + nw);

  new_in += batch_id * cap;
  incache += batch_id * (len + 1);
  pos_ids += batch_id * cap;
  recall_ids += batch_id * (k + 1);

  radix_topk_one_block_kernel_helper<T, IdxT, BitsPerPass, BlockSize>(
      in + ns, nullptr, (len + 1 - ns - nw), k, out, out_idx + ns, select_min,
      bufs, counter, histogram);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < ns; ++i)
      out_idx[i] = incache[i];
    for (int i = 0; i < nw; ++i)
      out_idx[ns + k + i] = incache[(len + 1) - nw + i];
    int n_recalls = 0;
    int n_incache = 0;
    for (int i = 0; i < k; ++i) {
      auto cpu_cache_pid = out_idx[ns + i] + ns;
      auto gpu_pool_pid = incache[cpu_cache_pid];
      if (gpu_pool_pid == -1) {
        recall_ids[++n_recalls] = cpu_cache_pid;
      } else {
        out_idx[ns + n_incache++] = gpu_pool_pid;
      }
    }
    recall_ids[0] = n_recalls;
  }
  __syncthreads();
  int n_recalls = recall_ids[0];
  int n_incache = k - n_recalls;
  if (n_recalls == 0)
    return;
  for (int i = threadIdx.x; i < cap; i += blockDim.x) {
    if (pos_ids[i] < len) {
      new_in[i] = in[pos_ids[i]];
    }
  }
  __syncthreads();
  radix_topk_one_block_kernel_helper<T, IdxT, BitsPerPass, BlockSize>(
      new_in + ns, nullptr, cap - ns - nw, n_recalls, out,
      out_idx + ns + n_incache, !select_min, bufs, counter, histogram);
  __syncthreads();
  for (int i = threadIdx.x; i < n_recalls; i += blockDim.x) {
    auto gpu_cache_pid = out_idx[ns + n_incache + i] + ns;
    auto old_cpu_cache_pid = pos_ids[gpu_cache_pid];
    auto gpu_pool_pid = incache[old_cpu_cache_pid];
    auto new_cpu_cache_pid = recall_ids[i + 1];
    pos_ids[gpu_cache_pid] = new_cpu_cache_pid;
    incache[old_cpu_cache_pid] = -1;
    incache[new_cpu_cache_pid] = gpu_pool_pid;
    out_idx[ns + n_incache + i] = gpu_pool_pid;
  }
}

template <typename T, typename IdxT, int BitsPerPass = 8, int BlockSize = 512>
void SelectTopk(const T *in,      // [bsz, len]
                T *out,           // [bsz, k]
                IdxT *out_idx,    // [bsz, k + ns + nw]
                T *new_in,        // [bsz, cap]
                int32_t *incache, // [bsz, len + 1]
                IdxT *pos_ids,    // [bsz, cap]
                IdxT *recall_ids, // [bsz, k + 1]
                char *bufs, int batch_size, IdxT len, IdxT cap, IdxT k, IdxT ns,
                IdxT nw, bool select_min = false) {

  int dev, sm_cnt;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev);
  const size_t max_chunk_size = calc_chunk_size<T, IdxT, BlockSize>(
      batch_size, len, sm_cnt,
      SelectTopkKernel<T, IdxT, BitsPerPass, BlockSize>, true);

  for (size_t offset = 0; offset < static_cast<size_t>(batch_size);
       offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);
    SelectTopkKernel<T, IdxT, BitsPerPass, BlockSize>
        <<<chunk_size, BlockSize, 0, nullptr>>>(
            in + offset * len, out + offset * k,
            out_idx + offset * (k + ns + nw), new_in + offset * cap,
            incache + offset * (len + 1), pos_ids + offset * cap,
            recall_ids + offset * (k + 1), bufs, len, cap, k, ns, nw,
            select_min);
  }
}

void select_topk(torch::Tensor scores,     // [bsz, len]
                 torch::Tensor out_data,   // [bsz, topk]
                 torch::Tensor out_inds,   // [bsz, topk + ns + nw]
                 torch::Tensor new_in,     // [bsz, cap]
                 torch::Tensor incache,    // [bsz, len + 1]
                 torch::Tensor pos_ids,    // [bsz, cap]
                 torch::Tensor recall_ids, // [bsz, topk + 1]
                 torch::Tensor buf, unsigned int topk,
                 unsigned int n_sink_pages, unsigned int n_win_pages) {

  CHECK_INPUT(scores);
  CHECK_INPUT(out_data);
  CHECK_INPUT(out_inds);
  CHECK_INPUT(new_in);
  CHECK_INPUT(incache);
  CHECK_INPUT(pos_ids);
  CHECK_INPUT(recall_ids);

  CHECK_DIM(2, scores);
  CHECK_DIM(2, out_data);
  CHECK_DIM(2, out_inds);
  CHECK_DIM(2, new_in);
  CHECK_DIM(2, incache);
  CHECK_DIM(2, pos_ids);
  CHECK_DIM(2, recall_ids);

  int batch_size = scores.size(0);
  int len = scores.size(1);
  int cap = pos_ids.size(1);

  CHECK_GE(len, topk);
  CHECK_EQ(topk, out_data.size(1));
  CHECK_EQ(topk + n_sink_pages + n_win_pages, out_inds.size(1));
  CHECK_EQ(topk + 1, recall_ids.size(1));
  CHECK_EQ(cap, new_in.size(1));
  CHECK_EQ(len + 1, incache.size(1));
  CHECK_EQ(batch_size, out_data.size(0));
  CHECK_EQ(batch_size, out_inds.size(0));
  CHECK_EQ(batch_size, new_in.size(0));
  CHECK_EQ(batch_size, incache.size(0));
  CHECK_EQ(batch_size, pos_ids.size(0));
  CHECK_EQ(batch_size, recall_ids.size(0));

  CHECK_EQ(out_inds.scalar_type(), torch::kInt32);
  CHECK_EQ(incache.scalar_type(), torch::kInt32);
  CHECK_EQ(pos_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(recall_ids.scalar_type(), torch::kInt32);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(scores.scalar_type(), c_type, [&] {
    SelectTopk<c_type, int32_t>(static_cast<c_type *>(scores.data_ptr()),
                                static_cast<c_type *>(out_data.data_ptr()),
                                static_cast<int32_t *>(out_inds.data_ptr()),
                                static_cast<c_type *>(new_in.data_ptr()),
                                static_cast<int32_t *>(incache.data_ptr()),
                                static_cast<int32_t *>(pos_ids.data_ptr()),
                                static_cast<int32_t *>(recall_ids.data_ptr()),
                                static_cast<char *>(buf.data_ptr()), batch_size,
                                len, cap, topk, n_sink_pages, n_win_pages,
                                /*select_min=*/false);
    return true;
  });
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
RAFT_KERNEL PrefillSelectTopkKernel(const T *in,   // [bsz, len]
                                    T *out,        // [bsz, k]
                                    IdxT *out_idx, // [bsz, k + ns + nw = cap]
                                    int32_t *incache,  // [bsz, len + 1]
                                    int32_t *incache1, // [bsz, len + 1]
                                    IdxT *pos_ids,     // [bsz, cap]
                                    char *bufs, const IdxT len, const IdxT cap,
                                    const IdxT k, const IdxT ns, const IdxT nw,
                                    const bool select_min) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  const size_t batch_id = blockIdx.x;
  in += batch_id * len;
  out += batch_id * k;
  out_idx += batch_id * (k + ns + nw);

  incache += batch_id * (len + 1);
  incache1 += batch_id * (len + 1);
  pos_ids += batch_id * cap;

  radix_topk_one_block_kernel_helper<T, IdxT, BitsPerPass, BlockSize>(
      in + ns, nullptr, (len + 1 - ns - nw), k, out, out_idx + ns, select_min,
      bufs, counter, histogram);

  __syncthreads();
  for (int i = threadIdx.x; i < cap; i += blockDim.x) {
    auto cpu_cache_pid = pos_ids[i] = (i < ns) ? i
                                      : (i >= ns + k)
                                          ? ((len + 1) - nw + i - (ns + k))
                                          : (out_idx[i] + ns);
    out_idx[i] = incache[cpu_cache_pid] = incache1[cpu_cache_pid];
    incache1[cpu_cache_pid] = -1;
  }
}

template <typename T, typename IdxT, int BitsPerPass = 8, int BlockSize = 512>
void PrefillSelectTopk(const T *in,       // [bsz, len]
                       T *out,            // [bsz, k]
                       IdxT *out_idx,     // [bsz, k + ns + nw = cap]
                       int32_t *incache,  // [bsz, len + 1]
                       int32_t *incache1, // [bsz, len + 1]
                       IdxT *pos_ids,     // [bsz, cap]
                       char *bufs, int batch_size, IdxT len, IdxT cap, IdxT k,
                       IdxT ns, IdxT nw, bool select_min = false) {

  int dev, sm_cnt;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev);
  const size_t max_chunk_size = calc_chunk_size<T, IdxT, BlockSize>(
      batch_size, len, sm_cnt,
      PrefillSelectTopkKernel<T, IdxT, BitsPerPass, BlockSize>, true);

  for (size_t offset = 0; offset < static_cast<size_t>(batch_size);
       offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);
    PrefillSelectTopkKernel<T, IdxT, BitsPerPass, BlockSize>
        <<<chunk_size, BlockSize, 0, nullptr>>>(
            in + offset * len, out + offset * k,
            out_idx + offset * (k + ns + nw), incache + offset * (len + 1),
            incache1 + offset * (len + 1), pos_ids + offset * cap, bufs, len,
            cap, k, ns, nw, select_min);
  }
}

void prefill_select_topk(torch::Tensor scores,   // [bsz, len]
                         torch::Tensor out_data, // [bsz, topk]
                         torch::Tensor out_inds, // [bsz, topk + ns + nw = cap]
                         torch::Tensor incache,  // [bsz, len + 1]
                         torch::Tensor incache1, // [bsz, len + 1]
                         torch::Tensor pos_ids,  // [bsz, cap]
                         torch::Tensor buf, unsigned int topk,
                         unsigned int n_sink_pages, unsigned int n_win_pages) {

  CHECK_INPUT(scores);
  CHECK_INPUT(out_data);
  CHECK_INPUT(out_inds);
  CHECK_INPUT(incache);
  CHECK_INPUT(pos_ids);

  CHECK_DIM(2, scores);
  CHECK_DIM(2, out_data);
  CHECK_DIM(2, out_inds);
  CHECK_DIM(2, incache);
  CHECK_DIM(2, pos_ids);

  int batch_size = scores.size(0);
  int len = scores.size(1);
  int cap = pos_ids.size(1);

  CHECK_GE(len, topk);
  CHECK_EQ(topk, out_data.size(1));
  CHECK_EQ(topk + n_sink_pages + n_win_pages, out_inds.size(1));
  CHECK_EQ(cap, topk + n_sink_pages + n_win_pages);
  CHECK_EQ(len + 1, incache.size(1));
  CHECK_EQ(batch_size, out_data.size(0));
  CHECK_EQ(batch_size, out_inds.size(0));
  CHECK_EQ(batch_size, incache.size(0));
  CHECK_EQ(batch_size, pos_ids.size(0));

  CHECK_EQ(out_inds.scalar_type(), torch::kInt32);
  CHECK_EQ(incache.scalar_type(), torch::kInt32);
  CHECK_EQ(pos_ids.scalar_type(), torch::kInt32);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(scores.scalar_type(), c_type, [&] {
    PrefillSelectTopk<c_type, int32_t>(
        static_cast<c_type *>(scores.data_ptr()),
        static_cast<c_type *>(out_data.data_ptr()),
        static_cast<int32_t *>(out_inds.data_ptr()),
        static_cast<int32_t *>(incache.data_ptr()),
        static_cast<int32_t *>(incache1.data_ptr()),
        static_cast<int32_t *>(pos_ids.data_ptr()),
        static_cast<char *>(buf.data_ptr()), batch_size, len, cap, topk,
        n_sink_pages, n_win_pages, /*select_min=*/false);
    return true;
  });
}