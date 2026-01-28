/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <torch/extension.h>

#include <flashinfer/attention/handler.cuh>
#include <flashinfer/layout.cuh>

torch::Tensor rms_norm(
	torch::Tensor input, // [bsz, len, hidden_dim]
	torch::Tensor weight, // [hidden_dim]
	float epsilon);

void qk_apply_rotary_in_place(
    torch::Tensor q, // [bsz, len, n_qo_heads, head_dim]
    torch::Tensor k, // [bsz, len, n_kv_heads, head_dim]
    unsigned int past_kv_len, 
    float rope_scale, 
    float rope_theta);

void qkq_apply_rotary_in_place(
    torch::Tensor q, // [bsz, len, n_qo_heads, head_dim]
    torch::Tensor k, // [bsz, len, n_kv_heads, head_dim]
    torch::Tensor q1, // [bsz, len, n_qo_heads, head_dim]
    unsigned int past_kv_len, 
    float rope_scale, 
    float rope_theta);

void append_paged_kv_cache_prefill(
	torch::Tensor k, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor v, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor kv_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
	torch::Tensor kv_indices, // [bsz, num_pages]
	torch::Tensor kv_indptr, // [bsz+1]
	torch::Tensor kv_last_page_len, // [bsz]
	unsigned int layout);

void append_paged_kv_cache_decode(
	torch::Tensor k, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor v, // [bsz, kv_len, n_kv_heads, head_dim]
	torch::Tensor kv_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
	torch::Tensor kv_indices, // [bsz, num_pages]
	torch::Tensor kv_indptr, // [bsz+1]
	torch::Tensor kv_last_page_len, // [bsz]
	unsigned int layout);

torch::Tensor estimate_scores(
	torch::Tensor q, // [bsz, 1, num_heads, head_dim]
	torch::Tensor dg_data, // [n_max_pages, 2, page_size, n_kv_heads, head_dim]
	torch::Tensor dg_indices, // [bsz, num_pages]
	torch::Tensor dg_indptr, // [bsz+1]
	torch::Tensor dg_last_page_len, // [bsz]
	unsigned int dg_seq_len, 
	unsigned int layout,
  unsigned int n_groups);

void select_topk(
	torch::Tensor scores, // [bsz, len]
	torch::Tensor out_data, // [bsz, topk]
	torch::Tensor out_inds, // [bsz, topk + ns + nw]
	torch::Tensor new_in, // [bsz, cap]
	torch::Tensor incache, // [bsz, len]
	torch::Tensor pos_ids, // [bsz, cap]
	torch::Tensor recall_ids, // [bsz, topk + 1]
	torch::Tensor buf,
	unsigned int topk,
	unsigned int n_sink_pages,
	unsigned int n_win_pages);

void prefill_select_topk(
  torch::Tensor scores, // [bsz, len]
  torch::Tensor out_data, // [bsz, topk]
  torch::Tensor out_inds, // [bsz, topk + ns + nw = cap]
  torch::Tensor incache, // [bsz, len + 1]
  torch::Tensor incache1, // [bsz, len + 1]
  torch::Tensor pos_ids, // [bsz, cap]
  torch::Tensor buf,
  unsigned int topk,
  unsigned int n_sink_pages,
  unsigned int n_win_pages);

class BatchPrefillWithPagedKVCachePyTorchWrapper {
 public:
  static BatchPrefillWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
    return BatchPrefillWithPagedKVCachePyTorchWrapper(layout);
  }
  void BeginForward(torch::Tensor workspace_buffer, torch::Tensor qo_indptr,
                    unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
                    unsigned int head_dim);
  void EndForward();
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor qo_indptr,
                                     torch::Tensor paged_kv_data, torch::Tensor paged_kv_indptr,
                                     torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len, bool causal,
                                     unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
                                     float sm_scale, float rope_scale, float rope_theta,
                                     bool return_lse);

 private:
  BatchPrefillWithPagedKVCachePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchPrefillHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};

class BatchDecodeWithPagedKVCachePyTorchWrapper {
 public:
  static BatchDecodeWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
    return BatchDecodeWithPagedKVCachePyTorchWrapper(layout);
  }
  void BeginForward(torch::Tensor workspace_buffer, torch::Tensor indptr,
                    torch::Tensor last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
                    unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
                    unsigned int pos_encoding_mode, torch::Tensor empty_data);
  void EndForward();
  std::vector<torch::Tensor> Forward(torch::Tensor q, torch::Tensor paged_kv_data,
                                     torch::Tensor paged_kv_indptr, torch::Tensor paged_kv_indices,
                                     torch::Tensor paged_kv_last_page_len,
                                     unsigned int pos_encoding_mode, float sm_scale,
                                     float rope_scale, float rope_theta, bool return_lse);

 private:
  BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchDecodeHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};
