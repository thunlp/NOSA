import torch
import math
from typing import Optional, Union

# import arkvale_cpp as _cpp
from . import arkvale_cpp as _cpp

from .utils import (
    PosEncodingMode,
    TensorLayout,
    expand_5d,
    check_pos_encoding_mode,
    check_kv_layout,
    is_float8,
)


def rms_norm(
    inp: torch.Tensor,
    wgt: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return _cpp.rms_norm(
        inp,
        wgt,
        epsilon,
    )


def qk_apply_rotary_in_place(
    q: torch.Tensor,
    k: torch.Tensor,
    past_kv_len: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
):
    _cpp.qk_apply_rotary_in_place(
        q,
        k,
        past_kv_len,
        rope_scale,
        rope_theta,
    )


def qkq_apply_rotary_in_place(
    q: torch.Tensor,
    k: torch.Tensor,
    q1: torch.Tensor,
    past_kv_len: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
):
    _cpp.qkq_apply_rotary_in_place(
        q,
        k,
        q1,
        past_kv_len,
        rope_scale,
        rope_theta,
    )


def append_paged_kv_cache(
    k: torch.Tensor,  # [bsz, kv_len, n_kv_heads, head_dim]
    v: torch.Tensor,  # [bsz, kv_len, n_kv_heads, head_dim]
    kv_data: torch.Tensor,  # [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    kv_indices: torch.Tensor,  # [bsz, num_pages]
    kv_indptr: torch.Tensor,  # [bsz+1]
    kv_last_page_len: torch.Tensor,  # [bsz]
    layout: str = "NHD",
):
    (
        _cpp.append_paged_kv_cache_prefill
        if k.size(1) > 1
        else _cpp.append_paged_kv_cache_decode
    )(
        k,
        v,
        kv_data,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        TensorLayout[layout].value,
    )


def estimate_scores(
    q: torch.Tensor,  # [bsz, 1, num_heads, head_dim]
    dg_data: torch.Tensor,  # [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    dg_indices: torch.Tensor,  # [bsz, num_pages]
    dg_indptr: torch.Tensor,  # [bsz+1]
    dg_last_page_len: torch.Tensor,  # [bsz]
    dg_seq_len: int,
    layout: str = "NHD",
    n_groups: int = 1,
) -> torch.Tensor:  # [bsz, n_groups, n_kv_pages - 1]
    return _cpp.estimate_scores(
        q,
        dg_data,
        dg_indices,
        dg_indptr,
        dg_last_page_len,
        dg_seq_len,
        TensorLayout[layout].value,
        n_groups,
    )


def select_topk(
    scores: torch.Tensor,  # [bsz, n_kv_pages - 1]
    out_data: torch.Tensor,  # [bsz, topk]
    out_inds: torch.Tensor,  # [bsz, topk + ns + nw]
    new_in: torch.Tensor,  # [bsz, cap]
    incache: torch.Tensor,  # [bsz, n_kv_pages]
    pos_ids: torch.Tensor,  # [bsz, cap]
    recall_ids: torch.Tensor,  # [bsz, topk + 1]
    buf: torch.Tensor,
    topk: int,
    n_sink_pages: int,
    n_win_pages: int,
):
    _cpp.select_topk(
        scores,
        out_data,
        out_inds,
        new_in,
        incache,
        pos_ids,
        recall_ids,
        buf,
        topk,
        n_sink_pages,
        n_win_pages,
    )


def prefill_select_topk(
    scores: torch.Tensor,  # [bsz, n_kv_pages - 1]
    out_data: torch.Tensor,  # [bsz, topk]
    out_inds: torch.Tensor,  # [bsz, topk + ns + nw = cap]
    incache: torch.Tensor,  # [bsz, n_kv_pages]
    incache1: torch.Tensor,  # [bsz, n_kv_pages]
    pos_ids: torch.Tensor,  # [bsz, cap]
    buf: torch.Tensor,
    topk: int,
    n_sink_pages: int,
    n_win_pages: int,
):
    _cpp.prefill_select_topk(
        scores,
        out_data,
        out_inds,
        incache,
        incache1,
        pos_ids,
        buf,
        topk,
        n_sink_pages,
        n_win_pages,
    )


class BatchPrefillWithPagedKVCacheWrapper:
    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _cpp.BatchPrefillWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value
        )
        self._qo_indptr = None
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        # paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        batch_size = len(qo_indptr) - 1
        self._qo_indptr = qo_indptr
        self._paged_kv_indptr = paged_kv_indptr
        # self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._wrapper.begin_forward(
            self._workspace_buffer,
            qo_indptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
        self._qo_indptr = None
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        causal: bool = True,
        pos_encoding_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        assert not is_float8(q)
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        return self._wrapper.forward(
            q.reshape(-1, *q.shape[-2:]),
            self._qo_indptr,
            paged_kv_data,
            self._paged_kv_indptr,
            # self._paged_kv_indices,
            paged_kv_indices.reshape(-1),
            self._paged_kv_last_page_len,
            causal,
            PosEncodingMode[pos_encoding_mode].value,
            allow_fp16_qk_reduction,
            sm_scale,
            rope_scale,
            rope_theta,
            False,
        )[0]


class BatchDecodeWithPagedKVCacheWrapper:
    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _cpp.BatchDecodeWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value
        )
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        paged_kv_indptr: torch.Tensor,
        # paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        data_type: Union[str, torch.dtype] = "float16",
    ):
        self._paged_kv_indptr = paged_kv_indptr
        # self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len

        batch_size = len(paged_kv_indptr) - 1
        # NOTE(Zihao): the following tensor acts as placeholder to pass dtype info
        empty_data = torch.empty(
            0,
            dtype=(
                getattr(torch, data_type) if isinstance(data_type, str) else data_type
            ),
        )
        self._wrapper.begin_forward(
            self._workspace_buffer,
            paged_kv_indptr,
            paged_kv_last_page_len,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            PosEncodingMode[pos_encoding_mode].value,
            empty_data,
        )

    def end_forward(self):
        self._paged_kv_indptr = None
        # self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        return self._wrapper.forward(
            q.reshape(-1, *q.shape[-2:]),
            paged_kv_data,
            self._paged_kv_indptr,
            # self._paged_kv_indices,
            paged_kv_indices.reshape(-1),
            self._paged_kv_last_page_len,
            PosEncodingMode[pos_encoding_mode].value,
            sm_scale,
            rope_scale,
            rope_theta,
            False,
        )[0]
