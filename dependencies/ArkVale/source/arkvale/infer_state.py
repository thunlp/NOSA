from typing import List, Union, Dict, Tuple
from collections import defaultdict

import torch
from torch import Tensor

from .kv_cache import KvPool, KvCache
from . import kernels
from . import utils

Digest = Tuple[Tensor, Tensor]


class InferState:
    def __init__(
        self,
        n_layers,
        n_qo_heads,
        n_kv_heads,
        head_dim,
        page_size,
        dtype: torch.dtype,
        device: torch.device,
        page_budgets: Union[int, List[int]] = None,
        n_max_pages=None,
        n_unlimited_layers=None,
        n_max_bytes=None,
        page_topks: Union[int, List[int]] = None,
        n_max_cpu_pages=None,
        n_max_cpu_bytes=None,
        n_sink_pages=2,
        n_win_pages=2,
        use_sparse_attn=False,
        n_prefetch_layers=None,
        group_size=None,
        n_groups=None,
        **kwargs,
    ) -> None:
        self.n_layers = n_layers
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        if n_max_pages is None:
            assert n_max_bytes is not None
            n_max_pages = n_max_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        if n_max_cpu_pages is None:
            assert n_max_cpu_bytes is not None
            n_max_cpu_pages = n_max_cpu_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        # breakpoint()
        if n_unlimited_layers is None:
            n_unlimited_layers = 0
        if not isinstance(page_budgets, (list, tuple)):
            page_budgets = [None] * n_unlimited_layers + [page_budgets] * (
                n_layers - n_unlimited_layers
            )
        if page_topks is None:
            page_topks = [b and b // 2 for b in page_budgets]
        elif not isinstance(page_topks, (list, tuple)):
            page_topks = [None] * n_unlimited_layers + [page_topks] * (
                n_layers - n_unlimited_layers
            )

        self.page_size = page_size
        self.n_max_pages = n_max_pages
        self.layer2budget = page_budgets
        self.budget2layers = defaultdict(list)
        for i, b in enumerate(page_budgets):
            self.budget2layers[b].append(i)
        self.layer2topk = page_topks
        self.n_sink_pages = n_sink_pages
        self.n_win_pages = n_win_pages
        assert n_win_pages >= 2
        self.use_sparse_attn = use_sparse_attn

        for i in range(n_layers):
            b = self.layer2budget[i]
            k = self.layer2topk[i]
            if b is not None:
                assert k is not None and k < b
                k = k - n_sink_pages - (n_win_pages - 1)
                assert k > 0
                self.layer2topk[i] = k

        self.layout = "NHD"
        self._i32 = dict(dtype=torch.int32, device=self.device)
        self._u8 = dict(dtype=torch.uint8, device=self.device)
        self._fp = dict(dtype=self.dtype, device=self.device)

        self._pool = KvPool(n_max_pages, page_size, n_kv_heads, head_dim, dtype, device)
        self.kv_caches: List[KvCache] = [None] * self.n_layers
        #dg means digest
        self.dg_caches: List[KvCache] = [None] * self.n_layers
        self._cpu_pool = KvPool(
            n_max_cpu_pages, page_size, n_kv_heads, head_dim, dtype, torch.device("cpu")
        )
        self.cpu_kv_caches: List[KvCache] = [None] * self.n_layers
        self.topk_dout: Tensor = None
        self.topk_iout: Tensor = None
        self.topk_newi: Tensor = None
        
        #rids means recall ids
        self.topk_rids: Tensor = None
        self.topk_buff: Tensor = None

        self.kv_last_page_len = 0
        self.kv_last_page_lens: Tensor = None
        self.kv_indptrs_tab: Dict[int, Tensor] = {b: None for b in self.budget2layers}
        self.kv_decode_indptrs_tab: Dict[int, Tensor] = {
            b: None for b in self.budget2layers
        }

        self.dg_last_page_len = 0
        self.dg_last_page_lens: Tensor = None
        self.dg_indptrs: Tensor = None

        wbufs = [torch.empty(16 * 1024 * 1024, **self._u8) for _ in self.budget2layers]
        self.prefill_handler = kernels.BatchPrefillWithPagedKVCacheWrapper(
            wbufs[0], self.layout
        )
        self.prefill_handler_tab = {
            b: kernels.BatchPrefillWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }
        self.decode_handler_tab = {
            b: kernels.BatchDecodeWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }

        self.n_prefetch_layers = n_prefetch_layers
        self.attn_layers = [None] * n_layers

        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.decode_backup_stream = torch.cuda.Stream(self.device)
        self.prefetch_streams = None
        self.on_decode_prefetch = None
        if n_prefetch_layers is not None:
            self.prefetch_streams = [
                torch.cuda.Stream(self.device) for _ in range(n_prefetch_layers + 1)
            ]
            self.on_decode_prefetch = [False] * (n_prefetch_layers + 1)

        assert group_size is None or n_groups is None
        if group_size is None and n_groups is None:
            n_groups = 1
            group_size = n_kv_heads
        elif group_size is None:
            assert n_kv_heads % n_groups == 0
            group_size = n_kv_heads // n_groups
        else:
            assert n_kv_heads % group_size == 0
            n_groups = n_kv_heads // group_size
        self.group_size = group_size
        self.n_groups = n_groups

    @property
    def seq_len(self):
        return self.kv_caches[0].seq_len

    @property
    def n_pages(self):
        return self.kv_caches[0].n_pages

    @property
    def batch_size(self):
        return self.kv_caches[0].batch_size

    def _prepare_prefill(self, bsz, q_len):
        self._pool.clear()
        self.kv_caches = [
            KvCache(
                self._pool,
                bsz,
                self.layer2budget[i],
                self.n_sink_pages,
                self.n_win_pages,
                n_groups=self.n_groups,
            )
            for i in range(self.n_layers)
        ]
        self.dg_caches = [
            self.layer2budget[i] and KvCache(self._pool, bsz)
            for i in range(self.n_layers)
        ]
        self._cpu_pool.clear()
        self.cpu_kv_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]

        max_topk = max([k for k in self.layer2topk if k] + [0])
        max_cap = max([b for b in self.layer2budget if b] + [0])
        bsz1 = bsz * self.n_groups
        self.topk_dout = torch.empty([bsz1 * max_cap], **self._fp)
        self.topk_iout = torch.empty(
            [bsz1 * (max_topk + self.n_sink_pages + self.n_win_pages)], **self._i32
        )
        self.topk_newi = torch.empty([bsz1 * max_cap], **self._fp)
        self.topk_rids = torch.empty([bsz1 * (max_topk + 1)], **self._i32)
        self.topk_buff = torch.empty([bsz1, 1 << 10], **self._u8)

        # we do not pre-allocate real kv-pages before prefill
        [kvc.prefill_alloc_n_tokens(q_len) for kvc in self.cpu_kv_caches if kvc]
        n_kv_pages = (q_len + self.page_size - 1) // self.page_size
        self.kv_last_page_len = (q_len - 1) % self.page_size + 1
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        for b in self.kv_indptrs_tab:
            self.kv_indptrs_tab[b] = kv_indptr = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )

        n_filled_kv_pages = n_kv_pages - 1
        [
            dgc.prefill_alloc_n_tokens(n_filled_kv_pages, self.alloc_page)
            for dgc in self.dg_caches
            if dgc
        ]
        n_dg_pages = (n_filled_kv_pages + self.page_size - 1) // self.page_size
        self.dg_last_page_len = (n_filled_kv_pages - 1) % self.page_size + 1
        self.dg_last_page_lens = torch.tensor(
            [self.dg_last_page_len] * bsz, **self._i32
        )
        self.dg_indptrs = torch.arange(0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32)

        qo_indptr = torch.arange(0, bsz * q_len + 1, q_len, **self._i32)
        self.prefill_handler.begin_forward(
            qo_indptr,
            kv_indptr,
            self.kv_last_page_lens,
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
        )

    def _finish_prefill(self, bsz, q_len):
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(self.kv_caches[l].n_real_pages for l in ls)
            self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )
            if b is not None and self.use_sparse_attn:
                topk = utils.all_eq(self.layer2topk[l] for l in ls)
                n_kv_pages = min(
                    self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                )
                self.kv_decode_indptrs_tab[b] = torch.arange(
                    0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                )
        self.prefill_handler.end_forward()

    def _prepare_decode(self, bsz):
        if self.kv_last_page_len + 1 >= self.page_size:
            self.default_stream.wait_stream(self.decode_backup_stream)
        pre = [kvc.n_real_pages for kvc in self.kv_caches]
        n_new_kv_pages = utils.all_eq(
            kvc.decode_alloc_1_token(self.alloc_page) for kvc in self.kv_caches
        )
        self.kv_last_page_len = utils.all_eq(
            kvc.last_page_len for kvc in self.kv_caches
        )
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        [kvc.decode_alloc_1_token() for kvc in self.cpu_kv_caches if kvc]

        if n_new_kv_pages > 0:
            assert n_new_kv_pages == 1

            cur = [kvc.n_real_pages for kvc in self.kv_caches]
            for b, ls in self.budget2layers.items():
                n_new_kv_real_pages = utils.all_eq(cur[l] - pre[l] for l in ls)
                if n_new_kv_real_pages > 0:
                    n_kv_pages = self.kv_caches[ls[0]].n_real_pages
                    self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = (
                        torch.arange(0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32)
                    )
                    if b is not None and self.use_sparse_attn:
                        topk = utils.all_eq(self.layer2topk[l] for l in ls)
                        n_kv_pages = min(
                            self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                        )
                        self.kv_decode_indptrs_tab[b] = torch.arange(
                            0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                        )

            dg_caches = [dgc for dgc in self.dg_caches if dgc]
            # if len(dg_caches) > 0 and self.kv_last_page_len == self.page_size:
            if len(dg_caches) > 0:
                n_new_dg_pages = utils.all_eq(
                    dgc.decode_alloc_1_token(self.alloc_page) for dgc in dg_caches
                )
                self.dg_last_page_len = utils.all_eq(
                    dgc.last_page_len for dgc in dg_caches
                )
                self.dg_last_page_lens = torch.tensor(
                    [self.dg_last_page_len] * bsz, **self._i32
                )
                if n_new_dg_pages > 0:
                    assert n_new_dg_pages == 1
                    n_dg_pages = utils.all_eq(dgc.n_real_pages for dgc in dg_caches)
                    self.dg_indptrs = torch.arange(
                        0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32
                    )

                with torch.cuda.stream(self.decode_backup_stream):
                    [self.decode_backup_1_page(l) for l in range(self.n_layers)]
                [self.decode_save_1_digest(l) for l in range(self.n_layers)]

        for b, h in self.decode_handler_tab.items():
            h.begin_forward(
                self.kv_decode_indptrs_tab[b],
                self.kv_last_page_lens,
                self.n_qo_heads,
                self.n_kv_heads,
                self.head_dim,
                self.page_size,
                data_type=self.dtype,
            )

    def _finish_decode(self, bsz):
        for handler in self.decode_handler_tab.values():
            handler.end_forward()

    def begin_forward(self, bsz, q_len):
        if q_len > 1:
            self._prepare_prefill(bsz, q_len)
        else:
            self._prepare_decode(bsz)

    def end_forward(self, bsz, q_len):
        if q_len > 1:
            self._finish_prefill(bsz, q_len)
        else:
            self._finish_decode(bsz)

    def append_paged_kv_cache(self, layer_idx: int, keys: Tensor, vals: Tensor):
        kvc = self.kv_caches[layer_idx]
        kernels.append_paged_kv_cache(
            keys,
            vals,
            kvc.buffer,
            kvc.c2p,
            self.kv_indptrs_tab[self.layer2budget[layer_idx]],
            self.kv_last_page_lens,
            self.layout,
        )

    def save_digests(self, layer_idx: int, digest: Digest):
        dgc = self.dg_caches[layer_idx]
        kernels.append_paged_kv_cache(
            *digest,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            self.layout,
        )

    def estimate_scores(
        self, layer_idx: int, query_states: Tensor, n_groups: int = None
    ):
        if n_groups is None:
            n_groups = self.n_groups
        dgc = self.dg_caches[layer_idx]
        return kernels.estimate_scores(
            query_states,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            dgc.seq_len,
            self.layout,
            n_groups,
        )

    def select_topk(self, layer_idx: int, scores: Tensor):
        bsz = self.batch_size * self.n_groups
        budget = self.layer2budget[layer_idx]
        topk = self.layer2topk[layer_idx]
        kvc = self.kv_caches[layer_idx]
        ns = self.n_sink_pages
        nw = self.n_win_pages
        eids_range = topk + ns + nw
        dout = self.topk_dout[: bsz * topk].view(bsz, topk)
        eids = self.topk_iout[: bsz * eids_range].view(bsz, eids_range)
        newi = self.topk_newi[: bsz * budget].view(bsz, budget)
        rids = self.topk_rids[: bsz * (topk + 1)].view(bsz, topk + 1)
        buff = self.topk_buff
        scores = scores.reshape(bsz, -1)
        cc2gp = kvc.cc2gp.reshape(bsz, -1)
        gc2cc = kvc.gc2cc.reshape(bsz, -1)
        #dout is the score, and eids is the index of the score
        #newi is the index of the new page that not in the gpu cache


        # torch.save(scores, "scores.pt")
        # torch.save(dout, "dout.pt")
        # torch.save(eids, "eids.pt")
        # torch.save(newi, "newi.pt")
        # torch.save(cc2gp, "cc2gp.pt")
        # torch.save(gc2cc, "gc2cc.pt")
        # torch.save(rids, "rids.pt")
        # torch.save(buff, "buff.pt")
        # torch.save(topk, "topk.pt")
        # torch.save(ns, "ns.pt")
        # torch.save(nw, "nw.pt")


        # torch.cuda.synchronize()
        # breakpoint()
        kernels.select_topk(
            scores, dout, eids, newi, cc2gp, gc2cc, rids, buff, topk, ns, nw
        )
        # torch.cuda.synchronize()
        return eids, rids

    def recall(self, layer_idx: int, eids: Tensor, rids: Tensor):
        nw = self.n_win_pages
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        bsz = kvc.batch_size
        gs = self.group_size
        ng = self.n_groups
        eids = eids.reshape(bsz, ng, -1)
        rids = rids.reshape(bsz, ng, -1)

        for i in range(kvc.batch_size):
            for j in range(self.n_groups):
                nr = rids[i, j, 0]
                if nr == 0:
                    continue
                heads = slice(j * gs, (j + 1) * gs)
                for ei, ri in zip(eids[i, j, -(nr + nw) : -nw], rids[i, j, 1 : nr + 1]):
                    kvc.pool[ei][..., heads, :].copy_(
                        cpu_kvc[i, ri][..., heads, :], non_blocking=True
                    )

    def estimate_select_recall(self, layer_idx: int, q: Tensor):
        scores = self.estimate_scores(layer_idx, q)
        #eid means evict ids, rid means recall ids
        eids, rids = self.select_topk(layer_idx, scores)
        self.recall(layer_idx, eids, rids)
        return scores, eids, rids

    def prefill_backup_pages(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for ci, gi in zip(cpu_kvc.c2p.reshape(-1), kvc.c2p.reshape(-1)):
            cpu_kvc.buffer[ci].copy_(kvc.buffer[gi], non_blocking=True)

    def decode_backup_1_page(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for i in range(kvc.batch_size):
            cpu_kvc[i, -2].copy_(kvc[i, kvc.evict_idx], non_blocking=True)

    def _summarize_keys(self, filled_keys: Tensor) -> Digest:
        maxs = filled_keys.max(dim=2).values
        mins = filled_keys.min(dim=2).values
        centers = (maxs + mins) / 2
        dists = (
            (
                centers.reshape(*filled_keys.shape[:2], 1, -1, self.head_dim)
                - filled_keys
            )
            .abs()
            .mean(dim=2)
        )
        mins = centers - dists
        maxs = centers + dists
        return maxs, mins

    def prefill_save_digests(self, layer_idx: int, keys: Tensor):
        if self.layer2budget[layer_idx] is None:
            return
        bsz, q_len, *_ = keys.shape
        n_filled_pages = (q_len + self.page_size - 1) // self.page_size - 1
        filled_keys = keys[:, : n_filled_pages * self.page_size, ...].reshape(
            bsz, n_filled_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def decode_save_1_digest(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        if kvc.budget is None:
            return
        filled_keys = torch.cat(
            [kvc[i, kvc.evict_idx][:1][None, ...] for i in range(self.batch_size)],
            dim=0,
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def prefill_evict_extra_pages(self, layer_idx: int, query_states: Tensor):
        kvc = self.kv_caches[layer_idx]
        bsz = kvc.batch_size
        ng = self.n_groups
        if kvc.budget is None:
            return
        if kvc.n_real_pages > kvc.budget:
            ns = max(2, kvc.n_sink_pages)
            nw = kvc.n_win_pages
            assert ns + nw <= kvc.budget
            topk = kvc.budget - ns - nw
            kvc.c2p = torch.empty([bsz, kvc.budget], **kvc._i32)
            kvc.gc2cc = torch.empty([bsz, kvc.budget], **kvc._i32)
            ev_gpi = kvc.cc2gp.clone()
            kvc.cc2gp.fill_(-1)
            dout = self.topk_dout[: bsz * topk].view(bsz, topk)
            buff = self.topk_buff
            scores = self.estimate_scores(layer_idx, query_states, n_groups=1)
            scores = scores.reshape(bsz, -1)
            kernels.prefill_select_topk(
                scores, dout, kvc.c2p, kvc.cc2gp, ev_gpi, kvc.gc2cc, buff, topk, ns, nw
            )
            self.prefill_evicted_pages[layer_idx] = ev_gpi
        kvc.cc2gp = (
            kvc.cc2gp.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.cc2gp.shape[-1])
            .contiguous()
        )
        kvc.gc2cc = (
            kvc.gc2cc.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.gc2cc.shape[-1])
            .contiguous()
        )

    def alloc_page(self):
        if len(self._pool._free_ids) <= 0:
            for i in range(self.n_layers):
                kvc = self.kv_caches[i]
                evt = self.prefill_backup_events[i]
                ev_gpi = self.prefill_evicted_pages[i]
                if ev_gpi is None:
                    continue
                assert evt is not None
                self.default_stream.wait_event(evt)
                [
                    kvc.pool.free_page(pid)
                    for pid in ev_gpi.reshape(-1).tolist()
                    if pid >= 0
                ]
                self.prefill_backup_events[i] = None
                self.prefill_evicted_pages[i] = None
                break
        return self._pool.alloc_page()

    def prefill_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.prefill_handler.forward(q, kvc.buffer, page_ids)

    def decode_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.decode_handler_tab[kvc.budget].forward(q, kvc.buffer, page_ids)
