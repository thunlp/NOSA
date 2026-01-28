#include <torch/extension.h>
#include "flashinfer_ops.h"

PYBIND11_MODULE(arkvale_cpp, m) {
	m.def("rms_norm", &rms_norm);
	m.def("qk_apply_rotary_in_place", &qk_apply_rotary_in_place);
	m.def("qkq_apply_rotary_in_place", &qkq_apply_rotary_in_place);
	m.def("append_paged_kv_cache_prefill", &append_paged_kv_cache_prefill);
	m.def("append_paged_kv_cache_decode", &append_paged_kv_cache_decode);
	m.def("estimate_scores", &estimate_scores);
	m.def("select_topk", &select_topk);
	m.def("prefill_select_topk", &prefill_select_topk);
  py::class_<BatchPrefillWithPagedKVCachePyTorchWrapper>(
      m, "BatchPrefillWithPagedKVCachePyTorchWrapper")
      .def(py::init(&BatchPrefillWithPagedKVCachePyTorchWrapper::Create))
      .def("begin_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward)
      .def("forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::Forward);
  py::class_<BatchDecodeWithPagedKVCachePyTorchWrapper>(
	  m, "BatchDecodeWithPagedKVCachePyTorchWrapper")
      .def(py::init(&BatchDecodeWithPagedKVCachePyTorchWrapper::Create))
      .def("begin_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
      .def("forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
}
