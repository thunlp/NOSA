#define _DISPATCH_CASES_head_dim(const_var, ...)         \
  _DISPATCH_CASE(128, HEAD_DIM, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_group_size(case_var, ...)         \
  _DISPATCH_CASE(1, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(4, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(8, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_page_size(case_var, ...)         \
  _DISPATCH_CASE(16, case_var, __VA_ARGS__) \
  _DISPATCH_CASE(32, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_kv_layout(case_var, ...)         \
  _DISPATCH_CASE(QKVLayout::kNHD, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_pos_encoding_mode(case_var, ...)         \
  _DISPATCH_CASE(PosEncodingMode::kNone, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_allow_fp16_qk_reduction(case_var, ...)         \
  _DISPATCH_CASE(false, case_var, __VA_ARGS__) \
// EOL

#define _DISPATCH_CASES_causal(case_var, ...)         \
  _DISPATCH_CASE(true, case_var, __VA_ARGS__) \
// EOL
