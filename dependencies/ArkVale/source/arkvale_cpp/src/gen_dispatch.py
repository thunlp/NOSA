import os
import sys
import argparse
import pathlib

root = pathlib.Path(__name__).parent
sys.path.append(str(root / "../../../3rdparty/flashinfer/python"))

import generate_dispatch_inc


def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)


prefix = "generated"
(root / prefix).mkdir(parents=True, exist_ok=True)

group_sizes = os.environ.get("FLASHINFER_GROUP_SIZES", "1,4,8").split(",")
page_sizes = os.environ.get("FLASHINFER_PAGE_SIZES", "16,32").split(",")
head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "128").split(",")
kv_layouts = os.environ.get("FLASHINFER_KV_LAYOUTS", "0").split(",")
pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0").split(",")
allow_fp16_qk_reduction_options = os.environ.get(
    "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0"
).split(",")
causal_options = os.environ.get("FLASHINFER_CAUSAL_OPTIONS", "1").split(",")
# dispatch.inc
path = root / prefix / "dispatch.inc"
write_if_different(
    path,
    generate_dispatch_inc.get_dispatch_inc_str(
        argparse.Namespace(
            group_sizes=map(int, group_sizes),
            page_sizes=map(int, page_sizes),
            head_dims=map(int, head_dims),
            kv_layouts=map(int, kv_layouts),
            pos_encoding_modes=map(int, pos_encoding_modes),
            allow_fp16_qk_reductions=map(int, allow_fp16_qk_reduction_options),
            causals=map(int, causal_options),
        )
    ),
)
