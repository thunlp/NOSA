from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="shadowkv",
    version="0.1.0",
    description="ShadowKV CUDA extension",
    packages=find_packages(),   # 👈 自动安装 shadowkv 包
    ext_modules=[
        CUDAExtension(
            name="shadowkv.shadowkv",  # 👈 关键：生成 shadowkv/shadowkv*.so
            sources=[
                "kernels/main.cu",
                "kernels/rope.cu",
                "kernels/rope_new.cu",
                "kernels/gather_copy.cu",
                "kernels/batch_gather_gemm.cu",
                "kernels/batch_gemm_softmax.cu",
            ],
            include_dirs=[
                os.path.join(current_dir, "3rdparty/cutlass/include"),
                os.path.join(current_dir, "3rdparty/cutlass/examples/common"),
                os.path.join(current_dir, "3rdparty/cutlass/tools/util/include"),
                os.path.join(current_dir, "kernels"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    zip_safe=False,
)
