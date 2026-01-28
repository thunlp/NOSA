from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import subprocess
import torch


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda",
            "--trace",
            "--trace-expand",
            "--debug-find",
            "--log-level=VERBOSE",
            "-Drmm_subbuild_repo=https://github.com/rapidsai/rmm.git",
            "-GNinja",
        ]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name], cwd=build_temp
        )


setup(
    name="arkvale",
    packages=find_packages(),
    ext_modules=[CMakeExtension("arkvale_cpp", "arkvale_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
