import os
import sys
from pathlib import Path
import warnings
from setuptools import setup, find_packages
from packaging.version import parse, Version
import platform
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


os.environ["CC"] = "hipcc"
os.environ["CXX"] = "hipcc"

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))
    
def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )



def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))


ext_modules = []

sources = [
    # 'torch_ck/kernels/fp8gemm.cpp',
    # 'torch_int/kernels/linear.cpp',
    # 'torch_int/kernels/bmm.cpp',
    # 'torch_ck/kernels/machete_kernels/machete_256_16x64x256_16x16_1x1_16x16_intrawave_v3_2.cpp',
    'torch_ck/kernels/machete_mm.cpp',
    'torch_ck/kernels/machete_prepack.cpp',
    'torch_ck/kernels/pybind.cpp', 
]

include_dirs = ['torch_ck/kernels']
extra_link_args = ['libutility.a']
extra_compile_args = ['-O3','-DNDEBUG', '-std=c++17', '--offload-arch=gfx942', '-DCK_ENABLE_INT8', '-D__HIP_PLATFORM_AMD__=1']


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

archs=["gfx942"]

cc_flag = [f"--offload-arch={arch}" for arch in archs]

# if FORCE_CXX11_ABI:
#     torch._C._GLIBCXX_USE_CXX11_ABI = True
cc_flag += ["-O3","-std=c++17",
            "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
            "-fgpu-flush-denormals-to-zero",
            "-DCK_ENABLE_BF16",
            "-DCK_ENABLE_BF8",
            "-DCK_ENABLE_FP16",
            "-DCK_ENABLE_FP32",
            "-DCK_ENABLE_FP64",
            "-DCK_ENABLE_FP8",
            "-DCK_ENABLE_INT8",
            "-DCK_USE_XDL",
            "-DUSE_PROF_API=1",
            # "-DFLASHATTENTION_DISABLE_BACKWARD",
            "-D__HIP_PLATFORM_HCC__=1"]

cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get('CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT', 3)}"]

# Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
hip_version = get_hip_version()
if hip_version > Version('5.7.23302'):
    cc_flag += ["-fno-offload-uniform-block"]
if hip_version > Version('6.1.40090'):
    cc_flag += ["-mllvm", "-enable-post-misched=0"]
if hip_version > Version('6.2.41132'):
    cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                "-mllvm", "-amdgpu-function-calls=false"]
if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
    cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"] + generator_flag,
    "nvcc": cc_flag + generator_flag,
}

include_dirs = [
    Path(this_dir) / "composable_kernel" / "include",
    Path(this_dir) / "composable_kernel" / "library" / "include",
    # Path(this_dir) / "csrc" / "composable_kernel" / "example" / "ck_tile" / "01_fmha",
]

# include_dirs.append(
#     Path(this_dir) / "torch_ck" / "kernels" / "include"
# )
ext_modules.append(
    CUDAExtension(
        name="torch_ck",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    )
)
setup(
    name='torch_ck',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)