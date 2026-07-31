from setuptools import setup
import os
import platform
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

ext_modules = []
cmdclass = {}

if BUILD_TARGET == "cpu" or (BUILD_TARGET == "auto" and platform.system() == "Darwin"):
    # CPU-only build: compile only C++ sources (Eigen-based), skip CUDA
    from torch.utils.cpp_extension import CppExtension, BuildExtension

    cpu_sources = [
        "src/convert/flexible_dual_grid.cpp",
        "src/convert/volumetic_attr.cpp",
        "src/io/svo.cpp",
        "src/io/filter_parent.cpp",
        "src/io/filter_neighbor.cpp",
        "src/ext_cpu.cpp",
    ]
    # Only build if the CPU ext entry point exists
    ext_cpu_path = os.path.join(ROOT, "src/ext_cpu.cpp")
    if os.path.exists(ext_cpu_path):
        ext_modules = [
            CppExtension(
                name="o_voxel._C",
                sources=cpu_sources,
                include_dirs=[
                    os.path.join(ROOT, "third_party/eigen"),
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                },
            )
        ]
    else:
        # No C++ extension — pure Python fallback
        print("[o_voxel] Building without C++ extension (pure Python fallback)")
        ext_modules = []
    cmdclass = {'build_ext': BuildExtension}
else:
    # CUDA/ROCm build (original)
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION

    if BUILD_TARGET == "auto":
        IS_HIP = IS_HIP_EXTENSION
    elif BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

    if not IS_HIP:
        cc_flag = []
    else:
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        cc_flag = [f"--offload-arch={arch}" for arch in archs]

    ext_modules = [
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                "src/hash/hash.cu",
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                "src/rasterize/rasterize.cu",
                "src/ext.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/eigen"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"] + cc_flag,
            },
        )
    ]
    cmdclass = {'build_ext': BuildExtension}

setup(
    name="o_voxel",
    packages=[
        'o_voxel',
        'o_voxel.convert',
        'o_voxel.io',
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
