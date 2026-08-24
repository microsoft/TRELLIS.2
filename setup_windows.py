#!/usr/bin/env python3
"""Windows setup helper for TRELLIS.2.

Locates the Visual Studio 2022 C++ build tools and CUDA Toolkit, then runs
``pip install -r requirements.txt --no-build-isolation`` with the environment
variables required to compile the native CUDA extensions (CuMesh, FlexGEMM,
o-voxel) on Windows/MSVC.

This is a Python equivalent of setup_windows.ps1 / setup_windows.bat for
users who prefer not to invoke PowerShell or a batch file directly, e.g.:

    python setup_windows.py
    python setup_windows.py --python .\\venv\\Scripts\\python.exe --cuda-home "D:\\CUDA\\v13.0"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
DEFAULT_TORCH_CUDA_ARCH_LIST = "12.0"

VCVARS_CANDIDATES = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=sys.executable, help="Path to the Python interpreter to install into (default: current interpreter)")
    parser.add_argument("--cuda-home", default=DEFAULT_CUDA_HOME, help=f"Path to the CUDA Toolkit install (default: {DEFAULT_CUDA_HOME})")
    parser.add_argument("--torch-cuda-arch-list", default=DEFAULT_TORCH_CUDA_ARCH_LIST, help=f"Value for TORCH_CUDA_ARCH_LIST (default: {DEFAULT_TORCH_CUDA_ARCH_LIST})")
    return parser.parse_args()


def find_vcvars() -> str:
    for candidate in VCVARS_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit(
        "Visual Studio 2022 C++ build tools were not found. "
        "Install the 'Desktop development with C++' workload."
    )


def resolve_python(python_arg: str) -> str:
    resolved = shutil.which(python_arg)
    if resolved is None:
        raise SystemExit(f"Could not find python executable: {python_arg}")
    return resolved


def main() -> int:
    if os.name != "nt":
        raise SystemExit("setup_windows.py is intended to be run on Windows.")

    args = parse_args()

    vcvars = find_vcvars()

    cuda_home = args.cuda_home
    if not os.path.isdir(cuda_home):
        raise SystemExit(f"CUDA toolkit path not found: {cuda_home}")

    python_exe = resolve_python(args.python)
    requirements = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements.is_file():
        raise SystemExit(f"requirements.txt not found: {requirements}")

    # vcvars64.bat only sets environment variables for the cmd.exe process it
    # runs in, so vcvars + pip install must happen in the same cmd.exe call.
    command = " && ".join(
        [
            f'call "{vcvars}" >nul',
            'set "DISTUTILS_USE_SDK=1"',
            'set "MSSdk=1"',
            f'set "CUDA_HOME={cuda_home}"',
            f'set "TORCH_CUDA_ARCH_LIST={args.torch_cuda_arch_list}"',
            f'"{python_exe}" -m pip install -r "{requirements}" --no-build-isolation',
        ]
    )

    print(f"Using Python: {python_exe}")
    print(f"Using CUDA_HOME: {cuda_home}")
    print(f"Using TORCH_CUDA_ARCH_LIST: {args.torch_cuda_arch_list}")
    print(f"Using vcvars64.bat: {vcvars}")

    result = subprocess.run(["cmd.exe", "/c", command])
    if result.returncode != 0:
        raise SystemExit(f"Windows setup failed with exit code {result.returncode}")

    print("Windows setup completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
