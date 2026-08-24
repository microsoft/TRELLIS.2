param(
    [string]$Python = "python",
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    [string]$TorchCudaArchList = "12.0"
)

$ErrorActionPreference = "Stop"

$vcvars = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $vcvars) {
    throw "Visual Studio 2022 C++ build tools were not found. Install the Desktop development with C++ workload."
}

if (-not (Test-Path $CudaHome)) {
    throw "CUDA toolkit path not found: $CudaHome"
}

$pythonExe = (Get-Command $Python -ErrorAction Stop).Source
$requirements = Join-Path $PSScriptRoot "requirements.txt"

$command = @(
    "call `"$vcvars`" >nul",
    "set `"DISTUTILS_USE_SDK=1`"",
    "set `"MSSdk=1`"",
    "set `"CUDA_HOME=$CudaHome`"",
    "set `"TORCH_CUDA_ARCH_LIST=$TorchCudaArchList`"",
    "`"$pythonExe`" -m pip install -r `"$requirements`" --no-build-isolation"
) -join " && "

cmd.exe /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Windows setup failed with exit code $LASTEXITCODE"
}
