@echo off
REM Convenience wrapper so setup_windows.ps1 can be run from cmd.exe.
REM Any arguments are forwarded as-is, e.g.:
REM   setup_windows.bat -Python .\venv\Scripts\python.exe -CudaHome "D:\CUDA\v13.0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
exit /b %ERRORLEVEL%
