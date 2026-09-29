@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run setup.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" FaceSlim_v1.py %*
endlocal
