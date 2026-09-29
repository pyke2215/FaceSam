@echo off
setlocal
cd /d "%~dp0"

rem Interpreter choice: FACESLIM_PYTHON (full path to python.exe), else py launcher 3.11/3.12/3.10, else python on PATH.
set "PYLAUNCH="
if defined FACESLIM_PYTHON set "PYLAUNCH="%FACESLIM_PYTHON%""
for %%V in (3.11 3.12 3.10) do (
    if not defined PYLAUNCH (
        py -%%V -c "import sys" >nul 2>&1 && set "PYLAUNCH=py -%%V"
    )
)
if not defined PYLAUNCH (
    call python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PYLAUNCH=python"
)
if not defined PYLAUNCH (
    echo Python 3.10-3.12 was not found.
    echo Install Python 3.11 from https://www.python.org/downloads/ with "py launcher" enabled,
    echo or set FACESLIM_PYTHON to a python.exe path, then run setup.bat again.
    exit /b 1
)

echo Using %PYLAUNCH%
call %PYLAUNCH% tools\setup_machine.py %*
if errorlevel 1 (
    echo.
    echo Setup failed. See docs\SETUP_WINDOWS_GPU.md, section Troubleshooting.
    exit /b 1
)
endlocal
