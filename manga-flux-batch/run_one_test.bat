@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "COMFY_PY=C:\Users\zs\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe"

if exist "%COMFY_PY%" (
  "%COMFY_PY%" "%SCRIPT_DIR%batch_run.py" --config "%SCRIPT_DIR%config.json" --limit 1 --no-pdf
) else (
  py -3 "%SCRIPT_DIR%batch_run.py" --config "%SCRIPT_DIR%config.json" --limit 1 --no-pdf
)

echo.
pause
