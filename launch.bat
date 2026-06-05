@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%CD%\bootstrap.py"
  set "EXIT_CODE=%ERRORLEVEL%"
  goto :done
)

where python >nul 2>nul
if not errorlevel 1 (
  python "%CD%\bootstrap.py"
  set "EXIT_CODE=%ERRORLEVEL%"
  goto :done
)

echo Python 3.10 or newer is required.
echo Install Python from https://www.python.org/downloads/
set "EXIT_CODE=1"

:done
if not "%EXIT_CODE%"=="0" (
  echo.
  pause
)
exit /b %EXIT_CODE%
