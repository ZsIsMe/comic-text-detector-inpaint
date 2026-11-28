@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul
  if not errorlevel 1 goto :venv
)

where py >nul 2>nul
if not errorlevel 1 goto :py

where python >nul 2>nul
if not errorlevel 1 goto :python

echo Python 3.11 or newer is required. Python 3.12 is recommended.
echo Install Python from https://www.python.org/downloads/
set "EXIT_CODE=1"
goto :done

:venv
".venv\Scripts\python.exe" "%CD%\bootstrap.py"
set "EXIT_CODE=%ERRORLEVEL%"
goto :done

:py
py -3 "%CD%\bootstrap.py"
set "EXIT_CODE=%ERRORLEVEL%"
goto :done

:python
python "%CD%\bootstrap.py"
set "EXIT_CODE=%ERRORLEVEL%"

:done
if not "%EXIT_CODE%"=="0" (
  echo.
  pause
)
exit /b %EXIT_CODE%
