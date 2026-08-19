@echo off
setlocal

cd /d "%~dp0"

title GeoEarthworm

where py >nul 2>nul
if not errorlevel 1 (
    set "PYCMD=py"
    goto :foundpython
)

where python >nul 2>nul
if not errorlevel 1 (
    set "PYCMD=python"
    goto :foundpython
)

echo.
echo ============================================================
echo   Python was not found on this computer.
echo.
echo   Install it from https://python.org/downloads - on the
echo   first installer screen, tick "Add python.exe to PATH" -
echo   then run this file again.
echo ============================================================
echo.
pause
exit /b 1

:foundpython

if exist ".venv\Scripts\activate.bat" (
    goto :runapp
)

echo.
echo ============================================================
echo   First-time setup - this only happens once and can take
echo   a few minutes. Please keep this window open.
echo ============================================================
echo.

%PYCMD% -m venv .venv
if errorlevel 1 (
    echo.
    echo Something went wrong creating the Python environment.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip
pip install -r requirements-app.txt
if errorlevel 1 (
    echo.
    echo Something went wrong installing the required packages.
    echo Please check your internet connection, then run this file again.
    pause
    exit /b 1
)

echo.
echo Setup complete.

:runapp

call ".venv\Scripts\activate.bat"

rem Skip Streamlit's first-run "enter your email" prompt, which otherwise
rem blocks this window on first launch.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo.
echo ============================================================
echo   Starting GeoEarthworm...
echo   A browser tab will open automatically in a few seconds.
echo.
echo   To stop the program, close this black window.
echo ============================================================
echo.

streamlit run app.py

echo.
echo GeoEarthworm has stopped.
pause
