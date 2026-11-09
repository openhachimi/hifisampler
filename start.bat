@echo off
setlocal
cd /d "%~dp0"

rem --- Fast path: if .venv already exists, start hifiserver.py directly ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" hifiserver.py %*
    goto :eof
)

rem --- First-time setup: ensure uv is installed and run hifiserver.py ---
call :SetupUv
if %errorlevel% neq 0 (
    echo.
    echo Fatal error during uv setup. The script cannot continue.
    pause
    goto :eof
)

uv run hifiserver.py %*

endlocal
goto :eof


:SetupUv
    echo --- Checking for uv...
    uv --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo uv is already installed and accessible.
        exit /b 0
    )

    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
        exit /b 0
    )

    echo uv not found. Attempting to install...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo Error: Failed to execute uv installation script.
        exit /b 1
    )

    set "UV_BIN_PATH=%USERPROFILE%\.local\bin"
    set "PATH=%UV_BIN_PATH%;%PATH%"

    uv --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: uv was installed but is still not accessible in %UV_BIN_PATH%.
        exit /b 1
    )
    exit /b 0
