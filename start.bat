@echo off
setlocal
cd /d "%~dp0"

rem --- Fast path 1: bundled portable environment (env\python.exe) ---
if exist "env\python.exe" if exist "env\Lib\site-packages\aiohttp" (
    "env\python.exe" hifiserver.py %*
    if errorlevel 1 pause
    goto :eof
)

rem --- Fast path 2: if .venv is fully initialized, start hifiserver.py directly ---
if exist ".venv\Scripts\python.exe" if exist ".venv\Lib\site-packages\aiohttp" (
    ".venv\Scripts\python.exe" hifiserver.py %*
    if errorlevel 1 pause
    goto :eof
)

rem --- First-time setup: ensure uv is installed and initialize .venv by GPU type ---
call :SetupUv
if %errorlevel% neq 0 (
    echo.
    echo Fatal error during uv setup. The script cannot continue.
    pause
    goto :eof
)

if not exist ".venv\Scripts\python.exe" (
    echo --- Creating Python 3.11 virtual environment ^(.venv^)...
    uv venv .venv --python 3.11
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        pause
        goto :eof
    )
)

if not exist ".venv\Lib\site-packages\torch" (
    where nvidia-smi >nul 2>&1 && nvidia-smi -L >nul 2>&1
    if not errorlevel 1 (
        echo --- NVIDIA GPU detected. Installing PyTorch with CUDA support...
        uv pip install --python .venv "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu129
        if errorlevel 1 (
            echo Error: Failed to install PyTorch with CUDA support.
            pause
            goto :eof
        )
    ) else (
        echo --- No NVIDIA GPU detected. Installing lightweight PyTorch ^(CPU^) + DirectML support...
    )
)

echo --- Installing HifiSampler dependencies...
uv pip install --python .venv -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies.
    pause
    goto :eof
)
echo ok > ".venv\.hifisampler_ready"

".venv\Scripts\python.exe" hifiserver.py %*

endlocal
goto :eof


:SetupUv
    echo --- Checking for uv...
    if exist "%~dp0uv.exe" (
        set "PATH=%~dp0;%PATH%"
        exit /b 0
    )

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
