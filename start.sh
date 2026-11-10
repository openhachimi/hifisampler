#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fast path: if .venv is fully initialized, run hifiserver.py directly without uv overhead
if [ -x "$SCRIPT_DIR/.venv/bin/python" ] && [ -f "$SCRIPT_DIR/.venv/.hifisampler_ready" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/hifiserver.py" "$@"
fi

# Ensure ~/.local/bin and ~/.cargo/bin are in PATH for uv
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "--- uv not found. Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Error: Neither curl nor wget found; cannot install uv automatically." >&2
        exit 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "--- Creating Python 3.11 virtual environment (.venv)..."
uv venv "$SCRIPT_DIR/.venv" --python 3.11

OS_NAME="$(uname -s)"
TORCH_EXTRA_INDEX=""
if [ "$OS_NAME" = "Linux" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "--- NVIDIA GPU detected. Installing PyTorch (CUDA)..."
        TORCH_EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cu129"
        uv pip install --python "$SCRIPT_DIR/.venv" "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu129
    elif [ -e /dev/kfd ] || command -v rocminfo >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1; then
        echo "--- AMD ROCm GPU detected. Installing PyTorch (ROCm)..."
        TORCH_EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/rocm6.3"
        uv pip install --python "$SCRIPT_DIR/.venv" "torch>=2.6.0" --index-url https://download.pytorch.org/whl/rocm6.3
    else
        echo "--- No NVIDIA/ROCm GPU detected. Installing lightweight PyTorch (CPU)..."
        TORCH_EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
        uv pip install --python "$SCRIPT_DIR/.venv" "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu
    fi
fi

echo "--- Installing HifiSampler dependencies..."
# shellcheck disable=SC2086
uv pip install --python "$SCRIPT_DIR/.venv" $TORCH_EXTRA_INDEX -r "$SCRIPT_DIR/requirements.txt"
touch "$SCRIPT_DIR/.venv/.hifisampler_ready"

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/hifiserver.py" "$@"
