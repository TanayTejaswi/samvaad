#!/usr/bin/env bash
# ==============================================================================
# Samvaad - Linux / POSIX Setup Script
# ==============================================================================
# Sets up Python virtual environment and dependencies for local development.
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo "  SAMVAAD: Development Environment Setup                    "
echo "============================================================"

VENV_PATH="${1:-.venv}"

# Check Python version
echo "[1/5] Checking Python version..."
PYTHON_BIN=$(command -v python3 || command -v python)
PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  -> Using Python $PY_VER from $PYTHON_BIN"

# Create venv if needed
echo "[2/5] Setting up venv at '$VENV_PATH'..."
if [ ! -d "$VENV_PATH" ]; then
    $PYTHON_BIN -m venv "$VENV_PATH"
    echo "  -> Created venv."
else
    echo "  -> Venv already exists."
fi

# Upgrade pip & install requirements
echo "[3/5] Installing dependencies..."
"$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel
"$VENV_PATH/bin/pip" install -r requirements.txt

# Create directories
echo "[4/5] Creating runtime directories..."
mkdir -p models/whisper data/test_audio benchmarks/results logs

# Check providers
echo "[5/5] Checking ONNX Execution Providers..."
"$VENV_PATH/bin/python" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available ONNX Providers:', providers)
if 'QNNExecutionProvider' in providers:
    print('SUCCESS: QNNExecutionProvider is active.')
else:
    print('INFO: QNNExecutionProvider not active (running on non-Snapdragon host; CPU fallback enabled).')
"

echo "============================================================"
echo "  Setup Complete. Activate with: source $VENV_PATH/bin/activate"
echo "============================================================"
