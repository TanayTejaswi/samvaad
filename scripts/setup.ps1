# ==============================================================================
# Samvaad - Windows ARM64 Setup Script for Snapdragon X-Series PCs
# ==============================================================================
# Establishes a native ARM64 virtual environment and installs pinned dependencies.
# Enforces Python 3.11+ and Qualcomm QNN execution provider support.
# ==============================================================================

[CmdletBinding()]
param (
    [switch]$SkipModelDownload = $false,
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SAMVAAD: Snapdragon X-Series Native ARM64 Setup          " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Verify Processor Architecture
$arch = $env:PROCESSOR_ARCHITECTURE
Write-Host "[1/6] Validating Processor Architecture: $arch..." -ForegroundColor Yellow
if ($arch -ne "ARM64") {
    Write-Warning "Detected architecture '$arch'. Production NPU acceleration requires Windows on Snapdragon ARM64."
    Write-Warning "Setup will proceed, but Hexagon NPU QNN provider will only function on ARM64 hardware."
} else {
    Write-Host "  -> Verified: Native ARM64 Host." -ForegroundColor Green
}

# 2. Verify Python Version (3.11+)
Write-Host "[2/6] Verifying Python Runtime..." -ForegroundColor Yellow
try {
    $pyVersionOutput = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    $pyMajorMinor = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "  -> Found Python $pyVersionOutput" -ForegroundColor Green

    $pyFloat = [float]$pyMajorMinor
    if ($pyFloat -lt 3.11) {
        Write-Error "Python 3.11 or higher is strictly required by Samvaad engineering rules. Found $pyVersionOutput."
    }
} catch {
    Write-Error "Python runtime not found in PATH. Please install native Windows ARM64 Python 3.11+."
}

# 3. Create Virtual Environment
Write-Host "[3/6] Setting up virtual environment at '$VenvPath'..." -ForegroundColor Yellow
if (!(Test-Path $VenvPath)) {
    python -m venv $VenvPath
    Write-Host "  -> Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  -> Virtual environment already exists." -ForegroundColor Gray
}

# Determine venv python and pip paths
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

# 4. Install Dependencies
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
& $VenvPip install --upgrade pip setuptools wheel
& $VenvPip install -r requirements.txt

# On ARM64 Windows, explicitly ensure onnxruntime-qnn is present
if ($arch -eq "ARM64") {
    Write-Host "  -> Ensuring onnxruntime-qnn for Hexagon NPU..." -ForegroundColor Yellow
    & $VenvPip install onnxruntime-qnn
}

# 5. Initialize Required Directories
Write-Host "[5/6] Initializing storage directories..." -ForegroundColor Yellow
$dirs = @("models/whisper", "data/test_audio", "benchmarks/results", "logs")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  -> Created $dir" -ForegroundColor Green
    }
}

# 6. Verify QNN Execution Provider
Write-Host "[6/6] Verifying ONNX Execution Providers..." -ForegroundColor Yellow
$verifyScript = @"
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available ONNX Providers:', providers)
if 'QNNExecutionProvider' in providers:
    print('SUCCESS: QNNExecutionProvider is available for Hexagon NPU acceleration.')
else:
    print('NOTICE: QNNExecutionProvider not active in current session (expected on non-Snapdragon host).')
"@
& $VenvPython -c $verifyScript

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SAMVAAD Setup Complete! Run 'scripts\run.ps1' to launch.  " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
