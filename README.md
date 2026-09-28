# SAMVAAD (Shravan Engine)

> **Offline, NPU-Accelerated Real-Time Speech Perception & Captioning for Qualcomm Snapdragon® X-Series PCs.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2011%20ARM64-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/NPU-Qualcomm%20Hexagon%20HTP-orange.svg)]()
[![License](https://img.shields.io/badge/privacy-100%25%20Offline-success.svg)]()

---

## Overview

**Samvaad** (code-named **Shravan** for its speech perception module) is a high-performance, privacy-first transcription and real-time captioning engine engineered for Qualcomm Snapdragon® X-Series laptops. It leverages the dedicated Qualcomm Hexagon NPU via `onnxruntime-qnn` to deliver sub-second speech-to-text without cloud connectivity or battery drain.

### Core Capabilities
- **Local NPU Execution**: Whisper encoder and decoder loops executed on the Hexagon NPU using the QNN Execution Provider (HTP backend) with `session.disable_cpu_ep_fallback = 1`.
- **100% Privacy & Zero-Cloud**: Operates completely in airplane mode. Zero audio or telemetry ever leaves the device.
- **Accessible Design System**: Built with the official **Samvaad Design System** featuring high-contrast ergonomics, WCAG AAA readability, and multi-modal state indicators.
- **WebSocket Event Bus**: Live asynchronous stream emitting `STATUS` and `TRANSCRIPT` events to any client application.
- **Auditable Benchmarking**: Every latency and power claim is backed by raw, reproducible CSV datasets.

---

## Repository Structure

```
samvaad/
├── config/             # Dynamic YAML settings (no hardcoded constants)
├── app/                # FastAPI application, audio capture ring buffer, energy VAD, SQLite storage
├── inference/          # QNN session factory, Whisper NPU/CPU inference engines, mel filterbank
├── frontend/           # React + Vite + Tailwind CSS accessibility UI
├── models/             # Qualcomm AI Hub model manifests, checksums, and download instructions
├── benchmarks/         # Latency, WER, and psutil resource monitoring harnesses
├── data/               # Test audio assets and local SQLite databases
├── scripts/            # Single-command setup and launch scripts for Windows ARM64 and Linux
├── docs/               # Architecture diagrams, decision logs, and benchmark methodology
└── tests/              # Automated pytest verification test suite
```

---

## Quick Start

### Windows 11 on Snapdragon X-Series (ARM64)
```powershell
# 1. Clone repository
git clone https://github.com/TanayTejaswi/samvaad.git
cd samvaad

# 2. Setup native ARM64 virtual environment & dependencies
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 3. Launch application
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

### Linux / POSIX Development
```bash
# Setup environment
bash scripts/setup_linux.sh

# Run test suite
.venv/bin/pytest -v
```

---

## Engineering Standards & Rules
- **Python**: Python 3.11+ strictly typed.
- **Linting**: Enforced via Ruff (`ruff check .`).
- **Testing**: Automated coverage via Pytest.
- **Hardware Honesty**: No synthetic, estimated, or extrapolated benchmarks.
