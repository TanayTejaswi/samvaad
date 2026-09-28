# Samvaad / Shravan

An offline, Hexagon NPU-accelerated live captioning application designed specifically for the Snapdragon X-Series (Windows on ARM64). This project adheres to strict local execution and data privacy rules, eliminating all external cloud dependencies.

## Architecture

- **Hardware Target**: Snapdragon X Elite / X Plus (ARM64)
- **Engine**: Qualcomm AI Hub Whisper-Small-Quantized
- **Runtime**: ONNX Runtime (QNN Execution Provider)
- **Backend**: FastAPI + WebSockets + SQLite (Local storage)
- **Frontend**: React + Vite (Tailwind UI with Glassmorphism)
- **Audio Capture**: Real-time ring buffer using `sounddevice`

## Setup & Installation

### Windows on ARM64 (Primary Target)
1. Ensure Python 3.11+ is installed.
2. Run the setup script to create the environment:
   ```powershell
   .\scripts\setup.ps1
   ```
3. Download the Whisper model files from Qualcomm AI Hub (see `models/README.md`) and place them in the `models/` folder.

### Linux x86_64 (Development / Evaluation Fallback)
The engine automatically falls back to `CPUExecutionProvider` when running unit tests or working on a non-ARM64 dev host.
```bash
./scripts/setup_linux.sh
```

## Running the Application

### 1. Start the Backend Server
This hosts the REST API, WebSocket bus, and SQLite background daemon:
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
In a new terminal:
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to view the live dashboard.

## Benchmarks & Evaluation

To evaluate the latency (P95, P99) of the NPU versus CPU fallback:
```bash
python benchmarks/latency.py
```
To calculate the Word Error Rate (WER):
```bash
python benchmarks/wer.py
```

All outputs are saved as CSV files inside the `data/` directory.

## Testing
Run the comprehensive unit test suite:
```bash
pytest tests/
```
