# Samvaad / Shravan

An offline, live captioning application designed for real-time speech perception. 
This project features a beautiful shiny yellow/black/white UI and currently runs locally on the CPU using `faster-whisper` (supporting English and Hinglish seamlessly out of the box). It is architected to ultimately target the Snapdragon X-Series Hexagon NPU.

## Architecture

- **Backend**: FastAPI + WebSockets + SQLite (Local storage)
- **Frontend**: React + Vite + Tailwind CSS (Beautiful Custom Aesthetic)
- **ML Engine**: `faster-whisper` (Base model) via CTranslate2 CPU backend
- **Audio Capture**: Browser Web Audio API Float32 streaming via WebSockets
- **VAD**: Custom real-time Energy-based Voice Activity Detection

## Setup & Installation

You need Python 3.11+ and Node.js v20+.

### 1. Backend Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Build
Navigate to the frontend folder, install dependencies, and build the static assets:
```bash
cd frontend
npm install
npm run build
cd ..
```

*(Note: The backend is configured to automatically serve the `frontend/dist` folder on port 8000, so you don't need a separate frontend dev server once built).*

## Running the Application

Start the backend server (which also serves the frontend):
```bash
source .venv/bin/activate
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Navigate to **http://localhost:8000** in your browser. 
Click **Start Mic** and start speaking! The app will automatically detect your speech (English or Hinglish) and transcribe it locally.

## Features

- **100% Offline**: No data leaves your machine.
- **Hinglish Support**: The `base` model automatically detects and transcribes English, Hindi, or a mix of both.
- **Ultra-Sensitive VAD**: Mathematically tuned to ignore room hiss but capture quiet speech.
- **Live Local Database**: Transcripts are instantly saved to `data/transcripts.db`.
