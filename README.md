# Samvaad: Two-Way ISL Communication Bridge

Samvaad is an offline, real-time communication bridge between Indian Sign Language (ISL) users and hearing people, specifically optimized for Windows 11 on Snapdragon X-series (ARM64) laptops with Hexagon NPUs.

## Quick Start on Your Snapdragon Laptop

To get the full pipeline (including Hand Tracking, SignNet AI, and the Local LLM) running natively on your laptop, follow these exact steps:

### 1. Clone & Install Dependencies
Open your Windows Terminal (PowerShell) and run:
```powershell
git clone https://github.com/TanayTejaswi/samvaad.git
cd samvaad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn websockets onnxruntime numpy opencv-python pyttsx3 pyyaml requests
```
*(Note: Do not install `mediapipe`. The hand tracking engine was custom-written to run natively without it).*

### 2. Download Qualcomm ONNX Models
To enable the camera to track your hands, you must download the pre-compiled models from AI Hub:
1. Make sure you have a Python `< 3.14` environment.
2. `pip install qai-hub-models`
3. Download the `MediaPipe-Hand-Detection` ONNX files.
4. Place them in: `samvaad/models/hands/palm_detection.onnx` and `samvaad/models/hands/hand_landmark.onnx`.

*(If you skip this step, the engine will gracefully fall back to "Simulate" mode).*

### 3. Install Ollama (For Fluent English Translation)
To translate broken ISL grammar (`"ME COLLEGE GO"`) into fluent English (`"I am going to college."`), you need the local LLM running.
1. Download Ollama for Windows from [ollama.com](https://ollama.com).
2. Open terminal and run: `ollama run llama3.2:3b`
3. Keep it running in the background.

### 4. Run the Samvaad Server
With your models in place and Ollama running, start the server!
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your Chrome/Edge browser. Allow camera and microphone permissions.

---

### How to use Personal Sign Enrollment (Phase 9)
Want to teach the AI a custom slang word? Open a new terminal and run:
```powershell
python scripts/enroll_cli.py
```
Follow the prompt to sign the word 5 times. The AI will extract the 128-d NPU embedding and hijack the neural network the next time you sign it!
