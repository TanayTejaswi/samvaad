"""FastAPI server for Samvaad backend.

Serves the REST API, WebSocket bus, and hosts the React frontend.
Orchestrates the Audio Capture and Inference engines.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio_capture import AudioStreamer
from app.storage import StorageManager
from inference.mel import log_mel_spectrogram
from inference.qnn_session import load_config
from inference.whisper_cpu import WhisperCPUEngine
from inference.whisper_npu import WhisperNPUEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("samvaad.server")

# Global state
config = load_config()
storage = StorageManager(config)
engine: Any = None
audio_streamer: AudioStreamer | None = None
active_connections: set[WebSocket] = set()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total clients: %d", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:  # noqa: BLE001
                logger.error("Error broadcasting to client: %s", e)

manager = ConnectionManager()


def handle_speech_segment(segment: np.ndarray) -> None:
    """Callback fired when the VAD identifies a complete speech chunk."""
    # Notify clients that we are transcribing
    asyncio.run_coroutine_threadsafe(
        manager.broadcast({"type": "STATUS", "status": "transcribing"}),
        loop
    )
    
    # 1. Mel extraction
    start_time = time.perf_counter()
    mel = log_mel_spectrogram(segment)
    
    # 2. Inference
    transcript = engine.infer(mel)
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    
    from datetime import timezone
    device_label = "npu" if "QNNExecutionProvider" in engine.backend_name else "cpu"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # 3. Save to DB
    storage.save_transcript(timestamp, transcript, latency_ms, device_label)
    
    # 4. Broadcast
    asyncio.run_coroutine_threadsafe(
        manager.broadcast({
            "type": "TRANSCRIPT",
            "timestamp": timestamp,
            "text": transcript,
            "latency_ms": latency_ms,
            "device": device_label
        }),
        loop
    )
    
    # Notify idle
    asyncio.run_coroutine_threadsafe(
        manager.broadcast({"type": "STATUS", "status": "idle"}),
        loop
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    global engine, audio_streamer, loop
    loop = asyncio.get_running_loop()
    
    # Initialize Engine
    model_dir = config.get("inference", {}).get("model_dir", "models/whisper")
    default_device = config.get("inference", {}).get("default_device", "npu")
    
    if default_device == "npu":
        engine = WhisperNPUEngine()
    else:
        engine = WhisperCPUEngine()
        
    engine.load(model_dir)
    engine.warmup()
    
    # Initialize Audio Streamer
    # For testing without a microphone, we can mock or disable this.
    try:
        audio_streamer = AudioStreamer(on_segment_ready=handle_speech_segment, config=config)
        audio_streamer.start()
    except Exception as e:  # noqa: BLE001
        logger.error("Microphone capture disabled: %s", e)
    
    yield
    
    # Shutdown
    if audio_streamer:
        audio_streamer.stop()


app = FastAPI(title="Samvaad Backend", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("server", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    active_device: str


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        models_loaded=engine is not None,
        active_device=engine.backend_name if engine else "none"
    )


@app.get("/api/history")
async def get_history():
    return storage.get_history()

@app.post("/api/simulate")
async def simulate_speech():
    """Simulates a speech segment for testing UI without a mic."""
    # Create a dummy 16kHz audio array (3 seconds)
    import numpy as np
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
    # Fire the handler asynchronously so we don't block the HTTP response
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, handle_speech_segment, dummy_audio)
    return {"status": "simulating"}

@app.websocket("/captions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                # Browser sends Float32 PCM array buffer
                audio_bytes = message["bytes"]
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # We can route it directly through the VAD if we have an instance
                if audio_streamer and audio_streamer.vad:
                    segment = audio_streamer.vad.process(audio_array)
                    if segment is not None:
                        # VAD completed a chunk! Run it!
                        # Use executor to avoid blocking the async event loop
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(None, handle_speech_segment, segment)
            elif "text" in message:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mount static frontend
static_dir = Path(config.get("server", {}).get("static_dir", "frontend/dist"))
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
else:
    logger.warning("Frontend static directory %s not found. API only mode.", static_dir)
