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
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Total clients: %d", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:  # noqa: BLE001
                logger.error("Error broadcasting to client: %s", e)
                dead_connections.append(connection)
        
        for dead_conn in dead_connections:
            self.disconnect(dead_conn)

manager = ConnectionManager()


def handle_speech_segment(segment: np.ndarray) -> None:
    """Callback fired when the VAD identifies a complete speech chunk."""
    try:
        # Notify clients that we are transcribing
        loop = asyncio.get_running_loop() if hasattr(asyncio, 'get_running_loop') else asyncio.get_event_loop()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({"type": "STATUS", "status": "transcribing"}),
                loop
            )
        
        # 1 & 2. Inference
        start_time = time.perf_counter()
        
        # If using WhisperRealtimeEngine, pass raw audio
        if hasattr(engine, 'set_raw_audio'):
            engine.set_raw_audio(segment)
            transcript = engine.infer(None)
        else:
            mel = log_mel_spectrogram(segment)
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
    except Exception as e:
        logger.error("Error in speech handler: %s", e, exc_info=True)
    finally:
        # Notify idle
        try:
            current_loop = asyncio.get_running_loop()
            if current_loop and current_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({"type": "STATUS", "status": "idle"}),
                    current_loop
                )
        except RuntimeError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    global engine, audio_streamer, loop
    loop = asyncio.get_running_loop()
    
    # Initialize Realtime Engine
    from inference.whisper_realtime import WhisperRealtimeEngine
    engine = WhisperRealtimeEngine(model_size="base")
    engine.load(config.get("inference", {}).get("model_dir", "models/whisper"))
    engine.warmup()
    
    # Disable background server mic since we use WebSockets for browser audio
    audio_streamer = None
    logger.info("Server microphone capture disabled in favor of WebSocket audio.")
    
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
    # Give this connection its own dedicated VAD instance!
    from app.vad import EnergyVAD
    ws_vad = EnergyVAD(config)
    
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
                
            if message.get("bytes") is not None:
                audio_bytes = message["bytes"]
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Debug logging every 50th block
                energy = np.sqrt(np.mean(np.square(audio_array)))
                if getattr(websocket, "block_count", 0) % 50 == 0:
                    logger.info("Received WS audio chunk. Energy: %f, Size: %d", energy, len(audio_array))
                websocket.block_count = getattr(websocket, "block_count", 0) + 1
                
                # Use the dedicated VAD!
                segment = ws_vad.process(audio_array)
                if segment is not None:
                    logger.info("WebSocket VAD triggered! Segment length: %d", len(segment))
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, handle_speech_segment, segment)
    except Exception as e:
        logger.warning("WebSocket connection dropped: %s", e)
    finally:
        manager.disconnect(websocket)

@app.websocket("/video")
async def video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    from engine.sign_engine import SignEngine
    from language.gloss2text import GlossToTextEngine
    import cv2
    import time
    
    sign_engine = SignEngine()
    llm_engine = GlossToTextEngine()
    
    from app.tts import TTSEngine
    tts_engine = TTSEngine()
    
    # Sentence buffer to accumulate glosses
    sentence_buffer = []
    last_sign_time = time.time()
    sentence_timeout = 2.5 # translate after 2.5s of no signing
    
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
                
            if message.get("bytes") is not None:
                # Decode JPEG frame
                np_arr = np.frombuffer(message["bytes"], np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # 1. Check for sentence timeout
                    if sentence_buffer and (time.time() - last_sign_time) > sentence_timeout:
                        gloss_seq = " ".join(sentence_buffer)
                        translation = llm_engine.translate(gloss_seq)
                        
                        # Phase 8: Speak the translated sentence out loud!
                        tts_engine.speak(translation["text"])
                        
                        await manager.broadcast({
                            "type": "TRANSCRIPT",
                            "text": translation["text"],
                            "source": "sign_translated",
                            "timestamp": int(time.time() * 1000),
                            "latency_ms": translation["latency_ms"],
                            "device": translation["backend"]
                        })
                        sentence_buffer.clear()
                        
                    # 2. Process frame
                    result = sign_engine.process_frame(frame)
                    
                    if "landmarks" in result:
                        del result["landmarks"]
                        
                    if result.get("type") == "SIGN_RECOGNIZED":
                        gloss = result.get("gloss", "")
                        if gloss and gloss != "UNSURE":
                            sentence_buffer.append(gloss)
                            last_sign_time = time.time()
                            
                        await manager.broadcast(result)
    except Exception as e:
        logger.warning("Video WebSocket dropped: %s", e)
    finally:
        manager.disconnect(websocket)

# Mount static frontend
static_dir = Path(config.get("server", {}).get("static_dir", "frontend/dist"))
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
else:
    logger.warning("Frontend static directory %s not found. API only mode.", static_dir)
