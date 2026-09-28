"""Audio Capture Pipeline using sounddevice and ring buffer.

Captures real-time 16kHz audio from the default microphone, buffers it,
and routes it through VAD to extract discrete speech segments.
"""

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd

from app.vad import EnergyVAD
from inference.qnn_session import load_config

logger = logging.getLogger("samvaad.audio_capture")


class AudioStreamer:
    """Non-blocking audio capture stream with VAD integration."""

    def __init__(
        self,
        on_segment_ready: Callable[[np.ndarray], None],
        config: dict[str, Any] | None = None,
    ):
        cfg = config or load_config()
        audio_cfg = cfg.get("audio", {})
        
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.channels = int(audio_cfg.get("channels", 1))
        
        # Block size for callback (e.g., 30ms)
        block_ms = float(audio_cfg.get("block_size_ms", 30))
        self.block_size = int(self.sample_rate * block_ms / 1000)
        
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.on_segment_ready = on_segment_ready
        
        self.vad = EnergyVAD(cfg)
        self.stream: sd.InputStream | None = None
        
        self._stop_event = threading.Event()
        self._process_thread: threading.Thread | None = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Called by sounddevice for each new audio block."""
        if status:
            logger.warning("Audio capture status: %s", status)
        
        # sounddevice gives (frames, channels) layout
        # We need a 1D float32 array
        audio_block = indata.copy().flatten()
        self.q.put(audio_block)

    def _process_loop(self) -> None:
        """Background thread consuming the audio queue."""
        while not self._stop_event.is_set():
            try:
                # Use timeout to allow checking _stop_event
                block = self.q.get(timeout=0.1)
                
                # Pass through VAD
                segment = self.vad.process(block)
                if segment is not None:
                    # Valid speech segment completed
                    self.on_segment_ready(segment)
                    
            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                logger.error("Error in audio process loop: %s", e)

    def start(self) -> None:
        """Starts the audio capture stream and processing thread."""
        if self.stream is not None:
            return
            
        self._stop_event.clear()
        
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="AudioProcessThread"
        )
        self._process_thread.start()
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=None,  # default input device
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        logger.info("Audio capture started at %d Hz.", self.sample_rate)

    def stop(self) -> None:
        """Stops capture and cleans up resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        self._stop_event.set()
        if self._process_thread is not None:
            self._process_thread.join(timeout=1.0)
            self._process_thread = None
            
        logger.info("Audio capture stopped.")
