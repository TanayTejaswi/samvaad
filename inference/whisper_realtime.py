"""Real Whisper inference using faster-whisper (CTranslate2 backend)."""

import logging
import numpy as np
from faster_whisper import WhisperModel
from inference.base import WhisperEngine

logger = logging.getLogger("samvaad.whisper_realtime")


class WhisperRealtimeEngine(WhisperEngine):
    """Real Whisper engine using faster-whisper for actual transcription."""
    
    def __init__(self, model_size: str = "tiny.en"):
        self.model_size = model_size
        self.model = None
        self._backend = "Uninitialized"
        self._raw_audio = None
    
    def load(self, model_dir: str) -> None:
        logger.info("Loading faster-whisper model: %s", self.model_size)
        self.model = WhisperModel(
            self.model_size, 
            device="cpu", 
            compute_type="int8"
        )
        self._backend = f"CTranslate2-CPU ({self.model_size})"
        logger.info("Model loaded: %s", self._backend)
    
    def infer(self, mel: np.ndarray | None) -> str:
        """Transcribe from raw audio segment (not mel - faster-whisper does its own)."""
        if self._raw_audio is None:
            return "[No audio segment]"
        
        segments, info = self.model.transcribe(
            self._raw_audio,
            beam_size=1,
            vad_filter=False,  # We already did VAD
        )
        text = " ".join(seg.text.strip() for seg in segments)
        self._raw_audio = None
        return text if text else "[silence]"
    
    def set_raw_audio(self, audio: np.ndarray):
        """Store raw audio for transcription."""
        self._raw_audio = audio
    
    def warmup(self) -> None:
        if self.model is None:
            return
        logger.info("Warming up faster-whisper...")
        dummy = np.zeros(16000, dtype=np.float32)
        self._raw_audio = dummy
        self.infer(None)
    
    @property
    def backend_name(self) -> str:
        return self._backend
