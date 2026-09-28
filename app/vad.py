"""Energy-based Voice Activity Detection (VAD).

Slices continuous audio streams into discrete speech chunks using 
configurable energy thresholds, hangover logic, and duration limits.
"""

from typing import Any

import numpy as np

from inference.qnn_session import load_config


class EnergyVAD:
    """Stateful Energy-based VAD for real-time chunking."""
    
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        vad_cfg = cfg.get("vad", {})
        audio_cfg = cfg.get("audio", {})
        
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.threshold = float(vad_cfg.get("energy_threshold", 0.015))
        self.silence_frames = int(
            float(vad_cfg.get("silence_duration_s", 0.8)) * self.sample_rate
        )
        self.min_speech_frames = int(
            float(vad_cfg.get("min_speech_duration_s", 0.4)) * self.sample_rate
        )
        self.max_speech_frames = int(
            float(vad_cfg.get("max_speech_duration_s", 30.0)) * self.sample_rate
        )
        self.pad_frames = int(
            float(vad_cfg.get("speech_pad_s", 0.2)) * self.sample_rate
        )
        
        self.is_speaking = False
        self.current_chunk: list[np.ndarray] = []
        self.current_length = 0
        self.silence_counter = 0

    def process(self, audio_block: np.ndarray) -> np.ndarray | None:
        """Processes an incoming block of audio.
        
        Args:
            audio_block: 1D numpy array of float32 PCM audio.
            
        Returns:
            np.ndarray if a valid speech chunk is completed, else None.
        """
        # Calculate RMS energy of the block
        energy = np.sqrt(np.mean(np.square(audio_block)))
        
        if energy > self.threshold:
            self.is_speaking = True
            self.silence_counter = 0
            self.current_chunk.append(audio_block)
            self.current_length += len(audio_block)
            
            # Force yield if max duration reached
            if self.current_length >= self.max_speech_frames:
                return self._yield_chunk()
        else:
            if self.is_speaking:
                self.silence_counter += len(audio_block)
                self.current_chunk.append(audio_block)
                self.current_length += len(audio_block)
                
                # If silence duration exceeded, close the chunk
                if self.silence_counter >= self.silence_frames:
                    return self._yield_chunk()
        
        return None

    def _yield_chunk(self) -> np.ndarray | None:
        """Assembles the current chunk and resets state."""
        if not self.current_chunk:
            self.is_speaking = False
            self.silence_counter = 0
            self.current_length = 0
            return None
            
        full_audio = np.concatenate(self.current_chunk)
        
        # Reset state
        self.is_speaking = False
        self.current_chunk = []
        self.current_length = 0
        self.silence_counter = 0
        
        # Validate minimum length
        if len(full_audio) < self.min_speech_frames:
            return None
            
        return full_audio
