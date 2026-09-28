"""Base interface for Whisper Inference Engines."""

from abc import ABC, abstractmethod

import numpy as np


class WhisperEngine(ABC):
    """Abstract base class for all speech-to-text inference backends."""
    
    @abstractmethod
    def load(self, model_dir: str) -> None:
        """Loads the models into memory and initializes the session."""
        
    @abstractmethod
    def infer(self, mel: np.ndarray) -> str:
        """Runs inference on a log-mel spectrogram.
        
        Args:
            mel: [1, 80, 3000] float32 log-mel features.
            
        Returns:
            Decoded string transcript.
        """
        
    @abstractmethod
    def warmup(self) -> None:
        """Performs a dummy inference pass to initialize hardware caches."""
        
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Returns the active provider backend (e.g., QNNExecutionProvider)."""
