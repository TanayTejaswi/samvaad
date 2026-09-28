"""Tests for Phase 1: Audio pipeline (mel & VAD)."""

import numpy as np

from app.vad import EnergyVAD
from inference.mel import log_mel_spectrogram


def test_log_mel_spectrogram_shape():
    """Verify standard Whisper mel feature shape."""
    # Create 3 seconds of dummy audio
    audio = np.random.randn(16000 * 3).astype(np.float32)
    
    mel = log_mel_spectrogram(audio)
    
    # Whisper always expects [1, 80, 3000] regardless of input length
    assert mel.shape == (1, 80, 3000)
    assert mel.dtype == np.float32


def test_vad_processing():
    """Verify VAD chunks audio based on energy."""
    vad = EnergyVAD()
    vad.threshold = 0.5  # High threshold
    vad.silence_frames = 16000 * 1  # 1 sec silence
    
    # Silent block
    silent = np.zeros(16000, dtype=np.float32)
    assert vad.process(silent) is None
    
    # Loud block (trigger speech)
    loud = np.ones(16000, dtype=np.float32) * 1.0
    assert vad.process(loud) is None
    assert vad.is_speaking
    
    # Another silent block (should yield chunk after silence duration)
    # The silence counter will exceed silence_frames
    result = vad.process(silent)
    assert result is not None
    # Result length = loud block + silent block
    assert len(result) == 32000
    assert not vad.is_speaking
