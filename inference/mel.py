"""Log-Mel Spectrogram Feature Extraction for Whisper.

Implements identical preprocessing for live audio capture and offline testing.
Converts 16kHz PCM audio into 80-bin log-mel spectrograms of shape [1, 80, 3000].
"""

import numpy as np


# Exact Mel filterbank generation matching OpenAI Whisper
def exact_div(x, y):
    assert x % y == 0
    return x // y

def get_mel_filters(sr: int, n_fft: int, n_mels: int = 80) -> np.ndarray:
    """Initialize Mel filterbank.
    
    Ported to pure numpy to avoid librosa dependency while maintaining
    exact parity with Whisper's expected filterbank.
    """
    # HTK Mel scaling
    fmin = 0.0
    fmax = sr / 2.0
    
    # 2595.0 * np.log10(1.0 + f / 700.0)
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    
    # Whisper drops the Nyquist frequency bin
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)[:-1]
    
    fdiff = np.diff(hz)
    ramps = np.subtract.outer(hz, fft_freqs)
    
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    
    weights = np.maximum(0, np.minimum(lower, upper))
    
    # Slaney-style normalization
    enorm = 2.0 / (hz[2:n_mels+2] - hz[:n_mels])
    weights *= enorm[:, np.newaxis]
    
    return weights

# Pre-compute filterbank on module load
MEL_FILTERS = get_mel_filters(16000, 400, 80)

def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    padding: int = 0,
) -> np.ndarray:
    """Compute the log-mel spectrogram of an audio signal.
    
    Args:
        audio: 1D numpy array of 16kHz PCM float32 audio.
        n_mels: Number of Mel bins (default 80).
        padding: Frames to pad.
        
    Returns:
        np.ndarray of shape [1, 80, 3000].
    """
    if audio.ndim != 1:
        audio = audio.flatten()
        
    # Whisper expects exactly 480000 samples (30 seconds at 16kHz)
    if audio.shape[0] < 480000:
        audio = np.pad(audio, (0, 480000 - audio.shape[0]), mode='constant')
    elif audio.shape[0] > 480000:
        audio = audio[:480000]

    # STFT parameters for Whisper
    n_fft = 400
    hop_length = 160
    
    # Pad audio to center frames
    audio = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')
    
    # Frame audio
    n_frames = 1 + (audio.shape[0] - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0])
    )
    
    # Apply Hann window
    window = np.hanning(n_fft + 1)[:-1]
    frames = frames * window
    
    # RFFT and magnitude
    stft = np.fft.rfft(frames, n=n_fft, axis=-1)
    magnitudes = np.abs(stft[:, :-1]) ** 2
    
    # Apply Mel filterbank
    mel_spec = np.dot(magnitudes, MEL_FILTERS.T)
    
    # Log compression
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    
    # Dynamic range scaling (max - 8.0)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    
    # Normalize to [-1.0, 1.0] approx
    log_spec = (log_spec + 4.0) / 4.0
    
    # Transpose to [n_mels, n_frames]
    out = log_spec.T
    
    # Pad or truncate to exactly 3000 frames (30 seconds)
    if out.shape[1] > 3000:
        out = out[:, :3000]
    elif out.shape[1] < 3000:
        pad_width = 3000 - out.shape[1]
        out = np.pad(out, ((0, 0), (0, pad_width)), mode='constant')
        
    return np.expand_dims(out, axis=0).astype(np.float32)
