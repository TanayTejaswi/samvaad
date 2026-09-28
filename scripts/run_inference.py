#!/usr/bin/env python3
"""Standalone script to execute inference on a sample audio file.

Validates the end-to-end integration of Audio -> Mel -> Encoder -> Decoder on NPU/CPU.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure samvaad root is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from inference.mel import log_mel_spectrogram
from inference.qnn_session import load_config
from inference.whisper_cpu import WhisperCPUEngine
from inference.whisper_npu import WhisperNPUEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("samvaad.run_inference")


def main():
    parser = argparse.ArgumentParser(description="Test Samvaad Whisper Inference")
    parser.add_argument("--wav", type=str, help="Path to input .wav file (16kHz)")
    parser.add_argument("--device", type=str, choices=["npu", "cpu"], default="cpu")
    args = parser.parse_args()

    config = load_config()
    model_dir = config.get("inference", {}).get("model_dir", "models/whisper")

    # 1. Initialize Engine
    if args.device == "npu":
        engine = WhisperNPUEngine()
    else:
        engine = WhisperCPUEngine()

    logger.info("Initializing %s engine...", args.device.upper())
    engine.load(model_dir)
    engine.warmup()
    logger.info("Engine initialized. Backend: %s", engine.backend_name)

    # 2. Get Audio
    if args.wav:
        logger.info("Loading audio from %s", args.wav)
        sr, audio_int16 = wavfile.read(args.wav)
        if sr != 16000:
            logger.error("Audio must be 16kHz.")
            sys.exit(1)
        audio = audio_int16.astype(np.float32) / 32768.0
    else:
        logger.info("No wav file provided. Recording 3 seconds from microphone...")
        audio = sd.rec(3 * 16000, samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        logger.info("Recording complete.")

    # 3. Preprocess
    start_time = time.perf_counter()
    mel = log_mel_spectrogram(audio)
    mel_time = time.perf_counter() - start_time
    logger.info("Mel spectrogram extracted. Shape: %s (Time: %.2f ms)", mel.shape, mel_time * 1000)

    # 4. Infer
    start_time = time.perf_counter()
    transcript = engine.infer(mel)
    infer_time = time.perf_counter() - start_time

    # 5. Output
    print("\n" + "="*50)
    print(" SAMVAAD INFERENCE RESULT")
    print("="*50)
    print(f" DEVICE:     {args.device.upper()}")
    print(f" BACKEND:    {engine.backend_name}")
    print(f" LATENCY:    {infer_time * 1000:.2f} ms")
    print("-" * 50)
    print(f" TRANSCRIPT: {transcript}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
