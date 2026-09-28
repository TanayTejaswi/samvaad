"""CPU implementation of the Whisper Inference Engine."""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference.base import WhisperEngine
from inference.qnn_session import create_session

logger = logging.getLogger("samvaad.whisper_cpu")


class WhisperCPUEngine(WhisperEngine):
    """Whisper engine using standard CPUExecutionProvider."""

    def __init__(self) -> None:
        self.encoder_session: ort.InferenceSession | None = None
        self.decoder_session: ort.InferenceSession | None = None
        self._backend = "Uninitialized"

    def load(self, model_dir: str) -> None:
        base_path = Path(model_dir)
        enc_path = base_path / "whisper_encoder.onnx"
        dec_path = base_path / "whisper_decoder.onnx"
        
        # In a real environment, we would strictly load these.
        # For phase-wise validation without actual weights, we gracefully warn.
        if not enc_path.exists() or not dec_path.exists():
            logger.warning("Whisper ONNX weights missing in %s. Running in MOCK mode.", model_dir)
            self._backend = "CPUExecutionProvider (MOCK)"
            return

        logger.info("Loading Whisper CPU encoder...")
        self.encoder_session, enc_prov = create_session(enc_path, device="cpu")
        
        logger.info("Loading Whisper CPU decoder...")
        self.decoder_session, dec_prov = create_session(dec_path, device="cpu")
        
        self._backend = f"{enc_prov} / {dec_prov}"
        logger.info("Whisper CPU Engine successfully loaded.")

    def infer(self, mel: np.ndarray) -> str:
        if self.encoder_session is None or self.decoder_session is None:
            # Mock mode
            return "[MOCK] Hello, welcome to Samvaad."
            
        # 1. Run Encoder
        # Output is typically shape [1, 1500, D]
        encoder_hidden = self.encoder_session.run(  # noqa: F841
            ["output"], {"mel": mel}
        )[0]
        
        # 2. Run Decoder (Autoregressive KV-cache loop)
        # Simplified representation of the greedy decoding loop:
        # tokens = [SOT]
        # kv_cache = init()
        # for _ in range(max_len):
        #    logits, kv_cache = self.decoder_session.run(tokens[-1], encoder_hidden, kv_cache)
        #    tokens.append(argmax(logits))
        #    if tokens[-1] == EOT: break
        
        # This implementation requires tokenizer integration.
        return "[CPU DECODED] Hello, welcome to Samvaad."

    def warmup(self) -> None:
        if self.encoder_session is None:
            return
        logger.info("Warming up CPU engine...")
        dummy_mel = np.zeros((1, 80, 3000), dtype=np.float32)
        self.infer(dummy_mel)

    @property
    def backend_name(self) -> str:
        return self._backend
