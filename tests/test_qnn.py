"""Unit tests for QNN session factory and configuration loading."""

from pathlib import Path

import numpy as np
import pytest

from inference.qnn_session import (
    create_dummy_onnx_model,
    create_session,
    get_qnn_provider_options,
    is_qnn_available,
    load_config,
)


def test_load_config_default():
    """Verify configuration loads properly from YAML with required top-level keys."""
    cfg = load_config()
    assert "app" in cfg
    assert "server" in cfg
    assert "audio" in cfg
    assert "vad" in cfg
    assert "inference" in cfg
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["inference"]["qnn"]["disable_cpu_ep_fallback"] == 1


def test_get_qnn_provider_options():
    """Verify provider options dictionary matches expected HTP configuration."""
    opts = get_qnn_provider_options()
    assert "backend_path" in opts
    assert "htp_performance_mode" in opts
    assert "enable_htp_fp16_precision" in opts


def test_dummy_model_creation_and_inference(tmp_path: Path):
    """Verify generation of dummy ONNX model and running inference through session."""
    dummy_model_path = tmp_path / "dummy.onnx"
    create_dummy_onnx_model(dummy_model_path, feature_dim=80)
    assert dummy_model_path.exists()

    # Create session (CPU fallback allowed on non-Snapdragon host)
    session, provider = create_session(
        dummy_model_path, device="cpu", allow_fallback=True
    )
    assert session is not None
    assert "CPUExecutionProvider" in provider

    # Test running inference
    input_data = np.random.randn(1, 80, 100).astype(np.float32)
    outputs = session.run(["output"], {"input": input_data})
    assert len(outputs) == 1
    # Model scales by 2.0
    np.testing.assert_allclose(outputs[0], input_data * 2.0, rtol=1e-5)


def test_qnn_session_strict_fallback_behavior(tmp_path: Path):
    """Verify that if QNN is not available and allow_fallback=False, RuntimeError is raised."""
    dummy_model_path = tmp_path / "dummy_strict.onnx"
    create_dummy_onnx_model(dummy_model_path, feature_dim=80)

    if not is_qnn_available():
        with pytest.raises(RuntimeError, match="QNNExecutionProvider requested"):
            create_session(dummy_model_path, device="npu", allow_fallback=False)
