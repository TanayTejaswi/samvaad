"""QNN Session Factory for Qualcomm Snapdragon Hexagon NPU.

Instantiates ONNX Runtime sessions targeting the Qualcomm Hexagon NPU via
the QNN Execution Provider (HTP backend) with strict enforcement of
`session.disable_cpu_ep_fallback = 1`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper

logger = logging.getLogger("samvaad.qnn_session")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads configuration from YAML file dynamically without hardcoded paths."""
    if config_path is None:
        # Default relative to repository root
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def is_qnn_available() -> bool:
    """Checks whether QNNExecutionProvider is available in the onnxruntime build."""
    available = ort.get_available_providers()
    return "QNNExecutionProvider" in available


def get_qnn_provider_options(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Builds provider options dictionary for QNNExecutionProvider from config."""
    cfg = config or load_config()
    qnn_cfg = cfg.get("inference", {}).get("qnn", {})

    backend_path = qnn_cfg.get("backend_path", "QnnHtp.dll")
    perf_mode = qnn_cfg.get("perf_mode", "burst")
    enable_fp16 = str(qnn_cfg.get("enable_htp_fp16_precision", "1"))

    return {
        "backend_path": backend_path,
        "htp_performance_mode": perf_mode,
        "enable_htp_fp16_precision": enable_fp16,
    }


def create_session(
    model_path: str | Path,
    device: str = "npu",
    config: dict[str, Any] | None = None,
    allow_fallback: bool | None = None,
) -> tuple[ort.InferenceSession, str]:
    """Creates and configures an ONNX Runtime InferenceSession.

    Args:
        model_path: Path to the .onnx model file.
        device: Target execution device, either 'npu' or 'cpu'.
        config: Optional loaded configuration dictionary.
        allow_fallback: Whether to permit fallback to CPU if NPU is unavailable.
            Defaults to configuration value if not specified.

    Returns:
        Tuple of (InferenceSession, active_provider_name).

    Raises:
        RuntimeError: If device='npu' is requested, QNN is unavailable,
            and allow_fallback is False.
        FileNotFoundError: If model_path does not exist.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    cfg = config or load_config()
    inf_cfg = cfg.get("inference", {})
    if allow_fallback is None:
        allow_fallback = bool(inf_cfg.get("allow_cpu_fallback", True))

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    requested_device = device.lower()

    if requested_device == "npu":
        if is_qnn_available():
            # Strictly enforce zero CPU fallback in hardware execution per rules
            sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            qnn_options = get_qnn_provider_options(cfg)

            providers: list[Any] = [("QNNExecutionProvider", qnn_options)]
            logger.info("Initializing ONNX session with QNNExecutionProvider (HTP NPU).")
            session = ort.InferenceSession(
                str(model_path), sess_options=sess_options, providers=providers
            )
            active_providers = session.get_providers()
            logger.info("Active providers on session: %s", active_providers)
            return session, "QNNExecutionProvider"

        # QNN not available on current host
        if not allow_fallback:
            raise RuntimeError(
                "QNNExecutionProvider requested with CPU fallback disabled, "
                "but QNN is not available in the current environment."
            )

        logger.warning(
            "QNNExecutionProvider not found in runtime. "
            "Falling back to CPUExecutionProvider as permitted by configuration."
        )
        requested_device = "cpu"

    # CPU Session creation
    cpu_cfg = inf_cfg.get("cpu", {})
    sess_options.intra_op_num_threads = int(cpu_cfg.get("intra_op_num_threads", 4))
    providers = ["CPUExecutionProvider"]
    logger.info("Initializing ONNX session with CPUExecutionProvider.")
    session = ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers
    )
    active_providers = session.get_providers()
    logger.info("Active providers on session: %s", active_providers)
    return session, "CPUExecutionProvider"


def create_dummy_onnx_model(output_path: str | Path, feature_dim: int = 80) -> Path:
    """Generates a minimal valid ONNX model for smoke-testing and NPU validation.

    Computes: Y = X * 2.0 (Identity/Scale)
    Input shape:  [1, feature_dim, 100]
    Output shape: [1, feature_dim, 100]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, feature_dim, 100]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, feature_dim, 100]
    )

    scale_tensor = helper.make_tensor(
        name="scale",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[2.0],
    )

    mul_node = helper.make_node(
        "Mul",
        inputs=["input", "scale"],
        outputs=["output"],
        name="scale_node",
    )

    graph = helper.make_graph(
        nodes=[mul_node],
        name="DummyTestModel",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[scale_tensor],
    )

    opset = helper.make_opsetid("", 17)
    model = helper.make_model(
        graph,
        producer_name="samvaad_test",
        opset_imports=[opset],
        ir_version=10,
    )
    onnx.save(model, str(output_path))
    logger.info("Created dummy test ONNX model at %s", output_path)
    return output_path
