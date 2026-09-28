# Samvaad Agent Rules & Engineering Directives

## 1. Target Platform Constraints
- **Primary Runtime Target**: Windows 11 on Qualcomm Snapdragon® X-Series architectures (ARM64).
- **Execution Engine**: NPU inference via `onnxruntime-qnn` (QNN Execution Provider with Hexagon NPU / HTP backend).
- **Session Rules**: NPU sessions must be configured with `session.disable_cpu_ep_fallback = 1` to strictly enforce hardware acceleration.
- **Portability**: Provide clean CPU fallback execution providers (`CPUExecutionProvider`) for baseline comparison and cross-platform verification.

## 2. Integrity & Metrics Guidelines (Non-Negotiable)
- **Zero Extrapolations**: Never invent, estimate, or synthesize benchmark numbers, accuracies, or latencies.
- **Auditable Evidence**: Every number cited in documentation, slides, or READMEs must map directly to raw CSV files committed in `benchmarks/results/`.
- **Hardware Verification**: Never claim a model runs on the NPU unless the session was initialized with `session.disable_cpu_ep_fallback = 1` and logged through `session.get_providers()`.
- **Honest Limitations**: State verified capabilities and known constraints transparently.

## 3. Engineering & Code Standards
- **Python Runtime**: Python 3.11+ required across all service modules.
- **Type Annotations**: Complete strict type hints (`mypy` compliant) on all function signatures, class members, and module interfaces.
- **Formatting & Linting**: Strictly formatted and checked via `Ruff`.
- **Testing**: Automated test suite powered by `pytest`. Every phase must have reproducible automated tests.
- **Configuration Management**: Absolute or dynamic local path hardcoding is strictly forbidden. All paths, thresholds, and execution modes must be loaded from `config/*.yaml`.
- **Symmetric Preprocessing**: Feature extraction and preprocessing logic (e.g. `inference/mel.py`) must remain 100% identical between offline verification (`tests/test_smoke.py`) and live audio streaming (`app/audio_capture.py`). Never duplicate preprocessing.
- **Model Storage**: Model weights are never committed to the git repository. `models/README.md` and `models/manifest.yaml` document download URLs, shapes, and SHA256 checksums.
- **Offline Guarantee**: Zero external network calls, cloud APIs, or telemetry at runtime. The application must run completely in airplane mode.

## 4. Design Guidelines (Samvaad Design System)
- **High-Contrast Dark Mode**:
  - Background Primary: `#0F172A` (Deep Slate base layer)
  - Background Secondary: `#1E293B` (Charcoal surface layer)
  - Text Primary: `#F8FAFC` (High-contrast Off-White)
  - Text Muted: `#94A3B8` (Mid-tone slate metadata)
  - Primary Accent: `#6366F1` (Electric Indigo)
  - Secondary Accent: `#14B8A6` (Vibrant Teal)
  - Border / Separator: `#334155` (Subtle boundary)
- **Typography**:
  - Headings & Display: Plus Jakarta Sans
  - Body & UI Controls: Roboto / Inter
  - Code & Telemetry: JetBrains Mono
- **Multi-Modal Accessibility**: Color must NEVER be used as the sole visual indicator for system status or state. Every visual cue must combine color with iconography, high-contrast labels, or distinctive structural patterns.
