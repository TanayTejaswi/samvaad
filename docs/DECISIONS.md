# Engineering Decisions & Architecture Log

This document tracks all foundational architectural decisions made during development to ensure auditable engineering standards and rigorous defense during evaluation.

---

## DEC-0001: Native Windows ARM64 & QNN HTP Provider Targeting
- **Date**: 2026-09-29
- **Status**: Accepted
- **Context**: Qualcomm Snapdragon® X-Series PCs feature a dedicated Hexagon NPU capable of up to 45 TOPS. Native execution requires avoiding x86 emulation overhead for performance-critical ML inference.
- **Decision**: Target Windows 11 on ARM64 as primary runtime. Use `onnxruntime-qnn` with `QNNExecutionProvider` targeting the `QnnHtp.dll` backend.
- **Enforcement**: Configure `session.disable_cpu_ep_fallback = 1` so that any failure to delegate to the NPU causes an immediate explicit exception rather than silent CPU degradation.
- **Fallback Policy**: Provide an explicit configuration switch (`inference.allow_cpu_fallback: true`) for development, CI/CD, and non-Snapdragon evaluation environments.

---

## DEC-0002: Dynamic Configuration Over Hardcoded Constants
- **Date**: 2026-09-29
- **Status**: Accepted
- **Context**: The Samvaad engineering rules explicitly forbid hardcoding paths, audio parameters, or environmental thresholds.
- **Decision**: All application components dynamically load their settings from `config/default.yaml` via `inference/qnn_session.py:load_config()`. No absolute file paths are committed to the codebase.

---

## DEC-0003: Samvaad High-Contrast Accessibility Design System
- **Date**: 2026-09-29
- **Status**: Accepted
- **Context**: Accessibility tools must minimize cognitive fatigue and maintain strict WCAG AAA contrast ratios.
- **Decision**: Enforce the official Samvaad color palette in Tailwind CSS:
  - Base: `#0F172A` (Deep Slate)
  - Surface: `#1E293B` (Charcoal)
  - Typography: `#F8FAFC` (Off-White) & `#94A3B8` (Muted Slate)
  - Primary Accent: `#6366F1` (Electric Indigo)
  - Secondary Accent: `#14B8A6` (Vibrant Teal)
  - Borders: `#334155`
- **Rule**: Never use color as the sole visual status indicator; all status elements combine color with iconography and high-contrast text badges.

---

## DEC-0004: Strict Offline Execution & Zero-Telemetry Contract
- **Date**: 2026-09-29
- **Status**: Accepted
- **Context**: Medical and personal conversation transcription requires total data privacy. Cloud dependency introduces latency and privacy violations.
- **Decision**: Zero external network calls at runtime. No external CDN fonts or scripts; all static assets bundled locally. Fully functional in airplane mode.
