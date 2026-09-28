# Models Directory

This directory stores the ONNX weights for the Samvaad / Shravan inference engine.
**Note**: Model weights (`*.onnx`, `*.bin`) are explicitly ignored from version control to save space and respect licenses.

## Required Models (Qualcomm AI Hub)

Samvaad relies on **Whisper-Small-Quantized** optimized for the Snapdragon X-Series Hexagon NPU.

### How to obtain the models:
1. Log in to [Qualcomm AI Hub](https://aihub.qualcomm.com/).
2. Navigate to the `Whisper-Small` model card.
3. Download the pre-compiled `QNN ONNX` assets for `Snapdragon X Elite` (or X Plus).
4. Place the `.onnx` files and the tokenizer assets into `models/whisper/`.

*See `manifest.yaml` for exact expected filenames and SHA256 checksums.*
