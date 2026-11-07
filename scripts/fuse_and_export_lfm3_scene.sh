#!/usr/bin/env bash
set -euo pipefail

MLX_BF16_DIR="/Users/liuboto/mlx-models/LFM3-1.7B-3D-Coder-bf16"
MLX_4BIT_DIR="/Users/liuboto/mlx-models/LFM3-1.7B-3D-Coder-4bit"
ONNX_DIR="/Users/liuboto/onnx-models/LFM3-1.7B-3D-Coder-ONNX-q4f16"
ADAPTER_PATH="/Users/liuboto/dev/threecoder/adapters/lfm3-1.7b-scene"
GGUF_DIR="/Users/liuboto/mlx-models/gguf"

echo "=== 1. Fusing LFM3-1.7B-3D-Coder LoRA adapter ==="
.venv/bin/python -m mlx_lm.fuse \
  --model LiquidAI/LFM3-1.7B-MLX-bf16 \
  --adapter-path "${ADAPTER_PATH}" \
  --save-path "${MLX_BF16_DIR}"

echo "=== 2. Creating 4-bit MLX Quantization ==="
.venv/bin/python -m mlx_lm.convert \
  --hf-path "${MLX_BF16_DIR}" \
  -q --q-bits 4 \
  --save-path "${MLX_4BIT_DIR}"

echo "=== 3. Exporting to WebGPU ONNX q4f16 ==="
/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python scripts/export_lfm3_onnx_q4f16.py \
  --mlx-dir "${MLX_BF16_DIR}" \
  --out-dir "${ONNX_DIR}"

echo "=== 4. Exporting GGUF & Registering Ollama custom card ==="
mkdir -p "${GGUF_DIR}"
.venv/bin/python scripts/export_lfm3_gguf_and_ollama.py \
  --mlx-dir "${MLX_BF16_DIR}" \
  --gguf-out "${GGUF_DIR}/LFM3-1.7B-3D-Coder-Q8_0.gguf" \
  --ollama-name "lfm3-3d-coder"

echo "=== LFM3-1.7B-3D-Coder Export Complete! ==="
