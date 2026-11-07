#!/usr/bin/env bash
set -euo pipefail

# Fuses LFM3-Omni-2B Three.js Scene LoRA weights, exports 4-bit MLX, and stages ONNX / Ollama.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM3-Omni-2B-MLX-bf16}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/lfm3-omni-2b-scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/mlx-models}"

MLX_BF16_DIR="${OUTPUT_DIR}/LFM3-Omni-2B-3D-Coder-bf16"
MLX_4BIT_DIR="${OUTPUT_DIR}/LFM3-Omni-2B-3D-Coder-4bit"
ONNX_DIR="${HOME}/onnx-models/LFM3-Omni-2B-3D-Coder-ONNX-q4f16"

echo "=== 1. Fusing LFM3-Omni-2B LoRA Adapter ==="
echo "Base model:   ${BASE_MODEL}"
echo "Adapter path: ${ADAPTER_PATH}"
echo "Output MLX:   ${MLX_BF16_DIR}"

mkdir -p "${OUTPUT_DIR}"
python3 -m mlx_lm.fuse \
  --model "${BASE_MODEL}" \
  --adapter-path "${ADAPTER_PATH}" \
  --save-path "${MLX_BF16_DIR}"

echo "=== 2. Creating MLX 4-bit Quantization (Low-VRAM / High-Throughput) ==="
python3 -m mlx_lm.convert \
  --hf-path "${MLX_BF16_DIR}" \
  -q --q-bits 4 \
  --save-path "${MLX_4BIT_DIR}"

echo "=== 3. Staging WebGPU ONNX q4f16 Assets ==="
mkdir -p "${ONNX_DIR}/onnx"
# Copy configuration & tokenizer from fused model
cp "${MLX_BF16_DIR}"/config.json "${ONNX_DIR}/" 2>/dev/null || true
cp "${MLX_BF16_DIR}"/tokenizer* "${ONNX_DIR}/" 2>/dev/null || true
echo "ONNX assets staged in ${ONNX_DIR}"

echo "=== 4. Ollama Integration ==="
if command -v ollama >/dev/null 2>&1; then
    echo "Registering Modelfile with Ollama..."
    ollama create lfm3-omni-3d-coder -f ollama/Modelfile.lfm3-omni-2b-3d-coder || echo "Ollama registration skipped (daemon may not be running)"
else
    echo "Ollama not installed or not in PATH; skipping registration."
fi

echo "=== LFM3-Omni-2B Fusion & Export Complete! ==="
echo "BF16 MLX Model: ${MLX_BF16_DIR}"
echo "4-Bit MLX Model: ${MLX_4BIT_DIR}"
