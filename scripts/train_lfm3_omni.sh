#!/usr/bin/env bash
set -euo pipefail

# LFM3-Omni-2B Three.js Scene Fine-Tuning Runner
# Supports both full BF16 LoRA and 4-Bit LoRA for lower-spec / lower-VRAM systems.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-bf16}"

if [ "${MODE}" = "4bit" ] || [ "${MODE}" = "--4bit" ]; then
    CONFIG="configs/lfm3_omni_scene_lora_4bit.yaml"
    echo "=== Running LFM3-Omni-2B Scene LoRA Training [4-Bit Low-VRAM Mode] ==="
else
    CONFIG="configs/lfm3_omni_scene_lora.yaml"
    echo "=== Running LFM3-Omni-2B Scene LoRA Training [BF16 High-Precision Mode] ==="
fi

# 1. Verify Dataset
if [ ! -f "data/lfm3_scene/train.jsonl" ]; then
    echo "Dataset not found in data/lfm3_scene/. Preparing dataset from ccore/ThreeJS-3D-Scenes..."
    python3 scripts/prepare_lfm3_scene_dataset.py \
        --model "LiquidAI/LFM3-Omni-2B-MLX-bf16" \
        --out-dir "data/lfm3_scene" \
        --max-seq-length 16384
fi

# 2. Run LoRA training via mlx_lm
echo "Starting mlx_lm.lora fine-tuning with ${CONFIG}..."
python3 -m mlx_lm.lora --config "${CONFIG}"

echo "=== Omni Training Complete! Adapters saved according to ${CONFIG} ==="
