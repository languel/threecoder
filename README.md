# threecoder

A complete toolkit for training, exporting, and benchmarking small local LLMs (`LFM3-1.7B`, `LFM3-Omni-2B`, `Qwen3.5-4B Coder`) to generate interactive, self-contained Three.js 3D HTML scenes directly on WebGPU, Apple Silicon (MLX), and Ollama.

> **Handoff & Rerunning Training from Scratch?**  
> Check [`HANDOFF.md`](./HANDOFF.md) for full reproduction commands, dataset preparation, LoRA training configs, WebGPU ONNX / MLX export scripts, and benchmark instructions.

---

## Highlights

- **Pure Three.js Scene Generation**: Zero GLB boilerplate. Models generate complete, standalone HTML scenes using Three.js ES modules via CDN (`three.module.min.js`), complete with lighting, procedural geometries, materials, animation loops, and mouse/touch orbit navigation.
- **WebGPU-First Architecture**: Exported to Transformers.js v4-compatible `q4f16` ONNX graphs for client-side WebGPU execution directly inside the browser.
- **High-Throughput Local MLX & Ollama**: Full BF16 and 4-bit MLX fine-tunes providing up to 620 tok/s batched generation on Apple Silicon, with one-click Ollama Modelfiles.
- **Rebuilt Eval-10 Benchmark**: 10 progressive 3D test cases (from beveled studio cubes to Platonic solid rings, particle systems, characters, and animated snowglobes) enforcing automatic camera framing and minimal UI.
- **Live 3D Arena (`compare.html`)**: Interactive side-by-side comparison matrix rendering live WebGL iframes for `LFM3 1.7B`, `LFM3-Omni 2B`, and `Qwen3.5 4B` simultaneously with zero broken static snapshots.

---

## Supported Models

| Model | WebGPU ONNX Size | Architecture | Best For |
| :--- | :---: | :--- | :--- |
| **`LFM3-1.7B-3D-Coder`** | 1.03 GB | Hybrid Conv + GQA | Ultra-fast client generation (~98 tok/s WebGPU) |
| **`LFM3-Omni-2B-3D-Coder`** | 1.32 GB | Liquid Omni Multimodal | Omni multimodal conditioning & native 3D representations |
| **`Qwen3.5-4B-3D-Coder`** | 2.81 GB | Full Dense Transformer | High-precision zero-shot syntax |

---

## Quickstart

### 1. Training LFM3-Omni-2B from Scratch
```bash
# 1. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare SFT dataset from ccore/ThreeJS-3D-Scenes
python3 scripts/prepare_lfm3_scene_dataset.py \
  --model "LiquidAI/LFM3-Omni-2B-MLX-bf16" \
  --dataset "ccore/ThreeJS-3D-Scenes" \
  --out-dir "data/lfm3_scene"

# 3. Run LoRA training (BF16 or 4-bit)
bash scripts/train_lfm3_omni.sh bf16

# 4. Fuse weights and export to MLX / ONNX / Ollama
bash scripts/fuse_and_export_omni.sh
```

### 2. Running the Eval-10 Benchmark & Live Arena
```bash
# Run benchmark across models
python3 benchmarks/eval10/run_eval10.py

# Launch local comparison server
python3 playground/server.py --port 8765
```
Open **`http://localhost:8765/benchmarks/eval10/compare.html`** in any modern browser to test scenes with interactive OrbitControls and camera auto-framing.

---

## Repository Structure

```
├── configs/                  # LoRA training configuration files (BF16 & 4-bit)
│   ├── lfm3_omni_scene_lora.yaml
│   └── lfm3_omni_scene_lora_4bit.yaml
├── scripts/                  # Training, fusion, ONNX export, and testing tools
│   ├── prepare_lfm3_scene_dataset.py
│   ├── train_lfm3_omni.sh
│   ├── fuse_and_export_omni.sh
│   ├── export_lfm3_onnx_q4f16.py
│   └── export_qwen35_onnx_q4f16.py
├── benchmarks/eval10/        # 10-tier benchmark suite and comparison arena
│   ├── prompts.json          # Rebuilt prompts mandating framing & orbit navigation
│   ├── run_eval10.py         # Batched MLX benchmark evaluator
│   ├── compare.html          # Live 3D side-by-side WebGL comparison arena
│   └── outputs/              # Model-generated HTML scene outputs
├── ollama/                   # Ollama Modelfiles for local deployment
│   ├── Modelfile.lfm3-omni-2b-3d-coder
│   └── Modelfile.lfm3-3d-coder-bf16
├── playground/               # WebGPU interactive browser generation studio
└── HANDOFF.md                # Comprehensive reproduction and handoff playbook
```

---

## License

Apache 2.0 (code). Model weights subject to their respective upstream licenses: [LiquidAI LFM License](https://huggingface.co/LiquidAI) and [Qwen License](https://huggingface.co/Qwen).
