# threecoder

Local 3D coding models (`CCORE Qwen3.5-4B-3D-Coder` & `GLB-3D-Coder`), benchmarks, and MLX fine-tuning experiments optimized for **Apple Silicon M5 (128 GB Unified Memory)**.

## Local Model Setup

### 1. Ollama Models (`ollama/`)

All four Modelfiles configure a **96K context window (`num_ctx 98304`)**, **unlimited output (`num_predict -1`)**, thinking enabled, and CCORE's official system prompts & sampling hyperparameters:

| Model Tag | Modelfile | Runner & Precision | Notes |
| :--- | :--- | :--- | :--- |
| `ccore-3d-coder` | `ollama/Modelfile.ccore-3d-coder` | `llamacpp` (`Q8_0`) | Interactive Three.js HTML scenes + BF16 vision (`mmproj`) |
| `ccore-glb-coder` | `ollama/Modelfile.ccore-glb-coder` | `llamacpp` (`Q8_0`) | Low-poly 3D assets with **Export GLB** button + BF16 vision |
| `ccore-3d-coder-mlx` | `ollama/Modelfile.ccore-3d-coder-mlx` | `mlx` (`BF16`) | Unquantized `bfloat16` Three.js scene coder via Ollama's MLX runner |
| `ccore-glb-coder-mlx` | `ollama/Modelfile.ccore-glb-coder-mlx` | `mlx` (`BF16`) | Unquantized `bfloat16` GLB asset coder via Ollama's MLX runner |

Rebuild any Ollama model with:
```bash
ollama create ccore-glb-coder -f ollama/Modelfile.ccore-glb-coder
```

### 2. Standalone MLX Checkpoints (`~/mlx-models/`)

Unquantized `bfloat16` MLX weights (7.9 GB each) converted via `mlx_lm.convert`:
- `~/mlx-models/Qwen3.5-4B-3D-Coder-bf16`
- `~/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16`

Serve via OpenAI-compatible API:
```bash
mlx_lm.server --model ~/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16 --port 8080
```

## Benchmarks

- `benchmarks/snowglobe/` — 4-way comparison (`Ollama Q8_0`, `Ollama MLX BF16`, and `Standalone mlx-lm BF16` across `3D-Coder` and `GLB-Coder`) on the Baba Yaga wintry snowglobe prompt.
  - Open `benchmarks/snowglobe/compare.html` in any browser to inspect all generated 3D scenes side-by-side or fullscreen.
