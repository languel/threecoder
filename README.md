# threecoder

Local 3D scene and `.glb` game-asset generation models, benchmarks, and Apple Silicon fine-tuning workspace—configured and benchmarked for **Apple M5 (128 GB Unified Memory)**.

> **Next session?** Start by reading [`HANDOFF.md`](./HANDOFF.md) for the full Session 1 build log, environment details, architectural trade-offs, and the step-by-step Session 2 local training plan (`LFM3-1.7B` & `Qwen3.5` via `mlx-lm`).

---

## Overview

This project runs and evaluates [CCORE's](https://huggingface.co/ccore) specialized 4B 3D-coding models locally across **Ollama (`llamacpp` GGUF)**, **Ollama (`mlx` native Safetensors)**, and **standalone `mlx-lm`**:

1. **[`ccore/Qwen3.5-4B-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-3D-Coder)** — Generates interactive, self-contained Three.js (`0.160.0` ES module) HTML scenes with procedural geometry, materials, animation, and custom camera controls.
2. **[`ccore/Qwen3.5-4B-GLB-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder)** — Generates low-poly 3D game assets in a standalone Three.js HTML preview complete with a working **Export GLB** button for Godot, Blender, or Unity.

---

## Local Models Installed on This Machine

### 1. Ollama Models (`ollama/`)

All four Modelfiles are tuned for 128 GB unified memory:
- **`num_ctx 98304` (96K tokens):** CCORE's training examples reach **72K tokens** (long `<think>` chain-of-thought traces + full HTML/JS). Default 4K–8K context windows truncate output mid-file.
- **`num_predict -1`:** Unlimited generation cap so long scenes finish cleanly.
- **Official System Prompts & Sampling:** Exact `temperature`, `top_k`, `top_p`, `min_p`, and `repeat_penalty` values from CCORE's evaluation configs.

| Ollama Model Tag | Modelfile | Runner & Precision | Disk / RAM (at 96K ctx) | Short-Ctx Speed | ~20K-Ctx Avg Speed | Vision (`mmproj`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`ccore-glb-coder`** | [`ollama/Modelfile.ccore-glb-coder`](./ollama/Modelfile.ccore-glb-coder) | `llamacpp` · `Q8_0` | 5.2 GB / 9.0 GB | **90.8 tok/s** | 26.3 tok/s | Yes (`BF16`) |
| **`ccore-3d-coder`** | [`ollama/Modelfile.ccore-3d-coder`](./ollama/Modelfile.ccore-3d-coder) | `llamacpp` · `Q8_0` | 5.2 GB / 9.0 GB | **~91 tok/s** | 24.4 tok/s | Yes (`BF16`) |
| **`ccore-glb-coder-mlx`** | [`ollama/Modelfile.ccore-glb-coder-mlx`](./ollama/Modelfile.ccore-glb-coder-mlx) | `mlx` · **`BF16`** | 8.4 GB / 8.5 GB | **48.7 tok/s** | ~20 tok/s | Text only |
| **`ccore-3d-coder-mlx`** | [`ollama/Modelfile.ccore-3d-coder-mlx`](./ollama/Modelfile.ccore-3d-coder-mlx) | `mlx` · **`BF16`** | 8.4 GB / 8.5 GB | **46.3 tok/s** | 21.0 tok/s | Text only |

#### Quick Usage (Ollama)

```bash
# GLB game asset (prompts work directly with "Create ..."):
ollama run ccore-glb-coder-mlx "Create a tiny fox companion wearing a red scarf."

# Interactive Three.js scene (prefix with self-contained HTML instruction to match training data):
ollama run ccore-3d-coder-mlx "Build a self-contained HTML scene with Three.js from cdnjs featuring a canal lock and weir complex with a central island."

# Rebuild any Ollama tag from its Modelfile:
ollama create ccore-3d-coder-mlx -f ollama/Modelfile.ccore-3d-coder-mlx
```

### 2. Standalone Unquantized MLX Checkpoints (`~/mlx-models/`)

Converted directly from CCORE's upstream Safetensors into `bfloat16` MLX format (`7.9 GB` each, zero quantization loss) via `mlx_lm.convert`:
- `/Users/liuboto/mlx-models/Qwen3.5-4B-3D-Coder-bf16`
- `/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16`

*(These directories also serve directly as unquantized base checkpoints if you fine-tune further with `mlx_lm.lora`.)*

```bash
# Launch an OpenAI-compatible local MLX server on port 8080:
mlx_lm.server --model ~/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16 --port 8080
```

---

## Benchmark Report: Baba Yaga Snowglobe (4-Way Comparison)

**Prompt tested:**
> *"a small wintry scene inside a snowglobe. baba yaga hut in a small clearing. eastern foklore, a fox, a raven and an oul and chimney smoke. dusk, starry sky, fluffy snow."*

To compare all 4 scenes interactively in Single, Side-by-Side (`2×`), or Quad (`2×2`) view:

```bash
open benchmarks/snowglobe/compare.html
```

### Results Summary (Apple M5 · 128 GB RAM)

Full machine-readable stats live in [`benchmarks/snowglobe/benchmark_results.json`](./benchmarks/snowglobe/benchmark_results.json).

| Output File | Model & Engine | Total Tokens | Avg Speed (at ~20K ctx) | Wall Time | HTML Lines | Zero-Shot Valid JS (`node --check`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| [`snowglobe_3d_ollama_mlx.html`](./benchmarks/snowglobe/snowglobe_3d_ollama_mlx.html) | `ccore-3d-coder-mlx` (`BF16` · Ollama `mlx`) | 16,934 | 21.0 tok/s | **13m 30s** | 1,351 | **Yes — 0 errors** |
| [`snowglobe_3d_ollama_q8.html`](./benchmarks/snowglobe/snowglobe_3d_ollama_q8.html) | `ccore-3d-coder` (`Q8_0` · Ollama `llamacpp`) | 19,697 | 24.4 tok/s | **13m 29s** | 1,468 | **No** — 2 duplicate `const`s (`ravenBeak`, `snowMaterial`)* |
| [`snowglobe_glb_ollama_q8.html`](./benchmarks/snowglobe/snowglobe_glb_ollama_q8.html) | `ccore-glb-coder` (`Q8_0` · Ollama `llamacpp`) | 21,601 | 26.3 tok/s | **13m 47s** | 1,325 | **Yes — 0 errors** (+ Export GLB) |
| [`snowglobe_glb_mlx_lm.html`](./benchmarks/snowglobe/snowglobe_glb_mlx_lm.html) | `GLB-3D-Coder` (`BF16` · standalone `mlx-lm`) | 23,282 | 18.2 tok/s | **21m 23s** | 869 | **Yes — 0 errors** (+ Export GLB, 9.42 GB peak RAM) |

*\*Note: [`snowglobe_3d_ollama_q8.html`](./benchmarks/snowglobe/snowglobe_3d_ollama_q8.html) is preserved unedited for benchmark integrity; [`snowglobe_3d_ollama_q8_fixed.html`](./benchmarks/snowglobe/snowglobe_3d_ollama_q8_fixed.html) renames the two colliding material constants (`ravenBeakMat`, `snowfallMaterial`) so the rendered scene can also be inspected in `compare.html`.*

### Key Empirical Findings

1. **`BF16` MLX (`ccore-3d-coder-mlx`) is the sweet spot for `3D-Coder`:** Although 8-bit `Q8_0` decodes ~1.9× faster on short 128-token bursts (`90.8 tok/s` vs `46.3 tok/s` due to half the memory-bandwidth bytes per token), across a full **17K–20K token generation** both finished in **13.5 minutes**, and unquantized `BF16` avoided the exact duplicate-variable `SyntaxError` bugs (`ravenBeak`, `snowMaterial`) that `Q8_0` hit.
2. **`GLB-3D-Coder` (Nov 5 checkpoint) is cleaner out of the box:** Both `Q8_0` (`ccore-glb-coder`) and `BF16` MLX (`Qwen3.5-4B-GLB-3D-Coder-bf16`) produced 100% syntactically valid JavaScript zero-shot, complete with working GLB exporters.
3. **Vision (`mmproj`) bonus on `Q8_0` tags:** Ollama `0.40.14` automatically pulled and linked CCORE's 675 MB `mmproj-Qwen3.5-4B-BF16.gguf` projector onto `ccore-3d-coder` and `ccore-glb-coder`, enabling image-guided 3D prompts when needed.

---

## Repository Structure

```text
threecoder/
├── README.md                                        # Project overview, usage, and benchmark report
├── HANDOFF.md                                       # Session 1 build log & Session 2 training guide
├── ollama/
│   ├── Modelfile.ccore-3d-coder                     # Ollama Q8_0 GGUF (Three.js scenes + vision)
│   ├── Modelfile.ccore-3d-coder-mlx                 # Ollama BF16 MLX  (Three.js scenes, unquantized)
│   ├── Modelfile.ccore-glb-coder                    # Ollama Q8_0 GGUF (GLB assets + vision)
│   └── Modelfile.ccore-glb-coder-mlx                # Ollama BF16 MLX  (GLB assets, unquantized)
└── benchmarks/
    └── snowglobe/
        ├── compare.html                             # Single / 2x Split / 2x2 Quad 3D viewer
        ├── run_snowglobe_bench.py                   # Sequential 4-way benchmark & HTML extractor
        ├── benchmark_results.json                   # Token counts, tok/s, wall times, JS syntax checks
        ├── snowglobe_3d_ollama_mlx.html             # 3D-Coder BF16 MLX output (zero-shot valid)
        ├── snowglobe_3d_ollama_q8.html              # 3D-Coder Q8_0 raw output (2 duplicate consts)
        ├── snowglobe_3d_ollama_q8_fixed.html        # 3D-Coder Q8_0 with 2-line const rename fix
        ├── snowglobe_glb_ollama_q8.html             # GLB-Coder Q8_0 output (zero-shot valid)
        └── snowglobe_glb_mlx_lm.html                # GLB-Coder standalone mlx-lm BF16 output (zero-shot valid)
```
