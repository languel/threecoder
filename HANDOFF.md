# Session Handoff & Next-Session Playbook

**Date:** November 6, 2026  
**Workspace:** `/Users/liuboto/dev/threecoder`  
**Hardware:** Apple Silicon M5 · 128 GB Unified Memory · macOS

---

## 1. What Was Accomplished in Session 1

### A. Evaluated & Configured Local Inference Options (Ollama + MLX)
1. **Inspected upstream CCORE repositories on Hugging Face:**
   - Models: [`ccore/Qwen3.5-4B-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-3D-Coder) and [`ccore/Qwen3.5-4B-GLB-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder) (both fine-tuned from `Qwen/Qwen3.5-4B`, `max_position_embeddings = 262144`).
   - Datasets: [`ccore/ThreeJS-3D-Scenes`](https://huggingface.co/datasets/ccore/ThreeJS-3D-Scenes) (3,933 train / 213 val, MIT) and [`ccore/ThreeJS-3D-GLB-Assets`](https://huggingface.co/datasets/ccore/ThreeJS-3D-GLB-Assets) (3,501 train / 417 val, MIT).
2. **Built Option 1 — Optimized Ollama `Q8_0` Models (`llamacpp` runner):**
   - Pulled `Q8_0` GGUFs (skipping `Q4_K_M` since 128 GB RAM has massive headroom and 4-bit quantization degrades 3D spatial math and JS variable tracking).
   - Ollama `0.40.14` automatically pulled and wired CCORE's `mmproj-Qwen3.5-4B-BF16.gguf` (675 MB) projector, giving both `ccore-3d-coder` and `ccore-glb-coder` native **image-to-3D vision** capabilities alongside `<think>` parsing.
   - Configured custom Modelfiles (`ollama/Modelfile.ccore-3d-coder` and `ollama/Modelfile.ccore-glb-coder`) with `num_ctx 98304` (96K), `num_predict -1`, and CCORE's exact system prompts and sampling parameters.
3. **Built Option 3 — Unquantized `BF16` MLX Models (Standalone `mlx-lm` + Ollama `mlx` runner):**
   - Converted CCORE's raw Safetensors checkpoints (`FP16` for `3D-Coder`; `FP32` text + `BF16` vision for `GLB-3D-Coder`) to clean `bfloat16` MLX models (`7.9 GB` each) at:
     - `/Users/liuboto/mlx-models/Qwen3.5-4B-3D-Coder-bf16`
     - `/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16`
   - Imported both `BF16` MLX directories into Ollama `0.40.14`'s native **`mlx` runner** as `ccore-3d-coder-mlx` and `ccore-glb-coder-mlx` (`ollama/Modelfile.ccore-*-mlx`).
4. **Disk & Tag Cleanup:**
   - Removed the duplicate `hf.co/ccore/...` tags from `ollama list` (Ollama's SHA-256 blob store kept all referenced weight/projector blobs intact).
   - Deleted the **24.5 GB** of redundant raw Hugging Face download cache (`~/.cache/huggingface/hub/models--ccore--*`), keeping only the active Ollama models and `~/mlx-models/`.
5. **Ran 4-Way Snowglobe Benchmark & Initialized Repo:**
   - Created `/Users/liuboto/dev/threecoder` (initially outside the user's unrelated `mediatheque` repo) with `compare.html`, `run_snowglobe_bench.py`, and `benchmark_results.json`.
   - Validated all 4 generated HTML files with `node --check` (see findings below).

---

## 2. Verified Local Environment & Tooling Paths

When picking up work in the next session, these tools are already installed and verified on this Mac:

| Tool / Asset | Version / Location | Notes |
| :--- | :--- | :--- |
| **Ollama** | `0.40.14` (`/usr/local/bin/ollama`) | Supports both `llamacpp` (GGUF) and native `mlx` (Safetensors) runners, including `qwen3_5` (`renderer: qwen3.5`, `parser: qwen3.5`). Also already has `lfm3:1.7b`, `lfm3-omni`, `qwen3-coder:30b`, `qwen3.8:27b-mlx` installed. |
| **MLX / `mlx-lm`** | `mlx 0.32.1`, `mlx-lm 0.31.3` | Installed via Homebrew formula. CLI binaries: `/opt/homebrew/bin/mlx_lm.*` (`mlx_lm.lora`, `mlx_lm.convert`, `mlx_lm.fuse`, `mlx_lm.server`). Python interpreter with `mlx` & `mlx_lm`: `/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python` *(note: `/opt/homebrew/bin/python3` does NOT have `mlx` in global site-packages—use the `mlx-lm` libexec Python or create a local `.venv` via `uv`)*. |
| **`uv`** | `/Users/liuboto/.local/bin/uv` | Recommended for creating an isolated `.venv` in `/Users/liuboto/dev/threecoder` when writing dataset-prep and training scripts. |
| **Standalone MLX Weights** | `/Users/liuboto/mlx-models/` | `Qwen3.5-4B-3D-Coder-bf16` (7.9 GB) & `Qwen3.5-4B-GLB-3D-Coder-bf16` (7.9 GB). |

---

## 3. Important Technical Gotchas Discovered

1. **Omit `RENDERER` in Ollama `qwen3_5` MLX Modelfiles:**
   - When `FROM` points at a `Qwen3.5` MLX directory without overriding `RENDERER`, Ollama `0.40.14` auto-detects `renderer: qwen3.5` + `parser: qwen3.5` with `thinking` (`true`/`false`, default `true`), matching Qwen3.5's native `<think>` template.
2. **Training-Distribution Prompt Phrasing Matters (`3D-Coder` vs `GLB-Coder`):**
   - `ccore/ThreeJS-3D-GLB-Assets` puts `"in one complete standalone HTML file"` inside the **system prompt**, so short user prompts like `"Create a coral-armored lighthouse keeper."` always emit standalone HTML.
   - `ccore/ThreeJS-3D-Scenes` puts `"Use only the Three.js core ES module..."` in the system prompt, while its **user prompts** consistently begin with `"Build a self-contained HTML scene with Three.js from cdnjs featuring ..."` or `"Create a single self-contained HTML file using Three.js from cdnjs that shows ..."`. Always include that prefix when prompting `ccore-3d-coder` / `ccore-3d-coder-mlx`.
3. **Why Long Generations Slow Down (~90 tok/s -> ~20–26 tok/s) & Why `BF16` MLX Won on `3D-Coder`:**
   - CCORE's models emit 4K–14K tokens of `<think>` reasoning *before* writing 800–1,500 lines of HTML/JS (17K–23K total tokens per generation).
   - Over ~17K–20K tokens, `ccore-3d-coder-mlx` (`BF16`) finished in the **exact same wall time (`810s`)** as `ccore-3d-coder` (`Q8_0`, `809s`) while producing **zero JavaScript syntax errors** (whereas `Q8_0` collided `const ravenBeak` and `const snowMaterial`).
4. **Clarification on Training Libraries (`mlx-learn` vs `mlx-lm` vs `Unsloth`):**
   - [`tabulai/mlx-learn`](https://github.com/tabulai/mlx-learn) is a **classical ML (`scikit-learn`) accelerator** (`KNN`, `SVC`, `LogisticRegression`), **not** an LLM deep-learning trainer.
   - Standard Python `unsloth` uses CUDA + Triton kernels (Linux/NVIDIA). On Apple Silicon, local LLM LoRA/DoRA training uses **`mlx-lm` (`mlx_lm.lora`)**, **Unsloth Studio/Desktop (Mac)**, or **`mlx-tune`** (Unsloth-compatible Python wrapper over MLX).

---

## 4. Instructions & Roadmap for Session 2 (Local Fine-Tuning)

When opening `/Users/liuboto/dev/threecoder` in the next session, here are the concrete next steps if you want to replicate CCORE's training pipeline locally on your M5 (128 GB):

### Step 1: Set Up a Clean Python Environment in `threecoder`
```bash
cd ~/dev/threecoder
uv venv --python 3.12
uv pip install mlx mlx-lm datasets huggingface_hub pyarrow
```

### Step 2: Choose Which Base-Model Experiment to Run First

| Option | Base Model | Architectural Hypothesis | Expected Behavior |
| :--- | :--- | :--- | :--- |
| **Experiment A (Speed / Agentic Loop)** | **`LiquidAI/LFM3-1.7B`** *(or `LFM3-1.7B-Base`)* | Tri-hybrid architecture (gated short convolutions + sliding-window + global attention, 1.73B params). | **2×–3× faster generation & tiny KV cache** at 20K+ tokens. 1.7B params is tight for complex one-shot 3D math, so test **both** with `<think>` traces and **distilled direct `prompt -> HTML` (no `<think>`)** for ultra-fast interactive iteration or agentic self-repair loops (`generate -> node --check / headless screenshot -> fix`). |
| **Experiment B (128 GB Quality Upgrade)** | **`Qwen/Qwen3.5-9B`** *(or `14B`)* | CCORE trained on 4B so 8 GB GPUs could run it; your M5 has 128 GB RAM. | Fine-tuning a 9B–14B backbone on CCORE's exact MIT dataset should substantially improve 3D spatial proportions and eliminate variable-collision bugs while still running fast in Ollama/MLX on 128 GB. |

### Step 3: Format CCORE's Dataset for `mlx_lm.lora`
Both [`ccore/ThreeJS-3D-GLB-Assets`](https://huggingface.co/datasets/ccore/ThreeJS-3D-GLB-Assets) and [`ccore/ThreeJS-3D-Scenes`](https://huggingface.co/datasets/ccore/ThreeJS-3D-Scenes) store Parquet rows with:
- `messages`: list of `{ "role", "content", "reasoning_content" }`.

When converting to `data/{train,valid}.jsonl` for `mlx_lm.lora`:
1. **Check sequence token lengths first!** Filter or bucket examples (e.g., `<= 16384` or `<= 24576` tokens for initial runs) so outlier 72K-token rows don't dominate step time.
2. **Two dataset variants worth writing side-by-side:**
   - **Reasoning variant:** Format assistant `content` as `<think>\n{reasoning_content}\n</think>\n\n{content}` (matches Qwen3.5's native template).
   - **Direct / Fast-Coder variant (especially great for `LFM3-1.7B`):** Strip `reasoning_content` (or summarize it) so the model outputs the `<!DOCTYPE html>...</html>` immediately—cutting generation time from **13 minutes down to ~45–90 seconds**!

### Step 4: Run a 10-Minute Smoke Test Before Any Overnight Run
With 128 GB unified memory, run **16-bit (`bf16`) LoRA** (no QLoRA quantization needed) with `--grad-checkpoint`:
1. Verify `mlx_lm` supports `Lfm3Config` (`python -c "from mlx_lm import load; load('LiquidAI/LFM3-1.7B')"`). Note that LFM3 is a tri-hybrid architecture (gated conv + attention), so configure LoRA to target all linear projection layers (`all-linear` / `keys` in `lora_parameters`), not just attention `q_proj`/`v_proj`.
2. Run **100–200 iterations** on a small subset (`--iters 150 --batch-size 1 --grad-accumulation-steps 4 --max-seq-length 16384 --grad-checkpoint`) to verify:
   - Peak unified memory usage (expected ~20–40 GB out of 128 GB),
   - Seconds/step to calculate exact wall-clock time for 1 full epoch (~3,500 examples),
   - Validation loss curve & a sample snowglobe generation.

### Step 5: Fuse & Import Back into Ollama
Once a LoRA adapter looks good:
```bash
# 1. Fuse LoRA adapter into a standalone BF16 MLX/Safetensors model:
mlx_lm.fuse --model LiquidAI/LFM3-1.7B --adapter-path adapters/lfm3-glb --save-path ~/mlx-models/LFM3-1.7B-GLB-bf16

# 2. Import directly into Ollama (either via Ollama's native MLX Safetensors importer or llama.cpp convert_hf_to_gguf.py):
ollama create lfm3-glb-coder -f ollama/Modelfile.lfm3-glb-coder
```

### Step 6: Optional Bonus — Automated Validation Harness
Notice in our Snowglobe benchmark how simple `node --check` caught `Q8_0`'s duplicate `const` declarations in <50 ms! In Session 2, we can add a lightweight **post-generation validator / self-repair loop** (`scripts/generate_and_validate.py`) that:
1. Streams HTML from Ollama/MLX,
2. Runs `node --check` + headless browser console/WebGL check + verifies `GLTFExporter` output, and
3. Automatically sends any syntax/runtime error back to the local model for a fast patch if needed.
