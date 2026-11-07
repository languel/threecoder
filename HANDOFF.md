# ThreeCoder: System Handoff & Training Playbook

**Workspace:** `threecoder`  
**GitHub Repository:** `https://github.com/languel/threecoder`  
**Target Models:** `LiquidAI/LFM3-1.7B`, `LiquidAI/LFM3-Omni-2B`, `Qwen/Qwen2.5-Coder-7B` / `Qwen3.5-4B`  
**Primary Output Formats:**
1. **WebGPU ONNX (`q4f16`)** — Direct browser execution via Transformers.js v4 & WebGPU.
2. **MLX (BF16 & 4-bit)** — High-throughput Apple Silicon and Ollama-compatible inference.
*(GGUF is lowest priority and optional for resource-constrained systems).*

---

## 1. Project Mission & Architecture

ThreeCoder focuses exclusively on **interactive, standalone Three.js HTML scene generation**:
- **Zero GLB boilerplate**: No export buttons, headless browser capture overhead, or complex exporter scripts.
- **Pure Web Standards**: Complete, self-contained HTML files importing standard Three.js via ES module CDN (`https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js`).
- **Framing & Controls Mandate**: Every scene must position its camera so the entire 3D subject is automatically framed and visible on initial load, with minimal UI and basic orbit navigation.
- **WebGPU-First Deployment**: Models are optimized to run directly inside client browsers using WebGPU ONNX runtime, or locally via MLX / Ollama.

---

## 2. Quickstart on Another System (From Scratch)

Follow these steps to clone, configure, train, and export ThreeCoder on a clean machine.

### Step 1: Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/languel/threecoder.git
cd threecoder

# Create Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> If running on Apple Silicon, ensure `mlx` and `mlx-lm` are installed (`pip install mlx mlx-lm`). If running on Linux/CUDA, you can use Hugging Face `peft` / `transformers` or Ollama runners.

---

### Step 2: Prepare SFT Training Dataset

The training pipeline fine-tunes on `ccore/ThreeJS-3D-Scenes` from Hugging Face. The data preparation script preserves split boundaries, bounds tokens to `<= 16,384` to prevent HTML tag truncation, and formats prompts into Direct-HTML mode:

```bash
# Prepare dataset for LFM3-Omni-2B (or LFM3-1.7B)
python3 scripts/prepare_lfm3_scene_dataset.py \
  --model "LiquidAI/LFM3-Omni-2B-MLX-bf16" \
  --dataset "ccore/ThreeJS-3D-Scenes" \
  --out-dir "data/lfm3_scene" \
  --max-seq-length 16384
```

This generates:
- `data/lfm3_scene/train.jsonl`
- `data/lfm3_scene/valid.jsonl`
- `data/lfm3_scene/stats.json`

---

### Step 3: Run Fine-Tuning (LoRA)

Two configuration presets are available depending on target hardware:

#### Option A: High-Performance BF16 LoRA (Recommended for >= 16 GB VRAM)
```bash
bash scripts/train_lfm3_omni.sh bf16
```
*Config:* `configs/lfm3_omni_scene_lora.yaml` (Rank 16, Alpha 32, max_seq_length 16,384, cosine warmup, grad checkpointing).

#### Option B: Low-Spec / Low-VRAM 4-Bit LoRA (For <= 8 GB VRAM)
```bash
bash scripts/train_lfm3_omni.sh 4bit
```
*Config:* `configs/lfm3_omni_scene_lora_4bit.yaml` (Base model 4-bit quantized, max_seq_length 8,192, memory threshold 2 GB).

LoRA adapters are saved to `adapters/lfm3-omni-2b-scene/`.

---

### Step 4: Fuse Weights & Export (MLX, WebGPU ONNX, Ollama)

Run the automated fusion and export pipeline:

```bash
bash scripts/fuse_and_export_omni.sh
```

What this script produces:
1. **Fused MLX BF16**: Fuses the LoRA adapter with base weights to `~/mlx-models/LFM3-Omni-2B-3D-Coder-bf16`.
2. **4-Bit MLX Model**: Quantizes the fused model to 4-bit (`~/mlx-models/LFM3-Omni-2B-3D-Coder-4bit`) for fast local inference (>600 tok/s batched).
3. **WebGPU ONNX Staging**: Prepares ONNX model assets for browser deployment under `~/onnx-models/LFM3-Omni-2B-3D-Coder-ONNX-q4f16`.
4. **Ollama Registration**: Automatically registers the Ollama custom card:
   ```bash
   ollama create lfm3-omni-3d-coder -f ollama/Modelfile.lfm3-omni-2b-3d-coder
   ```

---

## 3. Benchmark Suite & Live 3D Arena

The evaluation benchmark (`benchmarks/eval10/`) evaluates models across 10 progressive tiers from basic geometric primitives to complex animated multi-subject scenes.

### Prompt Design Philosophy
Prompts describe **WHAT** to build and enforce camera auto-framing, minimal UI, and orbit controls—**WITHOUT** micromanaging internal code implementation details:

> *"Build a complete standalone Three.js HTML scene of [subject]. The camera must be positioned so the whole scene is automatically framed and visible on initial load. Enable basic mouse/touch orbit navigation with minimal UI."*

### The 10 Evaluated Scenarios (`benchmarks/eval10/prompts.json`)
1. `P01_default_cube`: Default Studio Cube (three-point lighting).
2. `P02_gold_sphere`: Brushed Gold Sphere on architectural stone pedestal.
3. `P03_cube_sphere_tetrahedron`: Geometric Primitive Triad on illuminated dark plinth.
4. `P04_platonic_solids`: Five Platonic Solids in an orbital ring.
5. `P05_pine_tree`: Low-Poly Pine Tree with tiered snow-dusted foliage branches.
6. `P06_particle_system`: Firefly Lantern & glowing floating particle swarm.
7. `P07_fox_character`: Stylized Low-Poly Fox character.
8. `P08_owl_character`: Animated Low-Poly Owl perched on branch with idle head-tilt.
9. `P09_human_character`: Stylized Human Explorer adventurer.
10. `P10_snowglobe`: Winter Cabin Snowglobe with drifting snow particles inside a glass dome.

### Running the Evaluation
To evaluate all models in batch mode (~5-8 minutes total):
```bash
python3 benchmarks/eval10/run_eval10.py
```

To evaluate a specific model:
```bash
python3 benchmarks/eval10/run_eval10.py --models lfm3_omni_2b_3d_bf16
```

### Viewing the Live Side-by-Side 3D Arena
Launch the local playground web server:
```bash
python3 playground/server.py --port 8765
```
Open **`http://localhost:8765/benchmarks/eval10/compare.html`** in your browser.

**Features:**
- **Live 3D Side-by-Side Arena**: Runs `LFM3-1.7B`, `LFM3-Omni-2B`, and `Qwen3.5-4B` in synchronous live iframes with real-time OrbitControls and bounding-box auto-framing.
- **Interactive Prompt Selector**: 10-prompt navigation pills with keyboard shortcuts (`ArrowLeft` / `ArrowRight`).
- **Full Prompt-by-Model Matrix**: On-demand WebGL embedding for all 10 prompts across models.
- **Zero Broken Thumbnails**: No static snapshot blank screens—renders live WebGL canvases directly.

---

## 4. Key Scripts & Directories

| Path | Purpose |
| :--- | :--- |
| `configs/lfm3_omni_scene_lora.yaml` | BF16 LoRA training configuration for LFM3-Omni-2B. |
| `configs/lfm3_omni_scene_lora_4bit.yaml` | 4-bit LoRA training configuration for low-VRAM systems. |
| `scripts/prepare_lfm3_scene_dataset.py` | SFT dataset generation with token limits and split integrity. |
| `scripts/train_lfm3_omni.sh` | One-command fine-tuning runner (supports `bf16` or `4bit`). |
| `scripts/fuse_and_export_omni.sh` | Fuses adapter, creates 4-bit MLX, stages ONNX, and registers Ollama model. |
| `ollama/Modelfile.lfm3-omni-2b-3d-coder` | Ollama model card for LFM3-Omni-2B Three.js Coder. |
| `benchmarks/eval10/prompts.json` | 10 rebuilt framing-enforced benchmark prompts. |
| `benchmarks/eval10/run_eval10.py` | Fast batched benchmark execution engine. |
| `benchmarks/eval10/compare.html` | Live 3D comparison arena and model leaderboard. |
| `playground/` | WebGPU interactive generation studio with real-time auto-healing. |

---

## 5. Notes & Best Practices

1. **Auto-Framing In Three.js**:
   All prompts explicitly instruct models to auto-frame scenes. In code, this is best achieved via:
   ```javascript
   const box = new THREE.Box3().setFromObject(sceneGroup);
   const size = box.getSize(new THREE.Vector3());
   const center = box.getCenter(new THREE.Vector3());
   const maxDim = Math.max(size.x, size.y, size.z);
   camera.position.set(center.x, center.y + maxDim * 0.5, center.z + maxDim * 2.0);
   camera.lookAt(center);
   ```
2. **Auto-Healing JavaScript Identifiers**:
   Smaller 4-bit quantized models occasionally redeclare variables across code blocks (`Identifier '...' has already been declared`). The playground automatically heals this in `<1ms` by converting duplicate block declarations to `var`.
3. **WebGPU Deployment**:
   ONNX files can be deployed directly with Transformers.js via `env.backends.onnx.wasm.proxy = true` and `device: "webgpu"`.
