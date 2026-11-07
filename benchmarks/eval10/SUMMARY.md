# Eval-10: Graduated 3D & GLB Benchmark Results

- **Total Suite Wall Time (8 models × 10 prompts = 80 scenes):** `2769.8s` (`46.16 min`)
- **Fast 4-Model Subset (`--fast-suite`, skipping 13.5 GB 4B BF16):** `1068.3s` (`17.80 min`)
- **Capped `<think>` Budget (Stage 1):** `900 tokens` + graduated HTML completion (`9,216–14,336 tokens`, `batch=10` on Apple M5 Max)

## 1. Aggregate Leaderboard

| Model | Mode | 10-Prompt Wall Time | Batch Speed | Peak RAM | Complete HTML | Zero-Shot Valid JS | Auto-Healed Valid JS | Valid + `GLTFExporter` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LFM3-1.7B Base (BF16, Pre-SFT)** | `glb` | `265.6s` | `194.0 tok/s` | `5.09 GB` | `10/10` | **`8/10`** | **`8/10`** | `5/10` |
| **LFM3-1.7B-GLB-Coder (BF16)** | `glb` | `272.6s` | `240.2 tok/s` | `5.34 GB` | `9/10` | **`2/10`** | **`7/10`** | `7/10` |
| **LFM3-1.7B-GLB-Coder (4-bit)** | `glb` | `156.7s` | `365.0 tok/s` | `5.34 GB` | `9/10` | **`2/10`** | **`6/10`** | `6/10` |
| **LFM3-1.7B-3D-Coder (BF16 Scene)** | `scene` | `133.1s` | `545.7 tok/s` | `5.02 GB` | `10/10` | **`4/10`** | **`9/10`** | `0/10` |
| **LFM3-1.7B-3D-Coder (4-bit Scene)** | `scene` | `137.8s` | `619.9 tok/s` | `5.02 GB` | `9/10` | **`2/10`** | **`7/10`** | `0/10` |
| **Qwen3.5-4B-GLB-Coder (4-bit)** | `glb` | `373.4s` | `187.7 tok/s` | `10.86 GB` | `10/10` | **`7/10`** | **`8/10`** | `8/10` |
| **Qwen3.5-4B-GLB-Coder (BF16)** | `glb` | `601.9s` | `122.5 tok/s` | `18.12 GB` | `10/10` | **`9/10`** | **`9/10`** | `9/10` |
| **Qwen3.5-4B-3D-Coder (BF16 Scene)** | `scene` | `828.6s` | `88.3 tok/s` | `18.12 GB` | `10/10` | **`9/10`** | **`9/10`** | `0/10` |

## 2. Per-Prompt Difficulty Matrix (`PASS` = zero-shot valid JS, `HEAL` = valid after `<1ms` duplicate-var heal, `FAIL` = syntax/truncation)

| Prompt | LFM3-1.7B Base (BF16, Pre-SFT) | LFM3-1.7B-GLB-Coder (BF16) | LFM3-1.7B-GLB-Coder (4-bit) | LFM3-1.7B-3D-Coder (BF16 Scene) | LFM3-1.7B-3D-Coder (4-bit Scene) | Qwen3.5-4B-GLB-Coder (4-bit) | Qwen3.5-4B-GLB-Coder (BF16) | Qwen3.5-4B-3D-Coder (BF16 Scene) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P01 · Default Cube** | PASS (`155L`) | HEAL (`762L`) | HEAL (`639L`) | HEAL (`949L`) | HEAL (`815L`) | PASS (`392L`) | PASS (`307L`) | PASS (`221L`) |
| **P02 · Sphere on Pedestal** | PASS (`139L`) | HEAL (`609L`) | HEAL (`475L`) | PASS (`662L`) | PASS (`589L`) | PASS (`407L`) | PASS (`497L`) | PASS (`666L`) |
| **P03 · Cube, Sphere & Tetrahedron** | PASS (`162L`) | PASS (`552L`) | HEAL (`521L`) | HEAL (`966L`) | PASS (`476L`) | PASS (`523L`) | PASS (`560L`) | PASS (`558L`) |
| **P04 · Five Platonic Solids** | FAIL | HEAL (`862L`) | HEAL (`616L`) | PASS (`881L`) | HEAL (`1547L`) | FAIL | PASS (`584L`) | PASS (`586L`) |
| **P05 · Low-Poly Pine Tree** | PASS (`217L`) | FAIL | FAIL | HEAL (`992L`) | HEAL (`1275L`) | PASS (`810L`) | PASS (`817L`) | PASS (`842L`) |
| **P06 · Firefly Particle System** | FAIL | HEAL (`605L`) | PASS (`973L`) | FAIL | FAIL | PASS (`851L`) | PASS (`847L`) | PASS (`720L`) |
| **P07 · Stylized Fox** | PASS (`205L`) | PASS (`711L`) | PASS (`762L`) | PASS (`855L`) | HEAL (`1204L`) | PASS (`693L`) | FAIL | PASS (`696L`) |
| **P08 · Animated Owl** | PASS (`192L`) | FAIL | FAIL | HEAL (`1658L`) | FAIL | FAIL | PASS (`1009L`) | PASS (`741L`) |
| **P09 · Stylized Human Explorer** | PASS (`277L`) | FAIL | FAIL | HEAL (`1042L`) | FAIL | HEAL (`878L`) | PASS (`982L`) | FAIL |
| **P10 · Winter Cabin Snowglobe** | PASS (`231L`) | HEAL (`811L`) | FAIL | PASS (`546L`) | HEAL (`1208L`) | PASS (`896L`) | PASS (`890L`) | PASS (`973L`) |
