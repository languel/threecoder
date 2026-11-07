#!/usr/bin/env python3
"""
Zero-PyTorch, bit-verified exporter from fused MLX BF16 Safetensors (LiquidAI/LFM3-1.7B fine-tunes)
to Transformers.js v4 WebGPU ONNX q4f16 format (onnx-community/LFM3-1.7B-ONNX graph).

Key architectural details verified against onnx-community/LFM3-1.7B-ONNX:
1. Partial-RoPE head layout on full_attention layers (layers 2, 6, 10, 14, 18, 22; partial_rotary_factor=0.5):
   MLX checkpoints swap sub-blocks [32:64] and [64:96] inside every 128-dim head of q_proj, k_proj,
   q_norm, and k_norm for mx.fast.rope(..., traditional=False). We unswap [0, 2, 1, 3] x 32 back to
   standard HF/ONNX order (verified 0.0 fp16 norm diff and >0.9959 q4 cosine).
2. Fused gated attention query projection:
   ONNX concatenates self_attn.q_proj.weight [2048, 2048] and self_attn.gate_proj.weight [16, 2048]
   along axis 0 into model_layers_{N}_attn_q_proj_MatMul_weight [2064, 2048].
3. Mixed symmetric + asymmetric MatMulNBits (block_size=32):
   - Asymmetric (with _zp): embed_tokens, conv_in_proj, conv_out_proj, attn_o_proj, mlp_down_proj (w2).
   - Symmetric (no _zp, implicit zp=8): attn_q_proj, attn_k_proj, attn_v_proj, mlp_gate_proj (w1), mlp_up_proj (w3).
4. Static RoPE tables (cos/sin_cache_{global,local}, 100.66 MB total) are fetched once via HTTP Range
   from onnx-community/LFM3-1.7B-ONNX and cached locally.
"""
import argparse
import glob
import json
import os
import re
import shutil
import time
import urllib.request

import mlx.core as mx
import numpy as np

from export_qwen35_onnx_q4f16 import get_onnx_initializers, quantize_matmul_4bits

HF_ONNX_BASE = "https://huggingface.co/onnx-community/LFM3-1.7B-ONNX/resolve/main"
TEMPLATE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "onnx/model_q4f16.onnx",
]
STATIC_ROPE_NAMES = {
    "cos_cache_global",
    "sin_cache_global",
    "cos_cache_local",
    "sin_cache_local",
}


def quantize_matmul_4bits_symmetric(weight_2d: np.ndarray) -> tuple[bytes, bytes]:
    """
    Symmetric 4-bit blockwise quantization (block_size=32, implicit zero_point=8)
    matching ONNX Runtime MatMulNBits when zero_points input is omitted.
    Valid dequantized range per block is [-8 * scale, +7 * scale].
    """
    n_rows, k_cols = weight_2d.shape
    assert k_cols % 32 == 0, f"Expected K divisible by 32, got {weight_2d.shape}"
    blocks = weight_2d.reshape(n_rows, k_cols // 32, 32)

    b_min = np.minimum(blocks.min(axis=-1), 0.0)
    b_max = np.maximum(blocks.max(axis=-1), 0.0)
    scales_f16 = np.maximum(-b_min / 8.0, b_max / 7.0).astype(np.float16)
    scales_f32 = scales_f16.astype(np.float32)
    safe_scales = np.where(scales_f32 > 0.0, scales_f32, 1.0)

    q = np.clip(np.round(blocks / safe_scales[..., None] + 8.0), 0.0, 15.0).astype(np.uint8)
    quant_packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    return quant_packed.tobytes(), scales_f16.tobytes()


def unswap_partial_rope_heads(arr: np.ndarray) -> np.ndarray:
    """Unswap [32:64] <-> [64:96] inside every 128-dim head along axis 0."""
    orig_shape = arr.shape
    assert orig_shape[0] % 128 == 0, f"Expected axis 0 multiple of 128, got {orig_shape}"
    h = arr.reshape(-1, 4, 32, *orig_shape[1:])
    return h[:, [0, 2, 1, 3], ...].reshape(orig_shape)


def load_mlx_weights(mlx_dir: str) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for sf in sorted(glob.glob(os.path.join(mlx_dir, "model*.safetensors"))):
        for k, v in mx.load(sf).items():
            weights[k.removeprefix("language_model.")] = v
    return weights


def get_f32(weights: dict[str, mx.array], key: str) -> np.ndarray:
    if key not in weights:
        raise KeyError(f"Missing weight key in MLX checkpoint: {key}")
    return np.array(weights[key].astype(mx.float32))


def ensure_templates_and_rope_cache(out_dir: str, cache_dir: str, inits_out: list) -> dict[str, bytes]:
    os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    for rel in TEMPLATE_FILES:
        dst = os.path.join(out_dir, rel)
        if not os.path.exists(dst):
            print(f"Downloading template {rel}...", flush=True)
            urllib.request.urlretrieve(f"{HF_ONNX_BASE}/{rel}", dst)

    onnx_bytes = open(os.path.join(out_dir, "onnx/model_q4f16.onnx"), "rb").read()
    inits = get_onnx_initializers(onnx_bytes)
    inits_out.extend(inits)

    rope_blobs: dict[str, bytes] = {}
    data_url = f"{HF_ONNX_BASE}/onnx/model_q4f16.onnx_data"
    for info in inits:
        if info["name"] in STATIC_ROPE_NAMES:
            cpath = os.path.join(cache_dir, f"lfm3_{info['name']}.bin")
            exp_len = int(info["ext"]["length"])
            if not (os.path.exists(cpath) and os.path.getsize(cpath) == exp_len):
                off = int(info["ext"]["offset"])
                print(f"Caching static RoPE table {info['name']} ({exp_len / 1e6:.2f} MB)...", flush=True)
                req = urllib.request.Request(data_url, headers={"Range": f"bytes={off}-{off + exp_len - 1}"})
                buf = urllib.request.urlopen(req).read()
                assert len(buf) == exp_len, f"Range read mismatch: {len(buf)} != {exp_len}"
                with open(cpath, "wb") as f:
                    f.write(buf)
            rope_blobs[info["name"]] = open(cpath, "rb").read()
    return rope_blobs


def export_lfm3_onnx_q4f16(mlx_dir: str, out_dir: str, cache_dir: str) -> None:
    t0 = time.time()
    inits: list[dict] = []
    rope_blobs = ensure_templates_and_rope_cache(out_dir, cache_dir, inits)

    tok_src = os.path.join(mlx_dir, "tokenizer.json")
    if os.path.exists(tok_src):
        shutil.copy2(tok_src, os.path.join(out_dir, "tokenizer.json"))

    cfg = json.load(open(os.path.join(mlx_dir, "config.json")))
    layer_types: list[str] = cfg["layer_types"]

    print(f"Loading BF16 Safetensors from {mlx_dir}...", flush=True)
    weights = load_mlx_weights(mlx_dir)
    print(f"Loaded {len(weights)} LFM3 tensors.", flush=True)

    # 1. Patch inline q_norm / k_norm fp16 weights in onnx/model_q4f16.onnx
    onnx_path = os.path.join(out_dir, "onnx/model_q4f16.onnx")
    onnx_buf = bytearray(open(onnx_path, "rb").read())
    inline_patched = 0
    for info in inits:
        if info["data_loc"] == 0:
            m = re.match(r"^model\.layers\.(\d+)\.attn\.(q_norm|k_norm)\.layernorm\.weight$", info["name"])
            if not m:
                continue
            l_idx = int(m.group(1))
            norm_kind = m.group(2)
            arr = get_f32(weights, f"model.layers.{l_idx}.self_attn.{norm_kind}.weight")
            if layer_types[l_idx] == "full_attention":
                arr = unswap_partial_rope_heads(arr)
            raw_bytes = arr.astype(np.float16).tobytes()
            s, e = info["raw_span"]
            assert e - s == len(raw_bytes), f"Inline span mismatch on {info['name']}: {e - s} != {len(raw_bytes)}"
            onnx_buf[s:e] = raw_bytes
            inline_patched += 1

    with open(onnx_path, "wb") as f:
        f.write(onnx_buf)
    print(f"Patched {inline_patched} inline q_norm/k_norm fp16 tensors in onnx/model_q4f16.onnx.", flush=True)

    # 2. Build external onnx/model_q4f16.onnx_data
    ext_items = [x for x in inits if x["data_loc"] == 1]
    ext_items.sort(key=lambda x: int(x["ext"]["offset"]))
    ext_names = {x["name"] for x in ext_items}
    total_len = max(int(x["ext"]["offset"]) + int(x["ext"]["length"]) for x in ext_items)

    q4_cache: dict[str, dict[str, bytes]] = {}

    def build_2d_matrix(base_name: str) -> np.ndarray:
        if base_name == "model_embed_tokens_weight":
            return get_f32(weights, "model.embed_tokens.weight")
        m = re.match(r"^model_layers_(\d+)_conv_(in_proj|out_proj)_MatMul_weight$", base_name)
        if m:
            return get_f32(weights, f"model.layers.{m.group(1)}.conv.{m.group(2)}.weight")
        m = re.match(r"^model_layers_(\d+)_attn_(q_proj|k_proj|v_proj|o_proj)_MatMul_weight$", base_name)
        if m:
            l_idx = int(m.group(1))
            proj = m.group(2)
            w = get_f32(weights, f"model.layers.{l_idx}.self_attn.{proj}.weight")
            if proj in ("q_proj", "k_proj") and layer_types[l_idx] == "full_attention":
                w = unswap_partial_rope_heads(w)
            if proj == "q_proj":
                g = get_f32(weights, f"model.layers.{l_idx}.self_attn.gate_proj.weight")
                w = np.concatenate([w, g], axis=0)
            return w
        m = re.match(r"^model_layers_(\d+)_mlp_(gate_proj|up_proj|down_proj)_MatMul_weight$", base_name)
        if m:
            l_idx = int(m.group(1))
            w_map = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
            return get_f32(weights, f"model.layers.{l_idx}.feed_forward.{w_map[m.group(2)]}.weight")
        raise ValueError(f"Unrecognized 4-bit base initializer: {base_name}")

    def get_q4_part(base_name: str, suffix: str) -> bytes:
        if base_name not in q4_cache:
            mat = build_2d_matrix(base_name)
            has_zp = f"{base_name}_zp" in ext_names
            if has_zp:
                qb, sb, zb = quantize_matmul_4bits(mat)
                q4_cache[base_name] = {"quant": qb, "scales": sb, "zp": zb}
            else:
                qb, sb = quantize_matmul_4bits_symmetric(mat)
                q4_cache[base_name] = {"quant": qb, "scales": sb}
        blob = q4_cache[base_name].pop(suffix)
        if not q4_cache[base_name]:
            del q4_cache[base_name]
        return blob

    data_path = os.path.join(out_dir, "onnx/model_q4f16.onnx_data")
    print(f"Writing onnx/model_q4f16.onnx_data ({len(ext_items)} tensors, {total_len / 1e6:.2f} MB)...", flush=True)
    with open(data_path, "wb") as out_f:
        out_f.truncate(total_len)
        for idx_i, info in enumerate(ext_items):
            name = info["name"]
            offset = int(info["ext"]["offset"])
            expected_len = int(info["ext"]["length"])

            if name in STATIC_ROPE_NAMES:
                payload = rope_blobs[name]
            elif name.endswith(("_quant", "_scales", "_zp")):
                suffix = name.rsplit("_", 1)[1]
                base = name[: -(len(suffix) + 1)]
                payload = get_q4_part(base, suffix)
            elif name == "model.layers.24.final_norm_layernorm.weight":
                payload = get_f32(weights, "model.embedding_norm.weight").astype(np.float16).tobytes()
            else:
                m_norm = re.match(
                    r"^model\.layers\.(\d+)\.(operator|post_operator|ffn|post_ffn)_layernorm\.weight$",
                    name,
                )
                m_conv = re.match(r"^model\.layers\.(\d+)\.conv\.conv\.weight$", name)
                if m_norm:
                    l_idx, kind = m_norm.group(1), m_norm.group(2)
                    payload = get_f32(weights, f"model.layers.{l_idx}.{kind}_norm.weight").astype(np.float16).tobytes()
                elif m_conv:
                    l_idx = m_conv.group(1)
                    c = get_f32(weights, f"model.layers.{l_idx}.conv.conv.weight")
                    if c.shape[1] > c.shape[2]:
                        c = np.swapaxes(c, 1, 2)
                    payload = c.astype(np.float16).tobytes()
                else:
                    raise ValueError(f"Unmapped external ONNX initializer: {name}")

            assert len(payload) == expected_len, (
                f"Byte length mismatch on {name}: got {len(payload)}, expected {expected_len}"
            )
            out_f.seek(offset)
            out_f.write(payload)

            if (idx_i + 1) % 100 == 0 or (idx_i + 1) == len(ext_items):
                print(f"  [model_q4f16.onnx_data] {idx_i + 1}/{len(ext_items)} tensors written...", flush=True)

    elapsed = time.time() - t0
    print(f"\nSUCCESS! Exported LFM3 WebGPU ONNX q4f16 model to {out_dir} in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlx-dir",
        default="/Users/liuboto/mlx-models/LFM3-1.7B-GLB-3D-Coder-bf16",
        help="Path to fused BF16 MLX checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/liuboto/onnx-models/LFM3-1.7B-GLB-3D-Coder-ONNX-q4f16",
        help="Output directory for Transformers.js WebGPU ONNX model",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Users/liuboto/onnx-models/.cache",
        help="Local cache directory for static LFM3 RoPE tables",
    )
    args = parser.parse_args()
    export_lfm3_onnx_q4f16(args.mlx_dir, args.out_dir, args.cache_dir)
