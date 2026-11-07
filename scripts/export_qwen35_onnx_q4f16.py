#!/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python
"""
Zero-PyTorch, bit-verified exporter from MLX BF16 Safetensors (Qwen3.5-4B fine-tunes)
to Transformers.js v4 WebGPU ONNX-OPT q4f16 format.

Uses the fused WebGPU operator graph from onnx-community/Qwen3.5-4B-ONNX-OPT
(SimplifiedLayerNormalization, GroupQueryAttention, MatMulNBits block_size=32,
and GatedDeltaNet recurrent state graphs) and packs all fine-tuned weights directly.
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

HF_ONNX_OPT_BASE = "https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT/resolve/main"
TEMPLATE_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "onnx/decoder_model_merged_q4f16.onnx",
    "onnx/embed_tokens_q4f16.onnx",
    "onnx/vision_encoder_q4f16.onnx",
]


def read_varint(b: bytes, i: int) -> tuple[int, int]:
    res, shift = 0, 0
    while True:
        byte = b[i]
        i += 1
        res |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return res, i
        shift += 7


def parse_fields_with_spans(b: bytes):
    i, n = 0, len(b)
    out = []
    while i < n:
        start = i
        tag, i = read_varint(b, i)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:
            val_start = i
            v, i = read_varint(b, i)
        elif wtype == 2:
            ln, i = read_varint(b, i)
            val_start = i
            v = b[i : i + ln]
            i += ln
        elif wtype == 5:
            val_start = i
            v = b[i : i + 4]
            i += 4
        elif wtype == 1:
            val_start = i
            v = b[i : i + 8]
            i += 8
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wtype}")
        out.append((fnum, wtype, v, start, val_start, i))
    return out


def parse_tensor_info(tb: bytes, tb_val_offset: int):
    dims = []
    dtype = None
    name = ""
    ext = {}
    data_loc = 0
    raw_span = None
    for fn, wt, v, _, val_start, val_end in parse_fields_with_spans(tb):
        if fn == 1 and wt == 0:
            dims.append(v)
        elif fn == 1 and wt == 2:
            j = 0
            while j < len(v):
                d, j = read_varint(v, j)
                dims.append(d)
        elif fn == 2:
            dtype = v
        elif fn == 8:
            name = v.decode("utf-8")
        elif fn == 9 and wt == 2:
            raw_span = (tb_val_offset + val_start, tb_val_offset + val_end)
        elif fn == 13:
            sub = parse_fields_with_spans(v)
            ks = [x[2].decode("utf-8") for x in sub if x[0] == 1]
            vs = [x[2].decode("utf-8") for x in sub if x[0] == 2]
            if ks and vs:
                ext[ks[0]] = vs[0]
        elif fn == 14:
            data_loc = v
    return {
        "name": name,
        "dtype": dtype,
        "dims": dims,
        "data_loc": data_loc,
        "ext": ext,
        "raw_span": raw_span,
    }


def quantize_matmul_4bits(weight_2d: np.ndarray) -> tuple[bytes, bytes, bytes]:
    """
    Asymmetric 4-bit blockwise quantization (block_size=32) matching ONNX Runtime MatMulNBits.
    weight_2d: float32 array of shape (N, K) where K % 64 == 0.
    Returns (quant_bytes, scales_bytes, zp_bytes).
    """
    n_rows, k_cols = weight_2d.shape
    assert k_cols % 64 == 0, f"Expected K divisible by 64, got {weight_2d.shape}"
    blocks = weight_2d.reshape(n_rows, k_cols // 32, 32)

    w_min = np.minimum(blocks.min(axis=-1), 0.0)
    w_max = np.maximum(blocks.max(axis=-1), 0.0)

    scales_f16 = ((w_max - w_min) / 15.0).astype(np.float16)
    scales_f32 = scales_f16.astype(np.float32)
    safe_scales = np.where(scales_f32 > 0.0, scales_f32, 1.0)

    zp = np.clip(np.round(-w_min / safe_scales), 0.0, 15.0).astype(np.uint8)
    q = np.clip(np.round(blocks / safe_scales[..., None] + zp[..., None]), 0.0, 15.0).astype(np.uint8)

    quant_packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    zp_packed = (zp[:, 0::2] | (zp[:, 1::2] << 4)).astype(np.uint8)

    return quant_packed.tobytes(), scales_f16.tobytes(), zp_packed.tobytes()


def load_mlx_bf16_weights(mlx_dir: str) -> dict[str, mx.array]:
    weights = {}
    for sf in sorted(glob.glob(os.path.join(mlx_dir, "model*.safetensors"))):
        if "vision" in os.path.basename(sf):
            continue
        weights.update(mx.load(sf))
    return weights


def get_f32(weights: dict[str, mx.array], key: str) -> np.ndarray:
    if key not in weights:
        raise KeyError(f"Missing weight key in MLX checkpoint: {key}")
    return np.array(weights[key].astype(mx.float32))


def ensure_templates(out_dir: str, include_vision_data: bool = True) -> None:
    os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    for rel in TEMPLATE_FILES:
        dst = os.path.join(out_dir, rel)
        if not os.path.exists(dst):
            print(f"Downloading template {rel}...", flush=True)
            urllib.request.urlretrieve(f"{HF_ONNX_OPT_BASE}/{rel}", dst)

    if include_vision_data:
        vis_data = os.path.join(out_dir, "onnx/vision_encoder_q4f16.onnx_data")
        if not os.path.exists(vis_data):
            print("Downloading unmodified Qwen3.5-4B vision_encoder_q4f16.onnx_data (198 MB)...", flush=True)
            urllib.request.urlretrieve(f"{HF_ONNX_OPT_BASE}/onnx/vision_encoder_q4f16.onnx_data", vis_data)


def map_onnx_ext_name_to_mlx_key(onnx_base: str) -> tuple[str, str]:
    """
    Returns (kind, mlx_key) where kind in {'q4_group', 'f32_1d', 'conv1d_f16'}.
    """
    if onnx_base == "model_embed_tokens_weight":
        return "q4_group", "language_model.model.embed_tokens.weight"
    if onnx_base == "lm_head_MatMul_weight":
        return "q4_group", "language_model.model.embed_tokens.weight"
    if onnx_base == "model.layers.32.final_norm_layernorm.weight":
        return "f32_1d", "language_model.model.norm.weight"

    m = re.match(r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$", onnx_base)
    if m:
        return "f32_1d", f"language_model.model.layers.{m.group(1)}.{m.group(2)}.weight"

    m = re.match(r"^model\.layers\.(\d+)\.gdn\.conv1d\.weight_3d$", onnx_base)
    if m:
        return "conv1d_f16", f"language_model.model.layers.{m.group(1)}.linear_attn.conv1d.weight"

    m = re.match(r"^model\.layers\.(\d+)\.attn\.(q_norm|k_norm)\.layernorm\.weight$", onnx_base)
    if m:
        return "f32_1d", f"language_model.model.layers.{m.group(1)}.self_attn.{m.group(2)}.weight"

    m = re.match(r"^model_layers_(\d+)_gdn_(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)_MatMul_weight$", onnx_base)
    if m:
        return "q4_group", f"language_model.model.layers.{m.group(1)}.linear_attn.{m.group(2)}.weight"

    m = re.match(r"^model_layers_(\d+)_attn_(q_proj|k_proj|v_proj|o_proj)_MatMul_weight$", onnx_base)
    if m:
        return "q4_group", f"language_model.model.layers.{m.group(1)}.self_attn.{m.group(2)}.weight"

    m = re.match(r"^model_layers_(\d+)_mlp_(gate_proj|up_proj|down_proj)_MatMul_weight$", onnx_base)
    if m:
        return "q4_group", f"language_model.model.layers.{m.group(1)}.mlp.{m.group(2)}.weight"

    raise ValueError(f"Unrecognized ONNX initializer: {onnx_base}")


def get_onnx_initializers(onnx_bytes: bytes):
    model_fields = parse_fields_with_spans(onnx_bytes)
    graph_field = [x for x in model_fields if x[0] == 7][0]
    graph_bytes, graph_val_start = graph_field[2], graph_field[4]
    inits = []
    for fn, wt, v, _, val_start, _ in parse_fields_with_spans(graph_bytes):
        if fn == 5:
            inits.append(parse_tensor_info(v, graph_val_start + val_start))
    return inits


def export_onnx_q4f16(mlx_dir: str, out_dir: str, include_vision: bool = True) -> None:
    t0 = time.time()
    ensure_templates(out_dir, include_vision_data=include_vision)
    shutil.copy2(os.path.join(mlx_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    print(f"Loading BF16 Safetensors from {mlx_dir}...", flush=True)
    weights = load_mlx_bf16_weights(mlx_dir)
    print(f"Loaded {len(weights)} language tensors.", flush=True)

    # Cache quantized 2D weights (notably tied embed_tokens / lm_head so we only quantize 248K x 2560 once!)
    q4_cache: dict[str, tuple[bytes, bytes, bytes]] = {}

    def get_q4(mlx_key: str) -> tuple[bytes, bytes, bytes]:
        if mlx_key not in q4_cache:
            q4_cache[mlx_key] = quantize_matmul_4bits(get_f32(weights, mlx_key))
        return q4_cache[mlx_key]

    for onnx_rel in ["onnx/embed_tokens_q4f16.onnx", "onnx/decoder_model_merged_q4f16.onnx"]:
        onnx_path = os.path.join(out_dir, onnx_rel)
        onnx_buf = bytearray(open(onnx_path, "rb").read())
        inits = get_onnx_initializers(bytes(onnx_buf))

        # 1. Patch inline learned GDN 1D initializers (A_neg_exp, dt_bias, gdn.norm.weight)
        inline_patched = 0
        for info in inits:
            if info["data_loc"] == 0:
                m = re.match(r"^model\.layers\.(\d+)\.gdn\.(A_neg_exp|dt_bias|norm\.weight)$", info["name"])
                if not m:
                    continue
                layer_idx, suffix = m.group(1), m.group(2)
                if suffix == "A_neg_exp":
                    arr = -np.exp(get_f32(weights, f"language_model.model.layers.{layer_idx}.linear_attn.A_log"))
                elif suffix == "dt_bias":
                    arr = get_f32(weights, f"language_model.model.layers.{layer_idx}.linear_attn.dt_bias")
                else:
                    arr = get_f32(weights, f"language_model.model.layers.{layer_idx}.linear_attn.norm.weight")
                raw_bytes = arr.astype(np.float32).tobytes()
                s, e = info["raw_span"]
                assert e - s == len(raw_bytes), f"Inline size mismatch for {info['name']}: {e - s} != {len(raw_bytes)}"
                onnx_buf[s:e] = raw_bytes
                inline_patched += 1

        if inline_patched:
            with open(onnx_path, "wb") as f:
                f.write(onnx_buf)
            print(f"Patched {inline_patched} inline GDN tensors in {onnx_rel}.", flush=True)

        # 2. Group external initializers by data file and write at exact offsets
        by_file: dict[str, list[dict]] = {}
        for info in inits:
            if info["data_loc"] == 1:
                by_file.setdefault(info["ext"]["location"], []).append(info)

        for data_fname, items in by_file.items():
            items.sort(key=lambda x: int(x["ext"]["offset"]))
            total_len = max(int(x["ext"]["offset"]) + int(x["ext"]["length"]) for x in items)
            data_path = os.path.join(out_dir, "onnx", data_fname)
            print(f"Writing {data_fname} ({len(items)} tensors, {total_len / 1e6:.2f} MB)...", flush=True)

            with open(data_path, "wb") as out_f:
                out_f.truncate(total_len)
                for idx_i, info in enumerate(items):
                    name = info["name"]
                    offset = int(info["ext"]["offset"])
                    expected_len = int(info["ext"]["length"])

                    if name.endswith(("_quant", "_scales", "_zp")):
                        suffix = name.rsplit("_", 1)[1]
                        base = name[: -(len(suffix) + 1)]
                        kind, mlx_key = map_onnx_ext_name_to_mlx_key(base)
                        assert kind == "q4_group"
                        qb, sb, zb = get_q4(mlx_key)
                        payload = {"quant": qb, "scales": sb, "zp": zb}[suffix]
                    else:
                        kind, mlx_key = map_onnx_ext_name_to_mlx_key(name)
                        if kind == "f32_1d":
                            payload = get_f32(weights, mlx_key).astype(np.float32).tobytes()
                        elif kind == "conv1d_f16":
                            # MLX stores (8192, 4, 1); ONNX expects (8192, 1, 4) float16
                            c = np.swapaxes(get_f32(weights, mlx_key), 1, 2).astype(np.float16)
                            payload = c.tobytes()
                        else:
                            raise ValueError(kind)

                    assert len(payload) == expected_len, (
                        f"Byte length mismatch on {name}: got {len(payload)}, expected {expected_len}"
                    )
                    out_f.seek(offset)
                    out_f.write(payload)

                    if (idx_i + 1) % 150 == 0:
                        print(f"  [{data_fname}] {idx_i + 1}/{len(items)} tensors written...", flush=True)

            # Free single-use layer cache entries while keeping tied embed_tokens if still needed
            q4_cache = {k: v for k, v in q4_cache.items() if "embed_tokens" in k}

    elapsed = time.time() - t0
    print(f"\nSUCCESS! Exported WebGPU ONNX q4f16 model to {out_dir} in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlx-dir",
        default="/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16",
        help="Path to unquantized BF16 MLX checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/liuboto/onnx-models/Qwen3.5-4B-GLB-3D-Coder-ONNX-q4f16",
        help="Output directory for Transformers.js WebGPU ONNX model",
    )
    parser.add_argument("--no-vision", action="store_true", help="Skip downloading 198 MB vision encoder data")
    args = parser.parse_args()
    export_onnx_q4f16(args.mlx_dir, args.out_dir, include_vision=not args.no_vision)
