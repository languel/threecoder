#!/usr/bin/env python3
"""
Zero-llama.cpp-build exporter from fused MLX BF16 Safetensors (LiquidAI/LFM3-1.7B fine-tunes)
to GGUF v3 (BF16 and Q8_0) using cached 8.25 MB GGUF headers from LiquidAI/LFM3-1.7B-GGUF.

Verified bit-exact (max_diff = 0.0) against LiquidAI/LFM3-1.7B-GGUF on both ShortConv
and partial-RoPE / sliding-window attention layers.
"""
import argparse
import glob
import math
import os
import re
import struct
import time
import urllib.request

import mlx.core as mx
import numpy as np

HF_GGUF_BASE = "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main"
GGUF_SCALAR_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}
Q8_0_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])


def parse_gguf_header_and_tensors(buf: bytes) -> tuple[int, list[dict]]:
    magic, version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", buf, 0)
    assert magic == b"GGUF" and version == 3, f"Unexpected GGUF header: {magic} v{version}"
    pos = 24
    alignment = 32

    def read_str(p: int) -> tuple[str, int]:
        (ln,) = struct.unpack_from("<Q", buf, p)
        return buf[p + 8 : p + 8 + ln].decode("utf-8", errors="replace"), p + 8 + ln

    def skip_val(p: int, vtype: int) -> int:
        if vtype == 8:
            _, p = read_str(p)
            return p
        if vtype == 9:
            atype, alen = struct.unpack_from("<IQ", buf, p)
            p += 12
            if atype in GGUF_SCALAR_SIZES:
                return p + alen * GGUF_SCALAR_SIZES[atype]
            for _ in range(alen):
                p = skip_val(p, atype)
            return p
        return p + GGUF_SCALAR_SIZES[vtype]

    for _ in range(n_kv):
        k, pos = read_str(pos)
        (vtype,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if k == "general.alignment":
            (alignment,) = struct.unpack_from("<I", buf, pos)
        pos = skip_val(pos, vtype)

    tensors = []
    for _ in range(n_tensors):
        tname, pos = read_str(pos)
        (ndims,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        dims = list(struct.unpack_from(f"<{ndims}Q", buf, pos))
        pos += 8 * ndims
        ggml_type, rel_offset = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        tensors.append(
            {
                "name": tname,
                "dims": dims,
                "ggml_type": ggml_type,
                "rel_offset": rel_offset,
            }
        )

    data_offset = (pos + alignment - 1) // alignment * alignment
    return data_offset, tensors


def ensure_gguf_header_cache(gguf_filename: str, cache_dir: str) -> tuple[bytes, int, list[dict]]:
    os.makedirs(cache_dir, exist_ok=True)
    # 8.5 MB covers the 8,247,328-byte header + rope_freqs.weight at rel_offset 8192 (256 bytes)
    fetch_len = 8_600_000
    cpath = os.path.join(cache_dir, f"{gguf_filename}.prefix.bin")
    if not (os.path.exists(cpath) and os.path.getsize(cpath) == fetch_len):
        print(f"Caching 8.6 MB GGUF prefix for {gguf_filename}...", flush=True)
        url = f"{HF_GGUF_BASE}/{gguf_filename}"
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{fetch_len - 1}"})
        buf = urllib.request.urlopen(req).read()
        assert len(buf) == fetch_len, f"Expected {fetch_len} bytes, got {len(buf)}"
        with open(cpath, "wb") as f:
            f.write(buf)
    buf = open(cpath, "rb").read()
    data_offset, tensors = parse_gguf_header_and_tensors(buf)
    return buf, data_offset, tensors


def map_gguf_name_to_mlx_key(gname: str) -> str:
    if gname == "output_norm.weight":
        return "model.embedding_norm.weight"
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight"

    m = re.match(r"^blk\.(\d+)\.(.+)$", gname)
    if not m:
        raise KeyError(f"Unrecognized GGUF tensor: {gname}")
    idx, rest = m.group(1), m.group(2)

    direct = {
        "attn_norm.weight": f"model.layers.{idx}.operator_norm.weight",
        "post_attention_norm.weight": f"model.layers.{idx}.post_operator_norm.weight",
        "ffn_norm.weight": f"model.layers.{idx}.ffn_norm.weight",
        "post_ffw_norm.weight": f"model.layers.{idx}.post_ffn_norm.weight",
        "ffn_gate.weight": f"model.layers.{idx}.feed_forward.w1.weight",
        "ffn_up.weight": f"model.layers.{idx}.feed_forward.w3.weight",
        "ffn_down.weight": f"model.layers.{idx}.feed_forward.w2.weight",
        "shortconv.conv.weight": f"model.layers.{idx}.conv.conv.weight",
        "shortconv.in_proj.weight": f"model.layers.{idx}.conv.in_proj.weight",
        "shortconv.out_proj.weight": f"model.layers.{idx}.conv.out_proj.weight",
        "attn_q.weight": f"model.layers.{idx}.self_attn.q_proj.weight",
        "attn_k.weight": f"model.layers.{idx}.self_attn.k_proj.weight",
        "attn_v.weight": f"model.layers.{idx}.self_attn.v_proj.weight",
        "attn_output.weight": f"model.layers.{idx}.self_attn.o_proj.weight",
        "attn_gate.weight": f"model.layers.{idx}.self_attn.gate_proj.weight",
        "attn_q_norm.weight": f"model.layers.{idx}.self_attn.q_norm.weight",
        "attn_k_norm.weight": f"model.layers.{idx}.self_attn.k_norm.weight",
    }
    if rest not in direct:
        raise KeyError(f"Unmapped GGUF block tensor: {gname}")
    return direct[rest]


def quantize_q8_0(arr_f32: np.ndarray) -> bytes:
    blocks = arr_f32.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=-1)
    d_f16 = (amax / 127.0).astype(np.float16)
    d_f32 = d_f16.astype(np.float32)
    inv_d = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)
    qs = np.clip(np.round(blocks * inv_d[:, None]), -128, 127).astype(np.int8)

    out = np.empty(blocks.shape[0], dtype=Q8_0_DTYPE)
    out["d"] = d_f16
    out["qs"] = qs
    return out.tobytes()


def tensor_byte_size(dims: list[int], ggml_type: int) -> int:
    n_elem = math.prod(dims)
    if ggml_type == 0:  # F32
        return n_elem * 4
    if ggml_type == 30:  # BF16
        return n_elem * 2
    if ggml_type == 8:  # Q8_0
        return (n_elem // 32) * 34
    raise ValueError(f"Unsupported ggml_type: {ggml_type}")


def write_gguf_variant(
    weights: dict[str, mx.array],
    template_gguf_name: str,
    out_path: str,
    cache_dir: str,
) -> None:
    t0 = time.time()
    prefix_buf, data_offset, tensors = ensure_gguf_header_cache(template_gguf_name, cache_dir)
    tensors_sorted = sorted(tensors, key=lambda x: x["rel_offset"])
    last = tensors_sorted[-1]
    total_file_bytes = data_offset + last["rel_offset"] + tensor_byte_size(last["dims"], last["ggml_type"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Writing {out_path} ({len(tensors_sorted)} tensors, {total_file_bytes / 1e9:.2f} GB)...", flush=True)
    with open(out_path, "wb") as f:
        f.truncate(total_file_bytes)
        f.seek(0)
        f.write(prefix_buf[:data_offset])

        for idx_i, info in enumerate(tensors_sorted):
            name = info["name"]
            gtype = info["ggml_type"]
            rel_off = info["rel_offset"]
            exp_bytes = tensor_byte_size(info["dims"], gtype)

            if name == "rope_freqs.weight":
                s = data_offset + rel_off
                payload = prefix_buf[s : s + exp_bytes]
            else:
                mkey = map_gguf_name_to_mlx_key(name)
                arr_mx = weights[mkey]
                # GGUF dims [ne0, ne1] correspond to C row-major shape (ne1, ne0)
                exp_np_shape = tuple(reversed(info["dims"]))
                if name.endswith("shortconv.conv.weight") and arr_mx.ndim == 3:
                    # MLX stores (2048, 7, 1) or HF (2048, 1, 7) -> (2048, 7)
                    arr_mx = arr_mx.reshape(exp_np_shape)
                assert arr_mx.shape == exp_np_shape, (
                    f"Shape mismatch for {name} ({mkey}): got {arr_mx.shape}, expected {exp_np_shape}"
                )

                if gtype == 0:
                    payload = np.array(arr_mx.astype(mx.float32), dtype=np.float32).tobytes()
                elif gtype == 30:
                    payload = np.array(arr_mx.astype(mx.bfloat16).view(mx.uint16), dtype=np.uint16).tobytes()
                elif gtype == 8:
                    payload = quantize_q8_0(np.array(arr_mx.astype(mx.float32), dtype=np.float32))
                else:
                    raise ValueError(f"Unsupported ggml_type {gtype} on {name}")

            assert len(payload) == exp_bytes, f"Byte size mismatch on {name}: {len(payload)} != {exp_bytes}"
            f.seek(data_offset + rel_off)
            f.write(payload)

            if (idx_i + 1) % 100 == 0 or (idx_i + 1) == len(tensors_sorted):
                print(f"  [{os.path.basename(out_path)}] {idx_i + 1}/{len(tensors_sorted)} tensors written...", flush=True)

    print(f"Finished {out_path} in {time.time() - t0:.1f}s.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-dir", default="/Users/liuboto/mlx-models/LFM3-1.7B-GLB-3D-Coder-bf16")
    parser.add_argument("--gguf-dir", default="/Users/liuboto/mlx-models/gguf")
    parser.add_argument("--cache-dir", default="/Users/liuboto/onnx-models/.cache")
    parser.add_argument("--prefix", default="LFM3-1.7B-GLB-3D-Coder")
    args = parser.parse_args()

    print(f"Loading fused BF16 weights from {args.mlx_dir}...", flush=True)
    weights: dict[str, mx.array] = {}
    for sf in sorted(glob.glob(os.path.join(args.mlx_dir, "model*.safetensors"))):
        for k, v in mx.load(sf).items():
            weights[k.removeprefix("language_model.")] = v
    print(f"Loaded {len(weights)} tensors.", flush=True)

    bf16_out = os.path.join(args.gguf_dir, f"{args.prefix}-BF16.gguf")
    q8_out = os.path.join(args.gguf_dir, f"{args.prefix}-Q8_0.gguf")

    write_gguf_variant(weights, "LFM3-1.7B-BF16.gguf", bf16_out, args.cache_dir)
    write_gguf_variant(weights, "LFM3-1.7B-Q8_0.gguf", q8_out, args.cache_dir)


if __name__ == "__main__":
    main()
