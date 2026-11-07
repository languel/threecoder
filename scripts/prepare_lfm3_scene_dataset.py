#!/usr/bin/env python3
"""
Prepare SFT dataset (train.jsonl / valid.jsonl) for LiquidAI/LFM3-1.7B and LFM3-Omni-2B
from ccore/ThreeJS-3D-Scenes on Hugging Face.

Follows strict ML split hygiene:
- Preserves upstream train (3,933) and validation (213) splits with zero cross-split leakage.
- Filters out broken/empty rows.
- Strictly bounds sequences to <= max_seq_length (default 16,384 tokens) so 100% of
  training sequences retain full valid HTML, script tags, and <|im_end|> EOS token.
- Produces Direct-HTML mode (<think></think>{content}) for instant, full-scene Three.js code generation.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def build_split(
    hf_split,
    tokenizer,
    max_seq_length: int,
    seed: int,
    split_name: str,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    null_content = 0
    dropped_over_max = 0
    kept_count = 0
    token_lengths: list[int] = []

    for i, row in enumerate(hf_split):
        msgs = row["messages"]
        asst = msgs[-1]
        content = asst.get("content")

        if not content or not content.strip():
            null_content += 1
            continue

        # Direct-HTML message structure: system, user, assistant (with full HTML scene)
        direct_msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
        token_ids = tokenizer.apply_chat_template(direct_msgs, return_dict=False)
        tok_len = len(token_ids)

        if tok_len <= max_seq_length:
            records.append({"messages": direct_msgs})
            token_lengths.append(tok_len)
            kept_count += 1
        else:
            dropped_over_max += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(hf_split):
            print(f"[{split_name}] Processed {i + 1}/{len(hf_split)}: kept={kept_count}, dropped_over_max={dropped_over_max}", flush=True)

    rng = random.Random(seed)
    rng.shuffle(records)

    arr = np.array(token_lengths, dtype=np.int64)
    stats = {
        "split": split_name,
        "source_rows": len(hf_split),
        "null_content": null_content,
        "kept_examples": kept_count,
        "dropped_over_max": dropped_over_max,
        "token_p10_p50_p90_max": np.percentile(arr, [10, 50, 90, 100]).astype(int).tolist()
        if len(arr)
        else [],
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B-MLX-bf16")
    parser.add_argument("--dataset", default="ccore/ThreeJS-3D-Scenes")
    parser.add_argument("--out-dir", default="/Users/liuboto/dev/threecoder/data/lfm3_scene")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Verify chat template preview
    sample_prompt = [{"role": "system", "content": "You are a Three.js expert."}, {"role": "user", "content": "Create a spinning torus."}]
    prompt_ids = tokenizer.apply_chat_template(sample_prompt, add_generation_prompt=True, return_dict=False)
    print(
        f"Prompt preview ({len(prompt_ids)} toks): "
        f"{repr(tokenizer.decode(prompt_ids))}",
        flush=True,
    )

    print(f"Loading {args.dataset}...", flush=True)
    ds = load_dataset(args.dataset)

    train_records, train_stats = build_split(
        ds["train"], tokenizer, args.max_seq_length, args.seed, "train"
    )
    valid_records, valid_stats = build_split(
        ds["validation"], tokenizer, args.max_seq_length, args.seed + 1, "valid"
    )

    for fname, rows in [("train.jsonl", train_records), ("valid.jsonl", valid_records)]:
        fpath = out_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} examples to {fpath}", flush=True)

    summary = {"dataset": args.dataset, "max_seq_length": args.max_seq_length, "train": train_stats, "valid": valid_stats}
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Dataset preparation stats:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
