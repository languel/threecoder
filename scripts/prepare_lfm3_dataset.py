#!/Users/liuboto/dev/threecoder/.venv/bin/python
"""
Prepare Dual-Mode SFT dataset (train.jsonl / valid.jsonl) for LiquidAI/LFM3-1.7B
from ccore/ThreeJS-3D-GLB-Assets ('conversations' config).

Follows strict ML split hygiene:
- Fixed upstream train (3,501) and validation (417) splits are processed independently
  with zero cross-split leakage.
- Explicitly audits missing/null content and reasoning fields.
- Produces two complementary supervision modes, strictly bounded to <= max_seq_length
  so 100% of training sequences retain their closing </html> and <|im_end|> EOS token:
    1. Direct-HTML mode (no reasoning_content -> <think></think>{content}): teaches fast,
       untruncated WebGPU HTML+GLTFExporter generation when thinking is off/skipped.
    2. Concise-Think mode (with reasoning_content -> <think>{reasoning}</think>{content}
       where total tokens <= max_seq_length): preserves structured 3D geometry/animation
       planning without 25K-token reasoning bloat.
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
    null_reasoning = 0
    dropped_direct_over_max = 0
    dropped_think_over_max = 0
    kept_direct = 0
    kept_think = 0
    token_lengths: list[int] = []

    for row in hf_split:
        msgs = row["messages"]
        asst = msgs[-1]
        content = asst.get("content")
        reasoning = asst.get("reasoning_content")

        if not content or not content.strip():
            null_content += 1
            continue
        if not reasoning or not reasoning.strip():
            null_reasoning += 1

        # 1. Direct-HTML sample (<think></think>{content})
        direct_msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
        direct_ids = tokenizer.apply_chat_template(direct_msgs, return_dict=False)
        if len(direct_ids) <= max_seq_length:
            records.append({"messages": direct_msgs})
            token_lengths.append(len(direct_ids))
            kept_direct += 1
        else:
            dropped_direct_over_max += 1

        # 2. Concise-Think sample (<think>{reasoning}</think>{content}), kept only when
        #    full reasoning + complete HTML + <|im_end|> fit inside max_seq_length.
        if reasoning and reasoning.strip():
            think_msgs = []
            for m in msgs:
                item = {"role": m["role"], "content": m["content"]}
                if m.get("reasoning_content"):
                    item["reasoning_content"] = m["reasoning_content"]
                think_msgs.append(item)
            think_ids = tokenizer.apply_chat_template(think_msgs, return_dict=False)
            if len(think_ids) <= max_seq_length:
                records.append({"messages": think_msgs})
                token_lengths.append(len(think_ids))
                kept_think += 1
            else:
                dropped_think_over_max += 1

    rng = random.Random(seed)
    rng.shuffle(records)

    arr = np.array(token_lengths, dtype=np.int64)
    stats = {
        "split": split_name,
        "source_rows": len(hf_split),
        "null_content": null_content,
        "null_reasoning": null_reasoning,
        "kept_direct_html": kept_direct,
        "kept_concise_think": kept_think,
        "dropped_direct_over_max": dropped_direct_over_max,
        "dropped_think_over_max": dropped_think_over_max,
        "total_output_examples": len(records),
        "token_p10_p50_p90_max": np.percentile(arr, [10, 50, 90, 100]).astype(int).tolist()
        if len(arr)
        else [],
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B-MLX-bf16")
    parser.add_argument("--dataset", default="ccore/ThreeJS-3D-GLB-Assets")
    parser.add_argument("--subset", default="conversations")
    parser.add_argument("--out-dir", default="/Users/liuboto/dev/threecoder/data/lfm3_glb")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Verify prompt masking offset behavior on LFM3 chat template
    sample_prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    prompt_ids = tokenizer.apply_chat_template(sample_prompt, add_generation_prompt=True, return_dict=False)
    print(
        f"Prompt mask boundary preview ({len(prompt_ids)} toks): "
        f"{repr(tokenizer.decode(prompt_ids))}",
        flush=True,
    )

    print(f"Loading {args.dataset} ({args.subset})...", flush=True)
    ds = load_dataset(args.dataset, args.subset)

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

    summary = {"max_seq_length": args.max_seq_length, "train": train_stats, "valid": valid_stats}
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Dataset preparation stats:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
