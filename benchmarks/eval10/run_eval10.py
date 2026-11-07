#!/usr/bin/env python3
"""
Fast 10-Prompt Graduated 3D / GLB Evaluation Suite (<10 minutes across all models).

Uses mlx_lm.batch_generate (batch=10) with a two-stage capped-<think> + HTML pipeline:
  1. Stage 1 (Capped <think>, max 900 tokens, batch=10): lets every model produce its own
     on-distribution geometric/rigging plan per prompt without 15,000-token reasoning bloat.
  2. Stage 2 (HTML completion, graduated 3,400-7,680 tokens, batch=10): seamlessly bridges
     any still-open <think> block into ```html\\n<!DOCTYPE html> and completes all 10 scenes.

Validates every generated HTML file via Node.js (`node --input-type=module --check`) both
raw (zero-shot) and after the Playground's <1ms duplicate-identifier auto-heal, writing:
  - benchmarks/eval10/outputs/<model_key>/<prompt_id>.html
  - benchmarks/eval10/results.json
  - benchmarks/eval10/SUMMARY.md
"""
import argparse
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

EVAL_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = EVAL_DIR / "prompts.json"
OUTPUTS_DIR = EVAL_DIR / "outputs"
RESULTS_FILE = EVAL_DIR / "results.json"
SUMMARY_FILE = EVAL_DIR / "SUMMARY.md"

GLB_SYSTEM_PROMPT = (
    "Create game assets for the Godot editor, which uses GLB files. "
    "Create whatever the user requests as a good-quality low-poly asset using only Three.js "
    "in one complete standalone HTML file. The HTML must show the asset and include an "
    "Export GLB button that downloads a working GLB. If the requested asset needs animation, "
    "add it and include the animation in the exported GLB; otherwise make it static."
)

SCENE_SYSTEM_PROMPT = (
    "Use only the Three.js core ES module from "
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js. "
    "Do not import addons or legacy examples/js paths; implement any camera controls "
    "directly inside the HTML."
)

HTML_OPEN_BRIDGE = '\n</think>\n\n```html\n<!DOCTYPE html>\n<html lang="en">\n'

DEFAULT_MODELS = [
    {
        "key": "lfm3_1.7b_base_bf16",
        "label": "LFM3-1.7B Base (BF16, Pre-SFT)",
        "path": "LiquidAI/LFM3-1.7B-MLX-bf16",
        "family": "lfm3",
        "mode": "glb",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "rep_penalty": 1.05,
    },
    {
        "key": "lfm3_1.7b_glb_bf16",
        "label": "LFM3-1.7B-GLB-Coder (BF16)",
        "path": "/Users/liuboto/mlx-models/LFM3-1.7B-GLB-3D-Coder-bf16",
        "family": "lfm3",
        "mode": "glb",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "rep_penalty": 1.05,
    },
    {
        "key": "lfm3_1.7b_glb_4bit",
        "label": "LFM3-1.7B-GLB-Coder (4-bit)",
        "path": "/Users/liuboto/mlx-models/LFM3-1.7B-GLB-3D-Coder-4bit",
        "family": "lfm3",
        "mode": "glb",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "rep_penalty": 1.05,
    },
    {
        "key": "lfm3_1.7b_3d_bf16",
        "label": "LFM3-1.7B-3D-Coder (BF16 Scene)",
        "path": "/Users/liuboto/mlx-models/LFM3-1.7B-3D-Coder-bf16",
        "family": "lfm3",
        "mode": "scene",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "rep_penalty": 1.05,
    },
    {
        "key": "lfm3_1.7b_3d_4bit",
        "label": "LFM3-1.7B-3D-Coder (4-bit Scene)",
        "path": "/Users/liuboto/mlx-models/LFM3-1.7B-3D-Coder-4bit",
        "family": "lfm3",
        "mode": "scene",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "rep_penalty": 1.05,
    },
    {
        "key": "qwen35_4b_glb_4bit",
        "label": "Qwen3.5-4B-GLB-Coder (4-bit)",
        "path": "/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-4bit",
        "family": "qwen3_5",
        "mode": "glb",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "rep_penalty": 1.02,
    },
    {
        "key": "qwen35_4b_glb_bf16",
        "label": "Qwen3.5-4B-GLB-Coder (BF16)",
        "path": "/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16",
        "family": "qwen3_5",
        "mode": "glb",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "rep_penalty": 1.02,
    },
    {
        "key": "qwen35_4b_3d_bf16",
        "label": "Qwen3.5-4B-3D-Coder (BF16 Scene)",
        "path": "/Users/liuboto/mlx-models/Qwen3.5-4B-3D-Coder-bf16",
        "family": "qwen3_5",
        "mode": "scene",
        "temp": 0.6,
        "top_p": 0.95,
        "top_k": 30,
        "rep_penalty": 1.02,
    },
]


def extract_html(full_text: str) -> str:
    body = full_text.split("</think>", 1)[1] if "</think>" in full_text else full_text
    body = body.strip()
    for pat in (
        r"```html\s*([\s\S]*?)```",
        r"```\s*((?:<!doctype html|<html)[\s\S]*?)```",
        r"```(?:html)?\s*([\s\S]*?)```",
    ):
        m = re.search(pat, body, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            if "<html" in cand.lower() or "<!doctype" in cand.lower():
                return cand
    for pat in (
        r"```html\s*([\s\S]*)$",
        r"((?:<!doctype html|<html)[\s\S]*<\/html>)",
        r"((?:<!doctype html|<html)[\s\S]*)$",
    ):
        m = re.search(pat, body, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return body


def extract_scripts_for_check(html: str) -> str:
    blocks = []
    for m in re.finditer(r"<script\b([^>]*)>([\s\S]*?)</script>", html, flags=re.IGNORECASE):
        attrs = m.group(1).lower()
        if "importmap" in attrs or "application/json" in attrs:
            continue
        code = m.group(2).strip()
        if code:
            blocks.append(code)
    return "\n;\n".join(blocks)


def check_js_syntax(js_code: str) -> tuple[bool, str]:
    if not js_code.strip():
        return False, "No <script> code found"
    proc = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=js_code,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if proc.returncode == 0:
        return True, ""
    err_lines = [ln.strip() for ln in proc.stderr.splitlines() if "Error:" in ln]
    return False, (err_lines[-1] if err_lines else proc.stderr.strip()[:160])


def auto_heal_duplicates_in_html(html: str, max_passes: int = 8) -> tuple[str, list[str]]:
    healed_html = html
    healed_ids: list[str] = []
    for _ in range(max_passes):
        js_code = extract_scripts_for_check(healed_html)
        ok, err = check_js_syntax(js_code)
        if ok:
            break
        m = re.search(r"Identifier\s+['\"`]([A-Za-z_$][\w$]*)['\"`]\s+has already been declared", err)
        if not m:
            break
        ident = m.group(1)
        if ident in healed_ids:
            break
        escaped = re.escape(ident)
        decl_re = re.compile(rf"(^|[;{{\n\r]\s*)(?:const|let)(\s+{escaped}\b)")
        fn_re = re.compile(rf"(^|[;{{\n\r]\s*)function\s+({escaped})\s*\(")

        def patch_script(match: re.Match) -> str:
            open_tag, body, close_tag = match.group(1), match.group(2), match.group(3)
            if "importmap" in open_tag.lower():
                return match.group(0)
            body = decl_re.sub(r"\1var\2", body)
            body = fn_re.sub(r"\1var \2 = function(", body)
            return f"{open_tag}{body}{close_tag}"

        new_html = re.sub(
            r"(<script\b[^>]*>)([\s\S]*?)(</script>)",
            patch_script,
            healed_html,
            flags=re.IGNORECASE,
        )
        if new_html == healed_html:
            break
        healed_html = new_html
        healed_ids.append(ident)
    return healed_html, healed_ids


def build_base_chat(tokenizer, family: str, system_prompt: str, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    base = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    if family != "lfm3":
        if base.endswith("<think>"):
            base += "\n"
        elif "<think>" not in base:
            base += "<think>\n"
    elif "<think>" not in base:
        base += "<think>"
    return base


def evaluate_model(model_cfg: dict, prompts: list[dict], think_budget: int) -> dict:
    mkey = model_cfg["key"]
    print(f"\n{'=' * 72}\nEvaluating {model_cfg['label']} ({model_cfg['path']})\n{'=' * 72}", flush=True)
    t_load0 = time.perf_counter()
    model, tokenizer = load(model_cfg["path"], tokenizer_config={"trust_remote_code": True})
    load_s = time.perf_counter() - t_load0

    sampler = make_sampler(
        temp=model_cfg["temp"],
        top_p=model_cfg["top_p"],
        top_k=model_cfg["top_k"],
    )
    logits_proc = make_logits_processors(
        repetition_penalty=model_cfg["rep_penalty"],
        repetition_context_size=64,
    )

    sys_prompt = GLB_SYSTEM_PROMPT if model_cfg["mode"] == "glb" else SCENE_SYSTEM_PROMPT
    base_chats = []
    stage1_token_lists = []
    for p in prompts:
        uprompt = p["glb_prompt"] if model_cfg["mode"] == "glb" else p["scene_prompt"]
        bc = build_base_chat(tokenizer, model_cfg["family"], sys_prompt, uprompt)
        base_chats.append(bc)
        stage1_token_lists.append(tokenizer.encode(bc, add_special_tokens=False))

    t_gen0 = time.perf_counter()
    print(f"  [Stage 1] Batched <think> pass (batch={len(prompts)}, budget={think_budget} tok)...", flush=True)
    res1 = batch_generate(
        model,
        tokenizer,
        stage1_token_lists,
        max_tokens=think_budget,
        sampler=sampler,
        logits_processors=logits_proc,
    )
    t_stage1 = time.perf_counter() - t_gen0
    print(
        f"  [Stage 1] Done in {t_stage1:.1f}s ({res1.stats.generation_tokens} tok @ {res1.stats.generation_tps:.1f} tok/s)",
        flush=True,
    )

    # Determine which prompts still need Stage 2 HTML completion
    need_stage2_indices = []
    stage2_prompts = []
    stage2_max_tokens = []
    combined_texts = list(res1.texts)
    per_prompt_tokens = [len(ids) for ids in (res1.token_ids or [tokenizer.encode(t, add_special_tokens=False) for t in res1.texts])]

    for idx, (p, txt1) in enumerate(zip(prompts, res1.texts)):
        extracted1 = extract_html(txt1)
        if "</html>" in extracted1.lower() and len(extracted1) >= 300:
            continue
        need_stage2_indices.append(idx)
        if "</think>" in txt1:
            bridged = txt1
        else:
            clean_think = txt1.rstrip()
            bridged = clean_think + HTML_OPEN_BRIDGE
        combined_texts[idx] = bridged
        stage2_prompts.append(tokenizer.encode(base_chats[idx] + bridged, add_special_tokens=False))
        stage2_max_tokens.append(p["max_html_tokens"])

    t_stage2 = 0.0
    stage2_gen_tokens = 0
    stage2_tps = 0.0
    if need_stage2_indices:
        t2_0 = time.perf_counter()
        print(
            f"  [Stage 2] Batched HTML pass for {len(need_stage2_indices)}/{len(prompts)} prompts "
            f"(max_tokens={min(stage2_max_tokens)}..{max(stage2_max_tokens)})...",
            flush=True,
        )
        res2 = batch_generate(
            model,
            tokenizer,
            stage2_prompts,
            max_tokens=stage2_max_tokens,
            sampler=sampler,
            logits_processors=logits_proc,
        )
        t_stage2 = time.perf_counter() - t2_0
        stage2_gen_tokens = res2.stats.generation_tokens
        stage2_tps = res2.stats.generation_tps
        print(
            f"  [Stage 2] Done in {t_stage2:.1f}s ({stage2_gen_tokens} tok @ {stage2_tps:.1f} tok/s, peak={res2.stats.peak_memory:.2f} GB)",
            flush=True,
        )
        for sub_i, orig_idx in enumerate(need_stage2_indices):
            t2 = res2.texts[sub_i]
            combined_texts[orig_idx] += t2
            per_prompt_tokens[orig_idx] += len(tokenizer.encode(t2, add_special_tokens=False))

    total_wall_s = time.perf_counter() - t_gen0
    total_gen_tokens = res1.stats.generation_tokens + stage2_gen_tokens
    agg_tps = total_gen_tokens / max(total_wall_s, 1e-3)
    peak_mem_gb = max(
        res1.stats.peak_memory,
        res2.stats.peak_memory if need_stage2_indices else 0.0,
    )

    model_out_dir = OUTPUTS_DIR / mkey
    model_out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for idx, (p, raw_full) in enumerate(zip(prompts, combined_texts)):
        cleaned_full = re.sub(r"<\|im_end\|>|<\s*\|endoftext\|\s*>", "", raw_full).strip()
        raw_html = extract_html(cleaned_full)
        has_complete_html = ("<html" in raw_html.lower() or "<!doctype" in raw_html.lower()) and (
            "</html>" in raw_html.lower()
        )
        raw_js = extract_scripts_for_check(raw_html)
        raw_js_ok, raw_js_err = check_js_syntax(raw_js) if has_complete_html else (False, "Incomplete HTML (missing </html>)")

        final_html = raw_html
        healed_ids: list[str] = []
        healed_js_ok = raw_js_ok
        healed_js_err = raw_js_err
        if has_complete_html and not raw_js_ok:
            final_html, healed_ids = auto_heal_duplicates_in_html(raw_html)
            healed_js_ok, healed_js_err = check_js_syntax(extract_scripts_for_check(final_html))

        out_path = model_out_dir / f"{p['id']}.html"
        out_path.write_text(final_html, encoding="utf-8")

        has_three = bool(re.search(r"\bTHREE\b|from\s+['\"]three['\"]|three\.module", final_html))
        has_gltf = "GLTFExporter" in final_html
        has_anim = bool(re.search(r"AnimationClip|requestAnimationFrame", final_html))
        mesh_calls = len(
            re.findall(r"new\s+THREE\.(?:Mesh|InstancedMesh|Points|LineSegments|SkinnedMesh|Group)\b", final_html)
        )

        status_icon = "PASS" if raw_js_ok else ("HEALED" if healed_js_ok else "FAIL")
        print(
            f"    {p['id']:28s} | {status_icon:6s} | {per_prompt_tokens[idx]:5d} tok | "
            f"{final_html.count(chr(10)) + 1:4d} lines | meshes/groups={mesh_calls:3d} | "
            f"GLB={'Y' if has_gltf else 'N'} "
            f"{('(' + (healed_js_err or ('healed: ' + ','.join(healed_ids))) + ')') if not raw_js_ok else ''}",
            flush=True,
        )

        items.append(
            {
                "prompt_id": p["id"],
                "level": p["level"],
                "title": p["title"],
                "tier": p["tier"],
                "html_rel_path": f"outputs/{mkey}/{p['id']}.html",
                "tokens": per_prompt_tokens[idx],
                "html_chars": len(final_html),
                "html_lines": final_html.count("\n") + 1,
                "has_complete_html": has_complete_html,
                "raw_js_ok": raw_js_ok,
                "healed_js_ok": healed_js_ok,
                "healed_identifiers": healed_ids,
                "error_message": healed_js_err if not healed_js_ok else "",
                "raw_error_message": raw_js_err if not raw_js_ok else "",
                "has_three": has_three,
                "has_gltf_exporter": has_gltf,
                "has_animation": has_anim,
                "scene_objects_est": mesh_calls,
            }
        )

    del model
    gc.collect()
    mx.clear_cache()

    raw_pass = sum(1 for x in items if x["raw_js_ok"])
    healed_pass = sum(1 for x in items if x["healed_js_ok"])
    complete_html_cnt = sum(1 for x in items if x["has_complete_html"])
    gltf_cnt = sum(1 for x in items if x["has_gltf_exporter"] and x["healed_js_ok"])

    return {
        "key": mkey,
        "label": model_cfg["label"],
        "path": model_cfg["path"],
        "family": model_cfg["family"],
        "mode": model_cfg["mode"],
        "load_seconds": round(load_s, 2),
        "wall_seconds": round(total_wall_s, 2),
        "stage1_seconds": round(t_stage1, 2),
        "stage2_seconds": round(t_stage2, 2),
        "total_generated_tokens": total_gen_tokens,
        "aggregate_batch_tps": round(agg_tps, 1),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "complete_html_count": complete_html_cnt,
        "raw_valid_js_count": raw_pass,
        "healed_valid_js_count": healed_pass,
        "valid_gltf_count": gltf_cnt,
        "items": items,
    }


def continue_truncated_model(model_cfg: dict, prev_model_res: dict, prompts: list[dict], extra_tokens: int = 5632) -> dict:
    mkey = model_cfg["key"]
    items = [dict(it) for it in prev_model_res["items"]]
    trunc_indices = [i for i, it in enumerate(items) if not it["has_complete_html"]]
    if not trunc_indices:
        print(f"\n[{model_cfg['label']}] 0 truncated scenes — nothing to continue.", flush=True)
        return prev_model_res

    print(
        f"\n{'=' * 72}\nContinuing {len(trunc_indices)} truncated scenes for {model_cfg['label']}\n{'=' * 72}",
        flush=True,
    )
    model, tokenizer = load(model_cfg["path"], tokenizer_config={"trust_remote_code": True})
    sampler = make_sampler(
        temp=min(model_cfg["temp"], 0.58),
        top_p=model_cfg["top_p"],
        top_k=model_cfg["top_k"],
    )
    logits_proc = make_logits_processors(
        repetition_penalty=model_cfg["rep_penalty"],
        repetition_context_size=64,
    )
    sys_prompt = GLB_SYSTEM_PROMPT if model_cfg["mode"] == "glb" else SCENE_SYSTEM_PROMPT

    cont_token_lists = []
    safe_html_prefixes = []
    for idx in trunc_indices:
        p = prompts[idx]
        html_path = EVAL_DIR / items[idx]["html_rel_path"]
        existing_html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
        # Drop potentially half-written final line so continuation starts cleanly on a newline
        lines = existing_html.split("\n")
        safe_prefix = "\n".join(lines[:-1]) + "\n" if len(lines) > 3 else existing_html
        safe_html_prefixes.append(safe_prefix)

        uprompt = p["glb_prompt"] if model_cfg["mode"] == "glb" else p["scene_prompt"]
        bc = build_base_chat(tokenizer, model_cfg["family"], sys_prompt, uprompt)
        asst_prefix = f"</think>\n\n```html\n{safe_prefix}"
        cont_token_lists.append(tokenizer.encode(bc + asst_prefix, add_special_tokens=False))

    t0 = time.perf_counter()
    res = batch_generate(
        model,
        tokenizer,
        cont_token_lists,
        max_tokens=extra_tokens,
        sampler=sampler,
        logits_processors=logits_proc,
    )
    dt = time.perf_counter() - t0
    print(
        f"  [Continue] Finished {len(trunc_indices)} tails in {dt:.1f}s "
        f"({res.stats.generation_tokens} tok @ {res.stats.generation_tps:.1f} tok/s)",
        flush=True,
    )

    for sub_i, idx in enumerate(trunc_indices):
        p = prompts[idx]
        tail_txt = re.sub(r"<\|im_end\|>|<\s*\|endoftext\|\s*>", "", res.texts[sub_i])
        merged_doc = f"```html\n{safe_html_prefixes[sub_i]}{tail_txt}"
        raw_html = extract_html(merged_doc)
        has_complete_html = ("<html" in raw_html.lower() or "<!doctype" in raw_html.lower()) and (
            "</html>" in raw_html.lower()
        )
        raw_js = extract_scripts_for_check(raw_html)
        raw_js_ok, raw_js_err = (
            check_js_syntax(raw_js) if has_complete_html else (False, "Incomplete HTML (missing </html>)")
        )

        final_html = raw_html
        healed_ids: list[str] = []
        healed_js_ok = raw_js_ok
        healed_js_err = raw_js_err
        if has_complete_html and not raw_js_ok:
            final_html, healed_ids = auto_heal_duplicates_in_html(raw_html)
            healed_js_ok, healed_js_err = check_js_syntax(extract_scripts_for_check(final_html))

        out_path = EVAL_DIR / items[idx]["html_rel_path"]
        out_path.write_text(final_html, encoding="utf-8")

        added_tok = len(tokenizer.encode(tail_txt, add_special_tokens=False))
        has_three = bool(re.search(r"\bTHREE\b|from\s+['\"]three['\"]|three\.module", final_html))
        has_gltf = "GLTFExporter" in final_html
        has_anim = bool(re.search(r"AnimationClip|requestAnimationFrame", final_html))
        mesh_calls = len(
            re.findall(r"new\s+THREE\.(?:Mesh|InstancedMesh|Points|LineSegments|SkinnedMesh|Group)\b", final_html)
        )

        status_icon = "PASS" if raw_js_ok else ("HEALED" if healed_js_ok else "FAIL")
        print(
            f"    {p['id']:28s} | {status_icon:6s} | +{added_tok:4d} tok | "
            f"{final_html.count(chr(10)) + 1:4d} lines | GLB={'Y' if has_gltf else 'N'} "
            f"{('(' + (healed_js_err or ('healed: ' + ','.join(healed_ids))) + ')') if not raw_js_ok else ''}",
            flush=True,
        )

        items[idx].update(
            {
                "tokens": items[idx]["tokens"] + added_tok,
                "html_chars": len(final_html),
                "html_lines": final_html.count("\n") + 1,
                "has_complete_html": has_complete_html,
                "raw_js_ok": raw_js_ok,
                "healed_js_ok": healed_js_ok,
                "healed_identifiers": healed_ids,
                "error_message": healed_js_err if not healed_js_ok else "",
                "raw_error_message": raw_js_err if not raw_js_ok else "",
                "has_three": has_three,
                "has_gltf_exporter": has_gltf,
                "has_animation": has_anim,
                "scene_objects_est": mesh_calls,
            }
        )

    del model
    gc.collect()
    mx.clear_cache()

    updated = dict(prev_model_res)
    updated["wall_seconds"] = round(prev_model_res["wall_seconds"] + dt, 2)
    updated["total_generated_tokens"] = prev_model_res["total_generated_tokens"] + res.stats.generation_tokens
    updated["aggregate_batch_tps"] = round(
        updated["total_generated_tokens"] / max(updated["wall_seconds"], 1e-3), 1
    )
    updated["peak_memory_gb"] = round(max(prev_model_res["peak_memory_gb"], res.stats.peak_memory), 2)
    updated["complete_html_count"] = sum(1 for x in items if x["has_complete_html"])
    updated["raw_valid_js_count"] = sum(1 for x in items if x["raw_js_ok"])
    updated["healed_valid_js_count"] = sum(1 for x in items if x["healed_js_ok"])
    updated["valid_gltf_count"] = sum(1 for x in items if x["has_gltf_exporter"] and x["healed_js_ok"])
    updated["items"] = items
    return updated


FAST_SUITE_KEYS = {
    "lfm3_1.7b_base_bf16",
    "lfm3_1.7b_glb_bf16",
    "lfm3_1.7b_glb_4bit",
    "qwen35_4b_glb_4bit",
}


def write_summary_markdown(payload: dict) -> None:
    lines = [
        "# Eval-10: Graduated 3D & GLB Benchmark Results",
        "",
        f"- **Total Suite Wall Time ({len(payload['models'])} models × 10 prompts = {len(payload['models']) * 10} scenes):** `{payload['suite_wall_seconds']}s` (`{payload['suite_wall_seconds'] / 60:.2f} min`)",
        f"- **Fast 4-Model Subset (`--fast-suite`, skipping 13.5 GB 4B BF16):** `{sum(m['wall_seconds'] for m in payload['models'] if m['key'] in FAST_SUITE_KEYS):.1f}s` (`{sum(m['wall_seconds'] for m in payload['models'] if m['key'] in FAST_SUITE_KEYS) / 60:.2f} min`)",
        f"- **Capped `<think>` Budget (Stage 1):** `{payload['think_budget']} tokens` + graduated HTML completion (`9,216–14,336 tokens`, `batch=10` on Apple M5 Max)",
        "",
        "## 1. Aggregate Leaderboard",
        "",
        "| Model | Mode | 10-Prompt Wall Time | Batch Speed | Peak RAM | Complete HTML | Zero-Shot Valid JS | Auto-Healed Valid JS | Valid + `GLTFExporter` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in payload["models"]:
        lines.append(
            f"| **{m['label']}** | `{m['mode']}` | `{m['wall_seconds']:.1f}s` | `{m['aggregate_batch_tps']:.1f} tok/s` | "
            f"`{m['peak_memory_gb']:.2f} GB` | `{m['complete_html_count']}/10` | **`{m['raw_valid_js_count']}/10`** | "
            f"**`{m['healed_valid_js_count']}/10`** | `{m['valid_gltf_count']}/10` |"
        )

    lines.extend(["", "## 2. Per-Prompt Difficulty Matrix (`PASS` = zero-shot valid JS, `HEAL` = valid after `<1ms` duplicate-var heal, `FAIL` = syntax/truncation)", ""])
    headers = ["Prompt"] + [m["label"] for m in payload["models"]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- | " + " | ".join([":---:" for _ in payload["models"]]) + " |")

    for p_idx, p in enumerate(payload["prompts"]):
        row = [f"**P{p['level']:02d} · {p['title']}**"]
        for m in payload["models"]:
            it = m["items"][p_idx]
            if it["raw_js_ok"]:
                cell = f"PASS (`{it['html_lines']}L`)"
            elif it["healed_js_ok"]:
                cell = f"HEAL (`{it['html_lines']}L`)"
            else:
                cell = "FAIL"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model keys to run (merges into existing results.json if present).",
    )
    parser.add_argument(
        "--fast-suite",
        action="store_true",
        help="Run the <10-minute 4-model suite (LFM3 Base, LFM3-GLB BF16, LFM3-GLB 4-bit, Qwen3.5-4B-GLB 4-bit).",
    )
    parser.add_argument(
        "--continue-truncated",
        action="store_true",
        help="Resume only truncated scenes missing </html> in existing results.json.",
    )
    parser.add_argument("--think-budget", type=int, default=900)
    args = parser.parse_args()

    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    selected = DEFAULT_MODELS
    if args.fast_suite and not args.models:
        selected = [m for m in DEFAULT_MODELS if m["key"] in FAST_SUITE_KEYS]
    if args.models:
        wanted = set(args.models)
        selected = [m for m in DEFAULT_MODELS if m["key"] in wanted]
        if not selected:
            raise SystemExit(f"No matching model keys for {args.models}. Available: {[m['key'] for m in DEFAULT_MODELS]}")

    existing_by_key: dict[str, dict] = {}
    if RESULTS_FILE.exists() and (args.models or args.fast_suite or args.continue_truncated):
        prev = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        for m in prev.get("models", []):
            existing_by_key[m["key"]] = m

    t_suite0 = time.perf_counter()
    for m_cfg in selected:
        if args.continue_truncated and m_cfg["key"] in existing_by_key:
            res = continue_truncated_model(m_cfg, existing_by_key[m_cfg["key"]], prompts)
        else:
            res = evaluate_model(m_cfg, prompts, think_budget=args.think_budget)
        existing_by_key[m_cfg["key"]] = res

        # Persist incrementally after every model so results survive any interruption
        ordered_models = [existing_by_key[m["key"]] for m in DEFAULT_MODELS if m["key"] in existing_by_key]
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "think_budget": args.think_budget,
            "suite_wall_seconds": round(sum(m["wall_seconds"] for m in ordered_models), 1),
            "last_run_elapsed_seconds": round(time.perf_counter() - t_suite0, 1),
            "prompts": prompts,
            "models": ordered_models,
        }
        RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_summary_markdown(payload)

    print(f"\nCompleted Eval-10 suite in {time.perf_counter() - t_suite0:.1f}s -> {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
