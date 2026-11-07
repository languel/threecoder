#!/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python
import json
import os
import re
import subprocess
import tempfile
import time

import mlx_lm
from mlx_lm.sample_utils import make_logits_processors, make_sampler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_4BIT = "/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-4bit"

GLB_SYSTEM = (
    "Create game assets for the Godot editor, which uses GLB files. "
    "Create whatever the user requests as a good-quality low-poly asset using only Three.js "
    "in one complete standalone HTML file. The HTML must show the asset and include an "
    "Export GLB button that downloads a working GLB. If the requested asset needs animation, "
    "add it and include the animation in the exported GLB; otherwise make it static."
)

SNOWGLOBE_PROMPT = (
    "Create a small wintry scene inside a snowglobe. baba yaga hut in a small clearing. "
    "eastern foklore, a fox, a raven and an oul and chimney smoke. dusk, starry sky, fluffy snow."
)

OWL_ANIM_PROMPT = (
    "Create an animated clockwork brass owl companion perched on a small ornate gear pedestal. "
    "Give it layered copper and brass feathers, glowing amber glass eyes, and exposed spinning clockwork gears on its chest. "
    "Include a smooth looping idle animation in the exported GLB with wing flutter, head tilt, and rotating gears."
)


def extract_html(text: str) -> str:
    fenced = re.findall(r"```html\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    m = re.search(r"(<!doctype\s+html[\s\S]*?</html>|<html[\s\S]*?</html>)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def check_js_syntax(html: str) -> tuple[bool, str]:
    scripts = re.findall(
        r"<script(?![^>]*type=['\"]importmap['\"])[^>]*>([\s\S]*?)</script>",
        html,
        flags=re.IGNORECASE,
    )
    if not scripts:
        return False, "No <script> block found"
    js_code = "\n".join(scripts)
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as tmp:
        tmp.write(js_code)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(["node", "--check", tmp_path], capture_output=True, text=True)
        if proc.returncode == 0:
            return True, "OK"
        return False, (proc.stderr or proc.stdout).strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def has_repetition_loop(tail: str) -> bool:
    if len(tail) < 1200:
        return False
    for pat_len in (120, 200, 320):
        pat = tail[-pat_len:]
        if tail[:-pat_len].count(pat) >= 3:
            return True
    return False


def run_case(model, tokenizer, slug: str, prompt: str, enable_thinking: bool, max_tokens: int = 32768) -> dict:
    print(f"\n=== Running {slug} (enable_thinking={enable_thinking}) ===", flush=True)
    t0 = time.time()
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": GLB_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    sampler = make_sampler(temp=0.8, top_p=0.95, min_p=0.0, top_k=40)
    logits_processors = make_logits_processors(repetition_penalty=1.02, repetition_context_size=128)

    pieces = []
    tok_count = 0
    gen_tps = 0.0
    peak_mem_gb = 0.0
    stop_reason = "eos"

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        pieces.append(resp.text)
        tok_count = resp.generation_tokens
        gen_tps = resp.generation_tps
        peak_mem_gb = resp.peak_memory

        if tok_count % 250 == 0:
            tail = "".join(pieces[-600:])
            if has_repetition_loop(tail):
                stop_reason = "repetition_loop"
                print(f"  [{slug}] Early stop at {tok_count} tok due to repetition loop!", flush=True)
                break
            joined_tail = "".join(pieces[-80:])
            if "</html>" in joined_tail.lower():
                full_check = "".join(pieces)
                body_part = full_check.split("</think>", 1)[-1] if "</think>" in full_check else full_check
                if "</html>" in body_part.lower() and body_part.lower().rfind("```") > body_part.lower().rfind("</html>"):
                    stop_reason = "html_complete"
                    break

        if tok_count % 1000 == 0:
            elapsed = time.time() - t0
            joined = "".join(pieces)
            phase = "writing HTML" if (not enable_thinking or "</think>" in joined) else "thinking"
            print(f"  [{slug}] {tok_count} tok ({elapsed:.1f}s, {gen_tps:.1f} tok/s, {phase})", flush=True)

    wall_s = time.time() - t0
    full_text = "".join(pieces)
    if enable_thinking and "</think>" in full_text:
        thinking_text, content_text = full_text.split("</think>", 1)
    else:
        thinking_text, content_text = "", full_text

    html = extract_html(content_text)
    html_path = os.path.join(OUT_DIR, f"{slug}.html")
    raw_path = os.path.join(OUT_DIR, f"{slug}.raw.md")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- THINKING ({len(thinking_text)} chars, stop={stop_reason}) -->\n\n{content_text}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    js_valid, js_msg = check_js_syntax(html)
    has_exporter = "GLTFExporter" in html
    has_anim_clip = "AnimationClip" in html

    res = {
        "slug": slug,
        "quantization": "q4f16 (MLX 4-bit, 4.50 bpw)",
        "enable_thinking": enable_thinking,
        "stop_reason": stop_reason,
        "eval_tokens": tok_count,
        "tok_per_sec": round(gen_tps, 1),
        "wall_seconds": round(wall_s, 1),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "thinking_chars": len(thinking_text),
        "html_lines": len(html.splitlines()),
        "has_closing_html": "</html>" in html.lower(),
        "js_valid": js_valid,
        "js_check_detail": js_msg,
        "has_gltf_exporter": has_exporter,
        "has_animation_clip": has_anim_clip,
        "html_path": html_path,
    }
    print(
        f"=== Done {slug} ({stop_reason}): {tok_count} tok @ {res['tok_per_sec']} tok/s ({res['wall_seconds']}s) | "
        f"peak={res['peak_memory_gb']} GB | lines={res['html_lines']} | JS valid={js_valid} | "
        f"GLTFExporter={has_exporter} | AnimationClip={has_anim_clip} ===",
        flush=True,
    )
    if not js_valid:
        print(f"  JS Error: {js_msg[:300]}", flush=True)
    return res


def main():
    print(f"Loading 4-bit model from {MODEL_4BIT}...", flush=True)
    model, tokenizer = mlx_lm.load(MODEL_4BIT)
    results = []
    out_json = os.path.join(OUT_DIR, "quant_results.json")

    for slug, prompt, think, max_t in [
        ("snowglobe_glb_q4f16_think", SNOWGLOBE_PROMPT, True, 32768),
        ("clockwork_owl_glb_q4f16_think", OWL_ANIM_PROMPT, True, 32768),
        ("clockwork_owl_glb_q4f16_nothink", OWL_ANIM_PROMPT, False, 12288),
    ]:
        results.append(run_case(model, tokenizer, slug, prompt, enable_thinking=think, max_tokens=max_t))
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\nSaved all results to", out_json, flush=True)


if __name__ == "__main__":
    main()
