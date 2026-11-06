#!/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python
import json
import os
import re
import time
import urllib.request

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

USER_IDEA = (
    "a small wintry scene inside a snowglobe. baba yaga hut in a small clearing. "
    "eastern foklore, a fox, a raven and an oul and chimney smoke. dusk, starry sky, fluffy snow."
)

SCENE_PROMPT = f"Build a self-contained HTML scene with Three.js from cdnjs featuring {USER_IDEA}"
GLB_PROMPT = f"Create {USER_IDEA}"

GLB_SYSTEM = (
    "Create game assets for the Godot editor, which uses GLB files. "
    "Create whatever the user requests as a good-quality low-poly asset using only Three.js "
    "in one complete standalone HTML file. The HTML must show the asset and include an "
    "Export GLB button that downloads a working GLB. If the requested asset needs animation, "
    "add it and include the animation in the exported GLB; otherwise make it static."
)


def extract_html(text: str) -> str:
    fenced = re.findall(r"```html\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    m = re.search(r"(<!doctype\s+html[\s\S]*?</html>|<html[\s\S]*?</html>)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def unload_ollama_model(model_name: str) -> None:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps({"model": model_name, "messages": [], "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print(f"[warn] unload {model_name}: {e}", flush=True)


def run_ollama(slug: str, model_name: str, prompt: str) -> dict:
    print(f"\n=== Starting {slug} ({model_name}) ===", flush=True)
    t0 = time.time()
    payload = {
        "model": model_name,
        "stream": True,
        "options": {"num_predict": 65536},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    thinking_parts = []
    content_parts = []
    chunks = 0
    eval_count = 0
    eval_duration_ns = 0

    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            obj = json.loads(raw_line.decode("utf-8"))
            msg = obj.get("message", {})
            if msg.get("thinking"):
                thinking_parts.append(msg["thinking"])
            if msg.get("content"):
                content_parts.append(msg["content"])
            chunks += 1
            if chunks % 2000 == 0:
                elapsed = time.time() - t0
                phase = "writing HTML" if content_parts else "thinking"
                print(f"  [{slug}] ~{chunks} chunks ({elapsed:.1f}s, {phase})", flush=True)
            if obj.get("done"):
                eval_count = obj.get("eval_count", chunks)
                eval_duration_ns = obj.get("eval_duration", 0)
                break

    wall_s = time.time() - t0
    tok_s = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else (eval_count / max(wall_s, 0.001))
    thinking_text = "".join(thinking_parts)
    content_text = "".join(content_parts)
    html = extract_html(content_text)

    raw_path = os.path.join(OUT_DIR, f"{slug}.raw.md")
    html_path = os.path.join(OUT_DIR, f"{slug}.html")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- THINKING ({len(thinking_text)} chars) -->\n\n{content_text}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    unload_ollama_model(model_name)

    res = {
        "slug": slug,
        "engine": model_name,
        "eval_tokens": eval_count,
        "tok_per_sec": round(tok_s, 1),
        "wall_seconds": round(wall_s, 1),
        "thinking_chars": len(thinking_text),
        "html_lines": len(html.splitlines()),
        "html_bytes": len(html.encode("utf-8")),
        "has_closing_html": "</html>" in html.lower(),
        "html_path": html_path,
    }
    print(f"=== Finished {slug}: {res['eval_tokens']} tok @ {res['tok_per_sec']} tok/s ({res['wall_seconds']}s), HTML lines={res['html_lines']}, complete={res['has_closing_html']} ===", flush=True)
    return res


def run_standalone_mlx(slug: str, model_dir: str, system_prompt: str, user_prompt: str) -> dict:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    print(f"\n=== Starting {slug} (standalone mlx_lm: {model_dir}) ===", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_dir)
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=0.8, top_p=0.95, min_p=0.0, top_k=40)

    pieces = []
    gen_tps = 0.0
    tok_count = 0
    peak_mem_gb = 0.0

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=65536,
        sampler=sampler,
    ):
        pieces.append(resp.text)
        tok_count = resp.generation_tokens
        gen_tps = resp.generation_tps
        peak_mem_gb = resp.peak_memory
        if tok_count % 2000 == 0:
            elapsed = time.time() - t0
            phase = "writing HTML" if "</think>" in "".join(pieces) else "thinking"
            print(f"  [{slug}] {tok_count} tokens ({elapsed:.1f}s, {gen_tps:.1f} tok/s, {phase})", flush=True)

    wall_s = time.time() - t0
    full_text = "".join(pieces)
    if "</think>" in full_text:
        thinking_text, content_text = full_text.split("</think>", 1)
    else:
        thinking_text, content_text = "", full_text

    html = extract_html(content_text)
    raw_path = os.path.join(OUT_DIR, f"{slug}.raw.md")
    html_path = os.path.join(OUT_DIR, f"{slug}.html")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- THINKING ({len(thinking_text)} chars) -->\n\n{content_text}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    res = {
        "slug": slug,
        "engine": "mlx_lm (standalone BF16)",
        "eval_tokens": tok_count,
        "tok_per_sec": round(gen_tps, 1),
        "wall_seconds": round(wall_s, 1),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "thinking_chars": len(thinking_text),
        "html_lines": len(html.splitlines()),
        "html_bytes": len(html.encode("utf-8")),
        "has_closing_html": "</html>" in html.lower(),
        "html_path": html_path,
    }
    print(f"=== Finished {slug}: {res['eval_tokens']} tok @ {res['tok_per_sec']} tok/s ({res['wall_seconds']}s), HTML lines={res['html_lines']}, complete={res['has_closing_html']} ===", flush=True)
    return res


def main():
    results = []
    results.append(run_ollama("snowglobe_3d_ollama_q8", "ccore-3d-coder", SCENE_PROMPT))
    results.append(run_ollama("snowglobe_3d_ollama_mlx", "ccore-3d-coder-mlx", SCENE_PROMPT))
    results.append(run_ollama("snowglobe_glb_ollama_q8", "ccore-glb-coder", GLB_PROMPT))
    results.append(
        run_standalone_mlx(
            "snowglobe_glb_mlx_lm",
            "/Users/liuboto/mlx-models/Qwen3.5-4B-GLB-3D-Coder-bf16",
            GLB_SYSTEM,
            GLB_PROMPT,
        )
    )

    summary_path = os.path.join(OUT_DIR, "benchmark_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nALL DONE! Summary written to", summary_path, flush=True)


if __name__ == "__main__":
    main()
