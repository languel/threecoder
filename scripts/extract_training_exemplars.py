#!/usr/bin/env python3
"""
Extract curated gold exemplar scenes from ccore/ThreeJS-3D-Scenes validation split.
Provides ground-truth visual references for what high-quality training scenes look like.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks" / "training_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDICES = [0, 3, 4, 7, 8, 13]
SLUGS = [
    "T01_glassblowing_workshop",
    "T02_formal_garden_canal",
    "T03_prayer_wheel_bank",
    "T04_mountain_locomotive",
    "T05_timber_market_town",
    "T06_bohr_oxygen_atom"
]
TITLES = [
    "Glassblowing Workshop",
    "Formal Garden & Canal",
    "Prayer Wheel Bank",
    "Mountain Rack Locomotive",
    "Timber-Framed Market Town",
    "Bohr-Style Oxygen Atom"
]

exemplars = []
source_file = ROOT / "data" / "lfm3_scene" / "valid.jsonl"

with open(source_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i in INDICES:
            idx = INDICES.index(i)
            slug = SLUGS[idx]
            title = TITLES[idx]
            data = json.loads(line)
            msgs = data["messages"]
            prompt = [m["content"] for m in msgs if m["role"] == "user"][0]
            code = [m["content"] for m in msgs if m["role"] == "assistant"][0]

            html = code
            fence = "```html"
            if fence in html:
                html = html.split(fence, 1)[1]
            end_fence = "```"
            if end_fence in html:
                html = html.rsplit(end_fence, 1)[0]
            html = html.strip()

            target_path = OUT_DIR / f"{slug}.html"
            with open(target_path, "w", encoding="utf-8") as out_f:
                out_f.write(html)

            exemplars.append({
                "id": slug,
                "title": title,
                "prompt": prompt,
                "rel_path": f"/benchmarks/training_set/{slug}.html",
                "lines": len(html.splitlines()),
                "bytes": len(html),
                "chars": len(html)
            })
            print(f"Extracted {slug}: {len(html.splitlines())} lines ({len(html)} bytes)")

with open(OUT_DIR / "exemplars.json", "w", encoding="utf-8") as out_meta:
    json.dump(exemplars, out_meta, indent=2)

print(f"Successfully extracted {len(exemplars)} gold training-set scenes to {OUT_DIR}")
