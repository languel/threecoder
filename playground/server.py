#!/usr/bin/env python3
"""
Local server for the ThreeCoder WebGPU Browser Playground & Telemetry Flywheel.
- Serves playground static assets from ./playground
- Serves local ONNX WebGPU checkpoints from /models/* (-> ~/onnx-models/*) with HTTP Range (206) support
- Serves existing benchmark outputs read-only from /benchmarks/*
- Captures opt-in user telemetry, ratings, runtime errors, and GLB export events into ./playground/telemetry/events.jsonl
"""
import argparse
import base64
import datetime
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

PLAYGROUND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLAYGROUND_DIR)
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "benchmarks")
ONNX_MODELS_DIR = os.path.expanduser("~/onnx-models")
TELEMETRY_DIR = os.path.join(PLAYGROUND_DIR, "telemetry")
TELEMETRY_FILE = os.path.join(TELEMETRY_DIR, "events.jsonl")

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/octet-stream", ".onnx_data")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")


def safe_join(base_dir: str, rel_path: str) -> str | None:
    norm_base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(norm_base, rel_path.lstrip("/")))
    if target == norm_base or target.startswith(norm_base + os.sep):
        return target
    return None


class PlaygroundHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def add_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Permissions-Policy",
            "camera=(), geolocation=(), microphone=(), focus-without-user-activation=()",
        )
        self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Vary", "Sec-Fetch-Dest, Sec-Fetch-Mode, Sec-Fetch-Site")
        self.send_header("Accept-Ranges", "bytes")

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.add_security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def resolve_local_file(self, url_path: str) -> str | None:
        clean = unquote(url_path)
        if clean == "/" or clean == "":
            return os.path.join(PLAYGROUND_DIR, "index.html")
        if clean.startswith("/models/"):
            return safe_join(ONNX_MODELS_DIR, clean[len("/models/") :])
        if clean.startswith("/benchmarks/"):
            return safe_join(BENCHMARKS_DIR, clean[len("/benchmarks/") :])
        return safe_join(PLAYGROUND_DIR, clean)

    def do_HEAD(self) -> None:
        self.serve_static(head_only=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            local_models = []
            if os.path.isdir(ONNX_MODELS_DIR):
                for entry in sorted(os.listdir(ONNX_MODELS_DIR)):
                    full = os.path.join(ONNX_MODELS_DIR, entry)
                    if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                        # Determine metadata
                        total_bytes = sum(
                            os.path.getsize(os.path.join(root, f))
                            for root, _, files in os.walk(full)
                            for f in files
                        )
                        size_gb = round(total_bytes / (1024 ** 3), 2)
                        has_vision = os.path.exists(os.path.join(full, "onnx", "vision_encoder_q4f16.onnx"))

                        if "LFM3-1.7B" in entry and "GLB" in entry:
                            friendly_name = "LFM3 1.7B GLB Asset Coder"
                            tag = f"{friendly_name} (Local WebGPU · {size_gb} GB · ~98 tok/s · Fast)"
                            model_type = "glb"
                            desc = f"Locally fine-tuned hybrid conv + GQA model for standalone Three.js & Godot GLB assets. Extremely fast (~98 tok/s) in {size_gb} GB VRAM."
                        elif "LFM3-1.7B" in entry:
                            friendly_name = "LFM3 1.7B Three.js Scene Coder"
                            tag = f"{friendly_name} (Local WebGPU · {size_gb} GB · ~98 tok/s · Interactive Scenes)"
                            model_type = "scene"
                            desc = f"Locally fine-tuned hybrid conv + GQA model on ccore/ThreeJS-3D-Scenes for full Three.js interactive scenes. Extremely fast (~98 tok/s) in {size_gb} GB VRAM."
                        elif "3D-Coder" in entry and "GLB" not in entry:
                            friendly_name = "Qwen3.5 4B Three.js Scene Coder"
                            tag = f"{friendly_name} (Local WebGPU · {size_gb} GB · Interactive Scenes)"
                            model_type = "scene"
                            desc = f"Specialized 4B model for full interactive Three.js 3D web experiences with lighting, camera controls, and animation loops. {size_gb} GB VRAM."
                        elif "GLB-3D-Coder" in entry:
                            friendly_name = "Qwen3.5 4B GLB Asset Coder"
                            tag = f"{friendly_name} (Local WebGPU · {size_gb} GB · Vision · Studio)"
                            model_type = "glb"
                            desc = f"Specialized 4B low-poly 3D game asset generator with Export GLB button and multimodal image conditioning. {size_gb} GB VRAM."
                        else:
                            friendly_name = entry
                            tag = f"{entry} (Local WebGPU · {size_gb} GB)"
                            model_type = "general"
                            desc = f"Local ONNX WebGPU model ({size_gb} GB)."

                        local_models.append({
                            "id": entry,
                            "label": tag,
                            "short_name": friendly_name,
                            "group": "Local 3D WebGPU Models (Offline · Zero Download)",
                            "is_local": True,
                            "size_gb": size_gb,
                            "type": model_type,
                            "has_vision": has_vision,
                            "description": desc,
                        })

            remote_models = [
                {
                    "id": "onnx-community/Qwen3.5-4B-ONNX-OPT",
                    "label": "Qwen3.5 4B Base Instruct (Remote HF · ~2.5 GB download)",
                    "short_name": "Qwen3.5-4B Base Instruct",
                    "group": "Base Remote WebGPU Models (Hugging Face)",
                    "is_local": False,
                    "size_gb": 2.5,
                    "type": "general",
                    "has_vision": True,
                    "description": "Base instruction-tuned Qwen3.5 4B model from onnx-community. Requires one-time ~2.5 GB download cached in browser IndexedDB.",
                },
                {
                    "id": "onnx-community/LFM3-1.7B-ONNX",
                    "label": "LFM3 1.7B Base Instruct (Remote HF · ~1.0 GB download)",
                    "short_name": "LFM3-1.7B Base Instruct",
                    "group": "Base Remote WebGPU Models (Hugging Face)",
                    "is_local": False,
                    "size_gb": 1.0,
                    "type": "general",
                    "has_vision": False,
                    "description": "Base instruction-tuned Liquid Foundation Model 1.7B from onnx-community. Requires one-time ~1.0 GB download.",
                },
            ]

            self.send_json(200, {
                "models": local_models + remote_models,
                "local_count": len(local_models),
                "remote_count": len(remote_models),
                "total_count": len(local_models) + len(remote_models),
            })
            return

        if parsed.path == "/api/benchmarks":
            items = []
            if os.path.isdir(BENCHMARKS_DIR):
                for root, _, files in os.walk(BENCHMARKS_DIR):
                    for fn in sorted(files):
                        full_fn = os.path.join(root, fn)
                        if fn.endswith(".html") and fn != "compare.html" and os.path.getsize(full_fn) >= 1024:
                            rel = os.path.relpath(full_fn, BENCHMARKS_DIR).replace(os.sep, "/")
                            items.append({"path": f"/benchmarks/{rel}", "label": rel})
            items.sort(key=lambda x: x["label"])
            self.send_json(200, {"benchmarks": items})
            return

        if parsed.path == "/api/telemetry/stats":
            events = []
            if os.path.exists(TELEMETRY_FILE):
                with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            counts = {}
            for ev in events:
                act = ev.get("event_type", "unknown")
                counts[act] = counts.get(act, 0) + 1
            self.send_json(
                200,
                {
                    "total_events": len(events),
                    "counts": counts,
                    "recent": events[-15:],
                },
            )
            return

        if parsed.path == "/api/eval10/glbs":
            glbs_dir = os.path.join(BENCHMARKS_DIR, "eval10", "glbs")
            found = {}
            if os.path.isdir(glbs_dir):
                for mkey in sorted(os.listdir(glbs_dir)):
                    mdir = os.path.join(glbs_dir, mkey)
                    if os.path.isdir(mdir):
                        found[mkey] = {}
                        for fname in sorted(os.listdir(mdir)):
                            if fname.endswith(".glb"):
                                pid = fname[:-4]
                                fpath = os.path.join(mdir, fname)
                                fsize = os.path.getsize(fpath)
                                found[mkey][pid] = {
                                    "url": f"/benchmarks/eval10/glbs/{mkey}/{fname}",
                                    "bytes": fsize,
                                    "kb": round(fsize / 1024, 1),
                                }
            self.send_json(200, {"glbs": found})
            return

        self.serve_static(head_only=False)

    def do_POST(self) -> None:
        site = self.headers.get("Sec-Fetch-Site")
        if site and site not in ("same-origin", "same-site", "none"):
            self.send_json(403, {"error": "Cross-site requests forbidden"})
            return

        parsed = urlparse(self.path)
        if parsed.path == "/api/save_glb":
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50 * 1024 * 1024:
                self.send_json(400, {"error": "Invalid payload size"})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                model_key = re.sub(r"[^A-Za-z0-9_.-]", "", payload.get("model_key", ""))
                prompt_id = re.sub(r"[^A-Za-z0-9_.-]", "", payload.get("prompt_id", ""))
                if not model_key or not prompt_id:
                    self.send_json(400, {"error": "Missing model_key or prompt_id"})
                    return
                b64 = payload.get("data_base64", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                glb_bytes = base64.b64decode(b64)
                target_dir = os.path.join(BENCHMARKS_DIR, "eval10", "glbs", model_key)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, f"{prompt_id}.glb")
                with open(target_path, "wb") as f:
                    f.write(glb_bytes)
                self.send_json(200, {
                    "ok": True,
                    "rel_path": f"/benchmarks/eval10/glbs/{model_key}/{prompt_id}.glb",
                    "bytes": len(glb_bytes),
                    "kb": round(len(glb_bytes) / 1024, 1),
                })
                return
            except Exception as e:
                self.send_json(500, {"error": str(e)})
                return
        if parsed.path == "/api/telemetry":
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10 * 1024 * 1024:
                self.send_json(400, {"error": "Invalid payload size"})
                return
            raw = self.rfile.read(length)
            try:
                record = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_json(400, {"error": "Invalid JSON"})
                return

            os.makedirs(TELEMETRY_DIR, exist_ok=True)
            record["logged_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(TELEMETRY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.send_json(200, {"ok": True})
            return

        self.send_json(404, {"error": "Not found"})

    def serve_static(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        fpath = self.resolve_local_file(parsed.path)
        if not fpath or not os.path.isfile(fpath):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "File not found", "path": parsed.path})
            return

        file_size = os.path.getsize(fpath)
        ctype, _ = mimetypes.guess_type(fpath)
        if not ctype:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/json":
            ctype = f"{ctype}; charset=utf-8"

        range_hdr = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = HTTPStatus.OK

        if range_hdr:
            m = re.match(r"bytes=(\d*)-(\d*)", range_hdr)
            if m:
                s_str, e_str = m.groups()
                if s_str:
                    start = int(s_str)
                    end = int(e_str) if e_str else file_size - 1
                elif e_str:
                    start = max(0, file_size - int(e_str))
                    end = file_size - 1
                if start >= file_size or start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                end = min(end, file_size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

        content_len = (end - start + 1) if file_size > 0 else 0
        self.send_response(status)
        self.add_security_headers()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(content_len))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if head_only or content_len == 0:
            return

        chunk_size = 1024 * 1024
        with open(fpath, "rb") as f:
            f.seek(start)
            remaining = content_len
            while remaining > 0:
                buf = f.read(min(chunk_size, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(buf)

    def log_message(self, fmt: str, *args) -> None:
        # Keep console output concise (skip noisy chunk range spam unless error)
        msg = fmt % args
        if "206" not in msg:
            print(f"[playground] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PlaygroundHandler)
    print(f"ThreeCoder WebGPU Playground running at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
