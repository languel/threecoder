import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CHROME_PORT = 9777;
const SCENE_MODELS = [
  "qwen35_4b_3d_bf16",
  "lfm3_1.7b_3d_bf16",
  "lfm3_1.7b_3d_4bit",
];

const PROMPTS = [
  "P01_default_cube",
  "P02_gold_sphere",
  "P03_cube_sphere_tetrahedron",
  "P04_platonic_solids",
  "P05_pine_tree",
  "P06_particle_system",
  "P07_fox_character",
  "P08_owl_character",
  "P09_human_character",
  "P10_snowglobe",
];

const SNAPSHOTS_DIR = path.resolve("benchmarks/eval10/snapshots");

async function main() {
  console.log("Launching Chrome for true-surface WebGL snapshots with autoframe...");
  const chromeProc = spawn(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    [
      "--headless=new",
      `--remote-debugging-port=${CHROME_PORT}`,
      "--user-data-dir=/tmp/chrome_snap_v2",
      "--no-first-run",
      "--disable-gpu-sandbox",
      "--window-size=640,480",
      "about:blank",
    ],
    { stdio: "ignore" }
  );

  try {
    await new Promise((r) => setTimeout(r, 1500));
    const versionRes = await fetch(`http://127.0.0.1:${CHROME_PORT}/json/version`);
    const vdata = await versionRes.json();
    console.log("Connected to Chrome:", vdata.Browser);

    const ws = new WebSocket(vdata.webSocketDebuggerUrl);
    await new Promise((resolve) => (ws.onopen = resolve));

    let id = 1;
    function send(method, params = {}) {
      return new Promise((resolve, reject) => {
        const reqId = id++;
        const handler = (event) => {
          const msg = JSON.parse(event.data);
          if (msg.id === reqId) {
            ws.removeEventListener("message", handler);
            if (msg.error) reject(msg.error);
            else resolve(msg.result);
          }
        };
        ws.addEventListener("message", handler);
        ws.send(JSON.stringify({ id: reqId, method, params }));
      });
    }

    const { targetId } = await send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });

    function sendSession(method, params = {}) {
      return new Promise((resolve, reject) => {
        const reqId = id++;
        const handler = (event) => {
          const msg = JSON.parse(event.data);
          if (msg.id === reqId) {
            ws.removeEventListener("message", handler);
            if (msg.error) reject(msg.error);
            else resolve(msg.result);
          }
        };
        ws.addEventListener("message", handler);
        ws.send(JSON.stringify({ id: reqId, sessionId, method, params }));
      });
    }

    await sendSession("Page.enable");
    await sendSession("Runtime.enable");

    for (const mkey of SCENE_MODELS) {
      const outModelDir = path.join(SNAPSHOTS_DIR, mkey);
      fs.mkdirSync(outModelDir, { recursive: true });
      console.log(`\n=== Capturing Surface Snapshots for ${mkey} ===`);

      for (const pid of PROMPTS) {
        const outPng = path.join(outModelDir, `${pid}.png`);
        const targetUrl = `http://127.0.0.1:8765/benchmarks/eval10/outputs/${mkey}/${pid}.html`;

        try {
          await sendSession("Page.navigate", { url: targetUrl });
          
          // Wait for load + evaluate animation frame sync
          const frameRes = await sendSession("Runtime.evaluate", {
            expression: `new Promise((resolve) => {
              let count = 0;
              function check() {
                count++;
                if (count >= 12) {
                  // Attempt autoframe if window.THREE / scene exists
                  try {
                    const canvas = document.querySelector('canvas');
                    if (canvas && window.THREE && window.scene && window.camera) {
                      const box = new window.THREE.Box3().setFromObject(window.scene);
                      if (!box.isEmpty()) {
                        const size = box.getSize(new window.THREE.Vector3());
                        const center = box.getCenter(new window.THREE.Vector3());
                        const maxDim = Math.max(size.x, size.y, size.z);
                        if (maxDim > 0.1 && maxDim < 500) {
                          const fov = (window.camera.fov || 50) * (Math.PI / 180);
                          const dist = (maxDim / 2 / Math.tan(fov / 2)) * 1.4;
                          window.camera.position.set(center.x + dist * 0.7, center.y + dist * 0.5, center.z + dist * 0.9);
                          window.camera.lookAt(center);
                        }
                      }
                    }
                  } catch (_) {}
                  resolve(true);
                } else {
                  requestAnimationFrame(check);
                }
              }
              requestAnimationFrame(check);
            })`,
            awaitPromise: true,
            returnByValue: true,
          });

          // Small sleep to let autoframe re-render
          await new Promise((r) => setTimeout(r, 200));

          const shot = await sendSession("Page.captureScreenshot", {
            fromSurface: true,
            format: "png",
          });

          if (shot && shot.data) {
            const buf = Buffer.from(shot.data, "base64");
            fs.writeFileSync(outPng, buf);
            const kb = (buf.length / 1024).toFixed(1);
            console.log(`  ${pid.padEnd(28)} -> ${kb.padStart(6)} KB ${buf.length > 25000 ? '✅ 3D MESH VISIBLE' : '⚠️ Flat / Blank'}`);
          }
        } catch (e) {
          console.error(`  ${pid} error:`, e.message);
        }
      }
    }

    ws.close();
    console.log("\nFinished capturing surface snapshots!");
  } finally {
    chromeProc.kill();
  }
}

main().catch(console.error);
