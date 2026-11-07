import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const SCENE_MODELS = [
  "lfm3_1.7b_3d_bf16",
  "lfm3_1.7b_3d_4bit",
  "qwen35_4b_3d_bf16",
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
  fs.mkdirSync(SNAPSHOTS_DIR, { recursive: true });

  console.log("Launching Headless Chrome for scene snapshots...");
  const chromeProc = spawn(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    [
      "--headless=new",
      "--remote-debugging-port=9444",
      "--user-data-dir=/tmp/chrome_snapshot_profile",
      "--no-first-run",
      "--disable-gpu-sandbox",
      "--window-size=640,480",
      "about:blank",
    ],
    { stdio: "ignore" }
  );

  try {
    await new Promise((r) => setTimeout(r, 1500));
    const versionRes = await fetch("http://127.0.0.1:9444/json/version");
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
    await sendSession("Emulation.setDeviceMetricsOverride", {
      width: 640,
      height: 480,
      deviceScaleFactor: 1,
      mobile: false,
    });

    for (const mkey of SCENE_MODELS) {
      const outModelDir = path.join(SNAPSHOTS_DIR, mkey);
      fs.mkdirSync(outModelDir, { recursive: true });
      console.log(`\n=== Capturing snapshots for ${mkey} ===`);

      for (const pid of PROMPTS) {
        const htmlPath = path.resolve(`benchmarks/eval10/outputs/${mkey}/${pid}.html`);
        if (!fs.existsSync(htmlPath)) {
          console.log(`  ${pid}: Missing HTML`);
          continue;
        }

        const outPng = path.join(outModelDir, `${pid}.png`);
        const targetUrl = `http://127.0.0.1:8765/benchmarks/eval10/outputs/${mkey}/${pid}.html`;

        try {
          await sendSession("Page.navigate", { url: targetUrl });
          // Wait 1.2s for WebGL scene initialization and rendering
          await new Promise((r) => setTimeout(r, 1200));

          const shot = await sendSession("Page.captureScreenshot", {
            format: "webp",
            quality: 85,
          });

          if (shot && shot.data) {
            const buf = Buffer.from(shot.data, "base64");
            fs.writeFileSync(outPng, buf);
            console.log(`  ${pid} -> ${outPng} (${(buf.length / 1024).toFixed(1)} KB)`);
          }
        } catch (e) {
          console.error(`  ${pid} failed:`, e.message);
        }
      }
    }

    ws.close();
    console.log("\nAll scene snapshots captured successfully!");
  } finally {
    chromeProc.kill();
  }
}

main().catch(console.error);
