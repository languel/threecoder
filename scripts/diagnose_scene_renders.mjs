import { spawn } from "node:child_process";
import fs from "node:fs";

const CHROME_PORT = 9555;
const MODELS = ["qwen35_4b_3d_bf16", "lfm3_1.7b_3d_bf16", "lfm3_1.7b_3d_4bit"];
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

async function main() {
  const chromeProc = spawn(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    [
      "--headless=new",
      `--remote-debugging-port=${CHROME_PORT}`,
      "--user-data-dir=/tmp/chrome_diag_profile",
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

    let lastConsoleError = "";
    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      if (msg.sessionId === sessionId && msg.method === "Runtime.exceptionThrown") {
        lastConsoleError = msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text;
      }
    });

    for (const mkey of MODELS) {
      console.log(`\n==================================================`);
      console.log(`DIAGNOSING: ${mkey}`);
      console.log(`==================================================`);

      for (const pid of PROMPTS) {
        lastConsoleError = "";
        const targetUrl = `http://127.0.0.1:8765/benchmarks/eval10/outputs/${mkey}/${pid}.html`;
        await sendSession("Page.navigate", { url: targetUrl });
        await new Promise((r) => setTimeout(r, 1800));

        const res = await sendSession("Runtime.evaluate", {
          expression: `(() => {
            const canvas = document.querySelector('canvas');
            if (!canvas) return { error: 'No canvas found' };
            const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
            
            // Try reading canvas data
            let dataUrl = '';
            try { dataUrl = canvas.toDataURL('image/png'); } catch (e) { dataUrl = 'error: ' + e.message; }
            
            return {
              canvasWidth: canvas.width,
              canvasHeight: canvas.height,
              dataUrlLen: dataUrl.length,
              isBlack: dataUrl.length < 5000,
              title: document.title
            };
          })()`,
          returnByValue: true,
        });

        const val = res.result?.value;
        const err = lastConsoleError ? `[CRASH: ${lastConsoleError.slice(0, 70)}]` : "[CLEAN]";
        console.log(`  ${pid.padEnd(28)} | ${err.padEnd(72)} | canvas: ${val?.canvasWidth}x${val?.canvasHeight} | dataLen: ${val?.dataUrlLen}`);
      }
    }

    ws.close();
  } finally {
    chromeProc.kill();
  }
}

main().catch(console.error);
