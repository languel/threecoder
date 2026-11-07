import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const EVAL_DIR = path.resolve("benchmarks/eval10");
const RESULTS_FILE = path.join(EVAL_DIR, "results.json");
const GLB_RESULTS_FILE = path.join(EVAL_DIR, "glb_results.json");
const GLBS_DIR = path.join(EVAL_DIR, "glbs");

async function main() {
  if (!fs.existsSync(RESULTS_FILE)) {
    throw new Error(`results.json not found at ${RESULTS_FILE}`);
  }
  const resultsData = JSON.parse(fs.readFileSync(RESULTS_FILE, "utf-8"));
  fs.mkdirSync(GLBS_DIR, { recursive: true });

  const chromeProc = spawn(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    [
      "--headless=new",
      "--remote-debugging-port=9333",
      "--user-data-dir=/tmp/chrome_extract_profile",
      "--no-first-run",
      "--disable-gpu-sandbox",
      "about:blank",
    ],
    { stdio: "ignore" }
  );

  try {
    await new Promise((r) => setTimeout(r, 1200));
    const versionRes = await fetch("http://127.0.0.1:9333/json/version");
    const vdata = await versionRes.json();
    console.log("Connected to Headless Chrome:", vdata.Browser);

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

    let currentException = null;
    let consoleErrors = [];

    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      if (msg.sessionId === sessionId) {
        if (msg.method === "Runtime.exceptionThrown") {
          const desc =
            msg.params.exceptionDetails?.exception?.description ||
            msg.params.exceptionDetails?.text ||
            "Unknown runtime exception";
          currentException = desc.split("\n")[0];
        }
        if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
          const args = msg.params.args.map((a) => a.value || a.description || "").join(" ");
          if (args && !args.includes("404")) {
            consoleErrors.push(args.slice(0, 120));
          }
        }
      }
    });

    const checkScript = `(() => {
      const canvas = document.querySelector("canvas");
      const btn =
        document.getElementById("exportButton") ||
        document.getElementById("exportBtn") ||
        document.querySelector("button[id*='export' i]") ||
        document.querySelector("button");
      const hasGlbText = btn ? /glb|download/i.test(btn.innerText) : false;
      return {
        hasCanvas: Boolean(canvas),
        canvasWidth: canvas ? canvas.width : 0,
        canvasHeight: canvas ? canvas.height : 0,
        hasExportBtn: Boolean(btn && hasGlbText),
        btnText: btn ? btn.innerText : "",
      };
    })()`;

    const exportScript = `(async () => {
      return new Promise((resolve) => {
        const timer = setTimeout(() => resolve({ error: "TIMEOUT" }), 4000);
        const orig = URL.createObjectURL;
        URL.createObjectURL = (blob) => {
          URL.createObjectURL = orig;
          clearTimeout(timer);
          const reader = new FileReader();
          reader.onloadend = () => {
            resolve({
              b64: reader.result,
              size: blob.size,
            });
          };
          reader.readAsDataURL(blob);
          return orig(blob);
        };
        const btn =
          document.getElementById("exportButton") ||
          document.getElementById("exportBtn") ||
          document.querySelector("button[id*='export' i]") ||
          document.querySelector("button");
        if (btn) btn.click();
        else resolve({ error: "NO_BUTTON" });
      });
    })()`;

    const glbResults = {
      extracted_at: new Date().toISOString(),
      models: {},
      summary: {},
    };

    console.log(`\nEvaluating runtime & extracting GLBs for ${resultsData.models.length} models × 10 prompts...\n`);

    for (const model of resultsData.models) {
      console.log(`\n=== Model: ${model.label} (${model.key}) ===`);
      glbResults.models[model.key] = {
        label: model.label,
        mode: model.mode,
        items: [],
      };
      const modelGlbDir = path.join(GLBS_DIR, model.key);
      fs.mkdirSync(modelGlbDir, { recursive: true });

      for (const item of model.items) {
        currentException = null;
        consoleErrors = [];

        const targetUrl = `http://127.0.0.1:8765/benchmarks/eval10/${item.html_rel_path}`;

        try {
          await sendSession("Page.navigate", { url: targetUrl });
          await new Promise((r) => setTimeout(r, 1200));

          // Check canvas and export button
          const checkRes = await sendSession("Runtime.evaluate", {
            expression: checkScript,
            returnByValue: true,
          });

          const pageState = checkRes.result?.value || {};
          let glbStatus = "NONE";
          let glbBytes = 0;
          let glbRelPath = null;
          let runtimeStatus = currentException ? "CRASH" : "CLEAN";
          let runtimeError = currentException || (consoleErrors.length ? consoleErrors[0] : null);

          if (!currentException && pageState.hasExportBtn) {
            // Trigger export with a 4-second timeout
            const exportRes = await sendSession("Runtime.evaluate", {
              expression: exportScript,
              awaitPromise: true,
              returnByValue: true,
            });

            const expVal = exportRes.result?.value || {};
            if (expVal.size && expVal.b64) {
              const parts = expVal.b64.split(",");
              const b64Data = parts.length > 1 ? parts[1] : parts[0];
              const rawBuf = Buffer.from(b64Data, "base64");
              const glbFilePath = path.join(modelGlbDir, `${item.prompt_id}.glb`);
              fs.writeFileSync(glbFilePath, rawBuf);
              glbStatus = "EXPORTED";
              glbBytes = rawBuf.length;
              glbRelPath = `glbs/${model.key}/${item.prompt_id}.glb`;
            } else if (expVal.error) {
              glbStatus = expVal.error;
            }
          } else if (currentException) {
            glbStatus = "RUNTIME_CRASH";
          } else if (!pageState.hasExportBtn) {
            glbStatus = "NO_EXPORTER";
          }

          const glbEntry = {
            prompt_id: item.prompt_id,
            level: item.level,
            title: item.title,
            raw_js_ok: item.raw_js_ok,
            healed_js_ok: item.healed_js_ok,
            runtime_status: runtimeStatus,
            runtime_error: runtimeError,
            has_canvas: pageState.hasCanvas,
            glb_status: glbStatus,
            glb_bytes: glbBytes,
            glb_kb: Math.round((glbBytes / 1024) * 10) / 10,
            glb_rel_path: glbRelPath,
          };

          glbResults.models[model.key].items.push(glbEntry);
          const icon = glbStatus === "EXPORTED" ? "✅" : (runtimeStatus === "CRASH" ? "❌" : "⚠️");
          console.log(
            `  ${item.prompt_id.padEnd(28)} | ${icon} GLB: ${glbStatus.padEnd(14)} | ` +
            `Runtime: ${runtimeStatus.padEnd(6)} | ` +
            `${glbBytes ? (glbBytes / 1024).toFixed(1) + " KB" : "-"} ` +
            `${runtimeError ? "(" + runtimeError + ")" : ""}`
          );
        } catch (err) {
          console.error(`  Error processing ${item.prompt_id}:`, err.message);
        }
      }
    }

    fs.writeFileSync(GLB_RESULTS_FILE, JSON.stringify(glbResults, null, 2), "utf-8");
    console.log(`\nWrote complete GLB & runtime results to ${GLB_RESULTS_FILE}`);

    ws.close();
  } finally {
    chromeProc.kill();
  }
}

main().catch(console.error);
