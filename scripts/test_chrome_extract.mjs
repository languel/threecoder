import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

async function main() {
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

    // Create a target
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

    let runtimeException = null;
    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      if (msg.sessionId === sessionId && msg.method === "Runtime.exceptionThrown") {
        runtimeException = msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text;
      }
    });

    console.log("Navigating to P01 of Qwen3.5 GLB BF16...");
    await sendSession("Page.navigate", {
      url: "http://127.0.0.1:8765/benchmarks/eval10/outputs/qwen35_4b_glb_bf16/P01_default_cube.html",
    });

    await new Promise((r) => setTimeout(r, 1500));
    console.log("Runtime exception:", runtimeException || "None (clean execution)");

    const evalRes = await sendSession("Runtime.evaluate", {
      expression: `(${async function () {
        return new Promise((resolve) => {
          const orig = URL.createObjectURL;
          URL.createObjectURL = (blob) => {
            URL.createObjectURL = orig;
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
          const btn = document.getElementById("exportButton") || document.querySelector("button");
          if (btn) btn.click();
          else resolve({ error: "no export button" });
        });
      }.toString()})()`,
      awaitPromise: true,
      returnByValue: true,
    });

    console.log("Export result size:", evalRes.result?.value?.size);
    ws.close();
  } finally {
    chromeProc.kill();
  }
}

main().catch(console.error);
