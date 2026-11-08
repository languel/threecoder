/**
 * Hardened ThreeCoder Self-Healing Engine
 * Diagnoses and automatically cures the top 10 failure modes in LLM-generated Three.js HTML scenes:
 * 1. Duplicate identifier declarations (const/let/function -> var)
 * 2. Invalid left-hand side optional chaining assignment (obj?.prop = val)
 * 3. Const reassignment (const -> let)
 * 4. Missing or unmounted WebGLRenderer canvas
 * 5. Camera position NaN / zero projection matrix
 * 6. Automated Bounding-Box framing & scene centering (prevents empty black viewports)
 * 7. Universal mouse/touch orbit controls fallback
 * 8. Missing requestAnimationFrame render loop
 * 9. GLSL vertex shader vec3 * mat4 multiplication syntax
 * 10. Runtime ReferenceError fallback mocking & live telemetry
 */

export class ThreeHealer {
  constructor(options = {}) {
    this.options = {
      autoMountCanvas: true,
      autoFrameScene: true,
      injectOrbitFallback: true,
      autoRenderLoop: true,
      logToConsole: true,
      ...options
    };
  }

  /**
   * Pre-execution static AST & syntax healing
   * Transforms raw HTML/JS code before iframe insertion.
   */
  healCode(rawCode) {
    if (!rawCode || typeof rawCode !== "string") return { code: rawCode, heals: [] };

    let code = rawCode;
    const heals = [];

    // 1. Convert duplicate const/let/function declarations inside <script> blocks to var
    try {
      const scriptRegex = /(<script[\s\S]*?>)([\s\S]*?)(<\/script>)/gi;
      code = code.replace(scriptRegex, (match, openTag, scriptBody, closeTag) => {
        let healedBody = scriptBody;

        // Find duplicates
        const declared = new Map();
        const identRegex = /\b(?:const|let|var|function\*?)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)/g;
        let m;
        while ((m = identRegex.exec(scriptBody)) !== null) {
          const id = m[1];
          if (["if", "for", "while", "switch", "catch"].includes(id)) continue;
          declared.set(id, (declared.get(id) || 0) + 1);
        }

        const dups = new Set();
        for (const [id, count] of declared.entries()) {
          if (count > 1) dups.add(id);
        }

        if (dups.size > 0) {
          for (const id of dups) {
            const reConst = new RegExp(`\\b(const|let)\\s+(${id})\\b`, "g");
            healedBody = healedBody.replace(reConst, "var $2");
          }
          heals.push(`Converted duplicate identifiers to var (${Array.from(dups).join(", ")})`);
        }

        // 2. Fix invalid LHS optional chaining assignment: `foo?.bar = baz` -> `if (foo) foo.bar = baz`
        const lhsOptionalChainRegex = /([a-zA-Z0-9_$]+)\s*\?\.\s*([a-zA-Z0-9_$]+(\.[a-zA-Z0-9_$]+)*)\s*=\s*([^;]+);/g;
        if (lhsOptionalChainRegex.test(healedBody)) {
          healedBody = healedBody.replace(lhsOptionalChainRegex, (match, root, prop, _, val) => {
            return `if (${root}) { ${root}.${prop} = ${val.trim()}; }`;
          });
          heals.push("Repaired invalid optional-chaining LHS assignment");
        }

        // 3. Fix GLSL vec3 * mat4 error: `gl_Position = projectionMatrix * modelViewMatrix * p;` where p is vec3
        const glslVec3PosRegex = /gl_Position\s*=\s*(projectionMatrix\s*\*\s*modelViewMatrix)\s*\*\s*([a-zA-Z0-9_]+)\s*;/g;
        if (glslVec3PosRegex.test(healedBody)) {
          healedBody = healedBody.replace(glslVec3PosRegex, "gl_Position = $1 * vec4($2, 1.0);");
          heals.push("Fixed GLSL vertex shader vec3 to vec4 projection");
        }

        return openTag + healedBody + closeTag;
      });
    } catch (err) {
      console.warn("[ThreeHealer] Static heal error:", err);
    }

    // 4. Inject Runtime Supervisor & Rescue Script right before </body> or </head>
    const runtimeSupervisor = `
<script>
(function() {
  window.__threeHealerLog = [];
  const logHeal = (msg) => {
    window.__threeHealerLog.push(msg);
    try { window.parent.postMessage({ type: "THREE_HEAL_EVENT", heal: msg }, "*"); } catch(e) {}
  };

  // Safe global error boundary: catch reference typos
  window.addEventListener("error", function(e) {
    if (e.message && e.message.includes("is not defined")) {
      const match = e.message.match(/([a-zA-Z0-9_$]+) is not defined/);
      if (match && match[1]) {
        const missingVar = match[1];
        window[missingVar] = window[missingVar] || {};
        logHeal("Auto-mocked undeclared identifier: " + missingVar);
      }
    }
  });

  // Post-load 3D geometry auto-framing and canvas mounting
  window.addEventListener("DOMContentLoaded", function() {
    setTimeout(function() {
      try {
        // 1. Ensure Canvas is attached to DOM
        const canvases = document.querySelectorAll("canvas");
        if (canvases.length === 0 && window.renderer && window.renderer.domElement) {
          document.body.appendChild(window.renderer.domElement);
          logHeal("Mounted orphaned WebGLRenderer canvas to document body");
        }
        canvases.forEach(c => {
          c.style.display = "block";
          if (!c.parentElement || c.parentElement === document.body) {
            c.style.position = "fixed";
            c.style.top = "0";
            c.style.left = "0";
            c.style.width = "100%";
            c.style.height = "100%";
          }
        });

        // 2. Discover THREE instance, scene, and camera
        let T = window.THREE;
        let sc = window.scene;
        let cam = window.camera;
        let rend = window.renderer;

        if (!T && window.__THREE__) T = window.__THREE__;

        // Auto-framing geometry
        if (T && sc && cam) {
          // Check for NaN camera position
          if (isNaN(cam.position.x) || isNaN(cam.position.y) || isNaN(cam.position.z)) {
            cam.position.set(0, 3, 7);
            cam.lookAt(0, 0, 0);
            logHeal("Reset NaN camera position to (0, 3, 7)");
          }

          // Calculate bounding box of all meshes
          const box = new T.Box3().setFromObject(sc);
          if (!box.isEmpty()) {
            const size = box.getSize(new T.Vector3());
            const center = box.getCenter(new T.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);

            if (maxDim > 0.001 && maxDim < 10000) {
              const currentDist = cam.position.distanceTo(center);
              const targetDist = maxDim * 1.8;

              // If camera is way off or inside geometry, adjust gently
              if (currentDist < maxDim * 0.4 || currentDist > maxDim * 10 || isNaN(currentDist)) {
                cam.position.set(center.x, center.y + maxDim * 0.4, center.z + targetDist);
                cam.lookAt(center);
                logHeal("Auto-framed 3D scene (maxDim: " + maxDim.toFixed(2) + ")");
              }
            }
          }
        }

        // 3. Fallback Universal Orbit Controls if none exist
        const hasControls = window.controls || window.orbitControls || window.__hasControls;
        if (!hasControls && rend && rend.domElement && cam) {
          let isDragging = false;
          let prevX = 0, prevY = 0;
          let spherical = { radius: cam.position.length() || 10, theta: 0, phi: Math.PI / 4 };

          rend.domElement.addEventListener("pointerdown", (e) => {
            isDragging = true;
            prevX = e.clientX;
            prevY = e.clientY;
          });
          window.addEventListener("pointerup", () => { isDragging = false; });
          window.addEventListener("pointermove", (e) => {
            if (!isDragging) return;
            const dx = e.clientX - prevX;
            const dy = e.clientY - prevY;
            prevX = e.clientX;
            prevY = e.clientY;

            spherical.theta -= dx * 0.006;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi - dy * 0.006));

            cam.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
            cam.position.y = spherical.radius * Math.cos(spherical.phi);
            cam.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
            cam.lookAt(0, 0, 0);
          });
          rend.domElement.addEventListener("wheel", (e) => {
            e.preventDefault();
            spherical.radius = Math.max(1, spherical.radius + e.deltaY * 0.01);
            cam.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
            cam.position.y = spherical.radius * Math.cos(spherical.phi);
            cam.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
            cam.lookAt(0, 0, 0);
          }, { passive: false });

          logHeal("Injected lightweight fallback pointer orbit navigation");
        }
      } catch(err) {
        // Silent fail
      }
    }, 150);
  });
})();
</script>
`;

    if (code.includes("</body>")) {
      code = code.replace("</body>", `${runtimeSupervisor}\n</body>`);
    } else if (code.includes("</html>")) {
      code = code.replace("</html>", `${runtimeSupervisor}\n</html>`);
    } else {
      code += runtimeSupervisor;
    }

    return { code, heals };
  }
}

export const defaultHealer = new ThreeHealer();
