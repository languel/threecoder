const TRANSFORMERS_CDN =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";

const GLB_SYSTEM_PROMPT =
  "Create game assets for the Godot editor, which uses GLB files. " +
  "Create whatever the user requests as a good-quality low-poly asset using only Three.js " +
  "in one complete standalone HTML file. The HTML must show the asset and include an " +
  "Export GLB button that downloads a working GLB. If the requested asset needs animation, " +
  "add it and include the animation in the exported GLB; otherwise make it static.";

const FAST_PLAN_TEMPLATE =
  "Plan:\n" +
  "1. Create one complete, self-contained HTML document using an ES module importmap for Three.js (0.169.0), OrbitControls, and GLTFExporter.\n" +
  "2. Build clean low-poly procedural geometry grouped under an asset root with balanced PBR materials and scene lighting.\n" +
  "3. If animation is requested, construct named pivot groups and seamless looping THREE.AnimationClip keyframe tracks played via AnimationMixer, and pass animations: [clip], binary: true to GLTFExporter on Export GLB click.";

const HTML_OPEN_BRIDGE = '\n</think>\n\n```html\n<!DOCTYPE html>\n<html lang="en">\n';

let tf = null;
let stoppingCriteria = null;
let loadedModelKey = null;
let loadedModelType = null;
let isMultimodalModel = false;
let processor = null;
let tokenizer = null;
let model = null;

// Mid-stream control flags
let userHardStopRequested = false;
let skipThinkRequested = false;

function emit(type, extra = {}) {
  self.postMessage({ type, ts: Date.now(), ...extra });
}

self.addEventListener("error", (ev) => {
  emit("error", {
    stage: "worker_uncaught",
    message: ev?.message || "Uncaught worker error",
    detail: ev?.filename ? `${ev.filename}:${ev.lineno}:${ev.colno}` : "",
  });
});

self.addEventListener("unhandledrejection", (ev) => {
  const reason = ev?.reason;
  emit("error", {
    stage: "worker_rejection",
    message: reason?.message || String(reason || "Unhandled promise rejection in worker"),
    detail: reason?.stack || "",
  });
});

function hasCompletedRealHtml(fullText) {
  const bodyPart = fullText.includes("</think>")
    ? fullText.split("</think>").slice(1).join("</think>")
    : fullText;
  const lower = bodyPart.toLowerCase();
  if (!lower.includes("</html>")) return false;
  const docIdx = lower.indexOf("<!doctype html");
  const htmlStart = docIdx !== -1 ? docIdx : lower.indexOf("<html");
  const htmlEnd = lower.lastIndexOf("</html>");
  if (htmlStart === -1 || htmlEnd <= htmlStart) return false;
  const candidate = bodyPart.slice(htmlStart, htmlEnd + 7);
  if (candidate.length < 2200) return false;
  if (candidate.includes("<style>...</style>") || candidate.includes("<meta charset...")) {
    return false;
  }
  const tail = lower.slice(htmlEnd + 7);
  return tail.includes("```") || fullText.includes("<|im_end|>");
}

async function ensureRuntime() {
  if (tf) return tf;
  emit("status", {
    phase: "booting_runtime",
    message: "Loading Transformers.js v4.3.2 + ONNX Runtime WebGPU bundle…",
  });
  emit("log", { message: `Importing ${TRANSFORMERS_CDN}` });

  tf = await import(TRANSFORMERS_CDN);
  tf.env.allowLocalModels = true;
  tf.env.localModelPath = "/models/";
  tf.env.allowRemoteModels = true;
  stoppingCriteria = new tf.InterruptableStoppingCriteria();

  emit("log", { message: "Transformers.js v4.3.2 WebGPU runtime initialized." });
  return tf;
}

async function detectModelType(modelId, isLocal) {
  const cfgUrl = isLocal
    ? `/models/${modelId}/config.json`
    : `https://huggingface.co/${modelId}/resolve/main/config.json`;
  try {
    const res = await fetch(cfgUrl);
    if (res.ok) {
      const cfg = await res.json();
      return String(cfg.model_type || "").toLowerCase();
    }
  } catch {
    // Fall back to modelId heuristics
  }
  const lower = modelId.toLowerCase();
  if (lower.includes("qwen3.5") || lower.includes("qwen3_5")) return "qwen3_5";
  if (lower.includes("lfm3")) return "lfm3";
  if (lower.includes("lfm2")) return "lfm2";
  return "causal_lm";
}

async function ensureModel(modelId, dtypeMode = "q4f16") {
  const runtime = await ensureRuntime();
  const targetKey = `${modelId}:${dtypeMode}`;
  if (model && (processor || tokenizer) && loadedModelKey === targetKey) {
    return;
  }

  // Free previous WebGPU InferenceSession VRAM cleanly before switching models
  if (model && typeof model.dispose === "function") {
    try {
      emit("log", { message: `Releasing previous WebGPU session (${loadedModelKey})…` });
      await model.dispose();
    } catch {
      // Ignore disposal errors
    }
    model = null;
    processor = null;
    tokenizer = null;
    loadedModelKey = null;
  }

  const isLocal = !modelId.includes("/");
  const tLoadStart = performance.now();
  const modelType = await detectModelType(modelId, isLocal);
  loadedModelType = modelType;
  isMultimodalModel = modelType === "qwen3_5";

  emit("status", {
    phase: "loading_processor",
    message: `Loading ${isMultimodalModel ? "multimodal processor" : "tokenizer"} for ${modelId} (${modelType})…`,
  });
  emit("log", {
    message: `Loading ${modelId} (arch=${modelType}, ${isLocal ? "local /models/" : "remote HF Hub"})`,
  });

  const progressMap = new Map();
  let announcedCompilePhase = false;

  const progressCallback = (info) => {
    if (!info) return;
    if (info.status === "initiate" && info.file) {
      emit("log", { message: `Fetching ${info.file}…` });
    } else if (info.status === "progress" && info.file) {
      progressMap.set(info.file, {
        loaded: info.loaded || 0,
        total: info.total || 0,
      });
      let loadedBytes = 0;
      let totalBytes = 0;
      for (const item of progressMap.values()) {
        loadedBytes += item.loaded;
        loadedBytes += 0;
        totalBytes += item.total;
      }
      const pct = totalBytes > 0 ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : 0;
      const totalGB = (totalBytes / 1e9).toFixed(2);
      emit("load_progress", {
        file: info.file,
        filePercent: info.progress ? Math.round(info.progress) : pct,
        percent: pct,
        loadedMB: Math.round(loadedBytes / 1e6),
        totalMB: Math.round(totalBytes / 1e6),
      });
      if (pct >= 99 && totalBytes > 300_000_000 && !announcedCompilePhase) {
        announcedCompilePhase = true;
        emit("status", {
          phase: "compiling_webgpu",
          message: `Shards fetched — uploading ${totalGB} GB to GPU & compiling WebGPU shaders (~3–20s)…`,
        });
        emit("log", {
          message: "All ONNX shards transferred; creating WebGPU InferenceSessions…",
        });
      }
    } else if (info.status === "done" && info.file) {
      emit("log", { message: `Loaded ${info.file}` });
    }
  };

  if (isMultimodalModel) {
    processor = await runtime.AutoProcessor.from_pretrained(modelId, {
      local_files_only: isLocal,
      progress_callback: progressCallback,
    });
    tokenizer = processor.tokenizer;

    emit("status", {
      phase: "loading_weights",
      message: `Streaming ${modelId} (${dtypeMode}) multimodal ONNX shards into WebGPU…`,
    });

    model = await runtime.Qwen3_5ForConditionalGeneration.from_pretrained(modelId, {
      dtype: {
        embed_tokens: dtypeMode,
        vision_encoder: dtypeMode,
        decoder_model_merged: dtypeMode,
      },
      device: "webgpu",
      local_files_only: isLocal,
      progress_callback: progressCallback,
    });
  } else {
    tokenizer = await runtime.AutoTokenizer.from_pretrained(modelId, {
      local_files_only: isLocal,
      progress_callback: progressCallback,
    });
    processor = null;

    emit("status", {
      phase: "loading_weights",
      message: `Streaming ${modelId} (${dtypeMode}) causal LM ONNX shards into WebGPU…`,
    });

    model = await runtime.AutoModelForCausalLM.from_pretrained(modelId, {
      dtype: dtypeMode,
      device: "webgpu",
      local_files_only: isLocal,
      progress_callback: progressCallback,
    });
  }

  loadedModelKey = targetKey;
  const loadSec = ((performance.now() - tLoadStart) / 1000).toFixed(1);
  emit("log", { message: `WebGPU model ready in ${loadSec}s (${modelId}, arch=${modelType}, ${dtypeMode}).` });
  emit("ready", {
    modelId,
    modelType,
    isMultimodal: isMultimodalModel,
    dtype: dtypeMode,
    loadSeconds: Number(loadSec),
  });
}

async function decodeWithAssistantPrefix({
  runtime,
  baseChat,
  imageInput,
  assistantPrefix,
  maxNewTokens,
  temperature,
  thinkBudgetTokens,
  tJobStart,
  priorTokenCount = 0,
}) {
  stoppingCriteria.reset();
  const fullChatText = baseChat + assistantPrefix;
  let inputs;
  if (isMultimodalModel && processor) {
    inputs = imageInput
      ? await processor(fullChatText, imageInput)
      : await processor(fullChatText);
  } else {
    inputs = tokenizer(fullChatText, { add_special_tokens: false });
  }

  const promptTokens = inputs?.input_ids?.dims
    ? inputs.input_ids.dims[inputs.input_ids.dims.length - 1]
    : 0;

  const alreadyPastThink = assistantPrefix.includes("</think>");
  emit("status", {
    phase: "prefilling",
    promptTokens,
    message: alreadyPastThink
      ? `Prefilling ${promptTokens.toLocaleString()} context/HTML tokens on WebGPU…`
      : `Prefilling ${promptTokens.toLocaleString()} prompt tokens on WebGPU…`,
  });
  emit("log", {
    message: `WebGPU prefill (${promptTokens} input tokens, arch=${loadedModelType}, pastThink=${alreadyPastThink}, temp=${temperature})…`,
  });

  let phaseTokens = 0;
  let firstTokenTime = null;
  let announcedHtmlSwitch = alreadyPastThink;
  let interruptedForThinkCap = false;
  let fullText = assistantPrefix;

  const streamer = new runtime.TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (textChunk) => {
      const now = performance.now();
      let ttftSeconds = null;
      if (!firstTokenTime) {
        firstTokenTime = now;
        ttftSeconds = Number(((now - tJobStart) / 1000).toFixed(2));
        emit("log", { message: `Streaming started (TTFT: ${ttftSeconds}s).` });
      }

      fullText += textChunk;
      const totalTokens = priorTokenCount + phaseTokens;
      const decodeElapsed = (now - firstTokenTime) / 1000;
      const tokPerSec = decodeElapsed > 0.05 ? phaseTokens / decodeElapsed : 0;

      const inHtmlPhase = fullText.includes("</think>");
      if (inHtmlPhase && !announcedHtmlSwitch) {
        announcedHtmlSwitch = true;
        emit("log", {
          message: `Completed <think> block at token ${totalTokens}; streaming Three.js HTML…`,
        });
      }

      // Check if user clicked "Skip <think>" OR think token budget reached
      if (
        !inHtmlPhase &&
        !userHardStopRequested &&
        (skipThinkRequested || (thinkBudgetTokens > 0 && phaseTokens >= thinkBudgetTokens))
      ) {
        interruptedForThinkCap = true;
        stoppingCriteria.interrupt();
      } else if (hasCompletedRealHtml(fullText)) {
        stoppingCriteria.interrupt();
      }

      emit("token", {
        chunk: textChunk,
        tokenCount: totalTokens,
        streamPhase: inHtmlPhase ? "streaming_html" : "streaming_think",
        tokPerSec: Number(tokPerSec.toFixed(1)),
        ttftSeconds,
        elapsedSeconds: Number(((now - tJobStart) / 1000).toFixed(1)),
      });
    },
    token_callback_function: () => {
      phaseTokens += 1;
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    do_sample: temperature > 0,
    temperature: temperature > 0 ? temperature : undefined,
    top_p: loadedModelType === "lfm3" ? 0.95 : 0.92,
    top_k: 40,
    repetition_penalty: loadedModelType === "lfm3" ? 1.05 : 1.02,
    streamer,
    stopping_criteria: [stoppingCriteria],
  });

  const decodeSec = firstTokenTime ? (performance.now() - firstTokenTime) / 1000 : 0.1;
  const avgTps = decodeSec > 0.05 ? phaseTokens / decodeSec : 0;

  return {
    fullText,
    phaseTokens,
    avgTps,
    interruptedForThinkCap,
  };
}

async function runGenerationOrContinuation(payload) {
  const runtime = await ensureRuntime();
  const {
    modelId,
    dtype = "q4f16",
    prompt,
    systemPrompt = GLB_SYSTEM_PROMPT,
    thinkingMode = "budget", // "fast" | "budget" | "full"
    thinkBudgetTokens = 2000,
    assistantPrefix: explicitPrefix = null,
    maxNewTokens = 16384,
    temperature = 0.65,
    imageBlob = null,
  } = payload;

  userHardStopRequested = false;
  skipThinkRequested = false;

  await ensureModel(modelId, dtype);

  emit("status", {
    phase: "preparing_inputs",
    message: imageBlob ? "Encoding reference image & chat template…" : "Preparing chat template…",
  });

  const userContent = [];
  let imageInput = null;
  if (imageBlob && isMultimodalModel) {
    const rawImg = await runtime.RawImage.fromBlob(imageBlob);
    imageInput = await rawImg.resize(448, 448);
    userContent.push({ type: "image" });
    emit("log", { message: "Resized reference image to 448×448 for vision_encoder_q4f16." });
  } else if (imageBlob && !isMultimodalModel) {
    emit("log", {
      message: `Note: ${modelId} (${loadedModelType}) is text-only; ignoring attached reference image.`,
    });
  }
  userContent.push({ type: "text", text: prompt });

  const conversation = [
    { role: "system", content: systemPrompt },
    { role: "user", content: imageInput ? userContent : prompt },
  ];

  // Align <think> opener with each architecture's exact SFT chat_template:
  // - Qwen3.5 uses "<|im_start|>assistant\n<think>\n"
  // - LFM3 uses "<|im_start|>assistant\n<think>" (or "<think></think>" for native Direct-HTML)
  const templateRunner = processor || tokenizer;
  let baseChat = templateRunner.apply_chat_template(conversation, {
    add_generation_prompt: true,
    enable_thinking: true,
    tokenize: false,
  });
  const isNativeDualModeLfm = loadedModelType === "lfm3";
  if (!isNativeDualModeLfm) {
    if (baseChat.endsWith("<think>")) {
      baseChat += "\n";
    } else if (!baseChat.includes("<think>")) {
      baseChat += "<think>\n";
    }
  } else if (!baseChat.includes("<think>")) {
    baseChat += "<think>";
  }

  let initialPrefix = "";
  let activeBudget = 0;

  if (typeof explicitPrefix === "string" && explicitPrefix.length > 0) {
    initialPrefix = explicitPrefix;
    activeBudget = 0;
    emit("log", {
      message: `Resuming from prefilled assistant state (${initialPrefix.length.toLocaleString()} chars)…`,
    });
  } else if (thinkingMode === "fast") {
    initialPrefix = FAST_PLAN_TEMPLATE + HTML_OPEN_BRIDGE;
    activeBudget = 0;
    emit("log", {
      message: "Fast Direct HTML mode: prefilled concise <think> plan + <!DOCTYPE html> header.",
    });
  } else if (thinkingMode === "budget") {
    initialPrefix = "";
    activeBudget = Number(thinkBudgetTokens) || 2000;
    emit("log", {
      message: `Capped <think> mode: up to ${activeBudget} reasoning tokens before auto-switching to HTML.`,
    });
  } else {
    initialPrefix = "";
    activeBudget = 0;
    emit("log", { message: "Full uncapped <think> mode enabled." });
  }

  // Send initial prefix to UI so rawAccumulator matches fullText 1:1
  emit("stream_reset", { initialPrefix });

  const tJobStart = performance.now();
  let result = await decodeWithAssistantPrefix({
    runtime,
    baseChat,
    imageInput,
    assistantPrefix: initialPrefix,
    maxNewTokens,
    temperature,
    thinkBudgetTokens: activeBudget,
    tJobStart,
    priorTokenCount: 0,
  });

  let totalGeneratedTokens = result.phaseTokens;

  // Seamless Phase 2 transition if <think> hit budget or user clicked "Skip <think> -> Write HTML Now"
  if (result.interruptedForThinkCap && !userHardStopRequested) {
    const cleanThought = result.fullText.replace(/<\/think>[\s\S]*$/i, "").trimEnd();
    const bridgedPrefix = cleanThought + HTML_OPEN_BRIDGE;
    skipThinkRequested = false;

    emit("log", {
      message: `Closing <think> at ${totalGeneratedTokens} tokens and transitioning directly into <!DOCTYPE html>…`,
    });
    emit("stream_reset", { initialPrefix: bridgedPrefix });

    const remainingBudget = Math.max(4096, maxNewTokens - totalGeneratedTokens);
    const phase2 = await decodeWithAssistantPrefix({
      runtime,
      baseChat,
      imageInput,
      assistantPrefix: bridgedPrefix,
      maxNewTokens: remainingBudget,
      temperature: Math.min(temperature, 0.65),
      thinkBudgetTokens: 0,
      tJobStart,
      priorTokenCount: totalGeneratedTokens,
    });

    result = phase2;
    totalGeneratedTokens += phase2.phaseTokens;
  }

  const totalSec = (performance.now() - tJobStart) / 1000;
  const finalTps = totalSec > 0.1 ? totalGeneratedTokens / totalSec : result.avgTps;
  const cleanedFull = result.fullText.replace(/<\|im_end\|>|<\s*\|endoftext\|\s*>/g, "").trim();

  emit("log", {
    message: `Finished: ${totalGeneratedTokens} generated tokens in ${totalSec.toFixed(1)}s (${finalTps.toFixed(1)} tok/s).`,
  });
  emit("complete", {
    fullText: cleanedFull,
    stoppedByUser: userHardStopRequested,
    tokenCount: totalGeneratedTokens,
    tokPerSec: Number(finalTps.toFixed(1)),
    wallSeconds: Number(totalSec.toFixed(1)),
  });
}

self.addEventListener("message", async (event) => {
  const { action, payload } = event.data || {};
  try {
    if (action === "ping") {
      emit("pong");
    } else if (action === "preload") {
      await ensureModel(payload.modelId, payload.dtype || "q4f16");
    } else if (action === "generate" || action === "continue") {
      await runGenerationOrContinuation(payload);
    } else if (action === "skip_think") {
      skipThinkRequested = true;
      if (stoppingCriteria) {
        stoppingCriteria.interrupt();
        emit("log", {
          message: "Skip <think> requested — closing </think> and jumping into HTML generation…",
        });
      }
    } else if (action === "stop") {
      userHardStopRequested = true;
      if (stoppingCriteria) {
        stoppingCriteria.interrupt();
        emit("log", { message: "Stop requested — pausing WebGPU generation (can Continue anytime)…" });
      }
    }
  } catch (err) {
    emit("error", {
      stage: action || "worker_action",
      message: err?.message || String(err),
      detail: err?.stack || "",
    });
  }
});

emit("boot", { message: "Worker thread online (Model not loaded yet)" });
