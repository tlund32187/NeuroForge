/* NeuroForge Dashboard — Client-side JavaScript
   ──────────────────────────────────────────────── */
"use strict";

// ── State ────────────────────────────────────────────────────────────

const state = {
  ws: null,
  training: false,
  gate: "OR",
  topology: null,       // { layers: [...], edges: [...] }
  accuracyHistory: [],
  truthTable: {},
  weightHistory: {},    // { projName: { steps:[], weights:[] } }
  totalTrials: 0,
  converged: null,      // null | true | false
  lastAccuracy: 0,
  epoch: 0,
  trainStartTime: null, // Date.now() when training began
  _elapsedTimer: null,  // setInterval id for elapsed clock
  windowSteps: 50,      // sim steps per trial (for computing spike rates)
  resourceSeries: {
    cpuSystem: [],
    cpuProcess: [],
    ramSystemUsed: [],
    ramSystemTotal: [],
    ramProcessRss: [],
    gpuUtil: [],
    gpuMemUsed: [],
    gpuMemTotal: [],
    torchCudaAllocated: [],
    torchCudaReserved: [],
    torchCudaMax: [],
  },
  resourceLastStep: -1,
  resourceSamples: 0,

  // Network visualisation
  neurons: [],          // { id, layer, idx, x, y, r, color, label, totalSpikes, trialCount }
  connections: [],      // rendered connections after edge-mode filtering
  allConnections: [],   // full connection list from topology
  topologyVersion: 0,
  edgeSampleCache: {},  // key -> sampled connections

  // Mode: "live" | "replay"
  mode: "live",
  activeRunId: null,    // run_id of the currently loaded replay (or live run)
  runConfig: null,      // resolved config dict from run_start or replay API
  replayEvents: [],     // full event stream loaded from events.ndjson
  replayIndex: [],      // scrubber index metadata
  replayCursor: -1,     // last applied replay event index
  replayPlaying: false,
  replayTimer: null,
  replayIntervalMs: 80,

  // Topology view controls
  edgeMode: "sampled",  // off | sampled | full
  maxEdgesPerProjection: 200,
  strongestOnly: true,
  detailedNodes: false, // collapsed-by-default for large populations
};

// ── DOM refs ─────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const btnTrain    = $("#btn-train");
const btnStop     = $("#btn-stop");
const gateSelect  = $("#gate-select");
const deviceSelect= $("#device-select");
const maxTrials   = $("#max-trials");
const resourceMonitorEnabled = $("#resource-monitor-enabled");
const resourceMonitorEvery = $("#resource-monitor-every");
const statusBadge = $("#status-badge");

const canvasNet   = $("#canvas-network");
const canvasAcc   = $("#canvas-accuracy");
const canvasWgt   = $("#canvas-weights");
const canvasResCpu = $("#canvas-resource-cpu");
const canvasResRam = $("#canvas-resource-ram");
const canvasResGpu = $("#canvas-resource-gpu");
const canvasResCuda = $("#canvas-resource-cuda");
const resourceNote = $("#resource-note");
const resourceStatsCpu = $("#resource-stats-cpu");
const resourceStatsRam = $("#resource-stats-ram");
const resourceStatsGpu = $("#resource-stats-gpu");
const resourceStatsCuda = $("#resource-stats-cuda");
const tooltip     = $("#tooltip");
const edgeModeSel = $("#topology-edge-mode");
const edgeCountWrap = $("#topology-edge-count-wrap");
const edgeCountSlider = $("#topology-edge-count");
const edgeCountValue = $("#topology-edge-count-value");
const strongestOnlyChk = $("#topology-strongest");
const detailedNodesChk = $("#topology-detailed-nodes");
const truthTbody  = $("#truth-table tbody");
const logContainer= $("#log-container");
const configGrid  = $("#config-grid");
const panelConfig = $("#panel-config");

const statTrial   = $("#stat-trial");
const statAccuracy= $("#stat-accuracy");
const statConverged=$("#stat-converged");
const statGate    = $("#stat-gate");
const statEpoch   = $("#stat-epoch");
const statElapsed = $("#stat-elapsed");

// Replay controls
const liveControls   = $("#live-controls");
const replayControls = $("#replay-controls");
const runSelect      = $("#run-select");
const btnLoadRun     = $("#btn-load-run");
const btnRefreshRuns = $("#btn-refresh-runs");
const btnModeLive    = $("#btn-mode-live");
const btnModeReplay  = $("#btn-mode-replay");
const replayEventsControls = $("#replay-events-controls");
const btnReplayPlay = $("#btn-replay-play");
const replayScrubber = $("#replay-scrubber");
const replayScrubberLabel = $("#replay-scrubber-label");

// Run info banner (live → replay handoff)
const runInfoBanner    = $("#run-info-banner");
const runInfoText      = $("#run-info-text");
const runInfoReplayLink= $("#run-info-replay-link");

// ── High-DPI canvas setup ────────────────────────────────────────────

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}

// ── WebSocket ────────────────────────────────────────────────────────

function connectWS() {
  if (state.mode === "replay") return;  // Don't connect WS in replay mode.
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  state.ws = new WebSocket(`${proto}//${location.host}/ws`);

  state.ws.onopen = () => console.log("[ws] connected");

  state.ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    handleMessage(msg);
  };

  state.ws.onclose = () => {
    console.log("[ws] disconnected — reconnecting in 2s");
    if (state.mode === "live") setTimeout(connectWS, 2000);
  };
}

function handleMessage(msg) {
  if (msg.topic === "snapshot") {
    // Full state snapshot on connect.
    if (msg.training) applyTrainingSnapshot(msg.training);
    if (msg.weights)  applyWeightSnapshot(msg.weights);
    if (msg.config)   { state.runConfig = msg.config; renderConfigPanel(); }
    return;
  }

  switch (msg.topic) {
    case "run_start":
      if (msg.data && msg.data.config) {
        state.runConfig = msg.data.config;
        renderConfigPanel();
      }
      break;

    case "training_start":
      state.training = true;
      state.gate = msg.data.gate || "OR";
      state.multiGates = msg.data.gates || null;  // list for MULTI mode
      state.windowSteps = msg.data.window_steps || 50;
      state.perPatternStreak = msg.data.per_pattern_streak || 5;
      state.accuracyHistory = [];
      state.truthTable = {};
      state.totalTrials = 0;
      state.converged = null;
      state.epoch = 0;
      if (state.mode === "live") {
        state.trainStartTime = Date.now();
        if (state._elapsedTimer) clearInterval(state._elapsedTimer);
        state._elapsedTimer = setInterval(() => {
          if (statElapsed) statElapsed.textContent = formatElapsed(Date.now() - state.trainStartTime);
        }, 1000);
      } else {
        state.trainStartTime = null;
        if (state._elapsedTimer) {
          clearInterval(state._elapsedTimer);
          state._elapsedTimer = null;
        }
      }
      state.weightHistory = {};
      resetResourceSeries();
      updateResourceNote();
      drawResourceCharts();
      // Reset per-neuron spike stats.
      for (const n of state.neurons) {
        n.totalSpikes = 0;
        n.lastSpikes = 0;
        n.trialCount = 0;
      }
      logContainer.innerHTML = "";
      setStatus("training");
      statGate.textContent = state.gate;
      if (state.mode === "live") {
        btnTrain.disabled = true;
        btnStop.disabled = false;
      }
      break;

    case "topology":
      state.topology = msg.data;
      buildNetworkLayout();
      drawNetwork();
      break;

    case "training_trial":
      state.totalTrials = msg.step + 1;
      state.lastAccuracy = msg.data.accuracy || 0;
      if (msg.data.epoch !== undefined) state.epoch = msg.data.epoch + 1;
      state.accuracyHistory.push(state.lastAccuracy);
      ingestResourceMetrics(msg.data, msg.step);

      updateTruthTableEntry(msg.data);
      updateNeuronSpikes(msg.data);
      updateStats();

      // Append log line (keep last 200).
      appendLog(msg);

      // Redraw charts periodically.
      if (state.totalTrials % 5 === 0 || state.totalTrials < 50) {
        drawAccuracyChart();
        renderTruthTable();
        drawNetwork();
      }
      break;

    case "scalar":
      ingestResourceMetrics(msg.data, msg.step);
      break;

    case "weight":
      recordWeight(msg);
      drawWeightChart();
      break;

    case "training_end":
      state.training = false;
      state.converged = msg.data.converged;
      if (msg.data.stopped) {
        setStatus("stopped");
      } else {
        setStatus(msg.data.converged ? "done" : "failed");
      }
      // Show per-gate status for MULTI mode.
      if (msg.data.per_gate_converged) {
        const pgc = msg.data.per_gate_converged;
        const summary = Object.entries(pgc)
          .map(([g, ok]) => `${g}:${ok ? "✓" : "✗"}`)
          .join("  ");
        const line = document.createElement("div");
        line.className = "log-correct";
        line.style.fontWeight = "bold";
        line.textContent = `Per-gate: ${summary}`;
        logContainer.appendChild(line);
      }
      if (state._elapsedTimer) { clearInterval(state._elapsedTimer); state._elapsedTimer = null; }
      if (state.mode === "live") {
        btnTrain.disabled = false;
        btnStop.disabled = true;
      }
      updateStats();
      drawAccuracyChart();
      drawWeightChart();
      renderTruthTable();
      drawNetwork();
      break;
  }
}

// ── Training snapshot (on WS connect) ────────────────────────────────

function applyTrainingSnapshot(snap) {
  state.gate = snap.gate || "";
  state.converged = snap.converged;
  state.totalTrials = snap.total_trials || 0;
  state.accuracyHistory = snap.accuracy_history || [];
  state.lastAccuracy = state.accuracyHistory.length
    ? state.accuracyHistory[state.accuracyHistory.length - 1] : 0;

  if (snap.truth_table) {
    state.truthTable = snap.truth_table;
  }
  if (snap.topology) {
    state.topology = snap.topology;
    buildNetworkLayout();
  }

  statGate.textContent = state.gate || "—";
  if (snap.epoch) state.epoch = snap.epoch;
  updateStats();
  renderTruthTable();
  drawAccuracyChart();
  drawNetwork();
}

function applyWeightSnapshot(snap) {
  if (snap.projections) {
    for (const [name, data] of Object.entries(snap.projections)) {
      state.weightHistory[name] = data;
    }
    drawWeightChart();
  }
}

function resetResourceSeries() {
  state.resourceSeries = {
    cpuSystem: [],
    cpuProcess: [],
    ramSystemUsed: [],
    ramSystemTotal: [],
    ramProcessRss: [],
    gpuUtil: [],
    gpuMemUsed: [],
    gpuMemTotal: [],
    torchCudaAllocated: [],
    torchCudaReserved: [],
    torchCudaMax: [],
  };
  state.resourceLastStep = -1;
  state.resourceSamples = 0;
}

function parseMetricNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function latestSeriesValue(seriesName) {
  const series = state.resourceSeries[seriesName];
  if (!series || series.length === 0) return null;
  const point = series[series.length - 1];
  if (!Array.isArray(point) || point.length < 2) return null;
  return parseMetricNumber(point[1]);
}

function formatPct(value) {
  return value === null ? "--" : `${value.toFixed(1)}%`;
}

function formatMb(value) {
  if (value === null) return "--";
  if (value >= 1024) return `${(value / 1024).toFixed(2)} GB`;
  return `${value.toFixed(0)} MB`;
}

function ratioPct(num, den) {
  if (num === null || den === null || den <= 0) return null;
  return (num / den) * 100;
}

function setText(el, text) {
  if (el) el.textContent = text;
}

function updateResourceStats() {
  const cpuSystem = latestSeriesValue("cpuSystem");
  const cpuProcess = latestSeriesValue("cpuProcess");
  setText(resourceStatsCpu, `sys ${formatPct(cpuSystem)} | proc ${formatPct(cpuProcess)}`);

  const ramUsed = latestSeriesValue("ramSystemUsed");
  const ramTotal = latestSeriesValue("ramSystemTotal");
  const ramProc = latestSeriesValue("ramProcessRss");
  const ramPct = ratioPct(ramUsed, ramTotal);
  let ramText = `sys ${formatMb(ramUsed)}`;
  if (ramTotal !== null) {
    ramText += ` / ${formatMb(ramTotal)}`;
    if (ramPct !== null) ramText += ` (${formatPct(ramPct)})`;
  }
  ramText += ` | proc ${formatMb(ramProc)}`;
  setText(resourceStatsRam, ramText);

  const gpuUtil = latestSeriesValue("gpuUtil");
  const gpuMemUsed = latestSeriesValue("gpuMemUsed");
  const gpuMemTotal = latestSeriesValue("gpuMemTotal");
  const gpuMemPct = ratioPct(gpuMemUsed, gpuMemTotal);
  let gpuText = `gpu ${formatPct(gpuUtil)} | vram ${formatMb(gpuMemUsed)}`;
  if (gpuMemTotal !== null) {
    gpuText += ` / ${formatMb(gpuMemTotal)}`;
    if (gpuMemPct !== null) gpuText += ` (${formatPct(gpuMemPct)})`;
  }
  setText(resourceStatsGpu, gpuText);

  const cudaAlloc = latestSeriesValue("torchCudaAllocated");
  const cudaRes = latestSeriesValue("torchCudaReserved");
  const cudaMax = latestSeriesValue("torchCudaMax");
  setText(
    resourceStatsCuda,
    `alloc ${formatMb(cudaAlloc)} | res ${formatMb(cudaRes)} | max ${formatMb(cudaMax)}`,
  );
}

function pushMetricPoint(seriesName, step, value) {
  const series = state.resourceSeries[seriesName];
  if (!series) return;
  series.push([step, value]);
  if (series.length > 1500) series.shift();
}

function hasResourceData() {
  return Object.values(state.resourceSeries).some((series) => series.length > 0);
}

function updateResourceNote() {
  if (!resourceNote) return;
  const hasData = hasResourceData();
  const hasCpu =
    state.resourceSeries.cpuSystem.length > 0 ||
    state.resourceSeries.cpuProcess.length > 0 ||
    state.resourceSeries.ramSystemUsed.length > 0 ||
    state.resourceSeries.ramProcessRss.length > 0;
  const hasGpu =
    state.resourceSeries.gpuUtil.length > 0 ||
    state.resourceSeries.gpuMemUsed.length > 0 ||
    state.resourceSeries.gpuMemTotal.length > 0;
  const hasCuda =
    state.resourceSeries.torchCudaAllocated.length > 0 ||
    state.resourceSeries.torchCudaReserved.length > 0 ||
    state.resourceSeries.torchCudaMax.length > 0;

  if (!hasData) {
    const monitorEnabled = !!(resourceMonitorEnabled && resourceMonitorEnabled.checked);
    resourceNote.textContent = monitorEnabled
      ? "No resource data yet. CPU/RAM metrics require psutil (`pip install psutil`)."
      : "No resource data yet.";
    return;
  }

  if (!hasCpu) {
    resourceNote.textContent = "CPU/RAM metrics unavailable (monitor disabled or psutil missing).";
    return;
  }
  if (!hasGpu && !hasCuda) {
    resourceNote.textContent = "GPU metrics unavailable (monitor disabled, NVML missing, or non-NVIDIA runtime).";
    return;
  }
  resourceNote.textContent = "";
}

function ingestResourceMetrics(data, stepRaw, opts = {}) {
  const dataObj = data || {};
  const step = Number.isFinite(stepRaw) ? stepRaw : state.resourceLastStep + 1;
  const deferDraw = !!opts.deferDraw;
  let added = false;

  const mapping = [
    ["resource.cpu.system_percent", "cpuSystem", 1],
    ["resource.cpu.process_percent", "cpuProcess", 1],
    ["resource.ram.system_used_mb", "ramSystemUsed", 1],
    ["resource.ram.system_total_mb", "ramSystemTotal", 1],
    ["resource.ram.process_rss_mb", "ramProcessRss", 1],
    ["resource.gpu.util_percent", "gpuUtil", 1],
    ["resource.gpu.mem_used_mb", "gpuMemUsed", 1],
    ["resource.gpu.mem_total_mb", "gpuMemTotal", 1],
    ["resource.torch.cuda_allocated_mb", "torchCudaAllocated", 1],
    ["resource.torch.cuda_reserved_mb", "torchCudaReserved", 1],
    ["resource.torch.cuda_max_allocated_mb", "torchCudaMax", 1],
    ["torch_cuda_allocated_mb", "torchCudaAllocated", 1],
    ["torch_cuda_reserved_mb", "torchCudaReserved", 1],
    ["torch_cuda_max_allocated_mb", "torchCudaMax", 1],
    ["cuda_mem_allocated", "torchCudaAllocated", 1 / (1024 * 1024)],
    ["cuda_mem_reserved", "torchCudaReserved", 1 / (1024 * 1024)],
    ["cuda_mem_peak", "torchCudaMax", 1 / (1024 * 1024)],
  ];

  for (const [key, seriesName, scale] of mapping) {
    if (!(key in dataObj)) continue;
    const parsed = parseMetricNumber(dataObj[key]);
    if (parsed === null) continue;
    pushMetricPoint(seriesName, step, parsed * scale);
    added = true;
  }

  if (!added) return false;
  state.resourceLastStep = step;
  state.resourceSamples += 1;
  if (!deferDraw) {
    updateResourceNote();
    drawResourceCharts();
  }
  return true;
}

function collectPoints(seriesName) {
  return state.resourceSeries[seriesName] || [];
}

function drawResourceChart(canvas, lines, opts = {}) {
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const padL = 34, padR = 8, padT = 8, padB = 18;
  const cw = w - padL - padR;
  const ch = h - padT - padB;

  const allPoints = [];
  for (const line of lines) {
    for (const p of line.points) allPoints.push(p);
  }
  if (allPoints.length === 0) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("No data", w / 2, h / 2);
    return;
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of allPoints) {
    minX = Math.min(minX, Number(x));
    maxX = Math.max(maxX, Number(x));
    minY = Math.min(minY, Number(y));
    maxY = Math.max(maxY, Number(y));
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return;
  if (opts.fixedMin !== undefined) minY = opts.fixedMin;
  if (opts.fixedMax !== undefined) maxY = opts.fixedMax;
  if (minY === maxY) { minY -= 1; maxY += 1; }
  if (minX === maxX) maxX = minX + 1;

  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 0.5;
  for (let t = 0; t <= 1; t += 0.25) {
    const py = padT + ch * t;
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(w - padR, py);
    ctx.stroke();
  }

  for (const line of lines) {
    if (!line.points || line.points.length < 2) continue;
    ctx.beginPath();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 1.3;
    line.points.forEach((p, idx) => {
      const x = padL + ((Number(p[0]) - minX) / (maxX - minX)) * cw;
      const y = padT + ch * (1 - (Number(p[1]) - minY) / (maxY - minY));
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  let lx = padL;
  for (const line of lines) {
    if (!line.points || line.points.length === 0) continue;
    ctx.fillStyle = line.color;
    const label = line.label || "";
    ctx.fillText(label, lx, h - 5);
    lx += (label.length * 6) + 10;
  }
}

function drawResourceCharts() {
  drawResourceChart(canvasResCpu, [
    { points: collectPoints("cpuSystem"), color: "#58a6ff", label: "system" },
    { points: collectPoints("cpuProcess"), color: "#3fb950", label: "process" },
  ], { fixedMin: 0, fixedMax: 100 });

  drawResourceChart(canvasResRam, [
    { points: collectPoints("ramSystemUsed"), color: "#d29922", label: "system used" },
    { points: collectPoints("ramProcessRss"), color: "#79c0ff", label: "process rss" },
  ]);

  drawResourceChart(canvasResGpu, [
    { points: collectPoints("gpuUtil"), color: "#f85149", label: "gpu %" },
    { points: collectPoints("gpuMemUsed"), color: "#bc8cff", label: "vram used" },
    { points: collectPoints("gpuMemTotal"), color: "#8b949e", label: "vram total" },
  ]);

  drawResourceChart(canvasResCuda, [
    { points: collectPoints("torchCudaAllocated"), color: "#56d364", label: "alloc" },
    { points: collectPoints("torchCudaReserved"), color: "#ff7b72", label: "reserved" },
    { points: collectPoints("torchCudaMax"), color: "#d2a8ff", label: "max" },
  ]);
  updateResourceStats();
}

// ── Training control ─────────────────────────────────────────────────

// Adjust max-trials placeholder when MULTI is selected.
gateSelect.addEventListener("change", () => {
  if (gateSelect.value === "MULTI") {
    maxTrials.value = "1500";
    maxTrials.title = "Max epochs (24 trials each)";
    maxTrials.min = "10";
    maxTrials.step = "50";
  } else {
    if (maxTrials.title.includes("epoch")) {
      maxTrials.value = "5000";
    }
    maxTrials.title = "Max trials";
    maxTrials.min = "100";
    maxTrials.step = "100";
  }
});

btnTrain.addEventListener("click", async () => {
  btnTrain.disabled = true;
  btnStop.disabled = false;
  hideRunInfoBanner();
  const gate = gateSelect.value;
  const trials = parseInt(maxTrials.value, 10) || 5000;
  const device = deviceSelect ? deviceSelect.value : "cuda";
  const resourceEnabled = !!(resourceMonitorEnabled && resourceMonitorEnabled.checked);
  const resourceEvery = Math.max(
    1,
    parseInt(resourceMonitorEvery ? resourceMonitorEvery.value : "10", 10) || 10,
  );

  const reqBody = {
    gate,
    max_trials: trials,
    device,
    monitoring: {
      resource: {
        enabled: resourceEnabled,
        every_n_steps: resourceEvery,
        include_system: true,
        include_process: true,
        include_gpu: true,
        gpu_index: 0,
      },
    },
  };

  const resp = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(reqBody),
  });

  if (!resp.ok) {
    const err = await resp.json();
    alert(err.error || "Failed to start training");
    btnTrain.disabled = false;
    btnStop.disabled = true;
  } else {
    const data = await resp.json();
    if (data.run_id) {
      showRunInfoBanner(data.run_id);
    }
  }
});

btnStop.addEventListener("click", async () => {
  btnStop.disabled = true;
  await fetch("/api/stop", { method: "POST" });
});

function setStatus(s) {
  statusBadge.className = `badge badge-${s}`;
  statusBadge.textContent = s;
}

function setReplayControlsVisible(visible) {
  if (!replayEventsControls) return;
  replayEventsControls.style.display = visible ? "" : "none";
}

function clearReplayTimer() {
  if (state.replayTimer) {
    clearInterval(state.replayTimer);
    state.replayTimer = null;
  }
}

function setReplayPlaying(playing) {
  const canPlay = state.mode === "replay" && state.replayEvents.length > 0;
  if (!canPlay) {
    clearReplayTimer();
    state.replayPlaying = false;
    if (btnReplayPlay) btnReplayPlay.textContent = "Play";
    return;
  }

  if (!playing) {
    clearReplayTimer();
    state.replayPlaying = false;
    if (btnReplayPlay) btnReplayPlay.textContent = "Play";
    return;
  }

  if (state.replayCursor >= state.replayEvents.length - 1) {
    state.replayCursor = -1;
    replaySeek(0);
  }
  clearReplayTimer();
  state.replayPlaying = true;
  if (btnReplayPlay) btnReplayPlay.textContent = "Pause";
  state.replayTimer = setInterval(() => {
    if (state.replayCursor >= state.replayEvents.length - 1) {
      setReplayPlaying(false);
      return;
    }
    replaySeek(state.replayCursor + 1);
  }, state.replayIntervalMs);
}

function updateReplayScrubberUi() {
  const count = state.replayEvents.length;
  const idx = Math.max(0, state.replayCursor);
  if (replayScrubber) {
    replayScrubber.min = "0";
    replayScrubber.max = String(Math.max(0, count - 1));
    replayScrubber.value = count > 0 ? String(Math.min(idx, count - 1)) : "0";
    replayScrubber.disabled = count === 0;
  }
  if (replayScrubberLabel) {
    if (count === 0) {
      replayScrubberLabel.textContent = "0 / 0";
    } else {
      const rec = state.replayIndex[Math.min(idx, state.replayIndex.length - 1)] || {};
      const topic = rec.topic ? ` ${rec.topic}` : "";
      replayScrubberLabel.textContent = `${idx + 1} / ${count}${topic}`;
    }
  }
  if (btnReplayPlay) {
    btnReplayPlay.disabled = count === 0;
    btnReplayPlay.textContent = state.replayPlaying ? "Pause" : "Play";
  }
}

function normalizeReplayEvent(rec) {
  return {
    topic: String(rec.topic || "").toLowerCase(),
    step: Number.isFinite(rec.step) ? rec.step : (Number(rec.t_step) || 0),
    t: Number.isFinite(rec.t) ? rec.t : Number(rec.t || 0),
    source: rec.source || "replay",
    data: rec.data || {},
  };
}

function replaySeek(targetIdx) {
  if (state.replayEvents.length === 0) {
    state.replayCursor = -1;
    updateReplayScrubberUi();
    return;
  }

  const clamped = Math.max(0, Math.min(targetIdx, state.replayEvents.length - 1));
  const current = state.replayCursor;

  if (clamped < current) {
    const runId = state.activeRunId;
    const replayEvents = state.replayEvents;
    const replayIndex = state.replayIndex;
    resetDashboardState({ preserveReplay: true });
    state.activeRunId = runId;
    state.replayEvents = replayEvents;
    state.replayIndex = replayIndex;
    state.replayCursor = -1;
  }

  for (let i = state.replayCursor + 1; i <= clamped; i++) {
    const msg = normalizeReplayEvent(state.replayEvents[i]);
    if (!msg.topic) continue;
    handleMessage(msg);
    state.replayCursor = i;
  }

  updateReplayScrubberUi();
}

function syncTopologyControlUi() {
  if (edgeCountValue) edgeCountValue.textContent = String(state.maxEdgesPerProjection);
  const sampled = state.edgeMode === "sampled";
  if (edgeCountSlider) edgeCountSlider.disabled = !sampled;
  if (strongestOnlyChk) strongestOnlyChk.disabled = !sampled;
  if (edgeCountWrap) edgeCountWrap.style.opacity = sampled ? "1" : "0.55";
}

function handleTopologyEdgeControlChange() {
  if (edgeModeSel) state.edgeMode = edgeModeSel.value;
  if (edgeCountSlider) state.maxEdgesPerProjection = parseInt(edgeCountSlider.value, 10) || 200;
  if (strongestOnlyChk) state.strongestOnly = !!strongestOnlyChk.checked;
  syncTopologyControlUi();
  state.edgeSampleCache = {};
  refreshRenderedConnections();
  drawNetwork();
}

function handleTopologyDetailChange() {
  if (detailedNodesChk) state.detailedNodes = !!detailedNodesChk.checked;
  state.edgeSampleCache = {};
  buildNetworkLayout();
  drawNetwork();
}

function initTopologyControls() {
  if (edgeModeSel) state.edgeMode = edgeModeSel.value || state.edgeMode;
  if (edgeCountSlider) state.maxEdgesPerProjection = parseInt(edgeCountSlider.value, 10) || state.maxEdgesPerProjection;
  if (strongestOnlyChk) state.strongestOnly = !!strongestOnlyChk.checked;
  if (detailedNodesChk) state.detailedNodes = !!detailedNodesChk.checked;

  if (edgeModeSel) edgeModeSel.addEventListener("change", handleTopologyEdgeControlChange);
  if (edgeCountSlider) edgeCountSlider.addEventListener("input", handleTopologyEdgeControlChange);
  if (strongestOnlyChk) strongestOnlyChk.addEventListener("change", handleTopologyEdgeControlChange);
  if (detailedNodesChk) detailedNodesChk.addEventListener("change", handleTopologyDetailChange);
  syncTopologyControlUi();
}

// ── Config panel ─────────────────────────────────────────────────────

const CONFIG_DISPLAY_ORDER = [
  "gate", "gates", "device", "dtype", "seed",
  "n_hidden", "n_inhibitory", "lr",
  "max_trials", "max_epochs", "window_steps", "dt",
  "convergence_streak", "per_pattern_streak",
  "amplitude", "spike_threshold", "w_min", "w_max",
];

const CONFIG_LABELS = {
  gate: "Gate",
  gates: "Gates",
  device: "Device",
  dtype: "Dtype",
  seed: "Seed",
  n_hidden: "Hidden",
  n_inhibitory: "Inhibitory",
  lr: "Learn Rate",
  max_trials: "Max Trials",
  max_epochs: "Max Epochs",
  window_steps: "Window Steps",
  dt: "dt",
  convergence_streak: "Conv. Streak",
  per_pattern_streak: "Pattern Streak",
  amplitude: "Amplitude",
  spike_threshold: "Spike Thresh",
  w_min: "W Min",
  w_max: "W Max",
};

function renderConfigPanel() {
  if (!configGrid || !panelConfig) return;
  const cfg = state.runConfig;
  if (!cfg || Object.keys(cfg).length === 0) {
    panelConfig.style.display = "none";
    return;
  }

  // Build items in display order, then append any remaining keys.
  const seen = new Set();
  const items = [];
  for (const key of CONFIG_DISPLAY_ORDER) {
    if (key in cfg) {
      seen.add(key);
      items.push({ key, value: cfg[key] });
    }
  }
  for (const key of Object.keys(cfg)) {
    if (!seen.has(key)) items.push({ key, value: cfg[key] });
  }

  configGrid.innerHTML = items.map(({ key, value }) => {
    const label = CONFIG_LABELS[key] || key;
    let display = value;
    if (Array.isArray(value)) display = value.join(", ");
    else if (typeof value === "object" && value !== null) display = JSON.stringify(value);
    return `<div class="config-item"><span class="config-key">${label}:</span><span class="config-val">${display}</span></div>`;
  }).join("");
  panelConfig.style.display = "";
}

// ── Stats ────────────────────────────────────────────────────────────

function formatElapsed(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function updateStats() {
  statTrial.textContent = state.totalTrials.toLocaleString();
  statAccuracy.textContent = (state.lastAccuracy * 100).toFixed(1) + "%";
  if (state.converged === true)       statConverged.textContent = "✓ Yes";
  else if (state.converged === false) statConverged.textContent = "✗ No";
  else                                statConverged.textContent = "—";
  statEpoch.textContent = state.epoch > 0 ? state.epoch.toLocaleString() : "—";
  if (statElapsed) {
    if (state.trainStartTime) {
      statElapsed.textContent = formatElapsed(Date.now() - state.trainStartTime);
    } else {
      statElapsed.textContent = "—";
    }
  }
}

// ── Truth table ──────────────────────────────────────────────────────

function updateTruthTableEntry(d) {
  // Normalise the input to a canonical key like "(0, 1)".
  const inp = Array.isArray(d.input) ? d.input : [d.input];
  // For multi-gate, prefix with gate name so each gate×input combo
  // gets its own row.
  const gateName = d.gate || state.gate || "";
  const isMulti = gateName === "MULTI" || (state.gate === "MULTI");
  const rawKey = "(" + inp.join(", ") + ")";
  const key = isMulti && d.gate ? d.gate + " " + rawKey : rawKey;

  if (!state.truthTable[key]) {
    state.truthTable[key] = {
      gate: d.gate || state.gate || "",
      input: rawKey,
      expected: d.expected, last_predicted: d.predicted,
      confidence: 0, total_count: 0, _window: [],
    };
  }
  const e = state.truthTable[key];
  e.total_count++;
  e.last_predicted = d.predicted;
  // Rolling window sized to per_pattern_streak.
  e._window.push(d.correct ? 1 : 0);
  const wSize = state.perPatternStreak || 5;
  if (e._window.length > wSize) e._window.shift();
  e.confidence = e._window.reduce((a, b) => a + b, 0) / e._window.length;
}

function updateNeuronSpikes(d) {
  // Map per-layer spike counts from the training_trial event onto neurons.
  const layerSpikes = {
    input: d.input_spikes || [],
    hidden: d.hidden_spikes || [],
    output: d.output_spikes || [],
  };
  for (const n of state.neurons) {
    const arr = layerSpikes[n.layer];
    if (!arr || arr.length === 0) {
      n.lastSpikes = 0;
      n.trialCount++;
      continue;
    }
    if (n.isAggregate) {
      let total = 0;
      for (const v of arr) total += Number(v || 0);
      n.lastSpikes = total;
      n.totalSpikes += total;
    } else if (n.idx < arr.length) {
      n.lastSpikes = arr[n.idx];
      n.totalSpikes += arr[n.idx];
    } else {
      n.lastSpikes = 0;
    }
    n.trialCount++;
  }
}

function renderTruthTable() {
  const isMulti = state.gate === "MULTI";

  // Build sorted key list.
  let keys;
  if (isMulti) {
    // Sort by gate name then input.
    const gateOrder = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"];
    keys = Object.keys(state.truthTable).sort((a, b) => {
      const ga = gateOrder.indexOf(a.split(" ")[0]);
      const gb = gateOrder.indexOf(b.split(" ")[0]);
      if (ga !== gb) return ga - gb;
      return a.localeCompare(b);
    });
  } else {
    keys = ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"];
  }

  let html = "";
  let lastGate = "";
  for (const key of keys) {
    const e = state.truthTable[key];
    if (!e) continue;

    // In multi mode, insert a gate header row when the gate changes.
    if (isMulti && e.gate && e.gate !== lastGate) {
      lastGate = e.gate;
      html += `<tr class="gate-header"><td colspan="5">${e.gate}</td></tr>`;
    }

    const conf = (e.confidence * 100).toFixed(1);
    const cls = e.last_predicted === e.expected ? "correct" : "wrong";
    const barW = Math.min(e.confidence * 100, 100);
    const displayInput = isMulti ? e.input : key;
    html += `<tr class="${cls}">
      <td>${displayInput}</td>
      <td>${e.expected}</td>
      <td>${e.last_predicted}</td>
      <td>${conf}% <span class="confidence-bar" style="width:${barW}px"></span></td>
      <td>${e.total_count}</td>
    </tr>`;
  }
  truthTbody.innerHTML = html;
}

// ── Log ──────────────────────────────────────────────────────────────

function appendLog(msg) {
  const d = msg.data;
  const cls = d.correct ? "log-correct" : "log-wrong";
  const line = document.createElement("div");
  line.className = cls;
  const gatePfx = d.gate && state.gate === "MULTI" ? `[${d.gate}] ` : "";
  line.textContent = `#${msg.step} | ${gatePfx}in=${JSON.stringify(d.input)} ` +
    `exp=${d.expected} pred=${d.predicted} ` +
    `err=${d.error} acc=${(d.accuracy * 100).toFixed(1)}%`;
  logContainer.appendChild(line);
  // Keep bounded.
  while (logContainer.children.length > 200) {
    logContainer.removeChild(logContainer.firstChild);
  }
  logContainer.scrollTop = logContainer.scrollHeight;
}

// ── Network topology visualisation ───────────────────────────────────

function parseLayerInfo(topology) {
  const layers = topology.layers || [];
  const popHints = topology.populations || {};
  return layers.map((layerLabel, layerIdx) => {
    const m = layerLabel.match(/^([\w-]+)\((\d+)\)$/);
    const name = m ? m[1] : String(layerLabel);
    const count = m ? parseInt(m[2], 10) : 1;
    const hintObj = Array.isArray(popHints)
      ? popHints.find((p) => p && p.name === name)
      : popHints[name];
    return {
      name,
      count,
      layerIdx,
      viz: (hintObj && hintObj.viz) ? hintObj.viz : {},
    };
  });
}

function isInputLikeLayerName(layerName) {
  const s = String(layerName || "").toLowerCase();
  return s.includes("input") || s.includes("pixel") || s.includes("image");
}

function inferGridDims(layer) {
  const viz = layer.viz || {};
  if (viz.layout === "grid" && Number.isFinite(viz.grid_w) && Number.isFinite(viz.grid_h)) {
    return { w: Math.max(1, Math.floor(viz.grid_w)), h: Math.max(1, Math.floor(viz.grid_h)) };
  }
  if (!isInputLikeLayerName(layer.name)) return null;
  const n = layer.count;
  const side = Math.round(Math.sqrt(n));
  // Inference rule: use grid for input-like populations when n is square,
  // with an explicit 64->8x8 special-case for pixel-encoded tasks.
  if (n === 64) return { w: 8, h: 8 };
  if (side * side === n) return { w: side, h: side };
  return null;
}

function shouldCollapseLayer(layer) {
  if (layer.viz && layer.viz.collapsed === false) return false;
  if (state.detailedNodes) return false;
  return layer.count > 32;
}

function makeNeuronLabel(layerName, idx, count, isAggregate) {
  if (isAggregate) return `${layerName} (${count})`;
  return `${layerName}[${idx}]`;
}

function shouldShowNeuronLabel(layerName, idx, count, isAggregate) {
  if (isAggregate) return true;
  const lower = String(layerName).toLowerCase();
  if (lower.includes("output")) return true;
  if (isInputLikeLayerName(layerName)) return false;
  if (lower.includes("hidden")) {
    const step = Math.max(1, Math.ceil(count / 10)); // show ~10 hidden labels max
    return idx % step === 0;
  }
  return idx % 4 === 0;
}

function createDeterministicRandom(seedStr) {
  let h = 2166136261 >>> 0;
  const s = String(seedStr || "");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  let stateVal = h >>> 0;
  return () => {
    stateVal = (Math.imul(1664525, stateVal) + 1013904223) >>> 0;
    return stateVal / 4294967296;
  };
}

function deterministicSample(items, maxItems, seedStr) {
  if (items.length <= maxItems) return items.slice();
  const rnd = createDeterministicRandom(seedStr);
  const idx = Array.from({ length: items.length }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  const keep = idx.slice(0, maxItems).sort((a, b) => a - b);
  return keep.map((i) => items[i]);
}

function flattenWeights(weights) {
  if (!Array.isArray(weights)) return [];
  if (Array.isArray(weights[0])) return weights.flat().map((v) => Number(v || 0));
  return weights.map((v) => Number(v || 0));
}

function buildAllConnections(layerInfo) {
  const all = [];
  const byLayerIdx = {};
  for (const n of state.neurons) {
    if (!byLayerIdx[n.layerIdx]) byLayerIdx[n.layerIdx] = [];
    byLayerIdx[n.layerIdx].push(n);
  }
  const edges = state.topology.edges || [];
  edges.forEach((edge, edgeIdx) => {
    const srcLayer = layerInfo.findIndex((l) => l.name === edge.src);
    const dstLayer = layerInfo.findIndex((l) => l.name === edge.dst);
    if (srcLayer < 0 || dstLayer < 0) return;

    const srcNeurons = byLayerIdx[srcLayer] || [];
    const dstNeurons = byLayerIdx[dstLayer] || [];
    if (srcNeurons.length === 0 || dstNeurons.length === 0) return;

    const projection = edge.name || edge.projection || `${edge.src}->${edge.dst}#${edgeIdx}`;
    const weights = edge.weights;

    if (srcNeurons.length === 1 && dstNeurons.length === 1) {
      const flat = flattenWeights(weights);
      const meanW = flat.length
        ? flat.reduce((acc, v) => acc + Number(v || 0), 0) / flat.length
        : 0;
      all.push({
        projection,
        src: srcNeurons[0].id,
        dst: dstNeurons[0].id,
        weight: Number(meanW || 0),
        hasWeight: flat.length > 0,
        srcNeuron: srcNeurons[0],
        dstNeuron: dstNeurons[0],
      });
      return;
    }

    if (Array.isArray(weights) && Array.isArray(weights[0])) {
      for (let si = 0; si < srcNeurons.length; si++) {
        for (let di = 0; di < dstNeurons.length; di++) {
          const row = weights[si] || [];
          all.push({
            projection,
            src: srcNeurons[si].id,
            dst: dstNeurons[di].id,
            weight: Number(row[di] || 0),
            hasWeight: true,
            srcNeuron: srcNeurons[si],
            dstNeuron: dstNeurons[di],
          });
        }
      }
      return;
    }

    if (Array.isArray(weights)) {
      for (let si = 0; si < srcNeurons.length; si++) {
        for (let di = 0; di < dstNeurons.length; di++) {
          const wIdx = srcNeurons.length > 1 ? si : di;
          all.push({
            projection,
            src: srcNeurons[si].id,
            dst: dstNeurons[di].id,
            weight: Number(weights[wIdx] || 0),
            hasWeight: true,
            srcNeuron: srcNeurons[si],
            dstNeuron: dstNeurons[di],
          });
        }
      }
      return;
    }

    for (const sn of srcNeurons) {
      for (const dn of dstNeurons) {
        all.push({
          projection,
          src: sn.id,
          dst: dn.id,
          weight: 0,
          hasWeight: false,
          srcNeuron: sn,
          dstNeuron: dn,
        });
      }
    }
  });
  return all;
}

function refreshRenderedConnections() {
  if (!state.allConnections || state.allConnections.length === 0) {
    state.connections = [];
    return;
  }
  if (state.edgeMode === "off") {
    state.connections = [];
    return;
  }
  if (state.edgeMode === "full") {
    state.connections = state.allConnections;
    return;
  }

  const maxPerProj = Math.max(1, state.maxEdgesPerProjection || 200);
  const runSeed = state.activeRunId || (state.gate || "live");
  const cacheKey = `${state.topologyVersion}|${runSeed}|${state.edgeMode}|${maxPerProj}|${state.strongestOnly}`;
  if (state.edgeSampleCache[cacheKey]) {
    state.connections = state.edgeSampleCache[cacheKey];
    return;
  }

  const grouped = {};
  for (const c of state.allConnections) {
    if (!grouped[c.projection]) grouped[c.projection] = [];
    grouped[c.projection].push(c);
  }

  const sampled = [];
  for (const [projection, edges] of Object.entries(grouped)) {
    if (edges.length <= maxPerProj) {
      sampled.push(...edges);
      continue;
    }
    const hasExplicitWeights = edges.some((e) => e.hasWeight);
    if (state.strongestOnly && hasExplicitWeights) {
      const top = edges
        .slice()
        .sort((a, b) => Math.abs(Number(b.weight || 0)) - Math.abs(Number(a.weight || 0)))
        .slice(0, maxPerProj);
      sampled.push(...top);
      continue;
    }
    const pick = deterministicSample(edges, maxPerProj, `${runSeed}|${projection}`);
    sampled.push(...pick);
  }

  state.edgeSampleCache[cacheKey] = sampled;
  state.connections = sampled;
}

function buildNetworkLayout() {
  state.neurons = [];
  state.connections = [];
  state.allConnections = [];
  if (!state.topology) return;

  const layerInfo = parseLayerInfo(state.topology);
  const colors = {
    input: "#58a6ff",
    hidden: "#bc8cff",
    output: "#3fb950",
  };

  const { w, h } = setupCanvas(canvasNet);
  const padX = 60;
  const padY = 24;
  const usableW = Math.max(1, w - 2 * padX);
  const usableH = Math.max(1, h - 2 * padY);
  const nLayers = layerInfo.length;
  let id = 0;

  layerInfo.forEach((layer, li) => {
    const centerX = padX + (li / (nLayers - 1 || 1)) * usableW;
    const lower = layer.name.toLowerCase();
    const layerColor = colors[lower] || "#8b949e";
    const collapsed = shouldCollapseLayer(layer);
    const grid = !collapsed ? inferGridDims(layer) : null;

    if (collapsed) {
      state.neurons.push({
        id: id++,
        layer: layer.name,
        layerIdx: li,
        idx: 0,
        x: centerX,
        y: padY + usableH / 2,
        r: 11,
        color: layerColor,
        label: makeNeuronLabel(layer.name, 0, layer.count, true),
        showLabel: true,
        isAggregate: true,
        memberCount: layer.count,
        totalSpikes: 0,
        lastSpikes: 0,
        trialCount: 0,
      });
      return;
    }

    if (grid && grid.w * grid.h >= layer.count) {
      const bandW = Math.min(160, usableW / Math.max(3, nLayers));
      const bandH = Math.min(180, usableH * 0.9);
      const cellW = bandW / grid.w;
      const cellH = bandH / grid.h;
      const dotR = Math.max(2.4, Math.min(6.2, Math.min(cellW, cellH) * 0.28));
      const x0 = centerX - bandW / 2;
      const y0 = padY + (usableH - bandH) / 2;
      for (let ni = 0; ni < layer.count; ni++) {
        const row = Math.floor(ni / grid.w);
        const col = ni % grid.w;
        state.neurons.push({
          id: id++,
          layer: layer.name,
          layerIdx: li,
          idx: ni,
          x: x0 + (col + 0.5) * cellW,
          y: y0 + (row + 0.5) * cellH,
          r: dotR,
          color: layerColor,
          label: makeNeuronLabel(layer.name, ni, layer.count, false),
          showLabel: shouldShowNeuronLabel(layer.name, ni, layer.count, false),
          isAggregate: false,
          totalSpikes: 0,
          lastSpikes: 0,
          trialCount: 0,
        });
      }
      return;
    }

    for (let ni = 0; ni < layer.count; ni++) {
      const y = padY + ((ni + 0.5) / Math.max(1, layer.count)) * usableH;
      const radius = Math.max(3.6, Math.min(10, 16 - layer.count * 0.2));
      state.neurons.push({
        id: id++,
        layer: layer.name,
        layerIdx: li,
        idx: ni,
        x: centerX,
        y,
        r: radius,
        color: layerColor,
        label: makeNeuronLabel(layer.name, ni, layer.count, false),
        showLabel: shouldShowNeuronLabel(layer.name, ni, layer.count, false),
        isAggregate: false,
        totalSpikes: 0,
        lastSpikes: 0,
        trialCount: 0,
      });
    }
  });

  state.allConnections = buildAllConnections(layerInfo);
  state.topologyVersion += 1;
  state.edgeSampleCache = {};
  refreshRenderedConnections();
}

function drawNetwork() {
  const { ctx, w, h } = setupCanvas(canvasNet);
  ctx.clearRect(0, 0, w, h);

  for (const c of state.connections) {
    const s = c.srcNeuron;
    const d = c.dstNeuron;
    const absW = Math.abs(Number(c.weight || 0));
    const alpha = Math.min(0.05 + absW * 0.08, 0.2);
    const lw = Math.max(0.4, Math.min(0.7 + absW * 1.2, 2.4));

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(d.x, d.y);
    ctx.strokeStyle = c.weight >= 0
      ? `rgba(63, 185, 80, ${alpha})`
      : `rgba(248, 81, 73, ${alpha})`;
    ctx.lineWidth = lw;
    ctx.stroke();
  }

  for (const n of state.neurons) {
    const rate = n.trialCount > 0 ? n.totalSpikes / n.trialCount : 0;
    const glowAlpha = Math.min(rate / 20, 0.65);
    if (glowAlpha > 0.05) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 100, ${glowAlpha})`;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.strokeStyle = "#e6edf3";
    ctx.lineWidth = n.isAggregate ? 1.8 : 1.2;
    ctx.stroke();

    if (!n.showLabel) continue;
    ctx.fillStyle = "#e6edf3";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(n.label, n.x, n.y + n.r + 12);
  }
}

// ── Network tooltip on hover ─────────────────────────────────────────

canvasNet.addEventListener("mousemove", (ev) => {
  const rect = canvasNet.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const my = ev.clientY - rect.top;

  // Check neurons.
  for (const n of state.neurons) {
    const dx = mx - n.x, dy = my - n.y;
    if (dx * dx + dy * dy < (n.r + 4) * (n.r + 4)) {
      const avgSpikes = n.trialCount > 0 ? (n.totalSpikes / n.trialCount).toFixed(2) : "0.00";
      const spikeRate = n.trialCount > 0
        ? (n.totalSpikes / n.trialCount / state.windowSteps * 1000).toFixed(1)
        : "0.0";
      tooltip.classList.remove("hidden");
      tooltip.style.left = (ev.clientX - canvasNet.closest(".panel").getBoundingClientRect().left + 12) + "px";
      tooltip.style.top  = (ev.clientY - canvasNet.closest(".panel").getBoundingClientRect().top  + 12) + "px";
      const indexLabel = n.isAggregate ? "aggregate" : String(n.idx);
      tooltip.textContent =
        `${n.layer}\n` +
        `Neuron: ${indexLabel}\n` +
        `Position: (${n.x.toFixed(0)}, ${n.y.toFixed(0)})\n` +
        `Last spikes: ${n.lastSpikes}\n` +
        `Avg spikes/trial: ${avgSpikes}\n` +
        `Spike rate: ${spikeRate} Hz\n` +
        `Trials seen: ${n.trialCount}`;
      return;
    }
  }

  // Check connections (distance to line segment).
  for (const c of state.connections) {
    const s = c.srcNeuron, d = c.dstNeuron;
    const dist = pointToSegDist(mx, my, s.x, s.y, d.x, d.y);
    if (dist < 6) {
      tooltip.classList.remove("hidden");
      tooltip.style.left = (ev.clientX - canvasNet.closest(".panel").getBoundingClientRect().left + 12) + "px";
      tooltip.style.top  = (ev.clientY - canvasNet.closest(".panel").getBoundingClientRect().top  + 12) + "px";
      tooltip.textContent =
        `${s.layer}[${s.idx}] -> ${d.layer}[${d.idx}]\nProjection: ${c.projection}\nWeight: ${Number(c.weight || 0).toFixed(4)}`;
      return;
    }
  }

  tooltip.classList.add("hidden");
});

canvasNet.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));

function pointToSegDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

// ── Accuracy chart ───────────────────────────────────────────────────

function drawAccuracyChart() {
  const { ctx, w, h } = setupCanvas(canvasAcc);
  ctx.clearRect(0, 0, w, h);

  const data = state.accuracyHistory;
  if (data.length < 2) return;

  const padL = 40, padR = 10, padT = 10, padB = 30;
  const cw = w - padL - padR;
  const ch = h - padT - padB;

  // Grid.
  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 0.5;
  for (let y = 0; y <= 1; y += 0.25) {
    const py = padT + ch * (1 - y);
    ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(w - padR, py); ctx.stroke();
    ctx.fillStyle = "#8b949e";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText((y * 100).toFixed(0) + "%", padL - 4, py + 3);
  }

  // Downsample if too many points.
  const maxPts = 600;
  let plotData = data;
  let step = 1;
  if (data.length > maxPts) {
    step = Math.ceil(data.length / maxPts);
    plotData = [];
    for (let i = 0; i < data.length; i += step) plotData.push(data[i]);
  }

  // Line.
  ctx.beginPath();
  ctx.strokeStyle = "#58a6ff";
  ctx.lineWidth = 1.5;
  plotData.forEach((v, i) => {
    const x = padL + (i / (plotData.length - 1)) * cw;
    const y = padT + ch * (1 - v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // X label.
  ctx.fillStyle = "#8b949e";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`Trial (${data.length})`, w / 2, h - 4);
}

// ── Weight chart ─────────────────────────────────────────────────────

function recordWeight(msg) {
  const name = msg.source;
  if (!state.weightHistory[name]) {
    state.weightHistory[name] = { steps: [], weights: [] };
  }
  const rec = state.weightHistory[name];
  rec.steps.push(msg.step);
  // Flatten weights.
  let flat;
  const w = msg.data.weights;
  if (Array.isArray(w) && Array.isArray(w[0])) {
    flat = w.flat();
  } else if (Array.isArray(w)) {
    flat = w;
  } else {
    flat = [w];
  }
  rec.weights.push(flat);
  // Bound.
  if (rec.steps.length > 500) {
    rec.steps.shift();
    rec.weights.shift();
  }
}

function drawWeightChart() {
  const { ctx, w, h } = setupCanvas(canvasWgt);
  ctx.clearRect(0, 0, w, h);

  const projections = Object.entries(state.weightHistory);
  if (projections.length === 0) return;

  const padL = 40, padR = 10, padT = 10, padB = 30;
  const cw = w - padL - padR;
  const ch = h - padT - padB;

  // Flatten all weights to find range.
  let wmin = Infinity, wmax = -Infinity;
  projections.forEach(([, d]) => {
    d.weights.forEach((snap) => {
      snap.forEach((v) => { wmin = Math.min(wmin, v); wmax = Math.max(wmax, v); });
    });
  });
  if (wmin === wmax) { wmin -= 1; wmax += 1; }
  const wRange = wmax - wmin;

  // Grid.
  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 0.5;
  for (let frac = 0; frac <= 1; frac += 0.25) {
    const py = padT + ch * frac;
    ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(w - padR, py); ctx.stroke();
    const val = wmax - frac * wRange;
    ctx.fillStyle = "#8b949e";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(2), padL - 4, py + 3);
  }

  // Colours per weight index.
  const palette = [
    "#58a6ff","#3fb950","#f85149","#d29922",
    "#bc8cff","#f778ba","#79c0ff","#56d364",
    "#ff7b72","#e3b341","#d2a8ff","#ff9bce",
  ];

  projections.forEach(([, d]) => {
    if (d.weights.length < 2) return;
    const n = d.weights[0].length;
    for (let wi = 0; wi < n; wi++) {
      ctx.beginPath();
      ctx.strokeStyle = palette[wi % palette.length];
      ctx.lineWidth = 1;
      d.weights.forEach((snap, si) => {
        const x = padL + (si / (d.weights.length - 1)) * cw;
        const y = padT + ch * (1 - (snap[wi] - wmin) / wRange);
        if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  });

  ctx.fillStyle = "#8b949e";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Weight evolution", w / 2, h - 4);
}

// ── Resize handling ──────────────────────────────────────────────────

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.topology) {
      buildNetworkLayout();
      drawNetwork();
    }
    drawAccuracyChart();
    drawWeightChart();
    drawResourceCharts();
  }, 150);
});

// ── Mode switching ───────────────────────────────────────────────────

function setMode(mode) {
  if (state.mode !== mode) {
    setReplayPlaying(false);
  }
  state.mode = mode;

  // Update button visuals.
  btnModeLive.classList.toggle("mode-active", mode === "live");
  btnModeReplay.classList.toggle("mode-active", mode === "replay");

  // Toggle control groups.
  liveControls.style.display   = mode === "live" ? "" : "none";
  replayControls.style.display = mode === "replay" ? "" : "none";

  // Update URL without reload.
  const url = new URL(window.location);
  url.searchParams.set("mode", mode);
  if (mode === "live") {
    url.searchParams.delete("run");
  }
  window.history.replaceState(null, "", url);

  if (mode === "replay") {
    if (state.ws) {
      state.ws.close();
      state.ws = null;
    }
    setReplayControlsVisible(state.replayEvents.length > 0);
    fetchRunList();
  } else {
    setReplayControlsVisible(false);
    if (!state.ws) connectWS();
  }
}

btnModeLive.addEventListener("click", () => setMode("live"));
btnModeReplay.addEventListener("click", () => setMode("replay"));

// ── Replay: fetch run list ───────────────────────────────────────────

async function fetchRunList() {
  try {
    const resp = await fetch("/api/runs");
    if (!resp.ok) return;
    const runs = await resp.json();
    // Preserve current selection.
    const prevVal = runSelect.value;
    runSelect.innerHTML = '<option value="">— select run —</option>';
    for (const r of runs) {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      const task = r.task_name || "";
      const seed = r.seed != null ? ` seed=${r.seed}` : "";
      const device = r.device || "";
      opt.textContent = `${r.run_id}${task ? "  " + task : ""}${seed}  ${device}`;
      runSelect.appendChild(opt);
    }
    // Restore selection if still present.
    if (prevVal && runSelect.querySelector(`option[value="${prevVal}"]`)) {
      runSelect.value = prevVal;
    }
  } catch (e) {
    console.warn("[replay] failed to fetch run list:", e);
  }
}

btnRefreshRuns.addEventListener("click", fetchRunList);

async function fetchReplayEventPage(runId, start, limit) {
  const resp = await fetch(`/api/run/${runId}/events?start=${start}&limit=${limit}`);
  if (!resp.ok) return null;
  return await resp.json();
}

async function loadReplayEvents(runId) {
  try {
    const idxResp = await fetch(`/api/run/${runId}/events/index`);
    if (!idxResp.ok) return false;
    const idxPayload = await idxResp.json();
    const indexRows = Array.isArray(idxPayload.items) ? idxPayload.items : [];
    if (indexRows.length === 0) return false;

    const events = [];
    const pageSize = 500;
    let cursor = 0;
    while (cursor < indexRows.length) {
      const page = await fetchReplayEventPage(runId, cursor, pageSize);
      if (!page || !Array.isArray(page.events) || page.events.length === 0) break;
      events.push(...page.events);
      cursor += page.events.length;
      if (typeof page.total === "number" && cursor >= page.total) break;
    }
    if (events.length === 0) return false;

    state.replayEvents = events;
    state.replayIndex = indexRows;
    state.replayCursor = -1;
    setReplayControlsVisible(true);
    replaySeek(0);
    return true;
  } catch (e) {
    console.warn("[replay] failed to load replay events:", e);
    return false;
  }
}

// ── Replay: load a run ───────────────────────────────────────────────

async function loadRun(runId) {
  if (!runId) return;
  setReplayPlaying(false);
  setStatus("loading");

  try {
    const [metaResp, cfgResp, topoResp, scalarsResp] = await Promise.all([
      fetch(`/api/run/${runId}/meta`),
      fetch(`/api/run/${runId}/config`),
      fetch(`/api/run/${runId}/topology`),
      fetch(`/api/run/${runId}/scalars`),
    ]);

    if (!scalarsResp.ok) {
      setStatus("failed");
      alert(`Failed to load run: ${runId}`);
      return;
    }

    // Reset dashboard state for fresh replay.
    resetDashboardState({ preserveReplay: false });

    state.activeRunId = runId;

    // Parse responses.
    const meta    = metaResp.ok   ? await metaResp.json()    : null;
    const config  = cfgResp.ok    ? await cfgResp.json()     : null;
    const topo    = topoResp.ok   ? await topoResp.json()    : null;
    const scalars = await scalarsResp.json();

    // Render run info in log panel (compact summary).
    if (meta) {
      appendLogText(`Run: ${meta.run_id}`, "log-correct");
      appendLogText(`Device: ${meta.device}  Seed: ${meta.seed}`, "log-correct");
      if (meta.started_at) {
        appendLogText(`Started: ${meta.started_at}`, "log-correct");
      }
    }

    // Show config in dedicated panel.
    if (config) {
      state.runConfig = config;
      renderConfigPanel();
    }

    // Topology.
    if (topo) {
      state.topology = topo;
      buildNetworkLayout();
      drawNetwork();
    }

    // Primary replay path: recorded events. Fallback: scalars only.
    const loadedEvents = await loadReplayEvents(runId);
    if (!loadedEvents) {
      setReplayControlsVisible(false);
      renderReplayScalars(scalars);
      setStatus("replay");
    } else {
      setStatus("replay");
    }

    // Update URL.
    const url = new URL(window.location);
    url.searchParams.set("run", runId);
    window.history.replaceState(null, "", url);

  } catch (e) {
    console.error("[replay] error loading run:", e);
    setStatus("failed");
  }
}

function resetDashboardState(opts = {}) {
  const preserveReplay = !!opts.preserveReplay;
  if (state._elapsedTimer) {
    clearInterval(state._elapsedTimer);
    state._elapsedTimer = null;
  }
  state.training = false;
  state.trainStartTime = null;
  state.accuracyHistory = [];
  state.truthTable = {};
  state.weightHistory = {};
  state.totalTrials = 0;
  state.converged = null;
  state.lastAccuracy = 0;
  state.topology = null;
  state.neurons = [];
  state.connections = [];
  state.allConnections = [];
  state.edgeSampleCache = {};
  state.topologyVersion = 0;
  state.activeRunId = null;
  state.runConfig = null;
  state.replayCursor = -1;
  resetResourceSeries();
  if (!preserveReplay) {
    state.replayEvents = [];
    state.replayIndex = [];
    setReplayPlaying(false);
    setReplayControlsVisible(false);
  } else {
    setReplayPlaying(false);
  }
  logContainer.innerHTML = "";
  truthTbody.innerHTML = "";
  if (panelConfig) panelConfig.style.display = "none";
  updateStats();
  drawAccuracyChart();
  drawWeightChart();
  updateResourceNote();
  drawResourceCharts();
  drawNetwork();
  updateReplayScrubberUi();
}

function renderReplayScalars(scalars) {
  if (!scalars || scalars.length === 0) return;

  // Determine gate type — if there are multiple distinct gates, it's MULTI.
  const gateSet = new Set(scalars.map((r) => r.gate).filter(Boolean));
  const isMulti = gateSet.size > 1;
  state.gate = isMulti ? "MULTI" : (scalars[0].gate || "?");
  statGate.textContent = state.gate;

  for (const row of scalars) {
    ingestResourceMetrics(row, row.trial != null ? Number(row.trial) : state.totalTrials, { deferDraw: true });

    const acc = row.accuracy != null ? row.accuracy : 0;
    state.accuracyHistory.push(acc);
    state.totalTrials++;
    state.lastAccuracy = acc;

    // Build truth table entries if we have enough data.
    if (row.gate) {
      // We don't have input/expected/predicted in scalars.csv for
      // per-pattern truth table, but we can show per-gate accuracy.
      const key = row.gate;
      if (!state.truthTable[key]) {
        state.truthTable[key] = {
          gate: row.gate,
          input: "—",
          expected: "—",
          last_predicted: "—",
          confidence: 0,
          total_count: 0,
          _window: [],
        };
      }
      const e = state.truthTable[key];
      e.total_count++;
      const correct = row.correct === true || row.correct === 1;
      e._window.push(correct ? 1 : 0);
      if (e._window.length > 20) e._window.shift();
      e.confidence = e._window.reduce((a, b) => a + b, 0) / e._window.length;
      e.last_predicted = correct ? "✓" : "✗";
      e.expected = "—";
    }
  }

  // Determine convergence from last few scalars.
  if (scalars.length > 10) {
    const last10 = scalars.slice(-10);
    const allCorrect = last10.every((r) => r.correct === true || r.correct === 1);
    const highAcc = last10.every((r) => r.accuracy != null && r.accuracy >= 0.95);
    state.converged = allCorrect || highAcc ? true : false;
  }

  updateStats();
  drawAccuracyChart();
  renderReplayTruthTable();
  updateResourceNote();
  drawResourceCharts();
}

function renderReplayTruthTable() {
  // For replay, show per-gate summary (we don't have per-pattern input data).
  const gateOrder = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"];
  const keys = Object.keys(state.truthTable).sort((a, b) => {
    const ia = gateOrder.indexOf(a);
    const ib = gateOrder.indexOf(b);
    if (ia >= 0 && ib >= 0) return ia - ib;
    return a.localeCompare(b);
  });

  let html = "";
  for (const key of keys) {
    const e = state.truthTable[key];
    if (!e) continue;
    const conf = (e.confidence * 100).toFixed(1);
    const cls = e.confidence >= 0.8 ? "correct" : "wrong";
    const barW = Math.min(e.confidence * 100, 100);
    html += `<tr class="${cls}">
      <td>${e.gate}</td>
      <td>—</td>
      <td>${e.last_predicted}</td>
      <td>${conf}% <span class="confidence-bar" style="width:${barW}px"></span></td>
      <td>${e.total_count}</td>
    </tr>`;
  }
  truthTbody.innerHTML = html;
}

function appendLogText(text, cls) {
  const line = document.createElement("div");
  line.className = cls || "";
  line.textContent = text;
  logContainer.appendChild(line);
}

btnLoadRun.addEventListener("click", () => {
  const runId = runSelect.value;
  if (runId) loadRun(runId);
});

if (btnReplayPlay) {
  btnReplayPlay.addEventListener("click", () => {
    setReplayPlaying(!state.replayPlaying);
  });
}

if (replayScrubber) {
  replayScrubber.addEventListener("input", () => {
    setReplayPlaying(false);
    const idx = parseInt(replayScrubber.value, 10);
    replaySeek(Number.isFinite(idx) ? idx : 0);
  });
}

// ── Live → Replay handoff ────────────────────────────────────────────

function showRunInfoBanner(runId) {
  if (!runInfoBanner || !runId) return;
  state.activeRunId = runId;
  runInfoText.textContent = `Recording to: artifacts/${runId}`;
  runInfoReplayLink.href = `/?mode=replay&run=${runId}`;
  runInfoBanner.style.display = "";
}

function hideRunInfoBanner() {
  if (runInfoBanner) runInfoBanner.style.display = "none";
}

// ── Init ─────────────────────────────────────────────────────────────

// Check URL params.
(function init() {
  const params = new URLSearchParams(window.location.search);
  const mode = params.get("mode");
  const runId = params.get("run");
  initTopologyControls();
  updateResourceNote();
  drawResourceCharts();
  setReplayControlsVisible(false);
  updateReplayScrubberUi();

  if (mode === "replay") {
    setMode("replay");
    if (runId) {
      // Auto-load the specified run after the run list is fetched.
      fetchRunList().then(() => {
        runSelect.value = runId;
        loadRun(runId);
      });
    }
  } else {
    // Default: live mode.
    setMode("live");
  }

  // In live mode, always connect WS.
  if (mode !== "replay") {
    // Already called above.
  }
})();

// Always connect WS in live mode (reconnects are handled inside connectWS).
if (state.mode === "live" && !state.ws) {
  connectWS();
}
