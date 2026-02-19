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
const statusBadge = $("#status-badge");

const canvasNet   = $("#canvas-network");
const canvasAcc   = $("#canvas-accuracy");
const canvasWgt   = $("#canvas-weights");
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
      state.trainStartTime = Date.now();
      if (state._elapsedTimer) clearInterval(state._elapsedTimer);
      state._elapsedTimer = setInterval(() => {
        if (statElapsed) statElapsed.textContent = formatElapsed(Date.now() - state.trainStartTime);
      }, 1000);
      state.weightHistory = {};
      // Reset per-neuron spike stats.
      for (const n of state.neurons) {
        n.totalSpikes = 0;
        n.lastSpikes = 0;
        n.trialCount = 0;
      }
      logContainer.innerHTML = "";
      setStatus("training");
      statGate.textContent = state.gate;
      btnTrain.disabled = true;
      btnStop.disabled = false;
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
      btnTrain.disabled = false;
      btnStop.disabled = true;
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

  const resp = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gate, max_trials: trials, device }),
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
  }, 150);
});

// ── Mode switching ───────────────────────────────────────────────────

function setMode(mode) {
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
    fetchRunList();
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

// ── Replay: load a run ───────────────────────────────────────────────

async function loadRun(runId) {
  if (!runId) return;
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
    resetDashboardState();

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

    // Process scalars into accuracy history + truth table.
    renderReplayScalars(scalars);

    // Update URL.
    const url = new URL(window.location);
    url.searchParams.set("run", runId);
    window.history.replaceState(null, "", url);

    setStatus("replay");
  } catch (e) {
    console.error("[replay] error loading run:", e);
    setStatus("failed");
  }
}

function resetDashboardState() {
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
  logContainer.innerHTML = "";
  truthTbody.innerHTML = "";
  if (panelConfig) panelConfig.style.display = "none";
}

function renderReplayScalars(scalars) {
  if (!scalars || scalars.length === 0) return;

  // Determine gate type — if there are multiple distinct gates, it's MULTI.
  const gateSet = new Set(scalars.map((r) => r.gate).filter(Boolean));
  const isMulti = gateSet.size > 1;
  state.gate = isMulti ? "MULTI" : (scalars[0].gate || "?");
  statGate.textContent = state.gate;

  for (const row of scalars) {
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
    connectWS();
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
