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
  windowSteps: 50,      // sim steps per trial (for computing spike rates)

  // Network visualisation
  neurons: [],          // { id, layer, idx, x, y, r, color, label, totalSpikes, trialCount }
  connections: [],      // { src, dst, weight, srcNeuron, dstNeuron }

  // Mode: "live" | "replay"
  mode: "live",
  activeRunId: null,    // run_id of the currently loaded replay (or live run)
};

// ── DOM refs ─────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const btnTrain    = $("#btn-train");
const btnStop     = $("#btn-stop");
const gateSelect  = $("#gate-select");
const maxTrials   = $("#max-trials");
const statusBadge = $("#status-badge");

const canvasNet   = $("#canvas-network");
const canvasAcc   = $("#canvas-accuracy");
const canvasWgt   = $("#canvas-weights");
const tooltip     = $("#tooltip");
const truthTbody  = $("#truth-table tbody");
const logContainer= $("#log-container");

const statTrial   = $("#stat-trial");
const statAccuracy= $("#stat-accuracy");
const statConverged=$("#stat-converged");
const statGate    = $("#stat-gate");

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
    return;
  }

  switch (msg.topic) {
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
      if (state.totalTrials % 20 === 0) drawWeightChart();
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
      btnTrain.disabled = false;
      btnStop.disabled = true;
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

  const resp = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gate, max_trials: trials }),
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

// ── Stats ────────────────────────────────────────────────────────────

function updateStats() {
  statTrial.textContent = state.totalTrials.toLocaleString();
  statAccuracy.textContent = (state.lastAccuracy * 100).toFixed(1) + "%";
  if (state.converged === true)       statConverged.textContent = "✓ Yes";
  else if (state.converged === false) statConverged.textContent = "✗ No";
  else                                statConverged.textContent = "—";
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
    if (arr && n.idx < arr.length) {
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

function buildNetworkLayout() {
  state.neurons = [];
  state.connections = [];
  if (!state.topology) return;

  const layers = state.topology.layers; // e.g. ["input(2)", "hidden(6)", "output(1)"]
  const layerInfo = layers.map((l) => {
    const m = l.match(/^(\w+)\((\d+)\)$/);
    return m ? { name: m[1], count: parseInt(m[2], 10) } : { name: l, count: 1 };
  });

  const colors = {
    input:  "#58a6ff",
    hidden: "#bc8cff",
    output: "#3fb950",
  };

  const { w, h } = setupCanvas(canvasNet);
  const padX = 60, padY = 30;
  const usableW = w - 2 * padX;
  const usableH = h - 2 * padY;

  // Position neurons.
  const nLayers = layerInfo.length;
  let id = 0;
  layerInfo.forEach((layer, li) => {
    const x = padX + (li / (nLayers - 1 || 1)) * usableW;
    for (let ni = 0; ni < layer.count; ni++) {
      const y = padY + ((ni + 0.5) / layer.count) * usableH;
      state.neurons.push({
        id: id++,
        layer: layer.name,
        layerIdx: li,
        idx: ni,
        x, y,
        r: Math.max(8, 18 - layer.count),
        color: colors[layer.name] || "#8b949e",
        label: `${layer.name}[${ni}]`,
        totalSpikes: 0,
        lastSpikes: 0,
        trialCount: 0,
      });
    }
  });

  // Build connections from topology edges.
  const edges = state.topology.edges || [];
  edges.forEach((edge) => {
    const srcLayer = layerInfo.findIndex((l) => l.name === edge.src);
    const dstLayer = layerInfo.findIndex((l) => l.name === edge.dst);
    if (srcLayer < 0 || dstLayer < 0) return;

    const srcNeurons = state.neurons.filter((n) => n.layerIdx === srcLayer);
    const dstNeurons = state.neurons.filter((n) => n.layerIdx === dstLayer);

    const weights = edge.weights;
    if (Array.isArray(weights) && Array.isArray(weights[0])) {
      // 2D weights [src, dst].
      for (let si = 0; si < srcNeurons.length; si++) {
        for (let di = 0; di < dstNeurons.length; di++) {
          state.connections.push({
            src: srcNeurons[si].id,
            dst: dstNeurons[di].id,
            weight: weights[si][di],
            srcNeuron: srcNeurons[si],
            dstNeuron: dstNeurons[di],
          });
        }
      }
    } else if (Array.isArray(weights)) {
      // 1D weights — one per dst or one per src.
      for (let si = 0; si < srcNeurons.length; si++) {
        for (let di = 0; di < dstNeurons.length; di++) {
          const wIdx = srcNeurons.length > 1 ? si : di;
          state.connections.push({
            src: srcNeurons[si].id,
            dst: dstNeurons[di].id,
            weight: weights[wIdx] || 0,
            srcNeuron: srcNeurons[si],
            dstNeuron: dstNeurons[di],
          });
        }
      }
    }
  });
}

function drawNetwork() {
  const { ctx, w, h } = setupCanvas(canvasNet);
  ctx.clearRect(0, 0, w, h);

  // Draw connections.
  state.connections.forEach((c) => {
    const s = c.srcNeuron;
    const d = c.dstNeuron;
    const absW = Math.abs(c.weight);
    const alpha = Math.min(0.2 + absW * 0.4, 0.9);
    const lw = Math.max(0.5, Math.min(absW * 3, 5));

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(d.x, d.y);
    ctx.strokeStyle = c.weight >= 0
      ? `rgba(63, 185, 80, ${alpha})`
      : `rgba(248, 81, 73, ${alpha})`;
    ctx.lineWidth = lw;
    ctx.stroke();
  });

  // Draw neurons.
  state.neurons.forEach((n) => {
    // Glow effect for recently-active neurons.
    const rate = n.trialCount > 0 ? n.totalSpikes / n.trialCount : 0;
    const glowAlpha = Math.min(rate / 20, 0.7);
    if (glowAlpha > 0.05) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 6, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 100, ${glowAlpha})`;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.strokeStyle = "#e6edf3";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Label.
    ctx.fillStyle = "#e6edf3";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(n.label, n.x, n.y + n.r + 14);
  });
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
      tooltip.textContent =
        `${n.label}\n` +
        `Layer: ${n.layer}  |  Index: ${n.idx}\n` +
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
        `${s.label} → ${d.label}\nWeight: ${c.weight.toFixed(4)}`;
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

    // Render config info in log panel.
    if (meta) {
      appendLogText(`Run: ${meta.run_id}`, "log-correct");
      appendLogText(`Device: ${meta.device}  Seed: ${meta.seed}`, "log-correct");
      if (meta.started_at) {
        appendLogText(`Started: ${meta.started_at}`, "log-correct");
      }
    }
    if (config) {
      const cfgStr = Object.entries(config)
        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
        .join("  ");
      appendLogText(`Config: ${cfgStr}`, "log-correct");
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
  state.activeRunId = null;
  logContainer.innerHTML = "";
  truthTbody.innerHTML = "";
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
