/* NeuroForge - Topology view renderers
   Canvas and Three.js renderers used by the dedicated Topology tab. */
"use strict";

(function () {
  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function escText(value) {
    return String(value === null || value === undefined ? "" : value);
  }

  function clamp01(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1, n));
  }

  function formatCount(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
    if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "k";
    return String(Math.round(n));
  }

  function layerRows(topology, mapped) {
    if (mapped && Array.isArray(mapped.nodes) && mapped.nodes.length > 0) {
      return mapped.nodes.map((node) => ({
        name: node.data.id,
        type: node.data.type || "hidden",
        neurons: Number(node.data.neuronCount || 0),
        params: Number(node.data.paramCount || 0),
        color: node.data.borderColor || "#58a6ff",
      }));
    }
    const layers = (topology && topology.layers) || [];
    return layers.map((label, idx) => {
      const match = String(label).match(/^([\w-]+)\((.+)\)$/);
      const name = match ? match[1] : String(label);
      const inner = match ? match[2] : "0";
      const count = /^\d+$/.test(inner) ? Number(inner) : 1;
      return {
        name,
        type: idx === 0 ? "input" : idx === layers.length - 1 ? "output" : "hidden",
        neurons: count,
        params: 0,
        color: idx === 0 ? "#58a6ff" : idx === layers.length - 1 ? "#3fb950" : "#bc8cff",
      };
    });
  }

  function projectionRows(topology, mapped) {
    const edgeRows = (topology && Array.isArray(topology.edges)) ? topology.edges : [];
    const metaRows = (topology && Array.isArray(topology.projection_meta))
      ? topology.projection_meta
      : [];
    const rows = [];
    for (const edge of edgeRows) {
      const name = edge.name || edge.projection || `${edge.src}_${edge.dst}`;
      const meta = metaRows.find((row) => {
        const mName = row.name || row.projection || `${row.src}_${row.dst}`;
        return normName(mName) === normName(name)
          || (row.src === edge.src && row.dst === edge.dst);
      }) || {};
      rows.push({ ...meta, ...edge, name });
    }
    if (rows.length > 0) return rows;
    if (mapped && Array.isArray(mapped.edges)) {
      return mapped.edges.map((edge) => ({
        name: edge.data.projectionName || edge.data.id,
        src: edge.data.source,
        dst: edge.data.target,
        n_edges: edge.data.nEdges,
      }));
    }
    return metaRows.map((row) => ({ ...row, name: row.name || `${row.src}_${row.dst}` }));
  }

  function normName(value) {
    return String(value || "").replace(/[- ]/g, "_");
  }

  function traceLayerMap(trace) {
    const out = {};
    for (const row of (trace && trace.layers) || []) {
      out[String(row.name || "")] = row;
    }
    return out;
  }

  function traceProjectionMap(trace) {
    const out = {};
    for (const row of (trace && trace.projections) || []) {
      out[normName(row.name)] = row;
    }
    return out;
  }

  class LayerStackRenderer {
    constructor(canvas) {
      this.canvas = canvas;
    }

    render(topology, mapped, trace) {
      if (!this.canvas) return;
      const { ctx, w, h } = setupCanvas(this.canvas);
      ctx.clearRect(0, 0, w, h);
      const layers = layerRows(topology, mapped);
      const activity = traceLayerMap(trace);
      if (layers.length === 0) {
        drawEmpty(ctx, w, h, "No topology loaded.");
        return;
      }

      const pad = 34;
      const gap = Math.max(22, Math.min(56, (w - pad * 2) / Math.max(1, layers.length) * 0.18));
      const planeW = Math.max(82, Math.min(180, (w - pad * 2 - gap * (layers.length - 1)) / layers.length));
      const maxNeurons = Math.max(1, ...layers.map((l) => l.neurons || 1));
      const baseY = h * 0.54;

      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(139, 148, 158, 0.18)";
      for (let y = 40; y < h; y += 36) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      const centers = [];
      layers.forEach((layer, idx) => {
        const x = pad + idx * (planeW + gap) + planeW / 2;
        const logN = Math.log2(Math.max(2, layer.neurons || 1));
        const logMax = Math.log2(Math.max(2, maxNeurons));
        const planeH = Math.max(70, Math.min(h * 0.62, 80 + (logN / logMax) * h * 0.34));
        const y = baseY - planeH / 2;
        const act = clamp01((activity[layer.name] || {}).activity);
        centers.push({ x, y: baseY, layer });

        ctx.save();
        ctx.translate(x, y);
        ctx.fillStyle = "rgba(22, 27, 34, 0.94)";
        ctx.strokeStyle = layer.color;
        ctx.lineWidth = 2 + act * 5;
        roundRect(ctx, -planeW / 2, 0, planeW, planeH, 8);
        ctx.fill();
        ctx.stroke();

        const fillH = planeH * act;
        if (fillH > 1) {
          ctx.fillStyle = `rgba(242, 204, 96, ${0.18 + act * 0.35})`;
          roundRect(ctx, -planeW / 2 + 3, planeH - fillH + 3, planeW - 6, fillH - 6, 6);
          ctx.fill();
        }

        drawMiniGrid(ctx, -planeW / 2 + 12, 24, planeW - 24, Math.max(22, planeH - 76), layer, act);
        ctx.fillStyle = "#e6edf3";
        ctx.font = "600 12px monospace";
        ctx.textAlign = "center";
        ctx.fillText(layer.name.toUpperCase(), 0, -10);
        ctx.font = "11px monospace";
        ctx.fillStyle = "#8b949e";
        ctx.fillText(`${formatCount(layer.neurons)} neurons`, 0, planeH + 18);
        const traceRow = activity[layer.name];
        if (traceRow) {
          ctx.fillStyle = "#f2cc60";
          ctx.fillText(`${formatCount(traceRow.latest_spikes)} spikes`, 0, planeH + 34);
        }
        ctx.restore();
      });

      ctx.strokeStyle = "rgba(88, 166, 255, 0.38)";
      ctx.lineWidth = 2;
      for (let i = 0; i < centers.length - 1; i++) {
        const a = centers[i];
        const b = centers[i + 1];
        ctx.beginPath();
        ctx.moveTo(a.x + planeW / 2, a.y);
        ctx.bezierCurveTo(a.x + gap * 0.5, a.y - 60, b.x - gap * 0.5, b.y - 60, b.x - planeW / 2, b.y);
        ctx.stroke();
      }
    }
  }

  class MatrixRenderer {
    constructor(canvas) {
      this.canvas = canvas;
    }

    render(topology, mapped, trace, projectionName, sampleLimit) {
      if (!this.canvas) return;
      const { ctx, w, h } = setupCanvas(this.canvas);
      ctx.clearRect(0, 0, w, h);
      const projections = projectionRows(topology, mapped);
      if (projections.length === 0) {
        drawEmpty(ctx, w, h, "No projections available.");
        return;
      }
      const projection = projections.find((p) => normName(p.name) === normName(projectionName))
        || projections[0];
      const traceProj = traceProjectionMap(trace)[normName(projection.name)];
      const matrix = buildMatrix(projection, traceProj, Math.max(8, Number(sampleLimit || 64)));

      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, w, h);
      const title = `${projection.name || "projection"}  ${projection.src || "?"} -> ${projection.dst || "?"}`;
      ctx.fillStyle = "#e6edf3";
      ctx.font = "600 13px monospace";
      ctx.fillText(title, 18, 24);
      ctx.fillStyle = "#8b949e";
      ctx.font = "11px monospace";
      ctx.fillText(`${formatCount(projection.n_edges || matrix.values.length)} edges sampled`, 18, 42);

      if (matrix.values.length === 0) {
        drawEmpty(ctx, w, h, "No weight or trace sample for this projection.");
        return;
      }

      const left = 54;
      const top = 62;
      const plotW = Math.max(80, w - left - 28);
      const plotH = Math.max(80, h - top - 42);
      const rows = matrix.rows;
      const cols = matrix.cols;
      const cell = Math.max(2, Math.min(plotW / cols, plotH / rows));
      const gridW = cell * cols;
      const gridH = cell * rows;
      const maxAbs = Math.max(1e-9, ...matrix.values.map((v) => Math.abs(Number(v.value || 0))));

      ctx.strokeStyle = "rgba(139, 148, 158, 0.35)";
      ctx.strokeRect(left - 0.5, top - 0.5, gridW + 1, gridH + 1);
      for (const entry of matrix.values) {
        const x = left + entry.col * cell;
        const y = top + entry.row * cell;
        const val = Number(entry.value || 0);
        const intensity = Math.min(1, Math.abs(val) / maxAbs);
        ctx.fillStyle = val >= 0
          ? `rgba(63, 185, 80, ${0.12 + intensity * 0.78})`
          : `rgba(248, 81, 73, ${0.12 + intensity * 0.78})`;
        ctx.fillRect(x, y, Math.max(1, cell - 0.25), Math.max(1, cell - 0.25));
      }
      if (traceProj && Array.isArray(traceProj.sample_edges)) {
        ctx.strokeStyle = "#f2cc60";
        ctx.lineWidth = Math.max(1, cell * 0.16);
        for (const edge of traceProj.sample_edges) {
          const row = Number(edge.pre || 0) % rows;
          const col = Number(edge.post || 0) % cols;
          ctx.strokeRect(left + col * cell + 1, top + row * cell + 1, cell - 2, cell - 2);
        }
      }
      ctx.fillStyle = "#8b949e";
      ctx.font = "10px monospace";
      ctx.fillText("pre", left, top + gridH + 24);
      ctx.save();
      ctx.translate(22, top + gridH);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("post", 0, 0);
      ctx.restore();
    }
  }

  class TraceTimelineRenderer {
    constructor(canvas) {
      this.canvas = canvas;
    }

    render(frames, selectedTrace) {
      if (!this.canvas) return;
      const { ctx, w, h } = setupCanvas(this.canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, w, h);
      const data = Array.isArray(frames) ? frames.slice(-120) : [];
      if (data.length === 0) {
        drawEmpty(ctx, w, h, "No trace frames yet.");
        return;
      }
      const padL = 42;
      const padT = 28;
      const padB = 30;
      const plotW = Math.max(1, w - padL - 18);
      const plotH = Math.max(1, h - padT - padB);
      const layers = [...new Set(data.flatMap((frame) => (frame.layers || []).map((l) => l.name)))];
      const rowH = plotH / Math.max(1, layers.length);

      ctx.fillStyle = "#e6edf3";
      ctx.font = "600 13px monospace";
      ctx.fillText("Topology Trace Timeline", 16, 20);

      layers.forEach((layer, li) => {
        const y = padT + li * rowH;
        ctx.fillStyle = "#8b949e";
        ctx.font = "10px monospace";
        ctx.fillText(layer, 8, y + rowH * 0.62);
        ctx.strokeStyle = "rgba(139, 148, 158, 0.16)";
        ctx.beginPath();
        ctx.moveTo(padL, y + rowH);
        ctx.lineTo(w - 12, y + rowH);
        ctx.stroke();
      });

      data.forEach((frame, fi) => {
        const x = padL + (fi / Math.max(1, data.length - 1)) * plotW;
        const layerMap = traceLayerMap(frame);
        layers.forEach((layer, li) => {
          const act = clamp01((layerMap[layer] || {}).activity);
          if (act <= 0) return;
          const barH = Math.max(2, rowH * act);
          ctx.fillStyle = `rgba(242, 204, 96, ${0.16 + act * 0.72})`;
          ctx.fillRect(x - 2, padT + li * rowH + rowH - barH, 4, barH);
        });
      });

      const selectedStep = selectedTrace ? selectedTrace.step : data[data.length - 1].step;
      const selectedIdx = data.findIndex((frame) => Number(frame.step) === Number(selectedStep));
      const sx = padL + ((selectedIdx < 0 ? data.length - 1 : selectedIdx) / Math.max(1, data.length - 1)) * plotW;
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      ctx.moveTo(sx, padT);
      ctx.lineTo(sx, h - padB);
      ctx.stroke();

      const last = selectedTrace || data[data.length - 1];
      ctx.fillStyle = "#8b949e";
      ctx.font = "11px monospace";
      ctx.fillText(`step ${last.step}  mode ${last.mode || "activity"}`, padL, h - 10);
    }
  }

  class TopologyThreeRenderer {
    constructor(container) {
      this.container = container;
      this.scene = null;
      this.camera = null;
      this.renderer = null;
      this.group = null;
      this.animation = null;
      this.drag = null;
      this.rotation = { x: -0.45, y: 0.55 };
      this.data = null;
    }

    render(topology, mapped, trace) {
      if (!this.container) return;
      this.data = { topology, mapped, trace };
      if (typeof THREE === "undefined") {
        this.container.innerHTML = "<div class=\"topo-status topo-status-empty\">Three.js not loaded.</div>";
        return;
      }
      this._ensureScene();
      this._rebuildScene();
      this._start();
    }

    pause() {
      if (this.animation) {
        cancelAnimationFrame(this.animation);
        this.animation = null;
      }
    }

    destroy() {
      this.pause();
      if (this.renderer) {
        this.renderer.dispose();
        this.renderer.domElement.remove();
      }
      this.scene = null;
      this.camera = null;
      this.renderer = null;
      this.group = null;
    }

    _ensureScene() {
      if (this.renderer) return;
      const THREE_NS = THREE;
      this.container.innerHTML = "";
      const rect = this.container.getBoundingClientRect();
      this.scene = new THREE_NS.Scene();
      this.scene.background = new THREE_NS.Color(0x0d1117);
      this.camera = new THREE_NS.PerspectiveCamera(
        45,
        Math.max(1, rect.width) / Math.max(1, rect.height),
        0.1,
        2000,
      );
      this.camera.position.set(0, 120, 340);
      this.renderer = new THREE_NS.WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.setPixelRatio(window.devicePixelRatio || 1);
      this.renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height));
      this.container.appendChild(this.renderer.domElement);
      this.group = new THREE_NS.Group();
      this.scene.add(this.group);
      this.scene.add(new THREE_NS.AmbientLight(0xffffff, 0.72));
      const light = new THREE_NS.DirectionalLight(0xffffff, 0.55);
      light.position.set(100, 180, 140);
      this.scene.add(light);
      this._bindInput();
    }

    _bindInput() {
      const canvas = this.renderer.domElement;
      canvas.addEventListener("pointerdown", (ev) => {
        this.drag = { x: ev.clientX, y: ev.clientY, rx: this.rotation.x, ry: this.rotation.y };
        canvas.setPointerCapture(ev.pointerId);
      });
      canvas.addEventListener("pointermove", (ev) => {
        if (!this.drag) return;
        this.rotation.y = this.drag.ry + (ev.clientX - this.drag.x) * 0.008;
        this.rotation.x = this.drag.rx + (ev.clientY - this.drag.y) * 0.008;
      });
      canvas.addEventListener("pointerup", () => {
        this.drag = null;
      });
      canvas.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        this.camera.position.z = Math.max(120, Math.min(620, this.camera.position.z + ev.deltaY * 0.45));
      }, { passive: false });
      window.addEventListener("resize", () => this._resize());
    }

    _resize() {
      if (!this.renderer || !this.camera || !this.container) return;
      const rect = this.container.getBoundingClientRect();
      this.camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height));
    }

    _rebuildScene() {
      const THREE_NS = THREE;
      while (this.group.children.length > 0) {
        const child = this.group.children.pop();
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      }
      const layers = layerRows(this.data.topology, this.data.mapped);
      const traceLayers = traceLayerMap(this.data.trace);
      const traceProjs = traceProjectionMap(this.data.trace);
      if (layers.length === 0) return;

      const spacing = 86;
      const start = -((layers.length - 1) * spacing) / 2;
      const maxNeurons = Math.max(1, ...layers.map((l) => l.neurons || 1));
      const positions = {};

      layers.forEach((layer, idx) => {
        const x = start + idx * spacing;
        const act = clamp01((traceLayers[layer.name] || {}).activity);
        const size = 36 + Math.log2(Math.max(2, layer.neurons || 1))
          / Math.log2(Math.max(2, maxNeurons)) * 42;
        const mat = new THREE_NS.MeshStandardMaterial({
          color: new THREE_NS.Color(layer.color || "#58a6ff"),
          transparent: true,
          opacity: 0.22 + act * 0.35,
          side: THREE_NS.DoubleSide,
          emissive: new THREE_NS.Color(act > 0 ? "#f2cc60" : "#000000"),
          emissiveIntensity: act * 0.45,
        });
        const plane = new THREE_NS.Mesh(new THREE_NS.PlaneGeometry(size, size), mat);
        plane.position.set(x, 0, 0);
        plane.rotation.y = Math.PI / 2;
        this.group.add(plane);
        positions[layer.name] = { x, y: 0, z: 0, size };

        const dots = Math.min(48, Math.max(4, Math.round(Math.sqrt(layer.neurons || 1))));
        const dotGeo = new THREE_NS.SphereGeometry(1.6 + act * 1.4, 10, 10);
        const dotMat = new THREE_NS.MeshStandardMaterial({
          color: act > 0 ? 0xf2cc60 : 0xe6edf3,
          emissive: act > 0 ? 0xf2cc60 : 0x000000,
          emissiveIntensity: act * 0.9,
        });
        for (let i = 0; i < dots; i++) {
          const dot = new THREE_NS.Mesh(dotGeo, dotMat);
          const gx = (i % 8) / 7 - 0.5;
          const gy = Math.floor(i / 8) / Math.max(1, Math.ceil(dots / 8) - 1) - 0.5;
          dot.position.set(x, gy * size * 0.68, gx * size * 0.68);
          this.group.add(dot);
        }
      });

      for (const proj of projectionRows(this.data.topology, this.data.mapped)) {
        const a = positions[proj.src];
        const b = positions[proj.dst];
        if (!a || !b) continue;
        const traceProj = traceProjs[normName(proj.name)] || {};
        const signal = Math.max(
          clamp01(Math.abs(Number(traceProj.weight_delta_mean || 0))),
          clamp01(Number(traceProj.active_edge_count || 0) / Math.max(1, Number(proj.n_edges || 1))),
          ...((traceProj.sample_edges || []).map((edge) => clamp01(edge.signal))),
        );
        const mat = new THREE_NS.LineBasicMaterial({
          color: signal > 0 ? 0xf2cc60 : 0x58a6ff,
          transparent: true,
          opacity: 0.18 + signal * 0.72,
        });
        const points = [
          new THREE_NS.Vector3(a.x, 0, 0),
          new THREE_NS.Vector3((a.x + b.x) / 2, 32 + signal * 36, 0),
          new THREE_NS.Vector3(b.x, 0, 0),
        ];
        const curve = new THREE_NS.CatmullRomCurve3(points);
        const line = new THREE_NS.Line(new THREE_NS.BufferGeometry().setFromPoints(curve.getPoints(28)), mat);
        this.group.add(line);
      }
    }

    _start() {
      if (this.animation) return;
      const tick = () => {
        this.animation = requestAnimationFrame(tick);
        if (!this.renderer || !this.scene || !this.camera || !this.group) return;
        this._resize();
        this.group.rotation.x = this.rotation.x;
        this.group.rotation.y = this.rotation.y + Math.sin(Date.now() * 0.00045) * 0.03;
        this.camera.lookAt(0, 0, 0);
        this.renderer.render(this.scene, this.camera);
      };
      tick();
    }
  }

  function buildMatrix(projection, traceProj, limit) {
    const nPre = Math.max(1, Number(projection.n_pre || 0));
    const nPost = Math.max(1, Number(projection.n_post || 0));
    const rows = Math.max(1, Math.min(limit, nPre));
    const cols = Math.max(1, Math.min(limit, nPost));
    const values = [];
    const weights = projection.weights;
    if (Array.isArray(weights) && Array.isArray(weights[0])) {
      for (let r = 0; r < rows; r++) {
        const srcRow = Math.floor((r / rows) * weights.length);
        const row = weights[srcRow] || [];
        for (let c = 0; c < cols; c++) {
          const srcCol = Math.floor((c / cols) * Math.max(1, row.length));
          values.push({ row: r, col: c, value: Number(row[srcCol] || 0) });
        }
      }
      return { rows, cols, values };
    }

    const flat = Array.isArray(weights)
      ? weights
      : Array.isArray(projection.weights_sample)
        ? projection.weights_sample
        : [];
    if (flat.length > 0) {
      for (let idx = 0; idx < Math.min(flat.length, rows * cols); idx++) {
        values.push({
          row: Math.floor(idx / cols),
          col: idx % cols,
          value: Number(flat[idx] || 0),
        });
      }
      return { rows, cols, values };
    }

    for (const edge of (traceProj && traceProj.sample_edges) || []) {
      values.push({
        row: Number(edge.pre || 0) % rows,
        col: Number(edge.post || 0) % cols,
        value: Number(edge.weight === null || edge.weight === undefined ? edge.signal : edge.weight),
      });
    }
    return { rows, cols, values };
  }

  function drawMiniGrid(ctx, x, y, w, h, layer, activity) {
    const cells = Math.min(100, Math.max(4, Math.round(Math.sqrt(layer.neurons || 1))));
    const cols = Math.ceil(Math.sqrt(cells));
    const rows = Math.ceil(cells / cols);
    const cell = Math.min(w / cols, h / rows);
    const ox = x + (w - cols * cell) / 2;
    const oy = y + (h - rows * cell) / 2;
    for (let i = 0; i < cells; i++) {
      const cx = ox + (i % cols) * cell;
      const cy = oy + Math.floor(i / cols) * cell;
      const hot = activity > 0 && i / cells < activity;
      ctx.fillStyle = hot ? "rgba(242, 204, 96, 0.88)" : "rgba(139, 148, 158, 0.22)";
      ctx.fillRect(cx + 1, cy + 1, Math.max(1, cell - 2), Math.max(1, cell - 2));
    }
  }

  function drawEmpty(ctx, w, h, message) {
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#8b949e";
    ctx.font = "13px monospace";
    ctx.textAlign = "center";
    ctx.fillText(message, w / 2, h / 2);
    ctx.textAlign = "left";
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  window.TopologyViews = {
    LayerStackRenderer,
    MatrixRenderer,
    TraceTimelineRenderer,
    TopologyThreeRenderer,
    layerRows,
    projectionRows,
  };
})();
