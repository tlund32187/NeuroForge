/* NeuroForge — Topology Page Controller
   Manages the dedicated Topology tab: filter panel, inspector,
   layout switcher, and data binding.
   ──────────────────────────────────────────────────────────── */
"use strict";

(function () {
  // ── DOM refs ─────────────────────────────────────────────────────
  const grid       = document.querySelector("#grid-topology");
  const viewArea   = document.querySelector("#topo-cy-container");
  const cyEl       = document.querySelector("#topo-view-graph");
  const stackEl    = document.querySelector("#topo-view-stack");
  const matrixEl   = document.querySelector("#topo-view-matrix");
  const threeEl    = document.querySelector("#topo-view-three");
  const traceEl    = document.querySelector("#topo-view-trace");
  const filterBody = document.querySelector("#topo-filter-body");
  const infoEl     = document.querySelector("#topo-inspector-body");
  const infoTitle  = document.querySelector("#topo-inspector-title");
  const viewSel    = document.querySelector("#topo-view-select");
  const layoutSel  = document.querySelector("#topo-layout-select");
  const projectionSel = document.querySelector("#topo-projection-select");
  const projectionWrap = document.querySelector("#topo-projection-wrap");
  const sampleInput = document.querySelector("#topo-sample-size");
  const sampleValue = document.querySelector("#topo-sample-size-value");
  const tracePlayBtn = document.querySelector("#topo-trace-play-btn");
  const fitBtn     = document.querySelector("#topo-fit-btn");
  const clearBtn   = document.querySelector("#topo-clear-filters-btn");
  const statusEl   = document.querySelector("#topo-status");
  const metaEl     = document.querySelector("#topo-meta");

  if (!grid || !cyEl) return; // Topology page HTML not present — bail.

  // ── Page state ───────────────────────────────────────────────────
  /** @type {import('./topology-mapper.js').CyElements|null} */
  let mappedElements = null;
  /** @type {import('./topology-mapper.js').FilterOptions|null} */
  let filterOptions = null;
  /** @type {Object<string, Set<string>>} */
  const activeFilters = {};
  /** @type {CyGraph|null} */
  let graph = null;
  let latestTopology = null;
  let latestTrace = null;
  const initialView = new URLSearchParams(window.location.search).get("topoView");
  let currentView = ["graph", "stack", "matrix", "three", "trace"].includes(initialView)
    ? initialView
    : viewSel ? viewSel.value : "graph";
  let traceFrames = [];
  let traceTimer = null;
  let tracePlayIndex = 0;
  const stackRenderer = window.TopologyViews && stackEl
    ? new window.TopologyViews.LayerStackRenderer(stackEl)
    : null;
  const matrixRenderer = window.TopologyViews && matrixEl
    ? new window.TopologyViews.MatrixRenderer(matrixEl)
    : null;
  const traceRenderer = window.TopologyViews && traceEl
    ? new window.TopologyViews.TraceTimelineRenderer(traceEl)
    : null;
  const threeRenderer = window.TopologyViews && threeEl
    ? new window.TopologyViews.TopologyThreeRenderer(threeEl)
    : null;

  // ── Initialize Cytoscape wrapper ─────────────────────────────────
  graph = new window.CyGraph({
    container: cyEl,
    onNodeSelect: showNodeInspector,
    onEdgeSelect: showEdgeInspector,
    onSelectionClear: clearInspector,
  });

  // ── Populate layout selector ─────────────────────────────────────
  if (layoutSel) {
    window.CyGraph.layoutList.forEach((l) => {
      const opt = document.createElement("option");
      opt.value = l.name;
      opt.textContent = l.label;
      layoutSel.appendChild(opt);
    });
    layoutSel.value = "dagre";
    layoutSel.addEventListener("change", () => {
      graph.setLayout(layoutSel.value);
    });
  }

  if (viewSel) {
    viewSel.value = currentView;
    viewSel.addEventListener("change", () => {
      currentView = viewSel.value || "graph";
      syncView();
      renderCurrentView();
    });
  }

  if (projectionSel) {
    projectionSel.addEventListener("change", () => renderCurrentView());
  }

  if (sampleInput) {
    sampleInput.addEventListener("input", () => {
      if (sampleValue) sampleValue.textContent = String(sampleInput.value || "64");
      renderCurrentView();
    });
  }

  if (tracePlayBtn) {
    tracePlayBtn.addEventListener("click", toggleTracePlayback);
  }

  if (fitBtn) {
    fitBtn.addEventListener("click", () => graph.fit());
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      Object.keys(activeFilters).forEach((k) => delete activeFilters[k]);
      // Uncheck all filter checkboxes.
      if (filterBody) {
        filterBody.querySelectorAll("input[type=checkbox]").forEach((cb) => {
          cb.checked = true;
        });
      }
      graph.showAll();
      updateStatus();
    });
  }

  // ── Public API (called by dashboard.js when topology data arrives)
  function onTopologyData(topology) {
    if (!topology) {
      latestTopology = null;
      latestTrace = null;
      traceFrames = [];
      stopTracePlayback();
      setEmptyState("No topology data available.");
      renderCurrentView();
      return;
    }
    setLoadingState();
    latestTopology = topology;

    const mapper = window.TopologyMapper;
    if (!mapper) {
      setErrorState("TopologyMapper module not loaded.");
      return;
    }

    mappedElements = mapper.mapTopologyToElements(topology);

    if (mappedElements.nodes.length === 0) {
      setEmptyState("Topology has no layers.");
      return;
    }

    filterOptions = mapper.extractFilterOptions(mappedElements);
    buildFilterPanel(filterOptions);
    buildProjectionSelector(topology);
    updateMeta(mappedElements.meta);

    graph.load(
      { nodes: mappedElements.nodes, edges: mappedElements.edges },
      layoutSel ? layoutSel.value : "dagre",
    );
    graph.updateTrace(latestTrace);

    setReadyState();
    syncView();
    renderCurrentView();
  }

  function onTopologyTrace(trace) {
    if (!trace) return;
    latestTrace = trace;
    traceFrames.push(trace);
    if (traceFrames.length > 240) traceFrames.shift();
    if (graph) graph.updateTrace(trace);
    renderCurrentView();
  }

  function onTopologyTraceHistory(frames) {
    if (!Array.isArray(frames)) return;
    traceFrames = frames.filter(Boolean).slice(-240);
    if (!latestTrace && traceFrames.length > 0) {
      latestTrace = traceFrames[traceFrames.length - 1];
      if (graph) graph.updateTrace(latestTrace);
    }
    renderCurrentView();
  }

  // ── State displays ───────────────────────────────────────────────
  function setEmptyState(msg) {
    if (statusEl) {
      statusEl.textContent = msg || "No topology loaded.";
      statusEl.className = "topo-status topo-status-empty";
      statusEl.style.display = "";
    }
    setPaneVisibility(false);
  }

  function setLoadingState() {
    if (statusEl) {
      statusEl.textContent = "Loading topology\u2026";
      statusEl.className = "topo-status topo-status-loading";
      statusEl.style.display = "";
    }
    setPaneVisibility(false);
  }

  function setErrorState(msg) {
    if (statusEl) {
      statusEl.textContent = "\u26A0 " + (msg || "Error loading topology.");
      statusEl.className = "topo-status topo-status-error";
      statusEl.style.display = "";
    }
  }

  function setReadyState() {
    if (statusEl) statusEl.style.display = "none";
    setPaneVisibility(true);
  }

  function setPaneVisibility(visible) {
    if (!viewArea) return;
    viewArea.style.visibility = "visible";
    viewArea.querySelectorAll(".topo-view-pane").forEach((pane) => {
      pane.style.visibility = visible ? "visible" : "hidden";
    });
  }

  function updateStatus() {
    if (!mappedElements) return;
    // Count visible.
    if (Object.keys(activeFilters).length === 0) {
      if (metaEl) {
        updateMeta(mappedElements.meta);
      }
      return;
    }
    const { visibleNodeIds } = window.TopologyMapper.applyFilters(mappedElements, activeFilters);
    if (metaEl) {
      metaEl.textContent =
        visibleNodeIds.size + " / " + mappedElements.meta.layerCount + " layers visible";
    }
  }

  // ── Meta bar ─────────────────────────────────────────────────────
  function updateMeta(meta) {
    if (!metaEl) return;
    const parts = [];
    parts.push(meta.layerCount + " layers");
    parts.push(meta.edgeCount + " connections");
    parts.push(meta.totalNeurons.toLocaleString() + " neurons");
    if (meta.totalParams > 0) parts.push(meta.totalParams.toLocaleString() + " params");
    if (meta.isVision) parts.push("Vision CNN");
    else parts.push("SNN");
    metaEl.textContent = parts.join("  \u00B7  ");
  }

  function buildProjectionSelector(topology) {
    if (!projectionSel || !window.TopologyViews) return;
    const selected = projectionSel.value;
    const rows = window.TopologyViews.projectionRows(topology, mappedElements);
    projectionSel.innerHTML = "";
    for (const row of rows) {
      const opt = document.createElement("option");
      opt.value = row.name || (row.src + "_" + row.dst);
      opt.textContent = row.name || (row.src + " -> " + row.dst);
      projectionSel.appendChild(opt);
    }
    if (selected && [...projectionSel.options].some((opt) => opt.value === selected)) {
      projectionSel.value = selected;
    }
  }

  function syncView() {
    const panes = {
      graph: document.querySelector("#topo-view-graph"),
      stack: document.querySelector("#topo-view-stack"),
      matrix: document.querySelector("#topo-view-matrix"),
      three: document.querySelector("#topo-view-three"),
      trace: document.querySelector("#topo-view-trace"),
    };
    Object.entries(panes).forEach(([name, pane]) => {
      if (!pane) return;
      pane.classList.toggle("topo-view-active", name === currentView);
    });
    const layoutWrap = layoutSel ? layoutSel.closest(".topo-toolbar-label") : null;
    if (layoutWrap) layoutWrap.style.display = currentView === "graph" ? "" : "none";
    if (fitBtn) fitBtn.style.display = currentView === "graph" ? "" : "none";
    if (projectionWrap) {
      projectionWrap.style.display = currentView === "matrix" ? "" : "none";
    }
    if (tracePlayBtn) {
      tracePlayBtn.style.display = currentView === "trace" ? "" : "none";
    }
    if (threeRenderer && currentView !== "three") threeRenderer.pause();
    if (graph && currentView === "graph") graph.fit();
  }

  function renderCurrentView() {
    if (!latestTopology && !mappedElements) return;
    const sampleLimit = sampleInput ? Number(sampleInput.value || 64) : 64;
    if (currentView === "stack" && stackRenderer) {
      stackRenderer.render(latestTopology, mappedElements, latestTrace);
    } else if (currentView === "matrix" && matrixRenderer) {
      matrixRenderer.render(
        latestTopology,
        mappedElements,
        latestTrace,
        projectionSel ? projectionSel.value : "",
        sampleLimit,
      );
    } else if (currentView === "three" && threeRenderer) {
      threeRenderer.render(latestTopology, mappedElements, latestTrace);
    } else if (currentView === "trace" && traceRenderer) {
      traceRenderer.render(traceFrames, latestTrace);
    }
  }

  function toggleTracePlayback() {
    if (traceTimer) {
      stopTracePlayback();
      return;
    }
    if (traceFrames.length === 0) return;
    tracePlayIndex = 0;
    if (tracePlayBtn) tracePlayBtn.textContent = "Pause";
    traceTimer = window.setInterval(() => {
      if (traceFrames.length === 0) {
        stopTracePlayback();
        return;
      }
      latestTrace = traceFrames[tracePlayIndex % traceFrames.length];
      if (graph) graph.updateTrace(latestTrace);
      renderCurrentView();
      tracePlayIndex += 1;
    }, 180);
  }

  function stopTracePlayback() {
    if (traceTimer) {
      window.clearInterval(traceTimer);
      traceTimer = null;
    }
    if (tracePlayBtn) tracePlayBtn.textContent = "Play";
  }

  // ── Filter panel ─────────────────────────────────────────────────
  function buildFilterPanel(opts) {
    if (!filterBody) return;
    filterBody.innerHTML = "";

    const filterDefs = [
      { key: "module",       label: "Module",        values: opts.module },
      { key: "population",   label: "Population",    values: opts.population },
      { key: "neuronType",   label: "Neuron Type",   values: opts.neuronType },
      { key: "synapseType",  label: "Synapse Type",  values: opts.synapseType },
      { key: "receptorType", label: "Receptor Type",  values: opts.receptorType },
      { key: "delayBucket",  label: "Delay Bucket",  values: opts.delayBucket },
      { key: "plasticity",   label: "Plasticity",    values: opts.plasticity },
    ];

    for (const fd of filterDefs) {
      if (fd.values.length === 0) continue;

      const section = document.createElement("div");
      section.className = "topo-filter-section";

      const heading = document.createElement("h4");
      heading.className = "topo-filter-heading";
      heading.textContent = fd.label;
      section.appendChild(heading);

      for (const val of fd.values) {
        const lbl = document.createElement("label");
        lbl.className = "topo-filter-item";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.dataset.filterKey = fd.key;
        cb.dataset.filterValue = val;
        cb.addEventListener("change", () => onFilterChange(fd.key, val, cb.checked));

        const span = document.createElement("span");
        span.textContent = val;

        lbl.appendChild(cb);
        lbl.appendChild(span);
        section.appendChild(lbl);
      }

      filterBody.appendChild(section);
    }
  }

  function onFilterChange(key, value, checked) {
    if (!mappedElements) return;

    if (checked) {
      // Remove value from active filter.
      if (activeFilters[key]) {
        activeFilters[key].delete(value);
        if (activeFilters[key].size === 0) delete activeFilters[key];
      }
    } else {
      // Build the "selected" set = all values except unchecked ones.
      if (!activeFilters[key]) {
        // Initialize with all values minus this one.
        const allVals = filterOptions ? filterOptions[key] || [] : [];
        activeFilters[key] = new Set(allVals.filter((v) => v !== value));
      } else {
        activeFilters[key].delete(value);
      }
    }

    // Recompute visibility.
    const hasAnyFilter = Object.keys(activeFilters).length > 0;
    if (!hasAnyFilter) {
      graph.showAll();
      updateStatus();
      return;
    }

    const { visibleNodeIds, visibleEdgeIds } = window.TopologyMapper.applyFilters(
      mappedElements, activeFilters,
    );
    graph.applyFilterVisibility(visibleNodeIds, visibleEdgeIds);
    updateStatus();
  }

  // ── Inspector panel ──────────────────────────────────────────────
  function showNodeInspector(nodeData) {
    if (!infoEl || !infoTitle) return;
    infoTitle.textContent = (nodeData.id || "Node").toUpperCase();

    const rows = [
      ["Type", nodeData.type || "–"],
      ["Neurons", (nodeData.neuronCount || 0).toLocaleString()],
      ["Parameters", (nodeData.paramCount || 0).toLocaleString()],
      ["Module", nodeData.module || "–"],
      ["Neuron Type", nodeData.neuronType || "–"],
    ];

    infoEl.innerHTML = _buildInfoTable(rows);
    _showInspector();
  }

  function showEdgeInspector(edgeData) {
    if (!infoEl || !infoTitle) return;
    infoTitle.textContent = (edgeData.source || "?") + " → " + (edgeData.target || "?");

    const rows = [
      ["Parameters", (edgeData.paramCount || 0).toLocaleString()],
      ["Edges", (edgeData.nEdges || 0).toLocaleString()],
      ["Topology", edgeData.topologyType || "–"],
      ["Synapse", edgeData.synapseType || "–"],
      ["Dense", edgeData.dense ? "Yes" : "No"],
    ];

    infoEl.innerHTML = _buildInfoTable(rows);
    _showInspector();
  }

  function clearInspector() {
    if (infoTitle) infoTitle.textContent = "Inspector";
    if (infoEl) infoEl.innerHTML =
      '<p class="topo-inspector-empty">Click a node or edge to inspect.</p>';
    _hideInspector();
  }

  function _showInspector() {
    const panel = document.querySelector("#topo-inspector");
    if (panel) panel.classList.add("active");
  }

  function _hideInspector() {
    const panel = document.querySelector("#topo-inspector");
    if (panel) panel.classList.remove("active");
  }

  /** @param {Array<[string, string]>} rows */
  function _buildInfoTable(rows) {
    let html = '<table class="topo-inspector-table">';
    for (const [key, val] of rows) {
      html += "<tr><td class=\"topo-inspector-key\">" + _esc(key) + "</td>"
        + "<td class=\"topo-inspector-val\">" + _esc(val) + "</td></tr>";
    }
    html += "</table>";
    return html;
  }

  /** @param {string} s */
  function _esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Initial state ────────────────────────────────────────────────
  clearInspector();
  syncView();
  setEmptyState("Start a training run or load a replay to view topology.");

  // ── Expose to dashboard.js ───────────────────────────────────────
  window.TopologyPage = {
    /** Called when new topology data arrives (live or replay). */
    onTopologyData: onTopologyData,
    /** Called when bounded runtime topology trace data arrives. */
    onTopologyTrace: onTopologyTrace,
    /** Called after replay events are loaded to seed the timeline. */
    onTopologyTraceHistory: onTopologyTraceHistory,
    /** Destroy the graph instance (cleanup). */
    destroy: function () {
      stopTracePlayback();
      if (graph) graph.destroy();
      if (threeRenderer) threeRenderer.destroy();
    },
  };
})();
