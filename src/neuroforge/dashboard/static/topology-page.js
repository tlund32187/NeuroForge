/* NeuroForge — Topology Page Controller
   Manages the dedicated Topology tab: filter panel, inspector,
   layout switcher, and data binding.
   ──────────────────────────────────────────────────────────── */
"use strict";

(function () {
  // ── DOM refs ─────────────────────────────────────────────────────
  const grid       = document.querySelector("#grid-topology");
  const cyEl       = document.querySelector("#topo-cy-container");
  const filterBody = document.querySelector("#topo-filter-body");
  const infoEl     = document.querySelector("#topo-inspector-body");
  const infoTitle  = document.querySelector("#topo-inspector-title");
  const layoutSel  = document.querySelector("#topo-layout-select");
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
      setEmptyState("No topology data available.");
      return;
    }
    setLoadingState();

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
    updateMeta(mappedElements.meta);

    graph.load(
      { nodes: mappedElements.nodes, edges: mappedElements.edges },
      layoutSel ? layoutSel.value : "dagre",
    );

    setReadyState();
  }

  // ── State displays ───────────────────────────────────────────────
  function setEmptyState(msg) {
    if (statusEl) {
      statusEl.textContent = msg || "No topology loaded.";
      statusEl.className = "topo-status topo-status-empty";
      statusEl.style.display = "";
    }
    if (cyEl) cyEl.style.visibility = "hidden";
  }

  function setLoadingState() {
    if (statusEl) {
      statusEl.textContent = "Loading topology\u2026";
      statusEl.className = "topo-status topo-status-loading";
      statusEl.style.display = "";
    }
    if (cyEl) cyEl.style.visibility = "hidden";
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
    if (cyEl) cyEl.style.visibility = "visible";
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
  setEmptyState("Start a training run or load a replay to view topology.");

  // ── Expose to dashboard.js ───────────────────────────────────────
  window.TopologyPage = {
    /** Called when new topology data arrives (live or replay). */
    onTopologyData: onTopologyData,
    /** Destroy the graph instance (cleanup). */
    destroy: function () {
      if (graph) graph.destroy();
    },
  };
})();
