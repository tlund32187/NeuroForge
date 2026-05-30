/* NeuroForge — Topology Mapper
   Converts domain topology data into Cytoscape.js elements.
   ────────────────────────────────────────────────────────── */
"use strict";

/**
 * @typedef {{
 *   name: string, type: string, neurons: number, params: number,
 *   out_channels?: number, out_shape?: string, kernel_size?: number[],
 *   stride?: number[], operation?: string, activation?: string,
 *   norm?: string, in_channels?: number, in_features?: number,
 *   out_features?: number, out_h?: number, out_w?: number,
 *   spike_threshold?: number, pool_mode?: string, depth?: number,
 * }} LayerDetail
 *
 * @typedef {{
 *   name: string, src: string, dst: string,
 *   n_pre?: number, n_post?: number, n_edges?: number,
 *   dense?: boolean, dtype?: string, topology_type?: string,
 *   paramCount?: number,
 * }} EdgeDetail
 *
 * @typedef {{
 *   layers: string[],
 *   edges: EdgeDetail[],
 *   layer_details?: LayerDetail[],
 *   projection_meta?: object[],
 *   topology_stats?: object,
 *   vision_blocks?: string[],
 *   total_params?: number,
 *   populations?: object,
 * }} TopologyPayload
 *
 * @typedef {{ bg: string, border: string, text: string }} BlockColors
 *
 * @typedef {{
 *   nodes: object[],
 *   edges: object[],
 *   meta: { totalNeurons: number, totalParams: number, layerCount: number,
 *           edgeCount: number, isVision: boolean },
 * }} CyElements
 *
 * @typedef {{
 *   module: string[],
 *   population: string[],
 *   neuronType: string[],
 *   synapseType: string[],
 *   receptorType: string[],
 *   delayBucket: string[],
 *   plasticity: string[],
 * }} FilterOptions
 */

// ── Color palette for layer/block types ──────────────────────────────
const BLOCK_COLORS = {
  input:   { bg: "#1a3a5c", border: "#58a6ff", text: "#58a6ff" },
  conv:    { bg: "#2d1f4e", border: "#bc8cff", text: "#d4b8ff" },
  pool:    { bg: "#1a3344", border: "#58c4dc", text: "#7dd8ec" },
  res:     { bg: "#3b1f3b", border: "#e07cff", text: "#f0b0ff" },
  linear:  { bg: "#1a3d26", border: "#3fb950", text: "#6edc7e" },
  output:  { bg: "#1a3d26", border: "#3fb950", text: "#6edc7e" },
  hidden:  { bg: "#2d1f4e", border: "#bc8cff", text: "#d4b8ff" },
  default: { bg: "#21262d", border: "#8b949e", text: "#c9d1d9" },
};

/** @param {string} type @returns {BlockColors} */
function blockColor(type) {
  return BLOCK_COLORS[type] || BLOCK_COLORS.default;
}

// ── Layer info parsing (from layer label strings) ────────────────────

/**
 * Parse a layer label like "input(64)" or "conv_0(3x8x8)" into structured info.
 * @param {string} label
 * @param {number} idx
 * @returns {{ name: string, count: number, detail: string, layerIdx: number }}
 */
function parseLayerLabel(label, idx) {
  const m = label.match(/^([\w-]+)\((.+)\)$/);
  const name = m ? m[1] : String(label);
  const inner = m ? m[2] : "";
  const count = /^\d+$/.test(inner) ? parseInt(inner, 10) : 1;
  return { name, count, detail: inner, layerIdx: idx };
}

// ── Classify a layer for coloring/grouping ───────────────────────────

/**
 * Infer layer type from name, position, and detail info.
 * @param {string} name
 * @param {number} idx
 * @param {number} total
 * @param {LayerDetail|null} detail
 * @returns {string}
 */
function inferLayerType(name, idx, total, detail) {
  if (detail && detail.type) return detail.type;
  const lower = name.toLowerCase();
  if (idx === 0 || lower.includes("input") || lower.includes("image")) return "input";
  if (idx === total - 1 || lower.includes("output") || lower.includes("head")) return "output";
  if (lower.includes("conv")) return "conv";
  if (lower.includes("pool")) return "pool";
  if (lower.includes("res")) return "res";
  if (lower.includes("linear") || lower.includes("features")) return "linear";
  return "hidden";
}

// ── Build multi-line node label ──────────────────────────────────────

/**
 * Build a readable multi-line label for a topology node.
 * @param {{ name: string, type: string, neurons: number, params: number,
 *           operation?: string, out_shape?: string, activation?: string }} info
 * @returns {string}
 */
function buildNodeLabel(info) {
  const lines = [info.name.toUpperCase()];
  if (info.operation) lines.push(info.operation);
  if (info.out_shape) lines.push("Out: " + info.out_shape);
  lines.push(info.neurons.toLocaleString() + " neurons");
  if (info.params > 0) {
    lines.push(info.params.toLocaleString() + " params");
  } else if (info.type === "pool") {
    lines.push("no learned weights");
  }
  if (info.activation) lines.push("\u26A1 " + info.activation);
  return lines.join("\n");
}

// ── Main mapper: TopologyPayload → CyElements ────────────────────────

/**
 * Convert a raw topology payload into Cytoscape.js elements.
 * Works for both SNN (logic gates) and Vision (CNN) topologies.
 *
 * @param {TopologyPayload|null} topology
 * @returns {CyElements}
 */
function mapTopologyToElements(topology) {
  const empty = { nodes: [], edges: [], meta: {
    totalNeurons: 0, totalParams: 0, layerCount: 0, edgeCount: 0, isVision: false,
  }};
  if (!topology) return empty;

  const hasDetails = Array.isArray(topology.layer_details) && topology.layer_details.length > 0;
  const layers = topology.layers || [];
  const rawEdges = topology.edges || [];
  const layerDetails = topology.layer_details || [];

  if (layers.length === 0 && layerDetails.length === 0) return empty;

  const nodes = [];
  const edges = [];
  let totalNeurons = 0;
  let totalParams = topology.total_params || 0;

  // ── Build layer nodes ──────────────────────────────────────────────
  const layerInfos = hasDetails ? layerDetails : layers.map((l, i) => {
    const parsed = parseLayerLabel(l, i);
    const type = inferLayerType(parsed.name, i, layers.length, null);
    return {
      name: parsed.name,
      type: type,
      neurons: parsed.count,
      params: 0,
      out_shape: parsed.detail !== String(parsed.count) ? parsed.detail : undefined,
    };
  });

  const maxNeurons = Math.max(1, ...layerInfos.map((d) => d.neurons || 1));

  layerInfos.forEach((info, idx) => {
    const type = info.type || inferLayerType(info.name, idx, layerInfos.length, info);
    const colors = blockColor(type);
    const neurons = info.neurons || 1;
    const params = info.params || 0;
    totalNeurons += neurons;
    if (!topology.total_params) totalParams += params;

    // Scale node dimensions by neuron count (log scale).
    const logN = Math.log2(Math.max(2, neurons));
    const logMax = Math.log2(Math.max(2, maxNeurons));
    const sizeRatio = 0.4 + 0.6 * (logN / logMax);
    const nodeW = Math.max(150, sizeRatio * 230);
    const nodeH = Math.max(65, sizeRatio * 95);

    const label = buildNodeLabel({
      name: info.name, type, neurons, params,
      operation: info.operation, out_shape: info.out_shape, activation: info.activation,
    });

    nodes.push({
      group: "nodes",
      data: {
        id: info.name,
        label: label,
        type: type,
        idx: idx,
        neuronCount: neurons,
        paramCount: params,
        bgColor: colors.bg,
        borderColor: colors.border,
        textColor: colors.text,
        nodeW: nodeW,
        nodeH: nodeH,
        // Filter-relevant metadata.
        module: _inferModule(info),
        population: info.name,
        neuronType: _inferNeuronType(info),
      },
    });
  });

  // ── Build edges ────────────────────────────────────────────────────
  for (const edge of rawEdges) {
    const dst = layerInfos.find((d) => d.name === edge.dst);
    const dstParams = dst ? (dst.params || 0) : 0;
    const nEdges = edge.n_edges || 0;
    const edgeWidth = dstParams > 0
      ? Math.min(10, 1.5 + Math.log10(Math.max(1, dstParams)) * 1.5)
      : nEdges > 0
        ? Math.min(10, 1.5 + Math.log10(Math.max(1, nEdges)) * 1.0)
        : 1.5;

    edges.push({
      group: "edges",
      data: {
        id: "e_" + edge.src + "_" + edge.dst,
        source: edge.src,
        target: edge.dst,
        label: dstParams > 0
          ? dstParams.toLocaleString() + " params"
          : nEdges > 0
            ? nEdges.toLocaleString() + " edges"
            : "",
        edgeWidth: edgeWidth,
        paramCount: dstParams,
        nEdges: nEdges,
        dense: edge.dense || false,
        topologyType: edge.topology_type || (edge.dense ? "dense" : "unknown"),
        synapseType: edge.topology_type || "dense",
        // Delay/plasticity not yet in payload — set defaults.
        delayBucket: "default",
        plasticity: "unknown",
      },
    });
  }

  return {
    nodes,
    edges,
    meta: {
      totalNeurons,
      totalParams,
      layerCount: layerInfos.length,
      edgeCount: edges.length,
      isVision: hasDetails,
    },
  };
}

// ── Extract filter options from mapped elements ──────────────────────

/**
 * Gather the set of unique filter values from mapped elements.
 * @param {CyElements} elements
 * @returns {FilterOptions}
 */
function extractFilterOptions(elements) {
  const modules = new Set();
  const populations = new Set();
  const neuronTypes = new Set();
  const synapseTypes = new Set();
  const receptorTypes = new Set();
  const delayBuckets = new Set();
  const plasticityValues = new Set();

  for (const n of elements.nodes) {
    if (n.data.module) modules.add(n.data.module);
    if (n.data.population) populations.add(n.data.population);
    if (n.data.neuronType) neuronTypes.add(n.data.neuronType);
  }
  for (const e of elements.edges) {
    if (e.data.synapseType) synapseTypes.add(e.data.synapseType);
    if (e.data.delayBucket) delayBuckets.add(e.data.delayBucket);
    if (e.data.plasticity) plasticityValues.add(e.data.plasticity);
  }

  return {
    module: [...modules].sort(),
    population: [...populations].sort(),
    neuronType: [...neuronTypes].sort(),
    synapseType: [...synapseTypes].sort(),
    receptorType: [...receptorTypes].sort(),
    delayBucket: [...delayBuckets].sort(),
    plasticity: [...plasticityValues].sort(),
  };
}

/**
 * Apply filter selections to elements, returning visible IDs.
 * @param {CyElements} elements
 * @param {Object<string, Set<string>>} activeFilters — key=filterName, value=selected values
 * @returns {{ visibleNodeIds: Set<string>, visibleEdgeIds: Set<string> }}
 */
function applyFilters(elements, activeFilters) {
  const visibleNodeIds = new Set();
  const visibleEdgeIds = new Set();

  for (const n of elements.nodes) {
    let visible = true;
    if (activeFilters.module && activeFilters.module.size > 0) {
      if (!activeFilters.module.has(n.data.module || "")) visible = false;
    }
    if (activeFilters.population && activeFilters.population.size > 0) {
      if (!activeFilters.population.has(n.data.population || "")) visible = false;
    }
    if (activeFilters.neuronType && activeFilters.neuronType.size > 0) {
      if (!activeFilters.neuronType.has(n.data.neuronType || "")) visible = false;
    }
    if (visible) visibleNodeIds.add(n.data.id);
  }

  for (const e of elements.edges) {
    let visible = visibleNodeIds.has(e.data.source) && visibleNodeIds.has(e.data.target);
    if (visible && activeFilters.synapseType && activeFilters.synapseType.size > 0) {
      if (!activeFilters.synapseType.has(e.data.synapseType || "")) visible = false;
    }
    if (visible && activeFilters.delayBucket && activeFilters.delayBucket.size > 0) {
      if (!activeFilters.delayBucket.has(e.data.delayBucket || "")) visible = false;
    }
    if (visible && activeFilters.plasticity && activeFilters.plasticity.size > 0) {
      if (!activeFilters.plasticity.has(e.data.plasticity || "")) visible = false;
    }
    if (visible) visibleEdgeIds.add(e.data.id);
  }

  return { visibleNodeIds, visibleEdgeIds };
}

// ── Private helpers ──────────────────────────────────────────────────

/** @param {LayerDetail} info @returns {string} */
function _inferModule(info) {
  const type = (info.type || "").toLowerCase();
  if (type === "input" || type === "image") return "input";
  if (type === "conv" || type === "pool" || type === "res") return "backbone";
  if (type === "linear" || type === "output") return "head";
  return "network";
}

/** @param {LayerDetail} info @returns {string} */
function _inferNeuronType(info) {
  if (info.activation) return info.activation;
  const type = (info.type || "").toLowerCase();
  if (type === "input" || type === "image") return "source";
  if (type === "pool") return "pool";
  return "spiking";
}

// ── Exports (attach to window for cross-file access) ─────────────────
window.TopologyMapper = {
  mapTopologyToElements,
  extractFilterOptions,
  applyFilters,
  blockColor,
  parseLayerLabel,
  BLOCK_COLORS,
};
