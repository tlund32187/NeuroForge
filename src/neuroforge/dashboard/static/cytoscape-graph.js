/* NeuroForge — Cytoscape Graph Wrapper
   Reusable Cytoscape.js instance manager with layout switching,
   selection events, and filter visibility.
   ────────────────────────────────────────────────────────────── */
"use strict";

/**
 * @typedef {{
 *   container: HTMLElement,
 *   onNodeSelect?: (nodeData: object) => void,
 *   onEdgeSelect?: (edgeData: object) => void,
 *   onSelectionClear?: () => void,
 *   onReady?: (cy: object) => void,
 * }} CyGraphOptions
 */

// ── Available layouts ────────────────────────────────────────────────

const LAYOUTS = {
  dagre: {
    name: "dagre",
    label: "Hierarchical (dagre)",
    options: {
      rankDir: "TB",
      rankSep: 70,
      nodeSep: 50,
      edgeSep: 20,
      fit: true,
      padding: 30,
      nodeDimensionsIncludeLabels: true,
      animate: false,
    },
  },
  breadthfirst: {
    name: "breadthfirst",
    label: "Breadthfirst",
    options: {
      directed: true,
      spacingFactor: 1.6,
      avoidOverlap: true,
      nodeDimensionsIncludeLabels: true,
      animate: false,
      fit: true,
      padding: 30,
    },
  },
  circle: {
    name: "circle",
    label: "Circle",
    options: {
      fit: true,
      padding: 40,
      animate: false,
      nodeDimensionsIncludeLabels: true,
    },
  },
  grid: {
    name: "grid",
    label: "Grid",
    options: {
      fit: true,
      padding: 30,
      rows: undefined,
      cols: undefined,
      animate: false,
      nodeDimensionsIncludeLabels: true,
    },
  },
  cose: {
    name: "cose",
    label: "Force-directed (CoSE)",
    options: {
      fit: true,
      padding: 30,
      animate: false,
      nodeDimensionsIncludeLabels: true,
      nodeRepulsion: function () { return 8000; },
      idealEdgeLength: function () { return 80; },
      edgeElasticity: function () { return 100; },
      gravity: 0.25,
      numIter: 200,
    },
  },
};

// ── Cytoscape stylesheet ─────────────────────────────────────────────

const CY_STYLE = [
  // ── Node styling ──
  {
    selector: "node",
    style: {
      "shape": "round-rectangle",
      "width": "data(nodeW)",
      "height": "data(nodeH)",
      "background-color": "data(bgColor)",
      "background-opacity": 0.88,
      "border-color": "data(borderColor)",
      "border-width": 2.5,
      "border-opacity": 0.95,
      "label": "data(label)",
      "text-valign": "center",
      "text-halign": "center",
      "text-wrap": "wrap",
      "text-max-width": "data(nodeW)",
      "font-family": "'JetBrains Mono', 'Fira Code', monospace",
      "font-size": "10px",
      "color": "data(textColor)",
      "text-outline-color": "data(bgColor)",
      "text-outline-width": 1,
    },
  },
  // ── Hover / active ──
  {
    selector: "node:active, node:selected",
    style: {
      "border-width": 3.5,
      "border-color": "#ffffff",
      "background-opacity": 1,
      "z-index": 10,
    },
  },
  // ── Hidden nodes (filtered out) ──
  {
    selector: "node.hidden",
    style: {
      "display": "none",
    },
  },
  // ── Edges ──
  {
    selector: "edge",
    style: {
      "curve-style": "bezier",
      "target-arrow-shape": "triangle",
      "target-arrow-color": "rgba(88, 166, 255, 0.8)",
      "line-color": "rgba(88, 166, 255, 0.55)",
      "width": "data(edgeWidth)",
      "label": "data(label)",
      "font-family": "'JetBrains Mono', 'Fira Code', monospace",
      "font-size": "9px",
      "color": "rgba(230, 237, 243, 0.8)",
      "text-rotation": "autorotate",
      "text-margin-y": -12,
      "text-background-color": "#0d1117",
      "text-background-opacity": 0.85,
      "text-background-padding": "3px",
      "arrow-scale": 1.3,
    },
  },
  // ── Edge with params → brighter ──
  {
    selector: "edge[paramCount > 0]",
    style: {
      "line-color": "rgba(88, 166, 255, 0.7)",
      "target-arrow-color": "rgba(88, 166, 255, 0.9)",
      "opacity": 0.8,
    },
  },
  // ── Edge without params → dimmer dashed ──
  {
    selector: "edge[paramCount = 0]",
    style: {
      "line-color": "rgba(88, 166, 255, 0.2)",
      "target-arrow-color": "rgba(88, 166, 255, 0.3)",
      "line-style": "dashed",
      "opacity": 0.4,
    },
  },
  // ── Hidden edges (filtered out) ──
  {
    selector: "edge.hidden",
    style: {
      "display": "none",
    },
  },
  // ── Selected ──
  {
    selector: "edge:selected",
    style: {
      "line-color": "#ffffff",
      "target-arrow-color": "#ffffff",
      "opacity": 1,
      "width": 4,
    },
  },
  // ── Box-selection highlight ──
  {
    selector: ":selected",
    style: {
      "border-color": "#ffffff",
    },
  },
];

// ── CyGraph class ────────────────────────────────────────────────────

class CyGraph {
  /**
   * @param {CyGraphOptions} opts
   */
  constructor(opts) {
    this._container = opts.container;
    this._onNodeSelect = opts.onNodeSelect || null;
    this._onEdgeSelect = opts.onEdgeSelect || null;
    this._onSelectionClear = opts.onSelectionClear || null;
    this._onReady = opts.onReady || null;
    /** @type {object|null} */
    this._cy = null;
    this._currentLayout = "dagre";
    this._elements = { nodes: [], edges: [] };
  }

  /** @returns {boolean} */
  get isReady() {
    return this._cy !== null;
  }

  /** @returns {string} */
  get currentLayout() {
    return this._currentLayout;
  }

  /** @returns {string[]} */
  static get layoutNames() {
    return Object.keys(LAYOUTS);
  }

  /** @returns {{ name: string, label: string }[]} */
  static get layoutList() {
    return Object.entries(LAYOUTS).map(([k, v]) => ({ name: k, label: v.label }));
  }

  /**
   * Load elements and render the graph.
   * @param {{ nodes: object[], edges: object[] }} elements
   * @param {string} [layoutName]
   */
  load(elements, layoutName) {
    this._elements = elements;
    const layout = layoutName || this._currentLayout;
    this._currentLayout = layout;

    if (typeof cytoscape === "undefined") {
      console.warn("[CyGraph] cytoscape not available");
      return;
    }

    // Destroy previous instance.
    if (this._cy) {
      this._cy.destroy();
      this._cy = null;
    }

    const layoutDef = LAYOUTS[layout] || LAYOUTS.dagre;

    this._cy = cytoscape({
      container: this._container,
      elements: [...elements.nodes, ...elements.edges],
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: true,
      selectionType: "single",
      style: CY_STYLE,
      layout: { name: layoutDef.name, ...layoutDef.options },
    });

    this._bindEvents();

    if (this._onReady) this._onReady(this._cy);
  }

  /**
   * Switch layout without reloading elements.
   * @param {string} layoutName
   */
  setLayout(layoutName) {
    if (!this._cy) return;
    const layoutDef = LAYOUTS[layoutName];
    if (!layoutDef) return;
    this._currentLayout = layoutName;
    this._cy.layout({ name: layoutDef.name, ...layoutDef.options }).run();
  }

  /**
   * Apply filter visibility: show/hide nodes and edges by ID sets.
   * @param {Set<string>} visibleNodeIds
   * @param {Set<string>} visibleEdgeIds
   */
  applyFilterVisibility(visibleNodeIds, visibleEdgeIds) {
    if (!this._cy) return;
    this._cy.batch(() => {
      this._cy.nodes().forEach((node) => {
        if (visibleNodeIds.has(node.id())) {
          node.removeClass("hidden");
        } else {
          node.addClass("hidden");
        }
      });
      this._cy.edges().forEach((edge) => {
        if (visibleEdgeIds.has(edge.id())) {
          edge.removeClass("hidden");
        } else {
          edge.addClass("hidden");
        }
      });
    });
  }

  /** Show all elements (clear filters). */
  showAll() {
    if (!this._cy) return;
    this._cy.batch(() => {
      this._cy.elements().removeClass("hidden");
    });
  }

  /** Fit the graph to the viewport. */
  fit() {
    if (this._cy) this._cy.fit(undefined, 30);
  }

  /** @returns {object|null} Selected node data or null. */
  getSelectedNode() {
    if (!this._cy) return null;
    const sel = this._cy.nodes(":selected");
    return sel.length > 0 ? sel[0].data() : null;
  }

  /** @returns {object|null} Selected edge data or null. */
  getSelectedEdge() {
    if (!this._cy) return null;
    const sel = this._cy.edges(":selected");
    return sel.length > 0 ? sel[0].data() : null;
  }

  /** Destroy the Cytoscape instance. */
  destroy() {
    if (this._cy) {
      this._cy.destroy();
      this._cy = null;
    }
  }

  /** @private */
  _bindEvents() {
    const cy = this._cy;
    if (!cy) return;

    cy.on("tap", "node", (evt) => {
      if (this._onNodeSelect) {
        this._onNodeSelect(evt.target.data());
      }
    });

    cy.on("tap", "edge", (evt) => {
      if (this._onEdgeSelect) {
        this._onEdgeSelect(evt.target.data());
      }
    });

    // Click on background → clear selection.
    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        if (this._onSelectionClear) this._onSelectionClear();
      }
    });

    // Box selection.
    cy.on("boxselect", "node", () => {
      const selected = cy.nodes(":selected");
      if (selected.length === 1 && this._onNodeSelect) {
        this._onNodeSelect(selected[0].data());
      }
    });
  }
}

// ── Exports ──────────────────────────────────────────────────────────
window.CyGraph = CyGraph;
window.CyGraph.LAYOUTS = LAYOUTS;
