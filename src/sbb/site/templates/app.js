"use strict";
// The leaderboard page. It loads data.json, written by `sbb site build`, filters it in the
// browser and draws the views with Plotly. Nothing is computed here that the builder could do.

const KIND_ORDER = ["data_poisoning", "training_control", "supply_chain", "federated"];
const STAGE_ORDER = ["pre_training", "in_training", "post_training", "inference", "model_detection"];
const METRIC_ORDER = ["ca", "asr", "asr_all", "ra", "frr", "tpr", "auroc", "verdict"];

const state = {
  tab: "matrix",
  version: null,
  track: null,
  dataset: null,
  model: null,
  scenario: null,
  metric: "asr",
  attacks: new Set(), // the shown attacks
  defenses: new Set(),
  seen: { attack: new Set(), defense: new Set() }, // methods offered before, to keep unticked ones
  compareBy: "attack",
  compareItem: null,
  sort: { key: 0, dir: 1 }, // column index and direction of the table
};
let DATA = null;

// ---- Pure helpers (also used by the tests through module.exports) ----

function uniq(values) {
  return [...new Set(values)];
}

function isPercent(metric) {
  return metric !== "verdict";
}

function formatValue(metric, value) {
  if (value === null || value === undefined) return "";
  return isPercent(metric) ? (100 * value).toFixed(1) : value.toFixed(3);
}

function formatStat(metric, stat) {
  if (!stat) return "";
  return `${formatValue(metric, stat.mean)} ± ${formatValue(metric, stat.std)}`;
}

function groupIndex(method, methods) {
  const info = methods[method];
  if (!info) return 99;
  const order = info.role === "attack" ? KIND_ORDER : STAGE_ORDER;
  const i = order.indexOf(info.group);
  return i < 0 ? 98 : i;
}

function isBaseline(name) {
  return name === "no_attack" || name === "no_defense";
}

function orderMethods(names, methods) {
  // By scenario kind or defense stage, the baselines first in their group, then by name.
  return [...names].sort((a, b) =>
    groupIndex(a, methods) - groupIndex(b, methods) || isBaseline(b) - isBaseline(a) || a.localeCompare(b));
}

function matrix(cells, metric, methods) {
  const attacks = orderMethods(uniq(cells.map((c) => c.attack)), methods);
  const defenses = orderMethods(uniq(cells.map((c) => c.defense)), methods);
  const byPair = new Map(cells.map((c) => [`${c.attack}|${c.defense}`, c]));
  const z = [];
  const text = [];
  const hover = [];
  for (const attack of attacks) {
    const zRow = [];
    const textRow = [];
    const hoverRow = [];
    for (const defense of defenses) {
      const cell = byPair.get(`${attack}|${defense}`);
      const stat = cell ? cell.metrics[metric] : null;
      const scale = isPercent(metric) ? 100 : 1;
      zRow.push(stat ? stat.mean * scale : null);
      textRow.push(stat ? `${formatValue(metric, stat.mean)}<br>±${formatValue(metric, stat.std)} (${cell.seeds.length})` : cell ? cell.status : "");
      hoverRow.push(cell
        ? `${attack} × ${defense}<br>${stat ? formatStat(metric, stat) : cell.status}${cell.reason ? `<br>${cell.reason}` : ""}<br>seeds ${cell.seeds.join(", ")} · ${cell.sites.join(", ")}`
        : `${attack} × ${defense}<br>no result`);
    }
    z.push(zRow);
    text.push(textRow);
    hover.push(hoverRow);
  }
  return { attacks, defenses, z, text, hover };
}

if (typeof module !== "undefined") {
  module.exports = { uniq, formatValue, formatStat, orderMethods, matrix };
}

// ---- Page ----

function $(id) {
  return document.getElementById(id);
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setOptions(select, values, current, label = (v) => v) {
  select.replaceChildren(...values.map((v) => new Option(label(v), v)));
  const value = values.includes(current) ? current : values[0] ?? null;
  if (value !== null) select.value = value;
  return value;
}

function metricLabel(metric) {
  const info = DATA.metrics[metric];
  return info ? `${info.label}${isPercent(metric) ? " (%)" : ""}` : metric;
}

function versionCells() {
  return DATA.cells.filter((c) => c.version === state.version && c.track === state.track);
}

function scenarioCells() {
  return versionCells().filter((c) => c.scenario === state.scenario);
}

function visibleCells() {
  return scenarioCells().filter((c) => state.attacks.has(c.attack) && state.defenses.has(c.defense));
}

function syncControls() {
  state.version = setOptions($("version"), uniq(DATA.cells.map((c) => c.version)).sort().reverse(), state.version);
  state.track = setOptions($("track"), uniq(DATA.cells.filter((c) => c.version === state.version).map((c) => c.track)), state.track);
  const cells = versionCells();
  state.dataset = setOptions($("dataset"), uniq(cells.map((c) => c.dataset)).sort(), state.dataset);
  const ofDataset = cells.filter((c) => c.dataset === state.dataset);
  state.model = setOptions($("model"), uniq(ofDataset.map((c) => c.arch)).sort(), state.model);
  const scenarios = uniq(ofDataset.filter((c) => c.arch === state.model).map((c) => c.scenario)).sort();
  state.scenario = setOptions($("scenario"), scenarios, state.scenario);
  $("scenario-label").hidden = scenarios.length < 2;
  const metrics = uniq(scenarioCells().flatMap((c) => Object.keys(c.metrics)));
  const ordered = METRIC_ORDER.filter((m) => metrics.includes(m)).concat(metrics.filter((m) => !METRIC_ORDER.includes(m)));
  state.metric = setOptions($("metric"), ordered.length ? ordered : ["asr"], state.metric, metricLabel);
  syncChecks("attacks", "attack");
  syncChecks("defenses", "defense");
}

function syncChecks(id, role) {
  const names = orderMethods(uniq(scenarioCells().map((c) => c[role])), DATA.methods);
  const chosen = role === "attack" ? state.attacks : state.defenses;
  for (const name of names) {
    if (!state.seen[role].has(name)) chosen.add(name); // a new method starts ticked
    state.seen[role].add(name);
  }
  const box = $(id);
  box.replaceChildren();
  let group = null;
  for (const name of names) {
    const info = DATA.methods[name];
    if (info && info.group !== group) {
      group = info.group;
      const heading = document.createElement("div");
      heading.className = "group";
      heading.textContent = group.replaceAll("_", " ");
      box.append(heading);
    }
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = chosen.has(name);
    input.addEventListener("change", () => {
      input.checked ? chosen.add(name) : chosen.delete(name);
      render();
    });
    label.append(input, name);
    box.append(label);
  }
  const shown = names.filter((n) => chosen.has(n)).length;
  $(`${role}-count`).textContent = `${shown}/${names.length}`;
}

function plotLayout(extra) {
  return {
    paper_bgcolor: css("--panel"),
    plot_bgcolor: css("--na"),
    font: { color: css("--ink"), family: "system-ui, sans-serif", size: 12 },
    margin: { l: 150, r: 20, t: 30, b: 110 },
    ...extra,
  };
}

function empty(element, message) {
  if (window.Plotly) Plotly.purge(element);
  element.replaceChildren(Object.assign(document.createElement("div"), { className: "empty", textContent: message }));
}

function renderMatrix() {
  const element = $("matrix");
  const cells = visibleCells();
  if (!cells.length) return empty(element, "No results for this selection yet.");
  element.replaceChildren();
  const m = matrix(cells, state.metric, DATA.methods);
  const percent = isPercent(state.metric);
  Plotly.react(element, [{
    type: "heatmap",
    x: m.defenses,
    y: m.attacks,
    z: m.z,
    text: m.text,
    texttemplate: "%{text}",
    hovertext: m.hover,
    hoverinfo: "text",
    colorscale: "Viridis",
    zmin: percent ? 0 : undefined,
    zmax: percent ? 100 : undefined,
    xgap: 3,
    ygap: 3,
    colorbar: { title: { text: metricLabel(state.metric), side: "right" } },
  }], plotLayout({
    height: Math.max(360, 70 * m.attacks.length + 180),
    xaxis: { title: "Defense", side: "bottom", tickangle: -30, automargin: true },
    yaxis: { title: "Attack", autorange: "reversed", automargin: true },
  }), { responsive: true, displaylogo: false });
  renderModelAuroc();
}

function renderModelAuroc() {
  const rows = DATA.model_auroc.filter((r) => r.version === state.version && r.track === state.track && r.scenario === state.scenario && state.defenses.has(r.defense));
  const box = $("model-auroc");
  box.replaceChildren();
  if (!rows.length) return;
  const heading = Object.assign(document.createElement("h2"), { textContent: "Model detection AUROC" });
  box.append(heading, table(["Defense", "AUROC", "Clean models", "Backdoored models"], rows.map((r) => [
    r.defense, r.auroc === null ? "needs both" : r.auroc.toFixed(3), r.clean_models, r.backdoored_models,
  ])));
}

function renderCompare() {
  const element = $("compare");
  const cells = visibleCells();
  const by = state.compareBy;
  const items = orderMethods(uniq(cells.map((c) => c[by])), DATA.methods);
  state.compareItem = setOptions($("compare-item"), items, state.compareItem);
  if (!cells.length || state.compareItem === null) return empty(element, "No results for this selection yet.");
  element.replaceChildren();
  const other = by === "attack" ? "defense" : "attack";
  const rows = cells.filter((c) => c[by] === state.compareItem && c.metrics[state.metric]);
  const names = orderMethods(rows.map((c) => c[other]), DATA.methods);
  const byName = new Map(rows.map((c) => [c[other], c]));
  const scale = isPercent(state.metric) ? 100 : 1;
  Plotly.react(element, [{
    type: "bar",
    x: names,
    y: names.map((n) => byName.get(n).metrics[state.metric].mean * scale),
    error_y: { type: "data", array: names.map((n) => byName.get(n).metrics[state.metric].std * scale), visible: true },
    marker: { color: css("--accent") },
    hovertemplate: `%{x}<br>%{y:.1f}<extra></extra>`,
  }], plotLayout({
    plot_bgcolor: css("--panel"),
    height: 440,
    xaxis: { title: other === "attack" ? "Attack" : "Defense", tickangle: -30, automargin: true },
    yaxis: { title: metricLabel(state.metric), range: isPercent(state.metric) ? [0, 100] : undefined, gridcolor: css("--line") },
  }), { responsive: true, displaylogo: false });
}

function table(headers, rows, sortable = false) {
  const element = document.createElement("table");
  const head = element.createTHead().insertRow();
  headers.forEach((h, i) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (sortable) {
      if (state.sort.key === i) th.setAttribute("aria-sort", state.sort.dir > 0 ? "ascending" : "descending");
      th.addEventListener("click", () => {
        state.sort = { key: i, dir: state.sort.key === i ? -state.sort.dir : 1 };
        render();
      });
    }
    head.append(th);
  });
  const body = element.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) {
      const td = tr.insertCell();
      if (value instanceof Node) td.append(value);
      else td.textContent = value ?? "";
      if (typeof value === "number" || /^[\d.]+( ± [\d.]+)?$/.test(String(value))) td.className = "num";
    }
  }
  return element;
}

function badge(status) {
  return Object.assign(document.createElement("span"), { className: `badge ${status}`, textContent: status });
}

function renderTable() {
  const cells = visibleCells();
  if (!cells.length) return empty($("table"), "No results for this selection yet.");
  const metrics = METRIC_ORDER.filter((m) => cells.some((c) => c.metrics[m]));
  const headers = ["Attack", "Defense", "Seeds", "Status", ...metrics.map(metricLabel)];
  // Sort keys: the text of the first four columns, the mean of a metric column.
  const keyOf = (cell, column) =>
    [cell.attack, cell.defense, cell.seeds.length, cell.status][column] ?? cell.metrics[metrics[column - 4]]?.mean ?? -Infinity;
  const column = state.sort.key < headers.length ? state.sort.key : 0;
  const sorted = [...cells].sort((a, b) => {
    const [ka, kb] = [keyOf(a, column), keyOf(b, column)];
    const order = typeof ka === "number" ? ka - kb : String(ka).localeCompare(String(kb));
    return state.sort.dir * order;
  });
  const rows = sorted.map((c) => [c.attack, c.defense, c.seeds.length, badge(c.status), ...metrics.map((m) => formatStat(m, c.metrics[m]))]);
  $("table").replaceChildren(table(headers, rows, true));
}

function replicationRows() {
  const rows = [];
  for (const record of DATA.replications) {
    for (const target of record.targets) {
      for (const [metric, reported] of Object.entries(target.reported)) {
        const ours = target.ours ? target.ours[metric] : null;
        rows.push({ record, target, metric, reported, ours });
      }
    }
  }
  return rows;
}

function renderReplications() {
  const rows = replicationRows();
  const metric = state.metric === "asr" ? "asr_all" : state.metric;
  const plotted = rows.filter((r) => r.metric === metric);
  const element = $("replication-plot");
  if (!plotted.length) {
    empty(element, `No replication target reports ${metricLabel(metric)}. Choose clean accuracy or ASR.`);
  } else {
    element.replaceChildren();
    const x = plotted.map((r) => `${r.record.method}<br>${r.target.dataset}`);
    Plotly.react(element, [
      { type: "bar", name: "Paper", x, y: plotted.map((r) => r.reported), marker: { color: css("--muted") } },
      {
        type: "bar",
        name: "Ours",
        x,
        y: plotted.map((r) => (r.ours ? r.ours.mean : null)),
        error_y: { type: "data", array: plotted.map((r) => (r.ours ? r.ours.std : 0)), visible: true },
        marker: { color: css("--accent") },
      },
    ], plotLayout({
      plot_bgcolor: css("--panel"),
      barmode: "group",
      height: 460,
      yaxis: { title: `${metricLabel(metric)}`, range: [0, 105], gridcolor: css("--line") },
      xaxis: { automargin: true },
      legend: { orientation: "h", y: 1.08 },
    }), { responsive: true, displaylogo: false });
  }
  const headers = ["Attack", "Dataset", "Model", "Metric", "Paper %", "Ours %", "Tolerance", "Status", "Targets confirmed", "Source"];
  $("replication-table").replaceChildren(table(headers, rows.map((r) => [
    r.record.method,
    r.target.dataset,
    r.target.arch,
    r.metric,
    r.reported.toFixed(1),
    r.ours ? `${r.ours.mean.toFixed(1)} ± ${r.ours.std.toFixed(1)}` : "pending",
    `${r.record.tolerance_pp} pp`,
    badge(r.record.status),
    r.record.confirmed ? "yes" : "proposed",
    r.record.source,
  ])));
}

function renderMethods() {
  const box = $("methods");
  box.replaceChildren();
  const status = new Map(DATA.replications.map((r) => [r.method, r.status]));
  const names = orderMethods(Object.keys(DATA.methods), DATA.methods);
  for (const role of ["attack", "defense"]) {
    for (const name of names.filter((n) => DATA.methods[n].role === role)) {
      const m = DATA.methods[name];
      const card = document.createElement("article");
      card.className = "card";
      const title = Object.assign(document.createElement("h3"), { textContent: m.title || name });
      const meta = Object.assign(document.createElement("div"), { className: "meta", textContent: `${name} · ${role} · ${m.group.replaceAll("_", " ")} · v${m.version}` });
      card.append(title, meta);
      const lines = [
        m.paper,
        m.trigger_domain ? `Trigger domain: ${m.trigger_domain.replace("_", " ")}` : null,
        m.resources ? `Defender sees: ${m.resources.join(", ") || "nothing"}` : null,
        m.adaptation_note ? `SNN adaptation: ${m.adaptation_note}` : null,
        m.license_note ? `License: ${m.license_note}` : null,
      ];
      for (const line of lines.filter(Boolean)) card.append(Object.assign(document.createElement("p"), { textContent: line }));
      if (m.code_reference) {
        const p = document.createElement("p");
        const url = m.code_reference.split(" ")[0];
        p.append("Code: ", Object.assign(document.createElement("a"), { href: url, textContent: m.code_reference, rel: "noopener" }));
        card.append(p);
      }
      if (status.has(name)) {
        const p = Object.assign(document.createElement("p"), { textContent: "Replication " });
        p.append(badge(status.get(name)));
        card.append(p);
      }
      box.append(card);
    }
  }
}

// The filters each view uses: the cell filters and the metric.
const SCOPES = { matrix: ["cells", "metric"], compare: ["cells", "metric"], table: ["cells"], replications: ["metric"], methods: [] };

function render() {
  syncControls();
  const scopes = SCOPES[state.tab];
  document.querySelector(".controls").hidden = scopes.length === 0;
  for (const element of document.querySelectorAll("[data-scope]")) {
    const shown = scopes.includes(element.dataset.scope);
    if (!shown) element.hidden = true;
    else if (element.id !== "scenario-label") element.hidden = false; // shown only when needed
  }
  for (const section of document.querySelectorAll("[data-view]")) section.hidden = section.dataset.view !== state.tab;
  for (const button of document.querySelectorAll("[data-tab]")) button.setAttribute("aria-selected", String(button.dataset.tab === state.tab));
  if (!window.Plotly && ["matrix", "compare", "replications"].includes(state.tab)) {
    return empty($(state.tab === "replications" ? "replication-plot" : state.tab), "The plotting library did not load. Check the connection to cdn.plot.ly.");
  }
  ({ matrix: renderMatrix, compare: renderCompare, table: renderTable, replications: renderReplications, methods: renderMethods })[state.tab]();
}

async function main() {
  try {
    DATA = await (await fetch("data.json")).json();
  } catch (error) {
    document.querySelector("main").replaceChildren(Object.assign(document.createElement("p"), { className: "empty", textContent: `Could not load data.json: ${error}` }));
    return;
  }
  for (const id of ["version", "track", "dataset", "model", "scenario", "metric"]) {
    $(id).addEventListener("change", (event) => {
      state[id] = event.target.value;
      render();
    });
  }
  $("compare-by").addEventListener("change", (event) => {
    state.compareBy = event.target.value;
    state.compareItem = null;
    render();
  });
  $("compare-item").addEventListener("change", (event) => {
    state.compareItem = event.target.value;
    render();
  });
  const tabs = [...document.querySelectorAll("[data-tab]")];
  const linked = location.hash.slice(1); // a view can be linked to, as #replications
  if (tabs.some((b) => b.dataset.tab === linked)) state.tab = linked;
  for (const button of tabs) {
    button.addEventListener("click", () => {
      state.tab = button.dataset.tab;
      history.replaceState(null, "", `#${state.tab}`);
      render();
    });
  }
  $("footer").textContent = `${DATA.cells.length} cells · ${DATA.replications.length} replication records · sbb ${DATA.sbb_version} · built ${DATA.generated}. Statuses are self-reported until independent verification (v2).`;
  render();
}

if (typeof document !== "undefined") main();
