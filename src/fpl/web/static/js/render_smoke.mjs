// Headless render smoke-test: imports the REAL view modules against a minimal
// DOM, with fetch() served from captured API responses (payloads.json =
// {"<url path>": <json>, ...}). Catches what Python tests cannot: view code
// reading .length of undefined, leaked "undefined"/"NaN" text, shape drift.
//
// usage: node render_smoke.mjs <payloads.json>
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const payloads = JSON.parse(readFileSync(process.argv[2], "utf8"));
const here = dirname(fileURLToPath(import.meta.url));
const VIEWS = ["overview", "explorer", "team", "transfers", "league", "research"];

// ---- minimal DOM (only what views/index actually touch) --------------------
function node(tag = "div") {
  const n = {
    tagName: tag, children: [], _text: "", attrs: {}, _class: "",
    dataset: {}, style: {}, clientWidth: 560, width: 0, height: 0, nodeType: 1,
    classList: {
      add: (c) => { if (!n._class.split(" ").includes(c)) n._class = `${n._class} ${c}`.trim(); },
      remove: (c) => { n._class = n._class.split(" ").filter(x => x && x !== c).join(" "); },
      toggle: (c, on) => (on ? n.classList.add(c) : n.classList.remove(c)),
      contains: (c) => n._class.split(" ").includes(c),
    },
    set className(v) { n._class = v; }, get className() { return n._class; },
    setAttribute(k, v) { n.attrs[k] = String(v); if (k === "class") n._class = String(v); },
    getAttribute(k) { return n.attrs[k]; },
    appendChild(c) { n.children.push(c); return c; },
    append(...cs) { for (const c of cs) if (c != null) n.children.push(c); },
    replaceChildren(...cs) { n.children = []; n.append(...cs); },
    addEventListener(ev, fn) { if (ev === "click") n._onclick = fn; },
    removeEventListener() {}, closest() { return null; }, remove() {}, focus() {},
    replaceWith() {},
    insertBefore(c) { n.children.push(c); return c; },
    set innerHTML(h) { n.children = []; n._text = ""; if (h) n._html = h; },
    get innerHTML() { return n._html || ""; },
    set textContent(t) { n._text = String(t); n.children = []; },
    get textContent() { return textOf(n); },
    get lastChild() { return n.children.at(-1) ?? null; },
    getContext: () => ctx2d(),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 560, height: 280 }),
  };
  return n;
}
function textOf(n) {
  if (n._text !== "") return n._text;
  return (n.children || []).map(textOf).join(" ");
}
function ctx2d() {
  const noop = () => {};
  const g = { measureText: () => ({ width: 10 }), canvas: { width: 560, height: 280 } };
  for (const m of ["beginPath", "moveTo", "lineTo", "stroke", "fill", "fillText",
    "closePath", "rect", "fillRect", "clearRect", "save", "restore", "translate"])
    g[m] = noop;
  return g;
}
function findClass(n, cls, acc = []) {
  if (n._class?.split(" ").includes(cls)) acc.push(n);
  for (const c of n.children || []) findClass(c, cls, acc);
  return acc;
}
function findAttr(n, key, val, acc = []) {
  if (n.attrs?.[key] === val || (key === "title" && n[key] === val)) acc.push(n);
  for (const c of n.children || []) findAttr(c, key, val, acc);
  return acc;
}

// uPlot stub validates its contract — a repeat of the (opts, true) data-slot
// bug, or NaN series, throws exactly like the real thing would
globalThis.window = globalThis;
globalThis.uPlot = class uPlot {
  constructor(opts, data) {
    if (!Array.isArray(data) || !data.length || !Array.isArray(data[0]))
      throw new Error(`uPlot stub: bad data arg (${typeof data})`);
    for (const s of data)
      if (!Array.isArray(s) || s.length !== data[0].length)
        throw new Error("uPlot stub: series length mismatch");
    for (const v of data.flat())
      if (typeof v !== "number" || !Number.isFinite(v))
        throw new Error(`uPlot stub: non-finite value ${v}`);
    this.root = node("div");
  }
  static plot = {};
};
globalThis.FPL_META = {};
globalThis.requestAnimationFrame = (fn) => Promise.resolve().then(() => fn(0));
const byId = {};
globalThis.document = {
  createElement: (t) => node(t),
  createTextNode: (t) => ({ _text: String(t), children: [], nodeType: 3 }),
  getElementById: (id) => (byId[id] ??= node()),
  querySelectorAll: () => [],
  addEventListener: () => {},
  body: node("body"),
};
const fetched = [];
const Q_PCT = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
// mirror of fpl.dist.probability_below: CDF from a quantile vector
function probBelow(vals, th) {
  const pairs = vals.map((v, i) => [v, Q_PCT[i]]).sort((a, b) => a[0] - b[0]);
  if (th <= pairs[0][0]) return 0;
  if (th >= pairs.at(-1)[0]) return 1;
  for (let i = 0; i < pairs.length - 1; i++) {
    const [x0, q0] = pairs[i], [x1, q1] = pairs[i + 1];
    if (th >= x0 && th <= x1) return x0 === x1 ? q1 : q0 + (th - x0) / (x1 - x0) * (q1 - q0);
  }
  return 1;
}
globalThis.fetch = async (url) => {
  const u = String(url);
  fetched.push(u);
  const [path, query] = u.split("?");
  const q = new URLSearchParams(query ?? "");
  if (path === "/api/forecast/cdf") {
    const code = Number(q.get("player_code")), gw = Number(q.get("gw"));
    const row = (payloads["/api/forecast"]?.rows ?? [])
      .find(r => r.player_code === code && r.gw === gw);
    if (!row) {
      return { ok: false, status: 404, statusText: "not found",
        json: async () => ({ detail: "no forecast row for this player/gameweek" }) };
    }
    const vals = Q_PCT.map(pr => row.quantiles[`q${Math.round(pr * 100)}`]);
    const top = Math.max(Math.max(...vals) * 1.05, (row.pred ?? 0) + 1, 5);
    const n = 80, xs = [], cdf = [];
    for (let i = 0; i < n; i++) {
      const x = i * top / (n - 1);
      xs.push(Math.round(x * 1000) / 1000);
      cdf.push(probBelow(vals, x));
    }
    return { ok: true, status: 200, statusText: "OK",
      json: async () => ({ player_code: code, gw, pred: row.pred,
        web_name: row.web_name, xs, cdf, quantiles: row.quantiles }) };
  }
  let body = payloads[path];
  if (path === "/api/forecast" && body?.rows) {
    // per-player drawer calls: filter the pre-captured full window
    const codes = new Set((q.get("player_codes") ?? "").split(",")
      .filter(Boolean).map(Number));
    if (codes.size) body = { ...body, rows: body.rows.filter(r => codes.has(r.player_code)) };
  }
  if (body === undefined) {
    return { ok: false, status: 404, statusText: `no payload for ${path}`,
      json: async () => ({ detail: `no payload for ${u}` }) };
  }
  return { ok: true, status: 200, statusText: "OK", json: async () => body };
};
const flush = () => new Promise((res) => setTimeout(res, 0));

// ---- run -------------------------------------------------------------------
const report = {};
let content = node("div");
try {
  const meta = payloads["/api/meta"];
  if (meta === undefined) throw new Error("payloads missing /api/meta");
  Object.assign(globalThis.FPL_META, meta);

  for (const view of VIEWS) {
    content = node("div");
    const mod = await import(resolve(here, `views/${view}.js`));
    await mod.render(content);
    for (let i = 0; i < 6; i++) await flush();
    const row = findClass(content, "click")[0];
    if (row) { await row._onclick(); for (let i = 0; i < 8; i++) await flush(); }
    if (view === "explorer") {
      // click a sortable header (price) once: exercises sort param plumbing
      const th = findAttr(content, "title", "sort by Price")[0];
      if (!th?._onclick) throw new Error("explorer: Price header not sortable");
      th._onclick();
      for (let i = 0; i < 8; i++) await flush();
    }
    const txt = textOf(view === "explorer" ? content : content)
      + (view === "explorer" ? " " + textOf(byId.drawer ?? node()) : "");
    const bad = txt.match(/undefined|NaN|\[object Object\]/);
    if (bad) throw new Error(`${view}: leaked "${bad[0]}" in rendered text`);
    if (!txt.trim()) throw new Error(`${view}: rendered nothing`);
    if (/no payload for/.test(txt)) throw new Error(`${view}: ${txt.slice(0, 120)}`);
    report[view] = "ok";
  }
} catch (e) {
  console.log(JSON.stringify({ ok: false, error: String(e.stack || e), fetched }, null, 2));
  process.exit(1);
}
console.log(JSON.stringify({ ok: true, report, fetched: fetched.length }));
