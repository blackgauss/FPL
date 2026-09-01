// API client + formatting helpers for the FPL web PoC
const qs = (p) => Object.entries(p)
  .filter(([, v]) => v !== undefined && v !== null && v !== "")
  .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
  .join("&");

async function get(path, params = {}) {
  const url = params ? `${path}${Object.keys(params).length ? "?" + qs(params) : ""}` : path;
  const res = await fetch(url);
  if (res.status === 503) {
    const err = new Error("computing…");
    err.cold = true;
    throw err;
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail ?? detail; } catch { /* keep status text */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  meta: () => get("/api/meta"),
  players: (p) => get("/api/players", p),
  forecast: (p) => get("/api/forecast", p),
  forecastCdf: (p) => get("/api/forecast/cdf", p),
  teamFlags: (p) => get("/api/team/flags", p),
  transferSuggestions: (p) => get("/api/transfers/suggestions", p),
  leagueStandings: (p) => get("/api/league/standings", p),
  leagueReport: () => get("/api/league/standings/report"),
  leagueOwnership: () => get("/api/league/ownership"),
  leagueExposure: () => get("/api/league/exposure"),
  teamHistory: () => get("/api/team/history"),
  teamPerformance: (p) => get("/api/team/performance", p),
  researchMetrics: (run) => get("/api/research/metrics", { run }),
};

export function resolveForecastGw(preferred) {
  // ONE policy for "which GW has published features": a future GW whose
  // source GW is still settling clamps to the latest scoreable window.
  // Both fallback paths (drawer forecast, transfer CDF compare) use this.
  const m = window.FPL_META || {};
  const scoreable = m.max_forecast_gw || 0;
  if (!scoreable || !preferred || preferred <= scoreable) return preferred;
  return scoreable;
}

export function fmtPrice(tenths) {
  if (tenths === undefined || tenths === null) return "–";
  return `£${(tenths / 10).toFixed(1)}m`;
}

export function fmtPct(x, digits = 1) {
  if (x === undefined || x === null) return "–";
  return `${Number(x).toFixed(digits)}%`;
}

export function fmtNum(x) {
  if (x === undefined || x === null) return "–";
  return Number(x).toFixed(1);
}

export function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  for (const kid of kids) {
    if (kid === null || kid === undefined) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

export function empty(msg = "Nothing here yet.") {
  return el("div", { class: "empty" }, msg);
}

export function loading(msg = "Loading…") {
  return el("div", { class: "loading" }, el("span", { class: "spinner" }), ` ${msg}`);
}

export function cardRow(pairs) {
  return el("div", { class: "cards" }, ...pairs.map(([k, v]) =>
    el("div", { class: "card" },
      el("div", { class: "k" }, k),
      el("div", { class: "v" }, typeof v === "object" && v !== null
        ? JSON.stringify(v) : String(v ?? "–")))));
}

