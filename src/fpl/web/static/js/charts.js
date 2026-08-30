// Band+line chart over uPlot (canvas fallback if uPlot is absent)
const PALETTE = ["#2456d6", "#c62f2f", "#17803d", "#b07404", "#7a3cc2"];

// Multi-series overlay (e.g. per-GW CDFs) — series: [{label, ys}]
export function multiLineChart(host, xs, series, opts = {}) {
  host.innerHTML = "";
  const W = host.clientWidth || 520, H = opts.height || 240;
  if (!window.uPlot || xs.length < 2 || !series.length) return false;
  const s = [{}, ...series.map((se, i) => ({
    label: se.label, stroke: PALETTE[i % PALETTE.length], width: 2,
  }))];
  const cfg = {
    target: host, width: W, height: H, series: s,
    scales: { x: { time: false },
      ...(opts.scaleY ? { y: opts.scaleY } : {}) },
    axes: [{ size: 42, label: opts.xlabel || "" },
      { size: 46, ...(opts.yPercent
        ? { values: (u, t) => t.map((v) => Math.round(v * 100) + "%") }
        : {}) }],
  };
  const u = new window.uPlot(cfg, [xs, ...series.map((se) => se.ys)]);
  if (u.root && !u.root.parentNode) host.append(u.root);
  return true;
}

// Tiny inline sparkline of bars (no uPlot needed)
export function sparkBars(host, values, opts = {}) {
  host.innerHTML = "";
  const max = Math.max(opts.min || 1, ...values.map((v) => v ?? 0));
  for (const v of values) {
    const h = v == null ? 0 : Math.max(2, Math.round(v / max * (opts.height || 18)));
    host.append(el2("span", {
      style: `display:inline-block;width:6px;margin:0 1px;height:${h}px;`
        + `vertical-align:bottom;background:${opts.color || "#2456d6"};`
        + `border-radius:1px;opacity:${v == null ? 0.25 : 0.85}`,
      title: v == null ? "–" : String(v),
    }));
  }
}

function el2(tag, attrs) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") n.setAttribute("style", v);
    else if (k === "title") n.title = v;
  }
  return n;
}

export function cdfChart(host, xs, cdf, opts = {}) {
  host.innerHTML = "";
  const W = host.clientWidth || 520, H = opts.height || 230;
  if (!window.uPlot || xs.length < 2) return false;
  const u = new window.uPlot({
    target: host, width: W, height: H,
    series: [{}, { label: opts.label || "P(X ≤ x)", stroke: "#2456d6",
      fill: "rgba(36,86,214,.10)", width: 2 }],
    scales: { x: { time: false }, y: { min: 0, max: 1 } },
    axes: [{ size: 42, label: opts.xlabel || "points" },
      { size: 46, values: (u, t) => t.map(v => Math.round(v * 100) + "%") }],
  }, [xs, cdf]);
  // uPlot 1.6 builds (but does not insert) u.root; append it visibly
  if (u.root && !u.root.parentNode) host.append(u.root);
  return true;
}

export function bandChart(host, xs, ys, lo, hi, opts = {}) {
  host.innerHTML = "";
  const data = [xs, ys, lo, hi, opts.lo2, opts.hi2].filter(d => d);
  const W = host.clientWidth || 560, H = opts.height || 280;
  if (xs.length < 2) return false; // single GW: number/table says it all
  if (!window.uPlot) { canvasBand(host, data, opts, W, H); return true; }
  const names = opts.names || ["pred", "lo", "hi"];
  const bandPath = (lowerIdx) => (u, si, i0, i1) => {
    let d = "";
    for (let i = i0; i <= i1; i++) d += `${i === i0 ? "M" : "L"}${u.posX(si, i)} ${u.posY(si, i)}`;
    for (let i = i1; i >= i0; i--) d += `L${u.posX(lowerIdx, i)} ${u.posY(lowerIdx, i)}`;
    return [d + "Z"];
  };
  const series = [
    {},
    { label: names[0], stroke: "#2456d6", width: 2 },
    { label: names[1], stroke: "rgba(104,112,127,.45)" },
    { label: names[2], stroke: "rgba(104,112,127,.45)", fill: "rgba(104,112,127,.14)", paths: bandPath(2) },
  ];
  if (opts.lo2) {
    series.push({ label: "q25", stroke: "rgba(36,86,214,.4)" });
    series.push({ label: "q75", stroke: "rgba(36,86,214,.55)", fill: "rgba(36,86,214,.25)", paths: bandPath(4) });
  }
  const u = new window.uPlot({
    target: host, width: W, height: H, series,
    scales: { x: { time: false } },
    axes: [{ size: 40 }, { size: 40 }],
  }, data);
  if (u.root && !u.root.parentNode) host.append(u.root);
  return true;
}

function canvasBand(host, [xs, ys, lo, hi, lo2, hi2], opts, W, H) {
  const c = el2("canvas", W, H);
  const g = c.getContext("2d"), pad = { l: 42, r: 10, t: 10, b: 24 };
  const all = [ys, lo, hi, lo2, hi2].filter(Boolean).flat().filter(v => v != null);
  const ymin = Math.min(...all, 0), ymax = Math.max(...all, 1);
  const X = i => pad.l + (xs.length < 2 ? 0 : i / (xs.length - 1) * (W - pad.l - pad.r));
  const Y = v => H - pad.b - (v - ymin) / (ymax - ymin || 1) * (H - pad.t - pad.b);
  g.strokeStyle = "#dfe3e8";
  for (let t = 0; t <= 4; t++) {
    const v = ymin + (ymax - ymin) * t / 4, y = Y(v);
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillStyle = "#68707f"; g.font = "11px sans-serif"; g.fillText(v.toFixed(0), 4, y + 3);
  }
  g.fillStyle = "#68707f";
  for (let i = 0; i < xs.length; i++) {
    if (xs.length <= 12 || i % 2 === 0) g.fillText("GW" + xs[i], X(i) - 12, H - 8);
  }
  const area = (inner, outer, color) => {
    g.beginPath();
    outer.forEach((v, i) => g[i ? "lineTo" : "moveTo"](X(i), Y(v)));
    for (let i = inner.length - 1; i >= 0; i--) g.lineTo(X(i), Y(inner[i]));
    g.closePath(); g.fillStyle = color; g.fill();
  };
  area(lo, hi, "rgba(104,112,127,.14)");
  if (lo2) area(lo2, hi2, "rgba(36,86,214,.22)");
  g.strokeStyle = "#2456d6"; g.lineWidth = 2; g.beginPath();
  ys.forEach((v, i) => g[i ? "lineTo" : "moveTo"](X(i), Y(v))); g.stroke();
  let tip = null;
  c.addEventListener("mousemove", (ev) => {
    const r = c.getBoundingClientRect();
    const i = Math.round((ev.clientX - r.left - pad.l) / (W - pad.l - pad.r) * (xs.length - 1));
    const idx = Math.max(0, Math.min(xs.length - 1, i));
    if (tip) tip.remove();
    tip = el2("div");
    tip.style.cssText = "position:fixed;background:#1c2330;color:#fff;padding:4px 8px;border-radius:5px;font:12px sans-serif;pointer-events:none;z-index:99";
    tip.textContent = `GW${xs[idx]} · pred ${ys[idx]?.toFixed(1)} · [${lo[idx]?.toFixed(1)}, ${hi[idx]?.toFixed(1)}]`;
    document.body.append(tip);
    tip.style.left = ev.clientX + 12 + "px";
    tip.style.top = ev.clientY + 12 + "px";
  });
  c.addEventListener("mouseleave", () => { if (tip) { tip.remove(); tip = null; } });
  host.append(c);
  function el2(tag, w, h) {
    const n = document.createElement(tag);
    if (w) { n.width = w; n.height = h; n.style.width = "100%"; }
    return n;
  }
}
