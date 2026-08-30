// Band+line chart over uPlot (canvas fallback if uPlot is absent)
export function bandChart(host, xs, ys, lo, hi, opts = {}) {
  host.innerHTML = "";
  const data = [xs, ys, lo, hi, opts.lo2, opts.hi2].filter(d => d);
  const W = host.clientWidth || 560, H = opts.height || 280;
  if (!window.uPlot) { canvasBand(host, data, opts, W, H); return; }
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
  new window.uPlot({
    target: host, width: W, height: H, data, series,
    scales: { x: { time: false } },
    axes: [{ size: 40 }, { size: 40 }],
  }, true);
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
