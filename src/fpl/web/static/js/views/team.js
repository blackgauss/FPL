// Team view: GW result cards, CDF performance analysis, squad detail grid,
// per-GW form, next-GW projection
import { api, el, empty, fmtPrice, fmtNum } from "../api.js";
import { bandChart } from "../charts.js";
import { openPlayerDrawer } from "./explorer.js";

const num = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));

function flagClass(f) {
  const s = String(f).toLowerCase();
  if (s === "ok" || s.includes("fit") || s.includes("available")) return "ok";
  if (/(price|owned|risk|doubt|rotate|fixture)/.test(s)) return "warn";
  return "bad";
}

function statCard(label, value, sub, cls = "") {
  return el("div", { class: `card stat ${cls}` },
    el("div", { class: "stat-v" }, value),
    el("div", {}, label),
    sub ? el("div", { class: "mut" }, sub) : null);
}

function pctBar(pct) {
  if (pct == null) return el("td", {}, "–");
  const color = pct >= 95 ? "#17803d" : pct <= 5 ? "#c62f2f" : "#2456d6";
  return el("td", {},
    el("span", { style: "display:inline-block;width:44px;height:8px;"
      + "background:#eceff3;border-radius:2px;vertical-align:middle;margin-right:6px",
    }, el("span", { style: `display:block;height:8px;width:${pct.toFixed(0)}%;`
      + `background:${color};border-radius:2px` })),
    `${pct.toFixed(0)}%`);
}

export async function render(root) {
  root.innerHTML = "";
  let data;
  try {
    data = await api.teamFlags({ gw: window.FPL_META?.current_gw || undefined });
  } catch (e) {
    root.append(el("div", { class: "err" }, e.cold ? "computing… (retry in a moment)" : e.message));
    return;
  }
  if (!data.available) {
    root.append(el("h1", {}, "My team"), empty(data.reason ?? "no collected team data"));
    return;
  }
  const rows = data.rows ?? [];
  const gw = data.gw;

  root.append(el("h1", {}, `My team — GW${gw} result`));
  const c = data.comparison;
  if (c) {
    const err = (c.error ?? 0);
    root.append(el("div", { class: "cards" },
      statCard("points", num(c.actual_score, 0), "settled (incl. captains)"),
      statCard("xPoints", num(c.xscore), "expected from fixtures"),
      statCard("historical", num(c.history_score, 0), "same-xi average"),
      statCard("model error", (err > 0 ? "+" : "") + num(err),
        `actual − model · ${c.score_source ?? ""}`, err >= 0 ? "ok" : "bad")));
  }

  performanceSection(root).catch(() => { /* forecast cold */ });
  squadGrid(root, rows).catch(() => { /* optional */ });
  ownFormSection(root).catch(() => { /* optional */ });

  root.append(el("h2", {}, `Squad — GW${gw} picks`));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["Lineup", "Pos", "Price", "EP next", "Flag"].map(h => el("th", {}, h)))), el("tbody"));
  for (const r of rows) {
    const bench = (r.slot ?? 0) > 11;
    const name = el("td", {}, r.web_name ?? `#${r.player_id}`,
      ...(r.is_captain ? [el("span", { class: "chip ok" }, "C")] : []),
      ...(r.is_vice_captain ? [el("span", { class: "chip" }, "A")] : []),
      ...(bench ? [el("span", { class: "chip" }, "bench")] : []));
    const chips = [el("span", { class: `chip ${flagClass(r.flag)}`,
      title: r.news || null }, String(r.flag ?? "ok"))];
    t.lastChild.append(el("tr", { class: bench ? "dim" : "" },
      name,
      el("td", {}, `#${r.slot ?? "?"}`),
      el("td", {}, fmtPrice(r.now_cost)),
      el("td", {}, fmtNum(r.ep_next)),
      el("td", {}, ...chips)));
  }
  root.append(t);
}

// -- CDF-based performance: percentiles / tail probabilities -------------------
async function performanceSection(root) {
  let perf;
  try { perf = await api.teamPerformance({ gw: window.FPL_META?.current_gw || undefined }); }
  catch { return; }
  if (!perf.available) return;
  root.append(el("h2", {}, `Performance vs model — GW${perf.gw} percentiles`));
  const s = perf.summary ?? {};
  root.append(el("div", { class: "meta" },
    `${(s.beat_95th ?? []).length} over the 95th percentile`
    + ((s.beat_95th ?? []).length ? ` (${s.beat_95th.join(", ")})` : "")
    + ` · ${(s.below_5th ?? []).length} below the 5th`
    + ((s.below_5th ?? []).length ? ` (${s.below_5th.join(", ")})` : "")));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["Player", "Min", "Actual", "Model", "90% interval",
        "Percentile of actual", "P(X ≥ actual)", "Δ"].map(h => el("th", {}, h)))), el("tbody"));
  for (const r of perf.rows) {
    const d = (r.actual_points ?? 0) - (r.model_pred ?? r.expected_points ?? 0);
    const pe = r.p_exceed;
    let pCell;
    if (pe == null) pCell = el("td", {}, "–");
    else if (pe < 0.05) pCell = el("td", { class: "num-ok" }, pe.toFixed(3) + " ↑ over");
    else if (pe > 0.95) pCell = el("td", { class: "num-bad" }, (1 - pe).toFixed(3) + " ↓ under");
    else pCell = el("td", {}, pe.toFixed(2));
    t.lastChild.append(el("tr", { class: r.minutes ? "" : "dim", style: "cursor:pointer",
      onclick: () => openPlayerDrawer({ player_code: r.player_code, web_name: r.web_name }),
    },
      el("td", {}, r.web_name ?? String(r.player_code),
        ...(r.is_captain ? [el("span", { class: "chip ok" }, "C")] : []),
        ...(r.is_vice_captain ? [el("span", { class: "chip" }, "A")] : [])),
      el("td", {}, String(r.minutes ?? "–")),
      el("td", {}, num(r.actual_points, 0)),
      el("td", { class: "mut" }, num(r.model_pred ?? r.expected_points)),
      el("td", { class: "mut", title: "central 90% credible interval (q05–q95)" },
        r.q05 == null ? "–" : `${num(r.q05)} – ${num(r.q95)}`),
      pctBar(r.percentile),
      pCell,
      el("td", d >= 0 ? { class: "num-ok" } : { class: "num-bad" },
        (d >= 0 ? "+" : "") + num(d))));
  }
  root.append(t);
  if (perf.note) root.append(el("div", { class: "mut" }, perf.note));
}

// -- explorer-grade detail for just my squad ------------------------------------
async function squadGrid(root, rows) {
  let table;
  try { table = await api.players({ limit: 1000 }); }
  catch { return; }
  const byCode = new Map((table.rows ?? []).map(r => [r.player_code, r]));
  const mine = rows.map(r => byCode.get(r.player_code)).filter(Boolean);
  if (!mine.length) return;
  root.append(el("h2", {}, "Squad detail (click for forecast)"));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["Player", "Pos", "Team", "Price", "Status", "Own %", "Own % lg",
        "Pred next", "xDG nxt"].map(h => el("th", {}, h)))), el("tbody"));
  for (const r of mine) {
    t.lastChild.append(el("tr", { class: "click", onclick: () => openPlayerDrawer(r) },
      el("td", {}, r.web_name ?? "?"),
      el("td", {}, r.position ?? ""),
      el("td", {}, r.team ?? ""),
      el("td", {}, fmtPrice(r.now_cost)),
      el("td", {}, r.status === "a" ? "ok" : String(r.status ?? "–")),
      el("td", {}, r.selected_by_percent != null ? fmtNum(r.selected_by_percent) : "–"),
      el("td", {}, r.own_league != null ? fmtNum(r.own_league) + "%" : "–"),
      el("td", {}, fmtNum(r.pred_next ?? r.ep_next)),
      el("td", {}, r.xdg_next != null ? fmtNum(r.xdg_next) : "–")));
  }
  root.append(t);
}

async function ownFormSection(root) {
  let data;
  try { data = await api.teamHistory(); } catch { return; }
  const rows = (data.rows ?? []).filter(r => r.gw != null);
  if (!data.available || !rows.length) return;
  root.append(el("h2", {}, "My form — points per GW"));
  const xs = rows.map(r => r.gw);
  const pts = rows.map(r => Number(r.points ?? 0));
  const zeros = pts.map(() => 0);
  const chart = el("div", { class: "chart" });
  root.append(chart);
  if (!bandChart(chart, xs, pts, zeros, pts, { height: 200, names: ["points", "", ""] })) {
    chart.replaceChildren();
  }
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["GW", "Points", "xScore", "Bench", "Transfers", "Rank"].map(h => el("th", {}, h)))), el("tbody"));
  for (const r of rows) {
    t.lastChild.append(el("tr", {},
      el("td", {}, "GW" + r.gw),
      el("td", {}, String(r.points ?? "–")),
      el("td", { class: "mut" }, fmtNum(r.xscore)),
      el("td", r.bench_points > 0 ? { class: "num-bad" } : {}, String(r.bench_points ?? "–")),
      el("td", {}, String(r.transfers ?? "–")),
      el("td", { class: "mut" }, String(r.rank ?? "–"))));
  }
  root.append(t);
}
