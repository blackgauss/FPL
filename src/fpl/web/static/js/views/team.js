// Team view: GW result cards, per-player model-vs-actual review, squad flags
import { api, el, empty, fmtPrice, fmtNum } from "../api.js";
import { bandChart } from "../charts.js";

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

function deltaCell(delta, maxAbs) {
  const w = Math.min(Math.abs(delta) / (maxAbs || 1), 1) * 46;
  const pos = delta >= 0;
  const bar = el("span", {
    style: `display:inline-block;height:8px;width:${w.toFixed(0)}px;`
      + `background:${pos ? "#17803d" : "#c62f2f"};border-radius:2px;`
      + `margin-right:6px;vertical-align:middle`,
  });
  return el("td", { style: `text-align:${pos ? "left" : "right"}`,
    class: pos ? "num-ok" : "num-bad" },
    ...(pos ? [bar] : []), num(delta) + " pts", ...(pos ? [] : [bar]));
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

  if (c?.players?.length) {
    const ps = c.players;
    const maxAbs = Math.max(10, ...ps.map(p => Math.abs((p.actual_points ?? 0) - (p.expected_points ?? 0))));
    root.append(el("h2", {}, "Model vs actual"));
    const t = el("table", {}, el("thead", {}, el("tr", {},
      ...["Player", "Min", "Actual", "Model", "Δ"].map(h => el("th", {}, h)))), el("tbody"));
    for (const p of ps) {
      const d = (p.actual_points ?? 0) - (p.expected_points ?? 0);
      const chips = [];
      if (p.is_captain) chips.push(el("span", { class: "chip ok" }, "C"));
      if (p.is_vice_captain) chips.push(el("span", { class: "chip" }, "A"));
      t.lastChild.append(el("tr", { class: p.minutes ? "" : "dim" },
        el("td", {}, p.web_name ?? String(p.player_code), ...chips),
        el("td", {}, String(p.minutes ?? "–")),
        el("td", {}, num(p.actual_points, 0)),
        el("td", { class: "mut" }, num(p.expected_points)),
        deltaCell(d, maxAbs)));
    }
    root.append(t);
    if (c.score_source) root.append(el("div", { class: "mut" },
      `Actuals from ${c.score_source}. Rows with 0 minutes are bench/unused.`));
  }

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

  ownFormSection(root).catch(() => { /* optional section */ });

  // next-GW model outlook for the owned players (only when forecast is warm)
  const codes = rows.map(r => r.player_code).filter(x => x != null);
  const nextGw = (window.FPL_META?.current_gw || gw) + 1;
  if (codes.length) {
    try {
      const fc = await api.forecast({ player_codes: codes.join(","), gw_start: nextGw, horizon: 1 });
      const by = Object.fromEntries((fc.rows ?? []).map(r => [r.player_code, r]));
      if (fc.rows?.length) {
        root.append(el("h2", {}, `Projected GW${nextGw}`));
        const ft = el("table", {}, el("thead", {}, el("tr", {},
          ...["Player", "pred", "q25", "q75"].map(h => el("th", {}, h)))), el("tbody"));
        for (const r of rows) {
          const p = by[r.player_code];
          if (!p) continue;
          ft.lastChild.append(el("tr", {},
            el("td", {}, r.web_name ?? String(r.player_code)),
            el("td", {}, fmtNum(p.pred)),
            el("td", { class: "mut" }, fmtNum(p.quantiles?.q25)),
            el("td", { class: "mut" }, fmtNum(p.quantiles?.q75))));
        }
        root.append(ft);
      }
    } catch { /* forecast cold or no rows for this window: skip section */ }
  }
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
