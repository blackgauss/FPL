// League view: standings + per-manager form, H2H record, league ownership
import { api, el, empty, fmtNum } from "../api.js";
import { sparkBars } from "../charts.js";
import { bar, chip, fail } from "../ui.js";

const COLUMNS = {
  h2h_resolved: [
    ["rank", "#"], ["entry_name", "Team"], ["player_name", "Manager"],
    ["resolved_score", "Score"], ["league_points", "LPt"],
    ["record", "W-D-L"],
  ],
  classic: [
    ["rank", "#"], ["entry_name", "Team"], ["player_name", "Manager"],
    ["total", "Total"], ["gw_points", "GW pts"], ["last_rank", "Last rank"],
  ],
};

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "League"));
  let data;
  try {
    data = await api.leagueStandings({});
  } catch (e) {
    root.append(fail(e));
    return;
  }
  const rows = (data.rows ?? []).map(r => r.record == null && r.wins != null
    ? { ...r, record: `${r.wins}-${r.draws}-${r.losses}` } : r);
  if (!rows.length) {
    root.append(empty(data.reason ?? "No league standings collected yet."));
    return;
  }
  let report = null;
  try {
    const rr = await api.leagueReport();
    report = rr.available ? rr : null;
  } catch { /* no collected matches: table only */ }

  const kind = data.kind === "h2h_resolved" ? "h2h_resolved" : "classic";
  const cols = COLUMNS[kind];
  root.append(el("div", { class: "meta" },
    kind === "h2h_resolved"
      ? `Resolved from event-live points · GW ${data.current_gw}`
      : `Official FPL table · GW points for GW ${data.points_gw ?? "?"}`));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...cols.map(([, label]) => el("th", {}, label)),
    ...(report ? [el("th", { title: "points per GW, newest right" }, "Form")] : []))), el("tbody"));
  for (const r of rows) {
    const cells = cols.map(([k]) => el("td", {}, r[k] === null || r[k] === undefined ? "–"
      : typeof r[k] === "number" ? String(Math.round(r[k] * 10) / 10)
        : String(r[k])));
    if (report) {
      const series = report.managers[String(r.entry_id)] ?? {};
      const vals = report.events.map(g => (series[String(g)] ?? {}).points ?? null);
      const spark = el("span", {});
      sparkBars(spark, vals, {});
      cells.push(el("td", {}, spark));
    }
    const tr = el("tr", {}, ...cells);
    if (r.is_self) tr.className = "self-row";
    t.lastChild.append(tr);
  }
  root.append(t);

  if (report && data.entry_id != null) drawRecord(root, report, data.entry_id);
  exposureSection(root).catch(() => { /* optional panel */ });
  ownershipSection(root).catch(() => { /* optional panel */ });
}

function drawRecord(root, report, me) {
  const mine = report.managers[String(me)] ?? {};
  const gwKeys = Object.keys(mine).map(Number).sort((a, b) => a - b);
  if (!gwKeys.length) return;
  root.append(el("h2", {}, "My matches"));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["GW", "Me", "Opponent", "They", "Result"].map(h => el("th", {}, h)))), el("tbody"));
  for (const gw of gwKeys) {
    const cell = mine[String(gw)];
    const res = cell.result;
    t.lastChild.append(el("tr", {},
      el("td", {}, "GW" + gw),
      el("td", {}, fmtNum(cell.points)),
      el("td", { class: "mut" }, `#${cell.opponent}`),
      el("td", { class: "mut" }, fmtNum(cell.opponent_points)),
      el("td", {}, chip(res === "W" ? "ok" : res === "L" ? "bad" : "", res))));
  }
  root.append(t);
}

async function ownershipSection(root) {
  let data;
  try { data = await api.leagueOwnership(); } catch { return; }
  if (!data.available || !data.rows?.length) return;
  root.append(el("h2", {}, `League ownership — ${data.rows.length} owned players`));
  root.append(el("div", { class: "meta" }, data.basis
    + " · diff = league % minus official %; + big = over-owned by friends (differential upside elsewhere)"));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["Player", "Pos", "Team", "League %", "Official %", "Diff", "Pred next"]
      .map(h => el("th", {}, h)))), el("tbody"));
  for (const r of data.rows.slice(0, 40)) {
    t.lastChild.append(el("tr", {},
      el("td", {}, r.web_name ?? String(r.player_code)),
      el("td", {}, r.position ?? ""),
      el("td", {}, r.team ?? ""),
      el("td", {}, fmtNum(r.own_league) + "%"),
      el("td", { class: "mut" }, r.own_official == null ? "–" : fmtNum(r.own_official) + "%"),
      el("td", r.diff != null && r.diff >= 15 ? { class: "num-bad" }
        : r.diff != null && r.diff <= -5 ? { class: "num-ok" } : {},
      r.diff == null ? "–" : (r.diff > 0 ? "+" : "") + fmtNum(r.diff)),
      el("td", {}, fmtNum(r.pred_next))));
  }
  root.append(t);
}

async function exposureSection(root) {
  let data;
  try { data = await api.leagueExposure(); } catch { return; }
  if (!data.available || !data.rows?.length) return;
  root.append(el("h2", {}, `Your exposure — league GW${data.gw}`));
  root.append(el("div", { class: "meta" },
    `how many of the ${data.league_entries} league managers also own each of`
    + " yours · heavily owned assets are net-zero even when they score big"));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["Player", "XIs", "% owning", "Status"]
      .map(h => el("th", {}, h)))), el("tbody"));
  for (const r of data.rows) {
    const tag = r.pct >= 50 && r.started ? "public stock"
      : r.pct <= 15 && r.started ? "edge" : "";
    t.lastChild.append(el("tr", {},
      el("td", {}, (r.web_name ?? String(r.element)) + (r.captain ? " (C)" : "")),
      el("td", {}, bar(r.pct / 100, r.pct >= 50 ? "#c0392b" : "#7f8c8d")),
      el("td", r.pct >= 50 ? { class: "num-bad" } : r.pct <= 15 ? { class: "num-ok" } : {},
        `${r.pct}% (${r.managers_owning})`),
      el("td", {}, tag ? chip(r.pct >= 50 ? "warn" : "ok", tag) : "")));
  }
  root.append(t);
}
