// League view: standings from collection (resolved H2H or classic)
import { api, el, empty } from "../api.js";

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
    root.append(el("div", { class: "err" }, e.cold ? "computing… (retry in a moment)" : e.message));
    return;
  }
  const rows = (data.rows ?? []).map(r => r.record == null && r.wins != null
    ? { ...r, record: `${r.wins}-${r.draws}-${r.losses}` } : r);
  if (!rows.length) {
    root.append(empty(data.reason ?? "No league standings collected yet."));
    return;
  }
  const kind = data.kind === "h2h_resolved" ? "h2h_resolved" : "classic";
  const cols = COLUMNS[kind];
  root.append(el("div", { class: "meta" },
    kind === "h2h_resolved"
      ? `Resolved from event-live points · GW ${data.current_gw}`
      : `Official FPL table · GW points for GW ${data.points_gw ?? "?"}`));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...cols.map(([, label]) => el("th", {}, label)))), el("tbody"));
  for (const r of rows) {
    const tr = el("tr", {},
      ...cols.map(([k]) => el("td", {}, r[k] === null || r[k] === undefined ? "–"
        : typeof r[k] === "number" ? String(Math.round(r[k] * 10) / 10)
          : String(r[k]))));
    if (r.is_self) tr.className = "self-row";
    t.lastChild.append(tr);
  }
  root.append(t);
}
