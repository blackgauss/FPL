// Team view: squad flags + GW comparison summary
import { api, el, empty, fmtPrice, cardRow } from "../api.js";

function flagClass(f) {
  const s = String(f).toLowerCase();
  if (s === "ok" || s.includes("fit") || s.includes("available")) return "flag-ok";
  if (/(price|owned|risk|doubt|rotate|fixture)/.test(s)) return "flag-warn";
  return "flag-bad";
}

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "My team"));
  let data;
  try {
    data = await api.teamFlags({ gw: window.FPL_META?.current_gw || undefined });
  } catch (e) {
    root.append(el("div", { class: "err" }, e.cold ? "computing… (retry in a moment)" : e.message));
    return;
  }
  const rows = data.rows ?? data.squad ?? data.players ?? (Array.isArray(data) ? data : []);
  if (!rows.length) { root.append(empty("No team data collected yet.")); return; }
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["Player", "Pos", "Team", "Price", "Flags"].map(h => el("th", {}, h)))), el("tbody"));
  for (const r of rows) {
    const flags = Array.isArray(r.flags) ? r.flags : [r.flag ?? r.flags ?? "ok"];
    const chips = [];
    if (r.captain || r.is_captain) chips.push(el("span", { class: "chip ok" }, "C"));
    if (r.vice_captain || r.is_vice_captain) chips.push(el("span", { class: "chip" }, "A"));
    for (const f of (flags.length ? flags : ["ok"])) {
      const c = flagClass(f);
      chips.push(el("span", { class: `chip ${c === "flag-ok" ? "ok" : c === "flag-warn" ? "warn" : "bad"}` }, String(f)));
    }
    t.lastChild.append(el("tr", {},
      el("td", {}, r.web_name ?? r.name ?? "?"),
      el("td", {}, r.position ?? r.pos ?? ""),
      el("td", {}, r.team ?? r.team_short ?? ""),
      el("td", {}, fmtPrice(r.now_cost ?? r.price)),
      el("td", {}, ...chips)));
  }
  root.append(t);
  const summary = data.summary ?? data.gw_summary ?? data.comparison ??
    (Array.isArray(data.comparisons) ? data.comparisons[0]?.summary : null);
  if (summary && typeof summary === "object") {
    root.append(el("h2", {}, "GW comparison summary"));
    root.append(cardRow(Object.entries(summary)));
  }
}
