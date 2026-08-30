// League view: sortable standings with own-team highlight
import { api, el, empty } from "../api.js";

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "League"));
  const meta = window.FPL_META ?? {};
  let data;
  try {
    data = await api.leagueStandings({ entry_id: meta.entry_id ?? undefined });
  } catch (e) {
    root.append(el("div", { class: "err" }, e.cold ? "computing… (retry in a moment)" : e.message));
    return;
  }
  let rows = data.rows ?? data.standings ?? (Array.isArray(data) ? data : []);
  if (!rows.length) { root.append(empty("No league standings collected yet.")); return; }
  const cols = Object.keys(rows[0]).filter(k => !/^_/.test(k)).slice(0, 10);
  const selfId = meta.entry_id ?? data.entry_id;
  let sortKey = null, dir = -1;
  const box = el("div");
  const draw = () => {
    if (sortKey) rows = rows.slice().sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      return (typeof x === "number" && typeof y === "number" ? x - y
        : String(x).localeCompare(String(y))) * dir;
    });
    const t = el("table", {}, el("thead", {}, el("tr", {},
      ...cols.map(k => el("th", {
        onclick: () => { if (sortKey === k) dir = -dir; else { sortKey = k; dir = -1; } draw(); },
      }, k)))), el("tbody"));
    for (const r of rows) {
      const tr = el("tr", {},
        ...cols.map(k => el("td", {}, r[k] === null || r[k] === undefined ? "–"
          : typeof r[k] === "number" ? String(Math.round(r[k] * 10) / 10) : String(r[k]))));
      if (selfId != null && (r.entry_id ?? r.entry ?? r.epx) === selfId) {
        tr.className = "self-row";
      }
      t.lastChild.append(tr);
    }
    box.replaceChildren(t);
  };
  root.append(box);
  draw();
}
