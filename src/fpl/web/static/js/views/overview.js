// Overview view: GW banner, snapshot age, recent comparison/plan, artifacts
import { api, el, empty, cardRow } from "../api.js";

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "Overview"));
  let data;
  try {
    data = await api.meta();
  } catch (e) {
    if (e.cold) { root.append(el("div", { class: "loading" }, "computing… (retry in a moment)")); return; }
    root.append(el("div", { class: "err" }, e.message));
    return;
  }
  const age = data.snapshot_age_min != null ? `${Math.round(data.snapshot_age_min)} min`
    : data.snapshot_age ?? data.fetched_at ?? "unknown";
  root.append(cardRow([
    ["Season", data.season ?? "–"],
    ["Current GW", data.current_gw ?? "–"],
    ["Snapshot age", age],
    ["Artifacts", (data.artifacts ?? []).length],
  ]));
  const oneLiners = [].concat(
    (data.comparisons ?? []).slice(0, 3).map(c => [c.gw ?? c.file ?? "gw", c.summary ?? ""]),
    (data.plans ?? []).slice(0, 3).map(p => [p.gw ?? p.file ?? "gw", p.summary ?? ""]),
  );
  if (oneLiners.length) {
    root.append(el("h2", {}, "Recent comparisons & plans"));
    root.append(el("ul", {}, ...oneLiners.map(([g, s]) => el("li", {}, `GW${g}: `, String(s)))));
  }
  const arts = data.artifacts ?? [];
  root.append(el("h2", {}, "Artifact freshness"));
  if (!arts.length) { root.append(empty("No research artifacts on disk.")); return; }
  const max = Math.max(...arts.map(a => a.mtime ?? 0), 1);
  root.append(el("table", {}, el("tbody", {}, ...arts.map(a =>
    el("tr", {},
      el("td", {}, a.name ?? "?"),
      el("td", {}, a.mtime ? new Date(a.mtime * 1000).toLocaleString() : "–"),
      el("td", {}, a.mtime ? "·".repeat(1 + Math.round(9 * (a.mtime / max))) : ""))))));
}
