// Research view: artifact list + recursive metrics tables
import { api, el, empty } from "../api.js";

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "Research"));
  let artifacts = window.FPL_META?.artifacts;
  if (!artifacts?.length) {
    try { artifacts = (await api.meta()).artifacts; } catch { artifacts = []; }
  }
  if (!artifacts?.length) { root.append(empty("No research artifacts found.")); return; }
  root.append(el("ul", {}, ...artifacts.map(a =>
    el("li", {}, el("a", {
      href: "#", style: "cursor:pointer;color:var(--acc)",
      onclick: async (ev) => {
        ev.preventDefault();
        await showRun(a.name ?? a, out);
      },
    }, a.name ?? String(a))))));
  const out = el("div");
  root.append(el("h2", {}, "Metrics"), out);
}

async function showRun(run, out) {
  out.replaceChildren(el("div", { class: "loading" }, "Loading metrics…"));
  try {
    const data = await api.researchMetrics(run);
    out.replaceChildren(el("h3", {}, run), ...jsonTable(data));
  } catch (e) {
    out.replaceChildren(el("div", { class: "err" },
      e.cold ? "computing… (retry in a moment)" : e.message));
  }
}

export function jsonTable(obj) {
  if (obj === null || obj === undefined) return [el("em", {}, "null")];
  if (typeof obj !== "object") return [el("span", {}, typeof obj === "number"
    ? String(Math.round(obj * 1e4) / 1e4) : String(obj))];
  if (Array.isArray(obj)) {
    if (!obj.length) return [el("em", {}, "[]")];
    if (obj.every(v => v === null || typeof v !== "object")) {
      const arr = obj.map(v => String(v));
      const ncol = obj.length > 12 ? 4 : obj.length > 3 ? 2 : 1;
      const t = el("table", {}, el("tbody", {}, ...chunk(arr, ncol).map(row =>
        el("tr", {}, ...row.map(v => el("td", {}, v))))));
      return [t];
    }
    const cols = [...new Set(obj.flatMap(r => Object.keys(r ?? {})))];
    return [el("table", {},
      el("thead", {}, el("tr", {}, ...cols.map(c => el("th", {}, c)))),
      el("tbody", {}, ...obj.map(r => el("tr", {},
        ...cols.map(c => el("td", {}, ...jsonTable(r?.[c])))))))];
  }
  const entries = Object.entries(obj);
  if (!entries.length) return [el("em", {}, "{}")];
  return [el("table", {}, el("tbody", {}, ...entries.map(([k, v]) =>
    el("tr", {}, el("td", {}, el("b", {}, k)),
      el("td", {}, ...(typeof v === "object" && v !== null
        ? jsonTable(v) : [el("span", {}, typeof v === "number"
          ? String(Math.round(v * 1e4) / 1e4) : String(v))]))))))];
}

function chunk(arr, n) {
  const out = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
}
