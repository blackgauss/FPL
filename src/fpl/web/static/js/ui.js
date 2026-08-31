// Shared view atoms: chips, bars and the cold-forecast error banner.
// Views must not re-skin these locally — divergence here is exactly the
// drift pattern this layer is meant to kill.
import { el } from "./api.js";

export function fail(e) {
  // Every view's cold/retry banner is byte-identical; one definition.
  return el("div", { class: "err" },
    e?.cold ? "computing… (retry in a moment)" : String(e?.message ?? e));
}

export function chip(cls, text, title = null) {
  return el("span", { class: `chip ${cls || ""}`.trim(),
                      title: title ?? undefined }, text);
}

// horizontal proportion bar inside a fixed track; `frac` is 0..1
export function bar(frac, color) {
  return el("span", { style: "display:inline-block;width:60px;height:8px;"
    + "background:#eceff3;border-radius:2px;vertical-align:middle;"
    + "margin-right:6px" },
  el("span", { style: `display:block;height:8px;width:${
    Math.round(Math.max(0, Math.min(1, frac)) * 100)}%;background:${
    color};border-radius:2px` }));
}
