// Transfers view: suggested in/out cards from the weekly plan
import { api, el, empty, fmtNum, fmtPrice } from "../api.js";

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "Transfers"));
  let data;
  try {
    data = await api.transferSuggestions({ gw: window.FPL_META?.current_gw || undefined });
  } catch (e) {
    root.append(el("div", { class: "err" }, e.cold ? "computing… (retry in a moment)" : e.message));
    return;
  }
  const plans = data.plans ?? data.suggestions ??
    (Array.isArray(data) ? data : [data]);
  if (!plans.length || plans.every(p => !(p.in && p.out)) && !plans.some(p => p.in?.length || p.out?.length)) {
    root.append(empty("No plan collected yet."));
    return;
  }
  for (const plan of plans) {
    const ins = [].concat(plan.in ?? plan.in_players ?? []);
    const outs = [].concat(plan.out ?? plan.out_players ?? []);
    if (!ins.length && !outs.length) continue;
    root.append(el("h2", {}, plan.gw != null ? `GW${plan.gw} plan` : plan.file ?? "Plan"));
    root.append(el("div", { class: "cards" },
      ...outs.map(p => transferCard(p, "out")),
      ...ins.map(p => transferCard(p, "in"))));
  }
}

function transferCard(p, dir) {
  const name = typeof p === "string" ? p : (p.web_name ?? p.name ?? "?");
  const gain = typeof p === "object" ? (p.expected_gain ?? p.gain) : undefined;
  const penalty = typeof p === "object" ? (p.penalty ?? p.suggested?.penalty) : undefined;
  return el("div", { class: "card" },
    el("div", { class: "k" }, dir === "in" ? "IN · " : "OUT · ",
      String(p?.position ?? p?.pos ?? "")),
    el("div", { class: "v" }, name),
    el("div", { class: dir === "in" ? "flag-ok" : "flag-bad" },
      gain != null ? `Δ ${fmtNum(gain)} pts` : ""),
    el("div", { class: "k" },
      penalty != null ? `penalty ${fmtNum(penalty)} pts` :
      (p && typeof p === "object" && p.price != null ? fmtPrice(p.price) : "")));
}
