// Transfers view: weekly plan options with impact bars + in/out CDF preview
import { api, el, empty, fmtNum, fmtPrice, resolveForecastGw } from "../api.js";
import { multiLineChart } from "../charts.js";
import { bar, chip, fail } from "../ui.js";

const VALUES = {
  model_digest: "our t-digest model",
  "model_digest+ep_fallback": "t-digest; official ep for uncovered players",
  model_point: "our point model",
  "model_point+ep_fallback": "point model; official ep for uncovered players",
  official_ep: "official ep_next (no model coverage)",
};
// QS levels (fpl.dist) the XI quantile vectors are indexed by
const XI_Q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "Transfers"));
  let data;
  try {
    data = await api.transferSuggestions({});
  } catch (e) {
    root.append(fail(e));
    return;
  }
  const options = data.suggestions ?? [];
  if (!data.available || !options.length) {
    root.append(empty(data.reason ?? "No plan collected yet (run the weekly planner)."));
    return;
  }
  root.append(el("div", { class: "meta" },
    `GW${data.gw} plan · from ${data.source}`
    + (data.bank_tenths != null ? ` · bank ${fmtPrice(Math.abs(data.bank_tenths * 10)).replace("£", data.bank_tenths < 0 ? "-£" : "£")}` : "")
    + (data.ownership_basis ? ` · ownership: ${data.ownership_basis}` : "")
    + (data.expected_source ? ` · values: ${VALUES[data.expected_source] ?? data.expected_source}` : "")));

  const gainMin = Math.min(0, ...options.map(o => o.expected_gain ?? 0));
  const gainMax = Math.max(1, ...options.map(o => o.expected_gain ?? 0));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["#", "In", "Out", "Expected gain", "Exp. score", "XI floor–ceiling", "Own in", "Own out", "C / VC", "Compare"]
      .map(h => el("th", {}, h)))), el("tbody"));

  options.forEach((o, i) => {
    const gain = o.expected_gain ?? 0;
    const gainBar = bar((gain - gainMin) / (gainMax - gainMin || 1),
                        gain >= 0 ? "#17803d" : "#c62f2f");
    const ownChip = (v) => el("td", {},
      chip(v != null && v > 30 ? "warn" : "", v == null ? "–" : fmtNum(v) + "%"));
    const detail = el("div", { class: "meta" }, "loading forecast comparison…");
    const det = el("details", {}, el("summary", {}, "CDF in vs out"), detail);
    let loaded = false;
    det.addEventListener("toggle", () => {
      if (!det.open || loaded) return;
      loaded = true;
      compareCdf(detail, o, data.gw,
                 (data.suggestions.find(x => !x.transfer_out) ?? {}).xi_quantiles);
    });
    t.lastChild.append(el("tr", {},
      el("td", {}, String(i + 1)),
      el("td", {}, String(o.transfer_in ?? "?")),
      el("td", { class: "mut" }, String(o.transfer_out ?? "?")),
      el("td", gain >= 0 ? { class: "num-ok" } : { class: "num-bad" },
        gainBar, "+" + fmtNum(gain) + " pts"),
      el("td", {}, fmtNum(o.expected_score)),
      el("td", { title: o.prob_beat_hold != null
        ? `P(beats hold lineup) ${Math.round(o.prob_beat_hold * 100)}%` : "" },
        o.xi_q10 != null ? `${fmtNum(o.xi_q10)}–${fmtNum(o.xi_q90)} pts` : "–"),
      ownChip(o.ownership_in),
      ownChip(o.ownership_out),
      el("td", {}, el("span", {}, `C ${capName(data, o.captain)} · VC ${capName(data, o.vice_captain)}`))));
    t.lastChild.lastChild.append(el("td", {}, det));
  });
  root.append(t);
}

function capName(data, code) {
  if (code == null) return "–";
  return typeof code === "string" ? code : `#${code}`;
}

async function cdfFor(code, gw) {
  const g = resolveForecastGw(gw);  // one clamp policy (see api.js)
  return api.forecastCdf({ player_code: code, gw: g, n: 48, at: "5" });
}

async function compareCdf(box, o, gw, holdQ) {
  const codes = [o.transfer_in_code, o.transfer_out_code].filter(c => c != null);
  if (codes.length < 2) { box.textContent = "codes unavailable for CDF."; return; }
  try {
    const [inn, out] = await Promise.all(codes.map(c => cdfFor(c, gw)));
    // XI-total step CDFs on the same points axis: quantile vectors read
    // off the t-digest, added quantile by quantile (see weekly planner)
    const xiOn = (q) => inn.xs.map(x => {
      let p = 0;
      XI_Q.forEach((qv, k) => { if (x >= q[k]) p = qv; });
      return p;
    });
    const pct = (d) => Math.round((d.tails?.["5"]?.p_gt ?? 0) * 100);
    const label = (d, side) => `${side} ${d.web_name ?? d.player_code}`;
    const series = [
      { label: label(inn, "in"), ys: inn.cdf },
      { label: label(out, "out"), ys: out.cdf },
    ];
    if (o.xi_quantiles && holdQ) {
      series.push({ label: "hold XI", ys: xiOn(holdQ) },
                  { label: "plan XI", ys: xiOn(o.xi_quantiles) });
    }
    if (!multiLineChart(box2(box), inn.xs, series,
                        { yPercent: true, xlabel: "points" }))
      box.textContent = "chart unavailable";
    box.append(el("div", { class: "meta" },
      "P(≥5 pts): in " + pct(inn) + "% vs out " + pct(out) + "%"
      + ((inn.gw != null && inn.gw !== gw)
        ? ` · ${gw > inn.gw ? `GW${gw} window not published yet — GW${inn.gw} distributions shown`
          : `GW${inn.gw} window`}`
        : "")));
  } catch (e) {
    box.textContent = e.cold ? "computing forecast…" : `cdf: ${e.message}`;
  }
}

function box2(holder) {
  const b = el("div", { class: "chart" });
  holder.replaceChildren(b);
  return b;
}

