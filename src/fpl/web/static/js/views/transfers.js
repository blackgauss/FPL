// Transfers view: weekly plan options with impact bars + in/out CDF preview
import { api, el, empty, fmtNum, fmtPrice, resolveForecastGw } from "../api.js";
import { multiLineChart } from "../charts.js";

export async function render(root) {
  root.innerHTML = "";
  root.append(el("h1", {}, "Transfers"));
  let data;
  try {
    data = await api.transferSuggestions({});
  } catch (e) {
    root.append(el("div", { class: "err" }, e.cold ? "computing… (retry in a moment)" : e.message));
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
    + (data.ownership_basis ? ` · ownership: ${data.ownership_basis}` : "")));

  const gainMin = Math.min(0, ...options.map(o => o.expected_gain ?? 0));
  const gainMax = Math.max(1, ...options.map(o => o.expected_gain ?? 0));
  const t = el("table", {}, el("thead", {}, el("tr", {},
    ...["#", "In", "Out", "Expected gain", "Exp. score", "Own in", "Own out", "C / VC", "Compare"]
      .map(h => el("th", {}, h)))), el("tbody"));

  options.forEach((o, i) => {
    const gain = o.expected_gain ?? 0;
    const w = Math.round(((gain - gainMin) / (gainMax - gainMin || 1)) * 60);
    const bar = el("span", {
      style: `display:inline-block;height:8px;width:${w}px;background:${
        gain >= 0 ? "#17803d" : "#c62f2f"};border-radius:2px;margin-right:6px;vertical-align:middle`,
    });
    const ownChip = (v) => el("td", {}, el("span", {
      class: `chip ${v != null && v > 30 ? "warn" : ""}`,
    }, v == null ? "–" : fmtNum(v) + "%"));
    const detail = el("div", { class: "meta" }, "loading forecast comparison…");
    const det = el("details", {}, el("summary", {}, "CDF in vs out"), detail);
    let loaded = false;
    det.addEventListener("toggle", () => {
      if (!det.open || loaded) return;
      loaded = true;
      compareCdf(detail, o, data.gw);
    });
    t.lastChild.append(el("tr", {},
      el("td", {}, String(i + 1)),
      el("td", {}, String(o.transfer_in ?? "?")),
      el("td", { class: "mut" }, String(o.transfer_out ?? "?")),
      el("td", gain >= 0 ? { class: "num-ok" } : { class: "num-bad" },
        bar, "+" + fmtNum(gain) + " pts"),
      el("td", {}, fmtNum(o.expected_score)),
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

async function compareCdf(box, o, gw) {
  const codes = [o.transfer_in_code, o.transfer_out_code].filter(c => c != null);
  if (codes.length < 2) { box.textContent = "codes unavailable for CDF."; return; }
  try {
    const [inn, out] = await Promise.all(codes.map(c => cdfFor(c, gw)));
    const pct = (d) => Math.round((d.tails?.["5"]?.p_gt ?? 0) * 100);
    const label = (d, side) => `${side} ${d.web_name ?? d.player_code}`;
    if (!multiLineChart(box2(box), inn.xs, [
      { label: label(inn, "in"), ys: inn.cdf },
      { label: label(out, "out"), ys: out.cdf },
    ], { yPercent: true, xlabel: "points" })) box.textContent = "chart unavailable";
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
