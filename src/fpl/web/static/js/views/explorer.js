// Explorer view: faceted player table + forecast drawer
import { api, el, empty, fmtPrice, fmtNum, resolveForecastGw } from "../api.js";
import { bandChart, multiLineChart } from "../charts.js";

const PAGE = 25;
const CHIP = {
  a: ["ok", "OK"], d: ["warn", "?"], i: ["bad", "INJ"], s: ["bad", "SUSP"], u: ["bad", "OUT"],
};

export async function render(root) {
  const st = { search: "", position: "", availableOnly: false, maxPrice: "",
    offset: 0, total: 0, sort: "", dir: "asc" };
  const controls = el("div", { class: "controls" },
    el("input", {
      type: "search", placeholder: "search players…",
      oninput: debounce(ev => { st.search = ev.target.value; st.offset = 0; load(); }, 300),
    }),
    select(["", "GKP", "DEF", "MID", "FWD"], v => { st.position = v; st.offset = 0; load(); }),
    el("label", {}, el("input", {
      type: "checkbox", onchange: ev => { st.availableOnly = ev.target.checked; st.offset = 0; load(); },
    }), " available only"),
    el("label", {}, "max £m ", el("input", {
      type: "number", min: 4, max: 15, step: 0.1, size: 4,
      onchange: ev => { st.maxPrice = ev.target.value ? Math.round(ev.target.value * 10) : ""; st.offset = 0; load(); },
    })),
  );
  const tableBox = el("div");
  const pager = el("div", { class: "pager" });
  root.append(el("h1", {}, "Player explorer"), controls, tableBox, pager);

  async function load() {
    if (controls._loading) return;
    controls._loading = true;
    tableBox.replaceChildren(el("div", { class: "loading" }, "Loading…"));
    try {
      const data = await api.players({
        search: st.search, position: st.position, max_price: st.maxPrice,
        sort: st.sort || undefined,
        dir: st.sort ? st.dir : undefined,
        limit: PAGE, offset: st.offset,
      });
      let rows = data.rows ?? [];
      if (st.availableOnly) rows = rows.filter(r => !["i", "s", "u"].includes(r.status));
      st.total = data.total ?? rows.length;
      tableBox.replaceChildren(rows.length ? tbl(rows) : empty("No players match these filters."));
      pager.replaceChildren(
        el("button", { disabled: st.offset === 0, onclick: () => { st.offset = Math.max(0, st.offset - PAGE); load(); } }, "← prev"),
        el("span", {}, `${st.total ? st.offset + 1 : 0}–${Math.min(st.offset + PAGE, st.total)} of ${st.total}`),
        el("button", { disabled: st.offset + PAGE >= st.total, onclick: () => { st.offset += PAGE; load(); } }, "next →"),
      );
    } catch (e) {
      tableBox.replaceChildren(el("div", { class: "err" },
        e.cold ? "computing… (retry in a moment)" : e.message));
      pager.replaceChildren();
    } finally { controls._loading = false; }
  }

  const COLUMNS = [
    ["Player", "web_name"], ["Pos", "position"], ["Team", "team"],
    ["Price", "now_cost"], ["Status", "status"],
    ["Own %", "selected_by_percent"], ["Own % lg", "own_league"],
    ["Expect", "expected"], ["xDG nxt", "xdg_next"], ["xDG ~5", "xdg_next5"],
  ];

  function headerCell(h, key) {
    if (!key) return el("th", {}, h);
    const arrow = st.sort === key ? (st.dir === "asc" ? " ▲" : " ▼") : "";
    return el("th", {
      title: "sort by " + h,
      onclick: () => {
        if (st.sort === key) st.dir = st.dir === "asc" ? "desc" : "asc";
        else { st.sort = key; st.dir = "asc"; }
        st.offset = 0;
        load();
      },
    }, h + arrow);
  }

  function tbl(rows) {
    const t = el("table", {}, el("thead", {}, el("tr", {},
      ...COLUMNS.map(([h, key]) => headerCell(h, key)))), el("tbody"));
    for (const r of rows) {
      const [cls, txt] = CHIP[r.status] ?? ["", r.status ?? "–"];
      const price = r.now_cost ?? r.price ?? r.cost_in_ten_thousands;
      t.lastChild.append(el("tr", { class: "click", onclick: () => drawer(r) },
        el("td", {}, r.web_name ?? r.name ?? "?"),
        el("td", {}, r.position ?? r.pos ?? ""),
        el("td", {}, r.team ?? r.team_short ?? r.club ?? ""),
        el("td", {}, fmtPrice(price)),
        el("td", {}, el("span", { class: `chip ${cls}` }, txt)),
        el("td", {}, r.selected_by_percent != null ? fmtNum(r.selected_by_percent) : "–"),
        el("td", { title: "owned by league managers (latest collected picks)" },
          r.own_league != null ? fmtNum(r.own_league) + "%" : "–"),
        el("td", {
          title: r.expected_source === "official" ? "official FPL ep_next (no model row)" : "",
        }, fmtNum(r.expected)),
        el("td", { title: "opponent strength next GW, 0 (weak) – 100 (strong)" },
          r.xdg_next != null ? fmtNum(r.xdg_next) : "–"),
        el("td", {}, r.xdg_next5 != null ? fmtNum(r.xdg_next5) : "–")));
    }
    return t;
  }
  load();
}

async function drawer(p) {
  const box = document.getElementById("drawer");
  box.classList.remove("hidden");
  const code = p.player_code ?? p.code;
  const gw = (window.FPL_META?.current_gw || 0) + 1;
  box.replaceChildren(
    el("button", { class: "close", onclick: () => box.classList.add("hidden") }, "×"),
    el("h1", {}, `${p.web_name ?? "?"} `),
    el("div", { class: "meta" }, `${p.position ?? ""} · ${p.team ?? ""} · ${fmtPrice(p.now_cost)}`),
    el("div", { class: "loading" }, "Loading forecast…"),
  );
  if (code === undefined) return;
  let node;
  try { node = await fetchForecast(code, gw); }
  catch (e) { node = el("div", { class: "err" }, String(e?.message ?? e)); }
  box.lastChild.replaceWith(node);
}

async function fetchForecast(code, gw, retries = 2) {
  const holder = el("div");
  holder.replaceChildren(el("div", { class: "loading" }, "Loading forecast…"));
  try {
    const target = resolveForecastGw(gw);
    const data = await api.forecast({ player_codes: code, gw_start: target,
                                      horizon: 5 });
    const rows = (data.rows ?? data.forecast ?? (Array.isArray(data) ? data : []))
      .slice().sort((a, b) => a.gw - b.gw);
    const fallbackNote = target !== gw
      ? `GW${gw} features are not published yet (its source GW is still `
        + "settling) — showing latest scoreable window instead."
      : null;
    if (!rows.length) {
      return empty("No model forecast yet for this player: the feature store has "
        + "no rows in this GW window (run `dvc repro` after fixture/data updates).");
    }
    holder.replaceChildren();
    if (fallbackNote) holder.append(el("div", { class: "card" }, fallbackNote));
    const col = (keys) => rows.map(r => {
      for (const k of keys) if (r[k] !== undefined) return r[k];
      const qq = r.quantiles ?? r.quantiles_struct;
      if (qq) for (const k of keys) if (qq[k] !== undefined) return qq[k];
      return null;
    });
    const xs = rows.map(r => r.gw), ys = rows.map(r => r.pred);
    const q05 = col(["q05", "q5"]), q25 = col(["q25"]), q75 = col(["q75"]), q95 = col(["q95"]);
    const chart = el("div", { class: "chart" });
    holder.append(el("h2", {}, `Forecast GW${xs[0]}${xs.length > 1 ? "–" + xs[xs.length - 1] : ""}`), chart);
    if (!bandChart(chart, xs, ys, q05, q95, { lo2: q25, hi2: q75 })) {
      chart.replaceChildren(el("div", { class: "meta" },
        "Single gameweek in window — quantiles listed below."));
    }
    holder.append(el("table", {}, el("thead", {}, el("tr", {},
      ...["GW", "pred", "q05", "q25", "q75", "q95"].map(h => el("th", {}, h)))),
      el("tbody", {}, ...rows.map((r, i) => el("tr", {},
        el("td", {}, "GW" + r.gw), el("td", {}, fmtNum(ys[i])),
        el("td", {}, fmtNum(q05[i])), el("td", {}, fmtNum(q25[i])),
        el("td", {}, fmtNum(q75[i])), el("td", {}, fmtNum(q95[i])))))));

    // full CDFs from the t-digest quantiles, overlaid across the window
    holder.append(el("h2", {}, "Points CDF by GW"));
    const cdfNote = el("div", { class: "meta" }, "loading…");
    const cdfBox = el("div", { class: "chart" });
    const cdfTbl = el("table", {}, el("thead", {}, el("tr", {},
      ...["GW", "median", "P(≥5)", "P(≥10)", "blank"].map(h => el("th", {}, h)))), el("tbody"));
    holder.append(cdfNote, cdfBox, cdfTbl);
    (async () => {
      const per = [];
      for (const gwv of [...new Set(rows.map(r => r.gw))]) {
        try {
          per.push(await api.forecastCdf({ player_code: code, gw: gwv,
                                           at: "5,10,0.5" }));
        } catch { /* cold or outside window: remaining GWs still draw */ }
      }
      if (!per.length) { cdfNote.textContent = "CDF unavailable (cold forecast)"; return; }
      cdfNote.textContent = "cumulative probability of GW points";
      multiLineChart(cdfBox, per[0].xs,
        per.map((d) => ({ label: "GW" + d.gw, ys: d.cdf })),
        { yPercent: true, xlabel: "points" });
      for (const d of per) {
        const t = d.tails ?? {};
        cdfTbl.lastChild.append(el("tr", {},
          el("td", {}, "GW" + d.gw),
          el("td", {}, fmtNum(d.quantiles.q50)),
          el("td", {}, Math.round((t["5"]?.p_gt ?? 0) * 100) + "%"),
          el("td", {}, Math.round((t["10"]?.p_gt ?? 0) * 100) + "%"),
          el("td", {}, Math.round((t["0.5"]?.p_le ?? 0) * 100) + "%")));
      }
    })();
    return holder;
  } catch (e) {
    if (e.cold && retries > 0) {
      holder.replaceChildren(el("div", { class: "loading" },
        el("span", { class: "spinner" }), " computing forecast (one-time ~15s)…"));
      await new Promise(res => setTimeout(res, 5000));
      return fetchForecast(code, gw, retries - 1);
    }
    return el("div", { class: "err" },
      e.cold ? "Still computing forecast — try again shortly." : e.message);
  }
}

function select(options, onchange) {
  return el("select", { onchange: ev => onchange(ev.target.value) },
    ...options.map(o => el("option", { value: o }, o || "any position")));
}

function debounce(fn, ms) {
  let t;
  return (ev) => { clearTimeout(t); const e = { target: ev.target }; t = setTimeout(() => fn(e), ms); };
}

export { drawer as openPlayerDrawer };
