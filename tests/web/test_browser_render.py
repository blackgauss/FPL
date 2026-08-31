"""Real-browser paint gate for charts.js.

The DOM-stub smoke test cannot see *where* uPlot's nodes land: vendored
uPlot 1.6 ships a build that constructs u.root but never inserts it into
opts.target, so charts rendered invisibly while every stub stayed green.
This test drives charts.js through headless Chrome with the *vendored*
uPlot and asserts actual canvas ink — the only oracle that proves a
chart is visible.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "fpl" / "web" / "static"
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium",
            "chromium-browser")

PROBE = """<!doctype html><html><head>
<link rel=stylesheet href=uplot.min.css><script src=uplot.min.js></script>
</head><body style="background:#fff">
<div id=b1 class=chart style="width:460px"></div>
<div id=b2 class=chart style="width:460px"></div>
<div id=b3 class=chart style="width:460px"></div>
<div id=b4 style="width:200px"></div>
<script type=module>
window.addEventListener("error", (e) => {
  window.__perr = "PAGEERR " + e.message + " @" + (e.lineno || "?");
});
import { bandChart, cdfChart, multiLineChart, sparkBars } from "/js/charts.js";
const gss = (x0, y0) => Array.from({length: 4},
  (_, i) => Math.round(y0 * (1 - Math.abs(Math.sin(i + x0)))) + 1);
bandChart(b1, [1, 2, 3, 4], gss(1, 8), gss(1, 4), gss(1, 12));
const Q = [.01, .05, .1, .25, .5, .75, .9, .95, .99];
const vals = [.2, .8, 1.6, 3.1, 5.1, 7.4, 10.2, 12.5, 16.0];
const prob = (th) => {
  const p = vals.map((v, i) => [v, Q[i]]).sort((a, b) => a[0] - b[0]);
  if (th <= p[0][0]) return 0;
  if (th >= p.at(-1)[0]) return 1;
  for (let i = 0; i < p.length - 1; i++) {
    const [x0, q0] = p[i], [x1, q1] = p[i + 1];
    if (th >= x0 && th <= x1)
      return x0 === x1 ? q1 : q0 + (th - x0) / (x1 - x0) * (q1 - q0);
  }
  return 1;
};
const top = Math.max(...vals) * 1.05, n = 80, xs = [], cdf = [];
for (let i = 0; i < n; i++) { xs.push(i * top / (n - 1)); cdf.push(prob(i * top / (n - 1))); }
cdfChart(b2, xs, cdf);
multiLineChart(b3, xs, [{ label: "GW1", ys: cdf },
  { label: "GW2", ys: cdf.map((v) => Math.min(1, v + 0.2)) }], { yPercent: true });
sparkBars(b4, [3, 5, 1, 8]);
setTimeout(() => {
  const ink = (el) => {
    let t = 0;
    el.querySelectorAll("canvas").forEach((c) => {
      if (!c.width) return;
      const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 0) t++;
    });
    return t;
  };
  const bars = Array.from(b4.children)
    .filter((n) => /height:\s*[1-9]/.test(n.getAttribute("style") || "")).length;
  document.title = window.__perr
    || ("INK band=" + ink(b1) + " cdf=" + ink(b2)
        + " multi=" + ink(b3) + " bars=" + bars);
}, 400);
</script></body></html>
"""


def _find_browser() -> str | None:
    exe = next((b for b in map(shutil.which, BROWSERS) if b), None)
    return exe


@pytest.mark.skipif(_find_browser() is None, reason="no chrome binary")
def test_charts_paint_real_pixels(tmp_path: Path) -> None:
    docroot = tmp_path / "public"
    shutil.copytree(STATIC, docroot, ignore=shutil.ignore_patterns(
        "render_smoke.mjs"))
    (docroot / "paint_probe.html").write_text(PROBE)

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(SimpleHTTPRequestHandler, directory=str(docroot)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/paint_probe.html"
    try:
        r = subprocess.run(
            [_find_browser(), "--headless=new", "--disable-gpu",
             "--no-sandbox", "--virtual-time-budget=15000",
             "--dump-dom", url],
            capture_output=True, text=True, timeout=90)
    finally:
        server.shutdown()

    assert "INK band=" in r.stdout, (
        f"probe script never ran (chrome exit {r.returncode}):\n"
        f"{r.stderr[-800:]}")
    page_err = re.search(r"<title>PAGEERR ([^<]*)</title>", r.stdout)
    assert page_err is None, f"page threw during chart draw: {page_err.group(1)}"
    # scope to <title>: the dumped page source also contains the tokens
    m = re.search(r"<title>INK band=(\d+) cdf=(\d+) multi=(\d+) bars=(\d+)</title>",
                  r.stdout)
    assert m, f"paint probe never reported ink: {r.stdout[:400]}"
    # a painted curve is thousands of px; a detached/blank chart is 0
    for name, got in zip(("band", "cdf", "multi"), m.groups()[:3], strict=True):
        assert int(got) > 500, f"{name} chart painted {got} px (invisible?)"
    assert int(m.group(4)) == 4, "sparkBars rendered no bars"
