"""Infra observability: where time actually goes.

Model observability tells us what the numbers mean; INFRA observability tells
us where the wall-clock goes (data prep vs training vs inference vs candidate
comparison). This module makes any workload profile-able with the stdlib:

- :func:`time_phases`   - labelled wall-clock per high-level stage.
- :func:`profile_call`  - cProfile a callable, persist `*.prof`, and emit a
                          structured per-function + per-module JSON report an
                          agent (or MCP client) can read directly.

Deliberately dependency-free and plug-and-play: wrap a step, or run
`scripts/profile_pipeline.py` for the reference pipeline.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from cProfile import Profile
from pathlib import Path
from pstats import Stats
from typing import Any


def time_phases(phases: dict[str, Callable[[], Any]]) -> dict[str, float]:
    """Wall-clock, in seconds, for each named phase.

    Phases run in dict insertion order; callers own any shared state. This is
    the coarse 'where does the time go' answer; profile_call gives functions.
    """
    timings: dict[str, float] = {}
    for name, fn in phases.items():
        start = time.perf_counter()
        fn()
        timings[name] = time.perf_counter() - start
    return timings


def _stats_rows(stats: Stats) -> tuple[list[dict], list[dict]]:
    """Flatten pstats into per-function and per-module rows.

    Function rows: ncalls/tottime/cumtime/percall/where (file:line + func).
    Module rows: aggregate tottime/cumtime per source module.
    """
    func_rows: list[dict] = []
    module_tot: dict[str, list[float]] = {}
    for key, (_cc, nc, tottime, cumtime, _callers) in stats.stats.items():
        filename, lineno, func = key
        ncalls = int(nc) if isinstance(nc, int) else int(sum(nc))
        func_rows.append({
            "function": f"{func}",
            "where": f"{filename.split('/')[-1]}:{lineno}",
            "ncalls": ncalls,
            "tottime_s": float(tottime),
            "cumtime_s": float(cumtime),
            "percall_cum_s": float(cumtime / ncalls) if ncalls else 0.0,
        })
        module = filename.split("/")[-1]
        bucket = module_tot.setdefault(module, [0.0, 0.0])
        bucket[0] += tottime
        bucket[1] += cumtime
    module_rows = [
        {"module": module, "tottime_s": tot[0], "cumtime_s": tot[1]}
        for module, tot in sorted(module_tot.items(), key=lambda kv: -kv[1][1])
    ]
    func_rows.sort(key=lambda r: -r["cumtime_s"])
    return func_rows, module_rows


def profile_call(
    fn: Callable[..., Any],
    *args: Any,
    out_dir: str | Path,
    name: str,
    **kwargs: Any,
) -> dict:
    """Profile one callable under cProfile and persist reports.

    Writes:
      <out_dir>/<name>.prof   - raw cProfile dump (loadable by pstats).
      <out_dir>/<name>.json   - structured report for agents: phases,
                                function rows (sorted by cumulative), and
                                module rollup.

    Returns the JSON report dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiler = Profile()
    start = time.perf_counter()
    profiler.enable()
    try:
        fn(*args, **kwargs)
    finally:
        profiler.disable()
    wall = time.perf_counter() - start

    prof_path = out / f"{name}.prof"
    profiler.dump_stats(str(prof_path))
    stats = Stats(str(prof_path))
    func_rows, module_rows = _stats_rows(stats)

    report = {
        "profile": name,
        "wall_seconds": wall,
        "function_rows": func_rows,
        "module_rows": module_rows,
    }
    (out / f"{name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def summarize_profile(report: dict) -> str:
    """A one-screen text summary of a profile report (for humans/CLI)."""
    rows = report["function_rows"][:12]
    lines = [
        f"profile: {report['profile']}  wall={report['wall_seconds']:.2f}s",
        "",
        f"{'cumtime_s':>10} {'tottime_s':>10} {'ncalls':>8}  function",
        "-" * 72,
    ]
    for row in rows:
        lines.append(
            f"{row['cumtime_s']:>10.3f} {row['tottime_s']:>10.3f} "
            f"{row['ncalls']:>8}  {row['function']} ({row['where']})")
    lines.append("")
    lines.append("top modules by cumulative time:")
    for module in report["module_rows"][:8]:
        lines.append(
            f"  {module['module']:<28} cum={module['cumtime_s']:>9.3f}s "
            f"tot={module['tottime_s']:>9.3f}s")
    return "\n".join(lines)