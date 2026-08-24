# Profiling this project

Where does the time actually go? This repo now has per-scope and
per-function observability for its own pipeline (data prep, model training,
inference, the declared-run experiment harness, and candidate search).

Everything is stdlib `cProfile` + `pstats` behind `fpl.profiling` — no new
dependencies, and the output is agent/MCP-friendly JSON.

Latest findings and what's been fixed: [`docs/PerformanceDiagnostics.md`](PerformanceDiagnostics.md).

## Quickstart (read this first)

```bash
# Profile the reference pipeline (data prep / training / inference /
# declared-run harness / candidate search) and print a wall-clock table:
python scripts/profile_pipeline.py

# Outputs land in experiments/artifacts/profile/:
#   *.json  structured reports (agent-readable)  <- read these
#   *.prof  raw cProfile dumps                    <- pstats only
#
# The directory is gitignored: reports are REPRODUCIBLE via the script, not
# committed. Run the script to (re)generate them.
```

Then open any report:

```python
import json
d = json.load(open("experiments/artifacts/profile/run_experiment.json"))
print(f"wall: {d['wall_seconds']:.2f}s")
for r in d["function_rows"][:5]:          # sorted by cumulative time
    print(f"{r['cumtime_s']:8.3f}s  {r['function']} ({r['where']})")
```

## The two APIs

### 1. `fpl.profiling.time_phases(phases)` — coarse, labelled wall-clock

Time named stages, in order, and report seconds each. Use when you already
know the big blocks and just want the split (e.g. "is it data prep or
training?"):

```python
from fpl.profiling import time_phases
from fpl.model.train import load_training

by_season = time_phases({
    "data_prep/load": lambda: load_training("data/processed", seasons),
    "fit": lambda: fit(),
    "predict": lambda: predict(),
})
```

### 2. `fpl.profiling.profile_call(fn, name=..., out_dir=...)` — per-function

cProfile one callable, write `{name}.prof` and `{name}.json`, and return the
report dict. Use this to find the hot functions inside a stage:

```python
from fpl.profiling import profile_call, summarize_profile

report = profile_call(
    run_experiment, {"name": "probe", "seasons": seasons, "split": split,
                     "model": "lgbm"},
    name="run_experiment", out_dir="experiments/artifacts/profile",
)
print(summarize_profile(report))
```

## Report schema (so an agent can rely on it)

Each `*.json` has:

```json
{
  "profile": "run_experiment",
  "wall_seconds": 12.95,
  "function_rows": [
    {"function": "train", "where": "engine.py:108",
     "ncalls": 1, "tottime_s": 0.0, "cumtime_s": 6.5, "percall_cum_s": 6.5}
  ],
  "module_rows": [
    {"module": "basic.py", "tottime_s": 6.6, "cumtime_s": 7.1}
  ]
}
```

- `wall_seconds` — the elapsed call time (includes cProfile overhead).
- `function_rows` — per function, sorted by descending cumulative time.
  `where` is a short `filename:lineno`.
- `module_rows` — same numbers rolled up per source module.

`fpl.profiling.summarize_profile(report)` prints a human one-screen version.

## Project-specific workflows

- **"Why is my experiment run slow?"** — profile `run_experiment` (compare the
  JSON report); training will dominate. Check `training_fit.json` for LightGBM
  internals and `run_experiment.json/ranking_metrics` if cohort/ranking work
  is the culprit.
- **"Is data prep a bottleneck?"** — `data_prep_load.json`. Reference numbers
  put this ~0.15s (small).
- **"Compare two approaches fairly"** — profiles are not calibrated
  benchmarks; keep configurations identical and look at *relative* changes in
  `cumtime_s`, not the absolute wall time.
- **Agent/MCP** — read `experiments/artifacts/profile/*.json` directly (run
  `scripts/profile_pipeline.py` first if the files are missing); the schema
  above is stable. The whole profile directory is gitignored, so the JSON is
  always generated, never stale in the repo.

## Caveats / gotchas

- cProfile adds overhead (~20–50%), so absolute walls are higher than the
  raw run; use them directionally.
- Timings vary with machine load (training observed 13–35s across runs). Only
  compare within a run, or across runs on the same config/load.
- Profiling captures CPU time in the call tree; polars lazy `collect` appears
  as `collect`/`PyLazyFrame` rows — that's expected.
- This is a *known-good example* of the payoff: the first pipeline profile
  showed `pairwise_concordance` burning ~6s (O(n²) Python loop); vectorizing
  it cut the ranked/calibrated stage to ~0.29s in the harness.

## When NOT to use this

- For correctness/leakage — use `fpl.experiments.splits.validate_feature_leakage`.
- For model-quality comparison — use `python -m fpl.stages.experiments` and
  `scripts/compare_experiments.py`; profiling answers *where time goes*, not
  *which model is better*.
