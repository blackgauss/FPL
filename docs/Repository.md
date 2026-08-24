# Repository Structure

The repository has one canonical production path:

```text
raw FPL-Core data
        |
        v
fpl.stages.ingest -> fpl.stages.features -> fpl.stages.train
                                      |             |
                                      v             v
                              experiments       fpl.stages.fit_dist
                                      |             |
                                      v             v
                              ranking report    fpl search -> fpl gym
```

## Production Code

- `src/fpl/data` — data loading, contracts, ingestion, and features.
- `src/fpl/model` — training, inference, leakage checks, and evaluation.
- `src/fpl/team` — scoring, filtering, enumeration, simulation, and search.
- `src/fpl/experiments` — declared experiment execution and metrics.
- `src/fpl/stages` — executable DVC stage modules.
- `src/fpl/pipeline.py` — reusable orchestration, without CLI or file output concerns.
- `dvc.yaml` — dependency graph only.
- `params.yaml` — pipeline settings and invalidation inputs.

## Supporting Code

- `scripts/live_*.py` — operational live-data tools; not historical DVC stages.
- `scripts/profile_pipeline.py`, `scripts/train_time_analysis.py` — profiling tools.
- `scripts/build_notebook.py` — notebook generation utility.
- `scripts/compare_experiments.py` — experiment comparison utility.
- `scripts/*investigation*.py`, `experiments/ab_*.py` — research investigations.
- `analysis/` — generated reports and exploratory notebooks.

Supporting code may be promoted into the production path only after it has a
stable input/output contract and a DVC stage or package API.

## Canonical Commands

```bash
uv run dvc repro
uv run dvc dag
uv run dvc metrics show
uv run pytest -q
uv run ruff check src tests
```

Do not add a new production workflow as a standalone script. Add a package
stage under `src/fpl/stages`, declare it in `dvc.yaml`, and test its artifact
contract.
