"""Machine-readable experiment result artifacts.

An artifact is only as good as its completion guarantee: we write status
``complete`` only after every declared experiment ran; on an exception we
write status ``failed`` with the traceback so a result can never be reported
from a run that did not finish.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any


def write_artifact(
    path: Path,
    results: list[dict],
    *,
    metadata: dict,
    succeeded: bool = True,
    error: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "complete" if succeeded else "failed",
        "results": results,
        "metadata": metadata,
    }
    if error is not None:
        payload["error"] = error
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def write_failed_artifact(path: Path, exc: BaseException, *, metadata: dict) -> Path:
    return write_artifact(
        path, [], metadata=metadata, succeeded=False,
        error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def load_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flat_metric(result: dict) -> dict[str, float | int]:
    """Small scalar view of one experiment result (dvc/agent-metrics friendly)."""
    metrics = {m["cohort"]: m for m in result.get("metrics", [])}
    out: dict[str, float | int] = {
        "mae_all": metrics["all"]["mae"] if "all" in metrics else float("nan"),
        "top10_mae": metrics["top10"]["mae"] if "top10" in metrics else float("nan"),
    }
    for prefix, section in (("rank", result.get("ranking", {})),
                            ("cal", result.get("calibration", {}))):
        for key, value in section.items():
            if isinstance(value, (int, float)):
                out[f"{prefix}_{key}"] = value
    return out


def write_metrics_json(payload: dict, path: str | Path) -> Path:
    """Persist a flat metrics payload (JSON) for later access."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def write_flat_metrics(results: list[dict], path: str | Path) -> Path:
    """Flatten a list of experiment results into one metrics JSON."""
    payload = {"experiments": {r["name"]: flat_metric(r) for r in results}}
    return write_metrics_json(payload, path)


def compare_artifacts(*paths: str | Path) -> str:
    """Render a side-by-side text table of results across artifacts.

    Rows are per-experiment; columns are shared metric keys (cohort metrics
    flatten to ``<metric>@<cohort>``). Returns the table as a string.
    """
    artifacts = [load_artifact(p) for p in paths]
    rows: list[tuple[str, dict[str, float]]] = []
    keys: set[str] = set()
    for artifact in artifacts:
        for result in artifact.get("results", []):
            row: dict[str, float] = {}
            for metric in result.get("metrics", []):
                key = metric.get("cohort", "all")
                for field, value in metric.items():
                    if isinstance(value, (int, float)) and field not in (
                        "cohort", "n"):
                        row[f"{field}@{key}"] = float(value)
            for run in (result.get("gym") or {}).get("runs", []):
                for field, value in run.get("totals", {}).items():
                    if isinstance(value, (int, float)) and field != "settlement":
                        row[f"gym@{field}"] = float(value)
            for section, prefix in (("ranking", "rank@"), ("calibration", "cal@")):
                for field, value in result.get(section, {}).items():
                    if isinstance(value, (int, float)) and field != "n":
                        row[f"{prefix}{field}"] = float(value)
            rows.append((result.get("name", "?"), row))
            keys.update(row)
    header = ["experiment", *sorted(keys)]
    lines = [" | ".join(f"{h:>24}" for h in header)]
    lines.append("-+-".join("-" * 24 for _ in header))
    for name, row in rows:
        lines.append(" | ".join(
            f"{name:>24}" if col == "experiment" else
            f"{row.get(col, float('nan')):>24.4g}"
            for col in header))
    return "\n".join(lines)