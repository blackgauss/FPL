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