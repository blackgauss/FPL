"""Headless runner: execute analysis/ModelEnsemblesInvestigation.ipynb.

Runs every code cell in order with one shared namespace and prints the
outputs (markdown headers + stdout) that the kernel would show — so results
are reproducible from the CLI without a notebook server.

    python scripts/run_investigation.py
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "analysis" / "ModelEnsemblesInvestigation.ipynb"


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {}
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            headings = [ln for ln in cell.source.splitlines() if ln.startswith("#")]
            print("\n" + "=" * 72)
            print("\n".join(headings) or cell.source)
            continue
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(cell.source, "<cell>", "exec"), ns)  # noqa: S102
        print("\n" + "-" * 72)
        print(buf.getvalue() if buf.getvalue().strip() else "(no output)")


if __name__ == "__main__":
    main()