"""Render a side-by-side comparison table across experiment artifacts.

Usage:
    python scripts/compare_experiments.py A.json B.json
"""

from __future__ import annotations

import argparse

from fpl.experiments.artifacts import compare_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment artifacts")
    parser.add_argument("artifacts", nargs="+", help="result artifact JSON paths")
    args = parser.parse_args()
    print(compare_artifacts(*args.artifacts))


if __name__ == "__main__":
    main()