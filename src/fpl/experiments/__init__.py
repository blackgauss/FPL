"""fpl.experiments: the hardened experiment + gym harness.

A declared experiment is a :class:`dict` (or YAML) describing one model
comparison with explicit temporal windows and an optional gym section. The
runner owns splitting, leakage, schema guards, cohorts, metrics, and writes a
machine-readable artifact with ``status: complete`` or ``status: failed``.

Key idea: an experiment is a *declared object that produces a reproducible
result artifact*; no printed result is valid unless the artifact exists.
"""

from __future__ import annotations

from fpl.experiments import cohorts, forecast, metrics, splits  # noqa: F401