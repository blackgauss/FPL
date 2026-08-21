"""Leakage-safe evaluation cohorts.

All cohorts are derived ONLY from pre-GW prediction and history. This keeps
the "top players we'd actually select" from being hidden inside all-player
metrics, without ever using the realized outcomes to define a cohort.
"""

from __future__ import annotations

import numpy as np


def cohort_masks(
    predictions: np.ndarray,
    source_gw: np.ndarray,
    positions: np.ndarray | None = None,
    *,
    top: float = 0.10,
) -> dict[str, np.ndarray]:
    """Return boolean masks (over the same rows) for standard cohorts.

    - all
    - top10: predicted top `top` within each source GW
    - top10_by_position: predicted top `top` within (source GW, position)

    `positions` must be aligned to the rows; when None the by-position cohort
    is empty.
    """
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(predictions), dtype=bool),
        "top10": np.zeros(len(predictions), dtype=bool),
        "top10_by_position": np.zeros(len(predictions), dtype=bool),
    }
    for gw in np.unique(source_gw):
        idx = np.flatnonzero(source_gw == gw)
        n = max(1, int(np.ceil(len(idx) * top)))
        masks["top10"][idx[np.argsort(predictions[idx])[-n:]]] = True
        if positions is not None:
            for pos in np.unique(positions[idx]):
                pix = idx[positions[idx] == pos]
                n_pos = max(1, int(np.ceil(len(pix) * top)))
                masks["top10_by_position"][
                    pix[np.argsort(predictions[pix])[-n_pos:]]] = True
    return masks


def group_mae(actual: np.ndarray, predicted: np.ndarray,
              masks: dict[str, np.ndarray]) -> dict[str, float]:
    """MAE per cohort; cohorts with no rows are omitted."""
    return {
        name: float(np.abs(actual[mask] - predicted[mask]).mean())
        for name, mask in masks.items() if np.any(mask)
    }