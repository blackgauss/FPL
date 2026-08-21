# Experiments

Runnable model/variant experiments for the investigations in
[`analysis/Investigations.md`](../analysis/Investigations.md).

Each experiment lives in its own script/notebook and follows the repo
conventions: synthetic or tracked data inputs, deterministic seeds, and —
where model quality is being judged — evaluation through the **gym**
(`fpl.gym.Eval`, paired toggles), never a bare training-loss number.

Expected contents (as they land):

- position-model variants (global vs global + per-player correction),
- covariance / team-latent experiments (IID baseline vs covariance-aware),
- momentum feature ablations,
- matchup × position interaction checks.

Findings and conclusions are recorded in `analysis/Investigations.md`, and
only experiment *code + recorded outputs* live here.