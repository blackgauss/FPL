# Model Evaluation

Model results are only comparable when the temporal protocol is explicit. The
feature store uses source gameweek rows to predict the following gameweek, so
all fit, calibration, and test windows refer to source GWs.

The current ranking stage uses one declared holdout in
`config/experiments_ranking.yaml`. The reusable `rolling_splits()` helper in
`fpl.experiments.splits` provides leakage-safe rolling windows for the next
evaluation pass.

```python
from fpl.experiments.splits import rolling_splits

windows = rolling_splits(
    first_test_start=10,
    last_test_end=38,
    test_window=4,
    step=4,
)
```

Before accepting a model change, report ranking, calibration, and downstream
team value across the same windows. Do not average across windows without also
reporting variation between them.
