"""Contract tests: in-process experiment cache (load + fit reuse)."""

from fpl.experiments import cache


def test_training_cache_reuses_and_counts():
    cache.reset_experiment_cache()
    calls = []

    def loader():
        calls.append(1)
        return {"a": 1}

    for _ in range(2):
        cache.cached_training(loader, processed="p", seasons=("s",),
                              features=("f",), categorical=("c",))
    assert len(calls) == 1                  # loader ran once
    counts = cache.cache_counts()
    assert counts["load_calls"] == 2 and counts["load_hits"] == 1


def test_distinct_configs_do_not_share():
    cache.reset_experiment_cache()
    seen = []

    def loader_a():
        seen.append("a")
        return {"a": 1}

    def loader_b():
        seen.append("b")
        return {"b": 2}

    cache.cached_training(loader_a, processed="p", seasons=("s",),
                          features=("f1",), categorical=None)
    cache.cached_training(loader_b, processed="p", seasons=("s",),
                          features=("f2",), categorical=None)
    assert seen == ["a", "b"]


def test_fit_cache_reuses_predict():
    cache.reset_experiment_cache()
    fits = []
    key = ("s", 30, "lgbm", (("seed", 42),), None, None)

    def fitter():
        fits.append(1)
        return lambda x: sum(x)

    cache.cached_fit(fitter, key=key)
    predict = cache.cached_fit(fitter, key=key)
    assert len(fits) == 1                     # fit ran once
    assert predict([1, 2, 3]) == 6            # same predict callable usable
    counts = cache.cache_counts()
    assert counts["fit_calls"] == 2 and counts["fit_hits"] == 1


def test_reset_clears_state():
    cache.cached_fit(lambda: lambda x: 0, key=("k",))
    cache.cached_training(lambda: {}, processed="p", seasons=("s",),
                          features=None, categorical=None)
    cache.reset_experiment_cache()
    counts = cache.cache_counts()
    assert counts == {k: 0 for k in counts}