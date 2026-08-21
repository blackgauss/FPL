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


def test_fit_key_is_scoped_to_data_dir():
    # same config on a different data directory must NOT share a fit
    base = dict(seasons=("s",), fit_gw_max=30, model="lgbm", params={"seed": 42},
                features=None, categorical=None)
    a = cache.fit_cache_key(processed="/data/a", **base)
    b = cache.fit_cache_key(processed="/data/b", **base)
    assert a != b
    cache.reset_experiment_cache()
    cache.cached_fit(lambda: lambda x: 1, key=a)
    second = cache.cached_fit(lambda: lambda x: 2, key=b)  # different store
    assert second([1]) == 2


def test_disk_cache_survives_memory_reset(tmp_path):
    # simulate a cross-process run: reset clears memory, disk provides the fit
    import numpy as np

    from fpl.model.experiment import REGISTRY

    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] + x[:, 2]).ravel()
    fits = []

    def fitter():
        fits.append(1)
        return REGISTRY["lgbm"]({"num_boost_round": 5, "num_leaves": 8,
                                 "min_child_samples": 5, "seed": 42,
                                 "verbosity": -1})(x, y, [])

    key = cache.fit_cache_key(processed="p", seasons=("s",), fit_gw_max=30,
                              model="lgbm", params={"seed": 42},
                              features=None, categorical=None)
    cache.reset_experiment_cache()
    predict1 = cache.cached_fit(fitter, key=key, disk_dir=tmp_path)
    pred = predict1(x[:5])
    assert len(fits) == 1
    assert (tmp_path / f"fit-{cache.cache_id(key)}.txt").exists()

    cache.reset_experiment_cache()          # new "process"
    predict2 = cache.cached_fit(fitter, key=key, disk_dir=tmp_path)
    assert len(fits) == 1                   # fitter NOT invoked (disk hit)
    assert np.allclose(predict2(x[:5]), pred)
    counts = cache.cache_counts()
    assert counts["fit_hits"] == 1