"""Contract tests: artifact writer/reader/compare."""


from fpl.experiments.artifacts import (
    compare_artifacts,
    load_artifact,
    write_artifact,
    write_failed_artifact,
)


def test_complete_artifact_roundtrip(tmp_path):
    path = write_artifact(
        tmp_path / "r.json",
        [{"name": "a", "metrics": [{"cohort": "all", "mae": 1.0, "n": 5}]}],
        metadata={"git_sha": "x"},
    )
    payload = load_artifact(path)
    assert payload["status"] == "complete"
    assert payload["results"][0]["name"] == "a"
    assert payload["metadata"] == {"git_sha": "x"}


def test_failed_artifact_preserves_traceback(tmp_path):
    try:
        raise ValueError("boom")
    except ValueError as exc:
        path = write_failed_artifact(tmp_path / "f.json", exc, metadata={})
    payload = load_artifact(path)
    assert payload["status"] == "failed"
    assert "boom" in payload["error"]


def test_compare_renders_table(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_artifact(a, [{"name": "x", "metrics": [{"cohort": "all", "mae": 1.0}]}],
                   metadata={})
    write_artifact(b, [{"name": "y", "metrics": [{"cohort": "all", "mae": 2.0}]}],
                   metadata={})
    table = compare_artifacts(a, b)
    assert "x" in table and "y" in table
    # header row has metric keys, rows have values
    assert "mae@all" in table