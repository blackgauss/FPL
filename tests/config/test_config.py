"""Black-box tests: load_config reads YAML into a typed config."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fpl.data.config import DataConfig, load_config


def write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


VALID = {
    "fpl_core_root": "external/fpl_core/data",
    "processed_dir": "data/processed",
    "seasons": {"available": ["2025-2026", "2026-2027"], "default": "2025-2026"},
}


def test_loads_valid_config(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID))
    assert isinstance(cfg, DataConfig)
    assert cfg.fpl_core_root == Path("external/fpl_core/data")
    assert cfg.processed_dir == Path("data/processed")
    assert cfg.seasons.available == ["2025-2026", "2026-2027"]
    assert cfg.seasons.default == "2025-2026"


def test_rejects_missing_sections(tmp_path):
    bad = {k: v for k, v in VALID.items() if k != "seasons"}
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, bad))


def test_rejects_wrong_types(tmp_path):
    bad = {**VALID, "seasons": {"available": "2025-2026", "default": "2025-2026"}}
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, bad))