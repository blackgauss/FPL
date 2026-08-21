"""Config loading: typed config from YAML, paths never hardcoded."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Seasons(BaseModel):
    available: list[str]
    default: str


class DataConfig(BaseModel):
    fpl_core_root: Path
    processed_dir: Path
    seasons: Seasons


def load_config(path: str | Path) -> DataConfig:
    """Load and validate the data YAML config."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return DataConfig(**raw)