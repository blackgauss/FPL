"""Web test fixtures: populated and empty synthetic stores."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.web.support import _client, _write_root


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return _client(_write_root(tmp_path))


@pytest.fixture()
def empty_client(tmp_path: Path) -> TestClient:
    root = tmp_path / "empty"
    root.mkdir()
    return _client(root)
