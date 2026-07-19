"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "mvp-cache"))
