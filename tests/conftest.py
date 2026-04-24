"""Shared fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_config() -> Callable[[str], dict[str, Any]]:
    """Load a config.json fixture by stem (e.g. "deepseek_v4_flash")."""

    def _load(stem: str) -> dict[str, Any]:
        path = FIXTURES_DIR / "configs" / f"{stem}.json"
        return json.loads(path.read_text())

    return _load
