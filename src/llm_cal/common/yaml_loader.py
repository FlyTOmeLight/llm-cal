"""Pydantic-validated YAML loader.

Shared between engine_compat and hardware modules. Supports `lazy=True` param
(v0.1 does not implement lazy — signature reserved for v0.2 when matrix > 100).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class YamlLoadError(Exception):
    """YAML file could not be parsed or validated."""


def load_yaml(path: str | Path, schema: type[T], *, lazy: bool = False) -> T:
    """Load + validate a YAML file against a Pydantic schema.

    Args:
        path: YAML file to load.
        schema: Pydantic model the YAML is expected to conform to.
        lazy: Reserved for v0.2 (on-demand loading of large matrices). v0.1
              ignores this; document-scale data is small enough that eager
              loading is fine.
    """
    _ = lazy  # v0.1 behavior is always eager
    p = Path(path)
    if not p.exists():
        raise YamlLoadError(f"YAML file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise YamlLoadError(f"YAML parse error in {p}: {e}") from e

    if raw is None:
        raise YamlLoadError(f"YAML file {p} is empty")

    try:
        return schema.model_validate(raw)
    except ValidationError as e:
        raise YamlLoadError(f"Schema validation failed for {p}:\n{e}") from e
