"""Engine compatibility matrix loader + match function."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from llm_cal.common.yaml_loader import load_yaml

SupportLevel = Literal["full", "partial", "broken", "unverified"]
VerificationLevel = Literal["verified", "cited", "unverified"]


class EngineFlag(BaseModel):
    flag: str
    value: str | None = None
    note_en: str | None = None
    note_zh: str | None = None


class EngineSource(BaseModel):
    type: str  # release_notes | announcement | pr | tested
    url: str | None = None
    captured_date: str | None = None
    note_en: str | None = None
    note_zh: str | None = None
    # `tested` specific fields (may be absent on other types)
    tester: str | None = None
    date: str | None = None
    hardware: str | None = None


class EngineCompatEntry(BaseModel):
    engine: Literal["vllm", "sglang"]
    version_spec: str  # e.g. ">=0.19.0"
    matches_model_type: str
    support: SupportLevel
    verification_level: VerificationLevel
    required_flags: list[EngineFlag] = Field(default_factory=list)
    optional_flags: list[EngineFlag] = Field(default_factory=list)
    sources: list[EngineSource] = Field(default_factory=list)
    caveats_en: list[str] = Field(default_factory=list)
    caveats_zh: list[str] = Field(default_factory=list)


class EngineCompatMatrix(BaseModel):
    schema_version: int
    entries: list[EngineCompatEntry]


def _default_path() -> Path:
    return Path(str(files("llm_cal.engine_compat").joinpath("matrix.yaml")))


@lru_cache(maxsize=1)
def load_matrix(path: Path | None = None) -> EngineCompatMatrix:
    return load_yaml(path or _default_path(), EngineCompatMatrix)


def find_match(
    engine: str,
    model_type: str,
    version: str | None = None,
    matrix: EngineCompatMatrix | None = None,
) -> EngineCompatEntry | None:
    """Find the highest-version matching entry for (engine, model_type).

    If `version` is None, we return the broadest entry (any version matching
    model_type on the given engine). If `version` is given, we filter to entries
    whose version_spec covers it.
    """
    m = matrix or load_matrix()
    engine_norm = engine.lower().strip()
    model_type_norm = model_type.lower().strip()

    candidates = [
        e for e in m.entries if e.engine == engine_norm and e.matches_model_type == model_type_norm
    ]
    if not candidates:
        return None

    if version is None:
        # Return the entry with the "highest lower bound" as the most relevant
        return max(candidates, key=_lower_bound_key)

    try:
        v = Version(version)
    except InvalidVersion:
        return candidates[0]

    for entry in candidates:
        try:
            if v in SpecifierSet(entry.version_spec):
                return entry
        except InvalidSpecifier:
            continue
    return None


def _lower_bound_key(entry: EngineCompatEntry) -> Version:
    """Extract the lowest version a spec matches (approximate, used only for sort)."""
    try:
        spec = SpecifierSet(entry.version_spec)
    except InvalidSpecifier:
        return Version("0.0.0")
    for single in spec:
        if single.operator in (">=", "==", ">"):
            try:
                return Version(single.version)
            except InvalidVersion:
                continue
    return Version("0.0.0")
