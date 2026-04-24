"""Hardware database loader + lookup."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from llm_cal.common.yaml_loader import load_yaml


class GPUSpec(BaseModel):
    """One GPU entry in the hardware database."""

    id: str
    aliases: list[str] = Field(default_factory=list)
    memory_gb: int
    nvlink_bandwidth_gbps: int
    fp16_tflops: float
    fp8_support: bool
    fp4_support: bool
    notes_en: str | None = None
    notes_zh: str | None = None
    # Where the numeric specs came from. A URL to a vendor datasheet / trusted
    # benchmark, or a short note like "NVIDIA H100 datasheet 2024-Q3". Lets
    # users audit the source; honesty-over-convenience principle.
    spec_source: str | None = None

    def localized_notes(self, locale: Literal["en", "zh"]) -> str | None:
        if locale == "zh":
            return self.notes_zh or self.notes_en
        return self.notes_en or self.notes_zh


class GPUDatabase(BaseModel):
    schema_version: int
    gpus: list[GPUSpec]


class UnknownGPUError(Exception):
    """User asked for a GPU id we don't know."""


def _default_path() -> Path:
    """Locate the bundled gpu_database.yaml inside the installed package."""
    return Path(str(files("llm_cal.hardware").joinpath("gpu_database.yaml")))


@lru_cache(maxsize=1)
def load_database(path: Path | None = None) -> GPUDatabase:
    return load_yaml(path or _default_path(), GPUDatabase)


def lookup(gpu: str, db: GPUDatabase | None = None) -> GPUSpec:
    """Look up a GPU by id or alias. Case-insensitive."""
    database = db or load_database()
    target = gpu.strip().upper()
    for spec in database.gpus:
        if spec.id.upper() == target:
            return spec
        if any(alias.upper() == target for alias in spec.aliases):
            return spec
    # Helpful rejection
    if "X" in target and target.split("X")[-1].isdigit():
        raise UnknownGPUError(
            f"'{gpu}' looks like old 'H800x8' format. "
            f"Use `--gpu {target.split('X')[0]} --gpu-count {target.split('X')[-1]}` instead."
        )
    raise UnknownGPUError(f"Unknown GPU '{gpu}'. Known: {', '.join(s.id for s in database.gpus)}")
