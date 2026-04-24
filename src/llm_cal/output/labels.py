"""6-level label discipline — the soul of the tool.

Every number in the output must be wrapped in `AnnotatedValue` so users always know
where a value came from. Using `StrEnum` (not bare strings) means typos are caught by
mypy/ruff, not by users.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


class Label(StrEnum):
    VERIFIED = "verified"
    INFERRED = "inferred"
    ESTIMATED = "estimated"
    CITED = "cited"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"


T = TypeVar("T")


@dataclass(frozen=True)
class AnnotatedValue(Generic[T]):
    """A value paired with provenance metadata.

    Examples:
        AnnotatedValue(160_300_000_000, Label.VERIFIED, source="HF model_info.siblings")
        AnnotatedValue(4.52, Label.INFERRED, source="160.3 GB / 284B params")
        AnnotatedValue(2_600_000_000, Label.ESTIMATED,
                       source="compress_ratios=[0,0,4,128,...] at 128K ctx")
    """

    value: T
    label: Label
    source: str | None = None

    def render_tag(self) -> str:
        return f"[{self.label.value}]"
