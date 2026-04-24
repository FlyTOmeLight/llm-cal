"""ModelSource ABC — HF and ModelScope implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SiblingFile:
    """One file in the model repo. `size` is bytes, or None if unknown."""

    filename: str
    size: int | None


@dataclass(frozen=True)
class ModelArtifact:
    """The raw material a ModelSource returns.

    We do NOT interpret anything here — interpretation lives in `architecture/`
    and `weight_analyzer/`. This is the thin "fetch" layer.
    """

    source: str  # "huggingface" | "modelscope"
    model_id: str
    commit_sha: str | None  # HF provides this; used as cache key component
    config: dict[str, Any]  # parsed config.json
    siblings: tuple[SiblingFile, ...]  # all files in the repo


class ModelNotFoundError(Exception):
    """Model id does not exist on this source."""


class AuthRequiredError(Exception):
    """Model is gated / private — user must set a token."""


class SourceUnavailableError(Exception):
    """Network error, timeout, rate limit, etc."""


class ModelSource(ABC):
    """Abstract interface for HF / ModelScope / future sources."""

    name: str  # subclasses override

    @abstractmethod
    def fetch(self, model_id: str) -> ModelArtifact:
        """Fetch config.json + siblings for the given model.

        Raises:
            ModelNotFoundError: 404.
            AuthRequiredError: 401/403 (gated/private).
            SourceUnavailableError: 429, 5xx, timeout, network down.
        """
