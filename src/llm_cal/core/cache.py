"""Disk cache for model-source responses.

Key design decisions (from /plan-eng-review Issue #2 + Issue #10 critical):

- Key = (source, model_id, commit_sha). Commit sha is included so a repo update
  invalidates cache automatically — prevents the critical regression of serving
  stale data after the upstream model updates.
- TTL = 7 days default. Even without a commit change, we force re-fetch weekly.
- `--refresh` flag sets `bypass=True` on `get()` — caller drives it.
- Store location: platformdirs user cache dir, subdirectory `llm-cal`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import diskcache
from platformdirs import user_cache_dir

from llm_cal.model_source.base import ModelArtifact, SiblingFile

_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


@dataclass(frozen=True)
class CacheKey:
    source: str
    model_id: str
    commit_sha: str | None

    def to_string(self) -> str:
        return f"{self.source}::{self.model_id}::{self.commit_sha or 'HEAD'}"


class ArtifactCache:
    """Persistent cache for ModelArtifact instances."""

    def __init__(
        self, cache_dir: str | Path | None = None, ttl_seconds: int = _DEFAULT_TTL_SECONDS
    ) -> None:
        if cache_dir is None:
            cache_dir = user_cache_dir("llm-cal", appauthor=False)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))
        self._ttl = ttl_seconds

    def get(self, key: CacheKey, bypass: bool = False) -> ModelArtifact | None:
        """Look up an artifact. `bypass=True` always returns None (used by --refresh).

        If `key.commit_sha` is None (no revision pinning), we never serve from cache
        because we can't prove freshness.
        """
        if bypass or key.commit_sha is None:
            return None
        raw = self._cache.get(key.to_string())
        if raw is None:
            return None
        return _deserialize_artifact(raw)

    def set(self, key: CacheKey, artifact: ModelArtifact) -> None:
        """Cache an artifact. No-op if commit_sha is None (can't guarantee freshness)."""
        if key.commit_sha is None:
            return
        self._cache.set(key.to_string(), _serialize_artifact(artifact), expire=self._ttl)

    def invalidate(self, key: CacheKey) -> bool:
        """Explicit invalidation, returns True if something was removed."""
        return bool(self._cache.delete(key.to_string()))

    def clear(self) -> None:
        """Wipe the whole cache — for tests and `llm-cal cache clear` (future)."""
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()


def _serialize_artifact(a: ModelArtifact) -> dict[str, Any]:
    return {
        "source": a.source,
        "model_id": a.model_id,
        "commit_sha": a.commit_sha,
        "config": a.config,
        "siblings": [asdict(s) if is_dataclass(s) else s for s in a.siblings],
    }


def _deserialize_artifact(raw: dict[str, Any]) -> ModelArtifact:
    return ModelArtifact(
        source=raw["source"],
        model_id=raw["model_id"],
        commit_sha=raw["commit_sha"],
        config=raw["config"],
        siblings=tuple(SiblingFile(**s) for s in raw["siblings"]),
    )
