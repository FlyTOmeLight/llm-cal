"""Tests for `core/cache.py`.

CRITICAL regression: commit SHA mismatch must invalidate cache. Without this,
users get stale results after the upstream model updates — silent wrong answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_cal.core.cache import ArtifactCache, CacheKey
from llm_cal.model_source.base import ModelArtifact, SiblingFile


def _artifact(sha: str | None = "abc123") -> ModelArtifact:
    return ModelArtifact(
        source="huggingface",
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        commit_sha=sha,
        config={"model_type": "deepseek_v4", "hidden_size": 4096},
        siblings=(
            SiblingFile("model-00001-of-00002.safetensors", 100),
            SiblingFile("config.json", 10),
        ),
    )


@pytest.fixture
def cache(tmp_path: Path):
    c = ArtifactCache(cache_dir=tmp_path / "cache")
    yield c
    c.close()


class TestBasic:
    def test_set_then_get(self, cache):
        key = CacheKey("huggingface", "foo/bar", "abc")
        cache.set(key, _artifact("abc"))
        got = cache.get(key)
        assert got is not None
        assert got.model_id == "deepseek-ai/DeepSeek-V4-Flash"
        assert got.siblings[0].filename == "model-00001-of-00002.safetensors"

    def test_miss_returns_none(self, cache):
        key = CacheKey("huggingface", "foo/bar", "abc")
        assert cache.get(key) is None

    def test_bypass_flag_forces_miss(self, cache):
        key = CacheKey("huggingface", "foo/bar", "abc")
        cache.set(key, _artifact("abc"))
        assert cache.get(key, bypass=True) is None

    def test_invalidate_removes_entry(self, cache):
        key = CacheKey("huggingface", "foo/bar", "abc")
        cache.set(key, _artifact("abc"))
        assert cache.invalidate(key) is True
        assert cache.get(key) is None
        # Second invalidate returns False (nothing to remove)
        assert cache.invalidate(key) is False


class TestCommitShaInvalidation:
    """CRITICAL regression test.

    Scenario: user queries DeepSeek-V4-Flash today (sha=abc).
    Tomorrow the repo is updated (sha=def).
    User queries again — they MUST get the new data, not the cached old one.
    """

    def test_sha_mismatch_returns_none(self, cache):
        old_key = CacheKey("huggingface", "deepseek/V4", "abc")
        new_key = CacheKey("huggingface", "deepseek/V4", "def")
        cache.set(old_key, _artifact("abc"))

        # New query with different sha — cache key is different, so miss
        assert cache.get(new_key) is None

    def test_same_sha_hits(self, cache):
        key = CacheKey("huggingface", "deepseek/V4", "abc")
        cache.set(key, _artifact("abc"))
        assert cache.get(key) is not None


class TestNoShaMeansNoCache:
    """If we don't have a commit sha, we can't prove freshness → never cache.

    This is a conservative fallback for sources (like a future ModelScope REST
    adapter) that may not expose commit hashes.
    """

    def test_set_with_none_sha_is_noop(self, cache):
        key = CacheKey("huggingface", "foo/bar", None)
        cache.set(key, _artifact(sha=None))
        assert cache.get(key) is None

    def test_get_with_none_sha_returns_none_even_if_stored(self, cache):
        # If something was stored with a real sha and we query with None, no match
        real_key = CacheKey("huggingface", "foo/bar", "abc")
        none_key = CacheKey("huggingface", "foo/bar", None)
        cache.set(real_key, _artifact("abc"))
        assert cache.get(none_key) is None


class TestTTL:
    def test_explicit_zero_ttl_never_stores(self, tmp_path: Path):
        # diskcache expire=0 is "set and expire immediately"
        cache = ArtifactCache(cache_dir=tmp_path / "cache", ttl_seconds=0)
        key = CacheKey("huggingface", "foo/bar", "abc")
        cache.set(key, _artifact("abc"))
        # Expired immediately on read
        assert cache.get(key) is None
        cache.close()
