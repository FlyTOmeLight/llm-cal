"""Evaluator — the single orchestration layer.

v0.1 partial implementation: composes model_source + detector + weight_analyzer
+ reconciler + kv_cache formula. Fleet planner and engine-compat matching land
in Week 4-5 (not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_cal.architecture.detector import detect
from llm_cal.architecture.formulas.kv_cache import compute_kv_cache_bytes
from llm_cal.architecture.formulas.weight import (
    estimate_total_params,
    predicted_bytes_under_quant,
)
from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.core.cache import ArtifactCache, CacheKey
from llm_cal.model_source.base import ModelArtifact, ModelSource
from llm_cal.model_source.huggingface import HuggingFaceSource
from llm_cal.output.labels import AnnotatedValue
from llm_cal.weight_analyzer import WeightReport, analyze
from llm_cal.weight_analyzer.reconciler import ReconciliationReport, reconcile


@dataclass(frozen=True)
class EvaluationReport:
    """Everything the evaluator produces for one model."""

    model_id: str
    source: str
    commit_sha: str | None
    gpu: str
    engine: str
    profile: ArchitectureProfile
    weight: WeightReport
    total_params_estimate: AnnotatedValue[int]
    reconciliation: ReconciliationReport
    kv_cache_by_context: dict[int, AnnotatedValue[int]] = field(default_factory=dict)


class Evaluator:
    """Orchestrates: model_source -> detect -> analyze -> reconcile -> KV cache.

    Fleet planning and command generation are Week 4-5 additions.
    """

    def __init__(
        self,
        source: ModelSource | None = None,
        cache: ArtifactCache | None = None,
    ) -> None:
        self._source = source or HuggingFaceSource()
        self._cache = cache or ArtifactCache()

    def evaluate(
        self,
        model_id: str,
        gpu: str,
        engine: str,
        gpu_count: int | None = None,
        context_length: int | None = None,
        refresh: bool = False,
    ) -> EvaluationReport:
        artifact = self._fetch(model_id, refresh=refresh)
        profile = detect(artifact.config)

        # Param count — prefer config field if provided, else estimate from shape
        total_params_est = estimate_total_params(profile)
        total_params = total_params_est.value

        weight = analyze(artifact.siblings, total_params=total_params if total_params > 0 else None)

        # Reconciler compares observed bytes vs each quantization's predicted bytes.
        # If our estimate is off, the reconciliation report surfaces the best match
        # available — honesty over false precision.
        reconciliation = reconcile(weight.total_bytes.value, total_params or 1)

        # KV cache at common context lengths + the model's max if known
        contexts_to_report = self._select_context_lengths(profile, context_length)
        kv_by_ctx = {
            ctx: compute_kv_cache_bytes(profile, seq_len=ctx, dtype_bytes=2)
            for ctx in contexts_to_report
        }

        # Keep the vs-gpu_poor sidelight visible
        _ = predicted_bytes_under_quant  # reserved for formatter

        return EvaluationReport(
            model_id=model_id,
            source=artifact.source,
            commit_sha=artifact.commit_sha,
            gpu=gpu,
            engine=engine,
            profile=profile,
            weight=weight,
            total_params_estimate=total_params_est,
            reconciliation=reconciliation,
            kv_cache_by_context=kv_by_ctx,
        )

    def _fetch(self, model_id: str, refresh: bool) -> ModelArtifact:
        # Probe fetch first to learn the commit sha, then try cache with that key.
        # This is a small extra network call but it's the honest way to use
        # sha-keyed caching — without it we'd cache under None and never hit.
        artifact = self._source.fetch(model_id)
        key = CacheKey(
            source=self._source.name,
            model_id=model_id,
            commit_sha=artifact.commit_sha,
        )
        cached = self._cache.get(key, bypass=refresh)
        if cached is not None:
            return cached
        self._cache.set(key, artifact)
        return artifact

    @staticmethod
    def _select_context_lengths(profile: ArchitectureProfile, override: int | None) -> list[int]:
        if override is not None:
            return [override]
        candidates = [4_096, 32_768, 131_072]
        max_pos = profile.position.max_position_embeddings if profile.position else None
        if max_pos and max_pos > 131_072:
            candidates.append(max_pos)
        # Keep only contexts that the model actually supports
        if max_pos:
            candidates = [c for c in candidates if c <= max_pos]
        return candidates
