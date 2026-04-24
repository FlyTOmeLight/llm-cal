"""Evaluator — the single orchestration layer.

v0.1 partial implementation: composes model_source + detector + weight_analyzer
+ reconciler + kv_cache + engine_compat + hardware. Fleet planner and command
generator land in Week 5 remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_cal.architecture.detector import detect
from llm_cal.architecture.formulas.kv_cache import compute_kv_cache_bytes
from llm_cal.architecture.formulas.weight import estimate_total_params
from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.core.cache import ArtifactCache, CacheKey
from llm_cal.engine_compat.loader import EngineCompatEntry, find_match
from llm_cal.hardware.loader import GPUSpec, UnknownGPUError, lookup
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
    gpu_spec: GPUSpec | None
    gpu_error: str | None  # message if gpu wasn't found
    engine: str
    profile: ArchitectureProfile
    weight: WeightReport
    total_params_estimate: AnnotatedValue[int]
    reconciliation: ReconciliationReport
    kv_cache_by_context: dict[int, AnnotatedValue[int]] = field(default_factory=dict)
    engine_match: EngineCompatEntry | None = None


class Evaluator:
    """Orchestrates: model_source -> detect -> analyze -> reconcile -> KV cache
    -> engine compat -> hardware lookup.

    Fleet planning and command generation are remaining Week 5 additions.
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

        total_params_est = estimate_total_params(profile)
        total_params = total_params_est.value

        weight = analyze(
            artifact.siblings,
            total_params=total_params if total_params > 0 else None,
        )
        reconciliation = reconcile(weight.total_bytes.value, total_params or 1)

        contexts_to_report = self._select_context_lengths(profile, context_length)
        kv_by_ctx = {
            ctx: compute_kv_cache_bytes(profile, seq_len=ctx, dtype_bytes=2)
            for ctx in contexts_to_report
        }

        # Engine compatibility — match by model_type alone (v0.1). Version
        # filtering can be added via a future --engine-version flag.
        engine_match = find_match(engine=engine, model_type=profile.model_type)

        # Hardware lookup — never raises out to CLI, we embed the error message
        # so the user sees a partial report instead of aborting.
        gpu_spec: GPUSpec | None = None
        gpu_error: str | None = None
        try:
            gpu_spec = lookup(gpu)
        except UnknownGPUError as e:
            gpu_error = str(e)

        return EvaluationReport(
            model_id=model_id,
            source=artifact.source,
            commit_sha=artifact.commit_sha,
            gpu=gpu,
            gpu_spec=gpu_spec,
            gpu_error=gpu_error,
            engine=engine,
            profile=profile,
            weight=weight,
            total_params_estimate=total_params_est,
            reconciliation=reconciliation,
            kv_cache_by_context=kv_by_ctx,
            engine_match=engine_match,
        )

    def _fetch(self, model_id: str, refresh: bool) -> ModelArtifact:
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
        if max_pos:
            candidates = [c for c in candidates if c <= max_pos]
        return candidates
