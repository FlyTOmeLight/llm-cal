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
from llm_cal.command_generator.sglang import generate_sglang_command
from llm_cal.command_generator.vllm import generate_vllm_command
from llm_cal.core.cache import ArtifactCache, CacheKey
from llm_cal.engine_compat.loader import EngineCompatEntry, find_match
from llm_cal.fleet.planner import FleetRecommendation, plan
from llm_cal.hardware.loader import GPUSpec, UnknownGPUError, lookup
from llm_cal.performance.compute import (
    DEFAULT_DECODE_BW_UTILIZATION,
    DEFAULT_PREFILL_UTILIZATION,
    DecodeEstimate,
    PrefillEstimate,
    estimate_decode,
    estimate_prefill,
)
from llm_cal.performance.concurrency import ConcurrencyAnalysis, analyze as analyze_concurrency
from llm_cal.model_source.base import ModelArtifact, ModelSource
from llm_cal.model_source.huggingface import HuggingFaceSource
from llm_cal.output.labels import AnnotatedValue
from llm_cal.weight_analyzer import WeightReport, analyze
from llm_cal.weight_analyzer.reconciler import ReconciliationReport, reconcile

_KV_REFERENCE_CTX = 131_072  # matches fleet.planner's _REFERENCE_CTX_TOKENS


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
    fleet: FleetRecommendation | None = None
    generated_command: str | None = None
    # Performance analysis — filled when user passes SLA args (or defaults).
    prefill: PrefillEstimate | None = None
    decode: DecodeEstimate | None = None
    concurrency: ConcurrencyAnalysis | None = None
    perf_input_tokens: int | None = None
    perf_output_tokens: int | None = None
    perf_target_tokens_per_sec: float | None = None


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
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        target_tokens_per_sec: float | None = None,
        prefill_utilization: float = DEFAULT_PREFILL_UTILIZATION,
        decode_bw_utilization: float = DEFAULT_DECODE_BW_UTILIZATION,
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

        # Fleet planning — only if we have a known GPU. The planner's reference
        # context is 128K; derive KV bytes there (computing fresh in case the
        # user chose a non-overlapping context_length override).
        fleet: FleetRecommendation | None = None
        generated_command: str | None = None
        if gpu_spec is not None and weight.total_bytes.value > 0:
            kv_ref = compute_kv_cache_bytes(profile, _KV_REFERENCE_CTX, dtype_bytes=2)
            kv_by_context_bytes = {ctx: av.value for ctx, av in kv_by_ctx.items() if av.value > 0}
            fleet = plan(
                profile=profile,
                weight_bytes=weight.total_bytes.value,
                kv_bytes_per_request_at_ref=max(1, kv_ref.value),
                gpu=gpu_spec,
                forced_gpu_count=gpu_count,
                kv_bytes_by_context=kv_by_context_bytes,
            )
            # Pick the gpu_count to emit the command for: user's forced value,
            # else the best_tier's recommendation.
            chosen_count = gpu_count or next(
                (o.gpu_count for o in fleet.options if o.tier == fleet.best_tier),
                fleet.options[0].gpu_count,
            )
            generated_command = _generate_command(
                engine=engine,
                model_id=model_id,
                profile=profile,
                tp=chosen_count,
                entry=engine_match,
                max_model_len=context_length,
            )

        # Performance analysis — runs whenever we have hardware + fleet.
        prefill_est: PrefillEstimate | None = None
        decode_est: DecodeEstimate | None = None
        concurrency_est: ConcurrencyAnalysis | None = None
        if gpu_spec is not None and fleet is not None and total_params > 0:
            # Pick the fleet tier we're analyzing (user's forced count or best tier).
            chosen = gpu_count or next(
                (o.gpu_count for o in fleet.options if o.tier == fleet.best_tier),
                fleet.options[0].gpu_count,
            )
            # Resolve performance defaults when user didn't specify.
            eff_input = input_tokens or 2000
            eff_output = output_tokens or 512
            eff_target = target_tokens_per_sec or 30.0

            prefill_est = estimate_prefill(
                profile=profile,
                total_params=total_params,
                gpu=gpu_spec,
                num_gpus=chosen,
                input_tokens=eff_input,
                utilization=prefill_utilization,
            )
            # MoE active ratio: active/total = (shared + experts_per_tok) / (shared + routed)
            moe_active_ratio: float | None = None
            if profile.moe is not None:
                active_experts = profile.moe.num_experts_per_tok + profile.moe.num_shared_experts
                total_experts = profile.moe.num_routed_experts + profile.moe.num_shared_experts
                if total_experts > 0:
                    moe_active_ratio = active_experts / total_experts
            decode_est = estimate_decode(
                profile=profile,
                total_weight_bytes=weight.total_bytes.value,
                gpu=gpu_spec,
                num_gpus=chosen,
                bw_utilization=decode_bw_utilization,
                moe_active_params_ratio=moe_active_ratio,
            )
            # Compute cluster headroom at the chosen tier + KV per request at the
            # *longest* surveyed context (most conservative).
            chosen_option = next(
                (o for o in fleet.options if o.gpu_count == chosen),
                fleet.options[-1],
            )
            headroom_per_gpu = (
                chosen_option.usable_bytes_per_gpu - chosen_option.weight_bytes_per_gpu
            )
            cluster_headroom = max(0, headroom_per_gpu) * chosen
            # Reference context for the L bound: match K's headroom context (128K
            # if model supports it, else max).
            kv_ref_ctx = 131_072 if 131_072 in kv_by_ctx else max(kv_by_ctx.keys())
            kv_ref_bytes: int = kv_by_ctx[kv_ref_ctx].value
            # Apply TP-aware sharding (same rule fleet planner uses).
            from llm_cal.fleet.planner import _kv_shards

            shards = _kv_shards(profile, chosen)
            kv_ref_per_gpu = max(1, kv_ref_bytes // shards)
            # Request KV lives per-GPU; under replication, it's the same value on all.
            # We compare cluster headroom against per-GPU KV (each request consumes
            # per-GPU KV on every rank simultaneously).
            # To convert to "how many requests fit", we divide *per-GPU* headroom
            # by *per-GPU* KV.
            headroom_per_req_view = max(0, headroom_per_gpu)
            concurrency_est = analyze_concurrency(
                cluster_headroom_bytes=headroom_per_req_view,
                kv_bytes_per_request=kv_ref_per_gpu,
                decode=decode_est,
                target_tokens_per_sec=eff_target,
            )

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
            fleet=fleet,
            generated_command=generated_command,
            prefill=prefill_est,
            decode=decode_est,
            concurrency=concurrency_est,
            perf_input_tokens=input_tokens or 2000 if fleet else None,
            perf_output_tokens=output_tokens or 512 if fleet else None,
            perf_target_tokens_per_sec=target_tokens_per_sec or 30.0 if fleet else None,
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


def _generate_command(
    engine: str,
    model_id: str,
    profile: ArchitectureProfile,
    tp: int,
    entry: EngineCompatEntry | None,
    max_model_len: int | None,
) -> str:
    engine_norm = engine.lower().strip()
    if engine_norm == "sglang":
        return generate_sglang_command(model_id, profile, tp, entry, max_model_len=max_model_len)
    return generate_vllm_command(model_id, profile, tp, entry, max_model_len=max_model_len)
