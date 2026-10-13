"""Full derivation traces for each non-trivial number in the report.

This module is only invoked when the user passes `--explain`. It doesn't
recompute anything — it reads the values that the main evaluator already
produced and wraps them in a formatted explanation with formula, inputs,
step-by-step computation, and primary source citation.

Design rationale: the tool's core promise is deterministic, auditable
output. `--explain` makes that auditability human-readable. A user can:
  1. Read the explanation themselves
  2. Paste it into an LLM and ask "does this math check out?"
  3. Cross-reference docs/methodology.md for the primary source
All three preserve determinism — the LLM is the user's tool, not ours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from llm_cal.architecture.formulas.kv_cache import (
    _average_csa_hca_ratio,
    _per_layer_per_token_bytes,
)
from llm_cal.core.evaluator import EvaluationReport


@dataclass(frozen=True)
class ExplainInput:
    """One input variable to a formula."""

    name: str
    value: str  # pre-formatted for display
    label: str  # e.g. "[verified]", "[estimated]"
    note: str = ""  # optional disambiguation


@dataclass(frozen=True)
class ExplainEntry:
    """A full derivation trace for one output number."""

    heading: str  # localized section title, e.g. "KV cache @ 128K"
    formula: str  # the formula, literally
    inputs: list[ExplainInput] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)  # step-by-step computation
    result: str = ""  # final formatted answer with label
    source: str = ""  # primary source citation
    methodology_anchor: str = ""  # anchor in docs/methodology.md, e.g. "#prefill-latency"


def build(report: EvaluationReport) -> list[ExplainEntry]:
    """Produce explanation entries in the order they appear in the main report."""
    entries: list[ExplainEntry] = []

    _weight_bytes(report, entries)
    _params(report, entries)
    _quantization(report, entries)
    _kv_cache_contexts(report, entries)
    _fleet_tiers(report, entries)
    _prefill(report, entries)
    _decode(report, entries)
    _concurrency(report, entries)

    return entries


# ======================================================================
# Weight
# ======================================================================


def _weight_bytes(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    w = report.weight.total_bytes
    entries.append(
        ExplainEntry(
            heading="Weight bytes (safetensors file sum)",
            formula="sum(sibling.size for sibling in HF model_info(files_metadata=True).siblings if sibling.endswith('.safetensors'))",
            inputs=[
                ExplainInput(
                    name="HF model_info API",
                    value=f"source={report.source}, sha={report.commit_sha or 'HEAD'}",
                    label=f"[{w.label.value}]",
                ),
            ],
            steps=[
                f"Raw value from API = {w.value:,} bytes",
                f"= {w.value / 1e9:.2f} GB",
            ],
            result=f"{w.value:,} bytes [{w.label.value}]",
            source=w.source or "HF siblings API",
            methodology_anchor="#weight-bytes",
        )
    )


def _params(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    av = report.total_params_estimate
    entries.append(
        ExplainEntry(
            heading="Parameter estimate (#params-estimate)",
            formula=av.source or "unknown",
            steps=[f"result = {av.value:,} parameters"],
            result=f"{av.value:,} [{av.label.value}]",
            source="Architecture shape estimate; includes all resident MoE experts",
            methodology_anchor="#weight-by-assumption",
        )
    )


def _quantization(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    r = report.reconciliation
    if not r.candidates:
        return
    selected = next((c for c in r.candidates if c.scheme == r.best.value), None)
    candidates = [
        f"{c.scheme}: {c.predicted_bytes:,} predicted bytes, {c.relative_error:.1%} error"
        for c in r.candidates[:6]
    ]
    entries.append(
        ExplainEntry(
            heading="Quantization scheme (reconciliation)",
            formula="predicted_bytes = scheme.bpp × total_params; select using available evidence or nearest bytes anchor",
            inputs=[
                ExplainInput(
                    "observed_bytes",
                    f"{r.observed_bytes:,}",
                    f"[{report.weight.total_bytes.label.value}]",
                ),
                ExplainInput(
                    "total_params", f"{r.total_params:,}", "[estimated]", note="#params-estimate"
                ),
            ],
            steps=[
                *candidates,
                f"Selection: {r.best.value}"
                + (f" ({selected.relative_error:.1%} error)" if selected else ""),
            ],
            result=f"{r.best.value} [{r.best.label.value}]",
            source=r.best.source or "No authoritative quantization evidence",
            methodology_anchor="#quantization-scheme",
        )
    )


# ======================================================================
# KV cache
# ======================================================================


def _kv_cache_contexts(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    profile = report.profile
    attn = profile.attention
    if attn is None:
        return
    for ctx, av in report.kv_cache_by_context.items():
        if av.value == 0:
            continue
        effective_seq = ctx
        steps = []
        if profile.sliding_window and attn.variant not in ("CSA_HCA", "NSA"):
            effective_seq = min(ctx, profile.sliding_window)
            steps.append(
                f"effective_seq = min({ctx:,}, sliding_window={profile.sliding_window:,}) = {effective_seq:,}"
            )
        per_token = _per_layer_per_token_bytes(attn, 2)
        baseline = per_token * effective_seq * profile.num_hidden_layers
        steps.extend(
            [
                f"per_token_per_layer = {per_token:,} bytes",
                f"baseline = {per_token:,} × {effective_seq:,} × {profile.num_hidden_layers} = {baseline:,} bytes",
            ]
        )
        if attn.variant == "CSA_HCA" and attn.compress_ratios:
            ratio = _average_csa_hca_ratio(attn.compress_ratios)
            steps.append(
                f"compress_keep_fraction = {ratio:.8f}; int(baseline × fraction) = {av.value:,}"
            )
        if attn.variant == "NSA" and attn.nsa_topk and attn.nsa_topk > 0:
            ratio = min(1.0, attn.nsa_topk / effective_seq)
            steps.append(
                f"NSA keep_fraction = min(1, {attn.nsa_topk} / {effective_seq}) = {ratio:.8f}; int(baseline × fraction) = {av.value:,}"
            )
        steps.append(f"result = {av.value:,} bytes")
        entries.append(
            ExplainEntry(
                heading=f"KV cache @ {_fmt_ctx(ctx)} context",
                formula=av.source or "KV cache estimate",
                inputs=[
                    ExplainInput("seq_len", f"{ctx:,}", "[inferred]"),
                    ExplainInput("dtype_bytes", "2", "[estimated]", note="BF16/FP16 assumption"),
                    ExplainInput("num_layers", str(profile.num_hidden_layers), "[verified]"),
                ],
                steps=steps,
                result=f"{av.value:,} bytes = {av.value / 1e9:.2f} GB [{av.label.value}]",
                source=av.source or "Architecture KV estimate",
                methodology_anchor="#kv-cache-per-request",
            )
        )


# ======================================================================
# Fleet tiers
# ======================================================================


def _fleet_tiers(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    if report.fleet is None or report.gpu_spec is None:
        return

    # One explain block per tier (min / dev / prod)
    for opt in report.fleet.options:
        tier_label = opt.tier
        headroom = opt.usable_bytes_per_gpu - opt.weight_bytes_per_gpu
        steps = [
            f"per-GPU HBM usable (@ 90% util) = {opt.usable_bytes_per_gpu:,} bytes",
            f"weight per GPU = total_weight / TP_size = "
            f"{report.weight.total_bytes.value:,} / {opt.gpu_count} = "
            f"{opt.weight_bytes_per_gpu:,} bytes",
            f"headroom per GPU = usable - weight = {headroom:,} bytes ({headroom / 1e9:.2f} GB)",
        ]
        fit_criterion = {"min": 1, "dev": 8, "prod": 16}.get(tier_label, 1)
        steps.append(f"tier criterion: headroom ≥ {fit_criterion} × per_GPU_KV_at_reference_ctx")
        steps.append(
            f"selected TP count: {opt.gpu_count}; valid candidates: {list(report.fleet.valid_tp_sizes)}"
        )
        if not opt.fits:
            steps.append(
                f"NOTE: does not fit the criterion — the chosen {opt.gpu_count} "
                "is the best available."
            )

        entries.append(
            ExplainEntry(
                heading=f"Fleet tier: {tier_label} ({opt.gpu_count} GPUs)",
                formula=(
                    "smallest TP in valid_set where "
                    "weight_per_gpu + concurrent × kv_per_request ≤ usable_per_gpu"
                ),
                inputs=[
                    ExplainInput(
                        "total_weight_bytes",
                        f"{report.weight.total_bytes.value:,}",
                        "[verified]",
                    ),
                    ExplainInput(
                        "valid_TP_sizes",
                        str(list(report.fleet.valid_tp_sizes)),
                        "[estimated]",
                        note="divisors of num_attention_heads capped at 8 (single node)",
                    ),
                    ExplainInput(
                        "GPU memory_gb",
                        f"{report.gpu_spec.memory_gb} GB",
                        "[verified]",
                    ),
                ],
                steps=steps,
                result=f"{opt.gpu_count} GPUs, fit={opt.fits}",
                source="vLLM --gpu-memory-utilization 0.9 convention; TP divisibility required by vLLM/SGLang",
                methodology_anchor="#tp-aware-kv-sharding",
            )
        )


# ======================================================================
# Prefill
# ======================================================================


def _prefill(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    if (
        report.prefill is None
        or report.gpu_spec is None
        or report.fleet is None
        or report.perf_input_tokens is None
    ):
        return
    p = report.prefill
    # Figure out chosen GPU count from the fleet
    chosen = next(
        (o.gpu_count for o in report.fleet.options if o.tier == report.fleet.best_tier),
        report.fleet.options[0].gpu_count,
    )
    entries.append(
        ExplainEntry(
            heading="Prefill latency (single request)",
            formula=(
                "FLOPs = 2 × params × input_tokens\n"
                "effective_TFLOPS = peak_fp16_TFLOPS × num_gpus × utilization\n"
                "latency_ms = (FLOPs / (effective_TFLOPS × 1e12)) × 1000"
            ),
            inputs=[
                ExplainInput(
                    "params",
                    f"{report.total_params_estimate.value:,}",
                    "[estimated]",
                    note="from architecture formula (see weight.py)",
                ),
                ExplainInput("input_tokens", f"{report.perf_input_tokens:,}", "[user-set]"),
                ExplainInput(
                    "peak_fp16_TFLOPS",
                    f"{report.gpu_spec.fp16_tflops}",
                    "[verified]",
                    note=f"from GPU database, {report.gpu_spec.id} spec",
                ),
                ExplainInput("num_gpus", f"{chosen}", "[estimated]"),
                ExplainInput(
                    "utilization",
                    f"{p.utilization:.2f}",
                    "[user-set]",
                    note="empirical MFU, default 0.40 — override with --prefill-util",
                ),
            ],
            steps=[
                f"FLOPs = 2 × {report.total_params_estimate.value:,} × "
                f"{report.perf_input_tokens:,} = {p.total_flops.value:.3e}",
                f"effective_TFLOPS = {report.gpu_spec.fp16_tflops} × {chosen} × "
                f"{p.utilization:.2f} = {p.peak_effective_tflops.value:.1f}",
                f"latency = {p.total_flops.value:.3e} / "
                f"({p.peak_effective_tflops.value:.1f} × 1e12) × 1000 = "
                f"{p.latency_ms.value:.1f} ms",
            ],
            result=f"{p.latency_ms.value:.1f} ms [{p.latency_ms.label.value}]",
            source="Kaplan et al. 2020 'Scaling Laws for Neural Language Models' (arxiv.org/abs/2001.08361)",
            methodology_anchor="#prefill-latency",
        )
    )


# ======================================================================
# Decode
# ======================================================================


def _decode(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    if report.decode is None or report.gpu_spec is None or report.fleet is None:
        return
    d = report.decode
    bw = report.gpu_spec.memory_bandwidth_gbps or 0
    chosen = next(
        (o.gpu_count for o in report.fleet.options if o.tier == report.fleet.best_tier),
        report.fleet.options[0].gpu_count,
    )
    weight_per_gpu = d.active_weight_bytes_per_gpu.value
    if weight_per_gpu <= 0:
        return
    effective_bw_gbs = bw * d.bw_utilization
    steps = [
        f"weight_per_gpu = {report.weight.total_bytes.value:,} / {chosen} = "
        f"{weight_per_gpu:,} bytes ({weight_per_gpu / 1e9:.2f} GB)",
        f"effective_bw = {bw} × {d.bw_utilization:.2f} = {effective_bw_gbs:.0f} GB/s",
        f"rank_tok_per_sec = effective_bw / weight_per_gpu = "
        f"{effective_bw_gbs * 1e9 / weight_per_gpu:.1f} tok/s",
        f"TP_group_tok_per_sec = rank_tok_per_sec × "
        f"{d.cluster_comm_efficiency:.2f} = {d.cluster_tokens_per_sec.value:.1f} tok/s",
    ]
    entries.append(
        ExplainEntry(
            heading="Decode throughput (cluster)",
            formula=(
                "per_gpu_tok_per_sec = memory_bandwidth × bw_util / weight_bytes_per_gpu\n"
                "TP_group_tok_per_sec = rank_tok_per_sec × effective_comm_efficiency"
            ),
            inputs=[
                ExplainInput(
                    "GPU memory_bandwidth_gbps",
                    f"{bw}",
                    "[verified]",
                    note=f"from GPU database, {report.gpu_spec.id}",
                ),
                ExplainInput(
                    "bw_util",
                    f"{d.bw_utilization:.2f}",
                    "[user-set]",
                    note="empirical, default 0.50 — override with --decode-bw-util",
                ),
                ExplainInput("weight_bytes_per_gpu", f"{weight_per_gpu:,}", "[estimated]"),
                ExplainInput("num_gpus", f"{chosen}", "[estimated]"),
                ExplainInput(
                    "cluster_comm_efficiency",
                    f"{d.cluster_comm_efficiency:.2f}",
                    "[user-set]",
                    note="effective communication coefficient including the interconnect penalty",
                ),
            ],
            steps=steps,
            result=f"{d.cluster_tokens_per_sec.value:.1f} tok/s [estimated]",
            source=d.cluster_tokens_per_sec.source or "TP bandwidth model",
            methodology_anchor="#decode-tokens-per-second",
        )
    )


# ======================================================================
# Concurrency bounds
# ======================================================================


def _concurrency(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    if report.concurrency is None:
        return
    c = report.concurrency
    entries.append(
        ExplainEntry(
            heading="K bound (memory capacity)",
            formula="K = floor(per_GPU_headroom_bytes / per_GPU_kv_bytes_per_request)",
            inputs=[
                ExplainInput(
                    "per_GPU_headroom_bytes",
                    f"{c.k_source_headroom_bytes:,}",
                    "[estimated]",
                ),
                ExplainInput(
                    "per_GPU_kv_bytes_per_request",
                    f"{c.k_source_kv_per_req_bytes:,}",
                    "[estimated]",
                    note="post-TP-sharding via min(tp, num_kv_heads)",
                ),
            ],
            steps=[
                f"K = floor({c.k_source_headroom_bytes:,} / "
                f"{c.k_source_kv_per_req_bytes:,}) = {c.k_bound.value}",
            ],
            result=f"K = {c.k_bound.value} [{c.k_bound.label.value}]",
            source="TP sharding rule from vLLM source code (verified)",
            methodology_anchor="#k-bound-memory-capacity",
        )
    )
    l_tps = report.decode.cluster_tokens_per_sec.value if report.decode else 0
    entries.append(
        ExplainEntry(
            heading="L bound (compute/bandwidth at SLA)",
            formula=(
                "L = floor(cluster_tok_per_sec / target_per_user_tok_per_sec / degradation_factor)"
            ),
            inputs=[
                ExplainInput("cluster_tok_per_sec", f"{l_tps:.1f}", "[estimated]"),
                ExplainInput(
                    "target_per_user_tok_per_sec",
                    f"{c.target_tokens_per_sec:.1f}",
                    "[user-set]",
                    note="SLA, override with --target-tokens-per-sec",
                ),
                ExplainInput(
                    "degradation_factor",
                    f"{c.degradation_factor:.2f}",
                    "[user-set]",
                    note="default 1.0 = no degradation; override with --concurrency-degradation",
                ),
            ],
            steps=[
                f"L = floor({l_tps:.1f} / {c.target_tokens_per_sec:.1f} / "
                f"{c.degradation_factor:.2f}) = {c.l_bound.value}",
            ],
            result=f"L = {c.l_bound.value} [{c.l_bound.label.value}]",
            source="Standard SLA-based capacity planning",
            methodology_anchor="#l-bound-compute-bandwidth-at-sla",
        )
    )
    entries.append(
        ExplainEntry(
            heading="Max concurrent + bottleneck verdict",
            formula="max_concurrent = min(K, L); bottleneck = 'memory_capacity' if K ≤ L else 'memory_bandwidth / compute'",
            inputs=[
                ExplainInput("K", str(c.k_bound.value), f"[{c.k_bound.label.value}]"),
                ExplainInput("L", str(c.l_bound.value), f"[{c.l_bound.label.value}]"),
            ],
            steps=[
                f"max_concurrent = min(K={c.k_bound.value}, L={c.l_bound.value}) = "
                f"{c.max_concurrent.value}",
                f"bottleneck = {c.bottleneck}",
            ],
            result=(f"{c.max_concurrent.value} concurrent, bottleneck = {c.bottleneck}"),
            source=c.bottleneck_reason_en,
            methodology_anchor="#concurrency-bounds-k-l",
        )
    )
    # Sanity check to silence "unused math import" if no steps triggered math.
    _ = math.floor(0)


# ======================================================================
# Helpers
# ======================================================================


def _fmt_ctx(ctx: int) -> str:
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    if ctx >= 1024:
        return f"{ctx // 1024}K"
    return str(ctx)
