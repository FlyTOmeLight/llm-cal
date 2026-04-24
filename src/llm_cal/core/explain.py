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
                    label="[verified]",
                ),
            ],
            steps=[
                f"Raw value from API = {w.value:,} bytes",
                f"= {w.value / 1e9:.2f} GB",
            ],
            result=f"{w.value:,} bytes [verified]",
            source=w.source or "HF siblings API",
            methodology_anchor="#weight-bytes",
        )
    )


def _quantization(report: EvaluationReport, entries: list[ExplainEntry]) -> None:
    r = report.reconciliation
    if not r.candidates:
        return
    best = r.candidates[0]
    cands_table = "\n".join(
        f"      {c.scheme:<16} predicted={c.predicted_bytes / 1e9:.2f} GB  "
        f"error={c.relative_error * 100:.1f}%"
        for c in r.candidates[:6]
    )
    entries.append(
        ExplainEntry(
            heading="Quantization scheme (reconciliation)",
            formula="best_match = argmin_scheme |observed_bytes - scheme.bpp × total_params|",
            inputs=[
                ExplainInput(
                    name="observed_bytes",
                    value=f"{r.observed_bytes:,}",
                    label="[verified]",
                ),
                ExplainInput(
                    name="total_params",
                    value=f"{r.total_params:,}",
                    label="[estimated]",
                    note="from architecture formula — see '#params-estimate' entry below",
                ),
            ],
            steps=[
                "For each known quantization scheme, predict total bytes = bpp × params:",
                cands_table,
                f"Winner: {best.scheme} at {best.relative_error * 100:.1f}% error",
            ],
            result=f"{r.best.value} [{r.best.label.value}]",
            source="Nearest-anchor match against known bytes-per-param values",
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

    is_mla = attn.variant == "MLA"
    is_csa_hca = attn.variant == "CSA_HCA"

    for ctx, av in report.kv_cache_by_context.items():
        if av.value == 0:
            continue
        # Rebuild the computation for transparency
        if is_mla and attn.kv_lora_rank:
            per_tok_per_layer = attn.kv_lora_rank * 2  # kv_lora_rank × dtype(2)
            formula = "per_tok_per_layer = kv_lora_rank × dtype_bytes   (MLA: compressed latent KV)"
            inputs = [
                ExplainInput("kv_lora_rank", str(attn.kv_lora_rank), "[verified]"),
                ExplainInput("dtype_bytes", "2", "[verified]", note="BF16/FP16"),
                ExplainInput("seq_len", f"{ctx:,}", "[verified]"),
                ExplainInput("num_layers", str(profile.num_hidden_layers), "[verified]"),
            ]
        else:
            per_tok_per_layer = 2 * attn.num_kv_heads * attn.head_dim * 2
            formula = "per_tok_per_layer = 2 × num_kv_heads × head_dim × dtype_bytes   (standard attention)"
            inputs = [
                ExplainInput("num_kv_heads", str(attn.num_kv_heads), "[verified]"),
                ExplainInput("head_dim", str(attn.head_dim), "[verified]"),
                ExplainInput("dtype_bytes", "2", "[verified]", note="BF16/FP16"),
                ExplainInput("seq_len", f"{ctx:,}", "[verified]"),
                ExplainInput("num_layers", str(profile.num_hidden_layers), "[verified]"),
            ]

        baseline = per_tok_per_layer * ctx * profile.num_hidden_layers
        steps = [
            f"per_tok_per_layer = {per_tok_per_layer:,} bytes",
            f"baseline = per_tok_per_layer × seq_len × num_layers = {baseline:,} bytes",
        ]

        if is_csa_hca and attn.compress_ratios:
            ratios = attn.compress_ratios
            avg = sum(1.0 if r == 0 else 1.0 / r for r in ratios) / len(ratios)
            inputs.append(
                ExplainInput(
                    "compress_ratios",
                    f"len={len(ratios)} (avg keep-fraction={avg:.4f})",
                    "[verified]",
                )
            )
            formula += (
                "\napply_csa_hca: baseline × avg(1/r_i for r_i in compress_ratios, 0 = keep-all=1)"
            )
            steps.extend(
                [
                    f"avg_keep_fraction = {avg:.4f}",
                    f"result = baseline × avg_keep_fraction = {av.value:,} bytes",
                ]
            )
        else:
            steps.append(f"result = baseline = {av.value:,} bytes")

        entries.append(
            ExplainEntry(
                heading=f"KV cache @ {_fmt_ctx(ctx)} context",
                formula=formula,
                inputs=inputs,
                steps=steps,
                result=f"{av.value:,} bytes = {av.value / 1e9:.2f} GB [{av.label.value}]",
                source=(
                    "DeepSeek-V2 paper (MLA); DeepSeek-V4 tech report (CSA+HCA); "
                    "standard attention formula per Attention Is All You Need (Vaswani 2017)"
                ),
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
        steps.append(
            f"tier criterion: headroom ≥ weight_per_gpu + {fit_criterion} × kv_per_request_128K"
        )
        steps.append(
            f"smallest TP count in {list(report.fleet.valid_tp_sizes)} that "
            f"satisfies the criterion: {opt.gpu_count}"
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
    effective_bw_gbs = bw * d.bw_utilization
    steps = [
        f"weight_per_gpu = {report.weight.total_bytes.value:,} / {chosen} = "
        f"{weight_per_gpu:,} bytes ({weight_per_gpu / 1e9:.2f} GB)",
        f"effective_bw = {bw} × {d.bw_utilization:.2f} = {effective_bw_gbs:.0f} GB/s",
        f"per_gpu_tok_per_sec = effective_bw / weight_per_gpu = "
        f"{effective_bw_gbs * 1e9 / weight_per_gpu:.1f} tok/s",
        f"cluster_tok_per_sec = per_gpu × {chosen} × "
        f"{d.cluster_comm_efficiency:.2f} = {d.cluster_tokens_per_sec.value:.1f} tok/s",
    ]
    entries.append(
        ExplainEntry(
            heading="Decode throughput (cluster)",
            formula=(
                "per_gpu_tok_per_sec = memory_bandwidth × bw_util / weight_bytes_per_gpu\n"
                "cluster_tok_per_sec = per_gpu × num_gpus × cluster_comm_efficiency"
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
                    note="NCCL AllReduce efficiency on NVLink, default 0.90",
                ),
            ],
            steps=steps,
            result=f"{d.cluster_tokens_per_sec.value:.1f} tok/s [estimated]",
            source="vLLM paper (Kwon et al. SOSP 2023, arxiv.org/abs/2309.06180)",
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
