"""Performance modeling for prefill latency and decode throughput.

FORMULAS — with sources. See docs/methodology.md for the full audit.

Prefill (compute-bound):
    FLOPs = 2 × params × input_tokens
    latency = FLOPs / (peak_TFLOPS × num_gpus × utilization × 1e12)

    Source: Kaplan et al. 2020, "Scaling Laws for Neural Language Models".
    The "2" factor is the forward-pass cost per param per token, a standard
    approximation in transformer inference literature.

Decode (memory-bandwidth-bound):
    per_token_time = weight_bytes_per_gpu / (memory_bandwidth × utilization)
    tokens_per_second = memory_bandwidth × utilization / weight_bytes_per_gpu

    Source: Kwon et al. SOSP 2023 "Efficient Memory Management for Large
    Language Model Serving with PagedAttention"; NVIDIA "Mastering LLM
    Techniques: Inference Optimization" (2023 technical blog).

UTILIZATION FACTORS (all empirical, ALL user-overridable):
  - Prefill 40% — midpoint of vLLM-reported 30-50% MFU on H100
  - Decode BW 50% — midpoint of NVIDIA/vLLM-reported 40-65% achieved bandwidth
  - Cluster comm 90% — typical NCCL AllReduce efficiency at TP=8 on NVLink
  - Concurrency degradation 1.0 (no degradation by default)
    This is the most uncertain factor. Prior versions defaulted to 1.5
    (borrowed from an LLM-generated report), which was NOT from a primary
    source. v0.1 defaults to 1.0 (honest baseline) and exposes the knob
    so users can dial in whatever their engine actually achieves.

MoE "active" vs "total":
    Strictly, MoE decode only reads the active experts per token. The
    ratio used here is a rough approximation:
        active_ratio ≈ (experts_per_tok + shared_experts) / (routed + shared)
    This UNDERESTIMATES active weight because attention + embeddings are
    always active (not just experts). For a more accurate number, use the
    model card's stated "total / active" figure if available. The
    "active-only" throughput is labeled "optimistic" for this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.hardware.loader import GPUSpec
from llm_cal.output.labels import AnnotatedValue, Label

# Empirical defaults. All user-overridable via CLI.
DEFAULT_PREFILL_UTILIZATION = 0.40
DEFAULT_DECODE_BW_UTILIZATION = 0.50
DEFAULT_CLUSTER_COMM_EFFICIENCY = 0.90
# Honest baseline. Previously 1.5, borrowed from an LLM-generated report —
# that had no primary source, so we reset to 1.0. Users who observe actual
# degradation on their engine should dial this up via CLI.
DEFAULT_CONCURRENCY_DEGRADATION = 1.0


@dataclass(frozen=True)
class PrefillEstimate:
    total_flops: AnnotatedValue[int]  # [estimated] 2 * params * input_tokens
    peak_effective_tflops: AnnotatedValue[float]  # TFLOPS × utilization
    latency_ms: AnnotatedValue[float]
    utilization: float  # the factor used (for provenance)


@dataclass(frozen=True)
class DecodeEstimate:
    active_weight_bytes_per_gpu: AnnotatedValue[int]
    per_gpu_tokens_per_sec: AnnotatedValue[float]
    cluster_tokens_per_sec: AnnotatedValue[float]  # after comm efficiency
    bw_utilization: float
    cluster_comm_efficiency: float
    moe_active_weight_bytes_per_gpu: AnnotatedValue[int] | None = None
    moe_active_tokens_per_sec: AnnotatedValue[float] | None = None


def estimate_prefill(
    profile: ArchitectureProfile,
    total_params: int,
    gpu: GPUSpec,
    num_gpus: int,
    input_tokens: int,
    utilization: float = DEFAULT_PREFILL_UTILIZATION,
) -> PrefillEstimate:
    """Estimate single-request prefill latency.

    Based on compute: FLOPs = 2 × params × tokens; latency = FLOPs / effective_FLOPS.
    """
    flops = 2 * total_params * input_tokens
    # TP distributes compute, so aggregate TFLOPS = num_gpus × per-card × util
    aggregate_tflops = gpu.fp16_tflops * num_gpus * utilization
    # Guard against zero
    if aggregate_tflops <= 0 or total_params <= 0 or input_tokens <= 0:
        return PrefillEstimate(
            total_flops=AnnotatedValue(0, Label.UNKNOWN, source="insufficient inputs"),
            peak_effective_tflops=AnnotatedValue(0.0, Label.UNKNOWN),
            latency_ms=AnnotatedValue(0.0, Label.UNKNOWN),
            utilization=utilization,
        )
    latency_s = flops / (aggregate_tflops * 1e12)
    latency_ms = latency_s * 1000.0

    return PrefillEstimate(
        total_flops=AnnotatedValue(
            flops,
            Label.ESTIMATED,
            source=f"2 × {total_params:,} params × {input_tokens:,} tokens",
        ),
        peak_effective_tflops=AnnotatedValue(
            aggregate_tflops,
            Label.ESTIMATED,
            source=f"{gpu.fp16_tflops} × {num_gpus} GPUs × {utilization:.0%} util",
        ),
        latency_ms=AnnotatedValue(
            latency_ms,
            Label.ESTIMATED,
            source=(f"{flops:.2e} FLOPs / ({aggregate_tflops:.1f} effective TFLOPS × 1e12)"),
        ),
        utilization=utilization,
    )


def estimate_decode(
    profile: ArchitectureProfile,
    total_weight_bytes: int,
    gpu: GPUSpec,
    num_gpus: int,
    bw_utilization: float = DEFAULT_DECODE_BW_UTILIZATION,
    cluster_comm_efficiency: float = DEFAULT_CLUSTER_COMM_EFFICIENCY,
    moe_active_params_ratio: float | None = None,
) -> DecodeEstimate:
    """Estimate decode tokens/second.

    Decode is memory-bandwidth-bound: per-token time = weight_bytes / bw.
    Under TP, weights split across ranks, so per-GPU weight bytes = total / N.

    If the model is MoE and moe_active_params_ratio is given (e.g. 0.3 for
    active/total), we ALSO report an optimistic "active only" throughput.
    """
    if gpu.memory_bandwidth_gbps is None or gpu.memory_bandwidth_gbps <= 0:
        _unknown = AnnotatedValue(
            0, Label.UNKNOWN, source="GPU memory_bandwidth_gbps not in database"
        )
        _unknown_f = AnnotatedValue(
            0.0, Label.UNKNOWN, source="GPU memory_bandwidth_gbps not in database"
        )
        return DecodeEstimate(
            active_weight_bytes_per_gpu=_unknown,
            per_gpu_tokens_per_sec=_unknown_f,
            cluster_tokens_per_sec=_unknown_f,
            bw_utilization=bw_utilization,
            cluster_comm_efficiency=cluster_comm_efficiency,
        )

    bw_bytes_per_s = gpu.memory_bandwidth_gbps * 1e9  # GB/s → bytes/s
    effective_bw = bw_bytes_per_s * bw_utilization
    weight_per_gpu = max(1, total_weight_bytes // num_gpus)
    per_gpu_tps = effective_bw / weight_per_gpu
    # Cluster-level: per-GPU × N × comm_efficiency (since requests are batched)
    cluster_tps = per_gpu_tps * num_gpus * cluster_comm_efficiency

    # MoE active-only optimistic view
    moe_active_weight: AnnotatedValue[int] | None = None
    moe_active_tps: AnnotatedValue[float] | None = None
    if profile.is_moe and moe_active_params_ratio is not None and moe_active_params_ratio > 0:
        active_bytes = int(weight_per_gpu * moe_active_params_ratio)
        moe_active_weight = AnnotatedValue(
            active_bytes,
            Label.ESTIMATED,
            source=f"{weight_per_gpu:,} × {moe_active_params_ratio:.3f} (active/total ratio)",
        )
        if active_bytes > 0:
            active_per_gpu_tps = effective_bw / active_bytes
            active_cluster_tps = active_per_gpu_tps * num_gpus * cluster_comm_efficiency
            moe_active_tps = AnnotatedValue(
                active_cluster_tps,
                Label.ESTIMATED,
                source=(
                    f"optimistic MoE active-only: effective_bw / {active_bytes:,} × "
                    f"{num_gpus} × {cluster_comm_efficiency:.2f}"
                ),
            )

    return DecodeEstimate(
        active_weight_bytes_per_gpu=AnnotatedValue(
            weight_per_gpu,
            Label.ESTIMATED,
            source=f"{total_weight_bytes:,} bytes / {num_gpus} TP ranks",
        ),
        per_gpu_tokens_per_sec=AnnotatedValue(
            per_gpu_tps,
            Label.ESTIMATED,
            source=(
                f"{gpu.memory_bandwidth_gbps} GB/s × {bw_utilization:.0%} util / "
                f"{weight_per_gpu:,} weight bytes"
            ),
        ),
        cluster_tokens_per_sec=AnnotatedValue(
            cluster_tps,
            Label.ESTIMATED,
            source=(
                f"per-GPU × {num_gpus} GPUs × {cluster_comm_efficiency:.0%} cluster comm efficiency"
            ),
        ),
        bw_utilization=bw_utilization,
        cluster_comm_efficiency=cluster_comm_efficiency,
        moe_active_weight_bytes_per_gpu=moe_active_weight,
        moe_active_tokens_per_sec=moe_active_tps,
    )
