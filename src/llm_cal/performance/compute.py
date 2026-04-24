"""Performance modeling for prefill latency and decode throughput.

FORMULAS (standard, from transformer inference literature):

Prefill (compute-bound):
    FLOPs = 2 × active_params × input_tokens
    latency = FLOPs / (peak_TFLOPS × 1e12 × utilization)

Decode (memory-bandwidth-bound):
    Per-token time = active_weight_bytes / (memory_bandwidth × utilization)
    Tokens per second = memory_bandwidth × utilization / active_weight_bytes

Utilization factors are EMPIRICAL and workload-dependent. Defaults are
conservative midpoints from published benchmarks — see docstrings.

MoE "active" vs "total" param question:
    Strictly, MoE decode only reads `num_experts_per_tok` active experts.
    But vLLM batching typically touches most experts across a batch, so
    the worst-case "all weights" assumption is often closer to reality.
    This module returns BOTH numbers when the model is MoE — let the user
    decide.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.hardware.loader import GPUSpec
from llm_cal.output.labels import AnnotatedValue, Label

# Empirical defaults. Documented so users can override via CLI.
DEFAULT_PREFILL_UTILIZATION = 0.40  # compute utilization — vLLM typically hits 35-50%
DEFAULT_DECODE_BW_UTILIZATION = 0.50  # memory-bw utilization on decode — 40-65% common
DEFAULT_CLUSTER_COMM_EFFICIENCY = 0.90  # AllReduce overhead reduces per-GPU scaling
# Throughput degradation at high concurrency (page-table thrash, scheduler overhead).
DEFAULT_CONCURRENCY_DEGRADATION = 1.5


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
