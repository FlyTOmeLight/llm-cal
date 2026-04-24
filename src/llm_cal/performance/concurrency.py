"""Dual-bound concurrency analysis + bottleneck classification.

Models two concurrency ceilings:
  K = memory-capacity bound
      (usable GPU memory ÷ per-request KV cache)
  L = compute/bandwidth bound at a given SLA
      (cluster decode throughput ÷ target per-user tokens/sec ÷ degradation)

Max concurrent = min(K, L). Whichever is smaller names the bottleneck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from llm_cal.output.labels import AnnotatedValue, Label
from llm_cal.performance.compute import (
    DEFAULT_CONCURRENCY_DEGRADATION,
    DecodeEstimate,
)

Bottleneck = Literal[
    "memory_capacity",
    "memory_bandwidth",
    "compute",
    "insufficient_data",
]


@dataclass(frozen=True)
class ConcurrencyAnalysis:
    # K bound
    k_bound: AnnotatedValue[int]
    k_source_headroom_bytes: int
    k_source_kv_per_req_bytes: int
    # L bound
    l_bound: AnnotatedValue[int]
    target_tokens_per_sec: float
    degradation_factor: float
    # Verdict
    max_concurrent: AnnotatedValue[int]
    bottleneck: Bottleneck
    bottleneck_reason_en: str
    bottleneck_reason_zh: str


def analyze(
    *,
    cluster_headroom_bytes: int,  # total KV headroom across all GPUs at ref context
    kv_bytes_per_request: int,  # single-request KV cache at ref context
    decode: DecodeEstimate,
    target_tokens_per_sec: float,
    degradation: float = DEFAULT_CONCURRENCY_DEGRADATION,
) -> ConcurrencyAnalysis:
    """Compute K and L bounds and pick the tighter one.

    `cluster_headroom_bytes` and `kv_bytes_per_request` should be pre-adjusted
    for TP sharding (see fleet planner for the same rule).
    """
    # K: how many requests fit in KV memory
    if kv_bytes_per_request <= 0:
        k = 0
        k_label = Label.UNKNOWN
        k_source = "KV cache per request is zero or unknown"
    else:
        k = max(0, math.floor(cluster_headroom_bytes / kv_bytes_per_request))
        k_label = Label.ESTIMATED
        k_source = (
            f"{cluster_headroom_bytes:,} bytes headroom / "
            f"{kv_bytes_per_request:,} bytes per request"
        )

    # L: how many concurrent users can maintain target tokens/sec
    cluster_tps = decode.cluster_tokens_per_sec.value
    if cluster_tps <= 0 or target_tokens_per_sec <= 0 or degradation <= 0:
        l_bound = 0
        l_label = Label.UNKNOWN
        l_source = "cluster throughput or target is zero / unknown"
    else:
        l_bound = max(0, math.floor(cluster_tps / target_tokens_per_sec / degradation))
        l_label = Label.ESTIMATED
        l_source = (
            f"{cluster_tps:.1f} tok/s cluster / "
            f"{target_tokens_per_sec:.1f} target / {degradation:.2f} degradation"
        )

    # Pick the tighter bound
    if k == 0 and l_bound == 0:
        max_n = 0
        bottleneck: Bottleneck = "insufficient_data"
        reason_en = "Both K and L unknown — cannot conclude."
        reason_zh = "K 和 L 均未知，无法得出结论。"
    elif k <= l_bound:
        max_n = k
        bottleneck = "memory_capacity"
        reason_en = (
            f"K ({k}) ≤ L ({l_bound}) → memory-capacity bound. "
            "KV cache exhausts GPU headroom before throughput SLA does."
        )
        reason_zh = (
            f"K ({k}) ≤ L ({l_bound}) → 显存容量瓶颈。先达到 KV cache 容量上限，才到吞吐目标。"
        )
    else:
        max_n = l_bound
        # Whether it's "compute" or "bandwidth" depends on where decode is bound.
        # For v0.1 we just say "memory bandwidth / compute" since decode is
        # bw-bound by default and the two share the same formula output.
        bottleneck = "memory_bandwidth"
        reason_en = (
            f"L ({l_bound}) < K ({k}) → memory-bandwidth / compute bound. "
            "Cluster can't sustain target tok/s per user at this concurrency."
        )
        reason_zh = f"L ({l_bound}) < K ({k}) → 带宽/算力瓶颈。集群在此并发下无法维持目标 tok/s。"

    return ConcurrencyAnalysis(
        k_bound=AnnotatedValue(k, k_label, source=k_source),
        k_source_headroom_bytes=cluster_headroom_bytes,
        k_source_kv_per_req_bytes=kv_bytes_per_request,
        l_bound=AnnotatedValue(l_bound, l_label, source=l_source),
        target_tokens_per_sec=target_tokens_per_sec,
        degradation_factor=degradation,
        max_concurrent=AnnotatedValue(
            max_n,
            Label.ESTIMATED if max_n > 0 else Label.UNKNOWN,
            source=f"min(K={k}, L={l_bound})",
        ),
        bottleneck=bottleneck,
        bottleneck_reason_en=reason_en,
        bottleneck_reason_zh=reason_zh,
    )
