"""Fleet planner — reverse-inference of "how many GPUs do I need".

Three tiers:
  * min  — just enough to hold weights + light overhead
           (can run single requests at short context)
  * dev  — room for ~8 concurrent at 128K context
  * prod — room for ~16 concurrent at 128K context

TP-divisibility constraint (CRITICAL regression test): the number of attention
heads must be divisible by the number of GPUs. vLLM/SGLang with TP=3 on a
64-head model would fail to start; we only recommend counts in the valid set.

Reserved overhead per GPU = 10% of HBM (CUDA context + activations + framework),
which matches `--gpu-memory-utilization 0.9` in vLLM.

Per-GPU KV modeling is TP-aware:

  per_gpu_KV = total_KV / min(tp_size, max(1, num_kv_heads))

  * MQA (kv_heads=1): KV replicates fully across ranks → divisor is 1,
    per-GPU KV = total (accurate for DeepSeek V4-Flash, Qwen MQA variants).
  * GQA (kv_heads=8): KV splits across ranks up to num_kv_heads → at TP=8,
    per-GPU KV = total/8 (accurate for Llama 3 70B, Qwen 72B).
  * MHA: splits fully up to num_heads.

This matches vLLM/SGLang's actual sharding behavior. MLA-latent KV is
technically replicated in most frameworks, but since num_kv_heads is
typically 1 in MLA (DeepSeek V2/V3/V4), the formula degenerates to
replication anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.hardware.loader import GPUSpec

Tier = Literal["min", "dev", "prod"]

_OVERHEAD_FRACTION = 0.10
_KV_HEAD_ROOM_CONCURRENT: dict[Tier, int] = {
    "min": 1,  # one request worth of KV at 128K
    "dev": 8,
    "prod": 16,
}
# For recommendation logic, compute per-GPU fit at this reference context length.
_REFERENCE_CTX_TOKENS = 131_072
# Max recommended TP within a single 8-GPU node. Beyond this we'd want PP/EP,
# which is out of v0.1 scope.
_MAX_TP_SINGLE_NODE = 8


@dataclass(frozen=True)
class FleetOption:
    tier: Tier
    gpu_count: int
    weight_bytes_per_gpu: int
    kv_bytes_per_request: int  # at reference context (128K)
    max_concurrent_at_reference_ctx: int
    # concurrency ceiling at each context length the user asked about.
    # Key = context token count, value = max concurrent requests that fit.
    max_concurrent_by_context: tuple[tuple[int, int], ...]
    usable_bytes_per_gpu: int
    fits: bool  # False => the best we can do still overflows headroom at reference ctx
    reason_en: str
    reason_zh: str


@dataclass(frozen=True)
class FleetRecommendation:
    options: tuple[FleetOption, ...]
    best_tier: Tier
    valid_tp_sizes: tuple[int, ...]
    constraint_note_en: str
    constraint_note_zh: str


def plan(
    profile: ArchitectureProfile,
    weight_bytes: int,
    kv_bytes_per_request_at_ref: int,
    gpu: GPUSpec,
    forced_gpu_count: int | None = None,
    kv_bytes_by_context: dict[int, int] | None = None,
) -> FleetRecommendation:
    """Recommend GPU counts for the three tiers, or a single option when forced.

    `kv_bytes_by_context` is optional metadata used only for the per-option
    concurrency breakdown (e.g. "~23 concurrent @ 128K, ~2 @ 1M"). Tier-fit
    decisions still use `kv_bytes_per_request_at_ref` (the reference context).
    """
    kv_by_ctx = kv_bytes_by_context or {}
    bytes_per_gpu_total = gpu.memory_gb * 1_000_000_000
    usable_per_gpu = int(bytes_per_gpu_total * (1 - _OVERHEAD_FRACTION))
    valid_tp = _valid_tp_sizes(profile)

    constraint_en = _constraint_note_en(profile, valid_tp)
    constraint_zh = _constraint_note_zh(profile, valid_tp)

    if forced_gpu_count is not None:
        option = _evaluate_count(
            forced_gpu_count,
            profile=profile,
            weight_bytes=weight_bytes,
            kv_bytes=kv_bytes_per_request_at_ref,
            usable_per_gpu=usable_per_gpu,
            valid_tp=valid_tp,
            tier="dev",  # generic label when user forced
            kv_by_context=kv_by_ctx,
        )
        return FleetRecommendation(
            options=(option,),
            best_tier="dev",
            valid_tp_sizes=tuple(valid_tp),
            constraint_note_en=constraint_en,
            constraint_note_zh=constraint_zh,
        )

    options: list[FleetOption] = []
    for tier in ("min", "dev", "prod"):
        gpu_count = _smallest_fitting_count(
            valid_tp,
            profile=profile,
            weight_bytes=weight_bytes,
            kv_bytes=kv_bytes_per_request_at_ref,
            usable_per_gpu=usable_per_gpu,
            concurrent=_KV_HEAD_ROOM_CONCURRENT[tier],
        )
        # Fall back to the largest TP if nothing fits — flagged as `fits=False`.
        chosen = gpu_count if gpu_count is not None else max(valid_tp)
        option = _evaluate_count(
            chosen,
            profile=profile,
            weight_bytes=weight_bytes,
            kv_bytes=kv_bytes_per_request_at_ref,
            usable_per_gpu=usable_per_gpu,
            valid_tp=valid_tp,
            tier=tier,
            kv_by_context=kv_by_ctx,
        )
        options.append(option)

    # Best tier: dev if it fits, otherwise min, otherwise whatever exists
    best = "dev" if options[1].fits else ("min" if options[0].fits else "prod")
    return FleetRecommendation(
        options=tuple(options),
        best_tier=best,  # type: ignore[arg-type]
        valid_tp_sizes=tuple(valid_tp),
        constraint_note_en=constraint_en,
        constraint_note_zh=constraint_zh,
    )


def _valid_tp_sizes(profile: ArchitectureProfile) -> list[int]:
    """Divisors of num_heads, capped at the single-node maximum."""
    if profile.attention is None or profile.attention.num_heads <= 0:
        return [1]
    h = profile.attention.num_heads
    divisors = [i for i in range(1, min(h, _MAX_TP_SINGLE_NODE) + 1) if h % i == 0]
    return divisors or [1]


def _kv_shards(profile: ArchitectureProfile, tp_size: int) -> int:
    """How many ways KV cache can be split across TP ranks.

    Saturates at num_kv_heads: once tp_size > num_kv_heads, extra ranks
    just replicate, so the divisor stops growing.
    """
    if profile.attention is None:
        return 1
    kv_heads = max(1, profile.attention.num_kv_heads)
    return min(tp_size, kv_heads)


def _smallest_fitting_count(
    valid_tp: list[int],
    *,
    profile: ArchitectureProfile,
    weight_bytes: int,
    kv_bytes: int,
    usable_per_gpu: int,
    concurrent: int,
) -> int | None:
    for n in valid_tp:
        if _fits(n, profile, weight_bytes, kv_bytes, usable_per_gpu, concurrent):
            return n
    return None


def _fits(
    gpu_count: int,
    profile: ArchitectureProfile,
    weight_bytes: int,
    kv_bytes: int,
    usable_per_gpu: int,
    concurrent: int,
) -> bool:
    weight_per_gpu = math.ceil(weight_bytes / gpu_count)
    shards = _kv_shards(profile, gpu_count)
    kv_per_gpu = math.ceil(kv_bytes / shards)
    needed = weight_per_gpu + concurrent * kv_per_gpu
    return needed <= usable_per_gpu


def _evaluate_count(
    gpu_count: int,
    *,
    profile: ArchitectureProfile,
    weight_bytes: int,
    kv_bytes: int,
    usable_per_gpu: int,
    valid_tp: list[int],
    tier: Tier,
    kv_by_context: dict[int, int],
) -> FleetOption:
    weight_per_gpu = math.ceil(weight_bytes / gpu_count)
    shards = _kv_shards(profile, gpu_count)
    kv_per_gpu = math.ceil(kv_bytes / shards)
    headroom = usable_per_gpu - weight_per_gpu
    max_concurrent = max(0, headroom // kv_per_gpu) if kv_per_gpu > 0 else 0
    # Per-context concurrency, sorted by context length ascending, each using
    # the TP-sharded per-GPU KV.
    max_concurrent_by_ctx = tuple(
        (
            ctx,
            (max(0, headroom // math.ceil(kv / shards)) if kv > 0 else 0),
        )
        for ctx, kv in sorted(kv_by_context.items())
    )
    fits = _fits(
        gpu_count,
        profile,
        weight_bytes,
        kv_bytes,
        usable_per_gpu,
        _KV_HEAD_ROOM_CONCURRENT[tier],
    )

    # Reason strings
    if gpu_count not in valid_tp:
        reason_en = (
            f"GPU count {gpu_count} does not divide num_heads — valid TP sizes: {sorted(valid_tp)}"
        )
        reason_zh = f"GPU 张数 {gpu_count} 无法整除注意力头数——有效 TP 张数：{sorted(valid_tp)}"
    elif not fits:
        reason_en = (
            f"Weights + {_KV_HEAD_ROOM_CONCURRENT[tier]}x KV would exceed "
            f"{usable_per_gpu / 1e9:.1f} GB usable per GPU"
        )
        reason_zh = (
            f"权重 + {_KV_HEAD_ROOM_CONCURRENT[tier]} 份 KV 超过单卡可用的 "
            f"{usable_per_gpu / 1e9:.1f} GB"
        )
    else:
        reason_en = f"fits ~{max_concurrent} concurrent @ {_REFERENCE_CTX_TOKENS // 1024}K ctx"
        reason_zh = f"可容纳约 {max_concurrent} 并发请求 @ {_REFERENCE_CTX_TOKENS // 1024}K 上下文"

    return FleetOption(
        tier=tier,
        gpu_count=gpu_count,
        weight_bytes_per_gpu=weight_per_gpu,
        kv_bytes_per_request=kv_bytes,
        max_concurrent_at_reference_ctx=max_concurrent,
        max_concurrent_by_context=max_concurrent_by_ctx,
        usable_bytes_per_gpu=usable_per_gpu,
        fits=fits,
        reason_en=reason_en,
        reason_zh=reason_zh,
    )


def _constraint_note_en(profile: ArchitectureProfile, valid_tp: list[int]) -> str:
    heads = profile.attention.num_heads if profile.attention else 0
    return f"TP must divide num_heads={heads}. Candidates within one node (<=8 GPUs): {valid_tp}."


def _constraint_note_zh(profile: ArchitectureProfile, valid_tp: list[int]) -> str:
    heads = profile.attention.num_heads if profile.attention else 0
    return f"TP 张数必须整除 num_heads={heads}。单节点（≤8 卡）候选：{valid_tp}。"
