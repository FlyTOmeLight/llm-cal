"""Tests for fleet/planner.py.

CRITICAL regression: TP divisibility — num_heads=64 must NEVER recommend
TP=3/5/6/7 because vLLM/SGLang would fail to start.
"""

from __future__ import annotations

from llm_cal.architecture.profile import (
    ArchitectureProfile,
    AttentionTraits,
    Confidence,
    Family,
    MoETraits,
)
from llm_cal.fleet.planner import plan
from llm_cal.hardware.loader import lookup


def _deepseek_v4_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        model_type="deepseek_v4",
        architectures=("deepseekv4forcausallm",),
        family=Family.TRANSFORMER,
        num_hidden_layers=43,
        hidden_size=4096,
        vocab_size=129280,
        confidence=Confidence.HIGH,
        attention=AttentionTraits(
            variant="CSA_HCA",
            num_heads=64,
            num_kv_heads=1,
            head_dim=512,
            compress_ratios=tuple([0] + [4] * 42 + [0]),
        ),
        moe=MoETraits(
            num_routed_experts=256,
            num_shared_experts=1,
            num_experts_per_tok=6,
            moe_intermediate_size=2048,
        ),
        sliding_window=128,
    )


def _llama_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        model_type="llama",
        architectures=("llamaforcausallm",),
        family=Family.TRANSFORMER,
        num_hidden_layers=80,
        hidden_size=8192,
        vocab_size=128256,
        confidence=Confidence.HIGH,
        attention=AttentionTraits(variant="GQA", num_heads=64, num_kv_heads=8, head_dim=128),
    )


class TestTPDivisibility:
    """CRITICAL: fleet planner must never recommend counts that don't divide num_heads."""

    def test_64_heads_valid_tp_is_powers_of_two_up_to_8(self):
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000, 1_048_576: 17_640_000_000},
        )
        # All recommended counts must divide num_heads=64
        for opt in rec.options:
            assert 64 % opt.gpu_count == 0, (
                f"tier={opt.tier} recommended gpu_count={opt.gpu_count} "
                f"which does not divide num_heads=64"
            )

    def test_no_tp3_or_tp5_or_tp6_or_tp7(self):
        """Negative test: none of the options should be these counts."""
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000, 1_048_576: 17_640_000_000},
        )
        forbidden = {3, 5, 6, 7}
        actual_counts = {opt.gpu_count for opt in rec.options}
        assert not (actual_counts & forbidden)

    def test_valid_tp_sizes_reported(self):
        profile = _deepseek_v4_profile()
        gpu = lookup("H800")
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000, 1_048_576: 17_640_000_000},
        )
        # num_heads=64, capped at 8-GPU single-node → divisors in [1..8] of 64
        assert rec.valid_tp_sizes == (1, 2, 4, 8)


class TestThreeTierRecommendation:
    def test_all_three_tiers_present(self):
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000, 1_048_576: 17_640_000_000},
        )
        tiers = [o.tier for o in rec.options]
        assert tiers == ["min", "dev", "prod"]

    def test_deepseek_v4_on_h800_prod_recommends_8(self):
        """The reference case: 160 GB weights, ~2.2 GB KV @ 128K, on H800.
        Production tier (16 concurrent) needs 8 GPUs."""
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000, 1_048_576: 17_640_000_000},
        )
        prod = next(o for o in rec.options if o.tier == "prod")
        assert prod.gpu_count == 8
        assert prod.fits


class TestForcedGPUCount:
    def test_forced_count_returns_single_option(self):
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            forced_gpu_count=8,
            kv_bytes_by_context={131_072: 2_200_000_000, 1_048_576: 17_640_000_000},
        )
        assert len(rec.options) == 1
        assert rec.options[0].gpu_count == 8

    def test_forced_invalid_count_flags_constraint(self):
        """User forces TP=3 on num_heads=64 — option is returned but reason_en
        explains the divisibility violation."""
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            forced_gpu_count=3,
            kv_bytes_by_context={131_072: 2_200_000_000},
        )
        opt = rec.options[0]
        assert opt.gpu_count == 3
        assert "divide" in opt.reason_en.lower()


class TestOversizedModel:
    def test_weights_exceeding_all_reasonable_counts_returns_dont_fit(self):
        """A 2 TB model on H800 — even 8 cards won't hold it.

        Option is still returned (user can see the math), but fits=False.
        """
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            2_000_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000},
        )
        # The planner falls back to max TP (8), but fits=False on all tiers.
        for opt in rec.options:
            assert opt.gpu_count == 8  # fallback max
            assert not opt.fits


class TestMultiContextConcurrency:
    """Verify the planner reports concurrent capacity at each surveyed context."""

    def test_deepseek_v4_8_gpus_fits_1m_at_least_once(self):
        """Headline: 8xH800 on DeepSeek-V4-Flash supports at least one request at 1M."""
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={
                131_072: 2_200_000_000,
                1_048_576: 17_640_000_000,
            },
        )
        prod = next(o for o in rec.options if o.tier == "prod")
        ctx_map = dict(prod.max_concurrent_by_context)
        assert ctx_map[1_048_576] >= 1
        # And 128K density is of course much higher
        assert ctx_map[131_072] > ctx_map[1_048_576]


class TestLlamaDense:
    def test_llama3_70b_on_h100_fits_on_2(self):
        """70B at BF16 → ~140 GB, H100 has 80 GB → TP=2 works."""
        gpu = lookup("H100")
        profile = _llama_profile()
        rec = plan(
            profile,
            140_000_000_000,
            1_000_000_000,
            gpu,
            kv_bytes_by_context={131_072: 1_000_000_000},
        )
        min_opt = next(o for o in rec.options if o.tier == "min")
        assert min_opt.gpu_count <= 2
        assert min_opt.fits


class TestTPShardingOfKV:
    """CRITICAL modeling fix: GQA KV cache splits across TP ranks.

    A 70B GQA model with kv_heads=8 on TP=8 should give each rank 1/8 of
    the KV cache. Previous version treated it as replication (1x) — that
    conservatively overestimated KV pressure by ~8x.
    """

    def test_gqa_kv_divides_by_tp(self):
        """Llama-3 70B at 128K: total KV ~10 GB (rough), TP=8 → ~1.25 GB/GPU."""
        gpu = lookup("H100")
        profile = _llama_profile()  # GQA kv_heads=8
        total_kv_128k = 10_000_000_000  # 10 GB total
        rec = plan(
            profile,
            140_000_000_000,
            total_kv_128k,
            gpu,
            kv_bytes_by_context={131_072: total_kv_128k},
        )
        prod = next(o for o in rec.options if o.tier == "prod")
        # With TP=8 and 8 kv_heads, each GPU holds ~1/8 the KV.
        # Headroom at TP=8: (72 - 17.5) = ~54.5 GB per GPU.
        # kv_per_gpu = 10/8 = 1.25 GB → ~43 concurrent @ 128K.
        ctx_map = dict(prod.max_concurrent_by_context)
        assert ctx_map[131_072] > 30, (
            f"expected >30 concurrent under TP-aware sharding, got {ctx_map[131_072]}"
        )

    def test_mqa_kv_stays_replicated(self):
        """MQA (kv_heads=1) cannot shard — per-GPU KV = total KV."""
        gpu = lookup("H800")
        profile = _deepseek_v4_profile()  # MQA kv_heads=1
        rec = plan(
            profile,
            160_000_000_000,
            2_200_000_000,
            gpu,
            kv_bytes_by_context={131_072: 2_200_000_000},
        )
        prod = next(o for o in rec.options if o.tier == "prod")
        ctx_map = dict(prod.max_concurrent_by_context)
        # Unchanged from pre-refactor: ~23 concurrent @ 128K
        assert 15 <= ctx_map[131_072] <= 35

    def test_shards_saturate_at_num_kv_heads(self):
        """TP=16 on a kv_heads=8 model still shards only 8 ways (4090-scale tests)."""
        from llm_cal.fleet.planner import _kv_shards

        profile = _llama_profile()  # kv_heads=8
        assert _kv_shards(profile, tp_size=1) == 1
        assert _kv_shards(profile, tp_size=2) == 2
        assert _kv_shards(profile, tp_size=8) == 8
        assert _kv_shards(profile, tp_size=16) == 8  # saturated
