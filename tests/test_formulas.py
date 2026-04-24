"""Tests for architecture.formulas.{weight, kv_cache}.

The headline assertion: DeepSeek-V4-Flash KV cache at 128K context is within
1% of the hand-computed expected value (Success Criteria #3).
"""

from __future__ import annotations

from llm_cal.architecture.detector import detect
from llm_cal.architecture.formulas.kv_cache import compute_kv_cache_bytes
from llm_cal.architecture.formulas.weight import (
    estimate_total_params,
    predicted_bytes_under_quant,
)


class TestKVCacheBasic:
    def test_standard_mha_formula(self, load_config):
        """Llama-3-70B @ 2K context. GQA with 8 kv heads.

        expected_per_token_per_layer = 2 * 8 * 128 * 2 = 4096 bytes
        expected_total = 4096 * 2048 * 80 = ~671 MB
        """
        profile = detect(load_config("llama3_70b"))
        kv = compute_kv_cache_bytes(profile, seq_len=2048, dtype_bytes=2)
        expected = 2 * 8 * 128 * 2 * 2048 * 80
        assert abs(kv.value - expected) <= 1, f"got {kv.value}, expected {expected}"

    def test_sliding_window_caps_seq_len(self, load_config):
        """Mistral with sliding_window=4096 @ 32K context.

        KV cache should be computed as if seq_len == 4096.
        """
        profile = detect(load_config("mistral_sliding"))
        capped = compute_kv_cache_bytes(profile, seq_len=32_768, dtype_bytes=2)
        same_as_window = compute_kv_cache_bytes(profile, seq_len=4096, dtype_bytes=2)
        assert capped.value == same_as_window.value

    def test_zero_seq_len_is_zero(self, load_config):
        profile = detect(load_config("llama3_70b"))
        kv = compute_kv_cache_bytes(profile, seq_len=0)
        assert kv.value == 0


class TestKVCacheDeepSeekV4Flash:
    """Reference case. Must be < 1% error at 128K context (Success Criteria #3)."""

    def test_128k_kv_cache_within_1_percent(self, load_config):
        profile = detect(load_config("deepseek_v4_flash"))
        kv = compute_kv_cache_bytes(profile, seq_len=128_000, dtype_bytes=2)

        # Hand computation:
        #   per_token_per_layer = 2 * 1 (num_kv_heads) * 512 (head_dim) * 2 = 2048 bytes
        #   baseline = 2048 * 128000 * 43 = 11,272,192,000 bytes (~10.5 GB)
        #   compress_ratios = [0, 0, 4, 128, 4, 128, ..., 4, 0] len 44
        #     - zeros: 3 (positions 0, 1, 43) -> 3 * 1.0 = 3.0
        #     - fours: 21 -> 21 * 0.25 = 5.25
        #     - 128s: 20 -> 20 * (1/128) = 0.15625
        #     - average = 8.40625 / 44 = 0.19105
        #   expected = 11,272,192,000 * 0.19105 ≈ 2,153,540,000 bytes (~2 GB)
        per_token_per_layer = 2 * 1 * 512 * 2
        baseline = per_token_per_layer * 128_000 * 43
        ratios = profile.attention.compress_ratios  # type: ignore[union-attr]
        assert ratios is not None
        avg_ratio = sum(1.0 if r == 0 else 1.0 / r for r in ratios) / len(ratios)
        expected = int(baseline * avg_ratio)

        delta = abs(kv.value - expected)
        rel_err = delta / expected
        assert rel_err < 0.01, (
            f"KV cache error {rel_err * 100:.3f}% exceeds 1% budget. "
            f"Got {kv.value}, expected ~{expected}"
        )

    def test_label_is_estimated(self, load_config):
        profile = detect(load_config("deepseek_v4_flash"))
        kv = compute_kv_cache_bytes(profile, seq_len=128_000, dtype_bytes=2)
        assert kv.label.value == "estimated"

    def test_scales_linearly_with_context(self, load_config):
        """CSA_HCA ignores sliding_window (sparse mechanism already handles it),
        so KV cache grows with seq_len * avg_compress_ratio."""
        profile = detect(load_config("deepseek_v4_flash"))
        kv_32k = compute_kv_cache_bytes(profile, seq_len=32_000, dtype_bytes=2)
        kv_128k = compute_kv_cache_bytes(profile, seq_len=128_000, dtype_bytes=2)
        # 128K / 32K = 4, so kv should be ~4x
        ratio = kv_128k.value / kv_32k.value
        assert 3.9 < ratio < 4.1, f"expected ~4x scaling, got {ratio:.2f}x"


class TestKVCacheMLA:
    """MLA uses compressed latent KV — far smaller than standard MHA shape would suggest."""

    def test_mla_uses_kv_lora_rank(self):
        from llm_cal.architecture.profile import (
            ArchitectureProfile,
            AttentionTraits,
            Confidence,
            Family,
        )

        profile = ArchitectureProfile(
            model_type="deepseek_v2",
            architectures=("deepseekv2forcausallm",),
            family=Family.TRANSFORMER,
            num_hidden_layers=60,
            hidden_size=5120,
            vocab_size=102400,
            confidence=Confidence.HIGH,
            attention=AttentionTraits(
                variant="MLA",
                num_heads=128,
                num_kv_heads=128,
                head_dim=128,
                q_lora_rank=1536,
                kv_lora_rank=512,
            ),
        )
        kv = compute_kv_cache_bytes(profile, seq_len=8192, dtype_bytes=2)
        # MLA: kv_lora_rank * dtype * seq * layers
        expected = 512 * 2 * 8192 * 60
        assert kv.value == expected


class TestKVCacheUnknown:
    def test_unknown_family_returns_unknown_label(self):
        from llm_cal.architecture.profile import (
            ArchitectureProfile,
            Confidence,
            Family,
        )

        profile = ArchitectureProfile(
            model_type="mystery",
            architectures=(),
            family=Family.UNKNOWN,
            num_hidden_layers=0,
            hidden_size=0,
            vocab_size=0,
            confidence=Confidence.LOW,
        )
        kv = compute_kv_cache_bytes(profile, seq_len=128_000)
        assert kv.value == 0
        assert kv.label.value == "unknown"

    def test_state_space_returns_unknown(self, load_config):
        profile = detect(load_config("mamba"))
        kv = compute_kv_cache_bytes(profile, seq_len=8192)
        assert kv.value == 0
        assert kv.label.value == "unknown"


class TestWeightEstimation:
    def test_llama3_70b_param_count_close_to_70b(self, load_config):
        profile = detect(load_config("llama3_70b"))
        # Llama3-70B fixture has no `intermediate_size` so we fall back to 4*hidden.
        # The real model uses 28672 intermediate which makes it ~70B. Our fallback
        # gives a rougher estimate — but the formula label is [estimated] and the
        # reconciler step owns precise matching. Here we just ensure it's in the
        # right order of magnitude.
        params = estimate_total_params(profile)
        assert params.label.value == "estimated"
        assert 40_000_000_000 < params.value < 100_000_000_000

    def test_predicted_bytes_fp16(self):
        """70B params at FP16 should be ~140 GB."""
        out = predicted_bytes_under_quant(70_000_000_000, "FP16")
        assert out.value == 140_000_000_000

    def test_predicted_bytes_fp4_fp8_mixed(self):
        """284B at FP4+FP8 mixed should be around 156 GB (close to observed 160 GB)."""
        out = predicted_bytes_under_quant(284_000_000_000, "FP4_FP8_MIXED")
        # 284B * 0.55 = 156.2B
        assert 155_000_000_000 < out.value < 158_000_000_000

    def test_unknown_scheme_returns_unknown_label(self):
        out = predicted_bytes_under_quant(1_000_000, "UNKNOWN")
        assert out.label.value == "unknown"
