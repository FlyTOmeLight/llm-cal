"""Tests for `architecture.detector.detect()` and trait sub-detectors.

CRITICAL regression tests are marked. Do not delete or weaken them.
"""

from __future__ import annotations

from typing import Any

from llm_cal.architecture.detector import _fallback_unknown, detect
from llm_cal.architecture.profile import Confidence, Family
from llm_cal.architecture.traits import (
    detect_attention,
    detect_moe,
    detect_sliding_window,
)


class TestDeepSeekV4Flash:
    """DeepSeek-V4-Flash is the tool's reference case. Three traits stack here."""

    def test_family_is_transformer(self, load_config):
        p = detect(load_config("deepseek_v4_flash"))
        assert p.family == Family.TRANSFORMER

    def test_confidence_is_high(self, load_config):
        p = detect(load_config("deepseek_v4_flash"))
        assert p.confidence == Confidence.HIGH

    def test_attention_is_csa_hca(self, load_config):
        # MLA keys ARE present (q_lora_rank=1024) but CSA_HCA takes precedence.
        # compress_ratios length is n_layers (43) + num_nextn_predict_layers (1) = 44.
        p = detect(load_config("deepseek_v4_flash"))
        assert p.attention is not None
        assert p.attention.variant == "CSA_HCA"
        assert p.attention.compress_ratios is not None
        assert len(p.attention.compress_ratios) == 44

    def test_moe_present(self, load_config):
        p = detect(load_config("deepseek_v4_flash"))
        assert p.moe is not None
        assert p.moe.num_routed_experts == 256
        assert p.moe.num_shared_experts == 1
        assert p.moe.num_experts_per_tok == 6

    def test_sliding_window_present(self, load_config):
        p = detect(load_config("deepseek_v4_flash"))
        assert p.sliding_window == 128


class TestLlama3Dense:
    def test_family_is_transformer(self, load_config):
        p = detect(load_config("llama3_70b"))
        assert p.family == Family.TRANSFORMER

    def test_attention_is_gqa(self, load_config):
        p = detect(load_config("llama3_70b"))
        assert p.attention is not None
        assert p.attention.variant == "GQA"
        assert p.attention.num_kv_heads == 8
        assert p.attention.num_heads == 64

    def test_no_moe(self, load_config):
        p = detect(load_config("llama3_70b"))
        assert p.moe is None

    def test_no_sliding_window(self, load_config):
        p = detect(load_config("llama3_70b"))
        assert p.sliding_window is None


class TestMistralSliding:
    def test_sliding_window_detected(self, load_config):
        p = detect(load_config("mistral_sliding"))
        assert p.sliding_window == 4096


class TestMamba:
    """State-space models are identified but marked v0.1 unsupported."""

    def test_family_is_state_space(self, load_config):
        p = detect(load_config("mamba"))
        assert p.family == Family.STATE_SPACE

    def test_v0_1_unsupported_flag(self, load_config):
        p = detect(load_config("mamba"))
        assert p.auxiliary.get("v0_1_unsupported") is True


class TestFallback:
    """Graceful degradation when config is unusable.

    The design doc promises "works on day-0" — if a brand new model type shows up
    we must not crash, we must return UNKNOWN.
    """

    def test_missing_model_type_returns_unknown(self, load_config):
        p = detect(load_config("unknown_model"))
        assert p.family == Family.UNKNOWN
        assert p.confidence == Confidence.LOW
        assert "warning" in p.auxiliary

    def test_completely_empty_config(self):
        p = detect({})
        assert p.family == Family.UNKNOWN
        assert p.confidence == Confidence.LOW

    def test_unknown_model_type_is_medium_confidence(self):
        """A future model_type we don't recognize but has full config is MEDIUM."""
        config = {
            "model_type": "hypothetical_v1",
            "architectures": ["HypotheticalForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        p = detect(config)
        assert p.family == Family.TRANSFORMER
        assert p.confidence == Confidence.MEDIUM


class TestCSAHCALengthMismatchFallthrough:
    """CRITICAL regression test (reviewer flagged twice, skill IRON RULE).

    If `compress_ratios` exists but its length doesn't equal `num_hidden_layers`,
    we must fall through to standard attention detection. Failure to do so would
    cause silent KV-cache mis-estimation on future variants.
    """

    def test_length_mismatch_is_not_classified_as_csa_hca(self):
        config: dict[str, Any] = {
            "model_type": "hypothetical",
            "architectures": ["Hypothetical"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "compress_ratios": [4, 128, 4],  # length 3, n_layers 32 — mismatch!
        }
        p = detect(config)
        assert p.attention is not None
        assert p.attention.variant != "CSA_HCA"
        # Should fall through to GQA (num_kv_heads=8 < num_heads=32)
        assert p.attention.variant == "GQA"

    def test_mtp_extra_layer_is_accepted(self):
        """DeepSeek models with `num_nextn_predict_layers` extend the ratios array.

        Real DeepSeek-V4-Flash: num_hidden_layers=43, num_nextn_predict_layers=1,
        compress_ratios length=44. This must STILL be identified as CSA_HCA.
        """
        config: dict[str, Any] = {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_nextn_predict_layers": 1,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "vocab_size": 129280,
            "compress_ratios": [0] * 44,  # length = n_layers + n_nextn
            "q_lora_rank": 1024,
        }
        p = detect(config)
        assert p.attention is not None
        assert p.attention.variant == "CSA_HCA"


class TestAttentionVariantOrdering:
    """Trait detection is order-sensitive. These tests pin down the contract."""

    def test_mla_beats_gqa_when_q_lora_rank_present(self):
        config: dict[str, Any] = {
            "model_type": "deepseek_v2",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,  # looks like MQA
            "q_lora_rank": 1024,  # but q_lora_rank wins
            "vocab_size": 10,
        }
        p = detect(config)
        assert p.attention is not None
        assert p.attention.variant == "MLA"

    def test_mqa_vs_gqa_boundary(self):
        """num_kv_heads=1 → MQA; num_kv_heads > 1 but < num_heads → GQA."""
        base = {
            "model_type": "foo",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 10,
        }
        mqa = detect_attention({**base, "num_key_value_heads": 1})
        gqa = detect_attention({**base, "num_key_value_heads": 8})
        mha = detect_attention({**base, "num_key_value_heads": 32})
        assert mqa.variant == "MQA"
        assert gqa.variant == "GQA"
        assert mha.variant == "MHA"


class TestDirectTraitDetectors:
    """Unit-level tests bypassing detect() — for sharper failure messages."""

    def test_detect_moe_returns_none_for_dense(self, load_config):
        assert detect_moe(load_config("llama3_70b")) is None

    def test_detect_moe_picks_up_num_routed_experts(self, load_config):
        moe = detect_moe(load_config("deepseek_v4_flash"))
        assert moe is not None
        assert moe.num_routed_experts == 256

    def test_detect_sliding_window(self, load_config):
        assert detect_sliding_window(load_config("llama3_70b")) is None
        assert detect_sliding_window(load_config("mistral_sliding")) == 4096
        assert detect_sliding_window({"sliding_window": 0}) is None


class TestFallbackHelper:
    def test_fallback_returns_profile_shape(self):
        p = _fallback_unknown({})
        assert p.family == Family.UNKNOWN
        assert p.confidence == Confidence.LOW
        assert p.num_hidden_layers == 0
