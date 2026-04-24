"""Tests for weight_analyzer + reconciler.

CRITICAL regression: DeepSeek-V4-Flash FP4+FP8 pack identification. This is the
core "vs gpu_poor" story the tool is built around.
"""

from __future__ import annotations

from llm_cal.model_source.base import SiblingFile
from llm_cal.weight_analyzer import analyze
from llm_cal.weight_analyzer.reconciler import reconcile


class TestAnalyze:
    def test_verified_label_on_total_bytes(self):
        siblings = (
            SiblingFile("model-00001-of-00002.safetensors", 100),
            SiblingFile("model-00002-of-00002.safetensors", 200),
            SiblingFile("tokenizer.json", 5),
        )
        report = analyze(siblings, total_params=1000)
        assert report.total_bytes.value == 300
        assert report.total_bytes.label.value == "verified"

    def test_skips_non_safetensors(self):
        siblings = (
            SiblingFile("model.safetensors", 100),
            SiblingFile("pytorch_model.bin", 500),  # should NOT be counted
            SiblingFile("config.json", 10),
        )
        report = analyze(siblings, total_params=100)
        assert report.total_bytes.value == 100

    def test_inferred_label_on_bits_per_param(self):
        siblings = (SiblingFile("model.safetensors", 200),)
        report = analyze(siblings, total_params=100)
        assert report.bits_per_param is not None
        assert report.bits_per_param.label.value == "inferred"
        # 200 bytes / 100 params = 2 bytes/param = 16 bits/param → FP16/BF16
        assert report.bits_per_param.value == 16.0


class TestReconcilerDeepSeekV4Flash:
    """CRITICAL: DeepSeek-V4-Flash reconciliation.

    284B params, 160 GB observed -> must prefer FP4_FP8_MIXED over FP8/INT4.
    Competitor tool `gpu_poor` picks FP8 here and reports 284 GB (wrong by 1.8x).
    """

    def test_fp4_fp8_pack_identified(self):
        observed = 160_300_000_000  # ~160.3 GB
        total_params = 284_000_000_000
        report = reconcile(observed, total_params)

        assert report.best.value == "FP4_FP8_MIXED"
        assert report.best.label.value == "inferred"

        # Sanity: FP8 hypothesis should be the runner-up but clearly worse
        schemes = [c.scheme for c in report.candidates]
        assert schemes[0] == "FP4_FP8_MIXED"


class TestReconcilerPureSchemes:
    def test_fp16_model_picks_fp16(self):
        # 70B * 2 bytes = 140 GB
        report = reconcile(140_000_000_000, 70_000_000_000)
        assert report.best.value in ("FP16", "BF16")

    def test_fp8_model_picks_fp8(self):
        report = reconcile(70_000_000_000, 70_000_000_000)
        assert report.best.value == "FP8"


class TestReconcilerEdgeCases:
    def test_zero_observed_returns_unknown(self):
        report = reconcile(0, 1_000_000)
        assert report.best.value == "UNKNOWN"
        assert report.best.label.value == "unknown"

    def test_zero_params_returns_unknown(self):
        report = reconcile(1_000_000, 0)
        assert report.best.value == "UNKNOWN"

    def test_implausibly_large_observed_returns_unknown(self):
        """observed >> any predicted (e.g. corruption, or not a LLM).

        Tolerance gate should catch this.
        """
        # 10 bytes/param = way above FP16's 2.00
        report = reconcile(10 * 1_000_000, 1_000_000)
        assert report.best.value == "UNKNOWN"
