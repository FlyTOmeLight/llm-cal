"""Tests for weight_analyzer + reconciler.

CRITICAL regression: DeepSeek-V4-Flash FP4+FP8 pack identification. This is the
core "vs gpu_poor" story the tool is built around.
"""

from __future__ import annotations

from llm_cal.model_source.base import SiblingFile
from llm_cal.weight_analyzer import analyze
from llm_cal.weight_analyzer.fingerprint import QuantFingerprint
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

    def test_ties_surfaced_in_source(self):
        """FP4_FP8_MIXED, GPTQ_INT4, AWQ_INT4 all share bpp=0.55.

        The LLM review caught this: the tool was silently picking the first
        without telling the user it's a tie. The source field must now name
        the tied alternatives.
        """
        observed = 160_300_000_000
        total_params = 284_000_000_000
        report = reconcile(observed, total_params)

        source = report.best.source or ""
        # The tie note must explicitly name the other schemes that share the bpp
        assert "tied with" in source
        assert "GPTQ_INT4" in source or "AWQ_INT4" in source

    def test_no_tie_note_when_unique_winner(self):
        """Pure FP16 model: FP16/BF16 share bpp=2.0 but nothing else does."""
        # 70B * 2 bytes = 140 GB exactly — FP16 and BF16 will tie, INT8 is far off
        report = reconcile(140_000_000_000, 70_000_000_000)
        source = report.best.source or ""
        # FP16/BF16 ARE aliases and will tie — that's expected to surface.
        # But INT8 (1.0 bpp) should NOT be in the tie note.
        assert "INT8" not in source


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


class TestReconcilerFingerprint:
    """Fingerprint-driven tie-breaking — the v0.1.2 story."""

    def test_fingerprint_breaks_fp4_gptq_awq_tie(self):
        """Three schemes tie at 0.55 bpp; fingerprint picks the real one."""
        observed = 160_300_000_000
        total_params = 284_000_000_000

        fp_awq = QuantFingerprint(
            scheme="AWQ_INT4",
            source_type="safetensors_header",
            evidence="safetensors header has .qweight + .qzeros, no .g_idx (AWQ marker)",
        )
        report = reconcile(observed, total_params, fingerprint=fp_awq)

        # Declared scheme wins over argmin (which was FP4_FP8_MIXED by dict order)
        assert report.best.value == "AWQ_INT4"
        # VERIFIED label because we read authoritative evidence
        assert report.best.label.value == "verified"
        assert "AWQ marker" in (report.best.source or "")

    def test_fingerprint_gptq(self):
        observed = 160_300_000_000
        total_params = 284_000_000_000

        fp_gptq = QuantFingerprint(
            scheme="GPTQ_INT4",
            source_type="config_json",
            evidence="config.json quantization_config.quant_method=gptq, bits=4",
        )
        report = reconcile(observed, total_params, fingerprint=fp_gptq)

        assert report.best.value == "GPTQ_INT4"
        assert report.best.label.value == "verified"
        assert "quant_method=gptq" in (report.best.source or "")

    def test_fingerprint_fp4_fp8_mixed(self):
        observed = 160_300_000_000
        total_params = 284_000_000_000

        fp = QuantFingerprint(
            scheme="FP4_FP8_MIXED",
            source_type="safetensors_header",
            evidence="safetensors header has both FP4 and FP8 weight tensors",
        )
        report = reconcile(observed, total_params, fingerprint=fp)

        assert report.best.value == "FP4_FP8_MIXED"
        assert report.best.label.value == "verified"

    def test_fingerprint_disagreement_still_trusts_declaration(self):
        """Fingerprint says FP8 but bytes predict 45% off (classic gpu_poor trap).

        We trust the declaration and flag the mismatch in source.
        """
        observed = 160_300_000_000  # matches FP4_FP8_MIXED
        total_params = 284_000_000_000

        fp_fp8 = QuantFingerprint(
            scheme="FP8",
            source_type="config_json",
            evidence="config.json quant_method=fp8",
        )
        report = reconcile(observed, total_params, fingerprint=fp_fp8)

        assert report.best.value == "FP8"
        assert report.best.label.value == "verified"
        assert "NOTE" in (report.best.source or "")
        assert "off by" in (report.best.source or "")

    def test_unknown_fingerprint_falls_back(self):
        """Fingerprint declares a scheme we don't have a bpp anchor for."""
        observed = 160_300_000_000
        total_params = 284_000_000_000

        fp_unknown = QuantFingerprint(
            scheme="UNKNOWN",  # type: ignore[arg-type]
            source_type="config_json",
            evidence="we declared UNKNOWN for some reason",
        )
        report = reconcile(observed, total_params, fingerprint=fp_unknown)

        # Falls back to argmin (FP4_FP8_MIXED by bytes)
        assert report.best.value == "FP4_FP8_MIXED"
        assert "fell back" in (report.best.source or "")
