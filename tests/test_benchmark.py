"""Tests for benchmark runner.

These tests use stubbed evaluators — NO network. The actual benchmark
against live HF data runs via `llm-cal --benchmark` at user request.
"""

from __future__ import annotations

from llm_cal.architecture.profile import (
    ArchitectureProfile,
    AttentionTraits,
    Confidence,
    Family,
    MoETraits,
)
from llm_cal.benchmark.runner import (
    Expectation,
    _evaluate_field,
    exit_code_from,
    load_dataset,
)
from llm_cal.core.evaluator import EvaluationReport
from llm_cal.output.labels import AnnotatedValue, Label
from llm_cal.weight_analyzer import WeightReport
from llm_cal.weight_analyzer.reconciler import ReconciliationReport


def _fake_report(
    attention_variant: str = "CSA_HCA",
    quantization: str = "FP4_FP8_MIXED",
    weight_bytes: int = 160_000_000_000,
    is_moe: bool = True,
) -> EvaluationReport:
    """Construct an EvaluationReport without hitting the network."""
    attention = AttentionTraits(
        variant=attention_variant,  # type: ignore[arg-type]
        num_heads=64,
        num_kv_heads=1,
        head_dim=512,
    )
    moe = (
        MoETraits(
            num_routed_experts=256,
            num_shared_experts=1,
            num_experts_per_tok=6,
            moe_intermediate_size=2048,
        )
        if is_moe
        else None
    )
    profile = ArchitectureProfile(
        model_type="fake",
        architectures=(),
        family=Family.TRANSFORMER,
        num_hidden_layers=43,
        hidden_size=4096,
        vocab_size=129280,
        confidence=Confidence.HIGH,
        attention=attention,
        moe=moe,
    )
    weight = WeightReport(
        total_bytes=AnnotatedValue(weight_bytes, Label.VERIFIED),
        bits_per_param=AnnotatedValue(4.5, Label.INFERRED),
        quantization_guess=AnnotatedValue(quantization, Label.INFERRED),  # type: ignore[arg-type]
    )
    return EvaluationReport(
        model_id="fake",
        source="huggingface",
        commit_sha=None,
        gpu="H800",
        gpu_spec=None,
        gpu_error=None,
        engine="vllm",
        profile=profile,
        weight=weight,
        total_params_estimate=AnnotatedValue(284_000_000_000, Label.ESTIMATED),
        reconciliation=ReconciliationReport(
            observed_bytes=weight_bytes,
            total_params=284_000_000_000,
            candidates=(),
            best=AnnotatedValue(quantization, Label.INFERRED),  # type: ignore[arg-type]
        ),
    )


class TestCheckEvaluation:
    """_evaluate_field covers the individual field checks."""

    def test_attention_variant_pass(self):
        report = _fake_report(attention_variant="CSA_HCA")
        exp = Expectation(field="attention_variant", expected="CSA_HCA", source="test")
        predicted, status = _evaluate_field(report, exp)
        assert status == "PASS"
        assert predicted == "CSA_HCA"

    def test_attention_variant_fail(self):
        report = _fake_report(attention_variant="GQA")
        exp = Expectation(field="attention_variant", expected="CSA_HCA", source="test")
        predicted, status = _evaluate_field(report, exp)
        assert status == "FAIL"
        assert predicted == "GQA"

    def test_quantization_pass(self):
        report = _fake_report(quantization="FP4_FP8_MIXED")
        exp = Expectation(field="quantization", expected="FP4_FP8_MIXED", source="test")
        _, status = _evaluate_field(report, exp)
        assert status == "PASS"

    def test_weight_bytes_within_range(self):
        report = _fake_report(weight_bytes=160_000_000_000)
        exp = Expectation(
            field="weight_bytes",
            expected_min=150_000_000_000,
            expected_max=170_000_000_000,
            source="test",
        )
        _, status = _evaluate_field(report, exp)
        assert status == "PASS"

    def test_weight_bytes_outside_range_fails(self):
        report = _fake_report(weight_bytes=300_000_000_000)
        exp = Expectation(
            field="weight_bytes",
            expected_min=150_000_000_000,
            expected_max=170_000_000_000,
            source="test",
        )
        _, status = _evaluate_field(report, exp)
        assert status == "FAIL"

    def test_is_moe_pass(self):
        report = _fake_report(is_moe=True)
        exp = Expectation(field="is_moe", expected=True, source="test")
        _, status = _evaluate_field(report, exp)
        assert status == "PASS"

    def test_unknown_field_returns_skip(self):
        report = _fake_report()
        exp = Expectation(field="nonexistent_field", expected="whatever", source="test")
        predicted, status = _evaluate_field(report, exp)
        assert status == "SKIP"
        assert "unknown field" in predicted


class TestDataset:
    """Bundled dataset.yaml must be parseable and non-empty."""

    def test_dataset_loads(self):
        ds = load_dataset()
        assert ds.schema_version == 1
        assert len(ds.entries) >= 4

    def test_every_expectation_has_source(self):
        """Honesty constraint: every expected value must cite where it came from."""
        ds = load_dataset()
        for entry in ds.entries:
            for exp in entry.expectations:
                assert exp.source, (
                    f"Entry '{entry.name}' field '{exp.field}' missing source — "
                    "cite HF API / model card / vLLM recipe / etc."
                )

    def test_deepseek_v4_flash_is_in_dataset(self):
        """The signature case must always be in the benchmark dataset."""
        ds = load_dataset()
        entries_with_v4 = [e for e in ds.entries if "DeepSeek-V4-Flash" in e.model_id]
        assert len(entries_with_v4) >= 1
        v4 = entries_with_v4[0]
        # Must check both quantization (the tool's value prop) and attention variant.
        fields = {exp.field for exp in v4.expectations}
        assert "quantization" in fields
        assert "attention_variant" in fields


class TestExitCode:
    def test_exit_0_on_all_pass(self):
        from llm_cal.benchmark.runner import CheckResult

        results = [
            CheckResult("a", "f1", "PASS", "", "", "src"),
            CheckResult("a", "f2", "PASS", "", "", "src"),
        ]
        assert exit_code_from(results) == 0

    def test_exit_1_on_any_fail(self):
        from llm_cal.benchmark.runner import CheckResult

        results = [
            CheckResult("a", "f1", "PASS", "", "", "src"),
            CheckResult("a", "f2", "FAIL", "", "", "src"),
        ]
        assert exit_code_from(results) == 1

    def test_exit_0_when_only_skips(self):
        from llm_cal.benchmark.runner import CheckResult

        results = [
            CheckResult("a", "f1", "SKIP", "", "", "src"),
        ]
        assert exit_code_from(results) == 0


def test_benchmark_flag_in_cli_help():
    """Smoke: --benchmark flag is exposed in the CLI."""
    from typer.testing import CliRunner

    from llm_cal.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--benchmark" in result.stdout
