"""Offline user-visible evaluation and derivation regressions."""

from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from llm_cal.benchmark.runner import exit_code_from, run_all
from llm_cal.cli import app
from llm_cal.core.cache import ArtifactCache
from llm_cal.core.evaluator import Evaluator
from llm_cal.core.explain import build
from llm_cal.model_source.base import (
    ModelArtifact,
    ModelSource,
    SiblingFile,
    SourceUnavailableError,
)
from llm_cal.output.formatter import render


class FakeSource(ModelSource):
    name = "huggingface"
    endpoint = "https://example.invalid"
    token = "request-token"

    def __init__(self, config, size=14_000_000_000):
        self.config = config
        self.size = size
        self.sha = "revision-a"
        self.fetches = 0
        self.probes = 0

    def resolve_revision(self, model_id):
        self.probes += 1
        return self.sha

    def fetch(self, model_id, revision=None):
        self.fetches += 1
        return ModelArtifact(
            self.name,
            model_id,
            self.sha,
            self.config,
            (SiblingFile("model.safetensors", self.size),),
        )


@pytest.fixture
def evaluator(tmp_path, load_config):
    source = FakeSource(load_config("mistral_sliding"))
    cache = ArtifactCache(tmp_path / "cache")
    evaluator = Evaluator(source, cache)
    with patch(
        "llm_cal.core.evaluator.fetch_tensor_dtypes", return_value={"layer.weight": "BF16"}
    ) as header:
        yield evaluator, source, header
    evaluator.close()


def test_cache_reuses_metadata_and_header_but_probes_access(evaluator):
    ev, source, header = evaluator
    first = ev.evaluate("example/model", "H100", "vllm")
    second = ev.evaluate("example/model", "H800", "vllm")
    assert first.commit_sha == second.commit_sha
    assert source.probes == 2
    assert source.fetches == 1
    assert header.call_count == 1
    assert header.call_args.kwargs["endpoint"] == source.endpoint
    assert header.call_args.kwargs["token"] == source.token
    source.sha = "revision-b"
    ev.evaluate("example/model", "H100", "vllm")
    assert source.fetches == 2
    assert header.call_count == 2
    ev.evaluate("example/model", "H100", "vllm", refresh=True)
    assert source.fetches == 3
    assert header.call_count == 3


def test_revision_access_failure_cannot_serve_cached_private_metadata(evaluator):
    ev, source, _ = evaluator
    ev.evaluate("example/model", "H100", "vllm")
    with (
        patch.object(
            source, "resolve_revision", side_effect=SourceUnavailableError("access denied")
        ),
        pytest.raises(SourceUnavailableError),
    ):
        ev.evaluate("example/model", "H100", "vllm")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_count": 0},
        {"gpu_count": -1},
        {"context_length": 0},
        {"input_tokens": -1},
        {"output_tokens": 0},
        {"target_tokens_per_sec": 0},
        {"target_tokens_per_sec": float("nan")},
        {"prefill_utilization": 0},
        {"decode_bw_utilization": 1.1},
        {"concurrency_degradation": -1},
    ],
)
def test_invalid_numeric_inputs_fail_before_network(evaluator, kwargs):
    ev, source, _ = evaluator
    with pytest.raises(ValueError):
        ev.evaluate("example/model", "H100", "vllm", **kwargs)
    assert source.probes == 0


def test_invalid_engine_fails_before_network(evaluator):
    ev, source, _ = evaluator
    with pytest.raises(ValueError, match="engine"):
        ev.evaluate("example/model", "H100", "unknown")
    assert source.probes == 0


@pytest.mark.parametrize("count,size", [(3, 14_000_000_000), (8, 2_000_000_000_000)])
def test_invalid_or_oversized_fleet_has_no_command(evaluator, count, size):
    ev, source, _ = evaluator
    source.size = size
    report = ev.evaluate("example/model", "H100", "vllm", gpu_count=count)
    assert not report.fleet.options[0].fits
    assert report.generated_command is None
    assert report.decode is None


@pytest.mark.parametrize("size", [None, 0])
def test_missing_weight_size_cannot_produce_capacity_recommendation(evaluator, size):
    ev, source, _ = evaluator
    source.size = size
    report = ev.evaluate("example/model", "H100", "vllm")
    assert report.weight.total_bytes.label.value == "unknown"
    assert report.fleet is None
    assert report.generated_command is None
    assert report.reconciliation.best.label.value == "unknown"


def test_unknown_architecture_cannot_produce_capacity_recommendation(evaluator):
    ev, source, _ = evaluator
    source.config = {}
    report = ev.evaluate("example/model", "H100", "vllm")
    assert report.fleet is None
    assert report.generated_command is None


@pytest.mark.parametrize("variant", ["sliding", "nsa"])
def test_explain_shows_kv_modifiers_and_actual_result(evaluator, variant):
    ev, source, _ = evaluator
    if variant == "nsa":
        source.config = {**source.config, "nsa_config": {"topk": 1024}}
    report = ev.evaluate("example/model", "H100", "vllm", context_length=32768)
    entry = next(e for e in build(report) if e.heading.startswith("KV cache"))
    steps = " ".join(entry.steps)
    assert ("sliding_window" if variant == "sliding" else "NSA keep_fraction") in steps
    per_token = 2 * 8 * 128 * 2
    expected = per_token * (4096 if variant == "sliding" else 1024) * 32
    assert report.kv_cache_by_context[32768].value == expected
    assert entry.steps[-1] == f"result = {expected:,} bytes"


def test_fingerprint_selection_and_parameter_entry_are_explained(evaluator):
    ev, source, header = evaluator
    source.size = 4_400_000_000  # 8.047B shape parameters at the 0.55 B/param anchor
    header.return_value = {"layer.qweight": "I32", "layer.qzeros": "I32"}
    report = ev.evaluate("example/model", "H100", "vllm")
    entries = build(report)
    assert any("#params-estimate" in e.heading for e in entries)
    quant = next(e for e in entries if e.heading.startswith("Quantization"))
    assert "AWQ_INT4" in quant.result
    assert "inferred" in quant.result
    assert "safetensors" in quant.source
    assert any("Selection: AWQ_INT4" in s for s in quant.steps)


def test_short_max_context_still_produces_valid_report(evaluator):
    ev, source, _ = evaluator
    source.config = {**source.config, "max_position_embeddings": 2048}
    report = ev.evaluate("example/model", "H100", "vllm")
    assert list(report.kv_cache_by_context) == [2048]


def test_cli_and_formatter_render_offline_report(evaluator):
    ev, _, _ = evaluator
    with patch("llm_cal.cli.Evaluator", return_value=ev):
        result = CliRunner().invoke(
            app, ["example/model", "--gpu", "H100", "--lang", "en", "--explain"]
        )
    assert result.exit_code == 0, result.output
    assert "sliding_window" in result.output
    assert "vllm serve" in result.output


def test_formatter_preserves_unknown_weight_label(evaluator):
    from io import StringIO

    ev, source, _ = evaluator
    source.size = None
    report = ev.evaluate("example/model", "H100", "vllm")
    output = StringIO()
    render(report, Console(file=output, width=160, color_system=None))
    assert "unknown" in output.getvalue().lower()


@pytest.mark.parametrize(
    "error,status",
    [
        (SourceUnavailableError("offline"), "SKIP"),
        (RuntimeError("bug"), "FAIL"),
    ],
)
def test_failed_benchmark_never_returns_success(evaluator, error, status):
    ev, _, _ = evaluator
    with patch.object(ev, "evaluate", side_effect=error):
        results = run_all(evaluator=ev)
    assert results and {r.status for r in results} == {status}
    assert exit_code_from(results) == 1


def test_shared_cache_is_partitioned_by_request_credentials(evaluator):
    ev, source, header = evaluator
    ev.evaluate("example/model", "H100", "vllm")
    source.token = "another-request-token"
    ev.evaluate("example/model", "H100", "vllm")
    assert source.fetches == 2
    assert header.call_count == 2
    assert "another-request-token" not in ev._cache_key("example/model", source.sha).to_string()


def test_decode_explain_uses_effective_interconnect_coefficient(evaluator):
    ev, _, _ = evaluator
    report = ev.evaluate("example/model", "H800", "vllm", gpu_count=4)
    entry = next(e for e in build(report) if e.heading.startswith("Decode throughput"))
    efficiency = report.decode.cluster_comm_efficiency
    assert efficiency == pytest.approx(0.9 * (0.85 + 0.15 * 400 / 900))
    assert any(f"{efficiency:.2f}" in step for step in entry.steps)
    assert not any("rank_tok_per_sec " + chr(215) + " 4" in step for step in entry.steps)


def test_uncertain_weight_is_not_verified_in_explain(evaluator):
    ev, source, _ = evaluator
    source.size = None
    entry = build(ev.evaluate("example/model", "H100", "vllm"))[0]
    assert "[unknown]" in entry.result
    assert "[verified]" not in entry.result
