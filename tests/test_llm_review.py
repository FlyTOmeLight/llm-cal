"""Tests for llm_review module — no network, pure input/output logic."""

from __future__ import annotations

import pytest

from llm_cal.core.explain import ExplainEntry, ExplainInput
from llm_cal.llm_review.reviewer import (
    LLMReviewResult,
    _build_prompt,
    _format_entry,
    _system_prompt,
    run_review,
)


def _sample_entries() -> list[ExplainEntry]:
    return [
        ExplainEntry(
            heading="Weight bytes",
            formula="sum(safetensors.size)",
            inputs=[ExplainInput("api", "HF siblings", "[verified]")],
            steps=["result = 160 GB"],
            result="160 GB [verified]",
            source="HF API",
        ),
        ExplainEntry(
            heading="Prefill latency",
            formula="2 × params × input_tokens / TFLOPS",
            inputs=[
                ExplainInput("params", "284B", "[estimated]"),
                ExplainInput("input_tokens", "2000", "[user-set]"),
            ],
            steps=["FLOPs = 1.1e15", "latency = 735 ms"],
            result="735 ms [estimated]",
            source="Kaplan 2020",
        ),
    ]


class TestPromptConstruction:
    """Prompt must include every entry's heading, formula, and result."""

    def test_english_prompt_contains_entry_data(self):
        entries = _sample_entries()
        prompt = _build_prompt(entries, locale="en")
        for entry in entries:
            assert entry.heading in prompt
            assert entry.result in prompt

    def test_chinese_prompt_contains_entry_data(self):
        entries = _sample_entries()
        prompt = _build_prompt(entries, locale="zh")
        for entry in entries:
            assert entry.heading in prompt

    def test_system_prompt_language(self):
        en = _system_prompt("en")
        zh = _system_prompt("zh")
        assert "auditor" in en.lower()
        assert "审计" in zh

    def test_format_entry_includes_all_parts(self):
        entry = _sample_entries()[1]
        formatted = _format_entry(entry)
        assert "## Prefill latency" in formatted
        assert "Formula:" in formatted
        assert "Inputs:" in formatted
        assert "Steps:" in formatted
        assert "Result:" in formatted
        assert "Source:" in formatted


class TestMissingAPIKey:
    """Without LLM_CAL_REVIEWER_API_KEY, should return graceful error."""

    def test_missing_key_returns_error_result(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_CAL_REVIEWER_API_KEY", raising=False)
        result = run_review(_sample_entries(), locale="en")
        assert result.ok is False
        assert result.content is None
        assert result.error is not None
        assert "LLM_CAL_REVIEWER_API_KEY" in result.error


class TestEnvironmentConfig:
    """Base URL and model come from env vars with sensible defaults."""

    def test_default_model_is_gpt4o(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_CAL_REVIEWER_API_KEY", raising=False)
        monkeypatch.delenv("LLM_CAL_REVIEWER_MODEL", raising=False)
        result = run_review(_sample_entries(), locale="en")
        assert result.model == "gpt-4o"

    def test_default_base_url_is_openai(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_CAL_REVIEWER_API_KEY", raising=False)
        monkeypatch.delenv("LLM_CAL_REVIEWER_BASE_URL", raising=False)
        result = run_review(_sample_entries(), locale="en")
        assert result.base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_CAL_REVIEWER_API_KEY", raising=False)
        monkeypatch.setenv("LLM_CAL_REVIEWER_BASE_URL", "https://api.deepseek.com/v1/")
        result = run_review(_sample_entries(), locale="en")
        # Trailing slash should be stripped for consistent concatenation
        assert result.base_url == "https://api.deepseek.com/v1"

    def test_custom_model(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_CAL_REVIEWER_API_KEY", raising=False)
        monkeypatch.setenv("LLM_CAL_REVIEWER_MODEL", "deepseek-chat")
        result = run_review(_sample_entries(), locale="en")
        assert result.model == "deepseek-chat"


class TestResultShape:
    """LLMReviewResult contract."""

    def test_failure_result_always_has_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LLM_CAL_REVIEWER_API_KEY", raising=False)
        result = run_review(_sample_entries(), locale="en")
        assert isinstance(result, LLMReviewResult)
        assert result.ok is False
        assert result.error is not None


def test_cli_flag_in_help():
    """Smoke test: --llm-review flag is exposed in CLI help."""
    import os

    from typer.testing import CliRunner

    from llm_cal.cli import app

    runner = CliRunner()
    env = {**os.environ, "COLUMNS": "200", "TERM": "xterm-256color"}
    result = runner.invoke(app, ["--help"], env=env)
    assert result.exit_code == 0
    assert "llm-review" in result.stdout.lower()
