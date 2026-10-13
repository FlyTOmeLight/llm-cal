"""Request credentials remain isolated across simultaneous Web evaluations."""

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "llm_cal_web", Path(__file__).parents[1] / "web/app.py"
)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


def _inputs(token, reviewer_key, source="HuggingFace"):
    return dict(
        model_id="example/model",
        gpu=["H100"],
        engine="vllm",
        context_length=None,
        lang="English",
        source=source,
        gpu_count=None,
        input_tokens=2000,
        output_tokens=512,
        target_tps=30,
        prefill_util=0.4,
        decode_bw_util=0.5,
        concurrency_degradation=1,
        refresh=False,
        explain=False,
        llm_review=True,
        hf_token=token,
        ms_token=token,
        llm_api_key=reviewer_key,
        llm_base_url="https://review.invalid/v1",
        llm_model="test-model",
    )


@pytest.mark.parametrize("source", ["HuggingFace", "ModelScope"])
def test_interleaved_web_requests_do_not_change_process_credentials(monkeypatch, source):
    monkeypatch.setenv("HF_TOKEN", "server-hf")
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "server-ms")
    monkeypatch.setenv("LLM_CAL_REVIEWER_API_KEY", "server-review")
    before = dict(os.environ)
    barrier = Barrier(2)
    closed = []
    review_calls = []

    class RequestEvaluator:
        def __init__(self, source):
            self.source = source

        def evaluate(self, **kwargs):
            barrier.wait(timeout=10)
            return SimpleNamespace(token=self.source.token)

        def close(self):
            closed.append(self.source.token)

    def review(entries, **kwargs):
        review_calls.append(kwargs["api_key"])
        return SimpleNamespace(content=kwargs["api_key"], error=None, model=kwargs["model"])

    with (
        patch.object(web, "Evaluator", RequestEvaluator),
        patch.object(web, "_render", side_effect=lambda report, locale: report.token),
        patch.object(web, "build_explain", return_value=[]),
        patch.object(web, "run_review", side_effect=review),
        patch.object(web, "_render_llm_review", side_effect=lambda content, *args: content),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        a = pool.submit(web.calculate, **_inputs("user-a", "review-a", source))
        b = pool.submit(web.calculate, **_inputs("user-b", "review-b", source))
        assert a.result(timeout=15) == ("user-a", "", "review-a")
        assert b.result(timeout=15) == ("user-b", "", "review-b")
    assert sorted(review_calls) == ["review-a", "review-b"]
    assert sorted(closed) == ["user-a", "user-b"]
    assert dict(os.environ) == before


def test_web_reports_error_and_closes_request_evaluator():
    with patch.object(web, "Evaluator") as evaluator:
        evaluator.return_value.evaluate.side_effect = ValueError("invalid input")
        main, explain, review = web.calculate(**_inputs("", ""))
    assert "invalid input" in main
    assert explain == review == ""
    evaluator.return_value.close.assert_called_once()


def test_web_ui_builds_without_model_requests():
    with patch.object(web, "Evaluator") as evaluator:
        ui = web._build_ui()
    assert ui is not None
    evaluator.assert_not_called()
