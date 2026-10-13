"""Pinned metadata, endpoints and per-request HuggingFace authentication."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from llm_cal.model_source.base import SourceUnavailableError
from llm_cal.model_source.huggingface import HuggingFaceSource


def test_revision_probe_and_pinned_fetch_use_effective_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.invalid")
    info = SimpleNamespace(
        sha="sha-a", siblings=[SimpleNamespace(rfilename="model.safetensors", size=100)]
    )
    with (
        patch("llm_cal.model_source.huggingface.HfApi") as api,
        patch("httpx.get", return_value=httpx.Response(200, json={"model_type": "llama"})) as get,
    ):
        api.return_value.model_info.return_value = info
        source = HuggingFaceSource(token="request-token")
        revision = source.resolve_revision("a/b")
        artifact = source.fetch("a/b", revision=revision)
    assert artifact.commit_sha == revision
    calls = api.return_value.model_info.call_args_list
    assert calls[0].kwargs["files_metadata"] is False
    assert calls[1].kwargs["files_metadata"] is True
    assert calls[1].kwargs["revision"] == "sha-a"
    assert get.call_args.args[0] == "https://mirror.invalid/a/b/resolve/sha-a/config.json"
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer request-token"}


def test_explicit_anonymous_request_ignores_server_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "server-secret")
    with patch("llm_cal.model_source.huggingface.HfApi") as api:
        api.return_value.model_info.return_value = SimpleNamespace(sha="sha-a", gated=False)
        source = HuggingFaceSource(token="")
        source.resolve_revision("a/b")
    assert api.call_args.kwargs["token"] is False
    assert api.return_value.model_info.call_args.kwargs["token"] is False


@pytest.mark.parametrize("payload", [[], None, "text"])
def test_non_object_config_is_source_error(payload):
    with patch("httpx.get", return_value=httpx.Response(200, content=json.dumps(payload))):
        source = HuggingFaceSource(token="")
        with pytest.raises(SourceUnavailableError, match="JSON object"):
            source._fetch_config("a/b", "sha", None)


def test_gated_public_metadata_does_not_bypass_file_authorization():
    from llm_cal.model_source.base import AuthRequiredError

    info = SimpleNamespace(sha="sha-a", gated="auto")
    with (
        patch("llm_cal.model_source.huggingface.HfApi") as api,
        patch("httpx.get", return_value=httpx.Response(403)),
    ):
        api.return_value.model_info.return_value = info
        with pytest.raises(AuthRequiredError):
            HuggingFaceSource(token="").resolve_revision("a/b")
