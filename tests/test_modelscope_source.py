"""Tests for `ModelScopeSource` REST client.

We mock `httpx.get` directly — same pattern test_safetensors_reader uses.
ModelScope wraps every response in {Code, Message, Data, Success}; tests
exercise both wrapped shapes (Data dict + Data list) and the documented
error codes.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from llm_cal.model_source.base import (
    AuthRequiredError,
    ModelNotFoundError,
    SourceUnavailableError,
)
from llm_cal.model_source.modelscope import (
    DEFAULT_ENDPOINT,
    ModelScopeSource,
    _extract_files,
)


# ---------------------------------------------------------------------------
# Mock infrastructure


def _wrapped(data, code: int = 200):
    """Build a ModelScope-style envelope."""
    return {"Code": code, "Message": "ok", "RequestId": "test", "Success": True, "Data": data}


def _resp(payload, status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, content=json.dumps(payload).encode())


def _raw_resp(content: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, content=content)


def _route(url: str, kwargs: dict, info_resp, files_resp, config_resp):
    """Stand-in for httpx.get that picks a response by endpoint shape.

    httpx.get(url, params=...) passes URL without query string and params
    in kwargs — so we route by URL path + presence of FilePath param.
    """
    params = kwargs.get("params") or {}
    if "/repo/files" in url:
        return files_resp
    if "FilePath" in params:
        return config_resp
    return info_resp


# ---------------------------------------------------------------------------
# _extract_files: defensive payload parsing


class TestExtractFiles:
    def test_data_dict_files(self):
        payload = {"Data": {"Files": [{"Path": "a.bin", "Size": 1}]}}
        assert _extract_files(payload) == [{"Path": "a.bin", "Size": 1}]

    def test_data_list(self):
        payload = {"Data": [{"Path": "a.bin"}]}
        assert _extract_files(payload) == [{"Path": "a.bin"}]

    def test_no_data(self):
        assert _extract_files({"Code": 200}) is None

    def test_data_string(self):
        # Some error envelopes return a string in Data — should not crash
        assert _extract_files({"Data": "permission denied"}) is None

    def test_not_a_dict(self):
        assert _extract_files("garbage") is None
        assert _extract_files(None) is None


# ---------------------------------------------------------------------------
# ModelScopeSource.fetch: happy + error paths


class TestFetchHappyPath:
    def test_full_fetch(self, monkeypatch: pytest.MonkeyPatch):
        # Ensure no token leaks from the dev environment
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        monkeypatch.delenv("MODELSCOPE_TOKEN", raising=False)

        info_resp = _resp(_wrapped({"LatestSha": "deadbeef"}))
        files_resp = _resp(
            _wrapped(
                {
                    "Files": [
                        {"Path": "config.json", "Type": "blob", "Size": 642},
                        {
                            "Path": "model-00001-of-00002.safetensors",
                            "Type": "blob",
                            "Size": 5_000_000_000,
                        },
                        # Folder entries should be filtered out
                        {"Path": "tokenizer", "Type": "tree", "Size": None},
                    ]
                }
            )
        )
        config_payload = {"model_type": "qwen3_moe", "hidden_size": 4096}
        config_resp = _raw_resp(json.dumps(config_payload).encode())

        def _stub(url, *args, **kwargs):
            return _route(url, kwargs, info_resp, files_resp, config_resp)

        with patch("httpx.get", side_effect=_stub):
            artifact = ModelScopeSource().fetch("owner/repo")

        assert artifact.source == "modelscope"
        assert artifact.model_id == "owner/repo"
        assert artifact.commit_sha == "deadbeef"
        assert artifact.config == config_payload
        # Folder entry filtered out, blobs preserved with sizes
        assert len(artifact.siblings) == 2
        assert {s.filename for s in artifact.siblings} == {
            "config.json",
            "model-00001-of-00002.safetensors",
        }

    def test_data_list_shape_works(self):
        """Older endpoint shape: Data is the file list directly (no Files key)."""
        info_resp = _resp(_wrapped({}))
        files_resp = _resp(_wrapped([{"Path": "config.json", "Type": "blob", "Size": 100}]))
        config_resp = _raw_resp(b'{"model_type": "llama"}')

        def _stub(url, *args, **kwargs):
            return _route(url, kwargs, info_resp, files_resp, config_resp)

        with patch("httpx.get", side_effect=_stub):
            artifact = ModelScopeSource().fetch("owner/repo")
        assert artifact.commit_sha is None  # info had no LatestSha
        assert len(artifact.siblings) == 1

    def test_uses_custom_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        captured: list = []

        def _stub(url, *args, **kwargs):
            captured.append(url)
            if "files" in url:
                return _resp(_wrapped({"Files": []}))
            if "FilePath" in url:
                return _raw_resp(b"{}")
            return _resp(_wrapped({}))

        with patch("httpx.get", side_effect=_stub):
            ModelScopeSource(endpoint="https://my-mirror.example.com").fetch("o/r")

        assert all(u.startswith("https://my-mirror.example.com/") for u in captured)
        assert DEFAULT_ENDPOINT not in "".join(captured)


class TestFetchAuth:
    def test_token_in_authorization_header(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "ms-secret-xyz")
        captured_headers: list = []

        def _stub(url, *args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            if "files" in url:
                return _resp(_wrapped({"Files": []}))
            if "FilePath" in url:
                return _raw_resp(b"{}")
            return _resp(_wrapped({}))

        with patch("httpx.get", side_effect=_stub):
            ModelScopeSource().fetch("o/r")

        for h in captured_headers:
            assert h.get("Authorization") == "Bearer ms-secret-xyz"

    def test_legacy_token_env(self, monkeypatch: pytest.MonkeyPatch):
        """MODELSCOPE_TOKEN is honored as a fallback."""
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        monkeypatch.setenv("MODELSCOPE_TOKEN", "legacy-token")
        captured_headers: list = []

        def _stub(url, *args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            if "files" in url:
                return _resp(_wrapped({"Files": []}))
            if "FilePath" in url:
                return _raw_resp(b"{}")
            return _resp(_wrapped({}))

        with patch("httpx.get", side_effect=_stub):
            ModelScopeSource().fetch("o/r")

        assert all(h.get("Authorization") == "Bearer legacy-token" for h in captured_headers)


class TestFetchErrorMapping:
    def test_file_list_404_raises_not_found(self):
        info_resp = _resp(_wrapped({}))
        not_found = _resp({"Code": 404, "Message": "model not found"}, status=404)

        def _stub(url, *args, **kwargs):
            if "files" in url:
                return not_found
            return info_resp

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(ModelNotFoundError, match="not found"):
                ModelScopeSource().fetch("nonexistent/model")

    def test_file_list_401_raises_auth_required(self):
        info_resp = _resp(_wrapped({}))
        unauth = _resp({"Code": 401, "Message": "auth required"}, status=401)

        def _stub(url, *args, **kwargs):
            if "files" in url:
                return unauth
            return info_resp

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(AuthRequiredError, match="MODELSCOPE_API_TOKEN"):
                ModelScopeSource().fetch("private/repo")

    def test_file_list_403_raises_auth_required(self):
        info_resp = _resp(_wrapped({}))
        forbidden = _resp({"Code": 403, "Message": "forbidden"}, status=403)

        def _stub(url, *args, **kwargs):
            if "files" in url:
                return forbidden
            return info_resp

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(AuthRequiredError):
                ModelScopeSource().fetch("private/repo")

    def test_file_list_429_raises_source_unavailable(self):
        info_resp = _resp(_wrapped({}))
        rate_limited = httpx.Response(
            status_code=429,
            content=b'{"Code":429}',
            headers={"Retry-After": "30"},
        )

        def _stub(url, *args, **kwargs):
            if "files" in url:
                return rate_limited
            return info_resp

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(SourceUnavailableError, match="rate limit"):
                ModelScopeSource().fetch("o/r")

    def test_timeout_raises_source_unavailable(self):
        def _stub(url, *args, **kwargs):
            if "/api/v1/models/owner/repo" == url.split("?")[0].rstrip("/").rsplit(
                "/api/v1/models/owner/repo"
            )[0] + "/api/v1/models/owner/repo":
                # Info endpoint — soft fail (returns None internally), not raised
                return _resp(_wrapped({}))
            raise httpx.TimeoutException("network slow")

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(SourceUnavailableError):
                ModelScopeSource().fetch("owner/repo")

    def test_malformed_envelope_raises(self):
        info_resp = _resp(_wrapped({}))
        # Files endpoint returns an envelope with Data = string
        bad_files = _resp({"Code": 200, "Message": "ok", "Data": "permission denied"})

        def _stub(url, *args, **kwargs):
            if "files" in url:
                return bad_files
            return info_resp

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(SourceUnavailableError, match="unexpected shape"):
                ModelScopeSource().fetch("o/r")

    def test_config_invalid_json(self):
        info_resp = _resp(_wrapped({}))
        files_resp = _resp(_wrapped({"Files": []}))
        config_resp = _raw_resp(b"not valid json {{{")

        def _stub(url, *args, **kwargs):
            return _route(url, kwargs, info_resp, files_resp, config_resp)

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(SourceUnavailableError, match="config.json"):
                ModelScopeSource().fetch("o/r")

    def test_config_not_an_object(self):
        info_resp = _resp(_wrapped({}))
        files_resp = _resp(_wrapped({"Files": []}))
        # Valid JSON but a list, not a dict
        config_resp = _raw_resp(b"[1, 2, 3]")

        def _stub(url, *args, **kwargs):
            return _route(url, kwargs, info_resp, files_resp, config_resp)

        with patch("httpx.get", side_effect=_stub):
            with pytest.raises(SourceUnavailableError, match="JSON object"):
                ModelScopeSource().fetch("o/r")

    def test_info_endpoint_failure_falls_back_to_master(self):
        """If model info fails, we still call files + config with revision=master."""
        captured_revisions: list = []

        def _stub(url, *args, **kwargs):
            params = kwargs.get("params") or {}
            if "/repo/files" in url:
                captured_revisions.append(params.get("Revision"))
                return _resp(_wrapped({"Files": []}))
            if "FilePath" in params:
                captured_revisions.append(params.get("Revision"))
                return _raw_resp(b"{}")
            # Info endpoint fails with timeout
            raise httpx.TimeoutException("info slow")

        with patch("httpx.get", side_effect=_stub):
            artifact = ModelScopeSource().fetch("owner/repo")
        assert artifact.commit_sha is None
        assert all(r == "master" for r in captured_revisions)
