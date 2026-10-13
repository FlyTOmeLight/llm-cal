"""HuggingFace source. Uses `huggingface_hub` for metadata + `httpx` for config fetch.

Anti-pattern warning: do NOT call `list_repo_files()` then head-request each file.
Always use `model_info(files_metadata=True)` which returns all sibling sizes in
ONE request. Verified in `tests/test_huggingface_source.py` by asserting HTTP call count.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from llm_cal.model_source.auth import get_hf_token, hf_auth_error_message
from llm_cal.model_source.base import (
    AuthRequiredError,
    ModelArtifact,
    ModelNotFoundError,
    ModelSource,
    SiblingFile,
    SourceUnavailableError,
)


class HuggingFaceSource(ModelSource):
    name = "huggingface"

    def __init__(
        self, endpoint: str | None = None, timeout_s: float = 30.0, token: str | None = None
    ) -> None:
        self.endpoint = (
            endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
        ).rstrip("/")
        self.token = get_hf_token() if token is None else token
        self._api = HfApi(endpoint=self.endpoint, token=self.token or False)
        self._timeout_s = timeout_s

    def resolve_revision(self, model_id: str) -> str | None:
        info = self._model_info(model_id, files_metadata=False)
        sha = info.sha if isinstance(info.sha, str) else None
        # Gated model metadata is public: a SHA alone does not prove file access.
        if getattr(info, "gated", False):
            self._fetch_config(model_id, sha or self.revision, self.token)
        return sha

    def _model_info(
        self, model_id: str, *, files_metadata: bool, revision: str | None = None
    ) -> Any:
        # Step 1: siblings + commit sha in ONE request.
        # CRITICAL: files_metadata=True — see module docstring.
        try:
            info = self._api.model_info(
                repo_id=model_id,
                files_metadata=files_metadata,
                token=self.token or False,
                revision=revision,
            )
        except GatedRepoError as e:
            raise AuthRequiredError(hf_auth_error_message(model_id)) from e
        except RepositoryNotFoundError as e:
            raise ModelNotFoundError(f"Model '{model_id}' not found on HuggingFace.") from e
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403):
                raise AuthRequiredError(hf_auth_error_message(model_id)) from e
            if status == 429:
                retry = e.response.headers.get("Retry-After", "unknown")
                raise SourceUnavailableError(
                    f"HuggingFace rate limit (429). Retry-After: {retry}s. "
                    "Setting HF_TOKEN increases your quota."
                ) from e
            raise SourceUnavailableError(f"HuggingFace error ({status}): {e}") from e
        except (httpx.TimeoutException, TimeoutError) as e:
            raise SourceUnavailableError(
                f"HuggingFace request timed out after {self._timeout_s}s."
            ) from e

        return info

    def fetch(self, model_id: str, revision: str | None = None) -> ModelArtifact:
        info = self._model_info(model_id, files_metadata=True, revision=revision)
        siblings = tuple(
            SiblingFile(filename=s.rfilename, size=s.size) for s in (info.siblings or [])
        )
        commit_sha = info.sha

        # Step 2: fetch config.json. If commit sha is available, pin to it so we don't
        # race with repo updates between the two calls.
        config = self._fetch_config(model_id, commit_sha or self.revision, self.token)

        return ModelArtifact(
            source=self.name,
            model_id=model_id,
            commit_sha=commit_sha,
            config=config,
            siblings=siblings,
        )

    def _fetch_config(self, model_id: str, revision: str, token: str | None) -> dict[str, Any]:
        url = f"{self.endpoint}/{model_id}/resolve/{revision}/config.json"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            resp = httpx.get(url, headers=headers, timeout=self._timeout_s, follow_redirects=True)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            raise SourceUnavailableError(f"config.json fetch failed: {e}") from e

        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model_id}' exists but has no config.json. "
                "May be a GGUF-only or dataset repo (not supported in v0.1)."
            )
        if resp.status_code in (401, 403):
            raise AuthRequiredError(hf_auth_error_message(model_id))
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After", "unknown")
            raise SourceUnavailableError(f"HuggingFace rate limit (429). Retry-After: {retry}s.")
        if resp.status_code >= 400:
            raise SourceUnavailableError(f"config.json fetch returned HTTP {resp.status_code}")

        try:
            parsed: dict[str, Any] = json.loads(resp.text)
        except json.JSONDecodeError as e:
            raise SourceUnavailableError(
                f"config.json is not valid JSON (line {e.lineno} col {e.colno}): {e.msg}"
            ) from e
        if not isinstance(parsed, dict):
            raise SourceUnavailableError("config.json did not parse to a JSON object")
        return parsed
