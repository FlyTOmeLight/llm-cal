"""ModelScope source — REST-only via httpx.

Decision: Option B from ADR-001. We don't need the official `modelscope` SDK
because llm-cal only requires three things:
  1. List repo files + sizes (one API call)
  2. Fetch config.json (one API call)
  3. Range-GET a safetensors header (handled by safetensors_reader)

The SDK pulls heavy ML deps by default (torch / tf for some install paths).
REST keeps the install footprint flat, mirrors the existing httpx hot path,
and gives us identical exception semantics across HF + MS.

Endpoints (verified against modelscope.cn public docs, 2026-04):
  * GET /api/v1/models/{owner}/{name}                       — model meta
  * GET /api/v1/models/{owner}/{name}/repo/files?Recursive=true
                                                            — file tree + sizes
  * GET /api/v1/models/{owner}/{name}/repo?FilePath=...&Revision=...
                                                            — raw file content

ModelScope wraps every response in a {Code, Message, Data, Success} envelope.
Field casing is PascalCase. We parse defensively — fields may evolve.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from llm_cal.model_source.auth import (
    get_modelscope_token,
    modelscope_auth_error_message,
)
from llm_cal.model_source.base import (
    AuthRequiredError,
    ModelArtifact,
    ModelNotFoundError,
    ModelSource,
    SiblingFile,
    SourceUnavailableError,
)

DEFAULT_ENDPOINT = "https://www.modelscope.cn"
DEFAULT_REVISION = "master"

_INFO_PATH = "/api/v1/models/{model_id}"
_FILES_PATH = "/api/v1/models/{model_id}/repo/files"
_RAW_PATH = "/api/v1/models/{model_id}/repo"


class ModelScopeSource(ModelSource):
    name = "modelscope"

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float = 30.0,
        revision: str = DEFAULT_REVISION,
    ) -> None:
        self._endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self._timeout_s = timeout_s
        self._revision = revision

    def fetch(self, model_id: str) -> ModelArtifact:
        token = get_modelscope_token()
        headers = self._auth_headers(token)

        # Step 1: model info — gives us LatestSha (commit pin) when available.
        # We tolerate missing info; fall back to revision="master" so that the
        # file list + config calls still work.
        commit_sha = self._fetch_commit_sha(model_id, headers)

        # Step 2: file tree with sizes. ONE call, recursive, includes sub-folders.
        siblings = self._list_files(model_id, commit_sha or self._revision, headers)

        # Step 3: config.json. Pin to the commit sha when we have it so two
        # back-to-back calls don't race against a repo update.
        config = self._fetch_config(model_id, commit_sha or self._revision, headers)

        return ModelArtifact(
            source=self.name,
            model_id=model_id,
            commit_sha=commit_sha,
            config=config,
            siblings=siblings,
        )

    # ------------------------------------------------------------------ helpers

    def _auth_headers(self, token: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _fetch_commit_sha(self, model_id: str, headers: dict[str, str]) -> str | None:
        url = f"{self._endpoint}{_INFO_PATH.format(model_id=model_id)}"
        try:
            resp = httpx.get(
                url, headers=headers, timeout=self._timeout_s, follow_redirects=True
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
            # Soft fail — commit sha is best-effort. Caller will use "master".
            return None

        if resp.status_code != 200:
            return None
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return None

        data = payload.get("Data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None
        # Field name has bounced between LatestSha / latest_sha / Revision in
        # historical docs; check several.
        for key in ("LatestSha", "latest_sha", "Revision", "Sha"):
            v = data.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    def _list_files(
        self, model_id: str, revision: str, headers: dict[str, str]
    ) -> tuple[SiblingFile, ...]:
        url = f"{self._endpoint}{_FILES_PATH.format(model_id=model_id)}"
        params = {"Recursive": "true", "Revision": revision}
        try:
            resp = httpx.get(
                url,
                headers=headers,
                params=params,
                timeout=self._timeout_s,
                follow_redirects=True,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            raise SourceUnavailableError(f"ModelScope file list failed: {e}") from e

        self._raise_for_status(resp, model_id, what="file list")

        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise SourceUnavailableError(
                f"ModelScope file list returned non-JSON: {e}"
            ) from e

        files = _extract_files(payload)
        if files is None:
            raise SourceUnavailableError(
                "ModelScope file list payload had unexpected shape — "
                "neither Data.Files nor Data is a list."
            )
        return tuple(
            SiblingFile(filename=f["Path"], size=f.get("Size"))
            for f in files
            if isinstance(f, dict) and isinstance(f.get("Path"), str)
            # Only include blobs (not directories). Type=tree means folder.
            and f.get("Type", "blob") != "tree"
        )

    def _fetch_config(
        self, model_id: str, revision: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{_RAW_PATH.format(model_id=model_id)}"
        params = {"FilePath": "config.json", "Revision": revision}
        try:
            resp = httpx.get(
                url,
                headers=headers,
                params=params,
                timeout=self._timeout_s,
                follow_redirects=True,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            raise SourceUnavailableError(f"config.json fetch failed: {e}") from e

        self._raise_for_status(resp, model_id, what="config.json")

        try:
            parsed: Any = json.loads(resp.text)
        except json.JSONDecodeError as e:
            raise SourceUnavailableError(
                f"config.json is not valid JSON (line {e.lineno} col {e.colno}): {e.msg}"
            ) from e
        if not isinstance(parsed, dict):
            raise SourceUnavailableError(
                "config.json did not parse to a JSON object."
            )
        return parsed

    def _raise_for_status(
        self, resp: httpx.Response, model_id: str, what: str
    ) -> None:
        if resp.status_code == 200:
            return
        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found on ModelScope ({what})."
            )
        if resp.status_code in (401, 403):
            raise AuthRequiredError(modelscope_auth_error_message(model_id))
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After", "unknown")
            raise SourceUnavailableError(
                f"ModelScope rate limit (429). Retry-After: {retry}s. "
                "Setting MODELSCOPE_API_TOKEN increases your quota."
            )
        raise SourceUnavailableError(
            f"ModelScope {what} returned HTTP {resp.status_code}"
        )


def _extract_files(payload: Any) -> list[Any] | None:
    """Pull the file list out of the wrapped ModelScope envelope.

    Tolerates two known shapes:
      A) {Data: {Files: [...]}}      — most common
      B) {Data: [...]}               — older / list-only endpoints
    """
    if not isinstance(payload, dict):
        return None
    data = payload.get("Data")
    if isinstance(data, dict):
        files = data.get("Files")
        if isinstance(files, list):
            return files
    if isinstance(data, list):
        return data
    return None
