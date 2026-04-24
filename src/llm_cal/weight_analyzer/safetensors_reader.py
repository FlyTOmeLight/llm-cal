"""Fetch the safetensors header of one shard to recover per-tensor dtypes.

The safetensors binary format:
  bytes[0..8]       uint64 little-endian  header length N (JSON bytes)
  bytes[8..8+N]     UTF-8 JSON            tensor_name -> {dtype, shape, data_offsets}
  bytes[8+N..]      raw tensor data       (we never read this)

So we can identify every tensor's dtype without downloading any weight bytes.
Headers are usually 50 KB - 2 MB. We cap the Range request at 16 MB as a
safety net; anything larger is treated as malformed.

This module NEVER raises on network or parse error — it returns None so
the caller can degrade gracefully. The honesty principle: "we tried and
failed to resolve the tie" is a legitimate outcome, not a fatal error.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import httpx

from llm_cal.model_source.auth import get_hf_token
from llm_cal.model_source.base import SiblingFile

_MAX_HEADER_BYTES = 16 * 1024 * 1024  # 16 MB — far above any realistic header
_RANGE_FETCH_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 15.0


def pick_sample_shard(siblings: tuple[SiblingFile, ...]) -> SiblingFile | None:
    """Choose one safetensors file that's representative of the model.

    Preference order:
      1. `model.safetensors` (single-file case — always representative)
      2. First `model-00001-of-NNNNN.safetensors` shard (contains layer-0
         which has all weight tensor name patterns we need)
      3. Any `*.safetensors`
    """
    st_files = [s for s in siblings if s.filename.endswith(".safetensors")]
    if not st_files:
        return None

    for s in st_files:
        if s.filename == "model.safetensors":
            return s

    first_shard = sorted(st_files, key=lambda s: s.filename)[0]
    return first_shard


def fetch_tensor_dtypes(
    source: str,
    model_id: str,
    revision: str,
    shard_filename: str,
    endpoint: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, str] | None:
    """Range-fetch the safetensors header of one shard and return dtype map.

    Returns a dict of `{tensor_name: dtype_string}` on success, None on any
    failure (network, parse, unexpected format). Non-fatal by design.

    Only HuggingFace is supported in v0.1.2 — ModelScope URL scheme is
    different and its tie-breaking path falls back to "no fingerprint".
    """
    if source != "huggingface":
        return None

    base = (endpoint or "https://huggingface.co").rstrip("/")
    url = f"{base}/{model_id}/resolve/{revision}/{shard_filename}"

    token = get_hf_token()
    headers: dict[str, str] = {"Range": f"bytes=0-{_RANGE_FETCH_BYTES - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = httpx.get(url, headers=headers, timeout=timeout_s, follow_redirects=True)
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
        return None

    # 200 for small files returned in full; 206 for actual Range response.
    # Anything else (404, 403, 500, ...) we degrade silently.
    if resp.status_code not in (200, 206):
        return None

    content = resp.content
    return parse_header(content)


def parse_header(content: bytes) -> dict[str, str] | None:
    """Parse the safetensors binary header from a leading byte buffer.

    Pure function — safe to call on any bytes. Returns None on any malformed
    input rather than raising.
    """
    if len(content) < 8:
        return None

    try:
        (header_len,) = struct.unpack("<Q", content[:8])
    except struct.error:
        return None

    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
        return None

    if len(content) < 8 + header_len:
        return None

    header_bytes = content[8 : 8 + header_len]
    try:
        header: Any = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(header, dict):
        return None

    dtypes: dict[str, str] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            continue
        dtype = info.get("dtype")
        if isinstance(dtype, str):
            dtypes[name] = dtype

    return dtypes if dtypes else None
