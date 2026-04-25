"""Tests for safetensors header Range-fetching + parsing.

Network calls are mocked. The parse_header tests run on synthetic byte
buffers that mirror the real safetensors binary layout.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import patch

import httpx

from llm_cal.model_source.base import SiblingFile
from llm_cal.weight_analyzer.safetensors_reader import (
    fetch_tensor_dtypes,
    parse_header,
    pick_sample_shard,
)


def _build_safetensors_bytes(header: dict) -> bytes:
    """Encode a safetensors-format byte buffer from a header dict."""
    header_bytes = json.dumps(header).encode("utf-8")
    length_prefix = struct.pack("<Q", len(header_bytes))
    # Append 100 bytes of "fake tensor data" so total length looks realistic
    return length_prefix + header_bytes + b"\x00" * 100


class TestPickSampleShard:
    def test_prefers_single_file(self):
        siblings = (
            SiblingFile("model-00001-of-00010.safetensors", 1000),
            SiblingFile("model.safetensors", 100),  # single-file wins
            SiblingFile("config.json", 10),
        )
        pick = pick_sample_shard(siblings)
        assert pick is not None
        assert pick.filename == "model.safetensors"

    def test_picks_middle_shard(self):
        """Middle shards have decoder layers / MoE experts; first usually has
        embeddings + early-layer norms which often stay BF16/FP16 even in
        quantized models. Sampling the first shard misled the FP4+FP8 detection
        for DeepSeek-V4-Flash — middle is the honest representative."""
        siblings = (
            SiblingFile("model-00003-of-00010.safetensors", 1000),
            SiblingFile("model-00001-of-00010.safetensors", 1000),
            SiblingFile("model-00002-of-00010.safetensors", 1000),
        )
        pick = pick_sample_shard(siblings)
        assert pick is not None
        # 3 shards sorted: 00001, 00002, 00003 → middle index 1 → 00002
        assert pick.filename == "model-00002-of-00010.safetensors"

    def test_picks_middle_shard_even_count(self):
        siblings = tuple(
            SiblingFile(f"model-{i:05d}-of-00046.safetensors", 1000) for i in range(1, 47)
        )
        pick = pick_sample_shard(siblings)
        assert pick is not None
        # 46 shards, middle index = 23 → "00024"
        assert pick.filename == "model-00024-of-00046.safetensors"

    def test_no_safetensors(self):
        siblings = (
            SiblingFile("pytorch_model.bin", 1000),
            SiblingFile("config.json", 10),
        )
        assert pick_sample_shard(siblings) is None


class TestParseHeader:
    def test_well_formed_header(self):
        header = {
            "__metadata__": {"format": "pt"},
            "model.layers.0.weight": {
                "dtype": "F8_E4M3",
                "shape": [4096, 4096],
                "data_offsets": [0, 16777216],
            },
            "model.norm.weight": {
                "dtype": "BF16",
                "shape": [4096],
                "data_offsets": [16777216, 16785408],
            },
        }
        dtypes = parse_header(_build_safetensors_bytes(header))
        assert dtypes == {
            "model.layers.0.weight": "F8_E4M3",
            "model.norm.weight": "BF16",
        }

    def test_too_short(self):
        # Less than 8 bytes — can't even read header length
        assert parse_header(b"abc") is None

    def test_header_length_exceeds_buffer(self):
        """Claimed header length > actual bytes available."""
        # Claim 10_000_000 bytes of header but only provide 100
        buf = struct.pack("<Q", 10_000_000) + b"{}" * 50
        assert parse_header(buf) is None

    def test_header_length_absurdly_large(self):
        """Claimed length > our safety cap."""
        buf = struct.pack("<Q", 100 * 1024 * 1024) + b"{}"
        assert parse_header(buf) is None

    def test_zero_header_length(self):
        buf = struct.pack("<Q", 0) + b"{}"
        assert parse_header(buf) is None

    def test_malformed_json(self):
        bad_json = b"{ not valid json"
        buf = struct.pack("<Q", len(bad_json)) + bad_json + b"\x00" * 10
        assert parse_header(buf) is None

    def test_header_not_dict(self):
        """JSON parses but isn't a dict."""
        buf = _build_safetensors_bytes({})  # valid empty dict
        # Empty dict → no tensor dtypes found → None
        assert parse_header(buf) is None

    def test_skips_metadata_key(self):
        """The __metadata__ key should be excluded from dtype map."""
        header = {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F16", "shape": [10], "data_offsets": [0, 20]},
        }
        dtypes = parse_header(_build_safetensors_bytes(header))
        assert dtypes == {"weight": "F16"}

    def test_entry_missing_dtype(self):
        """Tensor entries without a dtype field are silently skipped."""
        header = {
            "a": {"dtype": "F16", "shape": [10], "data_offsets": [0, 20]},
            "b": {"shape": [10], "data_offsets": [20, 40]},  # no dtype
        }
        dtypes = parse_header(_build_safetensors_bytes(header))
        assert dtypes == {"a": "F16"}


class TestFetchTensorDtypes:
    def test_happy_path_http_206(self):
        header = {
            "layer.weight": {"dtype": "F8_E4M3", "shape": [10], "data_offsets": [0, 10]},
        }
        buf = _build_safetensors_bytes(header)

        fake_resp = httpx.Response(status_code=206, content=buf)

        with patch("httpx.get", return_value=fake_resp):
            dtypes = fetch_tensor_dtypes(
                source="huggingface",
                model_id="foo/bar",
                revision="main",
                shard_filename="model.safetensors",
            )
        assert dtypes == {"layer.weight": "F8_E4M3"}

    def test_happy_path_http_200(self):
        """Small file returned in full without Range header."""
        header = {"x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        fake_resp = httpx.Response(status_code=200, content=_build_safetensors_bytes(header))

        with patch("httpx.get", return_value=fake_resp):
            dtypes = fetch_tensor_dtypes(
                source="huggingface",
                model_id="a/b",
                revision="deadbeef",
                shard_filename="model.safetensors",
            )
        assert dtypes == {"x": "F16"}

    def test_modelscope_happy_path(self):
        """ModelScope: same Range GET, different URL + auth header."""
        header = {"layer.weight": {"dtype": "F4_E2M1", "shape": [10], "data_offsets": [0, 10]}}
        buf = _build_safetensors_bytes(header)

        captured: dict = {}

        def _capture(url, *args, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return httpx.Response(status_code=206, content=buf)

        with patch("httpx.get", side_effect=_capture):
            dtypes = fetch_tensor_dtypes(
                source="modelscope",
                model_id="Qwen/Qwen3-30B-A3B",
                revision="master",
                shard_filename="model-00003-of-00163.safetensors",
            )
        assert dtypes == {"layer.weight": "F4_E2M1"}
        assert "modelscope.cn" in captured["url"]
        assert "FilePath=model-00003-of-00163.safetensors" in captured["url"]
        # Range header is always set on the request
        assert captured["headers"].get("Range", "").startswith("bytes=0-")

    def test_unknown_source_returns_none(self):
        """Sources we don't recognize fall through silently."""
        assert (
            fetch_tensor_dtypes(
                source="some-future-mirror",
                model_id="foo",
                revision="main",
                shard_filename="model.safetensors",
            )
            is None
        )

    def test_timeout_returns_none(self):
        def _raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        with patch("httpx.get", side_effect=_raise_timeout):
            dtypes = fetch_tensor_dtypes(
                source="huggingface",
                model_id="a/b",
                revision="main",
                shard_filename="model.safetensors",
            )
        assert dtypes is None

    def test_404_returns_none(self):
        fake_resp = httpx.Response(status_code=404, content=b"Not found")
        with patch("httpx.get", return_value=fake_resp):
            dtypes = fetch_tensor_dtypes(
                source="huggingface",
                model_id="a/b",
                revision="main",
                shard_filename="model.safetensors",
            )
        assert dtypes is None

    def test_connect_error_returns_none(self):
        def _raise_conn(*args, **kwargs):
            raise httpx.ConnectError("refused")

        with patch("httpx.get", side_effect=_raise_conn):
            dtypes = fetch_tensor_dtypes(
                source="huggingface",
                model_id="a/b",
                revision="main",
                shard_filename="model.safetensors",
            )
        assert dtypes is None
