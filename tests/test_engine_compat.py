"""Tests for engine_compat matrix + find_match."""

from __future__ import annotations

from llm_cal.engine_compat.loader import find_match, load_matrix


def test_matrix_loads():
    m = load_matrix()
    assert m.schema_version == 2
    assert len(m.entries) > 0


def test_match_deepseek_v4_on_vllm():
    entry = find_match(engine="vllm", model_type="deepseek_v4")
    assert entry is not None
    assert entry.engine == "vllm"
    assert entry.support == "full"
    assert entry.verification_level == "cited"
    assert any("v0.19.0" in (s.url or "") for s in entry.sources)


def test_match_deepseek_v4_on_sglang_is_unverified():
    """CRITICAL: v0.1 SGLang V4 support is unverified — MUST NOT be upgraded."""
    entry = find_match(engine="sglang", model_type="deepseek_v4")
    assert entry is not None
    assert entry.verification_level == "unverified"
    assert entry.support == "unverified"


def test_match_with_specific_version():
    entry = find_match(engine="vllm", model_type="deepseek_v4", version="0.19.0")
    assert entry is not None
    assert entry.engine == "vllm"

    # Older version, no V4 support
    entry_old = find_match(engine="vllm", model_type="deepseek_v4", version="0.18.0")
    assert entry_old is None


def test_match_returns_none_for_unknown_combination():
    entry = find_match(engine="vllm", model_type="brand_new_model_type_2030")
    assert entry is None


def test_case_insensitive_matching():
    entry = find_match(engine="VLLM", model_type="DEEPSEEK_V4")
    assert entry is not None


def test_caveats_are_present_for_v4():
    entry = find_match(engine="vllm", model_type="deepseek_v4")
    assert entry is not None
    assert any("H800" in c for c in entry.caveats_en)
    assert any("H800" in c for c in entry.caveats_zh)


def test_llama_supported_on_both_engines():
    vllm_entry = find_match(engine="vllm", model_type="llama")
    sglang_entry = find_match(engine="sglang", model_type="llama")
    assert vllm_entry is not None and vllm_entry.support == "full"
    assert sglang_entry is not None and sglang_entry.support == "full"
