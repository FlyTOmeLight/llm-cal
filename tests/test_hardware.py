"""Tests for hardware database + GPU lookup."""

from __future__ import annotations

import pytest

from llm_cal.hardware.loader import (
    UnknownGPUError,
    load_database,
    lookup,
)


def test_database_loads_and_has_all_expected_gpus():
    db = load_database()
    ids = {g.id for g in db.gpus}
    assert {"H100", "H800", "H200", "A100-80G", "A100-40G", "B200", "910B", "RTX4090"} <= ids


def test_lookup_exact_id():
    spec = lookup("H800")
    assert spec.id == "H800"
    assert spec.memory_gb == 80
    assert spec.nvlink_bandwidth_gbps == 400  # halved vs H100
    assert spec.fp4_support is False  # Hopper doesn't have FP4


def test_lookup_alias():
    # H800-SXM5 is an alias, should resolve to the H800 entry
    spec = lookup("H800-SXM5")
    assert spec.id == "H800"


def test_lookup_case_insensitive():
    spec = lookup("h800")
    assert spec.id == "H800"


def test_lookup_unknown_gpu_helpful_message():
    with pytest.raises(UnknownGPUError) as e:
        lookup("H999")
    assert "H999" in str(e.value)
    # Error lists known GPUs so user can correct
    assert "H800" in str(e.value)


def test_lookup_rejects_h800x8_legacy_format():
    """Old 'H800x8' format was before we split into --gpu + --gpu-count."""
    with pytest.raises(UnknownGPUError) as e:
        lookup("H800x8")
    assert "--gpu-count 8" in str(e.value)


def test_b200_has_fp4_support():
    """Blackwell is the first GPU that hardware-accelerates FP4."""
    spec = lookup("B200")
    assert spec.fp4_support is True
    assert spec.fp8_support is True


def test_localized_notes():
    spec = lookup("H800")
    en = spec.localized_notes("en")
    zh = spec.localized_notes("zh")
    assert en is not None and "regulated" in en.lower()
    assert zh is not None and "合规" in zh
