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
    expected = {
        # NVIDIA Blackwell / Hopper
        "B200",
        "H100",
        "H800",
        "H200",
        "H20",
        # NVIDIA Ada
        "L40S",
        "L40",
        "L4",
        "RTX6000-Ada",
        "RTX4090",
        # NVIDIA Ampere
        "A100-80G",
        "A100-40G",
        "A40",
        "A10",
        "A10G",
        # NVIDIA Volta / Turing
        "V100-SXM2-32G",
        "T4",
        # AMD
        "MI300X",
        "MI250X",
        # Intel Habana
        "Gaudi3",
        "Gaudi2",
        # Huawei Ascend
        "910A",
        "910B1",
        "910B2",
        "910B3",
        "910B4",
        "910C",
        "Atlas-300I-Duo",
        # Chinese domestic (non-Huawei)
        "MXC500",
        "MXC550",
        "Kunlun-P800",
        "Kunlun-R200",
        "BR100",
        "BR104",
        "BI-V100",
        "MR-V100",
        "MLU370-X8",
        "MTT-S4000",
        "MTT-S3000",
        # More international
        "GH200",
        "GB200",
        "MI325X",
        "MI210",
        "V100-PCIe-32G",
        # More Chinese domestic
        "MLU590",
        "Hygon-K100-AI",
        "Hygon-Z100",
    }
    assert expected <= ids, f"missing: {expected - ids}"


def test_910b_bare_alias_resolves_to_910b3():
    """Bare `910B` id should resolve to 910B3 (most common training variant)."""
    spec = lookup("910B")
    assert spec.id == "910B3"
    assert spec.memory_gb == 64
    assert spec.fp16_tflops == 313


def test_910b4_is_inference_variant():
    """910B4 has half the memory (32 vs 64) — inference-oriented Atlas 800I A2."""
    spec = lookup("910B4")
    assert spec.memory_gb == 32
    assert spec.fp16_tflops == 280


def test_chinese_domestic_cards_present():
    """Spot-check the four Chinese vendors beyond Huawei."""
    assert lookup("MXC500").memory_gb == 64
    assert lookup("Kunlun-P800").memory_gb == 96  # largest Chinese HBM in v0.1
    assert lookup("BR100").fp16_tflops >= 1000  # PFLOPS-class BF16/FP16
    assert lookup("BI-V100").memory_gb == 32


def test_kunlun_p800_supports_fp8():
    """P800 is the only Chinese domestic card in v0.1 with FP8 support."""
    spec = lookup("Kunlun-P800")
    assert spec.fp8_support is True


def test_chinese_name_aliases_resolve():
    """Chinese names should work as aliases."""
    assert lookup("曦云C500").id == "MXC500"
    assert lookup("昆仑芯P800").id == "Kunlun-P800"
    assert lookup("壁仞BR100").id == "BR100"
    assert lookup("天数天垓100").id == "BI-V100"
    assert lookup("天数智铠100").id == "MR-V100"
    assert lookup("寒武纪MLU370-X8").id == "MLU370-X8"
    assert lookup("摩尔线程S4000").id == "MTT-S4000"


def test_gh200_hopper_plus_grace():
    """GH200 has 144 GB HBM3e (unique superchip design)."""
    spec = lookup("GH200")
    assert spec.memory_gb == 144
    assert spec.fp8_support is True
    assert spec.fp4_support is False


def test_mi210_no_fp8():
    """MI210 is CDNA 2 — no native FP8 support."""
    spec = lookup("MI210")
    assert spec.fp8_support is False
    assert spec.memory_gb == 64


def test_list_gpus_subcommand():
    """CLI --list-gpus should run without requiring model_id/gpu."""
    from typer.testing import CliRunner

    from llm_cal.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--list-gpus"])
    assert result.exit_code == 0
    assert "H100" in result.stdout
    assert "MI325X" in result.stdout
    # Should mention total count (text differs by locale; check both)
    assert "Total" in result.stdout or "共 " in result.stdout


def test_every_gpu_has_spec_source():
    """Honesty constraint: every spec must be traceable to a source.

    If this test fails, someone added a GPU without citing where the numbers
    came from. That violates the tool's label-discipline philosophy.
    """
    db = load_database()
    missing = [g.id for g in db.gpus if not g.spec_source]
    assert not missing, (
        f"GPUs without spec_source: {missing}. "
        "Add a spec_source field citing the vendor datasheet or benchmark URL."
    )


def test_h20_recognized():
    """H20 is a critical entry for the China market post-2023 export controls."""
    spec = lookup("H20")
    assert spec.id == "H20"
    assert spec.memory_gb == 96
    assert spec.fp8_support is True
    assert spec.fp16_tflops < 200  # Heavily throttled vs H100's 989


def test_mi325x_is_biggest_single_card():
    """MI325X (256 GB) is the largest single-card memory in v0.1 database."""
    db = load_database()
    largest = max(db.gpus, key=lambda g: g.memory_gb)
    assert largest.id == "MI325X"
    assert largest.memory_gb == 256


def test_mlu590_vs_mlu370_progression():
    """MLU590 should have more memory (80 GB vs 48 GB) and more compute than MLU370-X8."""
    m590 = lookup("MLU590")
    m370 = lookup("MLU370-X8")
    assert m590.memory_gb > m370.memory_gb
    assert m590.fp16_tflops > m370.fp16_tflops


def test_hygon_cards_use_rocm_stack():
    """Hygon cards should be findable via Chinese name."""
    assert lookup("海光K100AI").id == "Hygon-K100-AI"
    assert lookup("深算二号").id == "Hygon-Z100"


def test_gb200_has_fp4_like_b200():
    """GB200 (per-GPU view) inherits B200's FP4 capability."""
    spec = lookup("GB200")
    assert spec.fp4_support is True
    assert spec.memory_gb == 192


def test_t4_has_no_fp8_no_nvlink():
    """Sanity: cheapest cloud option shouldn't accidentally claim FP8."""
    spec = lookup("T4")
    assert spec.fp8_support is False
    assert spec.nvlink_bandwidth_gbps == 0


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
