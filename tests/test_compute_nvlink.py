"""Tests for the NVLink-aware decode penalty in performance.compute.

Why this matters: H100 and H800 share HBM, FLOPS, and memory bandwidth — they
differ ONLY in NVLink bandwidth (900 vs 400 GB/s). Without an NVLink penalty,
the calculator returns identical decode throughput for both, which surprised
users running side-by-side comparisons. The penalty differentiates them on
multi-GPU TP clusters while leaving single-GPU runs untouched.
"""

from __future__ import annotations

import pytest

from llm_cal.architecture.profile import ArchitectureProfile, Confidence, Family
from llm_cal.hardware.loader import GPUSpec
from llm_cal.performance.compute import _nvlink_efficiency, estimate_decode


def _gpu(*, nvlink: int, mem_bw: int = 3350, mem_gb: int = 80, tflops: float = 989) -> GPUSpec:
    return GPUSpec(
        id=f"test-nvlink-{nvlink}",
        memory_gb=mem_gb,
        nvlink_bandwidth_gbps=nvlink,
        memory_bandwidth_gbps=mem_bw,
        fp16_tflops=tflops,
        fp8_support=True,
        fp4_support=False,
    )


def _dense_profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        model_type="llama",
        architectures=("LlamaForCausalLM",),
        family=Family.TRANSFORMER,
        num_hidden_layers=32,
        hidden_size=4096,
        vocab_size=128000,
        confidence=Confidence.HIGH,
    )


# ----------------------------------------------------------------------- helper


class TestNvlinkEfficiencyHelper:
    def test_single_gpu_no_penalty(self):
        """1-GPU has no TP all-reduce, so NVLink is irrelevant."""
        assert _nvlink_efficiency(_gpu(nvlink=0), num_gpus=1) == 1.0
        assert _nvlink_efficiency(_gpu(nvlink=400), num_gpus=1) == 1.0
        assert _nvlink_efficiency(_gpu(nvlink=900), num_gpus=1) == 1.0

    def test_h100_class_baseline_no_penalty(self):
        """NVLink >= 900 GB/s (H100, B200, H200) is the reference: 1.0."""
        assert _nvlink_efficiency(_gpu(nvlink=900), num_gpus=8) == 1.0
        assert _nvlink_efficiency(_gpu(nvlink=1800), num_gpus=8) == 1.0  # B200

    def test_h800_pays_partial_penalty(self):
        """H800 (400 GB/s) sits between PCIe and H100. ~92% efficient."""
        eff = _nvlink_efficiency(_gpu(nvlink=400), num_gpus=8)
        assert 0.91 < eff < 0.92  # 0.85 + 0.15 * 400/900 ≈ 0.9167

    def test_pcie_only_heavy_penalty(self):
        """No NVLink (L40S, RTX) → 0.80 (20% penalty on TP all-reduce)."""
        assert _nvlink_efficiency(_gpu(nvlink=0), num_gpus=8) == 0.80

    def test_a100_class_mild_penalty(self):
        """A100 SXM4 has 600 GB/s NVLink → 0.95 exactly (0.85 + 0.15·2/3)."""
        eff = _nvlink_efficiency(_gpu(nvlink=600), num_gpus=8)
        assert eff == pytest.approx(0.95)

    def test_monotonic_in_bandwidth(self):
        """Higher NVLink bandwidth must never decrease efficiency."""
        prev = -1.0
        for bw in [0, 100, 200, 400, 600, 800, 900, 1800]:
            eff = _nvlink_efficiency(_gpu(nvlink=bw), num_gpus=8)
            assert eff >= prev, f"non-monotonic at NVLink={bw}: {eff} < {prev}"
            prev = eff


# ---------------------------------------------------------- end-to-end via decode


class TestEstimateDecodeAppliesPenalty:
    def test_h100_vs_h800_differ_on_multi_gpu(self):
        """The headline test: H100 and H800 must produce DIFFERENT cluster
        decode throughput when num_gpus > 1, since that's the whole point
        of this work."""
        profile = _dense_profile()
        weight_bytes = 80 * 1024**3  # 80 GB
        h100 = estimate_decode(profile, weight_bytes, _gpu(nvlink=900), num_gpus=8)
        h800 = estimate_decode(profile, weight_bytes, _gpu(nvlink=400), num_gpus=8)
        assert h100.cluster_tokens_per_sec.value > h800.cluster_tokens_per_sec.value
        # H800 should be ~91.7% of H100 (the NVLink penalty ratio)
        ratio = h800.cluster_tokens_per_sec.value / h100.cluster_tokens_per_sec.value
        assert 0.91 < ratio < 0.92

    def test_h100_vs_h800_identical_on_single_gpu(self):
        """Single-GPU decode has no TP comm — H100 and H800 must match."""
        profile = _dense_profile()
        weight_bytes = 14 * 1024**3
        h100 = estimate_decode(profile, weight_bytes, _gpu(nvlink=900), num_gpus=1)
        h800 = estimate_decode(profile, weight_bytes, _gpu(nvlink=400), num_gpus=1)
        assert h100.cluster_tokens_per_sec.value == pytest.approx(
            h800.cluster_tokens_per_sec.value
        )

    def test_pcie_only_drops_cluster_tps(self):
        """L40S-class with no NVLink should land below H100 on multi-GPU."""
        profile = _dense_profile()
        weight_bytes = 80 * 1024**3
        h100 = estimate_decode(profile, weight_bytes, _gpu(nvlink=900), num_gpus=8)
        pcie = estimate_decode(profile, weight_bytes, _gpu(nvlink=0), num_gpus=8)
        ratio = pcie.cluster_tokens_per_sec.value / h100.cluster_tokens_per_sec.value
        assert 0.79 < ratio < 0.81

    def test_per_gpu_tps_unaffected(self):
        """Per-GPU decode is purely memory-bandwidth bound. NVLink must not
        leak into per-GPU numbers, only into cluster aggregation."""
        profile = _dense_profile()
        weight_bytes = 80 * 1024**3
        h100 = estimate_decode(profile, weight_bytes, _gpu(nvlink=900), num_gpus=8)
        h800 = estimate_decode(profile, weight_bytes, _gpu(nvlink=400), num_gpus=8)
        assert h100.per_gpu_tokens_per_sec.value == pytest.approx(
            h800.per_gpu_tokens_per_sec.value
        )

    def test_provenance_string_mentions_nvlink(self):
        """Source string should let users see the NVLink penalty was applied."""
        profile = _dense_profile()
        h800 = estimate_decode(
            profile, 80 * 1024**3, _gpu(nvlink=400), num_gpus=8
        )
        assert "NVLink" in h800.cluster_tokens_per_sec.source
        assert "400" in h800.cluster_tokens_per_sec.source
