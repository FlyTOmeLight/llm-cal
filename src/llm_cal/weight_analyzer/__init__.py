"""Weight analyzer — observed bytes + inferred quantization scheme.

Rules:
- `[verified]` — directly from HF/ModelScope API (sum of siblings[].size). Nothing else.
- `[inferred]` — any derivation, including bits/param and quantization guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from llm_cal.model_source.base import SiblingFile
from llm_cal.output.labels import AnnotatedValue, Label

if TYPE_CHECKING:
    from llm_cal.weight_analyzer.fingerprint import QuantFingerprint

# Known byte-per-param values. bits/param = bpp * 8.
QuantizationScheme = Literal[
    "FP16",
    "BF16",
    "FP8",
    "INT8",
    "FP4_FP8_MIXED",  # DeepSeek-V4-Flash style
    "INT4",
    "GPTQ_INT4",
    "AWQ_INT4",
    "UNKNOWN",
]

# Rough bytes-per-param anchor points. Used by reconciler.
_QUANT_BPP: dict[QuantizationScheme, float] = {
    "FP16": 2.00,
    "BF16": 2.00,
    "FP8": 1.00,
    "INT8": 1.00,
    "FP4_FP8_MIXED": 0.55,  # DeepSeek V4 empirical (~4.5 bits/param)
    "INT4": 0.50,
    "GPTQ_INT4": 0.55,  # +scale tensors overhead
    "AWQ_INT4": 0.55,
    "UNKNOWN": 0.0,
}


@dataclass(frozen=True)
class WeightReport:
    """Everything the weight analyzer can determine from files + params."""

    total_bytes: AnnotatedValue[int]  # [verified]
    bits_per_param: AnnotatedValue[float] | None  # [inferred]
    quantization_guess: AnnotatedValue[QuantizationScheme]  # [inferred]


def _safetensors_total_bytes(siblings: tuple[SiblingFile, ...]) -> int:
    """Sum all *.safetensors file sizes. Ignores config, tokenizer, etc."""
    return sum((s.size or 0) for s in siblings if s.filename.endswith(".safetensors"))


def analyze(
    siblings: tuple[SiblingFile, ...],
    total_params: int | None,
    fingerprint: QuantFingerprint | None = None,
) -> WeightReport:
    """Compute weight report from sibling files + param count.

    `total_params` comes from summing across the architecture (computed elsewhere)
    or is None if we couldn't determine it — in which case we skip the inference
    step and return raw file size only.

    `fingerprint` (optional) is authoritative evidence from config.json or
    safetensors header. When present, it overrides the bpp nearest-match
    heuristic for quantization_guess (VERIFIED instead of INFERRED).
    """
    observed_bytes = _safetensors_total_bytes(siblings)
    total_bytes = AnnotatedValue(
        observed_bytes,
        Label.VERIFIED,
        source="sum of safetensors siblings from model_info API",
    )

    if not total_params or observed_bytes == 0:
        return WeightReport(
            total_bytes=total_bytes,
            bits_per_param=None,
            quantization_guess=AnnotatedValue(
                "UNKNOWN",
                Label.UNKNOWN,
                source="total_params unknown or no safetensors files",
            ),
        )

    bpp = observed_bytes / total_params
    bits_per_param = AnnotatedValue(
        bpp * 8,
        Label.INFERRED,
        source=f"{observed_bytes} bytes / {total_params} params",
    )

    if fingerprint is not None:
        quant: AnnotatedValue[QuantizationScheme] = AnnotatedValue(
            fingerprint.scheme,
            Label.VERIFIED,
            source=fingerprint.evidence,
        )
    else:
        quant = _guess_quantization(bpp)

    return WeightReport(
        total_bytes=total_bytes,
        bits_per_param=bits_per_param,
        quantization_guess=quant,
    )


def _guess_quantization(bpp: float) -> AnnotatedValue[QuantizationScheme]:
    """Nearest-match heuristic.

    Tolerance ±0.10 bits/param for mixed-precision schemes (scale tensors,
    FP16 embeddings, etc.); ±0.05 for pure schemes. See Success Criteria #2.
    """
    # Ordered so closest anchor wins on ties
    candidates: list[tuple[QuantizationScheme, float, float]] = [
        ("FP16", _QUANT_BPP["FP16"], 0.05),
        ("FP8", _QUANT_BPP["FP8"], 0.05),
        ("FP4_FP8_MIXED", _QUANT_BPP["FP4_FP8_MIXED"], 0.10),
        ("INT4", _QUANT_BPP["INT4"], 0.05),
        ("GPTQ_INT4", _QUANT_BPP["GPTQ_INT4"], 0.10),
    ]
    best: tuple[QuantizationScheme, float] | None = None
    for scheme, anchor_bpp, tolerance in candidates:
        delta = abs(bpp - anchor_bpp)
        if delta <= tolerance and (best is None or delta < best[1]):
            best = (scheme, delta)

    if best is None:
        return AnnotatedValue(
            "UNKNOWN",
            Label.UNKNOWN,
            source=f"bits/param {bpp * 8:.2f} does not match known schemes",
        )
    return AnnotatedValue(
        best[0],
        Label.INFERRED,
        source=f"bits/param {bpp * 8:.2f} within tolerance of {best[0]}",
    )
