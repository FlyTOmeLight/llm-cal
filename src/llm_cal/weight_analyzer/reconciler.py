"""Reconciler — compare observed weight bytes vs computed under each quantization assumption.

This is the module that outputs the DeepSeek-V4-Flash story (Problem Evidence in design doc):
"gpu_poor says 285 GB (assumes pure FP8); we say 160 GB (observed bytes match FP4+FP8
 pack hypothesis). Here's why."

Core value: makes the quantization inference step transparent. The user sees all
candidates considered, not just the winner.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_cal.output.labels import AnnotatedValue, Label
from llm_cal.weight_analyzer import _QUANT_BPP, QuantizationScheme


@dataclass(frozen=True)
class ReconciliationCandidate:
    scheme: QuantizationScheme
    predicted_bytes: int
    delta_bytes: int  # observed - predicted (positive = observed is larger)
    relative_error: float  # |delta| / predicted


@dataclass(frozen=True)
class ReconciliationReport:
    observed_bytes: int
    total_params: int
    candidates: tuple[ReconciliationCandidate, ...]  # sorted by |relative_error| asc
    best: AnnotatedValue[QuantizationScheme]

    def summary_line(self) -> str:
        """One-liner for output formatter."""
        if not self.candidates:
            return f"{self.observed_bytes:,} bytes — no quantization candidates tested"
        c = self.candidates[0]
        return (
            f"Observed {self.observed_bytes:,} bytes. "
            f"Best match: {c.scheme} "
            f"(predicts {c.predicted_bytes:,} bytes, "
            f"{c.relative_error * 100:.1f}% error)"
        )


def reconcile(observed_bytes: int, total_params: int) -> ReconciliationReport:
    """Compare observed file bytes against every known quantization scheme.

    Returns full ranking so the formatter can show "gpu_poor would say X; we say Y."
    """
    if observed_bytes == 0 or total_params == 0:
        return ReconciliationReport(
            observed_bytes=observed_bytes,
            total_params=total_params,
            candidates=(),
            best=AnnotatedValue(
                "UNKNOWN",
                Label.UNKNOWN,
                source="observed_bytes or total_params is zero",
            ),
        )

    candidates = []
    for scheme, bpp in _QUANT_BPP.items():
        if scheme == "UNKNOWN" or bpp == 0.0:
            continue
        predicted = int(bpp * total_params)
        delta = observed_bytes - predicted
        rel_err = abs(delta) / predicted if predicted else float("inf")
        candidates.append(
            ReconciliationCandidate(
                scheme=scheme,
                predicted_bytes=predicted,
                delta_bytes=delta,
                relative_error=rel_err,
            )
        )
    candidates.sort(key=lambda c: c.relative_error)

    best_scheme = candidates[0].scheme
    best_err = candidates[0].relative_error

    # Tolerance gate: if even the best match is off by >15%, call it UNKNOWN.
    if best_err > 0.15:
        return ReconciliationReport(
            observed_bytes=observed_bytes,
            total_params=total_params,
            candidates=tuple(candidates),
            best=AnnotatedValue(
                "UNKNOWN",
                Label.UNKNOWN,
                source=(
                    f"closest candidate ({candidates[0].scheme}) is off by "
                    f"{best_err * 100:.1f}% — no confident match"
                ),
            ),
        )

    return ReconciliationReport(
        observed_bytes=observed_bytes,
        total_params=total_params,
        candidates=tuple(candidates),
        best=AnnotatedValue(
            best_scheme,
            Label.INFERRED,
            source=(f"best match among {len(candidates)} candidates, {best_err * 100:.1f}% error"),
        ),
    )
