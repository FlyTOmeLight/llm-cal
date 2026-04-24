"""Reconciler — compare observed weight bytes vs computed under each quantization assumption.

This is the module that outputs the DeepSeek-V4-Flash story (Problem Evidence in design doc):
"gpu_poor says 285 GB (assumes pure FP8); we say 160 GB (observed bytes match FP4+FP8
 pack hypothesis). Here's why."

Core value: makes the quantization inference step transparent. The user sees all
candidates considered, not just the winner.

When multiple schemes share the same bytes-per-param anchor (FP4_FP8_MIXED,
GPTQ_INT4, and AWQ_INT4 all sit at bpp=0.55), bytes alone cannot pick a winner.
Pass a `QuantFingerprint` from `fingerprint.from_config()` or
`fingerprint.from_safetensors_dtypes()` to break the tie with authoritative
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_cal.output.labels import AnnotatedValue, Label
from llm_cal.weight_analyzer import _QUANT_BPP, QuantizationScheme
from llm_cal.weight_analyzer.fingerprint import QuantFingerprint


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


# Tolerance for tie detection — schemes within this relative-error delta of the
# winner are considered tied.
_TIE_THRESHOLD = 0.01

# Tolerance gate — if the closest candidate is off by more than this, call UNKNOWN.
_UNKNOWN_THRESHOLD = 0.15


def reconcile(
    observed_bytes: int,
    total_params: int,
    fingerprint: QuantFingerprint | None = None,
) -> ReconciliationReport:
    """Compare observed file bytes against every known quantization scheme.

    Args:
        observed_bytes: Sum of safetensors file sizes.
        total_params: Estimated param count.
        fingerprint: Optional authoritative evidence from config.json or
            safetensors header. Breaks bpp ties and annotates the source.

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

    candidates: list[ReconciliationCandidate] = []
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

    argmin_scheme = candidates[0].scheme
    argmin_err = candidates[0].relative_error

    # Fingerprint path: authoritative declaration from config.json or safetensors
    # header. This is the primary fix for the tie that LLM review caught.
    if fingerprint is not None:
        return _reconcile_with_fingerprint(
            observed_bytes=observed_bytes,
            total_params=total_params,
            candidates=tuple(candidates),
            fingerprint=fingerprint,
            argmin_scheme=argmin_scheme,
            argmin_err=argmin_err,
        )

    # Tolerance gate without fingerprint
    if argmin_err > _UNKNOWN_THRESHOLD:
        return ReconciliationReport(
            observed_bytes=observed_bytes,
            total_params=total_params,
            candidates=tuple(candidates),
            best=AnnotatedValue(
                "UNKNOWN",
                Label.UNKNOWN,
                source=(
                    f"closest candidate ({argmin_scheme}) is off by "
                    f"{argmin_err * 100:.1f}% — no confident match"
                ),
            ),
        )

    # Bytes-only tie detection
    tied_schemes = [
        c.scheme
        for c in candidates
        if abs(c.relative_error - argmin_err) < _TIE_THRESHOLD
        and c.relative_error <= _UNKNOWN_THRESHOLD
    ]
    if len(tied_schemes) > 1:
        tie_note = (
            f" — tied with {', '.join(s for s in tied_schemes if s != argmin_scheme)} "
            f"at the same bits/param; distinguishing requires config.json "
            f"quantization_config or safetensors per-tensor dtype "
            f"(neither available for this model)"
        )
        source_text = (
            f"best match among {len(candidates)} candidates, "
            f"{argmin_err * 100:.1f}% error{tie_note}"
        )
    else:
        source_text = (
            f"best match among {len(candidates)} candidates, {argmin_err * 100:.1f}% error"
        )

    return ReconciliationReport(
        observed_bytes=observed_bytes,
        total_params=total_params,
        candidates=tuple(candidates),
        best=AnnotatedValue(argmin_scheme, Label.INFERRED, source=source_text),
    )


def _reconcile_with_fingerprint(
    observed_bytes: int,
    total_params: int,
    candidates: tuple[ReconciliationCandidate, ...],
    fingerprint: QuantFingerprint,
    argmin_scheme: QuantizationScheme,
    argmin_err: float,
) -> ReconciliationReport:
    """Fingerprint-driven path.

    Rules:
      - If the declared scheme is in the candidates AND its bytes-error is within
        tolerance → adopt it. Label VERIFIED (we're reading authoritative metadata,
        not inferring).
      - If declared scheme's bytes-error is > 15% → conflict. Still adopt the
        declared scheme but log the discrepancy. This usually means our param
        estimate is off, not that the declaration is wrong.
      - If declared scheme is unknown to us → fall back to argmin with note.
    """
    declared = fingerprint.scheme
    match = next((c for c in candidates if c.scheme == declared), None)

    if match is None:
        # Unknown scheme from fingerprint — degrade gracefully to bytes-only.
        return ReconciliationReport(
            observed_bytes=observed_bytes,
            total_params=total_params,
            candidates=candidates,
            best=AnnotatedValue(
                argmin_scheme,
                Label.INFERRED,
                source=(
                    f"fingerprint declared {declared} ({fingerprint.evidence}) "
                    f"but we have no bpp anchor for it; fell back to bytes match "
                    f"{argmin_scheme} at {argmin_err * 100:.1f}% error"
                ),
            ),
        )

    if match.relative_error <= _UNKNOWN_THRESHOLD:
        # Agreement — fingerprint picks a plausible scheme. This is the happy path.
        note = ""
        # Extra context: if bytes alone would have chosen a different scheme, say so.
        if declared != argmin_scheme and argmin_err < match.relative_error:
            note = (
                f" (bytes alone would argmin to {argmin_scheme} at "
                f"{argmin_err * 100:.1f}%; we trust the declaration)"
            )
        return ReconciliationReport(
            observed_bytes=observed_bytes,
            total_params=total_params,
            candidates=candidates,
            best=AnnotatedValue(
                declared,
                Label.VERIFIED,
                source=(
                    f"{fingerprint.evidence} "
                    f"(predicts {match.predicted_bytes:,} bytes, "
                    f"{match.relative_error * 100:.1f}% error){note}"
                ),
            ),
        )

    # Disagreement: declared scheme's prediction is >15% off from observed bytes.
    # Still trust the declaration — usually means our param estimate drifted.
    return ReconciliationReport(
        observed_bytes=observed_bytes,
        total_params=total_params,
        candidates=candidates,
        best=AnnotatedValue(
            declared,
            Label.VERIFIED,
            source=(
                f"{fingerprint.evidence} "
                f"(NOTE: bytes predict {match.predicted_bytes:,}, off by "
                f"{match.relative_error * 100:.1f}% — likely our param estimate is off, "
                f"not the declaration)"
            ),
        ),
    )
