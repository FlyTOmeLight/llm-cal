"""Tests for the label enum and AnnotatedValue wrapper."""

from __future__ import annotations

from llm_cal.output.labels import AnnotatedValue, Label


def test_all_primary_labels_exist():
    """The 6 primary labels + llm-opinion (7th, opt-in) must be exhaustively encoded.

    The primary 6 are part of the tool's core value prop (label discipline).
    llm-opinion is the opt-in 7th for --llm-review output, never overriding.
    """
    primary = {"verified", "inferred", "estimated", "cited", "unverified", "unknown"}
    all_labels = {label.value for label in Label}
    assert primary <= all_labels, f"missing primary labels: {primary - all_labels}"
    assert "llm-opinion" in all_labels, "missing opt-in llm-opinion label"
    assert len(all_labels) == 7


def test_annotated_value_preserves_data():
    v = AnnotatedValue(160_300_000_000, Label.VERIFIED, source="HF siblings")
    assert v.value == 160_300_000_000
    assert v.label == Label.VERIFIED
    assert v.source == "HF siblings"


def test_render_tag_bracket_format():
    v = AnnotatedValue(4.52, Label.INFERRED)
    assert v.render_tag() == "[inferred]"


def test_label_is_string_enum():
    """StrEnum enables direct string comparison — no .value needed."""
    assert Label.VERIFIED == "verified"
    assert f"{Label.CITED}" == "cited"
