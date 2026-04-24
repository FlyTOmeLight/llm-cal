"""Tests for the label enum and AnnotatedValue wrapper."""

from __future__ import annotations

from llm_cal.output.labels import AnnotatedValue, Label


def test_all_six_labels_exist():
    """The 6-level discipline must be exhaustively encoded."""
    expected = {"verified", "inferred", "estimated", "cited", "unverified", "unknown"}
    assert {label.value for label in Label} == expected


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
