"""Tests for i18n layer."""

from __future__ import annotations

import pytest

from llm_cal.common import i18n
from llm_cal.common.i18n import detect_locale_from_env, set_locale, t


@pytest.fixture(autouse=True)
def restore_locale():
    original = i18n.get_locale()
    yield
    set_locale(original)


def test_default_locale_is_en():
    set_locale("en")
    assert t("section.architecture") == "Architecture"


def test_zh_translates():
    set_locale("zh")
    assert t("section.architecture") == "架构"
    assert t("section.weights") == "权重"
    assert t("section.kv_cache") == "单请求 KV Cache（BF16/FP16）"


def test_unknown_key_returns_key_itself():
    set_locale("en")
    assert t("this.key.does.not.exist") == "this.key.does.not.exist"


def test_template_substitution():
    set_locale("en")
    out = t("arch.attn_summary", variant="CSA_HCA", heads=64, kv_heads=1, head_dim=512)
    assert "CSA_HCA" in out
    assert "heads=64" in out

    set_locale("zh")
    out_zh = t("arch.attn_summary", variant="CSA_HCA", heads=64, kv_heads=1, head_dim=512)
    assert "CSA_HCA" in out_zh
    assert "（" in out_zh  # uses full-width punctuation in Chinese


def test_env_detection_zh_cn(monkeypatch):
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    assert detect_locale_from_env() == "zh"


def test_env_detection_en_us(monkeypatch):
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    assert detect_locale_from_env() == "en"


def test_env_detection_lc_all_wins(monkeypatch):
    monkeypatch.setenv("LC_ALL", "zh_TW.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert detect_locale_from_env() == "zh"


def test_env_detection_unset(monkeypatch):
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    assert detect_locale_from_env() == "en"


def test_label_section_has_zh_and_en():
    """Smoke check: a random label pair exists in both locales."""
    set_locale("en")
    assert t("weights.safetensors_bytes") == "safetensors bytes"
    set_locale("zh")
    assert t("weights.safetensors_bytes") == "safetensors 总字节"
