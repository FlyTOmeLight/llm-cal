"""Optional LLM-based second opinion on the tool's derivation trace.

Design constraints (from the tool's honesty principle):
  1. Never overrides the 6 primary labels. LLM responses are tagged
     [llm-opinion] — a distinct 7th label.
  2. Opt-in only — requires --llm-review flag AND env vars set.
  3. Non-fatal — if the API call fails, the main report still works.
  4. User-chosen provider — supports any OpenAI-compatible endpoint
     (OpenAI, DeepSeek, Moonshot, Zhipu, local vLLM, etc.)
  5. Deterministic input — the prompt is built from the --explain
     derivation trace, not free-form. The LLM gets structured math,
     not prose.
  6. The LLM's job is to CRITIQUE, not to REWRITE. The prompt
     explicitly forbids generating new numbers.

Environment variables:
  LLM_CAL_REVIEWER_API_KEY   (required)
  LLM_CAL_REVIEWER_BASE_URL  (default: https://api.openai.com/v1)
  LLM_CAL_REVIEWER_MODEL     (default: gpt-4o)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import httpx

from llm_cal.core.explain import ExplainEntry

Locale = Literal["en", "zh"]


@dataclass(frozen=True)
class LLMReviewResult:
    ok: bool
    content: str | None
    error: str | None
    model: str
    base_url: str


def run_review(
    entries: list[ExplainEntry],
    locale: Locale,
    timeout_s: float = 60.0,
) -> LLMReviewResult:
    """Send the derivation trace to an LLM for audit.

    Returns a LLMReviewResult. Never raises — always returns a result
    object even on failure.
    """
    api_key = os.environ.get("LLM_CAL_REVIEWER_API_KEY")
    base_url = os.environ.get("LLM_CAL_REVIEWER_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_CAL_REVIEWER_MODEL", "gpt-4o")

    if not api_key:
        return LLMReviewResult(
            ok=False,
            content=None,
            error=(
                "LLM_CAL_REVIEWER_API_KEY env var not set. "
                "Set it to the API key of an OpenAI-compatible endpoint "
                "(OpenAI, DeepSeek, Moonshot, Zhipu, etc.)."
            ),
            model=model,
            base_url=base_url,
        )

    prompt = _build_prompt(entries, locale)

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _system_prompt(locale)},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            )
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        return LLMReviewResult(
            ok=False,
            content=None,
            error=f"{type(e).__name__}: {e}",
            model=model,
            base_url=base_url,
        )

    if resp.status_code != 200:
        return LLMReviewResult(
            ok=False,
            content=None,
            error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            model=model,
            base_url=base_url,
        )

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, ValueError) as e:
        return LLMReviewResult(
            ok=False,
            content=None,
            error=f"Malformed response: {type(e).__name__}: {e}",
            model=model,
            base_url=base_url,
        )

    return LLMReviewResult(ok=True, content=content, error=None, model=model, base_url=base_url)


def _system_prompt(locale: Locale) -> str:
    if locale == "zh":
        return (
            "你是一个大模型推理硬件计算工具的独立审计者。工具产出确定性的推导链，"
            "你的工作是发现数学错误、不合理假设或遗漏。你不负责重新计算，"
            "只负责评论和确认。输出简体中文。"
        )
    return (
        "You are an independent auditor for a deterministic LLM inference hardware "
        "calculator. The tool produces a derivation trace; your job is to find math "
        "errors, unreasonable assumptions, or missing considerations. You do NOT "
        "recalculate; you only critique and confirm."
    )


def _build_prompt(entries: list[ExplainEntry], locale: Locale) -> str:
    trace = "\n\n".join(_format_entry(e) for e in entries)
    if locale == "zh":
        return _prompt_zh(trace)
    return _prompt_en(trace)


def _format_entry(entry: ExplainEntry) -> str:
    parts: list[str] = [f"## {entry.heading}"]
    parts.append(f"Formula:\n{entry.formula}")
    if entry.inputs:
        parts.append("Inputs:")
        for inp in entry.inputs:
            note = f" ({inp.note})" if inp.note else ""
            parts.append(f"  - {inp.name} = {inp.value} {inp.label}{note}")
    if entry.steps:
        parts.append("Steps:")
        for step in entry.steps:
            parts.append(f"  {step}")
    parts.append(f"Result: {entry.result}")
    if entry.source:
        parts.append(f"Source: {entry.source}")
    return "\n".join(parts)


def _prompt_en(trace: str) -> str:
    return f"""The deterministic tool produced this derivation trace for one model evaluation. \
Audit it.

<DERIVATION_TRACE>
{trace}
</DERIVATION_TRACE>

Respond in this structure. If a section has nothing to flag, write "none".

## Critical issues
(math errors or wrong formulas — would give wrong final answer)

## Moderate concerns
(unreasonable assumptions, factors off by 2x+, missing TP/sharding effects, etc.)

## Minor notes
(clarifications, stylistic, optional improvements)

## Consensus check
(which ExplainEntry headings look correct? name them explicitly)

Rules:
  - Cite specific ExplainEntry heading names. Be concrete.
  - Do NOT produce new numbers. Only critique.
  - If you don't know, say so. Do not hallucinate.
  - All your output must be tagged as a second opinion, NOT authoritative."""


def _prompt_zh(trace: str) -> str:
    return f"""下面是工具产出的一份完整推导链。请审计。

<DERIVATION_TRACE>
{trace}
</DERIVATION_TRACE>

按下面结构回复。没内容的段落写"无"。

## 关键错误
（数学错误或公式错误 —— 会导致最终答案错）

## 中度疑虑
（不合理假设、因子偏差 2x+、遗漏的 TP 分摊等）

## 次要备注
（澄清、风格、可选改进）

## 一致性核查
（哪些 ExplainEntry 标题看起来是对的？明确列出）

规则：
  - 必须引用具体的 ExplainEntry 标题名。具体点。
  - 不要产出新数字，只做评论。
  - 不确定的地方直说。不要编造。
  - 你的所有输出都只是 second opinion，不是权威答案。"""
