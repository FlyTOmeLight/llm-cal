"""Minimal i18n layer. No gettext, no external deps.

Supports `en` and `zh`. Defaults to `en` but auto-detects from LC_ALL/LANG
when they start with `zh` (covers zh_CN, zh_TW, zh_HK, etc.).

Usage:
    from llm_cal.common.i18n import t, set_locale
    set_locale("zh")
    print(t("labels.legend"))   # "标签"
"""

from __future__ import annotations

import os
from typing import Literal

Locale = Literal["en", "zh"]

_current_locale: Locale = "en"


_MESSAGES: dict[str, dict[Locale, str]] = {
    # CLI help text
    "cli.help": {
        "en": "LLM inference hardware calculator.",
        "zh": "大模型推理硬件计算器。",
    },
    "cli.arg.model_id": {
        "en": "HuggingFace or ModelScope model id",
        "zh": "HuggingFace 或 ModelScope 的 model id",
    },
    "cli.opt.gpu": {
        "en": "GPU type, e.g. H800, A100-80G",
        "zh": "GPU 型号，例如 H800、A100-80G",
    },
    "cli.opt.engine": {
        "en": "Inference engine: vllm | sglang",
        "zh": "推理引擎：vllm | sglang",
    },
    "cli.opt.gpu_count": {
        "en": "Force GPU count (otherwise tool recommends min/dev/prod)",
        "zh": "强制指定 GPU 张数（默认由工具推荐 min/dev/prod 三档）",
    },
    "cli.opt.context_length": {
        "en": "Context length for KV cache estimation",
        "zh": "用于 KV cache 估算的上下文长度",
    },
    "cli.opt.refresh": {
        "en": "Bypass cache and re-fetch",
        "zh": "绕过缓存重新拉取",
    },
    "cli.opt.lang": {
        "en": "Output language: en | zh",
        "zh": "输出语言：en | zh",
    },
    "cli.err.auth_required": {
        "en": "Authentication required:",
        "zh": "需要认证：",
    },
    "cli.err.model_not_found": {
        "en": "Model not found:",
        "zh": "模型未找到：",
    },
    "cli.err.source_unavailable": {
        "en": "Source unavailable:",
        "zh": "数据源不可用：",
    },
    # Panel / section titles
    "panel.via": {"en": "via", "zh": "来源"},
    "section.architecture": {"en": "Architecture", "zh": "架构"},
    "section.weights": {"en": "Weights", "zh": "权重"},
    "section.kv_cache": {
        "en": "KV cache per request (BF16/FP16)",
        "zh": "单请求 KV Cache（BF16/FP16）",
    },
    "section.reconciliation": {
        "en": "Quantization reconciliation (observed vs predicted per scheme)",
        "zh": "量化方案对账（观测值 vs 各方案预测值）",
    },
    "section.engine_compat": {
        "en": "Engine compatibility",
        "zh": "推理引擎兼容性",
    },
    "section.hardware": {"en": "Target hardware", "zh": "目标硬件"},
    "section.labels": {"en": "labels:", "zh": "标签："},
    # Architecture row labels
    "arch.model_type": {"en": "model_type", "zh": "模型类型"},
    "arch.family": {"en": "family", "zh": "架构族"},
    "arch.confidence": {"en": "confidence", "zh": "识别置信度"},
    "arch.layers": {"en": "layers", "zh": "层数"},
    "arch.hidden_size": {"en": "hidden_size", "zh": "隐藏维度"},
    "arch.vocab_size": {"en": "vocab_size", "zh": "词表大小"},
    "arch.attention": {"en": "attention", "zh": "注意力机制"},
    "arch.compress_ratios": {"en": "compress_ratios", "zh": "压缩比数组"},
    "arch.moe": {"en": "moe", "zh": "MoE"},
    "arch.sliding_window": {"en": "sliding_window", "zh": "滑动窗口"},
    "arch.max_position": {
        "en": "max_position_embeddings",
        "zh": "最大上下文长度",
    },
    "arch.none": {"en": "(none)", "zh": "（无）"},
    "arch.compress_ratios_summary": {
        "en": "len={n}, dense_layers={dense}",
        "zh": "长度={n}，dense 层数={dense}",
    },
    "arch.moe_summary": {
        "en": "{routed} routed + {shared} shared, top-{topk}",
        "zh": "{routed} 个 routed + {shared} 个 shared，top-{topk}",
    },
    "arch.attn_summary": {
        "en": "{variant} (heads={heads}, kv_heads={kv_heads}, head_dim={head_dim})",
        "zh": "{variant}（heads={heads}，kv_heads={kv_heads}，head_dim={head_dim}）",
    },
    "arch.unsupported_state_space": {
        "en": "State-space models are not supported in v0.1 (planned for v0.3+).",
        "zh": "状态空间模型（Mamba 类）在 v0.1 暂不支持，计划在 v0.3+ 加入。",
    },
    # Weights rows
    "weights.safetensors_bytes": {
        "en": "safetensors bytes",
        "zh": "safetensors 总字节",
    },
    "weights.params_estimated": {
        "en": "estimated total params",
        "zh": "参数量（估算）",
    },
    "weights.bits_per_param": {"en": "bits/param", "zh": "每参数位数"},
    "weights.quant_guess": {"en": "quantization guess", "zh": "量化方案推断"},
    # Reconciliation
    "recon.scheme": {"en": "scheme", "zh": "量化方案"},
    "recon.predicted": {"en": "predicted bytes", "zh": "预测字节"},
    "recon.delta": {"en": "delta", "zh": "差值"},
    "recon.error_pct": {"en": "error %", "zh": "误差 %"},
    "recon.over": {"en": "over", "zh": "偏高"},
    "recon.under": {"en": "under", "zh": "偏低"},
    "recon.best": {"en": "best match:", "zh": "最佳匹配："},
    # KV cache
    "kv.context": {"en": "context", "zh": "上下文"},
    "kv.kv_cache": {"en": "KV cache", "zh": "KV Cache"},
    "kv.label": {"en": "label", "zh": "标签"},
    "kv.tokens": {"en": "tokens", "zh": "tokens"},
    # Engine compatibility
    "engine.version_spec": {"en": "version", "zh": "版本要求"},
    "engine.support": {"en": "support", "zh": "支持程度"},
    "engine.verification": {"en": "verification", "zh": "验证等级"},
    "engine.required_flags": {"en": "required flags", "zh": "必需参数"},
    "engine.optional_flags": {"en": "optional flags", "zh": "可选参数"},
    "engine.caveats": {"en": "caveats", "zh": "注意事项"},
    "engine.sources": {"en": "sources", "zh": "来源"},
    "engine.no_match": {
        "en": "No compatibility entry for this model + engine in v0.1 matrix.",
        "zh": "v0.1 兼容矩阵中暂无此模型 + 引擎的条目。",
    },
    # Hardware
    "hw.memory": {"en": "memory", "zh": "显存"},
    "hw.nvlink_bandwidth": {"en": "NVLink bandwidth", "zh": "NVLink 带宽"},
    "hw.fp16_tflops": {"en": "FP16 TFLOPS", "zh": "FP16 算力"},
    "hw.fp8_support": {"en": "FP8 support", "zh": "FP8 支持"},
    "hw.fp4_support": {"en": "FP4 support", "zh": "FP4 支持"},
    "hw.notes": {"en": "notes", "zh": "备注"},
    "hw.spec_source": {"en": "spec source", "zh": "规格来源"},
    "hw.unknown": {
        "en": "Unknown GPU '{gpu}'. Known: {known}",
        "zh": "未知 GPU '{gpu}'。已知型号：{known}",
    },
    "hw.bool_yes": {"en": "yes", "zh": "是"},
    "hw.bool_no": {"en": "no", "zh": "否"},
    # Source attribution
    "source.pr": {"en": "PR", "zh": "PR"},
    "source.release_notes": {"en": "release notes", "zh": "release note"},
    "source.announcement": {"en": "announcement", "zh": "官方公告"},
    "source.tested": {"en": "tested", "zh": "实测"},
    "source.captured_on": {"en": "captured on", "zh": "采集于"},
    # Fleet planner
    "section.fleet": {
        "en": "Recommended fleet",
        "zh": "推荐 GPU 张数",
    },
    "fleet.col.tier": {"en": "tier", "zh": "档位"},
    "fleet.col.gpus": {"en": "GPUs", "zh": "GPU 数"},
    "fleet.col.weight_per_gpu": {
        "en": "weight / GPU",
        "zh": "单卡权重",
    },
    "fleet.col.headroom_per_gpu": {
        "en": "headroom / GPU",
        "zh": "单卡余量",
    },
    "fleet.col.fit": {"en": "fit", "zh": "评估"},
    "fleet.col.concurrent_at_ctx": {
        "en": "concurrent @ {ctx}",
        "zh": "并发 @ {ctx}",
    },
    "fleet.tier.min": {"en": "min", "zh": "最小"},
    "fleet.tier.dev": {"en": "dev", "zh": "开发"},
    "fleet.tier.prod": {"en": "prod", "zh": "生产"},
    "fleet.best_marker": {
        "en": "= recommended",
        "zh": "= 推荐档位",
    },
    "fleet.constraint": {"en": "constraint:", "zh": "约束："},
    "fleet.forced": {
        "en": "Forced GPU count (--gpu-count was set)",
        "zh": "已强制指定 GPU 张数（--gpu-count）",
    },
    "fleet.gpu_spec_unknown": {
        "en": "Fleet planning skipped — GPU spec unknown.",
        "zh": "GPU 规格未知，跳过 fleet 规划。",
    },
    # Command generator
    "section.command": {
        "en": "Generated command",
        "zh": "生成的启动命令",
    },
    "command.tier_note": {
        "en": "tier: {tier} ({gpus} GPUs)",
        "zh": "档位：{tier}（{gpus} 张）",
    },
}


def set_locale(loc: Locale) -> None:
    global _current_locale
    _current_locale = loc


def get_locale() -> Locale:
    return _current_locale


def detect_locale_from_env() -> Locale:
    """Auto-detect from standard locale env vars."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        val = os.environ.get(var, "").lower()
        if val.startswith("zh"):
            return "zh"
    return "en"


def t(key: str, **kwargs: object) -> str:
    """Translate a message key. Unknown keys return the key itself (fail loud)."""
    bundle = _MESSAGES.get(key)
    if bundle is None:
        return key
    template = bundle.get(_current_locale, bundle.get("en", key))
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template
