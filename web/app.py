"""llm-cal Gradio web app — deploys to HuggingFace Spaces.

User journey:
  1. Type a HuggingFace model id (or pick from examples)
  2. Choose target GPU
  3. Hit Calculate
  4. Read the same `--explain`-quality output the CLI gives you, but in a browser
     and shareable via URL parameters.

The whole compute is the existing Python `Evaluator`. No new logic.

Local run:
  python web/app.py
HF Spaces:
  This file is the entry point Spaces expects. requirements.txt sits next to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable. Two layouts supported:
#   1. Local dev:  /repo/web/app.py + /repo/src/        (parent.parent / src)
#   2. HF Space:   /space/app.py    + /space/src/       (parent / src)
# The deploy workflow flattens layout 1 → layout 2 when pushing to the Space.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "src", _HERE.parent / "src"):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break

import os  # noqa: E402

import gradio as gr  # noqa: E402

from llm_cal.common.i18n import set_locale, t  # noqa: E402
from llm_cal.core.evaluator import EvaluationReport, Evaluator  # noqa: E402
from llm_cal.core.explain import ExplainEntry  # noqa: E402
from llm_cal.core.explain import build as build_explain  # noqa: E402
from llm_cal.hardware.loader import load_database  # noqa: E402
from llm_cal.llm_review.reviewer import run_review  # noqa: E402
from llm_cal.model_source.huggingface import HuggingFaceSource  # noqa: E402
from llm_cal.model_source.modelscope import ModelScopeSource  # noqa: E402

# ---------------------------------------------------------------------------
# Static data the UI needs

_DB = load_database()


def _classify_vendor(gpu_id: str) -> tuple[str, str]:
    """Map a GPU id to (vendor_en, vendor_zh).

    Vendor isn't in the YAML schema (yet), so derive from the id prefix.
    """
    gid = gpu_id.upper()
    if gid in {"B200", "GB200", "H100", "H800", "H200", "H20", "GH200"} or gid.startswith(
        ("L4", "L40", "RTX", "A10", "A100", "A40", "V100", "T4")
    ):
        return ("NVIDIA", "NVIDIA")
    if gid.startswith("MI"):
        return ("AMD", "AMD")
    if gid.startswith("GAUDI"):
        return ("Intel Habana", "英特尔 Habana")
    if gid.startswith("910") or gid.startswith("ATLAS"):
        return ("Huawei Ascend", "华为昇腾")
    if gid.startswith("MXC"):
        return ("MetaX 沐曦", "沐曦 MetaX")
    if gid.startswith("KUNLUN"):
        return ("Kunlunxin 昆仑芯", "昆仑芯 Kunlunxin")
    if gid.startswith("BR"):
        return ("Biren 壁仞", "壁仞 Biren")
    if gid.startswith("BI-"):
        return ("Iluvatar 天数智芯", "天数智芯 Iluvatar")
    if gid.startswith(("MR-", "MTT")):
        return ("Moore Threads 摩尔线程", "摩尔线程 Moore Threads")
    if gid.startswith("MLU"):
        return ("Cambricon 寒武纪", "寒武纪 Cambricon")
    if gid.startswith("HYGON"):
        return ("Hygon 海光", "海光 Hygon")
    return ("Other", "其他")


# Stable vendor display order
_VENDOR_ORDER = [
    "NVIDIA",
    "AMD",
    "Intel Habana",
    "Huawei Ascend",
    "MetaX 沐曦",
    "Kunlunxin 昆仑芯",
    "Biren 壁仞",
    "Iluvatar 天数智芯",
    "Moore Threads 摩尔线程",
    "Cambricon 寒武纪",
    "Hygon 海光",
    "Other",
]


def _build_vendor_index() -> dict[str, list[str]]:
    """vendor_en -> sorted list of GPU ids"""
    out: dict[str, list[str]] = {v: [] for v in _VENDOR_ORDER}
    for g in _DB.gpus:
        v_en, _ = _classify_vendor(g.id)
        out.setdefault(v_en, []).append(g.id)
    for v in out:
        out[v].sort()
    # Drop empty buckets
    return {v: ids for v, ids in out.items() if ids}


_VENDOR_TO_GPUS = _build_vendor_index()
VENDOR_CHOICES_EN: list[str] = list(_VENDOR_TO_GPUS.keys())
DEFAULT_VENDOR = "NVIDIA"
DEFAULT_GPU = "H800"

EXAMPLE_MODELS: list[tuple[str, str, str, str, str]] = [
    # (model_id, vendor, gpu, engine, source)
    ("deepseek-ai/DeepSeek-V4-Flash", "NVIDIA", "H800", "vllm", "HuggingFace"),
    ("deepseek-ai/DeepSeek-V3", "NVIDIA", "H800", "vllm", "HuggingFace"),
    ("Qwen/Qwen2.5-72B-Instruct", "NVIDIA", "H100", "vllm", "HuggingFace"),
    ("Qwen/Qwen3-30B-A3B", "NVIDIA", "A100-80G", "vllm", "HuggingFace"),
    ("mistralai/Mixtral-8x7B-v0.1", "NVIDIA", "H100", "vllm", "HuggingFace"),
    ("microsoft/Phi-4", "NVIDIA", "RTX4090", "vllm", "HuggingFace"),
    ("deepseek-ai/DeepSeek-V4-Flash", "Huawei Ascend", "910B4", "vllm", "HuggingFace"),
    # ModelScope examples — same models, China-side mirror.
    ("Qwen/Qwen3-30B-A3B", "NVIDIA", "A100-80G", "vllm", "ModelScope"),
    ("deepseek-ai/DeepSeek-V3", "Huawei Ascend", "910B4", "vllm", "ModelScope"),
]

# ---------------------------------------------------------------------------
# Output rendering


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    f = float(n)
    for u in ["KB", "MB", "GB", "TB"]:
        f /= 1024
        if f < 1024:
            return f"{f:.2f} {u}"
    return f"{f:.2f} PB"


def _fmt_params(n: int | None) -> str:
    if not n:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def _label_color(label: str) -> str:
    """Map a provenance label to a CSS color (visible in both light and dark)."""
    return {
        "verified": "#16a34a",  # green-600
        "inferred": "#2563eb",  # blue-600
        "estimated": "#d97706",  # amber-600
        "cited": "#7c3aed",  # violet-600
        "unverified": "#9a3412",  # orange-800
        "unknown": "#6b7280",  # gray-500
        "llm-opinion": "#db2777",  # pink-600
    }.get(label, "#6b7280")


def _label_chip(label_key: str) -> str:
    """Render a [label] chip with the right color."""
    color = _label_color(label_key)
    text = t(f"label.{label_key}")
    return (
        f'<span class="lc-chip" style="background:{color}1a;color:{color};'
        f'border:1px solid {color}55">{text}</span>'
    )


def _stat_card(label: str, value: str, sublabel: str = "", chip: str = "") -> str:
    chip_html = f"<div class='lc-stat-chip'>{chip}</div>" if chip else ""
    sub_html = f"<div class='lc-stat-sub'>{sublabel}</div>" if sublabel else ""
    return (
        f"<div class='lc-stat'>"
        f"<div class='lc-stat-value'>{value}</div>"
        f"<div class='lc-stat-label'>{label}</div>"
        f"{sub_html}{chip_html}"
        f"</div>"
    )


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _render(report: EvaluationReport, locale: str) -> str:
    set_locale(locale)  # type: ignore[arg-type]
    is_zh = locale == "zh"

    p, w, r, f = report.profile, report.weight, report.reconciliation, report.fleet

    # ---- Headline stat cards -------------------------------------------------
    weight_str = _fmt_bytes(w.total_bytes.value)
    weight_chip = _label_chip(w.total_bytes.label.value)
    quant_chip = _label_chip(w.quantization_guess.label.value)
    prod_opt = (
        next((o for o in (f.options if f else []) if o.tier == "prod"), None) if f else None
    )
    prod_gpus = str(prod_opt.gpu_count) if prod_opt else "—"
    prod_concurrent = str(prod_opt.max_concurrent_at_reference_ctx) if prod_opt else "—"

    headline = (
        f"<div class='lc-header'>"
        f"<div class='lc-title'>{_esc(report.model_id)}</div>"
        f"<div class='lc-subtitle'>"
        f"{_esc(report.gpu)} · {_esc(report.engine)}"
        f"</div></div>"
        f"<div class='lc-stats'>"
        f"{_stat_card('Weight' if not is_zh else '权重', weight_str, sublabel='from safetensors API' if not is_zh else '取自 safetensors API', chip=weight_chip)}"
        f"{_stat_card('Quantization' if not is_zh else '量化', _esc(w.quantization_guess.value), sublabel='resolved scheme' if not is_zh else '已识别方案', chip=quant_chip)}"
        f"{_stat_card('Prod GPUs' if not is_zh else 'Prod GPU 数', prod_gpus, sublabel='for 16-user prod' if not is_zh else '生产档（16 路并发）')}"
        f"{_stat_card('Users @ 128K' if not is_zh else '用户 @ 128K', prod_concurrent, sublabel='concurrent at prod tier' if not is_zh else '生产档的并发')}"
        f"</div>"
    )

    # Provenance footer for the headline
    quant_source = _esc(w.quantization_guess.source or "")
    headline += f"<div class='lc-prov'>{quant_source}</div>"

    # ---- Architecture --------------------------------------------------------
    arch_rows: list[tuple[str, str]] = [("model_type", p.model_type)]
    if p.attention:
        arch_rows.append(
            (
                "attention",
                f"{p.attention.variant} (heads={p.attention.num_heads}, "
                f"kv_heads={p.attention.num_kv_heads}, hd={p.attention.head_dim})",
            )
        )
    if p.moe:
        arch_rows.append(
            (
                "moe",
                f"{p.moe.num_routed_experts} routed + "
                f"{p.moe.num_shared_experts} shared, top-{p.moe.num_experts_per_tok}",
            )
        )
    if p.sliding_window:
        arch_rows.append(("sliding_window", str(p.sliding_window)))

    arch_html = "".join(
        f"<tr><td><code>{_esc(k)}</code></td><td><code>{_esc(v)}</code></td></tr>"
        for k, v in arch_rows
    )
    arch_explainer = (
        "从模型 config.json 读出来的，决定后续所有公式怎么走（是否分组注意力、是否 MoE、是否滑动窗口）。"
        if is_zh
        else "Read straight from the model's config.json. Drives every formula "
        "downstream — attention sharding, MoE active-expert ratio, sliding window."
    )
    arch_section = (
        f"<div class='lc-section'><h3>{'架构' if is_zh else 'Architecture'}</h3>"
        f"<div class='lc-section-help'>{arch_explainer}</div>"
        f"<table class='lc-table'>{arch_html}</table></div>"
    )

    # ---- Reconciliation ------------------------------------------------------
    recon_rows = []
    for c in r.candidates[:5]:
        is_best = c.scheme == r.best.value
        cls = " class='lc-best'" if is_best else ""
        marker = " ✓" if is_best else ""
        recon_rows.append(
            f"<tr{cls}><td><code>{_esc(c.scheme)}</code>{marker}</td>"
            f"<td>{_fmt_bytes(c.predicted_bytes)}</td>"
            f"<td>{c.relative_error * 100:.1f}%</td></tr>"
        )
    recon_explainer = (
        "用每种量化方案预测应该有多少字节，跟实际 safetensors 字节对比。误差最小的胜出。"
        "FP4_FP8_MIXED / GPTQ_INT4 / AWQ_INT4 在 0.55 bpp 处会打平，需要 config 或 dtype 进一步区分。"
        if is_zh
        else "Predict bytes under each quantization hypothesis, compare against the real "
        "safetensors size. Lowest error wins. FP4_FP8_MIXED / GPTQ_INT4 / AWQ_INT4 tie "
        "at 0.55 bpp — broken via config.json or per-tensor dtype."
    )
    recon_section = (
        f"<div class='lc-section'>"
        f"<h3>{'量化反演' if is_zh else 'Quantization reconciliation'}</h3>"
        f"<div class='lc-section-help'>{recon_explainer}</div>"
        f"<table class='lc-table lc-table-recon'>"
        f"<thead><tr><th>Scheme</th>"
        f"<th>{'预测字节' if is_zh else 'Predicted'}</th>"
        f"<th>{'误差' if is_zh else 'Error'}</th></tr></thead>"
        f"<tbody>{''.join(recon_rows)}</tbody></table></div>"
    )

    # ---- Fleet ---------------------------------------------------------------
    fleet_section = ""
    if f and f.options:
        # Pick which context lengths get their own concurrency column.
        # Always include 128K if any option has it; also include the model max
        # if it's larger (e.g. 1M for DeepSeek-V4-Flash) so the user can compare
        # "fits 23 users at 128K but only 2 at 1M".
        all_ctxs: set[int] = set()
        for opt in f.options:
            for ctx, _ in opt.max_concurrent_by_context:
                all_ctxs.add(ctx)
        ctx_cols: list[int] = []
        if 131_072 in all_ctxs:
            ctx_cols.append(131_072)
        max_ctx = max(all_ctxs) if all_ctxs else 0
        if max_ctx > 131_072 and max_ctx not in ctx_cols:
            ctx_cols.append(max_ctx)
        if not ctx_cols and all_ctxs:
            ctx_cols.append(max_ctx)

        def _ctx_label(ctx: int) -> str:
            if ctx >= 1_000_000:
                return f"{ctx // 1_000_000}M" if ctx % 1_000_000 == 0 else f"{ctx / 1_000_000:.1f}M"
            if ctx >= 1024:
                return f"{ctx // 1024}K"
            return str(ctx)

        rows = []
        for opt in f.options:
            star = " ★" if opt.tier == f.best_tier else ""
            cls = " class='lc-best'" if opt.tier == f.best_tier else ""
            headroom = max(0, opt.usable_bytes_per_gpu - opt.weight_bytes_per_gpu)
            ctx_map = dict(opt.max_concurrent_by_context)
            ctx_cells = "".join(f"<td>{ctx_map.get(c, '—')}</td>" for c in ctx_cols)
            rows.append(
                f"<tr{cls}><td><code>{opt.tier}{star}</code></td>"
                f"<td>{opt.gpu_count}</td>"
                f"<td>{_fmt_bytes(opt.weight_bytes_per_gpu)}</td>"
                f"<td>{_fmt_bytes(headroom)}</td>"
                f"{ctx_cells}</tr>"
            )

        ctx_headers = "".join(
            f"<th>{('@ ' + _ctx_label(c) + ' 并发') if is_zh else ('Concurrent @ ' + _ctx_label(c))}</th>"
            for c in ctx_cols
        )
        fleet_explainer = (
            "min = 刚好放得下；dev = 8 路并发场景；prod = 16 路并发场景。★ = 推荐。"
            if is_zh
            else "min = barely fits weights; dev = sized for 8 concurrent at 128K; "
            "prod = sized for 16 concurrent at 128K. ★ = recommended."
        )
        fleet_section = (
            f"<div class='lc-section'>"
            f"<h3>{'推荐集群' if is_zh else 'Recommended fleet'}</h3>"
            f"<div class='lc-section-help'>{fleet_explainer}</div>"
            f"<table class='lc-table'>"
            f"<thead><tr><th>Tier</th><th>GPUs</th>"
            f"<th>Weight/GPU</th><th>Headroom/GPU</th>"
            f"{ctx_headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    # ---- Performance ---------------------------------------------------------
    perf_explainer = (
        "Prefill 用算力公式（FLOPs = 2 × 参数 × 输入 token），decode 用带宽公式（吞吐 = 带宽 × 利用率 / 权重字节）。"
        "Bottleneck 标 memory_bandwidth 说明 decode 是带宽瓶颈，加显存带宽更高的 GPU 比加算力更划算。"
        if is_zh
        else "Prefill uses the compute formula (FLOPs = 2 × params × input_tokens, Kaplan 2020). "
        "Decode uses memory-bandwidth formula (tps = BW × util / weight_bytes, vLLM paper). "
        "Bottleneck = memory_bandwidth means a higher-BW GPU helps more than more FLOPS."
    )
    perf_section = ""
    if report.prefill and report.decode and report.concurrency:
        max_users = report.concurrency.max_concurrent.value
        bn = report.concurrency.bottleneck
        items = [
            (
                "Prefill latency" if not is_zh else "Prefill 延迟",
                f"{report.prefill.latency_ms.value:.0f} ms",
                f"@ {report.perf_input_tokens or 2000} input tokens",
            ),
            (
                "Cluster decode" if not is_zh else "集群 decode 吞吐",
                f"{report.decode.cluster_tokens_per_sec.value:.0f} tok/s",
                "",
            ),
            (
                "Max concurrent users" if not is_zh else "最大并发用户",
                str(max_users),
                "",
            ),
            (
                "Bottleneck" if not is_zh else "瓶颈",
                f"<code>{_esc(bn)}</code>",
                "",
            ),
        ]
        items_html = "".join(
            f"<div class='lc-perf-item'>"
            f"<div class='lc-perf-value'>{v}</div>"
            f"<div class='lc-perf-label'>{_esc(label)}</div>"
            f"<div class='lc-perf-sub'>{_esc(sub)}</div></div>"
            for label, v, sub in items
        )
        perf_section = (
            f"<div class='lc-section'>"
            f"<h3>{'性能' if is_zh else 'Performance'}</h3>"
            f"<div class='lc-section-help'>{perf_explainer}</div>"
            f"<div class='lc-perf'>{items_html}</div></div>"
        )

    # ---- KV cache per request -----------------------------------------------
    kv_section = ""
    if report.kv_cache_by_context:
        rows = []
        for ctx, av in sorted(report.kv_cache_by_context.items()):
            rows.append(
                f"<tr><td>{ctx:,}</td><td>{_fmt_bytes(av.value)}</td>"
                f"<td>{_label_chip(av.label.value)}</td></tr>"
            )
        kv_explainer = (
            "单个请求在不同 context 长度下需要多少 KV 缓存。这是决定一张 GPU 能并发跑多少请求的关键。"
            "MLA / MQA 模型这里会比标准 GQA 小很多。"
            if is_zh
            else "How much KV cache one request consumes at each context length. "
            "This is what limits per-GPU concurrency. MLA / MQA models are "
            "dramatically smaller here than standard GQA."
        )
        kv_section = (
            f"<div class='lc-section'>"
            f"<h3>{'KV 缓存（每请求）' if is_zh else 'KV cache per request'}</h3>"
            f"<div class='lc-section-help'>{kv_explainer}</div>"
            f"<table class='lc-table lc-table-recon'>"
            f"<thead><tr><th>{'Context tokens' if not is_zh else 'Context 长度'}</th>"
            f"<th>{'KV bytes' if not is_zh else 'KV 字节'}</th>"
            f"<th>{'Label' if not is_zh else '标签'}</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    # ---- Engine compatibility -----------------------------------------------
    engine_section = ""
    em = report.engine_match
    if em:
        def _fmt_flag(f) -> str:  # noqa: ANN001
            base = f"{f.flag} {f.value}".strip()
            return base
        flags = ", ".join(_fmt_flag(f) for f in em.required_flags) if em.required_flags else "—"
        opt_flags = ", ".join(_fmt_flag(f) for f in em.optional_flags) if em.optional_flags else "—"
        caveats = em.caveats_zh if is_zh else em.caveats_en
        sources_html = "—"
        if em.sources:
            sources_html = "<br>".join(
                f'<a href="{_esc(s.url)}" target="_blank" rel="noopener">{_esc(s.url)}</a>'
                + (
                    f" <span class='lc-prov'>({_esc(s.captured_date)})</span>"
                    if s.captured_date
                    else ""
                )
                for s in em.sources
            )
        rows = [
            (("引擎" if is_zh else "Engine"), f"<code>{_esc(em.engine)}</code>"),
            (
                ("版本要求" if is_zh else "Version"),
                f"<code>{_esc(em.version_spec)}</code>",
            ),
            (
                ("支持级别" if is_zh else "Support"),
                _label_chip(em.support) if em.support in {"verified", "cited", "unverified"} else f"<code>{_esc(em.support)}</code>",
            ),
            (
                ("验证级别" if is_zh else "Verification"),
                _label_chip(em.verification_level),
            ),
            (("必需 flag" if is_zh else "Required flags"), f"<code>{_esc(flags)}</code>"),
            (("可选 flag" if is_zh else "Optional flags"), f"<code>{_esc(opt_flags)}</code>"),
        ]
        if caveats:
            rows.append((("注意事项" if is_zh else "Caveats"), _esc(caveats)))
        rows.append((("来源" if is_zh else "Sources"), sources_html))
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        engine_explainer = (
            "这个模型在 vLLM/SGLang 哪个版本起能跑、需要哪些必需 flag、有哪些优化 flag。"
            "verification_level 标 cited 表示从 PR / release note 引用，verified 表示实测过。"
            if is_zh
            else "Which engine version supports this model, what flags are required, "
            "and which optional flags help. verification_level=cited means we got it "
            "from a PR or release note; verified means we actually ran it."
        )
        engine_section = (
            f"<div class='lc-section'>"
            f"<h3>{'引擎兼容性' if is_zh else 'Engine compatibility'}</h3>"
            f"<div class='lc-section-help'>{engine_explainer}</div>"
            f"<table class='lc-table'>{body}</table></div>"
        )

    # ---- GPU spec ------------------------------------------------------------
    gpu_section = ""
    g = report.gpu_spec
    if g:
        notes = g.notes_zh if is_zh else g.notes_en
        rows = [
            ("HBM", f"{g.memory_gb} GB"),
            ("Memory BW", f"{g.memory_bandwidth_gbps or '—'} GB/s"),
            ("NVLink BW", f"{g.nvlink_bandwidth_gbps} GB/s"),
            ("FP16 TFLOPS", f"{g.fp16_tflops}"),
            ("FP8", "✓" if g.fp8_support else "—"),
            ("FP4", "✓" if g.fp4_support else "—"),
        ]
        rows_html = "".join(
            f"<tr><td>{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>"
            for k, v in rows
        )
        notes_html = (
            f"<div class='lc-prov' style='margin-top:8px'>{_esc(notes)}</div>" if notes else ""
        )
        source_html = (
            f"<div class='lc-prov'>{'来源' if is_zh else 'Source'}: "
            f"<a href='{_esc(g.spec_source)}' target='_blank' rel='noopener'>"
            f"{_esc(g.spec_source)}</a></div>"
            if g.spec_source and g.spec_source.startswith("http")
            else (f"<div class='lc-prov'>{_esc(g.spec_source)}</div>" if g.spec_source else "")
        )
        gpu_explainer = (
            "目标 GPU 的硬件规格。Memory BW 决定 decode 能跑多快，FP8/FP4 支持决定能用什么量化。"
            if is_zh
            else "Hardware spec of the chosen GPU. Memory BW caps decode throughput; "
            "FP8/FP4 support determines which quantization paths actually accelerate."
        )
        gpu_section = (
            f"<div class='lc-section'>"
            f"<h3>{'目标 GPU 规格' if is_zh else 'Target GPU spec'} — <code>{_esc(g.id)}</code></h3>"
            f"<div class='lc-section-help'>{gpu_explainer}</div>"
            f"<table class='lc-table'>{rows_html}</table>"
            f"{notes_html}{source_html}"
            f"</div>"
        )

    # ---- Generated command ---------------------------------------------------
    cmd_section = ""
    if report.generated_command:
        cmd_explainer = (
            "可以直接复制粘贴到带显卡的机器上跑。flag 是按推荐 tier 的 GPU 数 + 引擎兼容矩阵的必需 flag 自动拼的。"
            if is_zh
            else "Copy-pasteable on a machine with the right GPUs. Flags auto-assembled "
            "from the recommended fleet tier + engine compat matrix's required flags."
        )
        cmd_section = (
            f"<div class='lc-section'>"
            f"<h3>{'生成命令' if is_zh else 'Generated command'}</h3>"
            f"<div class='lc-section-help'>{cmd_explainer}</div>"
            f"<pre class='lc-cmd'><code>{_esc(report.generated_command)}</code></pre></div>"
        )

    return (
        "<div class='lc-result'>"
        + headline
        + arch_section
        + gpu_section
        + recon_section
        + kv_section
        + fleet_section
        + perf_section
        + engine_section
        + cmd_section
        + _render_star_cta(is_zh)
        + "</div>"
    )


def _render_compare(reports: list[EvaluationReport], locale: str) -> str:
    """Side-by-side comparison of N >= 2 reports for the same model on
    different GPUs.

    Each metric column declares whether higher or lower is better and we
    paint the winner cell in green so the eye snaps to it.
    """
    set_locale(locale)  # type: ignore[arg-type]
    is_zh = locale == "zh"

    # All reports share the same model_id + engine — pull from the first.
    head = reports[0]
    title = (
        f"<div class='lc-header'>"
        f"<div class='lc-title'>{_esc(head.model_id)}</div>"
        f"<div class='lc-subtitle'>"
        f"{('对比 ' + str(len(reports)) + ' 张 GPU') if is_zh else ('Comparing ' + str(len(reports)) + ' GPUs')}"
        f" · {_esc(head.engine)}"
        f"</div></div>"
    )

    # Metric definitions: (label_en, label_zh, value_fn, better=lower|higher|none)
    def _max_concurrent(r: EvaluationReport) -> int | None:
        if not r.fleet:
            return None
        prod = next((o for o in r.fleet.options if o.tier == "prod"), None)
        return prod.max_concurrent_at_reference_ctx if prod else None

    def _prod_gpu_count(r: EvaluationReport) -> int | None:
        if not r.fleet:
            return None
        prod = next((o for o in r.fleet.options if o.tier == "prod"), None)
        return prod.gpu_count if prod else None

    metrics = [
        # (en, zh, getter, better, formatter)
        ("Quantization", "量化方案",
         lambda r: r.weight.quantization_guess.value, "none",
         lambda v: _esc(str(v)) if v else "—"),
        ("Prod GPUs", "生产档 GPU 数",
         _prod_gpu_count, "lower",
         lambda v: str(v) if v is not None else "—"),
        ("Users @ 128K", "用户 @ 128K",
         _max_concurrent, "higher",
         lambda v: str(v) if v is not None else "—"),
        ("Prefill latency", "Prefill 延迟",
         lambda r: r.prefill.latency_ms.value if r.prefill else None, "lower",
         lambda v: f"{v:.0f} ms" if v is not None else "—"),
        ("Cluster decode", "集群 decode 吞吐",
         lambda r: r.decode.cluster_tokens_per_sec.value if r.decode else None, "higher",
         lambda v: f"{v:.0f} tok/s" if v is not None else "—"),
        ("Bottleneck", "瓶颈",
         lambda r: r.concurrency.bottleneck if r.concurrency else None, "none",
         lambda v: f"<code>{_esc(str(v))}</code>" if v else "—"),
    ]

    # GPU column headers
    gpu_headers = "".join(
        f"<th class='lc-cmp-gpu'>{_esc(r.gpu)}</th>" for r in reports
    )

    rows_html = []
    for label_en, label_zh, getter, better, fmt in metrics:
        values = [getter(r) for r in reports]

        # Pick the winning index. None values are excluded from the contest.
        winner_idx: int | None = None
        if better in ("higher", "lower"):
            numeric_pairs = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
            if numeric_pairs:
                if better == "higher":
                    winner_idx = max(numeric_pairs, key=lambda p: p[1])[0]
                else:
                    winner_idx = min(numeric_pairs, key=lambda p: p[1])[0]
                # If all values are equal, no winner (avoid arbitrary-tiebreak gold star)
                vals_set = {v for _, v in numeric_pairs}
                if len(vals_set) <= 1:
                    winner_idx = None

        cells = []
        for i, v in enumerate(values):
            cls = " class='lc-cmp-winner'" if i == winner_idx else ""
            cells.append(f"<td{cls}>{fmt(v)}</td>")

        label = label_zh if is_zh else label_en
        rows_html.append(
            f"<tr><th class='lc-cmp-row-label'>{_esc(label)}</th>{''.join(cells)}</tr>"
        )

    # Aggregate winner — count column wins across "higher/lower" metrics
    win_counts = [0] * len(reports)
    for label_en, label_zh, getter, better, fmt in metrics:
        if better == "none":
            continue
        values = [getter(r) for r in reports]
        numeric_pairs = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
        if not numeric_pairs:
            continue
        vals_set = {v for _, v in numeric_pairs}
        if len(vals_set) <= 1:
            continue
        if better == "higher":
            winner_idx = max(numeric_pairs, key=lambda p: p[1])[0]
        else:
            winner_idx = min(numeric_pairs, key=lambda p: p[1])[0]
        win_counts[winner_idx] += 1

    overall_text = ""
    if any(win_counts):
        max_wins = max(win_counts)
        leaders = [reports[i].gpu for i, c in enumerate(win_counts) if c == max_wins]
        if len(leaders) == 1:
            overall_text = (
                f"<div class='lc-cmp-summary'>"
                f"{'综合最优' if is_zh else 'Overall winner'}: "
                f"<strong>{_esc(leaders[0])}</strong> "
                f"({max_wins}/{sum(1 for m in metrics if m[3] != 'none')} "
                f"{'指标领先' if is_zh else 'metrics lead'})"
                f"</div>"
            )
        else:
            overall_text = (
                f"<div class='lc-cmp-summary'>"
                f"{'势均力敌' if is_zh else 'Tied'}: "
                f"<strong>{_esc(' / '.join(leaders))}</strong>"
                f"</div>"
            )

    table = (
        f"<div class='lc-section'>"
        f"<h3>{'对比' if is_zh else 'Side-by-side comparison'}</h3>"
        f"<div class='lc-cmp-wrap'>"
        f"<table class='lc-cmp-table'>"
        f"<thead><tr>"
        f"<th class='lc-cmp-row-label'></th>"
        f"{gpu_headers}"
        f"</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
        f"{overall_text}"
        f"</div>"
    )

    # Per-GPU detail headlines (small stat cards) below the table
    detail_blocks = []
    for r in reports:
        weight_str = _fmt_bytes(r.weight.total_bytes.value)
        prod = _prod_gpu_count(r)
        users = _max_concurrent(r)
        detail_blocks.append(
            f"<div class='lc-cmp-detail'>"
            f"<div class='lc-cmp-detail-gpu'>{_esc(r.gpu)}</div>"
            f"<div class='lc-cmp-detail-row'>"
            f"<span>{'权重' if is_zh else 'Weight'}</span><strong>{weight_str}</strong></div>"
            f"<div class='lc-cmp-detail-row'>"
            f"<span>{'生产 GPU' if is_zh else 'Prod GPUs'}</span>"
            f"<strong>{prod if prod is not None else '—'}</strong></div>"
            f"<div class='lc-cmp-detail-row'>"
            f"<span>{'用户 @ 128K' if is_zh else 'Users @ 128K'}</span>"
            f"<strong>{users if users is not None else '—'}</strong></div>"
            f"</div>"
        )
    detail_section = (
        f"<div class='lc-section'>"
        f"<h3>{'各档详情' if is_zh else 'Per-GPU detail'}</h3>"
        f"<div class='lc-cmp-details'>{''.join(detail_blocks)}</div>"
        f"</div>"
    )

    return (
        "<div class='lc-result'>"
        + title
        + table
        + detail_section
        + _render_star_cta(is_zh)
        + "</div>"
    )


def _render_star_cta(is_zh: bool) -> str:
    """Tail-of-result CTA — shown right after the user got their answer,
    which is when satisfaction is highest and the GitHub star ask reads as
    'thanks for the tool' rather than 'please give me attention'."""
    en_msg = "Saved you GPU-sizing math?"
    zh_msg = "省了你 GPU 选型的时间？"
    cta_en = "Star on GitHub"
    cta_zh = "给个 Star"
    text_top = zh_msg if is_zh else en_msg
    text_bottom = en_msg if is_zh else zh_msg
    cta = f"{cta_zh if is_zh else cta_en} · {cta_en if is_zh else cta_zh}"
    return (
        "<a class='lc-star-cta' href='https://github.com/FlyTOmeLight/llm-cal' "
        "target='_blank' rel='noopener'>"
        "<svg viewBox='0 0 16 16' width='18' height='18' aria-hidden='true' fill='currentColor'>"
        "<path d='M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38v-1.33c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.81.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.83-2.15-.08-.2-.36-1.02.08-2.13 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.11.16 1.93.08 2.13.51.56.83 1.27.83 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.19c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z'/>"
        "</svg>"
        f"<div class='lc-star-cta-text'>"
        f"<div class='lc-star-cta-q'>{text_top}</div>"
        f"<div class='lc-star-cta-q-en'>{text_bottom}</div>"
        f"</div>"
        f"<div class='lc-star-cta-action'>{cta} →</div>"
        "</a>"
    )


def _render_explain(entries: list[ExplainEntry], is_zh: bool) -> str:
    """Render --explain derivation trace as an HTML accordion."""
    if not entries:
        return ""
    blocks = []
    for e in entries:
        inputs_html = ""
        if e.inputs:
            inputs_html = "<ul class='lc-explain-inputs'>" + "".join(
                f"<li><code>{_esc(inp.name)}</code> = "
                f"<strong>{_esc(inp.value)}</strong> "
                f"<span class='lc-explain-label'>{_esc(inp.label)}</span>"
                + (f" — <em>{_esc(inp.note)}</em>" if inp.note else "")
                + "</li>"
                for inp in e.inputs
            ) + "</ul>"
        steps_html = ""
        if e.steps:
            steps_html = "<ol class='lc-explain-steps'>" + "".join(
                f"<li>{_esc(s)}</li>" for s in e.steps
            ) + "</ol>"
        source_html = (
            f"<div class='lc-prov'>{'来源' if is_zh else 'Source'}: {_esc(e.source)}</div>"
            if e.source
            else ""
        )
        blocks.append(
            f"<div class='lc-explain-entry'>"
            f"<div class='lc-explain-heading'>{_esc(e.heading)}</div>"
            f"<div class='lc-explain-formula'><code>{_esc(e.formula)}</code></div>"
            f"{inputs_html}{steps_html}"
            f"<div class='lc-explain-result'>"
            f"{'结果' if is_zh else 'Result'}: <strong>{_esc(e.result)}</strong></div>"
            f"{source_html}"
            f"</div>"
        )
    return (
        "<div class='lc-result'>"
        f"<div class='lc-section'>"
        f"<h3>{'推导链 (--explain)' if is_zh else 'Derivation trace (--explain)'}</h3>"
        + "".join(blocks)
        + "</div></div>"
    )


def _render_llm_review(content: str | None, error: str | None, model: str, is_zh: bool) -> str:
    if error:
        return _render_error(f"LLM review: {error}", is_zh)
    if not content:
        return ""
    # The LLM responds with markdown — convert to a simple HTML block for display.
    # gr.HTML doesn't run markdown, but the LLM's headers (## ...) still read OK as text.
    safe = _esc(content).replace("\n", "<br>")
    return (
        "<div class='lc-result'>"
        f"<div class='lc-section'>"
        f"<h3>{'LLM 审计 (--llm-review)' if is_zh else 'LLM review (--llm-review)'} "
        f"<span class='lc-llm-model'>{_esc(model)}</span></h3>"
        f"<div class='lc-llm-banner'>"
        f"{_label_chip('llm-opinion')} "
        f"{'仅供参考，不覆盖前 6 个 label' if is_zh else 'Second opinion — never overrides the 6 primary labels'}"
        f"</div>"
        f"<div class='lc-llm-content'>{safe}</div>"
        f"</div></div>"
    )


def _render_error(msg: str, is_zh: bool) -> str:
    label = "出错了" if is_zh else "Error"
    return (
        f"<div class='lc-result lc-error'>"
        f"<h3>{label}</h3>"
        f"<pre>{_esc(msg)}</pre></div>"
    )


def _render_loading(is_zh: bool) -> str:
    msg = (
        "正在拉取模型元数据 + 读 safetensors header… 首次大模型约 3-8 秒"
        if is_zh
        else "Fetching model metadata + reading safetensors header… "
        "first lookup of a large model takes 3-8 seconds"
    )
    return (
        "<div class='lc-result lc-loading'>"
        "<div class='lc-spinner'></div>"
        f"<div class='lc-loading-text'>{msg}</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Backend handler

_evaluators: dict[str, Evaluator] = {}


def _get_evaluator(source_key: str) -> Evaluator:
    """One evaluator per source — Evaluator caches an HfApi client internally
    so we don't want to rebuild it every keystroke."""
    if source_key not in _evaluators:
        if source_key == "modelscope":
            _evaluators[source_key] = Evaluator(source=ModelScopeSource())
        else:
            _evaluators[source_key] = Evaluator(source=HuggingFaceSource())
    return _evaluators[source_key]


def calculate(
    model_id: str,
    gpu,  # list[str] from multiselect; str also tolerated  # noqa: ANN001
    engine: str,
    context_length: int | None,
    lang: str,
    source: str,
    gpu_count: int | None,
    input_tokens: int,
    output_tokens: int,
    target_tps: float,
    prefill_util: float,
    decode_bw_util: float,
    concurrency_degradation: float,
    refresh: bool,
    explain: bool,
    llm_review: bool,
    hf_token: str,
    ms_token: str,
    llm_api_key: str,
    llm_base_url: str,
    llm_model: str,
) -> tuple[str, str, str]:
    """Returns (main_html, explain_html, llm_review_html)."""
    locale = "zh" if lang.startswith("中") else "en"
    is_zh = locale == "zh"

    # Normalize GPU input. Multiselect returns list; defensive coerce for safety.
    if isinstance(gpu, str):
        gpu_list = [gpu] if gpu else []
    elif isinstance(gpu, (list, tuple)):
        gpu_list = [g for g in gpu if g]
    else:
        gpu_list = []

    if not model_id or not model_id.strip():
        return (
            _render_error(
                "请输入模型 ID" if is_zh else "Enter a model id",
                is_zh,
            ),
            "",
            "",
        )
    if not gpu_list:
        return (_render_error("请选择 GPU" if is_zh else "Pick a GPU", is_zh), "", "")

    is_compare = len(gpu_list) >= 2

    # Resolve source key. The radio shows e.g. "HuggingFace" / "ModelScope".
    src_key = "modelscope" if "modelscope" in source.lower() else "huggingface"

    # Inject user-provided tokens into env for the duration of this call only.
    # We restore the prior values in the finally block so a token entered for
    # one model doesn't leak into the next request from a different user.
    token_env_keys = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "MODELSCOPE_API_TOKEN",
        "MODELSCOPE_TOKEN",
    )
    old_token_env = {k: os.environ.get(k) for k in token_env_keys}
    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()
    if ms_token and ms_token.strip():
        os.environ["MODELSCOPE_API_TOKEN"] = ms_token.strip()

    def _eval_one(g: str) -> EvaluationReport:
        return _get_evaluator(src_key).evaluate(
            model_id=model_id.strip(),
            gpu=g,
            engine=engine,
            gpu_count=gpu_count if gpu_count and gpu_count > 0 else None,
            context_length=context_length if context_length and context_length > 0 else None,
            refresh=refresh,
            input_tokens=int(input_tokens) if input_tokens else 2000,
            output_tokens=int(output_tokens) if output_tokens else 512,
            target_tokens_per_sec=float(target_tps) if target_tps else 30.0,
            prefill_utilization=float(prefill_util) if prefill_util else 0.40,
            decode_bw_utilization=float(decode_bw_util) if decode_bw_util else 0.50,
            concurrency_degradation=(
                float(concurrency_degradation) if concurrency_degradation else 1.0
            ),
        )

    try:
        # ---- Compare path: 2-4 GPUs --------------------------------------
        if is_compare:
            try:
                reports = [_eval_one(g) for g in gpu_list]
            except Exception as e:  # noqa: BLE001
                return (_render_error(f"{type(e).__name__}: {e}", is_zh), "", "")
            return _render_compare(reports, locale), "", ""

        # ---- Single-GPU path (existing flow) ------------------------------
        try:
            report = _eval_one(gpu_list[0])
        except Exception as e:  # noqa: BLE001
            return (_render_error(f"{type(e).__name__}: {e}", is_zh), "", "")

        main_html = _render(report, locale)
        explain_html = ""
        llm_html = ""

        if explain or llm_review:
            entries = build_explain(report)
            if explain:
                explain_html = _render_explain(entries, is_zh)
            if llm_review:
                # Only set env vars if user actually provided them — never persist
                # them in env beyond this call's scope (they live in process env
                # for the duration of the call, but we don't persist to disk).
                old_env = {
                    "LLM_CAL_REVIEWER_API_KEY": os.environ.get("LLM_CAL_REVIEWER_API_KEY"),
                    "LLM_CAL_REVIEWER_BASE_URL": os.environ.get("LLM_CAL_REVIEWER_BASE_URL"),
                    "LLM_CAL_REVIEWER_MODEL": os.environ.get("LLM_CAL_REVIEWER_MODEL"),
                }
                try:
                    if llm_api_key.strip():
                        os.environ["LLM_CAL_REVIEWER_API_KEY"] = llm_api_key.strip()
                    if llm_base_url.strip():
                        os.environ["LLM_CAL_REVIEWER_BASE_URL"] = llm_base_url.strip()
                    if llm_model.strip():
                        os.environ["LLM_CAL_REVIEWER_MODEL"] = llm_model.strip()
                    result = run_review(entries, locale=locale)  # type: ignore[arg-type]
                finally:
                    for k, v in old_env.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                llm_html = _render_llm_review(result.content, result.error, result.model, is_zh)

        return main_html, explain_html, llm_html
    finally:
        for k, v in old_token_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def show_loading(lang: str) -> tuple[str, str, str]:
    is_zh = lang.startswith("中")
    return _render_loading(is_zh), "", ""


# ---------------------------------------------------------------------------
# UI

THEME = gr.themes.Soft(primary_hue="indigo")

HERO_HTML = """
<div class='lc-hero'>
  <div class='lc-hero-top'>
    <div class='lc-hero-titleblock'>
      <div class='lc-hero-title'>llm-cal</div>
      <div class='lc-hero-tagline'>
        LLM inference hardware calculator · 大模型推理硬件计算器<br>
        Architecture-aware · Engine-aware · <strong>Honest-labeled</strong>
      </div>
    </div>
    <a class='lc-hero-gh' href='https://github.com/FlyTOmeLight/llm-cal' target='_blank' rel='noopener'>
      <svg viewBox='0 0 16 16' width='16' height='16' aria-hidden='true' fill='currentColor'>
        <path d='M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38v-1.33c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.81.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.83-2.15-.08-.2-.36-1.02.08-2.13 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.11.16 1.93.08 2.13.51.56.83 1.27.83 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.19c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z'/>
      </svg>
      <span class='lc-hero-gh-text'>GitHub</span>
      <img class='lc-hero-gh-stars' alt='stars'
        src='https://img.shields.io/github/stars/FlyTOmeLight/llm-cal?style=flat-square&logo=&label=&color=eef2ff&labelColor=eef2ff'
        loading='lazy' />
    </a>
  </div>
  <div class='lc-hero-pitch'>
    <div class='lc-pitch-card lc-pitch-bad'>
      <div class='lc-pitch-tool'>gpu_poor</div>
      <div class='lc-pitch-num-bad'>284 GB</div>
      <div class='lc-pitch-method'>assumes pure FP8 · 假设纯 FP8</div>
    </div>
    <div class='lc-pitch-arrow'>→</div>
    <div class='lc-pitch-card lc-pitch-good'>
      <div class='lc-pitch-tool'>llm-cal</div>
      <div class='lc-pitch-num-good'>160 GB</div>
      <div class='lc-pitch-method'>reads real safetensors bytes · 读真实字节</div>
    </div>
    <div class='lc-pitch-summary'>
      <div class='lc-pitch-model'>DeepSeek-V4-Flash · H800</div>
      <div class='lc-pitch-result'>0.2% error vs 45% · 误差 0.2% vs 45%</div>
    </div>
  </div>
</div>
"""


CUSTOM_CSS = """
/* Font stack — system fonts in both English + Chinese, no Gradio default serif */
* {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue",
    "PingFang SC", "Microsoft YaHei", "Segoe UI", Roboto, Arial, sans-serif !important;
}

/* Hide Gradio's default footer chrome that looks like part of our app */
footer { display: none !important; }
.show-api, .built-with, .settings { display: none !important; }

/* Tighter overall padding + center on wide screens — without margin:auto the
   container left-aligns and leaves ~800px empty on 1920+ displays.
   width:100% makes it shrink to viewport when narrower than max-width
   (otherwise on mobile align-items:stretch + max-width overflows). */
.gradio-container {
  max-width: 1100px !important;
  width: 100% !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Hero section */
.lc-hero {
  margin: 8px 0 24px 0;
  padding: 24px 0 18px 0;
  border-bottom: 1px solid #e5e7eb;
}
.dark .lc-hero { border-bottom-color: #374151; }

/* Top row: title block (left) + GitHub link (right). On mobile the GH link
   wraps to its own line above or below the title — order kept so it stays
   visible above the fold. */
.lc-hero-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.lc-hero-titleblock {
  flex: 1 1 320px;
  min-width: 0;
}
.lc-hero-gh {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border: 1px solid #c7d2fe;
  background: #eef2ff;
  border-radius: 999px;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: #4338ca !important;
  text-decoration: none !important;
  white-space: nowrap;
  transition: background 0.15s ease, border-color 0.15s ease;
  flex: 0 0 auto;
}
.lc-hero-gh:hover {
  background: #e0e7ff;
  border-color: #a5b4fc;
}
.dark .lc-hero-gh {
  background: #1e1b4b;
  border-color: #3730a3;
  color: #c7d2fe !important;
}
.dark .lc-hero-gh:hover { background: #312e81; border-color: #4338ca; }
.lc-hero-gh svg { display: block; }
.lc-hero-gh-stars {
  height: 18px;
  vertical-align: middle;
  border-radius: 4px;
}

.lc-hero-title {
  font-size: 32px !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em;
  color: #0f172a !important;
  margin: 0 !important;
  line-height: 1.15;
}
.dark .lc-hero-title { color: #f8fafc !important; }
.lc-hero-tagline {
  font-size: 16px !important;
  color: #6b7280 !important;
  margin: 6px 0 16px 0;
  line-height: 1.5;
}
.lc-hero-pitch {
  display: grid;
  /* 4 cells: bad-card / arrow / good-card / summary on wide screens */
  grid-template-columns: 1fr 30px 1fr 1.2fr;
  gap: 14px;
  align-items: stretch;
  padding: 0;
  font-size: 13px !important;
  color: #1e293b !important;
}
.dark .lc-hero-pitch { color: #f1f5f9 !important; }

/* Tablet: bad / arrow / good in row 1, summary full-width row 2 */
@media (max-width: 900px) {
  .lc-hero-pitch {
    grid-template-columns: 1fr 28px 1fr;
    grid-template-rows: auto auto;
  }
  .lc-pitch-summary { grid-column: 1 / -1; }
}

/* Mobile: stack everything, hide the arrow */
@media (max-width: 540px) {
  .lc-hero-pitch {
    grid-template-columns: 1fr;
    grid-template-rows: repeat(3, auto);
  }
  .lc-pitch-arrow { display: none; }
  .lc-pitch-summary { grid-column: auto; }
}

.lc-pitch-card {
  padding: 14px 18px;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-width: 0;
}
.dark .lc-pitch-card { background: #111827; border-color: #374151; }
/* Subtle accent bar on the left, not a screaming red/green border */
.lc-pitch-bad  { border-left: 3px solid #cbd5e1; }
.lc-pitch-good { border-left: 3px solid #4f46e5; }
.dark .lc-pitch-bad  { border-left-color: #475569; }
.dark .lc-pitch-good { border-left-color: #818cf8; }

.lc-pitch-tool {
  font-size: 12px !important;
  font-weight: 600 !important;
  color: #6b7280 !important;
  font-family: "SF Mono", "JetBrains Mono", Menlo, monospace !important;
  margin-bottom: 4px;
}
.lc-pitch-num-bad  { font-size: 24px !important; font-weight: 800 !important; color: #b91c1c !important; line-height: 1.1; letter-spacing: -0.01em; }
.lc-pitch-num-good { font-size: 24px !important; font-weight: 800 !important; color: #15803d !important; line-height: 1.1; letter-spacing: -0.01em; }
.dark .lc-pitch-num-bad  { color: #f87171 !important; }
.dark .lc-pitch-num-good { color: #4ade80 !important; }
.lc-pitch-method {
  font-size: 11px !important;
  color: #6b7280 !important;
  margin-top: 6px;
  line-height: 1.4;
}

.lc-pitch-arrow {
  display: flex;
  align-items: center;
  font-size: 22px !important;
  color: #9ca3af !important;
  font-weight: 300;
}

.lc-pitch-summary {
  flex: 1 1 200px;
  padding: 14px 18px;
  border-radius: 10px;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.dark .lc-pitch-summary { background: #1e1b4b; border-color: #3730a3; }
.lc-pitch-model {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #6366f1 !important;
  margin-bottom: 4px;
}
.dark .lc-pitch-model { color: #a5b4fc !important; }
.lc-pitch-result {
  font-size: 14px !important;
  font-weight: 700 !important;
  color: #312e81 !important;
}
.dark .lc-pitch-result { color: #e0e7ff !important; }

/* Primary button — match the indigo theme; constrain width so it's not a billboard */
button.primary,
button[variant="primary"],
.primary > button {
  background: #4f46e5 !important;
  border-color: #4f46e5 !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
  border-radius: 8px !important;
  padding: 10px 28px !important;
}
button.primary:hover,
button[variant="primary"]:hover,
.primary > button:hover { background: #4338ca !important; border-color: #4338ca !important; }

/* The wrapper around the Calculate button — center it, give it sane width */
.lc-submit-wrap {
  display: flex !important;
  justify-content: center !important;
  margin: 20px 0 8px 0 !important;
}
.lc-submit-wrap button {
  min-width: 220px !important;
  max-width: 320px !important;
  width: auto !important;
}

/* Form labels — kill Gradio's purple chip; make labels plain uppercase small text */
[data-testid="block-info"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 0 6px 0 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  color: #6b7280 !important;
  border-radius: 0 !important;
  display: block !important;
}
.dark [data-testid="block-info"] { color: #9ca3af !important; }

/* Tooltip / info-text — single line, secondary color, no italic */
.info-text {
  font-size: 11px !important;
  color: #94a3b8 !important;
  margin: 0 0 4px 0 !important;
  line-height: 1.4 !important;
  padding: 0 !important;
  font-style: normal !important;
  white-space: normal !important;
}
.info-text br { display: none !important; }
.dark .info-text { color: #64748b !important; }

/* Kill Gradio's grey form-panel chrome entirely — labels + inputs float on the page */
.block,
.block.padded,
.block.gradio-container,
.form,
.row,
[data-testid="block"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.block.padded { padding: 6px 0 !important; }
.form { padding: 0 !important; }
.row { padding: 0 !important; }

/* Tighten row gap so inputs cluster more naturally */
.form, .row { gap: 16px !important; }

/* Tablet (≤900px): Gradio's gr.Row() flex-direction: row keeps 3 inputs
   in one line. min-width: 320px forces 3-column rows to wrap to 2x1 +
   1x1 at this size while leaving 2-column rows at 2-up. */
@media (max-width: 900px) {
  .form,
  .row {
    flex-wrap: wrap !important;
  }
  .form > .block,
  .row > .block {
    flex: 1 1 calc(50% - 12px) !important;
    min-width: 320px !important;
    max-width: 100% !important;
  }
}

/* Mobile (≤540px): single-column form. */
@media (max-width: 540px) {
  .form,
  .row {
    flex-direction: column !important;
  }
  .form > .block,
  .row > .block {
    flex: 1 1 100% !important;
    min-width: 0 !important;
    width: 100% !important;
  }
  .gradio-container { padding: 12px !important; }
  .lc-hero-title { font-size: 26px !important; }
  .lc-pitch-num-bad, .lc-pitch-num-good { font-size: 22px !important; }
  .lc-pitch-arrow { display: none !important; }
}

/* Inputs themselves — light border, soft fill */
input[type="text"],
input[type="number"],
input[type="password"],
textarea,
select {
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  font-size: 14px !important;
  padding: 10px 12px !important;
}
.dark input,
.dark textarea,
.dark select {
  background: #111827 !important;
  border-color: #374151 !important;
}
input:focus,
textarea:focus {
  border-color: #4f46e5 !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}

/* Accordion — Gradio 6 has no .accordion class; the only signal is a .block
   that *contains* a button.label-wrap. Use :has() to match precisely. */
.block.padded:has(> button.label-wrap) {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;
  margin: 14px 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
.dark .block.padded:has(> button.label-wrap) {
  background: #111827 !important;
  border-color: #374151 !important;
}
button.label-wrap {
  background: #f8fafc !important;
  padding: 14px 18px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  color: #1f2937 !important;
  width: 100% !important;
  text-align: left !important;
  cursor: pointer !important;
  border: none !important;
  border-bottom: 1px solid #e5e7eb !important;
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  letter-spacing: 0.01em;
}
.dark button.label-wrap {
  background: #1e293b !important;
  color: #f1f5f9 !important;
  border-bottom-color: #374151 !important;
}
button.label-wrap:hover { background: #f1f5f9 !important; }
.dark button.label-wrap:hover { background: #334155 !important; }
/* Sibling content of the header (the body when expanded) */
.block.padded:has(> button.label-wrap) > *:not(button.label-wrap) {
  padding: 16px 18px !important;
  background: #ffffff !important;
}
.dark .block.padded:has(> button.label-wrap) > *:not(button.label-wrap) {
  background: #111827 !important;
}

/* gr.Examples table — the default Gradio render is a raw HTML table with black
   borders and no hover state. Style it to match the rest of the page. */
.gradio-dataset,
[data-testid="dataset"] {
  margin-top: 24px !important;
  background: transparent !important;
  border: none !important;
}
.gradio-dataset table,
[data-testid="dataset"] table {
  border-collapse: collapse !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  font-size: 13px !important;
  width: 100% !important;
}
.dark .gradio-dataset table,
.dark [data-testid="dataset"] table { border-color: #374151 !important; }
.gradio-dataset thead,
[data-testid="dataset"] thead { background: #f9fafb !important; }
.dark .gradio-dataset thead,
.dark [data-testid="dataset"] thead { background: #111827 !important; }
.gradio-dataset th,
[data-testid="dataset"] th {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  color: #6b7280 !important;
  text-align: left !important;
  padding: 10px 12px !important;
  border: none !important;
  border-bottom: 1px solid #e5e7eb !important;
}
.gradio-dataset td,
[data-testid="dataset"] td {
  padding: 9px 12px !important;
  border: none !important;
  border-bottom: 1px solid #f3f4f6 !important;
  color: #1f2937 !important;
  font-size: 13px !important;
  background: transparent !important;
  cursor: pointer !important;
}
.dark .gradio-dataset td,
.dark [data-testid="dataset"] td {
  color: #e5e7eb !important;
  border-bottom-color: #1f2937 !important;
}
.gradio-dataset tbody tr:last-child td,
[data-testid="dataset"] tbody tr:last-child td { border-bottom: none !important; }
.gradio-dataset tbody tr:hover,
[data-testid="dataset"] tbody tr:hover { background: rgba(79, 70, 229, 0.04) !important; }
.dark .gradio-dataset tbody tr:hover,
.dark [data-testid="dataset"] tbody tr:hover { background: rgba(129, 140, 248, 0.08) !important; }

/* Examples header label — Gradio puts a "Try one of these" label above */
.gradio-dataset > .label,
[data-testid="dataset"] > .label,
.gradio-dataset .block-label,
.dataset .block-label {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  color: #6b7280 !important;
  background: transparent !important;
  border: none !important;
  padding: 0 0 6px 0 !important;
  margin-bottom: 0 !important;
}

/* Footer link strip */
.lc-footer {
  margin-top: 28px;
  padding: 14px 0;
  border-top: 1px solid #e5e7eb;
  font-size: 13px !important;
  color: #6b7280 !important;
}
.dark .lc-footer { border-top-color: #374151; }
.lc-footer a { color: #4f46e5 !important; text-decoration: none; }
.lc-footer a:hover { text-decoration: underline; }
.dark .lc-footer a { color: #818cf8 !important; }

/* Result wrapper */
.lc-result {
  padding: 4px 0;
  font-size: 14px;
  line-height: 1.55;
  color: #111827 !important;
}
.dark .lc-result { color: #f3f4f6 !important; }

/* Headline */
.lc-header { padding: 4px 0 14px 0; border-bottom: 1px solid #e5e7eb; }
.dark .lc-header { border-bottom-color: #374151; }
.lc-title {
  font-size: 22px !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
  color: #0f172a !important;
}
.dark .lc-title { color: #f8fafc !important; }
.lc-subtitle {
  font-size: 13px !important;
  color: #6b7280 !important;
  margin-top: 2px;
}

/* Headline stat cards */
.lc-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 16px 0 8px 0;
}
.lc-stat {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 14px 16px;
  background: #ffffff;
}
.dark .lc-stat { background: #111827; border-color: #374151; }
.lc-stat-value {
  font-size: 24px !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
  line-height: 1.2;
  color: #0f172a !important;
}
.dark .lc-stat-value { color: #f8fafc !important; }
.lc-stat-label {
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #6b7280 !important;
  margin-top: 4px;
  font-weight: 500 !important;
}
.lc-stat-sub {
  font-size: 11px !important;
  color: #9ca3af !important;
  margin-top: 2px;
}
.lc-stat-chip { margin-top: 10px; }

.lc-chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
}

.lc-prov {
  margin-top: 6px;
  font-size: 12px !important;
  color: #6b7280 !important;
  font-style: italic;
}

/* Sections */
.lc-section { margin: 24px 0 0 0; }
.lc-section h3 {
  font-size: 13px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #6b7280 !important;
  margin: 0 0 6px 0 !important;
}
.lc-section-help {
  font-size: 12px !important;
  color: #6b7280 !important;
  margin: 0 0 10px 0;
  line-height: 1.5;
}

/* Tables */
.lc-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px !important;
  color: #111827 !important;
}
.dark .lc-table { color: #f3f4f6 !important; }
.lc-table th, .lc-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #f3f4f6;
  text-align: left;
}
.dark .lc-table th, .dark .lc-table td { border-bottom-color: #1f2937; }
.lc-table th {
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #6b7280 !important;
  font-weight: 500 !important;
}
.lc-table-recon td:nth-child(2),
.lc-table-recon td:nth-child(3) { text-align: right; }
.lc-best { background: rgba(22, 163, 74, 0.08); }
.dark .lc-best { background: rgba(22, 163, 74, 0.18); }

/* Performance grid */
.lc-perf {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}
.lc-perf-item {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 12px 14px;
  background: #ffffff;
}
.dark .lc-perf-item { border-color: #374151; background: #111827; }
.lc-perf-value {
  font-size: 20px !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
  color: #0f172a !important;
  line-height: 1.2;
}
.dark .lc-perf-value { color: #f8fafc !important; }
.lc-perf-value code {
  font-size: 16px !important;
  font-weight: 600 !important;
  background: transparent !important;
  color: #0f172a !important;
  padding: 0 !important;
}
.dark .lc-perf-value code { color: #f8fafc !important; }
.lc-perf-label {
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #6b7280 !important;
  margin-top: 4px;
  font-weight: 500 !important;
}
.lc-perf-sub {
  font-size: 11px !important;
  color: #9ca3af !important;
  margin-top: 1px;
}

/* Inline code */
.lc-result code {
  font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important;
  font-size: 0.92em !important;
  color: #0f172a !important;
  background: rgba(15, 23, 42, 0.06);
  padding: 1px 5px;
  border-radius: 4px;
}
.dark .lc-result code {
  color: #e2e8f0 !important;
  background: rgba(226, 232, 240, 0.08);
}

/* Generated command — ALWAYS dark theme regardless of mode */
.lc-cmd {
  background: #0b1220 !important;
  color: #f1f5f9 !important;
  padding: 16px 18px !important;
  border-radius: 8px;
  font-size: 12.5px !important;
  overflow-x: auto;
  white-space: pre;
  border: 1px solid #1e293b !important;
  margin: 0 !important;
}
.lc-cmd code {
  font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important;
  background: transparent !important;
  color: #f1f5f9 !important;
  padding: 0 !important;
  font-size: 12.5px !important;
  border-radius: 0 !important;
}

/* Comparison view — side-by-side metrics across GPUs */
.lc-cmp-wrap {
  overflow-x: auto;
  margin: 8px 0 12px 0;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  background: #ffffff;
}
.dark .lc-cmp-wrap { background: #111827; border-color: #374151; }
.lc-cmp-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px !important;
}
.lc-cmp-table th,
.lc-cmp-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #f3f4f6;
}
.dark .lc-cmp-table th,
.dark .lc-cmp-table td { border-bottom-color: #1f2937; }
.lc-cmp-table thead th {
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #6b7280 !important;
  font-weight: 600 !important;
  background: #f9fafb;
}
.dark .lc-cmp-table thead th { background: #1e293b; color: #9ca3af !important; }
.lc-cmp-row-label {
  font-size: 12px !important;
  color: #6b7280 !important;
  font-weight: 600 !important;
  white-space: nowrap;
}
.lc-cmp-gpu {
  font-family: "SF Mono", "JetBrains Mono", Menlo, monospace !important;
  font-size: 12px !important;
}
.lc-cmp-table tbody tr:last-child td { border-bottom: none; }
.lc-cmp-winner {
  background: rgba(22, 163, 74, 0.10) !important;
  font-weight: 700 !important;
  color: #15803d !important;
  position: relative;
}
.dark .lc-cmp-winner { background: rgba(74, 222, 128, 0.15) !important; color: #4ade80 !important; }
.lc-cmp-winner::before {
  content: "✓ ";
  font-size: 11px;
  font-weight: 700;
  color: #15803d;
  margin-right: 2px;
}
.dark .lc-cmp-winner::before { color: #4ade80; }
.lc-cmp-summary {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: 8px;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  font-size: 13px !important;
  color: #312e81 !important;
}
.dark .lc-cmp-summary {
  background: #1e1b4b;
  border-color: #3730a3;
  color: #e0e7ff !important;
}
.lc-cmp-summary strong { color: #4338ca; }
.dark .lc-cmp-summary strong { color: #a5b4fc; }

/* Per-GPU detail cards under the table */
.lc-cmp-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}
.lc-cmp-detail {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 12px 14px;
  background: #ffffff;
}
.dark .lc-cmp-detail { background: #111827; border-color: #374151; }
.lc-cmp-detail-gpu {
  font-family: "SF Mono", "JetBrains Mono", Menlo, monospace !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  color: #0f172a !important;
  margin-bottom: 6px;
  padding-bottom: 6px;
  border-bottom: 1px solid #e5e7eb;
}
.dark .lc-cmp-detail-gpu { color: #f8fafc !important; border-bottom-color: #374151; }
.lc-cmp-detail-row {
  display: flex;
  justify-content: space-between;
  font-size: 12px !important;
  padding: 3px 0;
}
.lc-cmp-detail-row span { color: #6b7280 !important; }
.lc-cmp-detail-row strong {
  color: #0f172a !important;
  font-size: 13px !important;
}
.dark .lc-cmp-detail-row strong { color: #f8fafc !important; }

/* Star-on-GitHub CTA — shown at the bottom of the result, capturing the
   peak-satisfaction moment. Card-style with indigo accent so it reads as
   "thanks", not as a banner ad. */
.lc-star-cta {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 28px 0 8px 0;
  padding: 14px 18px;
  border: 1px solid #c7d2fe;
  background: #eef2ff;
  border-radius: 10px;
  text-decoration: none !important;
  color: #312e81 !important;
  transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
}
.lc-star-cta:hover {
  background: #e0e7ff;
  border-color: #a5b4fc;
}
.lc-star-cta:active { transform: scale(0.995); }
.dark .lc-star-cta {
  background: #1e1b4b;
  border-color: #3730a3;
  color: #c7d2fe !important;
}
.dark .lc-star-cta:hover { background: #312e81; }
.lc-star-cta svg { flex: 0 0 auto; color: #4338ca; }
.dark .lc-star-cta svg { color: #a5b4fc; }
.lc-star-cta-text { flex: 1 1 auto; min-width: 0; }
.lc-star-cta-q {
  font-size: 14px !important;
  font-weight: 600 !important;
  line-height: 1.3;
  color: #312e81 !important;
}
.dark .lc-star-cta-q { color: #e0e7ff !important; }
.lc-star-cta-q-en {
  font-size: 12px !important;
  color: #6366f1 !important;
  margin-top: 2px;
  line-height: 1.3;
}
.dark .lc-star-cta-q-en { color: #a5b4fc !important; }
.lc-star-cta-action {
  flex: 0 0 auto;
  font-size: 13px !important;
  font-weight: 700 !important;
  color: #4338ca !important;
  white-space: nowrap;
}
.dark .lc-star-cta-action { color: #c7d2fe !important; }
@media (max-width: 540px) {
  .lc-star-cta { flex-wrap: wrap; gap: 10px; }
  .lc-star-cta-action { flex-basis: 100%; }
}

/* Loading + error */
.lc-loading {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 24px;
  color: #6b7280 !important;
  font-size: 14px !important;
}
.lc-spinner {
  width: 18px; height: 18px;
  border: 2px solid #cbd5e1;
  border-top-color: #4f46e5;
  border-radius: 50%;
  animation: lc-spin 0.7s linear infinite;
  flex: none;
}
@keyframes lc-spin { to { transform: rotate(360deg); } }

.lc-error pre {
  background: #fef2f2;
  color: #991b1b !important;
  padding: 12px 14px;
  border-radius: 8px;
  border: 1px solid #fecaca;
  font-size: 12px !important;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}
.dark .lc-error pre { background: #450a0a; color: #fca5a5 !important; border-color: #7f1d1d; }

/* Explain trace */
.lc-explain-entry {
  margin: 14px 0;
  padding: 14px 16px;
  border: 1px solid #e5e7eb;
  border-left: 3px solid #4f46e5;
  border-radius: 8px;
  background: #fafafa;
}
.dark .lc-explain-entry { background: #0f172a; border-color: #374151; border-left-color: #818cf8; }
.lc-explain-heading {
  font-weight: 700 !important;
  font-size: 14px !important;
  margin-bottom: 8px;
  color: #0f172a !important;
}
.dark .lc-explain-heading { color: #f8fafc !important; }
.lc-explain-formula {
  margin: 6px 0;
  font-size: 12.5px !important;
}
.lc-explain-formula code {
  background: rgba(79, 70, 229, 0.08) !important;
  color: #4338ca !important;
  padding: 4px 8px !important;
  border-radius: 4px;
}
.dark .lc-explain-formula code { color: #a5b4fc !important; background: rgba(165, 180, 252, 0.12) !important; }
.lc-explain-inputs, .lc-explain-steps {
  margin: 6px 0 6px 1.2em;
  font-size: 12.5px !important;
  line-height: 1.7;
}
.lc-explain-label {
  font-size: 11px !important;
  color: #6b7280 !important;
  font-style: italic;
}
.lc-explain-result {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #e5e7eb;
  font-size: 13px !important;
  color: #0f172a !important;
}
.dark .lc-explain-result { color: #f8fafc !important; border-top-color: #374151; }

/* LLM review */
.lc-llm-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  font-size: 12px !important;
  color: #4b5563 !important;
  margin-bottom: 12px;
}
.dark .lc-llm-banner { color: #d1d5db !important; background: #111827; border-color: #374151; }
.lc-llm-model {
  font-size: 11px !important;
  color: #6b7280 !important;
  font-weight: 500 !important;
  margin-left: 6px;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
.lc-llm-content {
  font-size: 13px !important;
  line-height: 1.7;
  color: #0f172a !important;
  padding: 12px 14px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
}
.dark .lc-llm-content { color: #f3f4f6 !important; background: #111827; border-color: #374151; }
"""


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="llm-cal — LLM hardware calculator") as demo:
        gr.HTML(HERO_HTML)

        # ---- Required ----------------------------------------------------
        with gr.Row():
            model_id = gr.Textbox(
                label="Model ID · 模型 ID",
                placeholder="e.g. deepseek-ai/DeepSeek-V4-Flash",
                info="Repo id · 仓库 ID（owner/name）",
                scale=3,
            )
            source = gr.Radio(
                choices=["HuggingFace", "ModelScope"],
                value="HuggingFace",
                label="Source · 来源",
                info="Where to pull model metadata · 拉取来源",
                scale=2,
            )

        with gr.Row():
            vendor = gr.Dropdown(
                choices=VENDOR_CHOICES_EN,
                value=DEFAULT_VENDOR,
                label="GPU vendor · GPU 厂商",
                info="11 vendors covered · 共 11 家",
                scale=1,
            )
            gpu = gr.Dropdown(
                choices=_VENDOR_TO_GPUS[DEFAULT_VENDOR],
                value=[DEFAULT_GPU],
                label="GPU model · GPU 型号",
                info="One GPU = single eval. 2-4 = compare side-by-side · 选 1 张单评估，2-4 张对比",
                scale=2,
                multiselect=True,
                max_choices=4,
                allow_custom_value=True,
            )

        with gr.Row():
            engine = gr.Radio(
                choices=["vllm", "sglang"],
                value="vllm",
                label="Engine · 引擎",
                info="Inference engine · 推理引擎",
            )
            context_length = gr.Number(
                label="Context length · Context 长度",
                value=None,
                precision=0,
                info="Empty = 4K/32K/128K/1M · 留空显示全档",
            )
            lang = gr.Radio(
                choices=["English", "中文"],
                value="English",
                label="Output language · 输出语言",
                info="Result area only · 仅影响下方结果区",
            )

        # ---- Performance tuning (collapsible) ----------------------------
        with gr.Accordion("Performance tuning · 性能参数", open=False):
            with gr.Row():
                input_tokens = gr.Number(
                    label="Input tokens · 输入 tokens",
                    value=2000,
                    precision=0,
                    info="Prefill budget · Prefill 预算",
                )
                output_tokens = gr.Number(
                    label="Output tokens · 输出 tokens",
                    value=512,
                    precision=0,
                    info="Decode budget · Decode 预算",
                )
                target_tps = gr.Number(
                    label="Target tok/s/user · 单用户目标 tok/s",
                    value=30.0,
                    info="SLA per user · 单用户 SLA（30 ≈ 流畅阅读）",
                )
            with gr.Row():
                prefill_util = gr.Number(
                    label="Prefill util · Prefill 利用率",
                    value=0.40,
                    info="0–1 · 0.40 = vLLM paper baseline",
                )
                decode_bw_util = gr.Number(
                    label="Decode BW util · Decode 带宽利用率",
                    value=0.50,
                    info="0–1 · 0.50 = community median",
                )
                concurrency_degradation = gr.Number(
                    label="Concurrency degradation · 并发衰减",
                    value=1.0,
                    info="1.0 = honest · 1.67 = 60% efficiency under load",
                )

        # ---- Advanced (collapsible) --------------------------------------
        with gr.Accordion("Advanced · 高级", open=False):
            with gr.Row():
                hf_token = gr.Textbox(
                    label="HF_TOKEN",
                    value="",
                    placeholder="hf_...",
                    type="password",
                    info="For gated HF models · 私有 HF 模型用",
                )
                ms_token = gr.Textbox(
                    label="MODELSCOPE_API_TOKEN",
                    value="",
                    placeholder="ms-...",
                    type="password",
                    info="For gated MS models · 私有 MS 模型用",
                )
            with gr.Row():
                gpu_count = gr.Number(
                    label="Force GPU count · 强制 GPU 数",
                    value=None,
                    precision=0,
                    info="Empty = auto min/dev/prod · 留空自动给三档",
                )
                refresh = gr.Checkbox(
                    label="Refresh cache · 刷新缓存",
                    value=False,
                    info="Bypass diskcache · 跳过本地缓存",
                )
            with gr.Row():
                explain = gr.Checkbox(
                    label="--explain · 推导链",
                    value=False,
                    info="Full derivation trace · 输出完整推导链",
                )
                llm_review = gr.Checkbox(
                    label="--llm-review · LLM 审计",
                    value=False,
                    info="Second opinion from an LLM · 第二意见审计",
                )
            with gr.Row():
                llm_api_key = gr.Textbox(
                    label="LLM API key · LLM API 密钥",
                    value="",
                    placeholder="sk-...",
                    type="password",
                    info="OpenAI-compatible endpoint · OpenAI 兼容端点",
                )
                llm_base_url = gr.Textbox(
                    label="LLM base URL · LLM 基地址",
                    value="",
                    placeholder="https://api.openai.com/v1",
                    info="e.g. https://api.deepseek.com/v1",
                )
                llm_model = gr.Textbox(
                    label="LLM model · LLM 模型名",
                    value="",
                    placeholder="gpt-4o",
                    info="e.g. gpt-4o / deepseek-chat / MiniMax-M2",
                )

        with gr.Row(elem_classes="lc-submit-wrap"):
            submit = gr.Button("Calculate · 计算", variant="primary", size="lg")

        # Three output panes — main always shows, explain/llm-review only when toggled
        output_main = gr.HTML(label="Result")
        output_explain = gr.HTML(label="Explain trace")
        output_llm = gr.HTML(label="LLM review")

        gr.Examples(
            examples=[
                # gpu wrapped in a list — the Dropdown is multiselect now
                [m, v, [g], e, None, "English", s]
                for m, v, g, e, s in EXAMPLE_MODELS
            ],
            inputs=[model_id, vendor, gpu, engine, context_length, lang, source],
            label="Try one of these · 试试这些组合",
        )

        gr.HTML(
            "<div class='lc-footer'>"
            "<a href='https://github.com/FlyTOmeLight/llm-cal' target='_blank'>GitHub</a> · "
            "<a href='https://flytomelight.github.io/llm-cal/' target='_blank'>Docs</a> · "
            "<a href='https://flytomelight.github.io/llm-cal/methodology/' target='_blank'>Methodology</a> · "
            "<code>pip install llm-cal</code>"
            "</div>"
        )

        # When vendor changes, repopulate the GPU dropdown but PRESERVE any
        # cross-vendor selections (the whole point of compare mode is to
        # stack e.g. H800 + MI300X + 910B4 across NVIDIA/AMD/Ascend).
        def _on_vendor_change(v: str, current):  # noqa: ANN001, ANN202
            gpus = _VENDOR_TO_GPUS.get(v, [])
            # multiselect returns list; harden against str/None for safety
            if isinstance(current, list):
                keep = list(current)
            elif current:
                keep = [current]
            else:
                keep = []
            # Empty selection? Seed with the first GPU so the form stays usable.
            if not keep:
                keep = [gpus[0]] if gpus else []
            return gr.Dropdown(choices=gpus, value=keep)

        vendor.change(fn=_on_vendor_change, inputs=[vendor, gpu], outputs=[gpu])

        # Click flow: instantly show "loading…", THEN run calculate.
        all_outputs = [output_main, output_explain, output_llm]
        submit.click(
            fn=show_loading,
            inputs=[lang],
            outputs=all_outputs,
        ).then(
            fn=calculate,
            inputs=[
                model_id, gpu, engine, context_length, lang, source,
                gpu_count, input_tokens, output_tokens, target_tps,
                prefill_util, decode_bw_util, concurrency_degradation,
                refresh, explain, llm_review,
                hf_token, ms_token,
                llm_api_key, llm_base_url, llm_model,
            ],
            outputs=all_outputs,
        )

    return demo


def _prewarm_cache() -> None:
    """Fill the artifact cache for every Examples row so first-click users
    don't pay the 3-8s HF/MS metadata roundtrip.

    Runs on a daemon thread alongside the Gradio server. Failures are
    swallowed (printed only) — pre-warm is a UX nicety, never a hard
    dependency. Set LLM_CAL_PREWARM=0 to disable (useful for local dev
    when you don't want 9 API calls every time you `python web/app.py`).
    """
    import time

    print(f"[prewarm] starting cache warm-up for {len(EXAMPLE_MODELS)} examples")
    for i, (model_id, _vendor, gpu, engine, source) in enumerate(EXAMPLE_MODELS, 1):
        src_key = "modelscope" if "modelscope" in source.lower() else "huggingface"
        label = f"{i}/{len(EXAMPLE_MODELS)} {src_key}:{model_id}"
        try:
            t0 = time.monotonic()
            _get_evaluator(src_key).evaluate(
                model_id=model_id,
                gpu=gpu,
                engine=engine,
            )
            print(f"[prewarm] {label} ok ({time.monotonic() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[prewarm] {label} skip — {type(e).__name__}: {e}")
        # Throttle to stay well under HF/MS anonymous rate limits.
        time.sleep(2)
    print("[prewarm] done")


if __name__ == "__main__":
    if os.environ.get("LLM_CAL_PREWARM", "1") == "1":
        import threading

        threading.Thread(target=_prewarm_cache, daemon=True).start()
    _build_ui().launch(theme=THEME, css=CUSTOM_CSS)
