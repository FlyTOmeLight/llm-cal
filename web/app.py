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

# Ensure src/ is importable when run as a script (Spaces sets cwd to repo root,
# but we also support `python web/app.py` from the repo root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import os  # noqa: E402

import gradio as gr  # noqa: E402

from llm_cal.common.i18n import set_locale, t  # noqa: E402
from llm_cal.core.evaluator import EvaluationReport, Evaluator  # noqa: E402
from llm_cal.core.explain import ExplainEntry  # noqa: E402
from llm_cal.core.explain import build as build_explain  # noqa: E402
from llm_cal.hardware.loader import load_database  # noqa: E402
from llm_cal.llm_review.reviewer import run_review  # noqa: E402

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

EXAMPLE_MODELS: list[tuple[str, str, str, str]] = [
    ("deepseek-ai/DeepSeek-V4-Flash", "NVIDIA", "H800", "vllm"),
    ("deepseek-ai/DeepSeek-V3", "NVIDIA", "H800", "vllm"),
    ("Qwen/Qwen2.5-72B-Instruct", "NVIDIA", "H100", "vllm"),
    ("Qwen/Qwen3-30B-A3B", "NVIDIA", "A100-80G", "vllm"),
    ("mistralai/Mixtral-8x7B-v0.1", "NVIDIA", "H100", "vllm"),
    ("microsoft/Phi-4", "NVIDIA", "RTX4090", "vllm"),
    ("deepseek-ai/DeepSeek-V4-Flash", "Huawei Ascend", "910B4", "vllm"),
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
        + "</div>"
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

_evaluator: Evaluator | None = None


def _get_evaluator() -> Evaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator


def calculate(
    model_id: str,
    gpu: str,
    engine: str,
    context_length: int | None,
    lang: str,
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
    llm_api_key: str,
    llm_base_url: str,
    llm_model: str,
) -> tuple[str, str, str]:
    """Returns (main_html, explain_html, llm_review_html)."""
    locale = "zh" if lang.startswith("中") else "en"
    is_zh = locale == "zh"

    if not model_id or not model_id.strip():
        return (
            _render_error(
                "请输入 HuggingFace model id" if is_zh else "Enter a HuggingFace model id",
                is_zh,
            ),
            "",
            "",
        )
    if not gpu:
        return (_render_error("请选择 GPU" if is_zh else "Pick a GPU", is_zh), "", "")

    try:
        report = _get_evaluator().evaluate(
            model_id=model_id.strip(),
            gpu=gpu,
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


def show_loading(lang: str) -> tuple[str, str, str]:
    is_zh = lang.startswith("中")
    return _render_loading(is_zh), "", ""


# ---------------------------------------------------------------------------
# UI

THEME = gr.themes.Soft(primary_hue="indigo")

HERO_HTML = """
<div class='lc-hero'>
  <div class='lc-hero-title'>llm-cal</div>
  <div class='lc-hero-tagline'>
    LLM inference hardware calculator · 大模型推理硬件计算器<br>
    Architecture-aware · Engine-aware · <strong>Honest-labeled</strong>
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
.gradio-container, .gradio-container * {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue",
    "PingFang SC", "Microsoft YaHei", "Segoe UI", Roboto, Arial, sans-serif !important;
}

/* Hide Gradio's default footer chrome that looks like part of our app */
footer { display: none !important; }
.show-api, .built-with, .settings { display: none !important; }

/* Tighter overall padding so the page feels more like a tool, less like a docs site */
.gradio-container { max-width: 1100px !important; }

/* Hero section */
.lc-hero {
  margin: 8px 0 24px 0;
  padding: 24px 0 18px 0;
  border-bottom: 1px solid #e5e7eb;
}
.dark .lc-hero { border-bottom-color: #374151; }
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
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  align-items: stretch;
  padding: 0;
  font-size: 13px !important;
  color: #1e293b !important;
}
.dark .lc-hero-pitch { color: #f1f5f9 !important; }

.lc-pitch-card {
  flex: 1 1 200px;
  padding: 14px 18px;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.dark .lc-pitch-card { background: #111827; border-color: #374151; }
.lc-pitch-bad  { border-left: 3px solid #b91c1c; }
.lc-pitch-good { border-left: 3px solid #15803d; }
.dark .lc-pitch-bad  { border-left-color: #f87171; }
.dark .lc-pitch-good { border-left-color: #4ade80; }

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
  background: #0f172a;
  color: #f8fafc !important;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.dark .lc-pitch-summary { background: #1e293b; }
.lc-pitch-model {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #94a3b8 !important;
  margin-bottom: 4px;
}
.lc-pitch-result {
  font-size: 14px !important;
  font-weight: 700 !important;
  color: #f8fafc !important;
}

/* Make the primary button less Gradio-purple */
.gradio-container button.primary,
.gradio-container button[variant="primary"],
.gradio-container .primary > button {
  background: #0f172a !important;
  border-color: #0f172a !important;
  color: #f8fafc !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
}
.gradio-container button.primary:hover { background: #1e293b !important; }

/* Form labels — use a neutral chip instead of Gradio's purple */
.gradio-container label > span {
  font-size: 12px !important;
  font-weight: 600 !important;
  color: #374151 !important;
  letter-spacing: 0.02em;
}
.dark .gradio-container label > span { color: #d1d5db !important; }

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
  background: rgba(219, 39, 119, 0.06);
  border: 1px solid rgba(219, 39, 119, 0.18);
  border-radius: 8px;
  font-size: 12px !important;
  color: #be185d !important;
  margin-bottom: 12px;
}
.dark .lc-llm-banner { color: #f9a8d4 !important; background: rgba(219, 39, 119, 0.12); }
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


def _bi(en: str, zh: str) -> str:
    """Two-line bilingual tooltip — EN above, 中文 below."""
    return f"{en}\n{zh}"


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="llm-cal — LLM hardware calculator") as demo:
        gr.HTML(HERO_HTML)

        # ---- Required ----------------------------------------------------
        with gr.Row():
            model_id = gr.Textbox(
                label="Model ID · 模型 ID",
                placeholder="e.g. deepseek-ai/DeepSeek-V4-Flash",
                info=_bi(
                    "HuggingFace repo id (owner/name). Gated models need HF_TOKEN.",
                    "HuggingFace 仓库 ID（owner/name 格式），私有/Gated 模型需要 HF_TOKEN。",
                ),
                scale=3,
            )

        with gr.Row():
            vendor = gr.Dropdown(
                choices=VENDOR_CHOICES_EN,
                value=DEFAULT_VENDOR,
                label="GPU vendor · GPU 厂商",
                info=_bi(
                    "Pick brand first; the model list filters accordingly. 11 vendors covered.",
                    "先选厂商，下方型号列表会跟着筛选。共 11 家。",
                ),
                scale=1,
            )
            gpu = gr.Dropdown(
                choices=_VENDOR_TO_GPUS[DEFAULT_VENDOR],
                value=DEFAULT_GPU,
                label="GPU model · GPU 型号",
                info=_bi(
                    "Target hardware. e.g. H800 (China-regulated H100), 910B4 (Ascend), MI300X (AMD).",
                    "目标硬件型号。例如 H800（中国合规版 H100）、910B4（昇腾）、MI300X（AMD）。",
                ),
                scale=2,
                allow_custom_value=True,
            )

        with gr.Row():
            engine = gr.Radio(
                choices=["vllm", "sglang"],
                value="vllm",
                label="Engine · 引擎",
                info=_bi(
                    "Inference engine. Drives generated command + required-flag source.",
                    "推理引擎。决定生成命令的形式 + 必需 flag 来源（compat 矩阵）。",
                ),
            )
            context_length = gr.Number(
                label="Context length · Context 长度",
                value=None,
                precision=0,
                info=_bi(
                    "Empty = show 4K/32K/128K/1M. Set = single-context override.",
                    "留空 = 显示 4K/32K/128K/1M（如模型支持）。填了就只显示这一个 context。",
                ),
            )
            lang = gr.Radio(
                choices=["English", "中文"],
                value="English",
                label="Output language · 输出语言",
                info=_bi(
                    "Affects label translations + explanations in the result area below.",
                    "影响下方结果区的标签翻译和说明文字。",
                ),
            )

        # ---- Performance tuning (collapsible) ----------------------------
        with gr.Accordion("Performance tuning · 性能参数", open=False):
            gr.Markdown(
                "_SLA assumptions + empirical coefficients — affects prefill latency, "
                "decode throughput, and concurrency estimates._<br>"
                "_SLA 假设和经验系数。改了会同步影响 prefill 延迟、decode 吞吐和并发上限。_"
            )
            with gr.Row():
                input_tokens = gr.Number(
                    label="Input tokens · 输入 tokens",
                    value=2000,
                    precision=0,
                    info=_bi(
                        "Prefill budget. Default 2000 (typical chat/RAG context).",
                        "Prefill 预算。默认 2000（典型对话/RAG 上下文）。",
                    ),
                )
                output_tokens = gr.Number(
                    label="Output tokens · 输出 tokens",
                    value=512,
                    precision=0,
                    info=_bi(
                        "Decode budget per request. Default 512.",
                        "Decode 预算。默认 512。",
                    ),
                )
                target_tps = gr.Number(
                    label="Target tok/s/user · 单用户目标 tok/s",
                    value=30.0,
                    info=_bi(
                        "SLA per user. Drives the L bound. 30 ≈ smooth reading speed.",
                        "单用户 decode SLA。决定 L 上界。30 ≈ 流畅阅读速度。",
                    ),
                )
            with gr.Row():
                prefill_util = gr.Number(
                    label="Prefill util · Prefill 利用率",
                    value=0.40,
                    info=_bi(
                        "Compute utilization 0–1. 0.40 = vLLM paper baseline.",
                        "算力利用率（0–1）。0.40 是 vLLM 论文经验值。",
                    ),
                )
                decode_bw_util = gr.Number(
                    label="Decode BW util · Decode 带宽利用率",
                    value=0.50,
                    info=_bi(
                        "Memory-BW utilization 0–1. 0.50 = community median.",
                        "显存带宽利用率（0–1）。0.50 是社区实测中位。",
                    ),
                )
                concurrency_degradation = gr.Number(
                    label="Concurrency degradation · 并发衰减",
                    value=1.0,
                    info=_bi(
                        "1.0 = honest baseline. 1.67 if engine drops to 60% efficiency under load.",
                        "1.0 = 诚实基线。1.67 表示满载下掉到 60% 效率。",
                    ),
                )

        # ---- Advanced (collapsible) --------------------------------------
        with gr.Accordion("Advanced · 高级", open=False):
            with gr.Row():
                gpu_count = gr.Number(
                    label="Force GPU count · 强制 GPU 数",
                    value=None,
                    precision=0,
                    info=_bi(
                        "Empty = auto min/dev/prod recommendation. Set = single-count eval.",
                        "留空 = 自动给 min/dev/prod 三档推荐。填了就只评估这一个 GPU 数。",
                    ),
                )
                refresh = gr.Checkbox(
                    label="Refresh cache · 刷新缓存",
                    value=False,
                    info=_bi(
                        "Bypass diskcache, re-fetch from HF. Useful when model just updated.",
                        "跳过本地缓存，重新从 HF 拉取 metadata。模型刚更新时勾上。",
                    ),
                )
            with gr.Row():
                explain = gr.Checkbox(
                    label="--explain · 推导链",
                    value=False,
                    info=_bi(
                        "Output the full derivation trace: every formula, input, step, source.",
                        "输出完整推导链：每个数字的公式、输入项（带 label）、计算步骤、来源。",
                    ),
                )
                llm_review = gr.Checkbox(
                    label="--llm-review · LLM 审计",
                    value=False,
                    info=_bi(
                        "Send the trace to an LLM for a second opinion. Needs API key below.",
                        "把推导链发给 LLM 做第二意见审计。需要下方填 API key 等。",
                    ),
                )
            with gr.Row():
                llm_api_key = gr.Textbox(
                    label="LLM API key · LLM API 密钥",
                    value="",
                    placeholder="sk-...",
                    type="password",
                    info=_bi(
                        "Any OpenAI-compatible endpoint: OpenAI / DeepSeek / MiniMax / Moonshot.",
                        "任意 OpenAI 兼容端点：OpenAI / DeepSeek / MiniMax / 月之暗面。",
                    ),
                )
                llm_base_url = gr.Textbox(
                    label="LLM base URL · LLM 基地址",
                    value="",
                    placeholder="https://api.openai.com/v1",
                    info=_bi(
                        "e.g. https://api.deepseek.com/v1 or https://api.minimaxi.com/v1",
                        "例如 https://api.deepseek.com/v1 或 https://api.minimaxi.com/v1",
                    ),
                )
                llm_model = gr.Textbox(
                    label="LLM model · LLM 模型名",
                    value="",
                    placeholder="gpt-4o",
                    info=_bi(
                        "e.g. gpt-4o / deepseek-chat / MiniMax-M2 / moonshot-v1-32k",
                        "如 gpt-4o / deepseek-chat / MiniMax-M2 / moonshot-v1-32k",
                    ),
                )

        submit = gr.Button("Calculate · 计算", variant="primary", size="lg")

        # Three output panes — main always shows, explain/llm-review only when toggled
        output_main = gr.HTML(label="Result")
        output_explain = gr.HTML(label="Explain trace")
        output_llm = gr.HTML(label="LLM review")

        gr.Examples(
            examples=[[m, v, g, e, None, "English"] for m, v, g, e in EXAMPLE_MODELS],
            inputs=[model_id, vendor, gpu, engine, context_length, lang],
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

        # When vendor changes, repopulate the GPU dropdown.
        def _on_vendor_change(v: str):  # noqa: ANN202
            gpus = _VENDOR_TO_GPUS.get(v, [])
            default = gpus[0] if gpus else None
            return gr.Dropdown(choices=gpus, value=default)

        vendor.change(fn=_on_vendor_change, inputs=[vendor], outputs=[gpu])

        # Click flow: instantly show "loading…", THEN run calculate.
        all_outputs = [output_main, output_explain, output_llm]
        submit.click(
            fn=show_loading,
            inputs=[lang],
            outputs=all_outputs,
        ).then(
            fn=calculate,
            inputs=[
                model_id, gpu, engine, context_length, lang,
                gpu_count, input_tokens, output_tokens, target_tps,
                prefill_util, decode_bw_util, concurrency_degradation,
                refresh, explain, llm_review,
                llm_api_key, llm_base_url, llm_model,
            ],
            outputs=all_outputs,
        )

    return demo


if __name__ == "__main__":
    _build_ui().launch(theme=THEME, css=CUSTOM_CSS)
