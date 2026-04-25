"""Pre-render llm-cal evaluation pages for the docs site.

For each (model, GPU) pair in COMBOS, run a real Evaluator pass and write
two Markdown pages — one English, one Chinese — under docs/models/. mkdocs
picks them up at build time. Each page is shareable as a static URL like
  https://flytomelight.github.io/llm-cal/models/deepseek-v4-flash-h800/

This script needs network (hits HF API) and respects HF_TOKEN env var.
Re-runs are idempotent and cache-warm because Evaluator uses diskcache.

Run:
  python scripts/generate_model_pages.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure src/ is importable when this is run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_cal.common.i18n import set_locale, t  # noqa: E402
from llm_cal.core.evaluator import EvaluationReport, Evaluator  # noqa: E402

# ---------------------------------------------------------------------------
# What to render. Curated for v1: signature stories + popular models.
# Add an entry here, re-run the script, get a new page.

COMBOS: list[tuple[str, str, str]] = [
    # The signature DeepSeek-V4 story across the GPU spectrum
    ("deepseek-ai/DeepSeek-V4-Flash", "H800", "vllm"),
    ("deepseek-ai/DeepSeek-V4-Flash", "H100", "vllm"),
    ("deepseek-ai/DeepSeek-V4-Flash", "B200", "vllm"),
    ("deepseek-ai/DeepSeek-V4-Flash", "910B4", "vllm"),
    # DeepSeek-V3 — classic MoE+MLA
    ("deepseek-ai/DeepSeek-V3", "H800", "vllm"),
    ("deepseek-ai/DeepSeek-V3", "H100", "vllm"),
    ("deepseek-ai/DeepSeek-V3", "B200", "vllm"),
    # Qwen2.5-72B — dense GQA reference
    ("Qwen/Qwen2.5-72B-Instruct", "H100", "vllm"),
    ("Qwen/Qwen2.5-72B-Instruct", "A100-80G", "vllm"),
    ("Qwen/Qwen2.5-72B-Instruct", "H800", "vllm"),
    # Qwen3-30B-A3B — small MoE
    ("Qwen/Qwen3-30B-A3B", "A100-80G", "vllm"),
    ("Qwen/Qwen3-30B-A3B", "H100", "vllm"),
    # Qwen2.5-7B — small dense
    ("Qwen/Qwen2.5-7B", "RTX4090", "vllm"),
    ("Qwen/Qwen2.5-7B", "L40S", "vllm"),
    # Mixtral 8x7B
    ("mistralai/Mixtral-8x7B-v0.1", "H100", "vllm"),
    ("mistralai/Mixtral-8x7B-v0.1", "A100-80G", "vllm"),
    # Phi-4
    ("microsoft/Phi-4", "RTX4090", "vllm"),
    ("microsoft/Phi-4", "L40S", "vllm"),
]


@dataclass(frozen=True)
class PageInfo:
    """Metadata for one generated page (for index)."""

    slug: str
    model_id: str
    gpu: str
    engine: str
    title_en: str
    title_zh: str
    weight_gb: float | None
    quant: str
    fleet_prod_gpus: int | None


# ---------------------------------------------------------------------------


def _slug(model_id: str, gpu: str) -> str:
    """Filesystem- and URL-safe slug."""
    s = f"{model_id.replace('/', '-')}-{gpu}".lower()
    return s.replace("_", "-").replace(".", "-")


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        f /= 1024
        if f < 1024:
            return f"{f:.2f} {u}"
    return f"{f:.2f} PB"


def _fmt_params(n: int | None) -> str:
    if n is None or n == 0:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def render_page(report: EvaluationReport, locale: str) -> str:
    """Render a single EvaluationReport to Markdown for the given locale."""
    set_locale(locale)  # type: ignore[arg-type]
    is_zh = locale == "zh"

    # Headers
    h_arch = "架构" if is_zh else "Architecture"
    h_weight = "权重" if is_zh else "Weights"
    h_recon = "量化反演" if is_zh else "Quantization reconciliation"
    h_kv = "KV 缓存（每请求）" if is_zh else "KV cache per request"
    h_fleet = "推荐集群" if is_zh else "Recommended fleet"
    h_perf = "性能" if is_zh else "Performance"
    h_cmd = "生成命令" if is_zh else "Generated command"
    h_other = "本模型其他配置" if is_zh else "Other configurations for this model"

    p = report.profile
    w = report.weight
    r = report.reconciliation
    f = report.fleet

    # ---- Architecture table
    arch_rows: list[tuple[str, str]] = [
        ("model_type", str(p.model_type)),
    ]
    if p.attention:
        att = p.attention
        arch_rows.append(
            (
                "attention",
                f"{att.variant} (heads={att.num_heads}, kv_heads={att.num_kv_heads}, hd={att.head_dim})",
            )
        )
    if p.moe:
        arch_rows.append(
            (
                "moe",
                f"{p.moe.num_routed_experts} routed + {p.moe.num_shared_experts} shared, top-{p.moe.num_experts_per_tok}",
            )
        )
    if p.sliding_window:
        arch_rows.append(("sliding_window", str(p.sliding_window)))

    arch_table = (
        "| Field | Value |\n|---|---|\n"
        + "\n".join(f"| `{k}` | `{v}` |" for k, v in arch_rows)
    )

    # ---- Weight table
    quant_label = (
        f"`{w.quantization_guess.value}`"
        f" `[{t('label.' + w.quantization_guess.label.value)}]`"
    )
    weight_rows = [
        (
            "safetensors bytes" if not is_zh else "safetensors 字节",
            _fmt_bytes(w.total_bytes.value),
            f"`[{t('label.' + w.total_bytes.label.value)}]`",
        ),
        (
            "params" if not is_zh else "参数量",
            _fmt_params(report.total_params_estimate.value),
            f"`[{t('label.' + report.total_params_estimate.label.value)}]`",
        ),
        (
            "quantization" if not is_zh else "量化方案",
            quant_label,
            "",
        ),
    ]
    weight_table = "| Field | Value | Label |\n|---|---|---|\n" + "\n".join(
        f"| {k} | {v} | {lab} |" for k, v, lab in weight_rows
    )

    # ---- Reconciliation table
    recon_rows: list[str] = []
    for c in r.candidates[:5]:
        direction = ("over" if c.delta_bytes > 0 else "under") if not is_zh else (
            "偏多" if c.delta_bytes > 0 else "偏少"
        )
        marker = " ✓" if c.scheme == r.best.value else ""
        recon_rows.append(
            f"| {c.scheme}{marker} | {_fmt_bytes(c.predicted_bytes)} | "
            f"{_fmt_bytes(abs(c.delta_bytes))} {direction} | {c.relative_error * 100:.1f}% |"
        )
    if recon_rows:
        recon_table = (
            "| Scheme | Predicted | Δ | Error |\n|---|---:|---:|---:|\n"
            + "\n".join(recon_rows)
            + f"\n\n_Best: **{r.best.value}** — {r.best.source or ''}_"
        )
    else:
        recon_table = "_No reconciliation data._"

    # ---- KV cache
    kv_rows: list[str] = []
    for ctx, kv in sorted(report.kv_cache_by_context.items()):
        kv_rows.append(f"| {ctx:,} | {_fmt_bytes(kv.value)} |")
    kv_table = (
        "| Context tokens | KV bytes |\n|---:|---:|\n" + "\n".join(kv_rows)
        if kv_rows
        else ""
    )

    # ---- Fleet
    fleet_md = ""
    if f and f.options:
        rows = []
        for opt in f.options:
            star = " ★" if opt.tier == f.best_tier else ""
            headroom_per_gpu = opt.usable_bytes_per_gpu - opt.weight_bytes_per_gpu
            rows.append(
                f"| {opt.tier}{star} | {opt.gpu_count} | "
                f"{_fmt_bytes(opt.weight_bytes_per_gpu)} | "
                f"{_fmt_bytes(max(headroom_per_gpu, 0))} | "
                f"{opt.max_concurrent_at_reference_ctx} |"
            )
        fleet_md = (
            "| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows)
        )

    # ---- Performance
    perf_md = ""
    if report.prefill and report.decode:
        bn = report.concurrency.bottleneck if report.concurrency else "—"
        max_users = (
            report.concurrency.max_concurrent.value if report.concurrency else None
        )
        perf_md = (
            f"- **Prefill latency** {report.prefill.latency_ms.value:.0f} ms "
            f"@ {report.perf_input_tokens or 2000} input tokens "
            f"`[{t('label.' + report.prefill.latency_ms.label.value)}]`\n"
            f"- **Cluster decode throughput** "
            f"{report.decode.cluster_tokens_per_sec.value:.0f} tok/s "
            f"`[{t('label.' + report.decode.cluster_tokens_per_sec.label.value)}]`\n"
            f"- **Max concurrent users** "
            f"{'—' if max_users is None else max_users}\n"
            f"- **Bottleneck** `{bn}`"
        )

    # ---- Command
    cmd = report.generated_command or "—"

    # ---- Title
    title_en = f"{report.model_id} on {report.gpu}"
    title_zh = f"{report.model_id} 跑在 {report.gpu}"
    title = title_zh if is_zh else title_en

    # ---- Description for og/meta
    desc = (
        f"{report.model_id} 在 {report.gpu} 上需要多少 GPU。"
        if is_zh
        else f"How many {report.gpu} GPUs to run {report.model_id}."
    )

    # ---- Compose the page
    sections = [
        "---",
        f"title: {title}",
        f"description: {desc}",
        "---",
        "",
        f"# {title}",
        "",
        f"_{desc}_",
        "",
        f"## {h_arch}",
        "",
        arch_table,
        "",
        f"## {h_weight}",
        "",
        weight_table,
        "",
        f"### {h_recon}",
        "",
        recon_table,
        "",
    ]

    if kv_table:
        sections += [f"## {h_kv}", "", kv_table, ""]

    if fleet_md:
        sections += [f"## {h_fleet}", "", fleet_md, ""]

    if perf_md:
        sections += [f"## {h_perf}", "", perf_md, ""]

    sections += [
        f"## {h_cmd}",
        "",
        "```bash",
        cmd,
        "```",
        "",
        "---",
        "",
        f"_{('生成方式' if is_zh else 'Generated by')}_: ",
        f"```bash\nllm-cal {report.model_id} --gpu {report.gpu} "
        f"--engine {report.engine} --lang {locale}\n```",
        "",
    ]

    return "\n".join(sections)


def render_index(pages: list[PageInfo], locale: str) -> str:
    is_zh = locale == "zh"
    title = "模型评估报告" if is_zh else "Model evaluations"
    desc = (
        "热门模型 × 主流 GPU 的预渲染评估页。每页都是 `llm-cal` 实跑结果。"
        if is_zh
        else "Pre-rendered evaluation pages for popular models on common GPUs. "
        "Every page is real `llm-cal` output."
    )
    by_model: dict[str, list[PageInfo]] = {}
    for p in pages:
        by_model.setdefault(p.model_id, []).append(p)

    sections = [
        "---",
        f"title: {title}",
        f"description: {desc}",
        "---",
        "",
        f"# {title}",
        "",
        desc,
        "",
    ]

    for model_id in sorted(by_model.keys()):
        sections.append(f"## {model_id}")
        sections.append("")
        sections.append(
            "| GPU | "
            + ("权重" if is_zh else "Weight")
            + " | "
            + ("量化" if is_zh else "Quant")
            + " | "
            + ("推荐 GPU 数" if is_zh else "Prod GPUs")
            + " | "
            + ("链接" if is_zh else "Page")
            + " |"
        )
        sections.append("|---|---:|---|---:|---|")
        for p in sorted(by_model[model_id], key=lambda x: x.gpu):
            weight_s = f"{p.weight_gb:.1f} GB" if p.weight_gb is not None else "—"
            prod_s = str(p.fleet_prod_gpus) if p.fleet_prod_gpus is not None else "—"
            # Both EN (docs/models/index.md) and ZH (docs/zh/models/index.md)
            # are siblings of their model pages. Same relative link.
            sections.append(
                f"| {p.gpu} | {weight_s} | `{p.quant}` | {prod_s} | "
                f"[→]({p.slug}.md) |"
            )
        sections.append("")

    return "\n".join(sections)


def main() -> int:
    out_en = ROOT / "docs" / "models"
    out_zh = ROOT / "docs" / "zh" / "models"
    out_en.mkdir(parents=True, exist_ok=True)
    out_zh.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator()
    pages: list[PageInfo] = []

    for model_id, gpu, engine in COMBOS:
        slug = _slug(model_id, gpu)
        print(f"==> {model_id} on {gpu} ({engine}) -> {slug}", flush=True)
        try:
            report = evaluator.evaluate(
                model_id=model_id,
                gpu=gpu,
                engine=engine,
                input_tokens=2000,
                output_tokens=512,
                target_tokens_per_sec=30.0,
            )
        except Exception as e:  # noqa: BLE001
            print(f"   SKIP: {type(e).__name__}: {e}", flush=True)
            continue

        (out_en / f"{slug}.md").write_text(render_page(report, "en"), encoding="utf-8")
        (out_zh / f"{slug}.md").write_text(render_page(report, "zh"), encoding="utf-8")

        pages.append(
            PageInfo(
                slug=slug,
                model_id=report.model_id,
                gpu=report.gpu,
                engine=report.engine,
                title_en=f"{report.model_id} on {report.gpu}",
                title_zh=f"{report.model_id} 跑在 {report.gpu}",
                weight_gb=(
                    report.weight.total_bytes.value / 1_000_000_000
                    if report.weight.total_bytes.value
                    else None
                ),
                quant=str(report.weight.quantization_guess.value),
                fleet_prod_gpus=(
                    next(
                        (o.gpu_count for o in (report.fleet.options if report.fleet else []) if o.tier == "prod"),
                        None,
                    )
                ),
            )
        )

    (out_en / "index.md").write_text(render_index(pages, "en"), encoding="utf-8")
    (out_zh / "index.md").write_text(render_index(pages, "zh"), encoding="utf-8")

    print(f"\nDone — {len(pages)} pages × 2 languages = {len(pages) * 2} files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
