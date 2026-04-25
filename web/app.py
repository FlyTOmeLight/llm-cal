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

import gradio as gr  # noqa: E402

from llm_cal.common.i18n import set_locale, t  # noqa: E402
from llm_cal.core.evaluator import EvaluationReport, Evaluator  # noqa: E402
from llm_cal.hardware.loader import load_database  # noqa: E402

# ---------------------------------------------------------------------------
# Static data the UI needs

_DB = load_database()
GPU_CHOICES: list[str] = sorted(g.id for g in _DB.gpus)

EXAMPLE_MODELS: list[tuple[str, str, str]] = [
    ("deepseek-ai/DeepSeek-V4-Flash", "H800", "vllm"),
    ("deepseek-ai/DeepSeek-V3", "H800", "vllm"),
    ("Qwen/Qwen2.5-72B-Instruct", "H100", "vllm"),
    ("Qwen/Qwen3-30B-A3B", "A100-80G", "vllm"),
    ("mistralai/Mixtral-8x7B-v0.1", "H100", "vllm"),
    ("microsoft/Phi-4", "RTX4090", "vllm"),
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


def _render(report: EvaluationReport, locale: str) -> str:
    set_locale(locale)  # type: ignore[arg-type]
    is_zh = locale == "zh"

    p, w, r, f = report.profile, report.weight, report.reconciliation, report.fleet

    # Headline summary card
    quant_label = f"`[{t('label.' + w.quantization_guess.label.value)}]`"
    headline = (
        f"### {report.model_id} · {report.gpu} · {report.engine}\n\n"
        f"**{'权重' if is_zh else 'Weight'}:** {_fmt_bytes(w.total_bytes.value)} "
        f"`[{t('label.' + w.total_bytes.label.value)}]`  \n"
        f"**{'量化' if is_zh else 'Quantization'}:** "
        f"`{w.quantization_guess.value}` {quant_label}  \n"
        f"_{w.quantization_guess.source or ''}_"
    )

    # Architecture
    arch_rows = [f"| `model_type` | `{p.model_type}` |"]
    if p.attention:
        arch_rows.append(
            f"| `attention` | `{p.attention.variant} (heads={p.attention.num_heads}, "
            f"kv_heads={p.attention.num_kv_heads}, hd={p.attention.head_dim})` |"
        )
    if p.moe:
        arch_rows.append(
            f"| `moe` | `{p.moe.num_routed_experts} routed + "
            f"{p.moe.num_shared_experts} shared, top-{p.moe.num_experts_per_tok}` |"
        )
    if p.sliding_window:
        arch_rows.append(f"| `sliding_window` | `{p.sliding_window}` |")
    arch_md = (
        f"#### {'架构' if is_zh else 'Architecture'}\n\n"
        f"| {'字段' if is_zh else 'Field'} | {'值' if is_zh else 'Value'} |\n"
        f"|---|---|\n" + "\n".join(arch_rows)
    )

    # Reconciliation
    recon_rows = []
    for c in r.candidates[:5]:
        marker = " ✓" if c.scheme == r.best.value else ""
        recon_rows.append(
            f"| {c.scheme}{marker} | {_fmt_bytes(c.predicted_bytes)} | "
            f"{c.relative_error * 100:.1f}% |"
        )
    recon_md = (
        f"#### {'量化反演' if is_zh else 'Quantization reconciliation'}\n\n"
        + (
            "| Scheme | Predicted | Error |\n|---|---:|---:|\n"
            + "\n".join(recon_rows)
            if recon_rows
            else "_no data_"
        )
    )

    # Fleet
    fleet_md = ""
    if f and f.options:
        rows = []
        for opt in f.options:
            star = " ★" if opt.tier == f.best_tier else ""
            headroom = max(0, opt.usable_bytes_per_gpu - opt.weight_bytes_per_gpu)
            rows.append(
                f"| {opt.tier}{star} | {opt.gpu_count} | "
                f"{_fmt_bytes(opt.weight_bytes_per_gpu)} | "
                f"{_fmt_bytes(headroom)} | "
                f"{opt.max_concurrent_at_reference_ctx} |"
            )
        fleet_md = (
            f"#### {'推荐集群' if is_zh else 'Recommended fleet'}\n\n"
            "| Tier | GPUs | Weight/GPU | Headroom/GPU | "
            f"{'@ 128K 并发' if is_zh else 'Concurrent @ 128K'} |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows)
        )

    # Performance
    perf_md = ""
    if report.prefill and report.decode and report.concurrency:
        max_users = report.concurrency.max_concurrent.value
        bn = report.concurrency.bottleneck
        perf_md = (
            f"#### {'性能' if is_zh else 'Performance'}\n\n"
            f"- **Prefill latency** {report.prefill.latency_ms.value:.0f} ms "
            f"@ {report.perf_input_tokens or 2000} input tokens  \n"
            f"- **Cluster decode throughput** "
            f"{report.decode.cluster_tokens_per_sec.value:.0f} tok/s  \n"
            f"- **Max concurrent users** {max_users}  \n"
            f"- **Bottleneck** `{bn}`"
        )

    # Generated command
    cmd_md = ""
    if report.generated_command:
        cmd_md = (
            f"#### {'生成命令' if is_zh else 'Generated command'}\n\n"
            f"```bash\n{report.generated_command}\n```"
        )

    return "\n\n".join([s for s in [headline, arch_md, recon_md, fleet_md, perf_md, cmd_md] if s])


def _render_error(msg: str, is_zh: bool) -> str:
    label = "出错了" if is_zh else "Error"
    return f"### {label}\n\n```\n{msg}\n```"


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
) -> str:
    locale = "zh" if lang.startswith("中") else "en"
    is_zh = locale == "zh"

    if not model_id or not model_id.strip():
        return _render_error(
            "请输入 HuggingFace model id" if is_zh else "Enter a HuggingFace model id",
            is_zh,
        )
    if not gpu:
        return _render_error("请选择 GPU" if is_zh else "Pick a GPU", is_zh)

    try:
        report = _get_evaluator().evaluate(
            model_id=model_id.strip(),
            gpu=gpu,
            engine=engine,
            context_length=context_length if context_length and context_length > 0 else None,
            input_tokens=2000,
            output_tokens=512,
            target_tokens_per_sec=30.0,
        )
    except Exception as e:  # noqa: BLE001
        return _render_error(f"{type(e).__name__}: {e}", is_zh)

    return _render(report, locale)


# ---------------------------------------------------------------------------
# UI

THEME = gr.themes.Soft(primary_hue="indigo")

INTRO_EN = """
# llm-cal — LLM inference hardware calculator

Architecture-aware. Engine-aware. **Honest-labeled** — every number tagged
with where it came from.

The headline story: **DeepSeek-V4-Flash** ships an FP4+FP8 mixed pack.
`gpu_poor` reports 284 GB (assumes pure FP8). `llm-cal` reports 160 GB by
reading actual safetensors bytes + per-tensor dtype. **0.2% error vs
gpu_poor's 45%.**

Try it on any HuggingFace model below.
"""

INTRO_ZH = """
# llm-cal — 大模型推理硬件计算器

架构感知、引擎感知、**诚实标签** —— 每个数字都标了出处。

招牌故事：**DeepSeek-V4-Flash** 是 FP4+FP8 混合 pack。`gpu_poor` 报 284 GB
（假设纯 FP8）；`llm-cal` 读真实 safetensors 字节 + per-tensor dtype，
报 160 GB。**0.2% 误差 vs gpu_poor 的 45%。**

下方填任意 HuggingFace 模型 ID 试一下。
"""


def _build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, title="llm-cal — LLM hardware calculator") as demo:
        intro = gr.Markdown(INTRO_EN)

        with gr.Row():
            with gr.Column(scale=2):
                model_id = gr.Textbox(
                    label="Model ID",
                    placeholder="e.g. deepseek-ai/DeepSeek-V4-Flash",
                    info="HuggingFace repo id",
                )
            with gr.Column(scale=1):
                gpu = gr.Dropdown(
                    choices=GPU_CHOICES,
                    value="H800",
                    label="GPU",
                    info="Target hardware",
                )

        with gr.Row():
            engine = gr.Radio(
                choices=["vllm", "sglang"],
                value="vllm",
                label="Engine",
            )
            context_length = gr.Number(
                label="Context length",
                value=None,
                precision=0,
                info="Optional override (defaults to model max)",
            )
            lang = gr.Radio(
                choices=["English", "中文"],
                value="English",
                label="Language",
            )

        submit = gr.Button("Calculate", variant="primary")
        output = gr.Markdown(label="Result")

        gr.Examples(
            examples=[[m, g, e, None, "English"] for m, g, e in EXAMPLE_MODELS],
            inputs=[model_id, gpu, engine, context_length, lang],
            label="Try one of these",
        )

        gr.Markdown(
            "---\n"
            "📦 [GitHub](https://github.com/FlyTOmeLight/llm-cal) · "
            "📚 [Docs](https://flytomelight.github.io/llm-cal/) · "
            "🐍 `pip install llm-cal` · "
            "📐 [Methodology](https://flytomelight.github.io/llm-cal/methodology/)"
        )

        # Wire events
        submit.click(
            fn=calculate,
            inputs=[model_id, gpu, engine, context_length, lang],
            outputs=output,
        )

        # Switch intro language when user toggles lang radio
        def _swap_intro(choice: str) -> str:
            return INTRO_ZH if choice.startswith("中") else INTRO_EN

        lang.change(fn=_swap_intro, inputs=[lang], outputs=[intro])

    return demo


if __name__ == "__main__":
    _build_ui().launch()
