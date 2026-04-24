"""Rich-formatted output for EvaluationReport.

This is a partial implementation — focuses on the label-discipline story and
model/weight/KV-cache reporting. Fleet planner and generated-command blocks are
Week 5.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_cal.core.evaluator import EvaluationReport
from llm_cal.output.labels import AnnotatedValue, Label

_LABEL_STYLES: dict[Label, str] = {
    Label.VERIFIED: "bold green",
    Label.INFERRED: "cyan",
    Label.ESTIMATED: "yellow",
    Label.CITED: "blue",
    Label.UNVERIFIED: "bold yellow",
    Label.UNKNOWN: "dim red",
}


def format_tag(av: AnnotatedValue[Any]) -> Text:
    style = _LABEL_STYLES.get(av.label, "white")
    return Text(f"[{av.label.value}]", style=style)


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.2f} KB"
    return f"{n} B"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return str(n)


def render(report: EvaluationReport, console: Console | None = None) -> None:
    console = console or Console()

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{report.model_id}[/bold cyan]  "
            f"[dim]via {report.source}"
            f"{' @ ' + report.commit_sha[:7] if report.commit_sha else ''}[/dim]",
            border_style="cyan",
        )
    )

    _render_architecture(report, console)
    _render_weight(report, console)
    _render_kv_cache(report, console)
    _render_label_legend(console)


def _render_architecture(report: EvaluationReport, console: Console) -> None:
    p = report.profile
    table = Table(title="Architecture", show_header=False, box=None, padding=(0, 2))
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_column("label")

    table.add_row(
        "model_type",
        p.model_type or "(none)",
        _verified_tag("config.json"),
    )
    table.add_row(
        "family",
        p.family.value,
        _verified_tag("config.json"),
    )
    table.add_row(
        "confidence",
        p.confidence.value,
        Text(f"[{p.confidence.value}]", style="magenta"),
    )
    table.add_row("layers", str(p.num_hidden_layers), _verified_tag())
    table.add_row("hidden_size", str(p.hidden_size), _verified_tag())
    table.add_row("vocab_size", f"{p.vocab_size:,}", _verified_tag())
    if p.attention is not None:
        table.add_row(
            "attention",
            f"{p.attention.variant} "
            f"(heads={p.attention.num_heads}, kv_heads={p.attention.num_kv_heads}, "
            f"head_dim={p.attention.head_dim})",
            _verified_tag(),
        )
        if p.attention.compress_ratios:
            ratios = p.attention.compress_ratios
            summary = f"len={len(ratios)}, dense_layers={sum(1 for r in ratios if r == 0)}"
            table.add_row("compress_ratios", summary, _verified_tag())
    if p.moe is not None:
        table.add_row(
            "moe",
            f"{p.moe.num_routed_experts} routed + {p.moe.num_shared_experts} shared, "
            f"top-{p.moe.num_experts_per_tok}",
            _verified_tag(),
        )
    if p.sliding_window:
        table.add_row("sliding_window", str(p.sliding_window), _verified_tag())
    if p.position and p.position.max_position_embeddings:
        table.add_row(
            "max_position_embeddings",
            f"{p.position.max_position_embeddings:,}",
            _verified_tag(),
        )

    console.print(table)
    if p.auxiliary.get("warning"):
        console.print(f"[red]⚠ {p.auxiliary['warning']}[/red]")
    if p.auxiliary.get("v0_1_unsupported"):
        console.print(
            "[yellow]⚠ State-space models are not supported in v0.1 (planned for v0.3+).[/yellow]"
        )


def _render_weight(report: EvaluationReport, console: Console) -> None:
    table = Table(title="Weights", show_header=False, box=None, padding=(0, 2))
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_column("label")

    w = report.weight
    table.add_row(
        "safetensors bytes",
        _fmt_bytes(w.total_bytes.value),
        format_tag(w.total_bytes),
    )
    table.add_row(
        "estimated total params",
        _fmt_params(report.total_params_estimate.value),
        format_tag(report.total_params_estimate),
    )
    if w.bits_per_param is not None:
        table.add_row(
            "bits/param",
            f"{w.bits_per_param.value:.2f}",
            format_tag(w.bits_per_param),
        )
    table.add_row(
        "quantization guess",
        str(w.quantization_guess.value),
        format_tag(w.quantization_guess),
    )
    console.print(table)

    # The reconciliation sidelight — DeepSeek-V4-Flash story lives here
    r = report.reconciliation
    if r.candidates:
        rec_table = Table(
            title="Quantization reconciliation (observed vs predicted per scheme)",
            title_justify="left",
            show_header=True,
            header_style="dim",
            box=None,
            padding=(0, 2),
        )
        rec_table.add_column("scheme")
        rec_table.add_column("predicted bytes", justify="right")
        rec_table.add_column("delta", justify="right")
        rec_table.add_column("error %", justify="right")
        for c in r.candidates[:6]:
            rec_table.add_row(
                c.scheme,
                _fmt_bytes(c.predicted_bytes),
                _fmt_bytes(abs(c.delta_bytes)) + (" over" if c.delta_bytes > 0 else " under"),
                f"{c.relative_error * 100:.1f}%",
            )
        console.print(rec_table)
        console.print(f"[bold]best match:[/bold] {r.best.value}  {format_tag(r.best)}")


def _render_kv_cache(report: EvaluationReport, console: Console) -> None:
    if not report.kv_cache_by_context:
        return
    table = Table(
        title="KV cache per request (BF16/FP16)",
        title_justify="left",
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 2),
    )
    table.add_column("context")
    table.add_column("KV cache", justify="right")
    table.add_column("label")
    for ctx, av in report.kv_cache_by_context.items():
        table.add_row(
            f"{ctx:,} tokens",
            _fmt_bytes(av.value),
            format_tag(av),
        )
    console.print(table)


def _render_label_legend(console: Console) -> None:
    legend = Text()
    legend.append("labels: ", style="dim")
    for label in Label:
        tag = f"[{label.value}] "
        legend.append(tag, style=_LABEL_STYLES.get(label, "white"))
    console.print(legend)


def _verified_tag(_source: str | None = None) -> Text:
    return Text(f"[{Label.VERIFIED.value}]", style=_LABEL_STYLES[Label.VERIFIED])
