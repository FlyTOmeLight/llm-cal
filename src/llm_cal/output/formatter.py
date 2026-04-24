"""Rich-formatted, fully i18n'd output for EvaluationReport.

Every visible string flows through `common.i18n.t()`. To add another locale,
add entries to `_MESSAGES` in i18n.py; no changes here needed.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_cal.common.i18n import get_locale, t
from llm_cal.core.evaluator import EvaluationReport
from llm_cal.engine_compat.loader import EngineCompatEntry, EngineFlag, EngineSource
from llm_cal.fleet.planner import FleetRecommendation
from llm_cal.hardware.loader import GPUDatabase
from llm_cal.output.labels import AnnotatedValue, Label

_LABEL_STYLES: dict[Label, str] = {
    Label.VERIFIED: "bold green",
    Label.INFERRED: "cyan",
    Label.ESTIMATED: "yellow",
    Label.CITED: "blue",
    Label.UNVERIFIED: "bold yellow",
    Label.UNKNOWN: "dim red",
    Label.LLM_OPINION: "magenta",
}


def format_tag(av: AnnotatedValue[Any]) -> Text:
    style = _LABEL_STYLES.get(av.label, "white")
    display = t(f"label.{av.label.value}")  # localized; falls back to English
    return Text(f"[{display}]", style=style)


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
    sha_frag = f" @ {report.commit_sha[:7]}" if report.commit_sha else ""
    console.print(
        Panel.fit(
            f"[bold cyan]{report.model_id}[/bold cyan]  "
            f"[dim]{t('panel.via')} {report.source}{sha_frag}[/dim]",
            border_style="cyan",
        )
    )

    _render_architecture(report, console)
    _render_weight(report, console)
    _render_kv_cache(report, console)
    _render_engine_compat(report, console)
    _render_hardware(report, console)
    _render_fleet(report, console)
    _render_performance(report, console)
    _render_command(report, console)
    _render_label_legend(console)


def _render_architecture(report: EvaluationReport, console: Console) -> None:
    p = report.profile
    table = Table(title=t("section.architecture"), show_header=False, box=None, padding=(0, 2))
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_column("label")

    table.add_row(t("arch.model_type"), p.model_type or t("arch.none"), _verified_tag())
    table.add_row(t("arch.family"), p.family.value, _verified_tag())
    table.add_row(
        t("arch.confidence"), p.confidence.value, Text(f"[{p.confidence.value}]", style="magenta")
    )
    table.add_row(t("arch.layers"), str(p.num_hidden_layers), _verified_tag())
    table.add_row(t("arch.hidden_size"), str(p.hidden_size), _verified_tag())
    table.add_row(t("arch.vocab_size"), f"{p.vocab_size:,}", _verified_tag())

    if p.attention is not None:
        table.add_row(
            t("arch.attention"),
            t(
                "arch.attn_summary",
                variant=p.attention.variant,
                heads=p.attention.num_heads,
                kv_heads=p.attention.num_kv_heads,
                head_dim=p.attention.head_dim,
            ),
            _verified_tag(),
        )
        if p.attention.compress_ratios:
            ratios = p.attention.compress_ratios
            table.add_row(
                t("arch.compress_ratios"),
                t(
                    "arch.compress_ratios_summary",
                    n=len(ratios),
                    dense=sum(1 for r in ratios if r == 0),
                ),
                _verified_tag(),
            )
    if p.moe is not None:
        table.add_row(
            t("arch.moe"),
            t(
                "arch.moe_summary",
                routed=p.moe.num_routed_experts,
                shared=p.moe.num_shared_experts,
                topk=p.moe.num_experts_per_tok,
            ),
            _verified_tag(),
        )
    if p.sliding_window:
        table.add_row(t("arch.sliding_window"), str(p.sliding_window), _verified_tag())
    if p.position and p.position.max_position_embeddings:
        table.add_row(
            t("arch.max_position"),
            f"{p.position.max_position_embeddings:,}",
            _verified_tag(),
        )

    console.print(table)
    if p.auxiliary.get("warning"):
        console.print(f"[red]⚠ {p.auxiliary['warning']}[/red]")
    if p.auxiliary.get("v0_1_unsupported"):
        console.print(f"[yellow]⚠ {t('arch.unsupported_state_space')}[/yellow]")


def _render_weight(report: EvaluationReport, console: Console) -> None:
    table = Table(title=t("section.weights"), show_header=False, box=None, padding=(0, 2))
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_column("label")

    w = report.weight
    table.add_row(
        t("weights.safetensors_bytes"),
        _fmt_bytes(w.total_bytes.value),
        format_tag(w.total_bytes),
    )
    table.add_row(
        t("weights.params_estimated"),
        _fmt_params(report.total_params_estimate.value),
        format_tag(report.total_params_estimate),
    )
    if w.bits_per_param is not None:
        table.add_row(
            t("weights.bits_per_param"),
            f"{w.bits_per_param.value:.2f}",
            format_tag(w.bits_per_param),
        )
    table.add_row(
        t("weights.quant_guess"),
        str(w.quantization_guess.value),
        format_tag(w.quantization_guess),
    )
    console.print(table)

    r = report.reconciliation
    if r.candidates:
        rec_table = Table(
            title=t("section.reconciliation"),
            title_justify="left",
            show_header=True,
            header_style="dim",
            box=None,
            padding=(0, 2),
        )
        rec_table.add_column(t("recon.scheme"))
        rec_table.add_column(t("recon.predicted"), justify="right")
        rec_table.add_column(t("recon.delta"), justify="right")
        rec_table.add_column(t("recon.error_pct"), justify="right")
        for c in r.candidates[:6]:
            direction = t("recon.over") if c.delta_bytes > 0 else t("recon.under")
            rec_table.add_row(
                c.scheme,
                _fmt_bytes(c.predicted_bytes),
                f"{_fmt_bytes(abs(c.delta_bytes))} {direction}",
                f"{c.relative_error * 100:.1f}%",
            )
        console.print(rec_table)
        console.print(f"[bold]{t('recon.best')}[/bold] {r.best.value}  {format_tag(r.best)}")


def _render_kv_cache(report: EvaluationReport, console: Console) -> None:
    if not report.kv_cache_by_context:
        return
    table = Table(
        title=t("section.kv_cache"),
        title_justify="left",
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 2),
    )
    table.add_column(t("kv.context"))
    table.add_column(t("kv.kv_cache"), justify="right")
    table.add_column(t("kv.label"))
    tokens_word = t("kv.tokens")
    for ctx, av in report.kv_cache_by_context.items():
        table.add_row(
            f"{ctx:,} {tokens_word}",
            _fmt_bytes(av.value),
            format_tag(av),
        )
    console.print(table)


def _render_engine_compat(report: EvaluationReport, console: Console) -> None:
    m = report.engine_match
    if m is None:
        console.print()
        console.print(
            f"[dim]{t('section.engine_compat')}:[/dim] [yellow]{t('engine.no_match')}[/yellow]"
        )
        return

    table = Table(
        title=f"{t('section.engine_compat')} — {m.engine}",
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_column("label")

    verif_label = _verif_label(m)
    table.add_row(t("engine.version_spec"), m.version_spec, Text(""))
    table.add_row(t("engine.support"), m.support, verif_label)
    table.add_row(t("engine.verification"), m.verification_level, verif_label)

    if m.required_flags:
        lines = [_fmt_flag(f) for f in m.required_flags]
        table.add_row(t("engine.required_flags"), "\n".join(lines), Text(""))
    if m.optional_flags:
        lines = [_fmt_flag(f) for f in m.optional_flags]
        table.add_row(t("engine.optional_flags"), "\n".join(lines), Text(""))

    caveats = m.caveats_zh if get_locale() == "zh" else m.caveats_en
    if caveats:
        table.add_row(t("engine.caveats"), "\n".join(f"• {c}" for c in caveats), Text(""))

    if m.sources:
        source_lines = [_fmt_source(s) for s in m.sources]
        table.add_row(t("engine.sources"), "\n".join(source_lines), Text(""))

    console.print(table)


def _render_hardware(report: EvaluationReport, console: Console) -> None:
    console.print()
    if report.gpu_spec is None:
        msg = report.gpu_error or f"Unknown GPU '{report.gpu}'"
        console.print(f"[bold red]{t('section.hardware')}:[/bold red] [red]{msg}[/red]")
        return

    spec = report.gpu_spec
    locale = get_locale()
    table = Table(
        title=f"{t('section.hardware')} — {spec.id}",
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("field", style="dim")
    table.add_column("value")

    table.add_row(t("hw.memory"), f"{spec.memory_gb} GB HBM")
    table.add_row(t("hw.nvlink_bandwidth"), f"{spec.nvlink_bandwidth_gbps} GB/s")
    table.add_row(t("hw.fp16_tflops"), f"{spec.fp16_tflops:.0f} TFLOPS")
    table.add_row(t("hw.fp8_support"), t("hw.bool_yes") if spec.fp8_support else t("hw.bool_no"))
    table.add_row(t("hw.fp4_support"), t("hw.bool_yes") if spec.fp4_support else t("hw.bool_no"))
    notes = spec.localized_notes(locale)
    if notes:
        table.add_row(t("hw.notes"), notes)
    if spec.spec_source:
        table.add_row(t("hw.spec_source"), spec.spec_source)
    console.print(table)


def _render_fleet(report: EvaluationReport, console: Console) -> None:
    f = report.fleet
    if f is None:
        if report.gpu_spec is None:
            return  # hardware section already surfaced the error
        console.print(f"[dim]{t('fleet.gpu_spec_unknown')}[/dim]")
        return

    # Decide which context lengths to surface as concurrency columns.
    ctx_cols = _select_concurrency_columns(f)

    table = Table(
        title=f"{t('section.fleet')} — {report.gpu_spec.id if report.gpu_spec else report.gpu}",
        title_justify="left",
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 2),
    )
    table.add_column(t("fleet.col.tier"))
    table.add_column(t("fleet.col.gpus"), justify="right")
    table.add_column(t("fleet.col.weight_per_gpu"), justify="right")
    table.add_column(t("fleet.col.headroom_per_gpu"), justify="right")
    for ctx in ctx_cols:
        table.add_column(
            t("fleet.col.concurrent_at_ctx", ctx=_fmt_ctx(ctx)),
            justify="right",
        )

    for opt in f.options:
        headroom = opt.usable_bytes_per_gpu - opt.weight_bytes_per_gpu
        label_tier = t(f"fleet.tier.{opt.tier}")
        marker = " ★" if opt.tier == f.best_tier else ""
        row_style = None if opt.fits else "dim red"
        conc_map = dict(opt.max_concurrent_by_context)
        row = [
            f"{label_tier}{marker}",
            str(opt.gpu_count),
            _fmt_bytes(opt.weight_bytes_per_gpu),
            _fmt_bytes(headroom) if headroom > 0 else "—",
        ]
        for ctx in ctx_cols:
            n = conc_map.get(ctx, 0)
            row.append(f"~{n}" if n > 0 else "✗")
        table.add_row(*row, style=row_style)

    console.print(table)

    locale = get_locale()
    note = f.constraint_note_zh if locale == "zh" else f.constraint_note_en
    console.print(f"[dim]{t('fleet.constraint')} {note}[/dim]")
    console.print(f"[dim]★ {t('fleet.best_marker')}[/dim]")


def _select_concurrency_columns(f: FleetRecommendation) -> list[int]:
    """Pick which context lengths become concurrency columns in the fleet table.

    Rule: always include 128K if the model supports it; additionally include the
    model's max context if it's larger than 128K. For shorter-context models,
    fall back to 32K or whatever the max is.
    """
    all_ctxs: set[int] = set()
    for opt in f.options:
        for ctx, _ in opt.max_concurrent_by_context:
            all_ctxs.add(ctx)
    if not all_ctxs:
        return []
    picks: list[int] = []
    if 131_072 in all_ctxs:
        picks.append(131_072)
    max_ctx = max(all_ctxs)
    if max_ctx > 131_072 and max_ctx not in picks:
        picks.append(max_ctx)
    if not picks:
        picks.append(32_768 if 32_768 in all_ctxs else max_ctx)
    return picks


def _fmt_ctx(ctx_tokens: int) -> str:
    if ctx_tokens >= 1_000_000:
        if ctx_tokens % 1_000_000 == 0:
            return f"{ctx_tokens // 1_000_000}M"
        return f"{ctx_tokens / 1_000_000:.1f}M"
    if ctx_tokens >= 1024:
        return f"{ctx_tokens // 1024}K"
    return str(ctx_tokens)


def _render_performance(report: EvaluationReport, console: Console) -> None:
    if (
        report.prefill is None
        or report.decode is None
        or report.concurrency is None
        or report.perf_input_tokens is None
        or report.perf_target_tokens_per_sec is None
    ):
        return

    console.print()
    # Assumption banner — surfaces the utilization factors, SLA, and
    # degradation factor. Every number in the performance section depends
    # on these.
    assumptions = t(
        "perf.assumptions_note",
        input_tokens=report.perf_input_tokens,
        output_tokens=report.perf_output_tokens,
        target_tps=report.perf_target_tokens_per_sec,
        prefill_util=report.prefill.utilization,
        decode_util=report.decode.bw_utilization,
        degradation=report.concurrency.degradation_factor,
    )
    console.print(f"[dim italic]{assumptions}[/dim italic]")

    table = Table(
        title=t("section.performance"),
        title_justify="left",
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("field", style="dim")
    table.add_column("value")
    table.add_column("label")

    p = report.prefill
    d = report.decode
    c = report.concurrency

    table.add_row(
        t("perf.prefill_latency"),
        f"{p.latency_ms.value:.1f} ms",
        format_tag(p.latency_ms),
    )
    table.add_row(
        t("perf.decode_throughput_per_gpu"),
        f"{d.per_gpu_tokens_per_sec.value:.1f} tok/s",
        format_tag(d.per_gpu_tokens_per_sec),
    )
    table.add_row(
        t("perf.decode_throughput_cluster"),
        f"{d.cluster_tokens_per_sec.value:.1f} tok/s",
        format_tag(d.cluster_tokens_per_sec),
    )
    if d.moe_active_tokens_per_sec is not None:
        table.add_row(
            t("perf.decode_moe_active_optimistic"),
            f"{d.moe_active_tokens_per_sec.value:.1f} tok/s",
            format_tag(d.moe_active_tokens_per_sec),
        )
    table.add_row(
        t("perf.k_bound"),
        str(c.k_bound.value),
        format_tag(c.k_bound),
    )
    table.add_row(
        t("perf.l_bound"),
        str(c.l_bound.value),
        format_tag(c.l_bound),
    )
    table.add_row(
        t("perf.max_concurrent"),
        str(c.max_concurrent.value),
        format_tag(c.max_concurrent),
    )
    bottleneck_label = t(f"perf.bottleneck.{c.bottleneck}")
    locale = get_locale()
    reason = c.bottleneck_reason_zh if locale == "zh" else c.bottleneck_reason_en
    table.add_row(
        t("perf.bottleneck"),
        f"{bottleneck_label} — {reason}",
        Text(""),
    )
    console.print(table)

    # Always show a short optimization list. Rules are currently static but
    # future versions can pick per bottleneck type.
    console.print(f"[bold]{t('perf.optimization.header')}:[/bold]")
    for key in (
        "perf.opt.quantize_int4",
        "perf.opt.relax_sla",
        "perf.opt.kv_fp8",
        "perf.opt.moe_offload",
    ):
        console.print(f"  • {t(key)}")


def _render_command(report: EvaluationReport, console: Console) -> None:
    if not report.generated_command or report.fleet is None:
        return
    # Figure out which tier we emitted the command for.
    best_tier_opt = next(
        (o for o in report.fleet.options if o.tier == report.fleet.best_tier),
        report.fleet.options[0],
    )
    tier_label = t(f"fleet.tier.{best_tier_opt.tier}")
    header_note = t("command.tier_note", tier=tier_label, gpus=best_tier_opt.gpu_count)
    console.print()
    console.print(
        Panel(
            report.generated_command,
            title=f"{t('section.command')} — {header_note}",
            title_align="left",
            border_style="green",
        )
    )


def _render_label_legend(console: Console) -> None:
    legend = Text()
    legend.append(f"{t('section.labels')} ", style="dim")
    for label in Label:
        display = t(f"label.{label.value}")
        legend.append(f"[{display}] ", style=_LABEL_STYLES.get(label, "white"))
    console.print(legend)


def _verified_tag() -> Text:
    return Text(f"[{t('label.verified')}]", style=_LABEL_STYLES[Label.VERIFIED])


def render_llm_review(result: Any, console: Console | None = None) -> None:
    """Render --llm-review block. Accepts an LLMReviewResult.

    Failure is non-fatal — shows setup hint and continues.
    """
    console = console or Console()
    console.print()
    console.print(Panel.fit(t("section.llm_review"), border_style="magenta"))

    if not result.ok:
        msg = t("llm_review.unavailable", error=result.error or "unknown")
        console.print(f"[yellow]{msg}[/yellow]")
        console.print(f"[dim]{t('llm_review.setup_hint')}[/dim]")
        return

    # Disclaimer first — make it visually distinctive so users don't confuse
    # LLM opinion with the tool's own output.
    disclaimer = t("llm_review.disclaimer", model=result.model, base_url=result.base_url)
    console.print(f"[bold yellow]{disclaimer}[/bold yellow]")
    console.print()
    # The actual review, prefixed with the [llm-opinion] tag so users see
    # it's tagged too.
    tag_style = _LABEL_STYLES[Label.LLM_OPINION]
    tag_display = t(f"label.{Label.LLM_OPINION.value}")
    console.print(f"[{tag_style}][{tag_display}][/{tag_style}]")
    # Print content verbatim (LLM output is markdown-ish; let it through).
    console.print(result.content or "")


def render_explain(entries: list[Any], console: Console | None = None) -> None:
    """Render `--explain` block: full derivation trace for each number.

    `entries` is a list of `core.explain.ExplainEntry`.
    """
    console = console or Console()

    console.print()
    console.print(Panel.fit(t("section.explain"), border_style="magenta"))
    console.print(f"[dim italic]{t('explain.intro')}[/dim italic]")
    console.print()

    for entry in entries:
        # Title bar per entry
        console.print(Panel.fit(f"[bold]{entry.heading}[/bold]", border_style="cyan"))

        # Formula (monospace)
        console.print(f"[bold]{t('explain.formula')}:[/bold]")
        for line in entry.formula.splitlines():
            console.print(f"  [magenta]{line}[/magenta]")

        # Inputs
        if entry.inputs:
            console.print(f"[bold]{t('explain.inputs')}:[/bold]")
            for inp in entry.inputs:
                note = f" [dim]({inp.note})[/dim]" if inp.note else ""
                console.print(
                    f"  [cyan]{inp.name}[/cyan] = {inp.value}  [dim]{inp.label}[/dim]{note}"
                )

        # Steps
        if entry.steps:
            console.print(f"[bold]{t('explain.steps')}:[/bold]")
            for step in entry.steps:
                for line in step.splitlines():
                    console.print(f"  {line}")

        # Result
        console.print(f"[bold]{t('explain.result')}:[/bold]  {entry.result}")

        # Source + methodology anchor
        if entry.source:
            console.print(f"[bold]{t('explain.source')}:[/bold]  {entry.source}")
        if entry.methodology_anchor:
            console.print(
                f"[dim]{t('explain.see_also')}: docs/methodology.md{entry.methodology_anchor}[/dim]"
            )
        console.print()


def render_gpu_list(db: GPUDatabase, console: Console | None = None) -> None:
    """Print the supported-GPU table. Invoked by `llm-cal --list-gpus`."""
    console = console or Console()
    locale = get_locale()

    table = Table(
        title=t("gpus.list.title"),
        title_justify="left",
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 2),
    )
    table.add_column(t("gpus.col.id"))
    table.add_column(t("gpus.col.memory"), justify="right")
    table.add_column(t("gpus.col.nvlink"), justify="right")
    table.add_column(t("gpus.col.fp16"), justify="right")
    table.add_column(t("gpus.col.fp8"), justify="center")
    table.add_column(t("gpus.col.fp4"), justify="center")
    table.add_column(t("gpus.col.aliases"))

    yes = t("hw.bool_yes")
    no = t("hw.bool_no")

    # Preserve YAML insertion order (vendors are grouped there).
    for spec in db.gpus:
        aliases_str = ", ".join(spec.aliases) if spec.aliases else "—"
        nvlink_str = f"{spec.nvlink_bandwidth_gbps} GB/s" if spec.nvlink_bandwidth_gbps else "—"
        table.add_row(
            spec.id,
            f"{spec.memory_gb} GB",
            nvlink_str,
            f"{spec.fp16_tflops:.0f}",
            yes if spec.fp8_support else no,
            yes if spec.fp4_support else no,
            aliases_str,
        )
    console.print(table)
    console.print(f"[dim]{t('gpus.total', count=len(db.gpus))}[/dim]")
    _ = locale  # suppress unused var warn until we add locale-dependent notes column


def _verif_label(entry: EngineCompatEntry) -> Text:
    """Engine compat rows use the same label vocabulary as AnnotatedValue."""
    label = {
        "verified": Label.VERIFIED,
        "cited": Label.CITED,
        "unverified": Label.UNVERIFIED,
    }.get(entry.verification_level, Label.UNKNOWN)
    return Text(f"[{t(f'label.{label.value}')}]", style=_LABEL_STYLES.get(label, "white"))


def _fmt_flag(f: EngineFlag) -> str:
    if f.value is None:
        return f.flag
    return f"{f.flag} {f.value}"


def _fmt_source(s: EngineSource) -> str:
    label = t(f"source.{s.type}")
    if s.type == "tested":
        return f"[{label}] {s.tester} @ {s.hardware} ({s.date})"
    if s.url:
        captured = f" ({t('source.captured_on')} {s.captured_date})" if s.captured_date else ""
        return f"[{label}] {s.url}{captured}"
    return f"[{label}]"
