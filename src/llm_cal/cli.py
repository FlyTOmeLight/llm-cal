"""CLI entry point. Thin shell over `Evaluator` + rich formatter."""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from llm_cal.benchmark.runner import exit_code_from, render_results, run_all
from llm_cal.common.i18n import detect_locale_from_env, set_locale, t
from llm_cal.core.evaluator import Evaluator
from llm_cal.hardware.loader import load_database
from llm_cal.model_source.base import (
    AuthRequiredError,
    ModelNotFoundError,
    SourceUnavailableError,
)
from llm_cal.output.formatter import render, render_gpu_list

# Set locale from env first; --lang flag can override inside main()
set_locale(detect_locale_from_env())

app = typer.Typer(
    name="llm-cal",
    help="LLM inference hardware calculator.",
    no_args_is_help=True,
)
_console = Console()
_err = Console(stderr=True)


@app.command()
def main(
    model_id: str | None = typer.Argument(None, help="HuggingFace or ModelScope model id"),
    gpu: str | None = typer.Option(None, "--gpu", help="GPU type, e.g. H800, A100-80G"),
    engine: str = typer.Option("vllm", "--engine", help="Inference engine: vllm | sglang"),
    gpu_count: int | None = typer.Option(
        None, "--gpu-count", help="Force GPU count (otherwise tool recommends)"
    ),
    context_length: int | None = typer.Option(
        None, "--context-length", help="Context length for KV cache estimation"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass cache and re-fetch"),
    lang: str | None = typer.Option(
        None,
        "--lang",
        help="Output language: en | zh (default auto-detects from LANG env)",
    ),
    list_gpus: bool = typer.Option(
        False,
        "--list-gpus",
        help="List all supported GPUs and exit (no model_id needed)",
    ),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help=(
            "Run the curated benchmark dataset: compare tool output against "
            "reference values from HF API, model cards, vLLM recipes. "
            "Requires network. Exit 0 on all-pass, 1 if any FAIL."
        ),
    ),
    input_tokens: int = typer.Option(
        2000,
        "--input-tokens",
        help="Input token budget for prefill-latency estimation (default: 2000).",
    ),
    output_tokens: int = typer.Option(
        512,
        "--output-tokens",
        help="Output token budget for total-latency math (default: 512).",
    ),
    target_tokens_per_sec: float = typer.Option(
        30.0,
        "--target-tokens-per-sec",
        help="SLA: per-user decode tokens/second (drives L bound). Default: 30.",
    ),
    prefill_util: float = typer.Option(
        0.40,
        "--prefill-util",
        help="Compute utilization factor for prefill (empirical, default 0.40).",
    ),
    decode_bw_util: float = typer.Option(
        0.50,
        "--decode-bw-util",
        help="Memory-bandwidth utilization factor for decode (default 0.50).",
    ),
    concurrency_degradation: float = typer.Option(
        1.0,
        "--concurrency-degradation",
        help=(
            "High-concurrency throughput degradation factor (default 1.0 = "
            "no degradation — the honest baseline). If your engine drops "
            "to 60% efficiency under load, pass 1.67. See docs/methodology.md."
        ),
    ),
) -> None:
    """Evaluate a model against target hardware."""
    if lang in ("en", "zh"):
        set_locale(lang)  # type: ignore[arg-type]

    # Meta commands short-circuit before requiring model_id + --gpu.
    if list_gpus:
        render_gpu_list(load_database(), _console)
        return

    if benchmark:
        results = run_all()
        render_results(results, _console)
        sys.exit(exit_code_from(results))

    if not model_id:
        _err.print("[red]Missing argument MODEL_ID. Use --help for usage.[/red]")
        raise typer.Exit(code=1)
    if not gpu:
        _err.print("[red]Missing option --gpu. Use --list-gpus to see choices.[/red]")
        raise typer.Exit(code=1)

    evaluator = Evaluator()
    try:
        report = evaluator.evaluate(
            model_id=model_id,
            gpu=gpu,
            engine=engine,
            gpu_count=gpu_count,
            context_length=context_length,
            refresh=refresh,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            target_tokens_per_sec=target_tokens_per_sec,
            prefill_utilization=prefill_util,
            decode_bw_utilization=decode_bw_util,
            concurrency_degradation=concurrency_degradation,
        )
    except AuthRequiredError as e:
        _err.print(f"[bold red]{t('cli.err.auth_required')}[/bold red] {e}")
        sys.exit(2)
    except ModelNotFoundError as e:
        _err.print(f"[bold red]{t('cli.err.model_not_found')}[/bold red] {e}")
        sys.exit(3)
    except SourceUnavailableError as e:
        _err.print(f"[bold red]{t('cli.err.source_unavailable')}[/bold red] {e}")
        sys.exit(4)

    render(report, _console)


if __name__ == "__main__":
    app()
