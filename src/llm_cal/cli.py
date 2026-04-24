"""CLI entry point. Thin shell over `Evaluator` + rich formatter."""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from llm_cal.common.i18n import detect_locale_from_env, set_locale, t
from llm_cal.core.evaluator import Evaluator
from llm_cal.model_source.base import (
    AuthRequiredError,
    ModelNotFoundError,
    SourceUnavailableError,
)
from llm_cal.output.formatter import render

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
    model_id: str = typer.Argument(..., help="HuggingFace or ModelScope model id"),
    gpu: str = typer.Option(..., "--gpu", help="GPU type, e.g. H800, A100-80G"),
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
) -> None:
    """Evaluate a model against target hardware."""
    if lang in ("en", "zh"):
        set_locale(lang)  # type: ignore[arg-type]

    evaluator = Evaluator()
    try:
        report = evaluator.evaluate(
            model_id=model_id,
            gpu=gpu,
            engine=engine,
            gpu_count=gpu_count,
            context_length=context_length,
            refresh=refresh,
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
