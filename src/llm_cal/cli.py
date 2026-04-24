"""CLI entry point. Thin shell over `Evaluator`. Real flags land in Week 5."""

from __future__ import annotations

import typer
from rich.console import Console

from llm_cal.core.evaluator import Evaluator

app = typer.Typer(
    name="llm-cal",
    help="LLM inference hardware calculator.",
    no_args_is_help=True,
)
_console = Console()


@app.command()
def main(
    model_id: str = typer.Argument(..., help="HuggingFace or ModelScope model id"),
    gpu: str = typer.Option(..., "--gpu", help="GPU type, e.g. H800, A100-80G"),
    engine: str = typer.Option("vllm", "--engine", help="Inference engine: vllm | sglang"),
    gpu_count: int | None = typer.Option(
        None, "--gpu-count", help="Force GPU count (otherwise tool recommends min/dev/prod)"
    ),
    context_length: int | None = typer.Option(
        None, "--context-length", help="Context length for KV cache estimation"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass cache and re-fetch"),
) -> None:
    """Evaluate a model against target hardware."""
    evaluator = Evaluator()
    report = evaluator.evaluate(
        model_id=model_id,
        gpu=gpu,
        engine=engine,
        gpu_count=gpu_count,
        context_length=context_length,
        refresh=refresh,
    )
    _console.print(f"[bold cyan]{report.model_id}[/bold cyan]")
    _console.print(report.message)


if __name__ == "__main__":
    app()
