"""Benchmark runner — validate llm-cal's output against curated references.

For each entry in dataset.yaml, run the evaluator against the model, then
compare each `expectations[]` field with the predicted value. Report a
table of pass/fail per check, plus a summary.

This is NOT a synthetic benchmark. Every expected value cites a source
(HF API, model card text, vLLM recipe, hand computation) so users can
audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from llm_cal.common.yaml_loader import load_yaml
from llm_cal.core.evaluator import EvaluationReport, Evaluator

Status = Literal["PASS", "FAIL", "SKIP"]


class Expectation(BaseModel):
    field: str
    # Exactly one of these is used depending on `field`
    expected: str | int | bool | None = None
    expected_min: int | None = None
    expected_max: int | None = None
    source: str


class BenchmarkEntry(BaseModel):
    name: str
    model_id: str
    gpu: str
    engine: str = "vllm"
    expectations: list[Expectation] = Field(default_factory=list)


class BenchmarkDataset(BaseModel):
    schema_version: int
    entries: list[BenchmarkEntry]


@dataclass(frozen=True)
class CheckResult:
    entry_name: str
    field: str
    status: Status
    predicted: str
    expected: str
    source: str
    note: str | None = None


def _default_dataset_path() -> Path:
    return Path(str(files("llm_cal.benchmark").joinpath("dataset.yaml")))


@lru_cache(maxsize=1)
def load_dataset(path: Path | None = None) -> BenchmarkDataset:
    return load_yaml(path or _default_dataset_path(), BenchmarkDataset)


def run_all(
    evaluator: Evaluator | None = None,
    dataset: BenchmarkDataset | None = None,
) -> list[CheckResult]:
    """Run every check in the dataset. Returns flat list of results."""
    evaluator = evaluator or Evaluator()
    dataset = dataset or load_dataset()
    results: list[CheckResult] = []
    for entry in dataset.entries:
        try:
            report = evaluator.evaluate(
                model_id=entry.model_id,
                gpu=entry.gpu,
                engine=entry.engine,
            )
        except Exception as e:
            for exp in entry.expectations:
                results.append(
                    CheckResult(
                        entry_name=entry.name,
                        field=exp.field,
                        status="SKIP",
                        predicted="(evaluation failed)",
                        expected=_fmt_expected(exp),
                        source=exp.source,
                        note=f"{type(e).__name__}: {e}",
                    )
                )
            continue
        for exp in entry.expectations:
            results.append(_check_one(entry.name, report, exp))
    return results


def _check_one(entry_name: str, report: EvaluationReport, exp: Expectation) -> CheckResult:
    predicted_str, status = _evaluate_field(report, exp)
    return CheckResult(
        entry_name=entry_name,
        field=exp.field,
        status=status,
        predicted=predicted_str,
        expected=_fmt_expected(exp),
        source=exp.source,
    )


def _evaluate_field(report: EvaluationReport, exp: Expectation) -> tuple[str, Status]:
    """Return (predicted_str, PASS/FAIL/SKIP) for this field.

    Each `field` name matches a documented check in dataset.yaml.
    """
    if exp.field == "attention_variant":
        attn_actual = report.profile.attention.variant if report.profile.attention else "(none)"
        return attn_actual, ("PASS" if attn_actual == exp.expected else "FAIL")

    if exp.field == "quantization":
        quant_actual = report.weight.quantization_guess.value
        return quant_actual, ("PASS" if quant_actual == exp.expected else "FAIL")

    if exp.field == "is_moe":
        actual_bool = report.profile.is_moe
        return str(actual_bool), ("PASS" if actual_bool == exp.expected else "FAIL")

    if exp.field == "weight_bytes":
        actual_int = report.weight.total_bytes.value
        low = exp.expected_min or 0
        high = exp.expected_max or (1 << 62)
        passed = low <= actual_int <= high
        return f"{actual_int:,}", ("PASS" if passed else "FAIL")

    if exp.field == "fleet_prod_gpus":
        if report.fleet is None:
            return "(no fleet)", "SKIP"
        prod = next((o for o in report.fleet.options if o.tier == "prod"), None)
        if prod is None:
            return "(no prod tier)", "SKIP"
        passed = prod.gpu_count == exp.expected
        return str(prod.gpu_count), ("PASS" if passed else "FAIL")

    if exp.field == "fleet_prod_gpus_at_most":
        if report.fleet is None:
            return "(no fleet)", "SKIP"
        prod = next((o for o in report.fleet.options if o.tier == "prod"), None)
        if prod is None:
            return "(no prod tier)", "SKIP"
        passed = prod.gpu_count <= int(exp.expected or 0)
        return f"{prod.gpu_count} (max {exp.expected})", ("PASS" if passed else "FAIL")

    return "(unknown field)", "SKIP"


def _fmt_expected(exp: Expectation) -> str:
    if exp.expected is not None:
        return str(exp.expected)
    if exp.expected_min is not None or exp.expected_max is not None:
        lo = f"{exp.expected_min:,}" if exp.expected_min is not None else "-∞"
        hi = f"{exp.expected_max:,}" if exp.expected_max is not None else "+∞"
        return f"[{lo}, {hi}]"
    return "(unspecified)"


def render_results(results: list[CheckResult], console: Console | None = None) -> None:
    console = console or Console()

    table = Table(
        title="Benchmark results",
        title_justify="left",
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 2),
    )
    table.add_column("entry")
    table.add_column("field")
    table.add_column("predicted")
    table.add_column("expected")
    table.add_column("status")

    status_styles = {
        "PASS": "bold green",
        "FAIL": "bold red",
        "SKIP": "dim yellow",
    }

    current_entry = None
    for r in results:
        entry_cell = r.entry_name if r.entry_name != current_entry else ""
        current_entry = r.entry_name
        table.add_row(
            entry_cell,
            r.field,
            r.predicted,
            r.expected,
            f"[{status_styles[r.status]}]{r.status}[/]",
        )

    console.print(table)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")

    summary = (
        f"Total: {total}   "
        f"[bold green]PASS: {passed}[/]   "
        f"[bold red]FAIL: {failed}[/]   "
        f"[dim yellow]SKIP: {skipped}[/]"
    )
    console.print(summary)

    if failed > 0:
        console.print(
            "[dim]Failures show the tool's prediction diverges from a curated "
            "source. Check the `source` column for the expected-value provenance.[/]"
        )


def exit_code_from(results: list[CheckResult]) -> int:
    """0 if all PASS or only SKIP; 1 if any FAIL."""
    return 1 if any(r.status == "FAIL" for r in results) else 0
