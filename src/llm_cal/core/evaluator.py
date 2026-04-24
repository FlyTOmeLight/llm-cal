"""Evaluator — the single orchestration layer.

Placeholder for Week 5. Stubbed now so `from llm_cal import Evaluator` works and
`cli.py` can import. Real orchestration logic is added in Week 5 once all the
downstream modules (fleet, command_generator, output/formatter) are in place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationReport:
    """What the evaluator returns. Populated progressively through the weeks."""

    model_id: str
    message: str


class Evaluator:
    """Orchestrates: model_source → architecture.detect → weight_analyzer →
    engine_compat → hardware → fleet.planner → command_generator.

    v0.1 skeleton — implementation lands in Week 5.
    """

    def evaluate(
        self,
        model_id: str,
        gpu: str,
        engine: str,
        gpu_count: int | None = None,
        context_length: int | None = None,
        refresh: bool = False,
    ) -> EvaluationReport:
        return EvaluationReport(
            model_id=model_id,
            message=(
                "llm-cal is under active development. "
                "End-to-end evaluation lands in Week 5 of the build plan."
            ),
        )
