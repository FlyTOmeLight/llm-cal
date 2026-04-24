"""ModelScope source.

Implementation strategy is pending the Week 0 ADR spike
(docs/adr/001-modelscope-integration-strategy.md). This file is a placeholder
that raises an informative error until the spike resolves whether to use the
SDK or the REST API.
"""

from __future__ import annotations

from llm_cal.model_source.base import (
    ModelArtifact,
    ModelSource,
    SourceUnavailableError,
)


class ModelScopeSource(ModelSource):
    name = "modelscope"

    def fetch(self, model_id: str) -> ModelArtifact:  # pragma: no cover
        raise SourceUnavailableError(
            "ModelScope source not yet implemented. "
            "Pending ADR-001 decision on SDK vs REST. "
            "Track progress in docs/adr/001-modelscope-integration-strategy.md."
        )
