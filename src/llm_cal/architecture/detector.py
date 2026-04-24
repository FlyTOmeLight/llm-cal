"""`detect()` — main orchestration over trait sub-detectors.

Step 1: Family dispatch (state_space vs transformer vs unknown).
Step 2: Gather traits (independent sub-detectors).
Step 3: Assemble Profile with a confidence level.

Fallback path: `_fallback_unknown()` for configs missing key fields. This is
the bedrock of "works on day-0" — new model types degrade gracefully.
"""

from __future__ import annotations

from typing import Any

from llm_cal.architecture.profile import (
    ArchitectureProfile,
    Confidence,
    Family,
)
from llm_cal.architecture.traits import (
    detect_attention,
    detect_moe,
    detect_position,
    detect_sliding_window,
)

# Model types we know we handle well. Maintained alongside engine_compat matrix.
KNOWN_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v3_2",
        "deepseek_v4",
        "gemma",
        "gemma2",
        "gemma3",
        "phi",
        "phi3",
    }
)

STATE_SPACE_TYPES: frozenset[str] = frozenset({"mamba", "mamba2", "falcon_mamba", "jamba"})


def detect(config: dict[str, Any]) -> ArchitectureProfile:
    """Main entry. Given a parsed config.json dict, return an ArchitectureProfile."""
    model_type = str(config.get("model_type", "")).lower()

    # Step 1: state_space family short-circuits — v0.1 unsupported, but we identify it
    if model_type in STATE_SPACE_TYPES or "ssm_cfg" in config:
        return ArchitectureProfile(
            model_type=model_type,
            architectures=tuple(str(a).lower() for a in config.get("architectures", [])),
            family=Family.STATE_SPACE,
            num_hidden_layers=int(config.get("num_hidden_layers", 0)),
            hidden_size=int(config.get("hidden_size", 0)),
            vocab_size=int(config.get("vocab_size", 0)),
            confidence=Confidence.HIGH,
            auxiliary={"v0_1_unsupported": True},
        )

    # Step 2: reject if fundamentally unidentifiable
    if not model_type and not config.get("architectures"):
        return _fallback_unknown(config)

    # Step 3: required fields
    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    if not num_layers or not hidden_size:
        return _fallback_unknown(config)

    # Step 4: gather traits (each is independent and may return None)
    attention = detect_attention(config)
    moe = detect_moe(config)
    position = detect_position(config)
    sliding = detect_sliding_window(config)

    # Step 5: confidence — HIGH iff model_type is in the registry
    confidence = Confidence.HIGH if model_type in KNOWN_MODEL_TYPES else Confidence.MEDIUM

    # Pass-through of config fields our formulas can use downstream. Keeps the
    # Profile schema stable while enabling richer computation (e.g. dense FFN
    # param count needs intermediate_size).
    auxiliary: dict[str, object] = {}
    if isinstance(config.get("intermediate_size"), int):
        auxiliary["intermediate_size"] = config["intermediate_size"]
    if config.get("tie_word_embeddings") is not None:
        auxiliary["tie_word_embeddings"] = bool(config["tie_word_embeddings"])

    return ArchitectureProfile(
        model_type=model_type,
        architectures=tuple(str(a).lower() for a in config.get("architectures", [])),
        family=Family.TRANSFORMER,
        num_hidden_layers=int(num_layers),
        hidden_size=int(hidden_size),
        vocab_size=int(config.get("vocab_size", 0)),
        confidence=confidence,
        attention=attention,
        moe=moe,
        position=position,
        sliding_window=sliding,
        auxiliary=auxiliary,
    )


def _fallback_unknown(config: dict[str, Any]) -> ArchitectureProfile:
    """Graceful degradation when config.json is unusable.

    Still returns a valid Profile. Consumers check `family == Family.UNKNOWN`
    or `confidence == Confidence.LOW` and skip KV-cache estimation accordingly.
    """
    return ArchitectureProfile(
        model_type=str(config.get("model_type", "")).lower(),
        architectures=tuple(str(a).lower() for a in config.get("architectures", [])),
        family=Family.UNKNOWN,
        num_hidden_layers=int(config.get("num_hidden_layers", 0)),
        hidden_size=int(config.get("hidden_size", 0)),
        vocab_size=int(config.get("vocab_size", 0)),
        confidence=Confidence.LOW,
        auxiliary={
            "warning": (
                "No recognizable model_type or missing essential config fields. "
                "Weight estimate from safetensors file size only; "
                "KV cache cannot be estimated; engine compatibility unknown."
            )
        },
    )
