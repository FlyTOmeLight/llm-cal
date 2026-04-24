"""Generate a ready-to-copy SGLang launch command."""

from __future__ import annotations

from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.engine_compat.loader import EngineCompatEntry


def generate_sglang_command(
    model_id: str,
    profile: ArchitectureProfile,
    tensor_parallel_size: int,
    entry: EngineCompatEntry | None,
    max_model_len: int | None = None,
) -> str:
    """Generate a multi-line `python -m sglang.launch_server ...` command string."""
    lines: list[str] = [
        "python -m sglang.launch_server",
        f"  --model-path {model_id}",
        f"  --tp {tensor_parallel_size}",
    ]

    effective_max = max_model_len
    if effective_max is None and profile.position is not None:
        effective_max = profile.position.max_position_embeddings
    if effective_max:
        lines.append(f"  --context-length {effective_max}")

    if _needs_trust_remote_code(profile.model_type):
        lines.append("  --trust-remote-code")

    lines.append("  --mem-fraction-static 0.9")

    if entry is not None:
        for flag in entry.required_flags:
            lines.append("  " + _render_flag(flag.flag, flag.value))
        for flag in entry.optional_flags:
            lines.append("  " + _render_flag(flag.flag, flag.value))

    return " \\\n".join(lines)


def _render_flag(flag: str, value: str | None) -> str:
    if value is None:
        return flag
    return f"{flag} {value}"


def _needs_trust_remote_code(model_type: str) -> bool:
    return model_type.startswith(("deepseek", "qwen2_moe", "qwen3_moe", "mixtral"))
