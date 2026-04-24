"""Token discovery + user-friendly auth error messages."""

from __future__ import annotations

import os


def get_hf_token() -> str | None:
    """Read HF token from standard env vars.

    `HF_TOKEN` wins over `HUGGING_FACE_HUB_TOKEN` for consistency with the
    huggingface-cli default.
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_modelscope_token() -> str | None:
    return os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")


def hf_auth_error_message(model_id: str) -> str:
    return (
        f"Model '{model_id}' requires authentication (gated or private).\n"
        "Set HF_TOKEN env var or run: huggingface-cli login"
    )


def modelscope_auth_error_message(model_id: str) -> str:
    # Chinese user-facing message — full-width punctuation is intentional.
    return (
        f"模型 '{model_id}' 需要登录（gated 或 私有）。\n"
        "设置 MODELSCOPE_API_TOKEN 环境变量，或执行：modelscope login"
    )
