"""ArchitectureProfile — the core data class the whole tool orbits.

Key insight: an architecture is NOT a single label. It's a combination of independent
traits that co-exist on a Profile. DeepSeek-V3.2 = MoE + MLA + NSA — three traits.
Single-module dispatch cannot express this; traits composition can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Family(StrEnum):
    TRANSFORMER = "transformer"
    STATE_SPACE = "state_space"  # Mamba, etc. — v0.1 unsupported
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"  # model_type in KNOWN_MODEL_TYPES, all fields present
    MEDIUM = "medium"  # model_type unknown but architectures[] or config partial
    LOW = "low"  # fallback path, config.json missing or malformed


AttentionVariant = Literal["MHA", "GQA", "MQA", "MLA", "NSA", "CSA_HCA"]


@dataclass(frozen=True)
class AttentionTraits:
    """Attention layer shape. Populated by `detect_attention()`."""

    variant: AttentionVariant
    num_heads: int
    num_kv_heads: int
    head_dim: int
    # MLA-specific (DeepSeek V2+)
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    # Sparse attention (CSA+HCA per DeepSeek V4)
    compress_ratios: tuple[int, ...] | None = None
    # Sparse attention (NSA per DeepSeek V3.2)
    nsa_topk: int | None = None


@dataclass(frozen=True)
class MoETraits:
    """MoE-specific layer shape. None on Profile means dense."""

    num_routed_experts: int
    num_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int


@dataclass(frozen=True)
class PositionTraits:
    """RoPE / YaRN / AliBi / none."""

    rope_type: Literal["rope", "yarn", "alibi", "none"] = "rope"
    rope_theta: float | None = None
    rope_scaling_factor: float | None = None
    max_position_embeddings: int | None = None


@dataclass(frozen=True)
class ArchitectureProfile:
    """Complete architectural snapshot of a model.

    This drives weight/KV-cache formulas, engine matching, and fleet planning.
    """

    model_type: str  # config.json's `model_type` (lowercase)
    architectures: tuple[str, ...]  # config.json's `architectures[]`
    family: Family
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    confidence: Confidence
    # Traits (composable — not all populated)
    attention: AttentionTraits | None = None
    moe: MoETraits | None = None
    position: PositionTraits | None = None
    sliding_window: int | None = None  # None = no window
    # Pass-through for traits we haven't categorised yet
    auxiliary: dict[str, object] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    @property
    def is_sparse_attention(self) -> bool:
        if self.attention is None:
            return False
        return self.attention.variant in ("NSA", "CSA_HCA")
