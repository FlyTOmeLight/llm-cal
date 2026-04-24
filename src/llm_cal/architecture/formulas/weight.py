"""Weight count estimation — total parameters and total bytes by assumption.

Two distinct purposes, kept separate by label:
  * estimate_total_params(profile) -> [estimated] param count
  * predicted_bytes_under_quant(params, scheme) -> [estimated] bytes

The weight_analyzer/reconciler compares predicted_bytes against observed file
sizes to identify the actual quantization scheme. That's the DeepSeek-V4 story.
"""

from __future__ import annotations

from llm_cal.architecture.profile import ArchitectureProfile
from llm_cal.output.labels import AnnotatedValue, Label
from llm_cal.weight_analyzer import _QUANT_BPP, QuantizationScheme


def estimate_total_params(profile: ArchitectureProfile) -> AnnotatedValue[int]:
    """Rough param count from Profile.

    Core components (transformer block):
      - Embedding: vocab_size * hidden_size (+ output head if not tied)
      - Per-layer attention: depends on variant
      - Per-layer FFN: depends on dense vs MoE

    Returns [estimated] — this is an arithmetic sum over config values, several
    simplifying assumptions (e.g. RMSNorm gamma counted in overhead).
    """
    if profile.num_hidden_layers <= 0 or profile.hidden_size <= 0:
        return AnnotatedValue(0, Label.UNKNOWN, source="insufficient shape info in profile")

    hidden = profile.hidden_size
    n_layers = profile.num_hidden_layers
    vocab = profile.vocab_size

    # Embedding + output head (assume untied).
    embed_params = vocab * hidden
    output_head_params = vocab * hidden

    # Per-layer attention projections.
    attn_params = _attention_params(profile)

    # Per-layer FFN (dense path) OR MoE expert block.
    ffn_params = _ffn_params(profile)

    # Per-layer LayerNorms (2 of them, one scalar per feature).
    norm_params = 2 * hidden

    per_layer = attn_params + ffn_params + norm_params
    total = embed_params + output_head_params + per_layer * n_layers

    return AnnotatedValue(
        total,
        Label.ESTIMATED,
        source=(
            f"{vocab} vocab * {hidden} hidden * 2 (embed+head) + "
            f"{n_layers} layers * ({attn_params:,} attn + {ffn_params:,} ffn + norms)"
        ),
    )


def _attention_params(profile: ArchitectureProfile) -> int:
    """Parameter count for attention projections (Q/K/V/O) in one layer."""
    attn = profile.attention
    if attn is None:
        return 0
    hidden = profile.hidden_size

    # MLA uses low-rank projections — very different shape.
    if attn.variant == "MLA" and attn.q_lora_rank:
        q_lora = attn.q_lora_rank
        kv_lora = attn.kv_lora_rank or attn.q_lora_rank  # approximate
        # W_q_down + W_q_up + W_kv_down + W_kv_up + W_o_down + W_o_up
        head_total = attn.num_heads * attn.head_dim
        return (
            hidden * q_lora  # Q down
            + q_lora * head_total  # Q up
            + hidden * kv_lora * 2  # K+V down (shared)
            + kv_lora * head_total  # K+V up
            + head_total * q_lora  # O down (reuse q_lora as o_lora approx)
            + q_lora * hidden  # O up
        )

    # Standard/GQA/MQA: Q + K + V + O projections
    q_out = attn.num_heads * attn.head_dim
    kv_out = attn.num_kv_heads * attn.head_dim
    return hidden * q_out + hidden * kv_out * 2 + q_out * hidden


def _ffn_params(profile: ArchitectureProfile) -> int:
    """Parameter count for the FFN (MoE or dense) in one layer.

    For MoE, counts all experts (routed + shared) because they all live in memory.
    Active parameters per token is a different metric (not our job here).
    """
    hidden = profile.hidden_size

    if profile.moe is not None:
        moe = profile.moe
        # SwiGLU-style expert: 3 matrices (gate, up, down), each hidden x moe_intermediate.
        single_expert = 3 * hidden * moe.moe_intermediate_size
        total_experts = moe.num_routed_experts + moe.num_shared_experts
        # Router: hidden x num_routed_experts
        router = hidden * moe.num_routed_experts
        return single_expert * total_experts + router

    # Dense: try to read intermediate_size from auxiliary; fallback to 4 * hidden.
    intermediate = profile.auxiliary.get("intermediate_size")
    if not isinstance(intermediate, int) or intermediate <= 0:
        intermediate = 4 * hidden
    # SwiGLU: 3 matrices
    return 3 * hidden * intermediate


def predicted_bytes_under_quant(
    total_params: int, scheme: QuantizationScheme
) -> AnnotatedValue[int]:
    """How many bytes `total_params` would occupy under a given quantization."""
    bpp = _QUANT_BPP.get(scheme, 0.0)
    if bpp == 0.0:
        return AnnotatedValue(
            0,
            Label.UNKNOWN,
            source=f"no bytes-per-param mapping for {scheme}",
        )
    predicted = int(total_params * bpp)
    return AnnotatedValue(
        predicted,
        Label.ESTIMATED,
        source=f"{total_params:,} params * {bpp} bytes/param ({scheme})",
    )
