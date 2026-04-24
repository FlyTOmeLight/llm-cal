"""KV cache estimation — traits-composed formula.

The formula is NOT owned by a single architecture module. Instead we compose it
from the traits on `ArchitectureProfile`:

  baseline = 2 (K+V) * num_kv_heads * head_dim * seq_len * dtype_bytes * num_layers

Then apply compositional modifiers:
  * MLA:            baseline uses kv_lora_rank instead of num_kv_heads * head_dim
                    (DeepSeek's compressed KV representation)
  * CSA_HCA:        multiply by an effective-ratio derived from compress_ratios
                    (most layers are heavily compressed, a few are dense)
  * Sliding window: cap `seq_len` at the window size
  * NSA:            multiply by (nsa_topk / seq_len), clamped — sparse attention
                    keeps only top-k keys

Returns AnnotatedValue tagged [estimated] unless we can't compute it at all.
"""

from __future__ import annotations

from llm_cal.architecture.profile import (
    ArchitectureProfile,
    AttentionTraits,
    Confidence,
    Family,
)
from llm_cal.output.labels import AnnotatedValue, Label


def compute_kv_cache_bytes(
    profile: ArchitectureProfile,
    seq_len: int,
    dtype_bytes: int = 2,  # BF16/FP16 default
) -> AnnotatedValue[int]:
    """KV cache per single request at `seq_len` tokens.

    Returns AnnotatedValue. The label tells the user whether we could compute it
    at all.
    """
    if seq_len <= 0:
        return AnnotatedValue(0, Label.ESTIMATED, source="seq_len <= 0")

    if profile.family == Family.STATE_SPACE:
        return AnnotatedValue(
            0,
            Label.UNKNOWN,
            source="state-space model has no KV cache concept",
        )

    if profile.family == Family.UNKNOWN or profile.confidence == Confidence.LOW:
        return AnnotatedValue(
            0,
            Label.UNKNOWN,
            source="unknown architecture — cannot estimate KV cache",
        )

    if profile.attention is None or profile.num_hidden_layers <= 0:
        return AnnotatedValue(
            0,
            Label.UNKNOWN,
            source="missing attention traits or layer count",
        )

    attn = profile.attention
    n_layers = profile.num_hidden_layers

    # Step 1: effective seq_len.
    # Sliding window applies ONLY to standard attention (MHA/GQA/MQA). For
    # explicitly-sparse variants (CSA_HCA, NSA), the sparse mechanism already
    # encodes per-layer reduction; stacking sliding cap would double-count and
    # produce absurdly small estimates (measured 1000x too low on DeepSeek-V4).
    effective_seq = seq_len
    sliding_note = ""
    is_sparse_variant = attn.variant in ("CSA_HCA", "NSA")
    if profile.sliding_window and profile.sliding_window > 0 and not is_sparse_variant:
        effective_seq = min(seq_len, profile.sliding_window)
        if effective_seq < seq_len:
            sliding_note = (
                f" (sliding_window={profile.sliding_window} caps {seq_len} -> {effective_seq})"
            )

    # Step 2: per-layer per-token cache size
    per_layer_per_token = _per_layer_per_token_bytes(attn, dtype_bytes)

    # Step 3: baseline for the full layer stack
    baseline = per_layer_per_token * effective_seq * n_layers

    # Step 4: compositional modifier for sparse attention
    result_bytes = baseline
    variant_note: str = str(attn.variant)

    if attn.variant == "CSA_HCA" and attn.compress_ratios:
        ratio = _average_csa_hca_ratio(attn.compress_ratios)
        result_bytes = int(baseline * ratio)
        variant_note = f"{variant_note} (avg compress ratio {ratio:.3f})"

    if attn.variant == "NSA" and attn.nsa_topk and attn.nsa_topk > 0:
        sparsity = min(1.0, attn.nsa_topk / effective_seq)
        result_bytes = int(baseline * sparsity)
        variant_note = f"{variant_note} (nsa_topk={attn.nsa_topk}, sparsity={sparsity:.3f})"

    return AnnotatedValue(
        result_bytes,
        Label.ESTIMATED,
        source=(
            f"{variant_note}: 2*kv_shape*{dtype_bytes}B*{effective_seq}*{n_layers}{sliding_note}"
        ),
    )


def _per_layer_per_token_bytes(attn: AttentionTraits, dtype_bytes: int) -> int:
    """Bytes of K+V storage per token per layer, given attention shape."""
    # MLA: KV is compressed into a single latent vector of size kv_lora_rank.
    # (Both K and V share it; it's NOT 2 * kv_lora_rank.)
    if attn.variant == "MLA" and attn.kv_lora_rank:
        return attn.kv_lora_rank * dtype_bytes

    # Standard / GQA / MQA / CSA+HCA (the sparse scaling is applied later).
    # K and V both stored: factor of 2.
    return 2 * attn.num_kv_heads * attn.head_dim * dtype_bytes


def _average_csa_hca_ratio(compress_ratios: tuple[int, ...]) -> float:
    """DeepSeek V4 compress_ratios semantics:

      0   -> dense attention (keep 100%)
      N>0 -> keep 1/N of tokens

    Returns the average "keep fraction" across all layers.

    Example: ratios = [0, 0, 4, 128, 4, 128, ...]
      - two dense layers (fraction = 1.0)
      - remaining alternating 1/4 and 1/128
      - weighted average across all layers
    """
    if not compress_ratios:
        return 1.0
    total_fraction = 0.0
    for r in compress_ratios:
        if r == 0:
            total_fraction += 1.0
        else:
            total_fraction += 1.0 / r
    return total_fraction / len(compress_ratios)
