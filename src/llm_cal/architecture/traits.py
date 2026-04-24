"""Independent trait sub-detectors.

Each function inspects config.json and returns a trait dataclass (or None).
They co-exist: a MoE+MLA+CSA_HCA model matches all three.

Dispatch order inside `detect_attention()` is critical because some keys are
ambiguous (e.g. num_kv_heads < num_heads can be GQA OR a side-effect of MLA
where there's a single compressed KV head).
"""

from __future__ import annotations

from typing import Any

from llm_cal.architecture.profile import (
    AttentionTraits,
    MoETraits,
    PositionTraits,
)


def detect_moe(config: dict[str, Any]) -> MoETraits | None:
    """MoE detection — presence of any routed-expert key signals MoE."""
    routed = (
        config.get("n_routed_experts")
        or config.get("num_local_experts")
        or config.get("num_experts")
    )
    if not routed:
        return None

    return MoETraits(
        num_routed_experts=int(routed),
        num_shared_experts=int(config.get("n_shared_experts", 0)),
        num_experts_per_tok=int(
            config.get("num_experts_per_tok") or config.get("num_experts_per_token", 1)
        ),
        moe_intermediate_size=int(
            config.get("moe_intermediate_size") or config.get("intermediate_size", 0)
        ),
    )


def detect_attention(config: dict[str, Any]) -> AttentionTraits:
    """Attention variant detection — order-sensitive.

    Priority (first match wins on variant, but shape fields always populated):
      1. CSA+HCA: compress_ratios array, length matches num_hidden_layers
      2. NSA: nsa_config / sparse_attention_cfg present
      3. MLA: q_lora_rank OR kv_lora_rank present
      4. GQA/MQA: num_kv_heads < num_heads
      5. MHA: default
    """
    num_heads = int(config.get("num_attention_heads", 1))
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    head_dim = int(config.get("head_dim") or (config.get("hidden_size", 0) // num_heads or 1))
    num_layers = int(config.get("num_hidden_layers", 0))

    q_lora = config.get("q_lora_rank")
    kv_lora = config.get("kv_lora_rank")
    compress_ratios = config.get("compress_ratios")
    has_nsa = "nsa_config" in config or "sparse_attention_cfg" in config

    # CSA+HCA: length check guards against future variants that happen to use the
    # same key name with different semantics. Reviewer flagged this.
    # Accepted lengths:
    #   - num_hidden_layers
    #   - num_hidden_layers + num_nextn_predict_layers (DeepSeek MTP: one extra
    #     ratio for the next-token prediction head)
    nextn = int(config.get("num_nextn_predict_layers", 0))
    accepted_lengths = {num_layers, num_layers + nextn} if num_layers > 0 else set()
    if (
        isinstance(compress_ratios, list)
        and num_layers > 0
        and len(compress_ratios) in accepted_lengths
    ):
        return AttentionTraits(
            variant="CSA_HCA",
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_lora_rank=int(q_lora) if q_lora else None,
            kv_lora_rank=int(kv_lora) if kv_lora else None,
            compress_ratios=tuple(compress_ratios),
        )

    if has_nsa:
        nsa_cfg = config.get("nsa_config") or config.get("sparse_attention_cfg", {})
        nsa_topk = None
        if isinstance(nsa_cfg, dict):
            nsa_topk = nsa_cfg.get("topk") or nsa_cfg.get("index_topk")
        return AttentionTraits(
            variant="NSA",
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            nsa_topk=int(nsa_topk) if nsa_topk else None,
        )

    if q_lora or kv_lora:
        return AttentionTraits(
            variant="MLA",
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_lora_rank=int(q_lora) if q_lora else None,
            kv_lora_rank=int(kv_lora) if kv_lora else None,
        )

    if num_kv_heads < num_heads:
        variant = "MQA" if num_kv_heads == 1 else "GQA"
        return AttentionTraits(
            variant=variant,  # type: ignore[arg-type]
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    return AttentionTraits(
        variant="MHA",
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def detect_position(config: dict[str, Any]) -> PositionTraits:
    rope_scaling = config.get("rope_scaling") or {}
    rope_type = (rope_scaling.get("type") or rope_scaling.get("rope_type") or "rope").lower()
    if rope_type not in ("rope", "yarn", "alibi", "none"):
        rope_type = "rope"

    return PositionTraits(
        rope_type=rope_type,  # type: ignore[arg-type]
        rope_theta=float(config["rope_theta"]) if config.get("rope_theta") else None,
        rope_scaling_factor=(float(rope_scaling["factor"]) if rope_scaling.get("factor") else None),
        max_position_embeddings=(
            int(config["max_position_embeddings"])
            if config.get("max_position_embeddings")
            else None
        ),
    )


def detect_sliding_window(config: dict[str, Any]) -> int | None:
    """Return window size if sliding-window attention is used, else None."""
    sw = config.get("sliding_window")
    if sw is None or sw == 0:
        return None
    return int(sw)
