"""Quantization fingerprinting — tie-breakers for the reconciler.

When `reconciler.reconcile` has multiple schemes tied at the same bits/param
(FP4_FP8_MIXED, GPTQ_INT4, and AWQ_INT4 all sit at bpp=0.55), bytes alone
cannot pick a winner. We resolve the ambiguity with two stronger signals:

  1. `quantization_config` in config.json — explicit declaration by the model
     author. Covers most GPTQ/AWQ/FP8 community uploads.

  2. safetensors per-tensor dtype + tensor-name patterns — the ground truth.
     Covers models like DeepSeek-V4-Flash that use custom mixed-precision
     packs without a config.json declaration.

Both return a `QuantFingerprint`. The reconciler uses the fingerprint's
`scheme` as a tie-breaker, and the `evidence` string flows into the
derivation trace.

This module is pure — no network, no file I/O. `safetensors_reader.py`
handles fetching; this module interprets what was fetched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from llm_cal.weight_analyzer import QuantizationScheme

SourceType = Literal["config_json", "safetensors_header"]


@dataclass(frozen=True)
class QuantFingerprint:
    scheme: QuantizationScheme
    source_type: SourceType
    evidence: str  # for the derivation trace


# ---------------------------------------------------------------------------
# Config.json: explicit quant_method declaration


def from_config(config: dict[str, Any]) -> QuantFingerprint | None:
    """Read `config.json` `quantization_config` and map to a scheme.

    Returns None if no `quantization_config` block exists (model either
    unquantized in-config or uses a per-tensor pack without declaration).
    """
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return None

    quant_method = qc.get("quant_method")
    bits = qc.get("bits")
    weight_dtype = qc.get("weight_dtype")

    # GPTQ family
    if quant_method == "gptq":
        if bits == 4:
            return QuantFingerprint(
                scheme="GPTQ_INT4",
                source_type="config_json",
                evidence="config.json quantization_config.quant_method=gptq, bits=4",
            )
        if bits == 8:
            return QuantFingerprint(
                scheme="INT8",
                source_type="config_json",
                evidence="config.json quantization_config.quant_method=gptq, bits=8",
            )

    # AWQ family
    if quant_method == "awq" and bits == 4:
        return QuantFingerprint(
            scheme="AWQ_INT4",
            source_type="config_json",
            evidence="config.json quantization_config.quant_method=awq, bits=4",
        )

    # FP8 (native or compressed-tensors wrapping)
    if quant_method == "fp8":
        return QuantFingerprint(
            scheme="FP8",
            source_type="config_json",
            evidence="config.json quantization_config.quant_method=fp8",
        )

    # compressed-tensors (RedHatAI etc.) — inspect inner weight dtype
    if quant_method == "compressed-tensors":
        # The config_groups.group_0.weights.type can be "float", "int", etc.
        # and num_bits gives 4/8. For v0.1.2 we handle the two common cases.
        groups = qc.get("config_groups") or {}
        # Pick the first group; schemas with heterogeneous groups degrade
        # gracefully to None (reconciler stays in tied state).
        for g in groups.values():
            if not isinstance(g, dict):
                continue
            weights = g.get("weights") or {}
            num_bits = weights.get("num_bits")
            wtype = weights.get("type")
            if num_bits == 8 and wtype in ("float", "fp8"):
                return QuantFingerprint(
                    scheme="FP8",
                    source_type="config_json",
                    evidence="config.json compressed-tensors group weights=fp8/8bit",
                )
            if num_bits == 8 and wtype == "int":
                return QuantFingerprint(
                    scheme="INT8",
                    source_type="config_json",
                    evidence="config.json compressed-tensors group weights=int/8bit",
                )
            if num_bits == 4 and wtype == "int":
                # Generic INT4 — don't claim GPTQ or AWQ without more evidence
                return QuantFingerprint(
                    scheme="INT4",
                    source_type="config_json",
                    evidence="config.json compressed-tensors group weights=int/4bit",
                )
            break  # first group only

    # bitsandbytes — load_in_4bit / load_in_8bit flags
    if quant_method == "bitsandbytes":
        if qc.get("load_in_4bit"):
            return QuantFingerprint(
                scheme="INT4",
                source_type="config_json",
                evidence="config.json quant_method=bitsandbytes, load_in_4bit=true",
            )
        if qc.get("load_in_8bit"):
            return QuantFingerprint(
                scheme="INT8",
                source_type="config_json",
                evidence="config.json quant_method=bitsandbytes, load_in_8bit=true",
            )

    # Standalone weight_dtype (no nested groups — some custom loaders)
    if weight_dtype in ("float8_e4m3fn", "float8_e5m2"):
        return QuantFingerprint(
            scheme="FP8",
            source_type="config_json",
            evidence=f"config.json quantization_config.weight_dtype={weight_dtype}",
        )

    return None


# ---------------------------------------------------------------------------
# Safetensors header: per-tensor dtype + tensor-name patterns

# safetensors dtype strings (from the format spec)
_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})
_FP4_DTYPES = frozenset({"F4_E2M1", "F4"})  # F4 is used by some toolchains
_FP16_DTYPES = frozenset({"F16"})
_BF16_DTYPES = frozenset({"BF16"})
_INT8_DTYPES = frozenset({"I8", "U8"})


def from_safetensors_dtypes(tensor_dtypes: dict[str, str]) -> QuantFingerprint | None:
    """Fingerprint from a parsed safetensors header (tensor_name -> dtype string).

    Only considers "weight-like" tensors. Non-weight tensors (norms, biases,
    embeddings, LayerNorm params) often stay in FP16/BF16 even in heavily
    quantized models, so counting them directly would give a wrong picture.
    """
    if not tensor_dtypes:
        return None

    names = set(tensor_dtypes.keys())

    # ------------------------------------------------------------------
    # Packed-int4 schemes first — they have distinctive tensor-name markers
    # even though the underlying dtype is I32 (bit-packed).

    has_qweight = any(n.endswith(".qweight") or n.endswith("_qweight") for n in names)
    has_g_idx = any(n.endswith(".g_idx") or n.endswith("_g_idx") for n in names)
    has_qzeros = any(n.endswith(".qzeros") or n.endswith("_qzeros") for n in names)

    if has_qweight and has_g_idx:
        return QuantFingerprint(
            scheme="GPTQ_INT4",
            source_type="safetensors_header",
            evidence="safetensors header has .qweight + .g_idx tensors (GPTQ marker)",
        )
    if has_qweight and has_qzeros and not has_g_idx:
        return QuantFingerprint(
            scheme="AWQ_INT4",
            source_type="safetensors_header",
            evidence="safetensors header has .qweight + .qzeros, no .g_idx (AWQ marker)",
        )

    # ------------------------------------------------------------------
    # Dtype histogram over weight-like tensors.
    # Skip norms / biases / embeddings which typically don't get quantized.

    def _is_weight_tensor(name: str) -> bool:
        lname = name.lower()
        if any(sub in lname for sub in (".norm", ".bias", "embed", "lm_head")):
            return False
        # Tensor names in transformer models usually contain "weight"
        return "weight" in lname or lname.endswith(".w") or lname.endswith(".proj")

    weight_dtypes: list[str] = [dt for n, dt in tensor_dtypes.items() if _is_weight_tensor(n)]
    if not weight_dtypes:
        # Fall back to all dtypes if the name heuristic found nothing
        weight_dtypes = list(tensor_dtypes.values())

    has_fp4 = any(dt in _FP4_DTYPES for dt in weight_dtypes)
    has_fp8 = any(dt in _FP8_DTYPES for dt in weight_dtypes)
    has_fp16 = any(dt in _FP16_DTYPES for dt in weight_dtypes)
    has_bf16 = any(dt in _BF16_DTYPES for dt in weight_dtypes)
    has_int8 = any(dt in _INT8_DTYPES for dt in weight_dtypes)

    # DeepSeek-V4-Flash pattern: real FP4 tensors mixed with FP8 tensors
    if has_fp4 and has_fp8:
        return QuantFingerprint(
            scheme="FP4_FP8_MIXED",
            source_type="safetensors_header",
            evidence=(
                f"safetensors header has both FP4 and FP8 weight tensors "
                f"({sum(dt in _FP4_DTYPES for dt in weight_dtypes)} FP4, "
                f"{sum(dt in _FP8_DTYPES for dt in weight_dtypes)} FP8)"
            ),
        )

    # Pure FP8 — every weight tensor is F8_E4M3 or F8_E5M2
    if has_fp8 and not (has_fp4 or has_int8):
        fp8_count = sum(dt in _FP8_DTYPES for dt in weight_dtypes)
        return QuantFingerprint(
            scheme="FP8",
            source_type="safetensors_header",
            evidence=f"safetensors header: {fp8_count}/{len(weight_dtypes)} weight tensors are FP8",
        )

    # Pure FP16
    if has_fp16 and not (has_fp8 or has_fp4 or has_int8 or has_bf16):
        return QuantFingerprint(
            scheme="FP16",
            source_type="safetensors_header",
            evidence=f"safetensors header: all {len(weight_dtypes)} weight tensors are F16",
        )

    # Pure BF16
    if has_bf16 and not (has_fp8 or has_fp4 or has_int8 or has_fp16):
        return QuantFingerprint(
            scheme="BF16",
            source_type="safetensors_header",
            evidence=f"safetensors header: all {len(weight_dtypes)} weight tensors are BF16",
        )

    # Pure INT8
    if has_int8 and not (has_fp8 or has_fp4 or has_fp16 or has_bf16):
        return QuantFingerprint(
            scheme="INT8",
            source_type="safetensors_header",
            evidence=f"safetensors header: {len(weight_dtypes)} weight tensors are INT8",
        )

    # Mixed in a way we don't have a named scheme for — stay silent
    return None
