"""Tests for quantization fingerprinting — pure functions, no network."""

from __future__ import annotations

from llm_cal.weight_analyzer.fingerprint import (
    from_config,
    from_safetensors_dtypes,
)


class TestFromConfig:
    def test_gptq_int4(self):
        fp = from_config({"quantization_config": {"quant_method": "gptq", "bits": 4}})
        assert fp is not None
        assert fp.scheme == "GPTQ_INT4"
        assert fp.source_type == "config_json"
        assert "gptq" in fp.evidence

    def test_gptq_int8(self):
        fp = from_config({"quantization_config": {"quant_method": "gptq", "bits": 8}})
        assert fp is not None
        assert fp.scheme == "INT8"

    def test_awq_int4(self):
        fp = from_config({"quantization_config": {"quant_method": "awq", "bits": 4}})
        assert fp is not None
        assert fp.scheme == "AWQ_INT4"

    def test_fp8(self):
        fp = from_config({"quantization_config": {"quant_method": "fp8"}})
        assert fp is not None
        assert fp.scheme == "FP8"

    def test_compressed_tensors_fp8(self):
        fp = from_config(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {"num_bits": 8, "type": "float"},
                        }
                    },
                }
            }
        )
        assert fp is not None
        assert fp.scheme == "FP8"

    def test_compressed_tensors_int4(self):
        fp = from_config(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {"num_bits": 4, "type": "int"},
                        }
                    },
                }
            }
        )
        assert fp is not None
        assert fp.scheme == "INT4"

    def test_bitsandbytes_4bit(self):
        fp = from_config(
            {
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                }
            }
        )
        assert fp is not None
        assert fp.scheme == "INT4"

    def test_weight_dtype_fp8(self):
        fp = from_config({"quantization_config": {"weight_dtype": "float8_e4m3fn"}})
        assert fp is not None
        assert fp.scheme == "FP8"

    def test_no_quantization_config(self):
        """Vanilla FP16 model: no quantization_config → no fingerprint."""
        fp = from_config({"torch_dtype": "bfloat16"})
        assert fp is None

    def test_empty_config(self):
        fp = from_config({})
        assert fp is None

    def test_malformed_quantization_config(self):
        """Non-dict quantization_config value — degrade gracefully."""
        fp = from_config({"quantization_config": "not a dict"})
        assert fp is None

    def test_unknown_quant_method(self):
        """Future quant methods we don't recognize → return None."""
        fp = from_config({"quantization_config": {"quant_method": "something_new", "bits": 4}})
        assert fp is None


class TestFromSafetensorsDtypes:
    def test_gptq_fingerprint(self):
        """GPTQ has distinctive .qweight + .g_idx tensor names."""
        dtypes = {
            "model.layers.0.self_attn.q_proj.qweight": "I32",
            "model.layers.0.self_attn.q_proj.qzeros": "I32",
            "model.layers.0.self_attn.q_proj.scales": "F16",
            "model.layers.0.self_attn.q_proj.g_idx": "I32",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "GPTQ_INT4"
        assert "g_idx" in fp.evidence

    def test_awq_fingerprint(self):
        """AWQ has .qweight + .qzeros but no .g_idx."""
        dtypes = {
            "model.layers.0.self_attn.q_proj.qweight": "I32",
            "model.layers.0.self_attn.q_proj.qzeros": "I32",
            "model.layers.0.self_attn.q_proj.scales": "F16",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "AWQ_INT4"

    def test_fp4_fp8_mixed(self):
        """DeepSeek-V4 pattern: FP4 + FP8 mixed (older toolchain naming)."""
        dtypes = {
            "model.layers.0.mlp.experts.0.w1.weight": "F4_E2M1",
            "model.layers.0.mlp.experts.0.w2.weight": "F4_E2M1",
            "model.layers.0.self_attn.q_proj.weight": "F8_E4M3",
            "model.layers.0.input_layernorm.weight": "BF16",  # norm stays higher
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "FP4_FP8_MIXED"

    def test_mx_block_scaled_fp4_fp8(self):
        """Real DeepSeek-V4-Flash pattern: MX block-scaled with F8_E8M0
        scale tensors + I8 packed FP4 weights + a few F8_E4M3 layers.

        This is the actual on-disk dtype profile we observed when fetching
        the middle shard. Older heuristic missed it because there's no
        literal F4_E2M1 dtype anywhere — the FP4 values are bit-packed into
        I8 with a separate F8_E8M0 scale tensor per block.
        """
        dtypes = {
            # Packed FP4 weights (the bulk)
            "model.layers.5.mlp.experts.0.w1.weight": "I8",
            "model.layers.5.mlp.experts.0.w2.weight": "I8",
            "model.layers.5.mlp.experts.0.w3.weight": "I8",
            # MX scale tensors (one per quantization block)
            "model.layers.5.mlp.experts.0.w1.weight_scale": "F8_E8M0",
            "model.layers.5.mlp.experts.0.w2.weight_scale": "F8_E8M0",
            # FP8 sub-pack for some layers
            "model.layers.5.self_attn.q_proj.weight": "F8_E4M3",
            # Norms stay BF16
            "model.layers.5.input_layernorm.weight": "BF16",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "FP4_FP8_MIXED"
        assert "MX" in fp.evidence

    def test_mxfp4_no_fp8_layers(self):
        """Pure MXFP4 (block-scaled FP4, no FP8 sub-pack)."""
        dtypes = {
            "model.layers.0.mlp.gate_proj.weight": "I8",
            "model.layers.0.mlp.gate_proj.weight_scale": "F8_E8M0",
            "model.layers.0.mlp.up_proj.weight": "I8",
            "model.layers.0.mlp.up_proj.weight_scale": "F8_E8M0",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        # MXFP4 maps to FP4_FP8_MIXED bpp anchor (closest existing scheme)
        assert fp.scheme == "FP4_FP8_MIXED"
        assert "MXFP4" in fp.evidence

    def test_pure_fp8(self):
        dtypes = {
            "model.layers.0.self_attn.q_proj.weight": "F8_E4M3",
            "model.layers.0.self_attn.k_proj.weight": "F8_E4M3",
            "model.layers.0.mlp.gate_proj.weight": "F8_E4M3",
            "model.layers.0.input_layernorm.weight": "BF16",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "FP8"

    def test_pure_fp16(self):
        dtypes = {
            "model.layers.0.self_attn.q_proj.weight": "F16",
            "model.layers.0.self_attn.k_proj.weight": "F16",
            "model.layers.0.mlp.gate_proj.weight": "F16",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "FP16"

    def test_pure_bf16(self):
        dtypes = {
            "model.layers.0.self_attn.q_proj.weight": "BF16",
            "model.layers.0.mlp.gate_proj.weight": "BF16",
        }
        fp = from_safetensors_dtypes(dtypes)
        assert fp is not None
        assert fp.scheme == "BF16"

    def test_empty_dtypes(self):
        assert from_safetensors_dtypes({}) is None

    def test_only_norms_and_embeddings(self):
        """Only non-weight tensors — heuristic should return None or fall through."""
        dtypes = {
            "model.embed_tokens.weight": "BF16",  # embed excluded but contains "weight"
            "model.norm.weight": "BF16",
        }
        # With only excluded tensors, fallback uses all dtypes, so returns BF16.
        fp = from_safetensors_dtypes(dtypes)
        # Either None or BF16 is acceptable — both are honest outcomes.
        assert fp is None or fp.scheme == "BF16"
