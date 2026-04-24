"""Tests for command_generator/{vllm, sglang}.py."""

from __future__ import annotations

from llm_cal.architecture.profile import (
    ArchitectureProfile,
    AttentionTraits,
    Confidence,
    Family,
    PositionTraits,
)
from llm_cal.command_generator.sglang import generate_sglang_command
from llm_cal.command_generator.vllm import generate_vllm_command
from llm_cal.engine_compat.loader import find_match


def _profile(model_type: str = "llama", max_pos: int = 131072) -> ArchitectureProfile:
    return ArchitectureProfile(
        model_type=model_type,
        architectures=(),
        family=Family.TRANSFORMER,
        num_hidden_layers=80,
        hidden_size=8192,
        vocab_size=128256,
        confidence=Confidence.HIGH,
        attention=AttentionTraits(variant="GQA", num_heads=64, num_kv_heads=8, head_dim=128),
        position=PositionTraits(rope_type="rope", max_position_embeddings=max_pos),
    )


class TestVllmCommand:
    def test_basic_shape(self):
        profile = _profile()
        cmd = generate_vllm_command(
            "meta-llama/Llama-3.3-70B",
            profile,
            tensor_parallel_size=2,
            entry=None,
        )
        assert "vllm serve meta-llama/Llama-3.3-70B" in cmd
        assert "--tensor-parallel-size 2" in cmd
        assert "--max-model-len 131072" in cmd
        assert "--gpu-memory-utilization 0.9" in cmd

    def test_no_trust_remote_for_llama(self):
        cmd = generate_vllm_command("meta-llama/Llama-3.3-70B", _profile("llama"), 2, None)
        assert "--trust-remote-code" not in cmd

    def test_trust_remote_for_deepseek(self):
        cmd = generate_vllm_command(
            "deepseek-ai/DeepSeek-V4-Flash",
            _profile("deepseek_v4", max_pos=1_048_576),
            8,
            None,
        )
        assert "--trust-remote-code" in cmd
        assert "--max-model-len 1048576" in cmd

    def test_max_model_len_override(self):
        cmd = generate_vllm_command(
            "foo/bar", _profile(max_pos=131072), 2, None, max_model_len=32768
        )
        assert "--max-model-len 32768" in cmd
        assert "--max-model-len 131072" not in cmd

    def test_entry_flags_are_appended(self):
        profile = _profile("deepseek_v4", max_pos=1_048_576)
        entry = find_match(engine="vllm", model_type="deepseek_v4")
        assert entry is not None
        cmd = generate_vllm_command("deepseek-ai/DeepSeek-V4-Flash", profile, 8, entry)
        # matrix's optional_flags includes --attention-backend auto
        assert "--attention-backend auto" in cmd


class TestSglangCommand:
    def test_basic_shape(self):
        cmd = generate_sglang_command(
            "deepseek-ai/DeepSeek-V3.2",
            _profile("deepseek_v3_2"),
            tensor_parallel_size=8,
            entry=None,
        )
        assert "python -m sglang.launch_server" in cmd
        assert "--model-path deepseek-ai/DeepSeek-V3.2" in cmd
        assert "--tp 8" in cmd
        assert "--context-length" in cmd

    def test_entry_required_flags_appended(self):
        profile = _profile("deepseek_v3_2")
        entry = find_match(engine="sglang", model_type="deepseek_v3_2")
        assert entry is not None
        cmd = generate_sglang_command("deepseek-ai/DeepSeek-V3.2", profile, 8, entry)
        # required flag: --attention-backend nsa
        assert "--attention-backend nsa" in cmd
