# llm-cal

LLM inference hardware calculator — **architecture-aware**, **engine-version-aware**, **honest-labeled**.

> Status: v0.1 pre-alpha. Scaffolding and core modules under active development.
> Design doc: `~/.gstack/projects/moonlight/moonlight-no-repo-design-20260424-152100.md`

## Why

You have a new model (DeepSeek-V4-Flash, Qwen-3-MoE-235B, ...). You need to answer:

- How much VRAM does it need?
- How many H800 / A100 / H100 do I need?
- Which vLLM / SGLang version supports it?
- What's the launch command?

Existing calculators use `参数量 × 精度` formulas that silently fail on new architectures
(MLA, NSA, CSA+HCA, FP4+FP8 mixed quantization). `llm-cal` reads the model file directly
and labels every number by where it came from.

## Core principles

1. **Honest labels**. Every number is tagged: `[verified]` / `[inferred]` / `[estimated]`
   / `[cited]` / `[unverified]` / `[unknown]`. No guessing dressed up as fact.
2. **Architecture-aware**. MLA, NSA, CSA+HCA, sliding window, MoE — each is a first-class
   trait, not a formula patch.
3. **Engine-version-aware**. A persistent matrix tracks which vLLM/SGLang versions support
   which model architectures, with source citations.
4. **Works on day-0**. Unknown architectures degrade gracefully: safetensors file size is
   still `[verified]`, even when KV cache estimation isn't possible.

## Install (once v0.1 is published)

```bash
pip install llm-cal
```

## Usage (target UX)

```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm
```

Outputs:
- Model architecture profile (`[verified]` from `config.json`)
- Real weight bytes (`[verified]` from HF `model_info().siblings`)
- Inferred quantization scheme (`[inferred]` from bits/param)
- Fleet size recommendations (min / dev / prod)
- KV cache per request at common context lengths (`[estimated]`)
- Engine version & required flags (`[cited]`)
- Ready-to-copy `vllm serve` / `sglang launch_server` command

## What's out of scope for v0.1

- Ollama / GGUF (planned for v0.2)
- Multimodal, LoRA, CPU offload (v0.2+)
- Diagnostic mode (paste existing command, audit it — v0.3+)
- Mamba / State Space Models (v0.3+)
- Real-hardware-measured entries (community PR channel in v0.2+)

## Development

```bash
# Install dev deps
pip install -e ".[dev]"

# Run tests
pytest

# Lint + typecheck
ruff check src tests
mypy src
```

## License

Apache-2.0. See [LICENSE](LICENSE).
