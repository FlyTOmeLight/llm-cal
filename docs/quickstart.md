# Quickstart

## The canonical run

```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm
```

This is the tool's reference case. You get back:

1. **Architecture profile** — DeepSeek-V4 detected, CSA+HCA + MoE + sliding window, `confidence: high`.
2. **Weights** — `safetensors bytes: 159.62 GB [verified]`, `quantization guess: FP4_FP8_MIXED [inferred]`.
3. **Reconciliation** — predicted bytes under each quantization scheme. FP4_FP8_MIXED wins at 0.2% error; FP8 is off by 45.1%.
4. **KV cache** — 4K / 32K / 128K / 1M context length estimates.
5. **Engine compatibility** — vLLM ≥0.19.0, `[cited]` with source URLs.
6. **Target hardware** — H800 spec with bilingual notes.
7. **Recommended fleet** — min / dev / prod tiers with TP-aware KV sharding.
8. **Generated command** — paste-ready `vllm serve ...`.

## Common flags

```bash
# Chinese output
llm-cal <model> --gpu H800 --engine vllm --lang zh

# Force a specific GPU count (skip min/dev/prod recommendation)
llm-cal <model> --gpu H100 --gpu-count 4

# Override context length for KV cache math
llm-cal <model> --gpu H800 --context-length 65536

# Bypass cache (useful after a model repo update)
llm-cal <model> --gpu H800 --refresh

# See all supported GPUs
llm-cal --list-gpus

# Validate tool output against curated reference values
llm-cal --benchmark
```

## Non-NVIDIA examples

```bash
# AMD flagship (256 GB HBM3E, the largest single-card memory)
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu MI325X --engine vllm

# Huawei Ascend 910B4 (inference variant, 32 GB)
llm-cal Qwen/Qwen2.5-7B-Instruct --gpu 910B4 --engine vllm

# Chinese accelerators by alias
llm-cal Qwen/Qwen2.5-14B-Instruct --gpu 曦云C500 --engine vllm
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu 昆仑芯P800 --engine vllm
llm-cal Qwen/Qwen2.5-7B-Instruct --gpu 摩尔线程S4000 --engine vllm
```

## Output labels explained

Every number in the report is tagged:

| Tag | Meaning | Example |
|---|---|---|
| `[verified]` | Direct read from API or file | `safetensors bytes: 159.62 GB` (HF siblings API) |
| `[inferred]` | One-step derivation from verified data | `bits/param: 4.39` (bytes ÷ params) |
| `[estimated]` | Formula-based computation | `KV cache @ 128K: 2.21 GB` |
| `[cited]` | External source (release note / PR) | `vLLM ≥0.19.0 supports CSA+HCA` |
| `[unverified]` | Matrix entry without evidence — flagged | `SGLang day-0 support pending` |
| `[unknown]` | Couldn't identify, graceful degrade | New model type not in registry |

**Do NOT trust any tool that gives you a single number without provenance.** That's the tool's value prop.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success, or `--benchmark` all PASS |
| 1 | `--benchmark` has failures |
| 2 | Authentication required (gated model without `HF_TOKEN`) |
| 3 | Model not found |
| 4 | Source unavailable (network / rate limit / 5xx) |

These make the tool scriptable in CI.

## Troubleshooting

### "需要认证 / Authentication required"

Set `HF_TOKEN`:

```bash
export HF_TOKEN=hf_xxxxx
```

Or, on first run, `huggingface-cli login`.

### Slow HF API in China

Set the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Tool says `model_type not in v0.1 matrix`

The engine compatibility matrix doesn't yet have an entry for this model family. The rest of the report (weights, KV cache, fleet recommendation) still works — just the engine section shows "no match".

Consider contributing a matrix entry via PR — see [Contributing](contributing.md).

### Tool reports `[unknown]` architecture

A brand-new model type the detector doesn't recognize yet. Tool falls back to:

- `[verified]` safetensors bytes (still reliable)
- **No** KV cache estimate
- **No** engine compatibility info
- Conservative fleet recommendation based on weight fit only

This is by design. See the "Graceful unknown" section in the [Architecture Guide](architecture-guide.md) for the fallback tree.
