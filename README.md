# llm-cal

[![CI](https://github.com/FlyTOmeLight/llm-cal/actions/workflows/ci.yml/badge.svg)](https://github.com/FlyTOmeLight/llm-cal/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-cal.svg)](https://pypi.org/project/llm-cal/)
[![Docs](https://img.shields.io/badge/docs-flytomelight.github.io-blue)](https://flytomelight.github.io/llm-cal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**LLM inference hardware calculator** — architecture-aware, engine-version-aware, honest-labeled.

English · [中文](README.zh.md) · [Docs](https://flytomelight.github.io/llm-cal/) · [中文文档](https://flytomelight.github.io/llm-cal/zh/)

Give it a HuggingFace / ModelScope model id and a GPU, get back:

- **real weight size** (summed from `safetensors` API, not `params × precision`)
- **architecture profile** — MHA / GQA / MQA / MLA / NSA / CSA+HCA, MoE active-expert ratio, sliding window, tied embeddings
- **KV cache per request** at multiple context lengths, with TP-aware sharding
- **fleet size** — `min` / `dev` / `prod` tiers that respect `num_heads` TP divisibility
- **prefill latency + decode throughput** with named coefficients and citations
- **K/L concurrency bounds** with bottleneck classification (memory vs compute vs bandwidth)
- **engine compatibility** from a curated matrix (vLLM + SGLang × 16 model families × 32 entries)
- a **ready-to-paste** `vllm serve` or `sglang launch_server` command

Every number in the output carries a provenance label. `--explain` prints the full derivation trace. `--llm-review` (opt-in) sends the trace to any OpenAI-compatible endpoint for a second opinion.

---

## Why another calculator?

Existing tools (`gpu_poor`, `llm-vram-calculator`, APXML, SelfHostLLM, ...) compute weight size with `params × precision`. That silently fails on mixed-precision quantization:

| Model | `gpu_poor` | Real `safetensors` | **llm-cal** |
|---|---:|---:|---:|
| DeepSeek-V4-Flash (FP4+FP8 pack) | 284 GB (FP8 assumed) | **160 GB** | **160 GB** ✓ |
| DeepSeek-V3 (pure FP8) | 685 GB | **688 GB** | **688 GB** ✓ |
| Qwen2.5-72B (FP16) | 140 GB | **145 GB** | **145 GB** ✓ |

llm-cal reads real bytes from the HF API, reconciles against every known quantization scheme, picks the best match, and **surfaces ties** when multiple schemes share the same bits/param:

```
Quantization reconciliation
  FP4_FP8_MIXED    160.01 GB   0.2%  ← wins (tied with GPTQ_INT4, AWQ_INT4
  GPTQ_INT4        160.01 GB   0.2%    at bpp=0.55 — need per-tensor dtype
  AWQ_INT4         160.01 GB   0.2%    to distinguish, deferred to v0.2)
  FP8              290.94 GB  45.1%  ← the gpu_poor trap
```

This tie was caught by `--llm-review` running MiniMax-M2 against the tool's own output during dogfood testing. First real bug from LLM review, fixed in v0.1.0.

---

## The honesty principle — 7 labels

Every number carries one of these:

| Label | Meaning | Example |
|---|---|---|
| `[verified]` | Direct read from API or file | `safetensors bytes: 159.62 GB` |
| `[inferred]` | One-step derivation from verified data | `bits/param: 4.39` (bytes ÷ params) |
| `[estimated]` | Formula-based, coefficient from source | `prefill latency: 735 ms` |
| `[cited]` | From a paper / PR / release note | `vLLM ≥0.19.0 supports CSA+HCA` |
| `[unverified]` | Matrix entry without evidence, flagged | `SGLang day-0 support pending` |
| `[unknown]` | Graceful degrade — unknown model type | New `model_type` not in registry |
| `[llm-opinion]` | **Opt-in** LLM audit, never overrides the 6 above | `--llm-review` output only |

The first 6 labels are deterministic. `[llm-opinion]` is explicitly tagged as non-authoritative.

---

## Install

Python 3.11+.

```bash
# pipx (cleanest)
pipx install git+https://github.com/FlyTOmeLight/llm-cal.git@v0.1.0

# uv
uv tool install git+https://github.com/FlyTOmeLight/llm-cal.git@v0.1.0

# pip
pip install git+https://github.com/FlyTOmeLight/llm-cal.git@v0.1.0
```

Gated models (Llama, Gemma):

```bash
export HF_TOKEN=hf_...
```

Mainland China HF mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Quickstart

```bash
# Basic evaluation
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800

# Chinese output + longer context
llm-cal Qwen/Qwen2.5-72B-Instruct --gpu A100-80G --context-length 32768 --lang zh

# Full derivation trace (every formula + input + step + source)
llm-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain

# LLM audit of the derivation (opt-in, needs env vars)
export LLM_CAL_REVIEWER_API_KEY=sk-...
export LLM_CAL_REVIEWER_BASE_URL=https://api.deepseek.com/v1
export LLM_CAL_REVIEWER_MODEL=deepseek-chat
llm-cal deepseek-ai/DeepSeek-V3 --gpu H800 --explain --llm-review

# All 53 supported GPUs
llm-cal --list-gpus

# Run the curated benchmark (8 models × 33 checks vs reference truth)
llm-cal --benchmark
```

Abbreviated output:

```
┌─ deepseek-ai/DeepSeek-V4-Flash  via huggingface @ 6c858e7 ─┐

Architecture
  model_type         deepseek_v4                             [verified]
  attention          CSA_HCA (heads=64, kv_heads=1, hd=512)  [verified]
  moe                256 routed + 1 shared, top-6            [verified]
  sliding_window     128                                     [verified]

Weights
  safetensors bytes  159.62 GB      [verified]
  quantization       FP4_FP8_MIXED  [inferred]  (tied with GPTQ_INT4, AWQ_INT4)

Fleet — H800
  tier       GPUs    concurrent @ 128K    concurrent @ 1.0M
  min          4           ~14                  ~1
  dev ★        4           ~14                  ~1
  prod         8           ~23                  ~2

Performance — dev tier (4× H800)
  prefill latency   735 ms @ 2000 input tokens     [estimated, Kaplan 2020]
  decode throughput 48 tok/s per user              [estimated, Kwon SOSP 2023]
  bottleneck        memory bandwidth               [inferred]

Generated command
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 4 --max-model-len 1048576 \
    --trust-remote-code --gpu-memory-utilization 0.9 \
    --attention-backend auto
```

---

## CLI reference

```
llm-cal [MODEL_ID] [OPTIONS]

Core:
  --gpu TEXT                     GPU id (see --list-gpus). Aliases accepted, case-insensitive.
  --engine [vllm|sglang]         Default: vllm
  --gpu-count INT                Force fleet size (skips min/dev/prod auto-pick)
  --context-length INT           Context length for KV cache estimation
  --lang [en|zh]                 Output language (default: auto-detect from LANG)

Performance tuning (all have honest defaults — see docs/methodology.md):
  --input-tokens INT             Prefill input budget. Default: 2000
  --output-tokens INT            Decode output budget. Default: 512
  --target-tokens-per-sec FLOAT  SLA for per-user decode. Default: 30
  --prefill-util FLOAT           Compute utilization factor. Default: 0.40
  --decode-bw-util FLOAT         Memory-BW utilization factor. Default: 0.50
  --concurrency-degradation FLOAT  High-load efficiency loss. Default: 1.0 (honest baseline)

Introspection:
  --explain                      Print full derivation trace for every non-trivial number
  --llm-review                   Send derivation to LLM for second opinion (opt-in)
                                 Requires: LLM_CAL_REVIEWER_API_KEY / _BASE_URL / _MODEL

Meta:
  --list-gpus                    List all 53 supported GPUs and exit
  --benchmark                    Run the curated dataset (8 models × 33 checks)
  --refresh                      Bypass cache, re-fetch from HF/ModelScope
```

---

## Supported hardware (53 GPUs)

| Vendor | Models |
|---|---|
| **NVIDIA** | B200, GB200, H100, H800, H200, H20, GH200, L40S, L40, L4, RTX6000-Ada, RTX4090, A100-80G/40G, A40, A10, A10G, V100-SXM2/PCIe-32G, T4 |
| **AMD** | MI325X, MI300X, MI250X, MI210 |
| **Intel Habana** | Gaudi3, Gaudi2 |
| **华为昇腾** | 910A, 910B1, 910B2, 910B3, 910B4, 910C, Atlas-300I-Duo |
| **沐曦** | MXC500, MXC550 |
| **昆仑芯** | Kunlun-P800, Kunlun-R200 |
| **壁仞** | BR100, BR104 |
| **天数智芯** | BI-V100 |
| **摩尔线程** | MTT-S4000, MTT-S3000, MR-V100 |
| **寒武纪** | MLU370-X8, MLU590 |
| **海光** | K100-AI, Z100 |

Each entry carries `spec_source` (vendor page, datasheet, or verified benchmark URL) and bilingual notes.

Full details: `llm-cal --list-gpus`. Missing one? PR `src/llm_cal/hardware/gpu_database.yaml` — data-only change, no code.

---

## Engine × architecture matrix (32 entries / 16 families)

Covers vLLM 0.6–0.19 and SGLang 0.4–0.5:

- Dense: `llama`, `mistral`, `qwen2`, `qwen3`, `phi`, `gemma`, `internlm`
- MoE: `mixtral`, `qwen3_moe`, `deepseek_v3`, `deepseek_v3_2`, `deepseek_v4`, `phi_moe`
- Sparse attention: `deepseek_v3_2` (NSA), `deepseek_v4` (CSA+HCA)
- Sliding window: `mistral`, `qwen3_moe`

Every matrix entry carries `verification_level` (`verified` / `cited` / `unverified`) and `sources[]` with URL + `captured_date`. v0.1 has no `verified` entries — the author has no test hardware. Community `tested` contributions welcome.

Full matrix: [`src/llm_cal/engine_compat/matrix.yaml`](src/llm_cal/engine_compat/matrix.yaml).

---

## Benchmark (8 models × 33 checks)

`llm-cal --benchmark` runs the curated dataset and compares tool output against reference truth (HF API sizes, model card claims, vLLM recipes).

| Model | Ref weight | llm-cal | Quant | Status |
|---|---:|---:|---|:-:|
| `deepseek-ai/DeepSeek-V4-Flash` | 160 GB | 159.62 GB | FP4_FP8_MIXED | ✓ |
| `deepseek-ai/DeepSeek-V3` | 688 GB | 688.59 GB | FP8 | ✓ |
| `deepseek-ai/DeepSeek-V3.2` | 688 GB | 687.84 GB | FP8 (NSA) | ✓ |
| `Qwen/Qwen2.5-72B-Instruct` | 145 GB | 145.41 GB | FP16 | ✓ |
| `Qwen/Qwen3-30B-A3B` | 61 GB | 60.82 GB | FP16 (MoE) | ✓ |
| `Qwen/Qwen2.5-7B` | 14.2 GB | 14.19 GB | FP16 | ✓ |
| `mistralai/Mixtral-8x7B-v0.1` | 93 GB | 93.41 GB | FP16 (MoE) | ✓ |
| `microsoft/Phi-4` | 28 GB | 28.17 GB | FP16 | ✓ |

Exit code 0 on all-pass, 1 on any FAIL. Runnable in CI.

---

## Methodology

Every formula and coefficient has a primary source. No magic numbers.

- **Prefill FLOPs**: `2 × params × input_tokens` (Kaplan et al. 2020, *Scaling Laws for Neural Language Models*)
- **Decode throughput**: `bandwidth × util / weight_bytes` (Kwon et al. SOSP 2023, *Efficient Memory Management for LLM Serving with PagedAttention*)
- **KV cache layout**: matches vLLM `PagedAttention` and SGLang `RadixAttention` source behavior
- **TP sharding**: `per_gpu_KV = total_KV / min(tp_size, num_kv_heads)` — empirically verified against vLLM runtime
- **Utilization coefficients**: `prefill_util=0.40`, `decode_bw_util=0.50`, `concurrency_degradation=1.0` (honest defaults; override per-workload via CLI flags)

Full writeup with citations: [`docs/methodology.md`](docs/methodology.md) · [中文](docs/zh/methodology.md).

---

## Documentation

- [Homepage (English)](https://flytomelight.github.io/llm-cal/)
- [Homepage (中文)](https://flytomelight.github.io/llm-cal/zh/)
- [Architecture guide](https://flytomelight.github.io/llm-cal/architecture-guide/) — 10-step checklist for adding a new model type
- [Methodology](https://flytomelight.github.io/llm-cal/methodology/) — every formula with source
- [Contributing](CONTRIBUTING.md)

---

## Scope of v0.1

**Shipped:**

- HuggingFace + ModelScope as model sources, real bytes from `safetensors` metadata
- Architecture detection: Dense / MoE / GQA / MQA / MLA / NSA / CSA+HCA / Sliding Window
- KV cache with traits composition, TP-aware sharding
- Fleet planner (min/dev/prod, TP divisibility)
- Prefill / decode performance estimator
- K/L concurrency bounds + bottleneck classification
- Engine compat matrix (vLLM + SGLang, 32 entries)
- Command generator (vLLM + SGLang with required flags)
- Bilingual output (en / zh) with label localization
- `--explain` derivation trace
- `--llm-review` opt-in LLM audit (any OpenAI-compatible endpoint)
- `--benchmark` curated regression suite
- `--list-gpus` discovery
- 53-GPU database with `spec_source` traceability

**v0.2 roadmap:**

- Per-tensor dtype read from `safetensors` metadata (distinguishes FP4/GPTQ/AWQ tie)
- Lazy matrix loading when entries > 100
- Ollama / GGUF support
- Multimodal models (Qwen-VL, InternVL)
- LoRA / adapter VRAM math
- `--offline` mode for air-gapped environments
- Community-contributed `verified` matrix entries (requires real hardware runs)

---

## Contributing

Especially welcome:

1. **New GPUs** — `src/llm_cal/hardware/gpu_database.yaml` (data only, no code)
2. **New engine entries** — `src/llm_cal/engine_compat/matrix.yaml` with `sources[]`
3. **New model architectures** — [10-step checklist](docs/architecture-guide.md)
4. **`verified` matrix entries** — if you have real hardware and can run a config, send us the tested result

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
