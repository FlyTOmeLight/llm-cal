# llm-cal

**LLM inference hardware calculator** — architecture-aware, engine-version-aware, honest-labeled.

Give it a HuggingFace / ModelScope model id and a GPU type, get back:
- real weight size (read from `safetensors` metadata, not guessed)
- architecture profile (MLA, NSA, CSA+HCA, MoE, sliding window — each treated as a first-class trait)
- KV cache per request at multiple context lengths
- recommended fleet size: `min` / `dev` / `prod` tiers with TP-aware KV sharding
- engine compatibility from a curated matrix (vLLM & SGLang × 7 architecture families)
- a ready-to-paste `vllm serve` or `sglang launch_server` command

Output is **bilingual** — English and 中文.

---

## Why another calculator?

Existing tools (`gpu_poor`, `llm-vram-calculator`, APXML, SelfHostLLM, ...) all compute weight size using `params × precision`. That silently fails on new architectures:

| Model | `gpu_poor` says | Real `safetensors` | llm-cal |
|---|---|---|---|
| DeepSeek-V4-Flash (FP4+FP8 pack) | 284 GB (FP8 assumption) | **160 GB** | **160 GB** ✓ |
| Standard FP8 models | correct | correct | correct ✓ |

llm-cal reads the real file sizes from the HuggingFace API, then compares against every known quantization scheme — the best match wins. The DeepSeek-V4 story becomes explicit:

```
Quantization reconciliation (observed vs predicted per scheme)
  scheme           predicted bytes    delta         error %
  FP4_FP8_MIXED        160.01 GB     397 MB under   0.2%  ← wins
  FP8                  290.94 GB     131 GB under   45.1% ← the gpu_poor trap
```

And every number has a tag telling you where it came from:

- `[verified]` — read directly from HF API / config.json
- `[inferred]` — derived from `[verified]` in a single step
- `[estimated]` — computed by a formula (KV cache, weight split)
- `[cited]` — from release notes / PR / announcement
- `[unverified]` — matrix entry without evidence (explicitly flagged)
- `[unknown]` — failed to recognize, graceful degrade

---

## Install

Requires Python 3.11+.

```bash
# pipx (cleanest)
pipx install git+https://github.com/FlyTOmeLight/llm-cal.git

# or uv
uv tool install git+https://github.com/FlyTOmeLight/llm-cal.git

# or pip
pip install git+https://github.com/FlyTOmeLight/llm-cal.git
```

Auth (for gated models like Llama, Gemma):
```bash
export HF_TOKEN=hf_...
```

Chinese mirror (if HF is slow in your region):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Quickstart

```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm
```

Abbreviated output (real terminal is color-tagged):

```
┌─ deepseek-ai/DeepSeek-V4-Flash  via huggingface @ 6c858e7 ─┐

Architecture
  model_type         deepseek_v4                             [verified]
  family             transformer                             [verified]
  layers             43                                      [verified]
  attention          CSA_HCA (heads=64, kv_heads=1, hd=512)  [verified]
  compress_ratios    len=44, dense_layers=3                  [verified]
  moe                256 routed + 1 shared, top-6            [verified]
  sliding_window     128                                     [verified]

Weights
  safetensors bytes     159.62 GB      [verified]
  total params          290.94B        [estimated]
  bits/param            4.39           [inferred]
  quantization guess    FP4_FP8_MIXED  [inferred]

KV cache per request (BF16)
  4,096 tokens    68.91 MB      [estimated]
  131,072         2.21 GB       [estimated]
  1,048,576       17.64 GB      [estimated]

Engine compatibility — vllm
  version           >=0.19.0                       [cited]
  optional flags    --attention-backend auto
  sources           release notes + day-0 announcement 2026-04-23

Target hardware — H800
  memory              80 GB HBM
  NVLink bandwidth    400 GB/s (half of H100 — China-regulated variant)

Recommended fleet — H800
  tier       GPUs    weight/GPU    headroom/GPU    concurrent @ 128K    concurrent @ 1.0M
  min          4      39.90 GB      32.10 GB               ~14                  ~1
  dev ★        4      39.90 GB      32.10 GB               ~14                  ~1
  prod         8      19.95 GB      52.05 GB               ~23                  ~2
  constraint: TP must divide num_heads=64. Candidates within one node: [1, 2, 4, 8].
  ★ = recommended

┌─ Generated command — tier: dev (4 GPUs) ─┐
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 4 \
    --max-model-len 1048576 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --attention-backend auto
```

**中文输出**：加 `--lang zh`，或设 `LANG=zh_CN.UTF-8` 自动识别。

---

## CLI reference

```
llm-cal MODEL_ID [OPTIONS]

Required:
  --gpu TEXT                GPU type: H100 / H800 / H200 / A100-80G / A100-40G /
                            B200 / 910B / RTX4090 (case-insensitive, aliases accepted)

Optional:
  --engine [vllm|sglang]    Default: vllm
  --gpu-count INT           Force a specific fleet size (skips min/dev/prod tiers)
  --context-length INT      Override default context for KV cache estimation
  --refresh                 Bypass cache and re-fetch from HF/ModelScope
  --lang [en|zh]            Output language (default: auto from LANG env)
```

---

## Supported GPUs

| ID | Aliases | HBM | NVLink | FP16 TFLOPS | FP8 | FP4 |
|---|---|---:|---:|---:|:-:|:-:|
| H100 | H100-SXM5, H100-80G | 80 GB | 900 GB/s | 989 | ✓ | — |
| H800 | H800-SXM5, H800-80G | 80 GB | 400 GB/s | 989 | ✓ | — |
| H200 | H200-SXM, H200-141G | 141 GB | 900 GB/s | 989 | ✓ | — |
| A100-80G | A100-80 | 80 GB | 600 GB/s | 312 | — | — |
| A100-40G | A100-40 | 40 GB | 600 GB/s | 312 | — | — |
| B200 | B200-192G | 192 GB | 1800 GB/s | 2250 | ✓ | **✓** |
| 910B | Ascend-910B, 910B2 | 64 GB | 400 GB/s (HCCS) | 376 | — | — |
| RTX4090 | 4090 | 24 GB | 0 (PCIe) | 165 | ✓ | — |

Missing a GPU? Open a PR against `src/llm_cal/hardware/gpu_database.yaml` — no code changes needed.

---

## Supported engines + architectures (matrix)

Current engine compatibility matrix (v0.1) covers 7 model-type families across vLLM 0.6–0.19 and SGLang 0.4–0.5:

| Model family | vLLM | SGLang |
|---|:-:|:-:|
| `llama` | ✓ (≥0.6.0) | ✓ (≥0.4.0) |
| `mistral` | ✓ (≥0.6.0) | — |
| `mixtral` | ✓ (≥0.6.0) | ✓ (≥0.4.0) |
| `qwen3` / `qwen3_moe` | ✓ (≥0.7.0) | ✓ (≥0.4.0) |
| `deepseek_v3` | ✓ (≥0.7.0) | ✓ (≥0.4.0) |
| `deepseek_v3_2` | ✓ (≥0.18.0, needs `--attention-backend nsa`) | ✓ (≥0.5.0) |
| `deepseek_v4` | ✓ (≥0.19.0, day-0 2026-04-23) | ⚠ unverified |

Every matrix entry carries a `verification_level` and a `sources[]` array. v0.1 has **no `verified` entries** — the author has no test hardware. Community-contributed `tested` entries are planned for v0.2.

---

## Benchmark: 4 reference models

All numbers are from real tool runs. Timestamps reflect the underlying HF commit.

| Model | GPU | Weight ([verified]) | Quant ([inferred]) | Recommended (dev) | concurrent @ 128K |
|---|---|---:|---|:-:|:-:|
| `deepseek-ai/DeepSeek-V4-Flash` | H800 | 159.62 GB | FP4_FP8_MIXED | 4 GPUs | ~14 |
| `deepseek-ai/DeepSeek-V3` | H800 | 688.59 GB | FP8 | 8 GPUs (tight) | (overflow — needs >8) |
| `Qwen/Qwen2.5-72B-Instruct` | H100 | 145.41 GB | FP16 | 8 GPUs | ~40* (GQA TP-shard) |
| `mistralai/Mixtral-8x7B-v0.1` | H100 | 93.41 GB | FP16 | 4 GPUs | ~45 |

*Qwen2.5-72B uses GQA (`kv_heads=8`). At TP=8, per-GPU KV is 1/8 of total — llm-cal applies this sharding automatically. Tools that assume KV replication (e.g. SelfHostLLM) would report ~5 concurrent here.

---

## Label discipline

Every number in the output has a provenance tag. This is the tool's value proposition — users know exactly what is measured vs computed.

| Tag | Meaning | Example |
|---|---|---|
| `[verified]` | Direct read from API or file | `safetensors bytes: 159.62 GB` (sum of HF siblings) |
| `[inferred]` | One-step derivation from verified data | `bits/param: 4.39` (bytes ÷ params) |
| `[estimated]` | Formula-based computation | `KV cache @ 128K: 2.21 GB` |
| `[cited]` | External source (release note / PR) | `vLLM ≥0.19.0 supports CSA+HCA` |
| `[unverified]` | Matrix entry without evidence — flagged | `SGLang day-0 support pending` |
| `[unknown]` | Couldn't identify, graceful degrade | New model type not in registry |

See [`docs/architecture-guide.md`](docs/architecture-guide.md) for how the tool handles each.

---

## Scope of v0.1

**In scope:**
- HuggingFace as the primary source (real bytes from `model_info().siblings`)
- ModelScope support (pending ADR-001 — SDK vs REST decision)
- Architecture detection: Dense / MoE / GQA / MQA / MLA / NSA / CSA+HCA / Sliding Window
- KV cache formulas with traits composition (DeepSeek V4 error < 1% vs hand math)
- TP-aware KV sharding (per vLLM/SGLang behavior)
- Fleet planner: min/dev/prod tiers, TP divisibility constraint
- Command generator: vLLM + SGLang, pulls required flags from compat matrix

**Out of scope (deferred):**
- Ollama / GGUF — v0.2
- Multimodal (Qwen-VL, InternVL) — v0.2
- LoRA / adapter VRAM math — v0.2
- Mamba / state-space models — v0.3+
- Diagnostic mode (audit existing `vllm serve` command) — v0.3+
- `--offline` mode for air-gapped environments — v0.2
- Real-hardware-measured matrix entries — v0.2+ community PR channel

---

## Contributing

We especially welcome:
1. **New GPUs** in `src/llm_cal/hardware/gpu_database.yaml` (data-only change)
2. **New engine compat entries** in `src/llm_cal/engine_compat/matrix.yaml` with sources
3. **New model architectures** — see [`docs/architecture-guide.md`](docs/architecture-guide.md) for the 10-step checklist
4. **`verified` matrix entries** — if you have hardware and can actually run a config, PR the tested result

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
