# llm-cal

**LLM inference hardware calculator** — architecture-aware, engine-version-aware, honest-labeled.

Give it a HuggingFace / ModelScope model id and a GPU type, get back:

- real weight size (read from `safetensors` metadata, not guessed)
- architecture profile: MLA, NSA, CSA+HCA, MoE, sliding window — each a first-class trait
- KV cache per request at multiple context lengths
- recommended fleet size: `min` / `dev` / `prod` with TP-aware KV sharding
- engine compatibility from a curated matrix (vLLM & SGLang × 16 architecture families)
- a ready-to-paste `vllm serve` or `sglang launch_server` command

Output is **bilingual** — English and 中文.

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

## Install

Requires Python 3.11+.

=== "pipx (recommended)"

    ```bash
    pipx install git+https://github.com/FlyTOmeLight/llm-cal.git
    ```

=== "uv"

    ```bash
    uv tool install git+https://github.com/FlyTOmeLight/llm-cal.git
    ```

=== "pip"

    ```bash
    pip install git+https://github.com/FlyTOmeLight/llm-cal.git
    ```

Auth (for gated models like Llama, Gemma):

```bash
export HF_TOKEN=hf_...
```

Chinese mirror (if HF is slow):

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Quickstart

```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm
```

See [Quickstart](quickstart.md) for full walkthrough, [Architecture Guide](architecture-guide.md) for how the tool works, and [Contributing](contributing.md) for how to add models, GPUs, or engine support.

## Validation

Run the built-in benchmark against curated reference data:

```bash
llm-cal --benchmark
```

Current result: **33/33 PASS** across 8 reference models, 6 check types. Every expected value in the dataset cites its source (HF API / model card / vLLM recipe / hand computation). See the [benchmark section of the contributing guide](contributing.md).

## Supported

- **47 GPUs** across NVIDIA / AMD / Intel Habana / Huawei Ascend / Cambricon / Moore Threads / MetaX / KunlunXin / Biren / Iluvatar / Hygon
- **16 architecture families** in the engine compatibility matrix
- **2 inference engines**: vLLM and SGLang
- **2 output languages**: English and 中文

Run `llm-cal --list-gpus` to see the full GPU table with aliases.

## License

Apache-2.0. See [LICENSE](https://github.com/FlyTOmeLight/llm-cal/blob/main/LICENSE).
