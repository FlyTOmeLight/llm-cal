---
title: llm-cal
emoji: 🧮
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: LLM inference sizing — honest, architecture-aware
---

# llm-cal — LLM inference hardware calculator

Web UI for [`llm-cal`](https://github.com/FlyTOmeLight/llm-cal). Pick a model, pick a GPU, get a hardware plan.

Architecture-aware (MLA, NSA, CSA+HCA, MoE, sliding window). Engine-aware (vLLM, SGLang). Honest-labeled — every number carries a provenance tag (`[verified]` / `[inferred]` / `[estimated]` / `[cited]` / `[unverified]` / `[unknown]`).

## The story this Space exists to tell

`gpu_poor` reports DeepSeek-V4-Flash as 284 GB by assuming pure FP8. The real safetensors weight is 160 GB — it ships an FP4+FP8 mixed pack. `llm-cal` reads the actual on-disk dtype (per-tensor metadata + MX block-scaled scale tensors) and gets 160.01 GB at **0.2% error**.

That's the whole pitch.

## Local

```bash
pip install llm-cal gradio
python app.py
```

## Links

- [GitHub repo](https://github.com/FlyTOmeLight/llm-cal)
- [Full docs](https://flytomelight.github.io/llm-cal/)
- [Methodology](https://flytomelight.github.io/llm-cal/methodology/) — every formula's primary source
- [Pre-rendered model pages](https://flytomelight.github.io/llm-cal/models/) — popular model × GPU combos
