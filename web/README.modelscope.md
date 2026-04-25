---
domain:
  - nlp
tags:
  - llm
  - inference
  - gpu
  - sizing
  - calculator
license: Apache License 2.0
language:
  - zh
  - en
deployspec:
  entry_file: app.py
  server_port: 7860
---

# llm-cal · 大模型推理硬件计算器

[GitHub](https://github.com/FlyTOmeLight/llm-cal) · [完整文档](https://flytomelight.github.io/llm-cal/) · `pip install llm-cal`

选模型，选 GPU，得到一份诚实标注的硬件配置方案。

## 跟 gpu_poor 的差别

`gpu_poor` 把 DeepSeek-V4-Flash 报成 284 GB——它假设全 FP8。模型实际是 FP4+FP8 混合包，safetensors 真实字节是 160 GB。`llm-cal` 直接读 safetensors header 的 per-tensor dtype + MX block-scaled scale tensor，得到 160.01 GB，**误差 0.2%**。

这就是核心卖点。

## 架构感知 · 引擎感知 · 诚实标注

每个数字都带 provenance label：
- `[verified]` 实测过的
- `[inferred]` 从 config 推出来的
- `[estimated]` 用经验公式算的
- `[cited]` 引用 PR / release note 的
- `[unverified]` 不确定的
- `[unknown]` 不知道的

不会拿"估算"冒充"实测"。

## 国内用户重点

- **支持 ModelScope 来源**：右上角切到 ModelScope，输入 `Qwen/Qwen3-30B-A3B` 这类 MS-原生 ID 即可。下载比 HF 快十倍
- **覆盖 11 家厂商**：NVIDIA / AMD / 华为昇腾 / 沐曦 / 昆仑芯 / 壁仞 / 天数智芯 / 摩尔线程 / 寒武纪 / 海光 / Intel Habana
- **中英文界面共存**：标签全部双语

## 本地用法

```bash
pip install llm-cal
llm-cal Qwen/Qwen3-30B-A3B --gpu A100-80G --source modelscope
```

或者直接在这个 Studio 里点 `Calculate · 计算`。
