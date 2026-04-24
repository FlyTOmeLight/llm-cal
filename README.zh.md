# llm-cal

[![CI](https://github.com/FlyTOmeLight/llm-cal/actions/workflows/ci.yml/badge.svg)](https://github.com/FlyTOmeLight/llm-cal/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-cal.svg)](https://pypi.org/project/llm-cal/)
[![Docs](https://img.shields.io/badge/docs-flytomelight.github.io-blue)](https://flytomelight.github.io/llm-cal/zh/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**大模型推理硬件计算器** —— 架构感知、引擎版本感知、诚实标签。

[English](README.md) · 中文 · [Docs](https://flytomelight.github.io/llm-cal/) · [中文文档](https://flytomelight.github.io/llm-cal/zh/)

给它一个 HuggingFace / ModelScope 模型 ID 和一块 GPU，它返回：

- **真实权重大小**（从 `safetensors` API 汇总，不是 `参数量 × 精度`）
- **架构画像** —— MHA / GQA / MQA / MLA / NSA / CSA+HCA、MoE 活跃专家比、滑动窗口、tied embeddings
- **每请求 KV 缓存**（多个 context 长度下的值，TP 感知分片）
- **集群规模** —— `min` / `dev` / `prod` 三档，遵守 `num_heads` 的 TP 整除约束
- **Prefill 延迟 + Decode 吞吐**，每个系数都有命名和引用来源
- **K/L 并发上界** 带 bottleneck 分类（显存 / 算力 / 带宽）
- **引擎兼容性** 来自一份 curated matrix（vLLM + SGLang × 16 模型族 × 32 条目）
- **可直接复制粘贴** 的 `vllm serve` 或 `sglang launch_server` 命令

每个数字都带来源标签。`--explain` 打印完整推导链。`--llm-review`（可选）把推导链发给任意 OpenAI 兼容端点做第二意见审计。

---

## 为什么又造一个计算器

现有工具（`gpu_poor`、`llm-vram-calculator`、APXML、SelfHostLLM 等）都用 `参数量 × 精度` 估权重。在混合精度量化面前会静默翻车：

| 模型 | `gpu_poor` | 真实 `safetensors` | **llm-cal** |
|---|---:|---:|---:|
| DeepSeek-V4-Flash（FP4+FP8 pack） | 284 GB（当成 FP8） | **160 GB** | **160 GB** ✓ |
| DeepSeek-V3（纯 FP8） | 685 GB | **688 GB** | **688 GB** ✓ |
| Qwen2.5-72B（FP16） | 140 GB | **145 GB** | **145 GB** ✓ |

llm-cal 从 HF API 读真实 bytes，逐一对比所有已知量化方案，选拟合最好的。当多个方案在相同 bits/param 处打平时，**工具会显式提示 tie**：

```
量化反演（实际观测 vs 各方案预测）
  FP4_FP8_MIXED    160.01 GB   0.2%  ← 最佳（与 GPTQ_INT4、AWQ_INT4 打平，
  GPTQ_INT4        160.01 GB   0.2%    bpp=0.55，区分需要读 per-tensor dtype，
  AWQ_INT4         160.01 GB   0.2%    v0.2 再做）
  FP8              290.94 GB  45.1%  ← gpu_poor 会掉进这个坑
```

这个 tie 是 v0.1.0 dogfood 测试时 `--llm-review` 用 MiniMax-M2 审计工具自己的输出抓到的。LLM 审计抓到的第一个真 bug，v0.1.0 已修。

---

## 诚实原则 —— 7 种标签

每个数字必带一种：

| 标签 | 含义 | 举例 |
|---|---|---|
| `[verified]` | 从 API 或文件直读 | `safetensors 字节：159.62 GB` |
| `[inferred]` | 从 verified 数据一步推导 | `bits/param: 4.39`（字节 ÷ 参数量） |
| `[estimated]` | 公式计算，系数有出处 | `prefill 延迟：735 ms` |
| `[cited]` | 引自 paper / PR / release note | `vLLM ≥0.19.0 支持 CSA+HCA` |
| `[unverified]` | 矩阵条目但无证据，显式标注 | `SGLang day-0 支持待验证` |
| `[unknown]` | 不识别的模型，graceful 降级 | 新 `model_type` 不在注册表 |
| `[llm-opinion]` | **opt-in** LLM 审计，永远不覆盖上面 6 个 | 仅 `--llm-review` 输出用 |

前 6 个都是确定性的。`[llm-opinion]` 显式标记为非权威。

---

## 安装

Python 3.11+。

```bash
# pipx（推荐）
pipx install git+https://github.com/FlyTOmeLight/llm-cal.git@v0.1.0

# uv
uv tool install git+https://github.com/FlyTOmeLight/llm-cal.git@v0.1.0

# pip
pip install git+https://github.com/FlyTOmeLight/llm-cal.git@v0.1.0
```

需要鉴权的模型（Llama、Gemma 等）：

```bash
export HF_TOKEN=hf_...
```

国内 HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 快速开始

```bash
# 基础评估
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800

# 中文输出 + 更长 context
llm-cal Qwen/Qwen2.5-72B-Instruct --gpu A100-80G --context-length 32768 --lang zh

# 完整推导链（每个公式 + 输入 + 步骤 + 来源）
llm-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain

# LLM 审计推导链（opt-in，需要环境变量）
export LLM_CAL_REVIEWER_API_KEY=sk-...
export LLM_CAL_REVIEWER_BASE_URL=https://api.deepseek.com/v1
export LLM_CAL_REVIEWER_MODEL=deepseek-chat
llm-cal deepseek-ai/DeepSeek-V3 --gpu H800 --explain --llm-review

# 全部 53 款支持的 GPU
llm-cal --list-gpus

# 跑 curated benchmark（8 模型 × 33 条检查对比真值）
llm-cal --benchmark
```

节选输出：

```
┌─ deepseek-ai/DeepSeek-V4-Flash  via huggingface @ 6c858e7 ─┐

架构
  model_type         deepseek_v4                             [verified]
  attention          CSA_HCA (heads=64, kv_heads=1, hd=512)  [verified]
  moe                256 routed + 1 shared, top-6            [verified]
  sliding_window     128                                     [verified]

权重
  safetensors 字节   159.62 GB      [verified]
  量化方案           FP4_FP8_MIXED  [inferred]  (与 GPTQ_INT4、AWQ_INT4 打平)

集群规模 — H800
  档位       GPU 数    @ 128K 并发    @ 1.0M 并发
  min          4          ~14            ~1
  dev ★        4          ~14            ~1
  prod         8          ~23            ~2

性能 — dev 档（4× H800）
  prefill 延迟       735 ms @ 2000 input tokens     [estimated, Kaplan 2020]
  decode 吞吐        48 tok/s per user              [estimated, Kwon SOSP 2023]
  bottleneck         显存带宽                        [inferred]

生成命令
  vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 4 --max-model-len 1048576 \
    --trust-remote-code --gpu-memory-utilization 0.9 \
    --attention-backend auto
```

---

## CLI 参数

```
llm-cal [MODEL_ID] [OPTIONS]

核心：
  --gpu TEXT                     GPU id（见 --list-gpus），支持别名，大小写不敏感
  --engine [vllm|sglang]         默认：vllm
  --gpu-count INT                强制集群大小（跳过 min/dev/prod 自动选择）
  --context-length INT           KV 缓存估算用的 context 长度
  --lang [en|zh]                 输出语言（默认从 LANG 自动识别）

性能调优（都有诚实默认值，见 docs/methodology.md）：
  --input-tokens INT             Prefill 输入预算。默认：2000
  --output-tokens INT            Decode 输出预算。默认：512
  --target-tokens-per-sec FLOAT  单用户 decode 的 SLA。默认：30
  --prefill-util FLOAT           算力利用率系数。默认：0.40
  --decode-bw-util FLOAT         显存带宽利用率系数。默认：0.50
  --concurrency-degradation FLOAT  高并发效率衰减。默认：1.0（诚实基线）

自省：
  --explain                      打印每个非平凡数字的完整推导链
  --llm-review                   把推导链发给 LLM 做第二意见（opt-in）
                                 需要：LLM_CAL_REVIEWER_API_KEY / _BASE_URL / _MODEL

元命令：
  --list-gpus                    列出全部 53 款支持的 GPU 并退出
  --benchmark                    跑 curated 数据集（8 模型 × 33 条检查）
  --refresh                      绕过缓存，重新从 HF/ModelScope 拉取
```

---

## 支持的硬件（53 款 GPU）

| 厂商 | 型号 |
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

每条都带 `spec_source`（厂商页、datasheet、或经验证的 benchmark URL）和中英文 notes。

完整信息：`llm-cal --list-gpus`。缺哪款？提 PR 到 `src/llm_cal/hardware/gpu_database.yaml`，纯数据改动，不需要改代码。

---

## 引擎 × 架构矩阵（32 条 / 16 个族）

覆盖 vLLM 0.6–0.19 和 SGLang 0.4–0.5：

- Dense：`llama`、`mistral`、`qwen2`、`qwen3`、`phi`、`gemma`、`internlm`
- MoE：`mixtral`、`qwen3_moe`、`deepseek_v3`、`deepseek_v3_2`、`deepseek_v4`、`phi_moe`
- 稀疏注意力：`deepseek_v3_2`（NSA）、`deepseek_v4`（CSA+HCA）
- 滑动窗口：`mistral`、`qwen3_moe`

每条都带 `verification_level`（`verified` / `cited` / `unverified`）和 `sources[]`（含 URL + `captured_date`）。v0.1 没有 `verified` 条目 —— 作者没有测试硬件。欢迎社区贡献 `tested` 结果。

完整矩阵：[`src/llm_cal/engine_compat/matrix.yaml`](src/llm_cal/engine_compat/matrix.yaml)。

---

## Benchmark（8 模型 × 33 条检查）

`llm-cal --benchmark` 跑 curated 数据集，把工具输出对照真值（HF API 大小、模型卡声明、vLLM 食谱）：

| 模型 | 参考权重 | llm-cal | 量化 | 结果 |
|---|---:|---:|---|:-:|
| `deepseek-ai/DeepSeek-V4-Flash` | 160 GB | 159.62 GB | FP4_FP8_MIXED | ✓ |
| `deepseek-ai/DeepSeek-V3` | 688 GB | 688.59 GB | FP8 | ✓ |
| `deepseek-ai/DeepSeek-V3.2` | 688 GB | 687.84 GB | FP8 (NSA) | ✓ |
| `Qwen/Qwen2.5-72B-Instruct` | 145 GB | 145.41 GB | FP16 | ✓ |
| `Qwen/Qwen3-30B-A3B` | 61 GB | 60.82 GB | FP16 (MoE) | ✓ |
| `Qwen/Qwen2.5-7B` | 14.2 GB | 14.19 GB | FP16 | ✓ |
| `mistralai/Mixtral-8x7B-v0.1` | 93 GB | 93.41 GB | FP16 (MoE) | ✓ |
| `microsoft/Phi-4` | 28 GB | 28.17 GB | FP16 | ✓ |

全过 exit 0，任何 FAIL exit 1。可以挂 CI。

---

## 方法论

每个公式和系数都有一级引用。不用 magic number。

- **Prefill FLOPs**：`2 × params × input_tokens`（Kaplan et al. 2020, *Scaling Laws for Neural Language Models*）
- **Decode 吞吐**：`bandwidth × util / weight_bytes`（Kwon et al. SOSP 2023, *Efficient Memory Management for LLM Serving with PagedAttention*，即 vLLM 论文）
- **KV 缓存布局**：对齐 vLLM `PagedAttention` 和 SGLang `RadixAttention` 源码行为
- **TP 分片**：`per_gpu_KV = total_KV / min(tp_size, num_kv_heads)`，经 vLLM 运行时实测验证
- **利用率系数**：`prefill_util=0.40`、`decode_bw_util=0.50`、`concurrency_degradation=1.0`（诚实默认；按负载通过 CLI flag 覆盖）

完整说明含引用：[`docs/methodology.md`](docs/methodology.md) · [`docs/zh/methodology.md`](docs/zh/methodology.md)。

---

## 文档

- [主页（English）](https://flytomelight.github.io/llm-cal/)
- [主页（中文）](https://flytomelight.github.io/llm-cal/zh/)
- [架构指南](https://flytomelight.github.io/llm-cal/zh/architecture-guide/) —— 新加一个模型类型的 10 步 checklist
- [方法论](https://flytomelight.github.io/llm-cal/zh/methodology/) —— 每个公式的来源
- [贡献指南](CONTRIBUTING.md)

---

## v0.1 范围

**已交付：**

- HuggingFace + ModelScope 作为模型源，`safetensors` 真实字节
- 架构识别：Dense / MoE / GQA / MQA / MLA / NSA / CSA+HCA / Sliding Window
- KV 缓存含 traits 组合、TP 感知分片
- Fleet planner（min/dev/prod、TP 整除约束）
- Prefill / Decode 性能估算
- K/L 并发上界 + bottleneck 分类
- 引擎兼容矩阵（vLLM + SGLang，32 条）
- 命令生成器（vLLM + SGLang，自动带必需 flag）
- 双语输出（en / zh），label 本地化
- `--explain` 推导链
- `--llm-review` opt-in LLM 审计（任意 OpenAI 兼容端点）
- `--benchmark` curated 回归测试
- `--list-gpus` 发现能力
- 53 款 GPU 数据库带 `spec_source` 可追溯

**v0.2 路线图：**

- 读 `safetensors` per-tensor dtype（区分 FP4 / GPTQ / AWQ 的 tie）
- 矩阵超过 100 条时走 lazy loading
- Ollama / GGUF 支持
- 多模态模型（Qwen-VL、InternVL）
- LoRA / adapter VRAM 估算
- `--offline` 模式（离线环境用）
- 社区贡献的 `verified` 矩阵条目（需要真实硬件跑过）

---

## 贡献

特别欢迎：

1. **新 GPU** —— `src/llm_cal/hardware/gpu_database.yaml`（纯数据，不改代码）
2. **新引擎条目** —— `src/llm_cal/engine_compat/matrix.yaml` 带 `sources[]`
3. **新模型架构** —— [10 步 checklist](docs/architecture-guide.md)
4. **`verified` 矩阵条目** —— 如果你有真硬件、真跑过一个 config，把测试结果发我们

开发环境见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

---

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
