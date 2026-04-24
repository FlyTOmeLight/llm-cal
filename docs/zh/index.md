# llm-cal

**大模型推理硬件计算器** — 架构感知、引擎版本感知、诚实标签。

给它一个 HuggingFace / ModelScope 模型 ID 和一款 GPU，你会得到：

- 真实权重大小（从 `safetensors` metadata 读，不是猜）
- 架构识别：MLA / NSA / CSA+HCA / MoE / 滑动窗口 — 每种都是一等公民 trait
- 多种上下文长度下的每请求 KV cache
- 推荐 GPU 张数：`min` / `dev` / `prod` 三档，带 TP 感知的 KV 分摊
- 推理引擎兼容性：精心整理的矩阵（vLLM + SGLang × 16 种架构族）
- 可直接复制粘贴的 `vllm serve` 或 `sglang launch_server` 命令

输出支持 **中英双语**。

## 为什么又一个 calculator？

现有工具（`gpu_poor`、`llm-vram-calculator`、APXML、SelfHostLLM 等）都用 `参数量 × 精度` 公式估算权重。这个公式在新架构上**会静默出错**：

| 模型 | `gpu_poor` 的答案 | 真实 safetensors | llm-cal |
|---|---|---|---|
| DeepSeek-V4-Flash（FP4+FP8 pack）| 284 GB（假设是 FP8）| **160 GB** | **160 GB** ✓ |
| 标准 FP8 模型 | 正确 | 正确 | 正确 ✓ |

llm-cal 从 HuggingFace API 读真实文件大小，再对比每一种已知量化方案——最匹配的胜出。DeepSeek-V4 的故事变得可见：

```
量化方案对账（观测值 vs 各方案预测值）
  量化方案          预测字节        差值            误差 %
  FP4_FP8_MIXED    160.01 GB     397 MB 偏低     0.2%  ← 胜出
  FP8              290.94 GB     131 GB 偏低     45.1% ← gpu_poor 的陷阱
```

而且每个数字都带标签，告诉你它来自哪：

- `[已验证]` — 直接从 HF API / config.json 读取
- `[推断]` — 基于 `[已验证]` 数据的单步推导
- `[估算]` — 公式计算（KV cache、权重分摊）
- `[引用]` — 来自 release notes / PR / 官方公告
- `[未经验证]` — 矩阵中未有证据的条目，明确标出
- `[未知]` — 识别失败时的 graceful 降级

## 安装

需要 Python 3.11+。

=== "pipx（推荐）"

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

认证（gated 模型如 Llama、Gemma 需要）：

```bash
export HF_TOKEN=hf_...
```

国内镜像（HF 慢时）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速上手

```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm --lang zh
```

详细用法见[快速开始](quickstart.md)，工具内部原理见[架构指南](architecture-guide.md)，贡献指南见[参与贡献](contributing.md)。

## 验证

针对精选参考数据跑内置 benchmark：

```bash
llm-cal --benchmark
```

当前结果：**33/33 PASS**，覆盖 8 个参考模型 × 6 种检查类型。每个预期值都在数据集里写明来源（HF API / 模型卡 / vLLM recipe / 手算）。

## 支持范围

- **47 款 GPU**：NVIDIA / AMD / Intel Habana / 华为昇腾 / 寒武纪 / 摩尔线程 / 沐曦 / 百度昆仑芯 / 壁仞 / 天数智芯 / 海光
- **16 种架构族**（引擎兼容矩阵覆盖）
- **2 个推理引擎**：vLLM 和 SGLang
- **2 种输出语言**：英文和中文

运行 `llm-cal --list-gpus --lang zh` 可以看完整 GPU 表和别名。

## 开源协议

Apache-2.0，详见 [LICENSE](https://github.com/FlyTOmeLight/llm-cal/blob/main/LICENSE)。
