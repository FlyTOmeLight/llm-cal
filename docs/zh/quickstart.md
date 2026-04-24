# 快速开始

## 经典调用

```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm --lang zh
```

这是工具的参考案例。你会得到：

1. **架构识别** — 检测到 DeepSeek-V4，CSA+HCA + MoE + 滑动窗口，`置信度：high`
2. **权重** — `safetensors 总字节：159.62 GB [已验证]`、`量化方案推断：FP4_FP8_MIXED [推断]`
3. **对账** — 每种量化方案预测字节对比。FP4_FP8_MIXED 胜出（0.2% 误差），FP8 差 45.1%
4. **KV Cache** — 4K / 32K / 128K / 1M 四个上下文长度的估算
5. **引擎兼容性** — vLLM ≥0.19.0，`[引用]` 带来源 URL
6. **目标硬件** — H800 规格，双语备注
7. **推荐 GPU 张数** — min / dev / prod 三档，含 TP 感知的 KV 分摊
8. **生成命令** — 可直接复制粘贴的 `vllm serve ...`

## 常用参数

```bash
# 中文输出
llm-cal <model> --gpu H800 --engine vllm --lang zh

# 强制指定 GPU 张数（跳过 min/dev/prod 推荐）
llm-cal <model> --gpu H100 --gpu-count 4

# 覆盖上下文长度以调整 KV cache 计算
llm-cal <model> --gpu H800 --context-length 65536

# 绕过缓存（模型 repo 更新后有用）
llm-cal <model> --gpu H800 --refresh

# 查看所有支持的 GPU
llm-cal --list-gpus --lang zh

# 跑内置 benchmark 验证
llm-cal --benchmark
```

## 非 NVIDIA 例子

```bash
# AMD 旗舰（256 GB HBM3E，单卡显存最大）
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu MI325X --engine vllm --lang zh

# 华为昇腾 910B4 推理卡（32 GB）
llm-cal Qwen/Qwen2.5-7B-Instruct --gpu 910B4 --engine vllm --lang zh

# 国产 AI 芯片（支持中文别名）
llm-cal Qwen/Qwen2.5-14B-Instruct --gpu 曦云C500 --engine vllm --lang zh
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu 昆仑芯P800 --engine vllm --lang zh
llm-cal Qwen/Qwen2.5-7B-Instruct --gpu 摩尔线程S4000 --engine vllm --lang zh
```

## 输出标签说明

报告里每个数字都带标签：

| 标签 | 含义 | 示例 |
|---|---|---|
| `[已验证]` | 从 API / 文件直接读取 | `safetensors 总字节：159.62 GB`（HF siblings API） |
| `[推断]` | 基于已验证数据的单步推导 | `每参数位数：4.39`（字节 ÷ 参数数） |
| `[估算]` | 基于公式的计算 | `KV cache @ 128K：2.21 GB` |
| `[引用]` | 外部来源（release note / PR） | `vLLM ≥0.19.0 支持 CSA+HCA` |
| `[未经验证]` | 矩阵条目但无证据，显式标出 | `SGLang Day-0 支持待定` |
| `[未知]` | 无法识别，graceful 降级 | 新模型类型未在注册表中 |

**不要相信任何不带来源的数字工具**——这就是本工具的核心价值主张。

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功，或 `--benchmark` 全通过 |
| 1 | `--benchmark` 有失败 |
| 2 | 需要认证（gated 模型但没设 `HF_TOKEN`）|
| 3 | 模型未找到 |
| 4 | 数据源不可用（网络 / 限流 / 5xx）|

方便 CI 脚本化使用。

## 常见问题

### "需要认证 / Authentication required"

设置 `HF_TOKEN`：

```bash
export HF_TOKEN=hf_xxxxx
```

或者首次使用前执行 `huggingface-cli login`。

### 国内访问 HF 慢

设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 工具报"model_type 不在 v0.1 矩阵中"

引擎兼容矩阵暂时没有这个模型家族的条目。报告其他部分（权重、KV cache、fleet 推荐）依然工作——只是引擎那一块显示"未匹配"。

欢迎通过 PR 贡献矩阵条目，见[参与贡献](contributing.md)。

### 工具报 `[未知]` 架构

检测器不认识的全新模型类型。工具会降级：

- `[已验证]` safetensors 字节数（依然可信）
- **无** KV cache 估算
- **无** 引擎兼容性信息
- 只基于权重塞得下给出保守 fleet 推荐

这是设计。详见[架构指南](architecture-guide.md) 的 "Graceful unknown" 部分。
