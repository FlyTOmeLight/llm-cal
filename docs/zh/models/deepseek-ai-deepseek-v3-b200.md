---
title: deepseek-ai/DeepSeek-V3 跑在 B200
description: deepseek-ai/DeepSeek-V3 在 B200 上需要多少 GPU。
---

# deepseek-ai/DeepSeek-V3 跑在 B200

_deepseek-ai/DeepSeek-V3 在 B200 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `deepseek_v3` |
| `attention` | `MLA (heads=128, kv_heads=128, hd=56)` |
| `moe` | `256 routed + 1 shared, top-8` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 641.30 GB | `[已验证]` |
| 参数量 | 695.7B | `[估算]` |
| 量化方案 | `FP8` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP8 ✓ | 647.96 GB | 6.66 GB 偏少 | 1.0% |
| INT8 | 647.96 GB | 6.66 GB 偏少 | 1.0% |
| FP16 | 1.27 TB | 654.62 GB 偏少 | 50.5% |
| BF16 | 1.27 TB | 654.62 GB 偏少 | 50.5% |
| FP4_FP8_MIXED | 356.38 GB | 284.92 GB 偏多 | 79.9% |

_Best: **FP8** — config.json quantization_config.quant_method=fp8 (predicts 695,742,322,688 bytes, 1.0% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 244.00 MB |
| 32,768 | 1.91 GB |
| 131,072 | 7.62 GB |
| 163,840 | 9.53 GB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min | 8 | 80.16 GB | 80.77 GB | 84 |
| dev ★ | 8 | 80.16 GB | 80.77 GB | 84 |
| prod | 8 | 80.16 GB | 80.77 GB | 84 |

## 性能

- **Prefill latency** 387 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 335 tok/s `[估算]`
- **Max concurrent users** 11
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --max-model-len 163840 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

---

_生成方式_: 
```bash
llm-cal deepseek-ai/DeepSeek-V3 --gpu B200 --engine vllm --lang zh
```
