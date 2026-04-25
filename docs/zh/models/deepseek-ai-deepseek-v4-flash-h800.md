---
title: deepseek-ai/DeepSeek-V4-Flash 跑在 H800
description: deepseek-ai/DeepSeek-V4-Flash 在 H800 上需要多少 GPU。
---

# deepseek-ai/DeepSeek-V4-Flash 跑在 H800

_deepseek-ai/DeepSeek-V4-Flash 在 H800 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `deepseek_v4` |
| `attention` | `CSA_HCA (heads=64, kv_heads=1, hd=512)` |
| `moe` | `256 routed + 1 shared, top-6` |
| `sliding_window` | `128` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 148.66 GB | `[已验证]` |
| 参数量 | 290.9B | `[估算]` |
| 量化方案 | `FP4_FP8_MIXED` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP4_FP8_MIXED ✓ | 149.02 GB | 378.76 MB 偏少 | 0.2% |
| GPTQ_INT4 | 149.02 GB | 378.76 MB 偏少 | 0.2% |
| AWQ_INT4 | 149.02 GB | 378.76 MB 偏少 | 0.2% |
| INT4 | 135.48 GB | 13.18 GB 偏多 | 9.7% |
| FP8 | 270.95 GB | 122.30 GB 偏少 | 45.1% |

_Best: **FP4_FP8_MIXED** — safetensors header: F8_E8M0 scale tensors + 768 packed-I8 (FP4) weights + 9 FP8 weights — MX block-scaled mixed pack (predicts 160,014,306,918 bytes, 0.2% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 65.72 MB |
| 32,768 | 525.77 MB |
| 131,072 | 2.05 GB |
| 1,048,576 | 16.43 GB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min | 4 | 37.16 GB | 29.89 GB | 14 |
| dev ★ | 4 | 37.16 GB | 29.89 GB | 14 |
| prod | 8 | 18.58 GB | 48.47 GB | 23 |

## 性能

- **Prefill latency** 735 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 151 tok/s `[估算]`
- **Max concurrent users** 5
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --attention-backend auto
```

---

_生成方式_: 
```bash
llm-cal deepseek-ai/DeepSeek-V4-Flash --gpu H800 --engine vllm --lang zh
```
