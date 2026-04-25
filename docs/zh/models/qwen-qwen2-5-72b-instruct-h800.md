---
title: Qwen/Qwen2.5-72B-Instruct 跑在 H800
description: Qwen/Qwen2.5-72B-Instruct 在 H800 上需要多少 GPU。
---

# Qwen/Qwen2.5-72B-Instruct 跑在 H800

_Qwen/Qwen2.5-72B-Instruct 在 H800 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `qwen2` |
| `attention` | `GQA (heads=64, kv_heads=8, hd=128)` |
| `sliding_window` | `131072` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 135.43 GB | `[已验证]` |
| 参数量 | 72.7B | `[估算]` |
| 量化方案 | `BF16` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP16 | 135.42 GB | 1.68 MB 偏多 | 0.0% |
| BF16 ✓ | 135.42 GB | 1.68 MB 偏多 | 0.0% |
| FP8 | 67.71 GB | 67.71 GB 偏多 | 100.0% |
| INT8 | 67.71 GB | 67.71 GB 偏多 | 100.0% |
| FP4_FP8_MIXED | 37.24 GB | 98.18 GB 偏多 | 263.6% |

_Best: **BF16** — safetensors header: all 23 weight tensors are BF16 (predicts 145,410,752,512 bytes, 0.0% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 1.25 GB |
| 32,768 | 10.00 GB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min | 4 | 33.86 GB | 33.20 GB | 3 |
| dev ★ | 8 | 16.93 GB | 50.13 GB | 10 |
| prod | 8 | 16.93 GB | 50.13 GB | 10 |

## 性能

- **Prefill latency** 92 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 663 tok/s `[估算]`
- **Max concurrent users** 22
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9
```

---

_生成方式_: 
```bash
llm-cal Qwen/Qwen2.5-72B-Instruct --gpu H800 --engine vllm --lang zh
```
