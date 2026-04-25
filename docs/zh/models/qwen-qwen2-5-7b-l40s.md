---
title: Qwen/Qwen2.5-7B 跑在 L40S
description: Qwen/Qwen2.5-7B 在 L40S 上需要多少 GPU。
---

# Qwen/Qwen2.5-7B 跑在 L40S

_Qwen/Qwen2.5-7B 在 L40S 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `qwen2` |
| `attention` | `GQA (heads=28, kv_heads=4, hd=128)` |
| `sliding_window` | `131072` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 14.19 GB | `[已验证]` |
| 参数量 | 7.6B | `[估算]` |
| 量化方案 | `BF16` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP16 | 14.18 GB | 296.95 KB 偏多 | 0.0% |
| BF16 ✓ | 14.18 GB | 296.95 KB 偏多 | 0.0% |
| FP8 | 7.09 GB | 7.09 GB 偏多 | 100.0% |
| INT8 | 7.09 GB | 7.09 GB 偏多 | 100.0% |
| FP4_FP8_MIXED | 3.90 GB | 10.28 GB 偏多 | 263.6% |

_Best: **BF16** — safetensors header: all 73 weight tensors are BF16 (predicts 15,230,967,808 bytes, 0.0% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 224.00 MB |
| 32,768 | 1.75 GB |
| 131,072 | 7.00 GB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min | 1 | 14.19 GB | 26.05 GB | 3 |
| dev ★ | 2 | 7.09 GB | 33.14 GB | 9 |
| prod | 4 | 3.55 GB | 36.69 GB | 20 |

## 性能

- **Prefill latency** 105 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 102 tok/s `[估算]`
- **Max concurrent users** 3
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve Qwen/Qwen2.5-7B \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9
```

---

_生成方式_: 
```bash
llm-cal Qwen/Qwen2.5-7B --gpu L40S --engine vllm --lang zh
```
