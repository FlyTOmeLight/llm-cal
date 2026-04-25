---
title: Qwen/Qwen3-30B-A3B 跑在 H100
description: Qwen/Qwen3-30B-A3B 在 H100 上需要多少 GPU。
---

# Qwen/Qwen3-30B-A3B 跑在 H100

_Qwen/Qwen3-30B-A3B 在 H100 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `qwen3_moe` |
| `attention` | `GQA (heads=32, kv_heads=4, hd=128)` |
| `moe` | `128 routed + 0 shared, top-8` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 56.87 GB | `[已验证]` |
| 参数量 | 30.5B | `[估算]` |
| 量化方案 | `BF16` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP16 | 56.87 GB | 2.25 MB 偏多 | 0.0% |
| BF16 ✓ | 56.87 GB | 2.25 MB 偏多 | 0.0% |
| FP8 | 28.44 GB | 28.44 GB 偏多 | 100.0% |
| INT8 | 28.44 GB | 28.44 GB 偏多 | 100.0% |
| FP4_FP8_MIXED | 15.64 GB | 41.23 GB 偏多 | 263.7% |

_Best: **BF16** — safetensors header: all 1262 weight tensors are BF16 (predicts 61,064,216,576 bytes, 0.0% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 384.00 MB |
| 32,768 | 3.00 GB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min | 2 | 28.44 GB | 38.62 GB | 6 |
| dev ★ | 4 | 14.22 GB | 52.84 GB | 17 |
| prod | 4 | 14.22 GB | 52.84 GB | 17 |

## 性能

- **Prefill latency** 77 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 395 tok/s `[估算]`
- **Max concurrent users** 13
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel
```

---

_生成方式_: 
```bash
llm-cal Qwen/Qwen3-30B-A3B --gpu H100 --engine vllm --lang zh
```
