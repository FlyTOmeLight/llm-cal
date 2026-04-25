---
title: mistralai/Mixtral-8x7B-v0.1 跑在 H100
description: mistralai/Mixtral-8x7B-v0.1 在 H100 上需要多少 GPU。
---

# mistralai/Mixtral-8x7B-v0.1 跑在 H100

_mistralai/Mixtral-8x7B-v0.1 在 H100 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `mixtral` |
| `attention` | `GQA (heads=32, kv_heads=8, hd=128)` |
| `moe` | `8 routed + 0 shared, top-2` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 86.99 GB | `[已验证]` |
| 参数量 | 46.7B | `[估算]` |
| 量化方案 | `BF16` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP16 | 86.99 GB | 133.09 KB 偏多 | 0.0% |
| BF16 ✓ | 86.99 GB | 133.09 KB 偏多 | 0.0% |
| FP8 | 43.50 GB | 43.50 GB 偏多 | 100.0% |
| INT8 | 43.50 GB | 43.50 GB 偏多 | 100.0% |
| FP4_FP8_MIXED | 23.92 GB | 63.07 GB 偏多 | 263.6% |

_Best: **BF16** — safetensors header: all 48 weight tensors are BF16 (predicts 93,405,577,216 bytes, 0.0% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 512.00 MB |
| 32,768 | 4.00 GB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min | 2 | 43.50 GB | 23.56 GB | 2 |
| dev ★ | 4 | 21.75 GB | 45.31 GB | 11 |
| prod | 8 | 10.87 GB | 56.18 GB | 28 |

## 性能

- **Prefill latency** 118 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 258 tok/s `[估算]`
- **Max concurrent users** 8
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve mistralai/Mixtral-8x7B-v0.1 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

---

_生成方式_: 
```bash
llm-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --engine vllm --lang zh
```
