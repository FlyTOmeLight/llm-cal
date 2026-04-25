---
title: microsoft/Phi-4 跑在 RTX4090
description: microsoft/Phi-4 在 RTX4090 上需要多少 GPU。
---

# microsoft/Phi-4 跑在 RTX4090

_microsoft/Phi-4 在 RTX4090 上需要多少 GPU。_

## 架构

| Field | Value |
|---|---|
| `model_type` | `phi3` |
| `attention` | `GQA (heads=40, kv_heads=10, hd=128)` |

## 权重

| Field | Value | Label |
|---|---|---|
| safetensors 字节 | 27.31 GB | `[已验证]` |
| 参数量 | 14.7B | `[估算]` |
| 量化方案 | `BF16` `[已验证]` |  |

### 量化反演

| Scheme | Predicted | Δ | Error |
|---|---:|---:|---:|
| FP16 | 27.31 GB | 37.92 KB 偏多 | 0.0% |
| BF16 ✓ | 27.31 GB | 37.92 KB 偏多 | 0.0% |
| FP8 | 13.65 GB | 13.65 GB 偏多 | 100.0% |
| INT8 | 13.65 GB | 13.65 GB 偏多 | 100.0% |
| FP4_FP8_MIXED | 7.51 GB | 19.80 GB 偏多 | 263.6% |

_Best: **BF16** — safetensors header: all 42 weight tensors are BF16 (predicts 29,319,004,160 bytes, 0.0% error)_

## KV 缓存（每请求）

| Context tokens | KV bytes |
|---:|---:|
| 4,096 | 800.00 MB |

## 推荐集群

| Tier | GPUs | Weight/GPU | Headroom/GPU | Concurrent @ 128K |
|---|---:|---:|---:|---:|
| min ★ | 4 | 6.83 GB | 13.29 GB | 2 |
| dev | 8 | 3.41 GB | 16.70 GB | 5 |
| prod | 8 | 3.41 GB | 16.70 GB | 5 |

## 性能

- **Prefill latency** 222 ms @ 2000 input tokens `[估算]`
- **Cluster decode throughput** 248 tok/s `[估算]`
- **Max concurrent users** 8
- **Bottleneck** `memory_bandwidth`

## 生成命令

```bash
vllm serve microsoft/Phi-4 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9
```

---

_生成方式_: 
```bash
llm-cal microsoft/Phi-4 --gpu RTX4090 --engine vllm --lang zh
```
