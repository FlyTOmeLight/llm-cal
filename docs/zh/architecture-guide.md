# 架构指南

工具如何识别模型、计算显存占用、匹配引擎和硬件。贡献代码前请阅读本文。

---

## 核心洞察：架构是 traits 的组合，不是单一标签

一个模型不只是"MoE"或"MLA"——而是**一组 traits 的组合**。DeepSeek-V3.2 = MoE + MLA + NSA。DeepSeek-V4 = MoE + MLA + CSA+HCA + 滑动窗口。Qwen3 = dense + GQA + RoPE。工具用 `ArchitectureProfile` 捕获这种组合：

```python
@dataclass(frozen=True)
class ArchitectureProfile:
    model_type: str
    family: Family                  # transformer | state_space | unknown
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    attention: AttentionTraits      # variant (MHA/GQA/MQA/MLA/NSA/CSA_HCA) + 维度
    moe: MoETraits | None           # None 表示 dense
    position: PositionTraits        # RoPE / YaRN / AliBi / none
    sliding_window: int | None
    auxiliary: dict                 # 未来 trait 的透传字段
    confidence: Confidence          # HIGH | MEDIUM | LOW
```

单 module 分派（`if model_type == "deepseek_v4": ...`）无法表达这种组合。Traits 可以。

---

## 检测流程

```
                     ┌──────────────────────┐
                     │   config.json 字典   │
                     └──────────┬───────────┘
                                │
                                ▼
             ┌──────────────────────────────────┐
             │  model_type ∈ STATE_SPACE_TYPES  │
             │    或存在 `ssm_cfg` 字段？        │
             └──────┬─────────────────┬─────────┘
                   是                  │否
                    ▼                  ▼
           Family.STATE_SPACE   ┌────────────────────────────┐
           （v0.1 不支持）      │ model_type 和 architectures│
                                │ 都缺失？                   │
                                └──────┬─────────────────┬───┘
                                      是                  │否
                                       ▼                  ▼
                             _fallback_unknown     ┌─────────────────────┐
                             （Family.UNKNOWN，    │  必需字段缺失？     │
                              confidence=LOW）     │  (layers/hidden)    │
                                                   └──────┬──────────┬───┘
                                                          │缺失       │有
                                                          ▼          ▼
                                                 _fallback_unknown   │
                                                                      ▼
                                                     ┌───────────────────────┐
                                                     │ 并行调用独立 trait    │
                                                     │ 子检测器：            │
                                                     │  detect_attention()   │
                                                     │  detect_moe()         │
                                                     │  detect_position()    │
                                                     │  detect_sliding_window│
                                                     └───────────┬───────────┘
                                                                 ▼
                                                     ┌───────────────────────┐
                                                     │ ArchitectureProfile   │
                                                     │  family=TRANSFORMER   │
                                                     │  confidence = HIGH 当 │
                                                     │    model_type 在注册  │
                                                     │    表中，否则 MEDIUM  │
                                                     └───────────────────────┘
```

---

## Attention 变体检测——顺序敏感

`detect_attention()` 是顺序敏感的。第一个匹配决定 variant，但形状字段（heads / kv_heads / head_dim）永远被填充。

```
优先级      关键字                        示例模型
─────────────────────────────────────────────────────────────
1  CSA_HCA  compress_ratios 数组          DeepSeek-V4-Flash
             （长度 == n_hidden_layers
              或 ± num_nextn_predict_layers）
2  NSA      存在 nsa_config                DeepSeek-V3.2
             或 sparse_attention_cfg
3  MLA      q_lora_rank 或 kv_lora_rank    DeepSeek-V2, V3
4  MQA      num_kv_heads == 1              （无 MLA keys 时）
5  GQA      num_kv_heads < num_heads       Llama-3, Qwen
6  MHA      默认                            老 Llama 1/2
```

**为什么 CSA_HCA 要做长度检查？** `compress_ratios` 这个键名可能在未来架构中被用于不同语义。长度相等检查（`len == num_hidden_layers` 或 `== num_hidden_layers + num_nextn_predict_layers`）是防止误报的保护。Reviewer 在设计阶段反复强调过这一点。Regression test：`tests/test_detector.py::test_length_mismatch_is_not_classified_as_csa_hca`。

---

## KV cache 公式——traits 组合式

```
baseline_per_token_per_layer_per_req = 2 (K+V) × num_kv_heads × head_dim × dtype_bytes
                                        （MLA 情况：kv_lora_rank × dtype_bytes）

baseline = baseline_per_token × effective_seq_len × num_hidden_layers

effective_seq_len:
  如果是 sparse variant (CSA_HCA, NSA): seq_len（sliding_window 不适用——稀疏机制
                                                  已经编码了每层的减量）
  否则如果存在 sliding_window:           min(seq_len, sliding_window)
  否则:                                   seq_len

组合式修正：
  CSA_HCA:  baseline × average_compress_ratio(compress_ratios)
            （0 → 保留 1.0，N>0 → 保留 1/N，跨所有层平均）
  NSA:      baseline × (nsa_topk / effective_seq_len)，clamp 到 [0, 1]

per_gpu_KV = total_KV / min(tp_size, max(1, num_kv_heads))
   — MQA (kv_heads=1): 始终复制（除数 = 1）
   — GQA (kv_heads=G): 最多切分 G 份
   — MHA: 完全切分到 num_heads
```

验证：DeepSeek-V4-Flash 在 128K 上下文下对比手算结果误差 < 1%，见 `tests/test_formulas.py::test_128k_kv_cache_within_1_percent`。

---

## 如何添加新架构（10 步清单）

假设一个新模型 `FooModel` 带 `model_type=foo_v1`，dense + GQA，并且有个新字段 `reuse_kv_across_layers: bool` 在 config 里，该字段能让 KV cache 减半。

1. **准备 config.json 样本** — 复制一份到 `tests/fixtures/configs/foo_v1.json`。这是你测试的锚点。越真实越好。

2. **注册 model_type** 到 `src/llm_cal/architecture/detector.py`：
   ```python
   KNOWN_MODEL_TYPES: frozenset[str] = frozenset({..., "foo_v1"})
   ```
   这会把 confidence 从 MEDIUM 提到 HIGH。

3. **扩展 `AttentionTraits`**（仅当需要新 variant）。这个例子中 `reuse_kv` 是对标准公式的乘数修正，不是新 variant——跳过此步。如果确实需要新 variant（比如 "FOO_SPARSE"），扩展：
   ```python
   AttentionVariant = Literal["MHA", "GQA", "MQA", "MLA", "NSA", "CSA_HCA", "FOO_SPARSE"]
   ```

4. **扩展 `traits.py` 的 `detect_attention()`**（仅当新 variant）。按正确优先级加检测逻辑。记住：第一个匹配胜出。

5. **通过 auxiliary 传递新字段**。在 `detector.py` 主路径里加：
   ```python
   if config.get("reuse_kv_across_layers") is True:
       auxiliary["reuse_kv_across_layers"] = True
   ```

6. **修改 `architecture/formulas/kv_cache.py` 的 KV 公式**读新字段：
   ```python
   result_bytes = baseline
   if profile.auxiliary.get("reuse_kv_across_layers"):
       result_bytes = result_bytes // 2
   ```

7. **加 fixture-based 的 detector 测试** 到 `tests/test_detector.py`：
   ```python
   def test_foo_v1_detection(self, load_config):
       p = detect(load_config("foo_v1"))
       assert p.family == Family.TRANSFORMER
       assert p.confidence == Confidence.HIGH
       assert p.attention.variant == "GQA"
   ```

8. **加 formula 测试** 到 `tests/test_formulas.py`。

9. **加引擎兼容矩阵条目** 到 `src/llm_cal/engine_compat/matrix.yaml`，至少一条 source（release notes / PR）。诚实标注 verification_level：
   ```yaml
   - engine: vllm
     version_spec: ">=0.20.0"
     matches_model_type: foo_v1
     support: full
     verification_level: cited
     sources:
       - type: release_notes
         url: https://github.com/vllm-project/vllm/releases/tag/v0.20.0
         captured_date: 2026-06-01
   ```

10. **为新 trait 字符串加 i18n** 到 `src/llm_cal/common/i18n.py`。如果新字段在 formatter 中面向用户，`en` 和 `zh` 两套都要有。

跑完整测试套件：`pytest` + `mypy src` + `ruff check`。全绿就 PR。

---

## 如何添加新 GPU

单文件改动：`src/llm_cal/hardware/gpu_database.yaml`。不用改代码。

```yaml
  - id: MI300X
    aliases: [MI300X-192G]
    memory_gb: 192
    nvlink_bandwidth_gbps: 0  # xGMI 互联时填 0，注释里写清楚
    fp16_tflops: 1307
    fp8_support: true
    fp4_support: false
    spec_source: "AMD Instinct MI300X datasheet 2023-12"
    notes_en: "AMD flagship. xGMI interconnect. vLLM support via ROCm."
    notes_zh: "AMD 旗舰，xGMI 互联，vLLM 通过 ROCm 支持。"
```

就这样。GPU 会立刻通过 ID 或任意别名可查。

想给新条目加 regression 保护的话，加个 `tests/test_hardware.py::test_mi300x_is_loaded` 测试。

---

## 如何添加引擎兼容条目

`src/llm_cal/engine_compat/matrix.yaml`。按既有格式。关键规则：

- **`verification_level: verified`** 要求至少一条 `type: tested` 来源带 `tester`、`date`、`hardware`、`metrics`。没实际跑过别声明 `verified`。
- **`verification_level: cited`** 要求至少一条 `sources[]` 带 URL + `captured_date`。
- **`verification_level: unverified`** 允许空 `sources[]`，但工具会在 UI 里大声提示用户这是猜的。
- **双语 notes**：`caveats_en` 和 `caveats_zh` 都必填（可以是空数组）。flag 级的 `note_en` / `note_zh` 可选。

---

## 标签纪律——工具的灵魂

**输出里每一个数字都带标签。** 这不是可选项。按来源分类：

| 数值来源 | 标签 |
|---|---|
| HF API `model_info().siblings[].size` | `[verified]` |
| `config.json` 字段直接读取 | `[verified]` |
| `sum(safetensors file sizes)` | `[verified]`（直接读取，即使是求和）|
| `观测字节 / 参数数` = bits/param | `[inferred]` |
| 最近锚点量化方案匹配 | `[inferred]` |
| 基于 profile 算出的 KV cache | `[estimated]` |
| 基于 profile 算出的权重（未观测）| `[estimated]` |
| 矩阵中带 sources 的引擎支持 | `[cited]` |
| 矩阵中无 sources 的引擎支持 | `[unverified]` |
| Graceful 降级路径（未知架构）| `[unknown]` |

**不要做的事**：
- 把算出来的数字标 `[verified]`。即使 `bits/param` 也是 `[inferred]`，因为是推导出来的。
- 在 `[unverified]` 条目旁显示绿色对号。UI 故意让这些条目显眼。
- 加新标签时不更新 i18n keys 和 legend 渲染。
