# Architecture Guide

How the tool identifies a model, computes its memory footprint, and matches it
to engines + hardware. Read this before contributing.

---

## The core insight: architectures are traits, not labels

A model isn't "MoE" or "MLA" — it's a **composition of traits**. DeepSeek-V3.2 is
MoE + MLA + NSA. DeepSeek-V4 is MoE + MLA + CSA+HCA + sliding window. Qwen3 is
dense + GQA + RoPE. The tool captures this with `ArchitectureProfile`:

```python
@dataclass(frozen=True)
class ArchitectureProfile:
    model_type: str
    family: Family                  # transformer | state_space | unknown
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    attention: AttentionTraits      # variant (MHA/GQA/MQA/MLA/NSA/CSA_HCA) + shape
    moe: MoETraits | None           # None = dense
    position: PositionTraits        # RoPE / YaRN / AliBi / none
    sliding_window: int | None
    auxiliary: dict                 # pass-through for future traits
    confidence: Confidence          # HIGH | MEDIUM | LOW
```

Single-module dispatch (`if model_type == "deepseek_v4": ...`) can't express
this combination. Traits can.

---

## Detection flow

```
                     ┌──────────────────────┐
                     │   config.json dict   │
                     └──────────┬───────────┘
                                │
                                ▼
             ┌──────────────────────────────────┐
             │ model_type in STATE_SPACE_TYPES? │
             │         or `ssm_cfg` present?    │
             └──────┬─────────────────┬─────────┘
                   yes                │no
                    ▼                 ▼
           Family.STATE_SPACE  ┌────────────────────────────┐
           (v0.1 unsupported)  │ model_type AND architectures│
                               │ both missing?              │
                               └──────┬─────────────────┬───┘
                                     yes                │no
                                      ▼                 ▼
                            _fallback_unknown    ┌─────────────────────┐
                            (Family.UNKNOWN,     │  required fields    │
                             confidence=LOW)     │  (layers/hidden)?   │
                                                 └──────┬──────────┬───┘
                                                        │missing   │ok
                                                        ▼          ▼
                                               _fallback_unknown    │
                                                                    ▼
                                                    ┌───────────────────────┐
                                                    │ gather independent    │
                                                    │ trait sub-detectors:  │
                                                    │  detect_attention()   │
                                                    │  detect_moe()         │
                                                    │  detect_position()    │
                                                    │  detect_sliding_window│
                                                    └───────────┬───────────┘
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │ ArchitectureProfile   │
                                                    │  family=TRANSFORMER   │
                                                    │  confidence = HIGH if │
                                                    │    model_type known,  │
                                                    │  else MEDIUM          │
                                                    └───────────────────────┘
```

---

## Attention variant detection — order matters

`detect_attention()` is order-sensitive. First match wins on variant, but shape
fields (heads / kv_heads / head_dim) are always populated.

```
priority        key                         example model
─────────────────────────────────────────────────────────────
1  CSA_HCA      compress_ratios (matched    DeepSeek-V4-Flash
                length vs n_hidden_layers
                ± num_nextn_predict_layers)
2  NSA          nsa_config present          DeepSeek-V3.2
                OR sparse_attention_cfg
3  MLA          q_lora_rank or kv_lora_rank DeepSeek-V2, V3
4  MQA          num_kv_heads == 1           (when no MLA keys)
5  GQA          num_kv_heads < num_heads    Llama-3, Qwen
6  MHA          default                     old Llama 1/2
```

**Why the CSA_HCA length check matters:** `compress_ratios` as a key name could
legitimately appear in future architectures with different semantics. The length
equality check (`len == num_hidden_layers` or `== num_hidden_layers +
num_nextn_predict_layers`) is the guard that prevents false positives. Reviewer
flagged this twice during design review; regression test:
`tests/test_detector.py::test_length_mismatch_is_not_classified_as_csa_hca`.

---

## KV cache formula — traits composition

```
baseline_per_token_per_layer_per_req = 2 (K+V) × num_kv_heads × head_dim × dtype_bytes
                                        (or, for MLA: kv_lora_rank × dtype_bytes)

baseline = baseline_per_token × effective_seq_len × num_hidden_layers

effective_seq_len:
  if sparse_variant (CSA_HCA, NSA): seq_len (sliding_window does NOT apply — the
                                             sparse mechanism already encodes
                                             per-layer reduction)
  elif sliding_window present:       min(seq_len, sliding_window)
  else:                              seq_len

compositional modifiers:
  CSA_HCA:  baseline × average_compress_ratio(compress_ratios)
            (0 → keep 1.0, N>0 → keep 1/N, averaged across all layers)
  NSA:      baseline × (nsa_topk / effective_seq_len), clamped to [0, 1]

per_gpu_KV = total_KV / min(tp_size, max(1, num_kv_heads))
   — MQA (kv_heads=1): always replicates (divisor = 1)
   — GQA (kv_heads=G): splits up to G ways
   — MHA: splits fully
```

Validation: DeepSeek-V4-Flash at 128K context vs hand-math:
error < 1% in `tests/test_formulas.py::test_128k_kv_cache_within_1_percent`.

---

## How to add a new architecture (10-step checklist)

Suppose a model `FooModel` ships with `model_type=foo_v1`, dense + GQA, and a
novel `reuse_kv_across_layers: bool` config flag that halves KV cache.

1. **Sample config.json** — save a copy under `tests/fixtures/configs/foo_v1.json`.
   This is your test anchor. The more realistic, the better.

2. **Register the model_type** in `src/llm_cal/architecture/detector.py`:
   ```python
   KNOWN_MODEL_TYPES: frozenset[str] = frozenset({..., "foo_v1"})
   ```
   This flips confidence from MEDIUM to HIGH.

3. **Extend `AttentionTraits` (only if a new variant is needed).** In this case,
   reuse_kv is a multiplier on the standard formula, not a new variant — skip
   this step. If you ARE introducing a new variant (say, "FOO_SPARSE"), extend:
   ```python
   AttentionVariant = Literal["MHA", "GQA", "MQA", "MLA", "NSA", "CSA_HCA", "FOO_SPARSE"]
   ```

4. **Extend `detect_attention()` in `traits.py`** (only if new variant). Add
   detection logic with the correct priority order. Remember: first match wins.

5. **Pass through new config fields via auxiliary**. In `detector.py`'s main
   path, add:
   ```python
   if config.get("reuse_kv_across_layers") is True:
       auxiliary["reuse_kv_across_layers"] = True
   ```

6. **Modify the KV formula** in `architecture/formulas/kv_cache.py` to read the
   new field:
   ```python
   result_bytes = baseline
   if profile.auxiliary.get("reuse_kv_across_layers"):
       result_bytes = result_bytes // 2
   ```

7. **Add a fixture-based detector test** in `tests/test_detector.py`:
   ```python
   def test_foo_v1_detection(self, load_config):
       p = detect(load_config("foo_v1"))
       assert p.family == Family.TRANSFORMER
       assert p.confidence == Confidence.HIGH
       assert p.attention.variant == "GQA"  # or FOO_SPARSE
   ```

8. **Add a formula test** in `tests/test_formulas.py`:
   ```python
   def test_foo_v1_kv_halved_by_reuse(self):
       profile = detect(load_config("foo_v1"))
       kv = compute_kv_cache_bytes(profile, seq_len=128_000, dtype_bytes=2)
       # Hand-compute the expected with reuse applied
       expected = ...
       assert abs(kv.value - expected) / expected < 0.01
   ```

9. **Add a compat matrix entry** in `src/llm_cal/engine_compat/matrix.yaml` with
   at least one source (release notes / PR). Mark verification_level honestly:
   ```yaml
   - engine: vllm
     version_spec: ">=0.20.0"
     matches_model_type: foo_v1
     support: full
     verification_level: cited     # or 'unverified' if only inferred
     sources:
       - type: release_notes
         url: https://github.com/vllm-project/vllm/releases/tag/v0.20.0
         captured_date: 2026-06-01
   ```

10. **Add i18n for any new trait string** in `src/llm_cal/common/i18n.py`. If
    your new field is user-facing in the formatter, both `en` and `zh` must exist.

Run the full suite: `pytest` + `mypy src` + `ruff check`. If green, PR.

---

## How to add a new GPU

Single-file change: `src/llm_cal/hardware/gpu_database.yaml`. No code.

```yaml
  - id: MI300X
    aliases: [MI300X-192G]
    memory_gb: 192
    nvlink_bandwidth_gbps: 0  # xGMI instead — caller handles interconnect notes
    fp16_tflops: 1307
    fp8_support: true
    fp4_support: false
    notes_en: "AMD flagship. xGMI interconnect. vLLM support via ROCm."
    notes_zh: "AMD 旗舰，xGMI 互联，vLLM 通过 ROCm 支持。"
```

That's it. The GPU becomes queryable by ID or any alias immediately.

Add `tests/test_hardware.py::test_mi300x_is_loaded` if you want to anchor
the new entry against regressions.

---

## How to add an engine compat entry

`src/llm_cal/engine_compat/matrix.yaml`. Follow the existing format. Critical
rules:

- **`verification_level: verified`** requires at least one `type: tested`
  source with `tester`, `date`, `hardware`, `metrics`. Don't claim `verified`
  unless you actually ran it.
- **`verification_level: cited`** requires at least one `sources[]` entry with
  a URL and `captured_date`.
- **`verification_level: unverified`** — empty `sources[]` is allowed, but the
  tool will surface this loudly in the UI so users know the entry is a guess.
- **Bilingual notes**: `caveats_en` and `caveats_zh` both required (can be
  empty arrays). Flag-level `note_en` / `note_zh` are optional.

---

## Label discipline — the tool's soul

**Every number in the output is tagged.** This is non-negotiable. Rules by
context:

| Source of value | Label |
|---|---|
| HF API `model_info().siblings[].size` | `[verified]` |
| `config.json` field read directly | `[verified]` |
| `sum(safetensors file sizes)` | `[verified]` (it IS a direct read, even though it's a sum) |
| `observed_bytes / total_params` = bits/param | `[inferred]` |
| Nearest-anchor quantization match | `[inferred]` |
| KV cache computed from profile | `[estimated]` |
| Weight computed from profile (not observed) | `[estimated]` |
| Engine support from matrix with sources | `[cited]` |
| Engine support from matrix without sources | `[unverified]` |
| Graceful degradation path (unknown architecture) | `[unknown]` |

**Do NOT:**
- Label a computed number `[verified]`. Even `bits/param` is `[inferred]`
  because it's a derivation.
- Show a green checkmark next to an `[unverified]` entry. The UI intentionally
  makes these loud.
- Introduce a new label without updating i18n keys and legend rendering.
