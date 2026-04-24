# 参与贡献

---

## 开发环境搭建

```bash
git clone https://github.com/FlyTOmeLight/llm-cal.git
cd llm-cal
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 一次性：安装 pre-commit hooks
pre-commit install
```

完整验证 loop：

```bash
ruff format src tests          # 自动格式化
ruff check src tests            # lint
mypy src                        # 类型检查（strict mode）
pytest -q                       # 测试（必须 100% 通过）
```

PR 合并前这些都是硬要求。

---

## 贡献方向（按价值排序）

### 1. 数据更新（无代码，最简单）

- **新 GPU** — 追加到 `src/llm_cal/hardware/gpu_database.yaml`，加一个测试到 `tests/test_hardware.py`。
- **新引擎兼容条目** — 追加到 `src/llm_cal/engine_compat/matrix.yaml`，必须带 `sources[]`。诚实标注 `verification_level`（见下）。
- **`verified` 矩阵条目** — 有真实硬件并实际跑过某个配置，PR 带 `type: tested` 来源（hardware、date、metrics）。**这是最有价值的贡献**，因为 v0.1 里零个 verified 条目。

### 2. 新架构

见[架构指南](architecture-guide.md)里的 10 步清单。一般涉及：

- Fixture（1 个文件）
- `detector.py` 的 KNOWN_MODEL_TYPES
- `traits.py`（仅当新 attention variant）
- `formulas/kv_cache.py`（仅当新 KV 行为）
- 测试（detector + formulas）
- `matrix.yaml` 条目

### 3. i18n

新语言 = 扩展 `src/llm_cal/common/i18n.py` 的翻译字典。每个 key 都要有条目。测试在 `tests/test_i18n.py`。

### 4. 新数据源

ModelScope SDK 决策待定——见 `docs/adr/001-modelscope-integration-strategy.md`。未来的数据源（本地目录、自定义 registry）放 `src/llm_cal/model_source/`。

---

## 两条硬规则

### 规则 1：标签纪律

每个面向用户的数字都要过 `AnnotatedValue[T]`，标签必须和数值来源匹配。见[架构指南](architecture-guide.md)的标签表。

**违反这条规则是 PR 被直接 reject 的唯一理由。** 工具的核心价值就是用户能信任标签的含义。

### 规则 2：诚实的 verification_level

`engine_compat/matrix.yaml` 条目：
- `verified` — 只有在有 `type: tested` 来源（含真实指标）时才能用
- `cited` — 至少一条 URL + `captured_date`
- `unverified` — 允许，但工具会在 UI 里大声提示

别在没实际跑过的情况下把 `cited` "升级"成 `verified`。

---

## 代码风格

- Python 3.11+（为了原生 `enum.StrEnum`、可能用 pattern match）
- Ruff 格式化，行宽 100
- mypy strict mode，`# type: ignore` 必须带 reason 注释
- Pydantic v2 用于所有 YAML schema
- `cli.py` 保持精简（< 60 行），orchestration 放 `core/evaluator.py`

中文标点在 i18n / 面向中文用户的错误消息里是故意的。如果你新加一个带中文内容的文件，记得把路径加到 `pyproject.toml` 的 ruff `per-file-ignores` 白名单里（RUF001/RUF002/RUF003）。

---

## Commit 规范

偏好 Conventional Commits 风格：

- `feat(scope): ...` — 新能力
- `fix(scope): ...` — 修 bug
- `docs: ...` — 仅文档改动
- `test: ...` — 仅测试改动
- `chore: ...` — tooling / 构建

合适的 scope 名对应顶层 module：`architecture`、`fleet`、`engine_compat`、`output`、`formulas`、`model_source`、`i18n`。

---

## 测试哲学

- **Critical regression 测试**在 docstring 前标 `CRITICAL:` 前缀，**永远不能被弱化或删除**。例如：
  - `test_csa_hca_length_mismatch_fallthrough`（检测器守卫）
  - `test_fp4_fp8_pack_identified`（工具核心卖点）
  - `test_commit_sha_mismatch_invalidates`（缓存正确性）
  - `test_tp_divisibility_constraint`（fleet 正确性）
  - `test_unverified_match_shows_warning_in_output`（诚实性约束）

- **优先 fixture 复用而非 mock** 做 config 相关测试。真实模型的 config.json 样本放 `tests/fixtures/configs/`。

- **CI 中禁止网络调用**。用 `responses` 库的 fixture 回放（`scripts/capture-fixtures.py` 负责采集——v0.1 收尾规划中）。

---

## 设计文档

v0.1 的设计文档存在仓库外（作者的 gstack workspace）。关键决策：

- Traits 组合模型（而非继承）——让新架构可以自由组合
- 引擎兼容矩阵作为数据，非代码——社区贡献是针对 YAML 的 PR，不是 Python
- 六级标签，由 `enum.StrEnum` 强制
- TP 感知的 KV 分摊（匹配 vLLM 行为，非朴素复制）

权衡和被拒绝的方案（薄壳聚合器、无 core 的纯社区数据库）都记录在设计文档里。

---

## 问题

开 issue 即可。维护者一周内回复（通常更快）。
