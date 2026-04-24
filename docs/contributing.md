# Contributing to llm-cal

---

## Dev setup

```bash
git clone https://github.com/FlyTOmeLight/llm-cal.git
cd llm-cal
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# one-time: install pre-commit hooks
pre-commit install
```

Run the full verification loop:

```bash
ruff format src tests          # auto-format
ruff check src tests            # lint
mypy src                        # type-check (strict mode)
pytest -q                       # tests (must be 100% passing)
```

All of these are gates for PR merge.

---

## What to work on

Welcome contributions, roughly in order of value:

### 1. Data updates (no code, easiest)

- **New GPUs** — append to `src/llm_cal/hardware/gpu_database.yaml` and add
  one test in `tests/test_hardware.py`.
- **New engine compat entries** — append to
  `src/llm_cal/engine_compat/matrix.yaml` with `sources[]`. Honest
  `verification_level` required (see below).
- **`verified` matrix entries** — if you have real hardware and actually ran
  a config, PR the result with `type: tested` sources (hardware, date,
  metrics). This is the most valuable kind of contribution because v0.1
  ships with zero verified entries.

### 2. New architectures

See [Architecture Guide](architecture-guide.md) for the 10-step
checklist. Typically touches:

- Fixture (1 file)
- `detector.py` KNOWN_MODEL_TYPES
- `traits.py` (only if new attention variant)
- `formulas/kv_cache.py` (only if new KV behavior)
- Tests (detector + formulas)
- `matrix.yaml` entry

### 3. i18n

New locale = extend `src/llm_cal/common/i18n.py` with translations. Every key
needs an entry. Test in `tests/test_i18n.py`.

### 4. New sources

ModelScope SDK decision pending — see `docs/adr/001-modelscope-integration-strategy.md`.
Future sources (local directories, custom registries) go in
`src/llm_cal/model_source/`.

---

## The two hard rules

### Rule 1: Label discipline

Every user-facing number passes through `AnnotatedValue[T]`, and the label must
match how the value was obtained. See the label table in
[Architecture Guide](architecture-guide.md).

**Violation of this rule is the only reason a PR will be rejected outright.**
The whole point of the tool is that users can trust what the labels mean.

### Rule 2: Honest verification levels

`engine_compat/matrix.yaml` entries:
- `verified` — only with `type: tested` sources containing real metrics
- `cited` — requires at least one URL with `captured_date`
- `unverified` — allowed but surfaces loudly in the UI

Don't "upgrade" a `cited` to `verified` without actually running hardware.

---

## Code style

- Python 3.11+ (for `enum.StrEnum`, structural-pattern match where useful)
- Ruff-formatted, line length 100
- mypy strict mode, no `# type: ignore` without a reason comment
- Pydantic v2 for all YAML schemas
- Keep `cli.py` thin (< 60 lines) — orchestration belongs in `core/evaluator.py`

Chinese punctuation is intentional in i18n / Chinese-facing error messages.
If you add a file with Chinese content, add the path to the ruff
`per-file-ignores` whitelist in `pyproject.toml` for RUF001/RUF002/RUF003.

---

## Commit messages

Conventional-commits style preferred:

- `feat(scope): ...` — new capability
- `fix(scope): ...` — bug fix
- `docs: ...` — doc-only changes
- `test: ...` — test-only changes
- `chore: ...` — tooling / build

Good scope names match top-level modules: `architecture`, `fleet`,
`engine_compat`, `output`, `formulas`, `model_source`, `i18n`.

---

## Testing philosophy

- **Critical regression tests** are marked with a `CRITICAL:` docstring prefix
  and **must never be weakened or removed**. Examples:
  - `test_csa_hca_length_mismatch_fallthrough` (detection guard)
  - `test_fp4_fp8_pack_identified` (tool's core value prop)
  - `test_commit_sha_mismatch_invalidates` (cache correctness)
  - `test_tp_divisibility_constraint` (fleet correctness)
  - `test_unverified_match_shows_warning_in_output` (honesty constraint)

- **Fixture reuse over mocks** for config-based tests. Real model `config.json`
  samples live in `tests/fixtures/configs/`.

- **No network in CI tests.** Use `responses` library fixtures (seeded by
  `scripts/capture-fixtures.py` — planned for v0.1 finalization).

---

## Design doc

The v0.1 design doc lives outside the repo (in the author's gstack workspace).
Key decisions:

- Pass-through "traits composition" model (not inheritance) — lets new
  architectures combine freely
- Engine compat matrix as data, not code — community contributions are PRs
  against YAML, not Python
- Six-level labels, enforced by `enum.StrEnum`
- TP-aware KV sharding (matches vLLM behavior, not naive replication)

The trade-offs and rejected alternatives (thin-aggregator approach, data-driven
community registry without core) are documented in the design doc.

---

## Questions

Open an issue. Maintainer replies within a week (probably faster).
