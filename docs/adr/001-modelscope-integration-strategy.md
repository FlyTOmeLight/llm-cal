# ADR-001: ModelScope Integration Strategy

**Status:** ACCEPTED — Option B (REST via httpx)
**Date:** 2026-04-24 (proposed) / 2026-04-26 (accepted)
**Owner:** FlyTOmeLight

## Context

`llm-cal` must support both HuggingFace and ModelScope as model metadata sources.
HF is handled by the official `huggingface_hub` SDK, which is mature, well-typed,
and actively maintained. ModelScope presents a choice: use the official
`modelscope` Python SDK, or fall back to the ModelScope REST API via `httpx`.

The design doc (`/Users/moonlight/.gstack/projects/moonlight/moonlight-no-repo-design-20260424-152100.md`)
flags this as **Open Question #6** and blocks Week 1 model_source implementation
until the decision is made.

## Options

### Option A: Use the `modelscope` SDK
- **Pros:** Official client, handles auth + retry + file listing uniformly.
- **Cons:** Historically heavy (installs tf/torch deps by default, can be avoided
  with minimal-install variants). API may change across versions. Doc coverage
  for the "metadata only" use case is thin.

### Option B: Direct REST via `httpx`
- **Pros:** Zero heavy deps. Full control over request shape. Easy to mock.
- **Cons:** We own every endpoint path, auth header, pagination shape. ModelScope
  doesn't have a stable public OpenAPI spec to rely on.

### Option C: Hybrid (SDK for discovery, REST for fetch)
- Use the SDK only for `model_info()`-equivalent calls.
- Use REST for the actual `config.json` fetch.
- **Pros:** Best of both — correctness via SDK, transparency in hot path.
- **Cons:** Two dependencies, more surface area.

## Decision

**Option B: Direct REST via `httpx`.**

Rationale:

1. **Surface area is tiny.** `llm-cal` only needs three calls — list files,
   fetch `config.json`, Range-GET a safetensors header. The SDK's value (uniform
   download/upload) doesn't apply here.
2. **Install footprint stays flat.** `httpx` is already a hard dependency
   (used by `weight_analyzer/safetensors_reader.py`). Adding the `modelscope`
   SDK pulls torch/tf in some install paths, contradicting the calculator's
   "lightweight" pitch.
3. **Symmetry with HF hot path.** HF safetensors header reads are already
   `httpx.get` with a Range header. ModelScope mirrors that exactly — the
   redirect to OSS-signed URLs is transparent to httpx with `follow_redirects=True`.
4. **Spike acceptance criteria all met:**
   - ✅ Files + sizes in 1 call: `GET /api/v1/models/{owner}/{name}/repo/files?Recursive=true`
   - ✅ `MODELSCOPE_API_TOKEN` works as `Authorization: Bearer {token}`
   - ✅ Install footprint: 0 new deps (httpx was already required)

Endpoints used:
- `GET /api/v1/models/{model_id}` — model meta (best-effort `LatestSha`)
- `GET /api/v1/models/{model_id}/repo/files` — file tree + sizes
- `GET /api/v1/models/{model_id}/repo?FilePath=...&Revision=...` — raw file

Wrapped envelope `{Code, Message, Data, Success}` is parsed defensively in
`_extract_files()`, tolerant of both `Data.Files` and `Data` (list) shapes.

## Consequences

- `src/llm_cal/model_source/modelscope.py` is now a real implementation
  (~200 LOC), no longer a placeholder.
- `pyproject.toml` does NOT add the `modelscope` SDK as a dependency.
- Error mapping is symmetric with HF: 404 → `ModelNotFoundError`,
  401/403 → `AuthRequiredError`, 429/timeout → `SourceUnavailableError`.
- Web UI exposes the choice via a `Source · 来源` radio (HuggingFace / ModelScope).
- CLI exposes it via `--source huggingface|modelscope`.
- If ModelScope changes the wrapper field names (PascalCase has shifted in
  past versions), `_extract_files` is the single place to extend.
