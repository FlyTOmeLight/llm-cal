# ADR-001: ModelScope Integration Strategy

**Status:** PROPOSED (spike pending)
**Date:** 2026-04-24
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

_Pending spike results. This ADR will be moved to ACCEPTED with the chosen option
once Week 0 validation completes. Criteria:_

1. Can we list repo files + sizes with ≤2 API calls?
2. Does the spike work under standard `MODELSCOPE_API_TOKEN` env?
3. Is the minimal install footprint ≤ 20 MB?

## Consequences

- Blocks `src/llm_cal/model_source/modelscope.py` implementation until merged.
- Dependency list in `pyproject.toml` may be adjusted downward if REST-only wins.
- Shape of the error-handling matrix (`docs/error-handling.md`) depends on which
  library throws which exceptions.
