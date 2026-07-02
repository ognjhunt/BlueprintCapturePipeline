# SPEC-01: Stop fabricating geometry (fallback + `local_sfm` relabel)

- Status: Proposed
- Priority: **P0 — launch blocker**
- Area: `src/blueprint_pipeline/geometry_stage.py`, `geometry_sources.py`
- Doctrine: "raw capture, timestamps, poses, device metadata, and provenance are authoritative"; no fabricated operational states.

## Problem

When the real geometry provider fails, the pipeline manufactures a complete camera model
and depth stack out of thin air, and one code path then relabels that synthetic output as
real SfM and clears the fallback provenance flag.

1. `_build_fallback_provider_result` (`geometry_stage.py:640-724`) writes:
   - hard-coded intrinsics `fx = fy = max(width, height)`, `cx = width/2`, `cy = height/2` (`:707-715`)
   - placeholder poses translating only along +x with identity rotation (`:681-692`)
   - constant depth `1.5 + idx*0.05` m (`:666`) and constant confidence `0.75` (`:667`)

   These fabricated arrays are written to `depth/`, `confidence/`, `camera/intrinsics.json`,
   and `poses.jsonl` and are consumed downstream by scene-memory and retrieval. The
   `fallback_used=True` flag and `launch_blockers` entries (`:1297-1301`) are the only
   mitigations.

2. `_build_local_sfm_provider_result` (`geometry_stage.py:727-778`) calls the same
   fabricator (`:734-743`) and then sets `fallback_used = False` (`:761`) and
   `geometry_source = "local_sfm"` (`:759`). Identical synthetic tensors are now presented
   as "local_sfm_offline" output. Downstream consumers keying on `fallback_used`
   (`geometry_stage.py:385-391`, `:1373`; `geometry_sources.py:251-257`) will treat
   fabricated geometry as genuine.

3. Provenance flags are mutable: a later stage can reset `fallback_used`, so synthetic
   provenance is not append-only.

## Why this blocks beta

Buyers pay for evaluation runs and data packages grounded in *real* site geometry.
Fabricated depth/pose/intrinsics flowing into retrieval, scene memory, Cosmos training
export, or eval artifacts silently corrupts every downstream product and violates the raw
capture truth hierarchy. The `local_sfm` relabel is the worst case: it actively hides the
fabrication from provenance consumers.

## Proposed fix

1. On provider failure, emit a **blocked geometry artifact** (status file + launch
   blocker), not synthetic tensors. No depth/pose/intrinsics files should be written that
   did not come from a real provider or a real local algorithm.
2. Delete the fabricated-tensor path inside `_build_local_sfm_provider_result`. If a real
   offline SfM backend (e.g. COLMAP/GLOMAP) is intended, integrate it explicitly; until it
   exists, `local_sfm` must be reported as `unavailable`, never fabricated.
3. Make synthetic/fallback provenance **append-only**: once `fallback_used`,
   `synthetic_geometry`, or an equivalent flag is set true by any stage, no later stage may
   reset it. Enforce in `geometry_sources.py` when re-reading `geometry_summary.json`.
4. If a dev/test mode genuinely needs placeholder geometry, gate it behind an explicit
   env (`BLUEPRINT_ALLOW_SYNTHETIC_GEOMETRY=1`), stamp every artifact with
   `synthetic=true`, and hard-block package/eval export when present.

## Acceptance criteria

- [ ] Provider failure produces zero depth/pose/intrinsics files; run status is `blocked_geometry_unavailable`.
- [ ] `local_sfm` path either runs a real SfM implementation or reports unavailable; it can never emit tensors from `_build_fallback_provider_result`.
- [ ] A regression test asserts `fallback_used`/synthetic flags cannot transition true→false across stages.
- [ ] Package/eval export refuses inputs carrying synthetic-geometry flags unless the explicit dev env is set, and stamps the output manifest when it is.
- [ ] Existing e2e (`python -m blueprint_pipeline.run_e2e`) passes with the real-provider path.
