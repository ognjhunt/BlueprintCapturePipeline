# Output Validation Checklist

Use this checklist before accepting a run as production-ready.

## 1) Required Artifacts (hard fail if missing)
- `pipeline/nurec/export_last.usdz`
- `pipeline/nurec/visual_mesh.glb`
- `pipeline/nurec/nvblox_mesh.ply`
- `pipeline/nurec/mesh_manifest.json`
- `pipeline/nurec/occupancy.bin`
- `pipeline/swap_quality_report.json`
- `pipeline/pipeline_summary.json`

Pass Rule: all files exist and are non-empty.

## 2) Quality Gate Status
- `swap_quality_report.json.status == "passed"`
- `assembly_gate` is `passed`
- In `full_required`: both `runtime_preflight_gate` and `advanced_quality_gate` are `passed`

Pass Rule: every required gate above is true.

## 3) Registration/Coverage Baseline
- Read `pipeline/nurec/capture_quality_report.json` when present.
- Recommended threshold:
  - SfM registered ratio `>= 0.80` (matches default `COLMAP_MIN_REGISTERED_RATIO`)
  - Hard reject below `0.75` (default retry floor)

Pass Rule: run is accepted only if coverage is at or above the selected floor.

## 4) Refinement Safety (when Stage 4.5/5 used)
- If `pipeline/nurec/refinement_quality_gate.json` exists, require gate status `passed`.
- Reject if sharpness or PSNR regressions exceed configured gate profile limits.

Pass Rule: no refinement rollback condition is triggered.

## 5) Scene Assembly Integrity
- `scene.usda` exists under `scenes/<scene_id>/usd/scene.usda`
- In `full_required`, scene must not be the standalone stub variant.

Pass Rule: assembled scene exists and is non-stub for strict mode.

## 6) Run-Level Sanity
- `full_pipeline/run_summary.json.runtime.status == "passed"`
- `full_pipeline/run_summary.json.failures` is empty, or all listed entries are acknowledged best-effort fallbacks.
- `full_pipeline/log_summary.json.errors` contains no unresolved fatal errors.

Pass Rule: runtime + logs do not show unresolved hard failures.

## Decision
1. `PASS`: all hard checks pass and no unresolved fatal quality issues.
2. `CONDITIONAL PASS`: best-effort fallback occurred but target deliverable did not require full downstream assembly.
3. `FAIL`: any hard check fails or quality gates reject output.
