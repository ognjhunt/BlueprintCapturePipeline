# Pipeline Current Process Audit

Date: 2026-06-06

Status: cleanup pass applied to defaults, docs, and command surfaces.

## Current Supported Process

The active repo path is:

1. Capture App output is staged or materialized into a capture root.
2. Qualification prepares or verifies privacy-safe World Labs input. SAM3,
   VIP/depth, and DeepPrivacy2 are optional runner implementations; production
   handoff depends on an audited `privacy/final_walkthrough.*` or derivative.
3. World Labs API upload/request/operation/world manifests are persisted when `preview_simulation` or `preview` is requested and the input is ready.
4. Evaluation prep and robot-eval package artifacts are generated as downstream package support.
5. CPU/pre-GPU processing writes scene asset inspection, frame estimate, episode specs, CPU simulator fixture setup, and CPU preflight manifests.
6. Simulation automation writes a fail-closed manifest, proof boundary, GPU
   handoff packet, and owner-proof schema.
7. Owner GPU proof ingestion validates logs, scene/spawn/action traces, artifact
   manifest, and attestation when `gpu_owner_system_proof.json` exists.
8. Provider-preview QA and final handoff readiness validators summarize local proof and blockers.
9. Robot-eval job orchestration and Post-Training Data Package export write
   per-job evidence without upgrading readiness booleans.
10. The capture batch registry tracks retry/resume status per site/capture.
11. Real simulator execution runs only through explicit env and CLI gates.

`blueprint-capture-pipeline --lane current` and `--lane all` now expand to:

```text
qualification -> evaluation_prep -> simulation_automation
```

## Cleaned Up

- Removed the stale smoke-only runtime command from `docs/GPU_VM_RUNBOOK.md`.
- Updated the capture orchestrator so `current` and `all` no longer expand to legacy scene-memory/retrieval/frame-alignment/Cosmos lanes.
- Added `simulation_automation` as a first-class orchestrator lane.
- Updated materialization and capture-bridge requested-lane defaults so Capture App output now seeds the current package/CPU-preflight path.
- Stopped evaluation prep from auto-building legacy SimReady, Marble, and Cosmos export artifacts unless explicit legacy env gates are set.
- Removed legacy runtime/model/simulator-review console scripts from the package entrypoint list: native runtime service, video-to-world runner, Cosmos Vast trainer, SimReady builder, and Marble builder.
- Deleted the deprecated `scripts/run_site_world_runtime_local.py` launcher.
- Removed detailed `video_to_world` live-geometry setup from the main README and left it as legacy compatibility material.
- Updated staging and end-to-end CLI defaults to use `current`.
- Reworked README language so the active process is first and older runtime/model lanes are labeled legacy/advisory.
- Added provider-preview QA and production handoff readiness manifests as local proof surfaces.
- Added final-walkthrough lineage enforcement before World Labs provider-ready
  status can pass in production.
- Added owner GPU proof ingestion as evidence validation, not as robot-readiness
  proof.
- Added Post-Training Data Package export and site/capture batch registry
  command surfaces for production handoff tracking.

## Legacy Or Advisory Paths Still Present

These paths are still in the repo because tests, advisory artifacts, or proof-boundary surfaces still reference them:

- `scene_memory`, `retrieval_index`, `frame_alignment`, and `synthesis_coverage_validation`
- Cosmos feasibility, export, benchmark, and optional training helpers
- single-VM GPU/native runtime helper scripts
- SimReady and Marble bridge helpers
- city-launch/readiness harnesses
- native site-world runtime service tests and placeholder render paths

They are not part of the active default process. Delete them only in a separate pass that also removes or rewrites their tests, docs, console scripts, and WebApp-facing artifact fields.

## Removal Candidates For A Later Pass

- `scripts/bootstrap_cosmos_official_repo.sh`
- `scripts/start_native_runtime_vast.sh`
- `scripts/check_cosmos3_readiness.py`
- `src/blueprint_pipeline/synthesis/cosmos_*`
- `src/blueprint_pipeline/native_runtime_service.py`
- `src/blueprint_pipeline/simready_assets.py`
- `src/blueprint_pipeline/marble_sim_assets.py`
- console scripts for legacy Cosmos, native runtime, SimReady, and Marble helpers
- legacy test files that only pin the paths above

## Proof Boundary

Local CPU smoke can be displayed only as local CPU preflight. It must not be
treated as accepted simulator execution, physics/contact validation, safety
proof, policy success, robot readiness, deployment readiness, or training proof.

`provider_preview_qa_manifest.json` and
`production_handoff_readiness_manifest.json` are local proof summaries. A
`ready_except_owner_gpu_simulator_execution` handoff status still leaves
simulator execution, policy success, contact validation, safety validation, and
robot readiness false until owner-system evidence is supplied.
In production mode, the handoff readiness validator also requires real WebApp
upstream-link truth (`site_submission_id`, derived or explicit `request_id`,
`buyer_request_id`, and `capture_job_id`) from a succeeded WebApp sync before
owner GPU can be the only remaining unproven step. Local IDs or skipped sync
artifacts remain advisory and must not be used as production proof.
Provider-preview QA also requires the World Labs input to be
`privacy/final_walkthrough.*` or an audited derivative; a configured privacy
model runner is not by itself production proof.
