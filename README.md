# BlueprintCapturePipeline

`BlueprintCapturePipeline` is the packaging, trust, and runtime service that turns raw Blueprint captures into site-specific world-model products with provenance, privacy, and rights safety.

For non-ARKit captures, the canonical world-model packaging path remains internal: `BlueprintCapture` evidence -> support/trust analysis -> privacy-aware geometry staging -> retrieval memory -> alignment -> synthesis/Cosmos conditioning. The default hosted preview path is now World Labs Marble from the privacy-safe walkthrough, while scene-memory, presentation, evaluation-prep, and runtime registration remain downstream derived lanes.

For public language, Google/Meta smart glasses are supported only for approved repeat walkthroughs where the assignment, hardware, launch proof, and downstream capture/package proof exist. This repo treats glasses outputs as partial/internal until that proof chain exists.

AI and engineer orientation maps live under [`docs/architecture/`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture):

- [`ai-onboarding-map.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/ai-onboarding-map.md)
- [`source-of-truth-map.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/source-of-truth-map.md)
- [`command-safety-matrix.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/command-safety-matrix.md)
- [`refactor-hotspots.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/refactor-hotspots.md)

## Scope

Primary product path:

- raw capture materialization from `BlueprintCapture`
- Gemini-backed multimodal capture review
- capture evidence analysis and agent review
- deterministic QA aggregation and trust/provenance assembly
- world-model fit scoring and capturer payout recommendation
- optional provider preview routing
- privacy-safe World Labs input preparation
- webapp sync for buyer-review surfaces
- deterministic object indexing and scene semantics when deeper work is requested
- optional scene-memory assembly
- optional presentation-world assembly
- optional evaluation-prep packaging
- optional runtime registration support for the built site-world package

Support / trust alpha artifacts:

- `qualification_summary.json`
- `capture_quality_summary.json`
- `rights_and_compliance_summary.json`
- `buyer_trust_score.json`
- `world_model_fit_summary.json`
- `capturer_payout_recommendation.json`
- `recapture_requirements.json`
- `provider_preview_status.json`
- `provenance_summary.json`
- `gemini_capture_fidelity_review.json`

Optional legacy downstream artifacts:

- `scene_memory/*`
- `presentation_world/presentation_bundle.json`
- `presentation_world/presentation_world_manifest.json`
- `presentation_world/runtime_demo_manifest.json`
- `evaluation_prep/site_world_spec.json`
- `evaluation_prep/site_world_registration.json`
- `evaluation_prep/site_world_health.json`
- `evaluation_prep/evaluation_prep_manifest.json`

## Local Development

```bash
uv sync --extra dev
```

This is a repository development setup only. It is not the supported single-VM GPU runtime bootstrap path.

Optional LLM support for the capture review agent:

```bash
uv sync --extra dev --extra llm
```

Local tests automatically add `src/` and the sibling `BlueprintContracts/src` to `sys.path` through [`tests/conftest.py`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/tests/conftest.py). If the contracts repo is not present beside this repo, install `blueprint-contracts` before running `pytest`.

Cross-repo external alpha gate:

```bash
python scripts/run_external_alpha_launch_gate.py
```

## Privacy Runner Services

The production preview path expects URL-first privacy runners:

- `PRIVACY_SAM3_URL`
- `PRIVACY_VIP_URL`
- `PRIVACY_DEPTH_ANYTHING_URL` (optional; otherwise `vip-inpaint` handles depth-only requests)
- `PRIVACY_DEEPPRIVACY2_URL`
- `PRIVACY_RUNNER_TOKEN`

For temporary internal demos, `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true` allows the World Labs preview path to fall back to the raw walkthrough video when privacy processing is unavailable. The bypass path is intentionally labeled as non-production and unredacted, and the input video is auto-trimmed/compressed to World Labs upload limits before submission.

The non-ARKit geometry path expects a dedicated GPU `video_to_world` runner. This
is the only path that can mark non-ARKit geometry as live world-model-ready:

- `VIDEO_TO_WORLD_URL`
- `VIDEO_TO_WORLD_RUNNER_TOKEN`
- `VIDEO_TO_WORLD_PIPELINE_PRESET` or `VIDEO_TO_WORLD_COMMAND_TEMPLATE`

The production deployment should use four GPU Cloud Run services:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`
- `video-to-world`

Recommended `video_to_world` presets:

- `preprocess_only` for DA3-only geometry bootstrap
- `preprocess_plus_alignment` for DA3 + non-rigid alignment outputs. This is the default deployment preset.
- `full_fast` for the end-to-end upstream reconstruction path with the lighter preset
- `full_extensive` for the full upstream path including global optimization and longer inverse-deformation / GS stages

If the runner is missing or fails, the geometry stage may write an explicitly
labeled internal fallback so local tests and contract-shape debugging can continue.
Fallback geometry is machine-readable as `geometry_source=fallback_geometry` and
`fallback_used=true`; it must remain `ready_for_world_model=false`,
`geometry_live_ready=false`, and `site_faithful_market_ready=false`.

`RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO=true` is now the default production expectation. Retrieval indexing fails closed unless it can resolve `world_model_video_uri`, `privacy_processed_video_uri`, or the concrete privacy artifact at `privacy/final_walkthrough.mov` / `privacy/final_walkthrough.mp4`.

The main `blueprint-pipeline` job stays CPU-only. The concrete service contract, storage behavior, and model-path rules are documented in [docs/PRIVACY_RUNNER_SERVICES.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/PRIVACY_RUNNER_SERVICES.md).

Live geometry validation command:

```bash
VIDEO_TO_WORLD_URL=https://<video-to-world-runner> \
VIDEO_TO_WORLD_RUNNER_TOKEN=<secret> \
VIDEO_TO_WORLD_PIPELINE_PRESET=preprocess_plus_alignment \
python3 scripts/run_geometry_lane.py \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider video_to_world \
  --model video_to_world-default
```

Before claiming live proof, inspect `pipeline/geometry/geometry_summary.json` and
`pipeline/geometry/logs/provider_result.json` for `geometry_source=video_to_world`,
`fallback_used=false`, `provider_native_result=true`, `ready_for_world_model=true`,
and `geometry_live_ready=true`.

The privacy path now treats depth generation as a first-class step:

- use ARKit depth/confidence when available
- otherwise run Depth Anything 3 for every non-ARKit capture, including glasses captures, even if no humans are detected
- persist the resulting depth and confidence manifests for downstream grounding
- pass those manifests into VIP so non-ARKit inpainting reuses the generated depth artifacts

## Local GPU Bring-Up

The older single-VM GPU runbook is still available for legacy downstream world-model work in [docs/GPU_VM_RUNBOOK.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/GPU_VM_RUNBOOK.md), but it is not the active preview path.

For privacy-service bring-up, use the service images under [`deploy/docker/`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/deploy/docker) and the Terraform stack under [`deploy/terraform/main.tf`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/deploy/terraform/main.tf).

The local repo bootstrap remains:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_ml_stack.sh
python3 scripts/setup_environment.py --check
```

Then stage and run:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /data/raw_bundle \
  --storage-root /data/blueprint-storage \
  --bucket local-blueprint \
  --copy \
  --run-qualification
```

## Entry Points

Pipeline lanes:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane qualification
```

Raw bundle staging:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /path/to/raw-download-folder \
  --storage-root /mnt/blueprint-storage \
  --bucket local-blueprint \
  --link \
  --run-qualification
```

Qualification agent review:

```bash
blueprint-agent-review \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider openai
```

Optional agent-review wrapper:

```bash
blueprint-run-e2e \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider openai
```

Optional legacy scene-memory build:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane scene_memory
```

Deeper local staging lanes can be requested through `scripts/stage_capture_bundle.py`
with `--pipeline-lane retrieval_index`, `frame_alignment`, `evaluation_prep`,
`synthesis_coverage_validation`, `cosmos_single_capture_smoke`, or `all` when
`--run-qualification` is set. These lanes still honor geometry/provider truth
and will not promote fallback geometry into live `video_to_world` proof.

Object index build:

```bash
blueprint-build-object-index \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Evaluation prep build:

```bash
blueprint-build-evaluation-prep \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider manual
```

## Contract Boundary

Shared contract code lives in `BlueprintContracts`:

- `handoff_contract`
- `site_world_contract`
- `runtime_layer_contract`
- `canonical_package`

The bridge contract for this repo is documented in [`docs/CAPTURE_BRIDGE_CONTRACT.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/CAPTURE_BRIDGE_CONTRACT.md).

Current cross-repo implementation status is tracked in [`docs/READINESS_MATRIX.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/READINESS_MATRIX.md). It is intentionally strict about what is shipped in-repo versus what still depends on live GPU/runtime/model access.
