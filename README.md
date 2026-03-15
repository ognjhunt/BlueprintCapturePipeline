# BlueprintCapturePipeline

`BlueprintCapturePipeline` is the qualification, provenance, and provider-routing service for raw Blueprint captures.

For alpha, the default output is a WebApp-facing qualification record built from `BlueprintCapture` evidence, Gemini-backed multimodal fidelity review, deterministic rights/provenance normalization, and optional provider-preview state. Scene-memory, presentation, and evaluation-prep artifacts remain available as explicit downstream derived lanes.

## Scope

Primary product path:

- raw capture materialization from `BlueprintCapture`
- Gemini-backed multimodal capture review
- qualification analysis and agent review
- deterministic QA aggregation and trust/provenance assembly
- world-model fit scoring and capturer payout recommendation
- optional provider preview routing
- webapp sync for buyer-review surfaces
- deterministic object indexing and scene semantics when deeper work is requested
- optional scene-memory assembly
- optional presentation-world assembly
- optional evaluation-prep packaging
- optional runtime registration support for the built site-world package

Authoritative alpha artifacts:

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

Optional downstream artifacts:

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

Optional LLM support for the qualification agent:

```bash
uv sync --extra dev --extra llm
```

## Supported GPU VM Path

The supported runtime path is documented in [docs/GPU_VM_RUNBOOK.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/GPU_VM_RUNBOOK.md).

The short version is:

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

Optional downstream scene-memory build:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane scene_memory
```

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
