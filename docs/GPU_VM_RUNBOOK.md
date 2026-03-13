# GPU VM Runbook

This is the supported bring-up path for the narrowed NeoVerse workflow on one Linux GPU VM.

## Scope

Supported path:

- stage a fresh raw Blueprint capture bundle
- run qualification
- run evaluation-prep
- emit canonical site-world artifacts
- emit presentation manifests when enabled
- hand off `site_world_registration.json` to BlueprintValidation preflight

Out of scope for this runbook:

- Isaac Sim / Replicator bring-up
- broad multi-platform install support
- legacy stage-1 nurec / simulator branches

## VM assumptions

- Ubuntu-like Linux VM with an NVIDIA GPU
- working NVIDIA driver
- Python 3.10+
- outbound network access for Python packages and model downloads during bootstrap

## Bootstrap

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_ml_stack.sh
python3 scripts/setup_environment.py --check
```

Default bootstrap installs the supported runtime path:

- CUDA PyTorch
- repo package plus `runtime` dependencies
- ffmpeg and base system tools
- YOLO-World runtime dependencies

Optional installs:

- `./scripts/install_ml_stack.sh --with-sam3`
- `./scripts/install_ml_stack.sh --with-da3`
- `./scripts/install_ml_stack.sh --with-fixer`
- `./scripts/install_ml_stack.sh --with-local-qwen`

SAM3 is optional. If it is not installed or `SAM3_WEIGHTS_PATH` does not point to weights, the SAM3 backend is skipped explicitly and the supported object-index path remains YOLO-World plus Grounding-DINO fallback.

## Environment variables

Required for downstream NeoVerse handoff:

```bash
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"
```

Optional:

```bash
export NEOVERSE_RUNTIME_SERVICE_API_KEY=""
export NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS="120"
export WORLD_MODEL_EMIT_PRESENTATION="true"
export WORLD_MODEL_ALLOW_GENERATIVE_COMPLETION="limited"
export HF_HOME="/opt/hf"
export SAM3_WEIGHTS_PATH="/opt/sam3_weights/sam3.pt"
```

## Stage and run

Stage a raw capture bundle into the supported local storage layout and run the narrowed pipeline:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /data/raw_bundle \
  --storage-root /data/blueprint-storage \
  --bucket local-blueprint \
  --copy \
  --run-qualification \
  --run-evaluation-prep
```

Supported layout after staging:

```text
<storage-root>/<bucket>/scenes/<scene>/captures/<capture>/
```

## Expected outputs

Canonical outputs:

- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/site_world_registration.json`
- `pipeline/evaluation_prep/site_world_health.json`
- `pipeline/evaluation_prep/evaluation_prep_manifest.json`

Presentation outputs when `WORLD_MODEL_EMIT_PRESENTATION=true`:

- `pipeline/presentation_world/presentation_world_manifest.json`
- `pipeline/presentation_world/runtime_demo_manifest.json`

Useful diagnostics:

- `raw/object_index_build_report.json`
- `pipeline/qualification_record.json`
- `pipeline/evaluation_prep/object_geometry_manifest.json`
- `pipeline/evaluation_prep/evaluation_prep_summary.json`

## BlueprintValidation handoff

From the BlueprintValidation repo:

```bash
cd /Users/nijelhunt_1/workspace/BlueprintValidation
export NEOVERSE_RUNTIME_SERVICE_URL="http://127.0.0.1:8787"

blueprint-validate --config configs/example_validation.yaml preflight \
  --site-world-registration /data/blueprint-storage/local-blueprint/scenes/<scene>/captures/<capture>/pipeline/evaluation_prep/site_world_registration.json
```

## Failure diagnosis

If object indexing is empty:

- inspect `raw/object_index_build_report.json`
- check `backend_summary.providers[*].status`
- check `backend_summary.providers[*].reason`
- check `empty_index_cause`

Common blockers:

- `ultralytics_missing:...`
- `sam3_not_installed`
- `sam3_weights_missing:...`
- `missing_runtime_service_url`
- `qualification_state:not_ready_yet`

If `site_world_health.json` is blocked:

- inspect `pipeline/evaluation_prep/site_world_health.json`
- inspect `pipeline/evaluation_prep/object_geometry_manifest.json`
- inspect `pipeline/qualification_record.json`

Canonical artifacts stay conservative:

- missing grounded objects blocks launchability
- missing evidence is surfaced as blockers or warnings
- presentation manifests never overwrite canonical truth

## Manual GPU smoke check

After bootstrap, you can validate real YOLO-World execution on a staged capture:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m blueprint_pipeline.object_index_stage \
  --capture-root /data/blueprint-storage/local-blueprint/scenes/<scene>/captures/<capture> \
  --force-rebuild
```

Success criteria:

- `raw/object_index_build_report.json` shows YOLO-World `status=ok` or Grounding-DINO fallback `status=ok`
- `raw/object_index.json` contains non-empty `objects` when visible manipulable objects exist
