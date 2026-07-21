# Legacy GPU VM Runbook

> **SUPERSEDED:** Historical single-VM workflow only. Do not use this as a paid
> resource launcher. Current allocation must go through
> `blueprint_pipeline.paid_resource_allocator` and the provider-specific runbook.

This is legacy reference material for the older single-VM site-world/Cosmos
workflow. It is not the supported path for the current Capture App -> World Labs
API -> upload -> CPU preflight -> simulation-manifest flow.

## Scope

Historical path:

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
source .env.vast.local   # if present; ignored local secrets file
# Optional: copy defaults from configs/native_runtime_vast.env.example into .env.vast.local
./scripts/install_ml_stack.sh
./scripts/bootstrap_cosmos_official_repo.sh
python3 scripts/setup_environment.py --check
```

Default bootstrap installs the supported runtime path:

- CUDA PyTorch
- repo package plus `runtime` dependencies
- official Cosmos runtime package plus LoRA-training support libraries
- ffmpeg and base system tools
- YOLO-World runtime dependencies
- helper scripts already present in this repo, including `scripts/sam3_detect.py`

Optional installs:

- `./scripts/install_ml_stack.sh --skip-cosmos`
- `./scripts/install_ml_stack.sh --with-sam3`
- `./scripts/install_ml_stack.sh --with-da3`
- `./scripts/install_ml_stack.sh --with-fixer`
- `./scripts/install_ml_stack.sh --with-local-qwen`

SAM3 is optional. If it is not installed or `SAM3_WEIGHTS_PATH` does not point to weights, the SAM3 backend is skipped explicitly and the supported object-index path remains YOLO-World plus Grounding-DINO fallback.

Splat Analyzer is also optional. It can enrich object indexing from local
World Labs/Marble `.ply` or `.spz` splats after visual assets have been
materialized, but it is a model-derived support path only. It may help propose
task objects, geometry hints, and advisory object relationships; it does not
prove raw capture truth, collision/contact, robot spawn validity, simulator
execution, policy execution, or physical readiness.

`install_colmap_cuda.sh` is not part of this narrowed path and is not required for the supported GPU VM bootstrap.

## Reusable WAM Perception Harness Image

For provider/harness runs, prefer a versioned image over reinstalling SAM3,
Depth Anything, YOLO pose, `transformers`, and `ultralytics` on every GPU:

```bash
blueprint-build-wam-perception-harness-gpu-image \
  --image-ref docker.io/nijelhunt/blueprint-wam-perception-harness:20260626-cu126
```

The generated build context writes:

- `Dockerfile.wam-perception-harness-gpu`
- `build_image.sh`
- `push_image.sh`
- `run_image_healthcheck.sh`
- `prepare_model_mounts.sh`
- `wam_perception_harness_gpu_image_manifest.json`

The image bakes the Blueprint WAM-derived harness code, real-provider probe,
sim-provider E2E runner, CUDA PyTorch, `transformers`, `ultralytics`, Depth
Anything V2 cache, and YOLO pose cache. It keeps SAM3 weights external by
default: mount or fetch them to `/models/sam3/sam3.pt` and set
`SAM3_WEIGHTS_PATH` only to that path. Registry, Hugging Face, DigitalOcean, and
object-store credentials must come from local secret files or provider-native
secrets; they must not be baked into the image or written into artifacts.

The image healthcheck verifies imports and a fixture-mode harness loop. It is a
runtime-readiness check only; it does not prove provider accuracy, sensor depth,
physical contact, off-scope validation, or generated-world rank fidelity.

## Environment variables

Required for downstream site-world runtime handoff:

```bash
export SITE_WORLD_RUNTIME_SERVICE_URL="http://127.0.0.1:8791"
export SITE_WORLD_RUNTIME_SERVICE_API_KEY="<nonempty-runtime-secret>"
```

Optional:

```bash
export SITE_WORLD_RUNTIME_SERVICE_TIMEOUT_SECONDS="120"
export WORLD_MODEL_EMIT_PRESENTATION="true"
export WORLD_MODEL_ALLOW_GENERATIVE_COMPLETION="limited"
export HF_HOME="/opt/hf"
export HF_TOKEN=""
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export NGC_API_KEY=""
export SAM3_WEIGHTS_PATH="/opt/sam3_weights/sam3.pt"
export SPLAT_ANALYZER_REPO="/opt/splat_analyzer"
# or:
export SPLAT_ANALYZER_RUN_LOCAL="/opt/splat_analyzer/run_local.py"
# or:
export SPLAT_ANALYZER_COMMAND='python /opt/splat_analyzer/run_local.py --ply {SPLAT_PATH} --prompt {PROMPT} --quality medium --job_dir {JOB_DIR}'
export COSMOS_MODEL_ID="nvidia/Cosmos-Predict2.5-2B"
export COSMOS_MODEL_REVISION="0d37c7498f54cee3c599d438d895a0a4a8608064"
export COSMOS_OFFICIAL_REPO_ROOT="$HOME/workspace/cosmos-predict2.5"
export COSMOS_OFFICIAL_REPO_REF="661da4774b0ca41d082a0ecbeb47550bcf07e03f"
export COSMOS_OFFICIAL_REPO_UV_EXTRA="cu128"
export COSMOS_WORKER_PYTHON_BIN="$COSMOS_OFFICIAL_REPO_ROOT/.venv/bin/python"
export COSMOS_DISABLE_GUARDRAILS="1"
export COSMOS_CHUNK_SIZE="33"
export COSMOS_CHUNK_OVERLAP="4"
export NATIVE_WORLD_MODEL_SYNTHESIS_MODE="cosmos_i2w"
export COSMOS_TRAINER_ENTRYPOINT="/opt/cosmos/train_lora.py"
export COSMOS_TRAINER_ENTRYPOINT_MODE="script"
export COSMOS_TRAINER_LAUNCHER="accelerate"
export COSMOS_TRAINER_NUM_PROCESSES="1"
export COSMOS_TRAINING_COMMAND="python -m blueprint_pipeline.synthesis.cosmos_vast_training_wrapper --trainer-config {trainer_config_path} --output-dir {output_dir} --export-manifest {export_manifest_path} --capture-root {capture_root} --paired-reference-target {paired_reference_target_path} --k-reference-conditioning {k_reference_conditioning_path} --train-val-split {train_val_split_path}"
export BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL=""
export BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL=""
```

The wrapper command above is the supported VM entrypoint for the LoRA lane. It standardizes dataset/env wiring and then delegates to the real trainer specified by `COSMOS_TRAINER_ENTRYPOINT`.

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
- `pipeline/presentation_demo_preflight_report.json`

Useful diagnostics:

- `raw/object_index_build_report.json`
- `pipeline/qualification_record.json`
- `pipeline/evaluation_prep/object_geometry_manifest.json`
- `pipeline/evaluation_prep/evaluation_prep_summary.json`

## BlueprintValidation handoff

From the BlueprintValidation repo:

```bash
cd $HOME/workspace/BlueprintValidation
uv sync --extra vision
export SITE_WORLD_RUNTIME_SERVICE_URL="http://127.0.0.1:8791"
export SITE_WORLD_RUNTIME_SERVICE_API_KEY="<same-nonempty-runtime-secret>"

cd $HOME/workspace/BlueprintCapturePipeline
source .venv/bin/activate
./scripts/start_native_runtime_vast.sh
```

The supported live-runtime path now depends on the tracked launcher and official repo bootstrap scripts rather than an ad hoc VM-local `/root/blueprint-storage/start_native_runtime.sh` file.

Use the production-only validation config in [`BlueprintValidation/configs/example_validation.yaml`](../../BlueprintValidation/configs/example_validation.yaml).

Run production preflight:

```bash
blueprint-validate --config configs/example_validation.yaml \
  --required-runtime-kind native_world_model \
  preflight \
  --site-world-registration /data/blueprint-storage/local-blueprint/scenes/<scene>/captures/<capture>/pipeline/evaluation_prep/site_world_registration.json
```

Minimal session flow against the production runtime:

```bash
blueprint-validate --config configs/example_validation.yaml \
  --required-runtime-kind native_world_model \
  session create \
  --session-id validation-session \
  --session-work-dir data/session-validation \
  --site-world-registration /data/blueprint-storage/local-blueprint/scenes/<scene>/captures/<capture>/pipeline/evaluation_prep/site_world_registration.json \
  --robot-profile-id mobile_manipulator_rgb_v1 \
  --task-id task-1 \
  --scenario-id scenario-default \
  --start-state-id start-default

blueprint-validate session reset \
  --session-id validation-session \
  --session-work-dir data/session-validation

blueprint-validate session step \
  --session-work-dir data/session-validation \
  --episode-id <episode-id> \
  --action-json '[0,0,0,0,0,0,0]'

blueprint-validate session export \
  --session-id validation-session \
  --session-work-dir data/session-validation
```

Retired smoke-only runtime branch:

The old smoke-only runtime command has been removed from the supported workflow.
Use the current capture -> World Labs -> CPU preflight -> simulation automation
manifest path instead, and treat this runbook as legacy GPU/runtime reference
material only.

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
- `missing_local_splat_asset`
- `splat_analyzer_command_not_configured`
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

## Presentation demo operator steps

Stage 6 requires a truthful presentation demo UI endpoint, not just the manifest files.

Before regenerating `runtime_demo_manifest.json` for a demo-capable run:

```bash
# 1. Launch the demo UI service on the GPU host
#    Example: operator-managed site-world demo UI or another truthful demo endpoint

# 2. Verify it really serves HTTP 200
curl -I "$BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL"

# 3. Export the URL(s) before rerunning qualification/evaluation-prep
export BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL="https://demo.example/internal"
export BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL="https://demo.example/public"

# 4. Regenerate the presentation manifest artifacts
python3 scripts/stage_capture_bundle.py \
  --source-bundle /data/raw_bundle \
  --storage-root /data/blueprint-storage \
  --bucket local-blueprint \
  --copy \
  --run-qualification \
  --run-evaluation-prep
```

If the URL is missing, `runtime_demo_manifest.json` may still exist, but stage 6 should be treated as blocked until the UI endpoint is real and reachable.

## Manual GPU smoke check

After bootstrap, you can validate real YOLO-World execution on a staged capture:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m blueprint_pipeline.object_index_stage \
  --capture-root /data/blueprint-storage/local-blueprint/scenes/<scene>/captures/<capture> \
  --force-rebuild
```

Success criteria:

- `raw/object_index_build_report.json` shows YOLO-World `status=ok`, Grounding-DINO fallback `status=ok`, or optional Splat Analyzer `status=ok`
- `raw/object_index.json` contains non-empty `objects` when visible manipulable objects exist
- optional Splat Analyzer relationships, if produced, appear in `raw/object_grounding_hints.json.scene_relationship_candidates` and remain review-required advisory candidates
