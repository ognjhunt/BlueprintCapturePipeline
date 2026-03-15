# BlueprintCapturePipeline

`BlueprintCapturePipeline` is the qualification, provenance, and provider-routing service for raw Blueprint captures.

For alpha, the default output is a qualification bundle with trust, rights/compliance, recapture, and optional provider-preview state. Scene-memory, presentation, and evaluation-prep artifacts remain available as downstream derived lanes.

## Scope

Primary product path:

- raw capture materialization
- qualification analysis and agent review
- deterministic QA aggregation and trust/provenance assembly
- optional provider preview routing
- webapp sync for buyer-review surfaces
- deterministic object indexing and scene semantics when deeper work is requested
- scene-memory assembly
- presentation-world assembly
- evaluation-prep packaging
- optional runtime registration support for the built site-world package

Retained public artifacts:

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
  --run-qualification \
  --run-evaluation-prep
```

## Entry Points

Pipeline lanes:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane evaluation_prep
```

Raw bundle staging:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /path/to/raw-download-folder \
  --storage-root /mnt/blueprint-storage \
  --bucket local-blueprint \
  --link \
  --run-qualification \
  --run-evaluation-prep
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
  --provider openai \
  --run-evaluation-prep
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
