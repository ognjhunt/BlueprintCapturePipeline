# Production Pipeline Profile

The canonical non-secret environment template is
`configs/production_pipeline.env.example`. Load it once at the process entrypoint;
deployment-specific credentials belong in the secret store, not the profile.

## Product path

With the profile loaded, run the current capture product lane:

```bash
python -m blueprint_pipeline.capture_orchestrator \
  --descriptor-gcs-uri gs://BUCKET/scenes/SCENE/captures/CAPTURE/capture_descriptor.json \
  --lane current
```

This produces qualification, card/package preparation, simulation-automation
contracts, WebApp handoff state, and a run summary. Capture rights, consent,
privacy, provenance, and requested outputs must be present in the capture bundle;
the profile does not fabricate or bypass them.

To build a read-only fleet view from capture roots that contain those summaries:

```bash
python -m blueprint_pipeline.run_summary_aggregation \
  --root /mnt/gcs/BUCKET/scenes \
  --output /path/to/pipeline_fleet_run_summary.json
```

Aggregation fails closed if any discovered `pipeline/run_summary.json` is malformed.
Requested budgets and known GPU seconds are reported separately; missing GPU time is
unknown, not zero. The aggregate is operational observability, not provider teardown,
semantic-success, or ranking evidence.

For a real local MuJoCo Task Evaluation Run, use an explicit simulator command:

```bash
python -m blueprint_pipeline.robot_eval_job_orchestrator \
  --capture-root /mnt/gcs/BUCKET/scenes/SCENE/captures/CAPTURE \
  --job-request /path/to/robot_eval_job_request.json \
  --job-id RUN_ID \
  --provisioner fixture_local \
  --simulator mujoco \
  --allow-simulator mujoco \
  --simulator-command 'mujoco=python -m blueprint_pipeline.mujoco_g1_simulator_command' \
  --allow-simulator-execution
```

`BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`, the CLI allow flag, and the explicit
command are all required. A model-specific simulator is never selected implicitly.

## Hosted support rendering

`BLUEPRINT_NATIVE_RUNTIME_BACKEND=site_splat` is the provider-neutral default. It
uses capture-grounded truthful-preview rendering behind the stable runtime-service
contract. A legacy Cosmos-Predict2.5 deployment must explicitly select
`cosmos_wam` and configure `COSMOS_OFFICIAL_REPO_ROOT`; the runtime no longer
discovers model repositories from machine-specific workspace paths. The old
`NATIVE_WORLD_MODEL_SYNTHESIS_MODE=cosmos_i2w` setting remains a compatibility
alias, and conflicting old/new settings fail closed.
Capture-grounded `site_splat` deployments do not activate Cosmos refinement
from ambient model files; refinement requires an explicit
`NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT=true` opt-in.

## Paid resources

The base production profile deliberately leaves
`BLUEPRINT_ALLOW_GPU_PROVISIONING=false`. Paid CPU builders, model volumes, and GPU
canaries must be admitted through exactly these commands:

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build ARGS
python -m blueprint_pipeline.paid_resource_allocator model-volume ARGS
python -m blueprint_pipeline.paid_resource_allocator gpu-canary ARGS
```

Follow `docs/runbooks/groot-oscar-thin-release.md` for the required inventory,
budget, watchdog, teardown, and provider-absence evidence. Loading this profile is
not authorization to spend.

## Claim boundary

- A completed current lane proves artifact contracts, not physical robot readiness.
- A local MuJoCo run proves only the evidence emitted by that run.
- Provider startup is not semantic/ranking success.
- Paid work is incomplete until exact-attempt and global inventory show teardown.
- WebApp sync is not deployment approval or buyer acceptance.
