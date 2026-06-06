# Simready Asset Lane

Status: local deterministic review lane.

Owner repo: `BlueprintCapturePipeline`

## Purpose

The simready asset lane converts existing capture-derived package evidence into
explicit simulator-review artifacts for Isaac Sim, MuJoCo, and PyBullet.

It does not run simulators, call live providers, download model checkpoints,
download robot assets, or claim robot readiness.

World Labs / Marble-specific SPZ, PLY, collider mesh GLB, and semantics
metadata handoff is handled by the sibling
[`docs/MARBLE_SIM_ASSET_HANDOFF.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/MARBLE_SIM_ASSET_HANDOFF.md)
lane.

## Inputs

The lane reads only local capture/package artifacts:

- `capture_descriptor.json`
- `raw/manifest.json`
- `pipeline/geometry/geometry_summary.json`
- `pipeline/evaluation_prep/object_geometry_manifest.json`
- `pipeline/evaluation_prep/task_anchor_manifest.json`
- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/hosted_session_runtime_manifest.json`
- `sites/{site_id}/reference_memory/site_reference_manifest.json`
- `sites/{site_id}/reference_memory/site_reference_index.jsonl`

Site-reference memory is summarized. Dense frames, depth maps, confidence maps,
embeddings, thumbnails, and full per-reference rows are not copied into the
simready summary.

## Outputs

The direct CLI and evaluation-prep hook write:

```text
pipeline/simready/
  simready_scene_manifest.json
  simready_validation.json
  framework_review_manifest.json
  evidence_boundaries.json
  site_reference_summary.json
  task_scenarios.json
  robot_profiles.json
  isaac_sim/site_scene.usda
  mujoco/site_scene.xml
  pybullet/site_scene.urdf
```

Evaluation prep also writes:

```text
pipeline/evaluation_prep/simready_prep_manifest.json
```

and includes `simready_prep_manifest` in
`evaluation_prep_manifest.json.artifacts`.

## Claim Boundary

Allowed claim:

- Blueprint emitted local simulator-review artifacts from capture/package
  evidence.

Blocked claims unless owner-system proof exists:

- Isaac Sim, MuJoCo, or PyBullet execution completed.
- The scene loaded successfully in any simulator.
- Physics/contact behavior is valid.
- Robot policy execution succeeded.
- A robot team profile or robot asset is ready for deployment.
- A real robot or accepted simulator trial passed.
- A live provider/runtime/model backend generated or validated the result.

Robot readiness requires real simulator load traces, action logs, physics/contact
validation logs, robot-team-owned robot assets, and accepted simulator or robot
trial evidence.

## Legacy Module Command

Direct local build:

```bash
PYTHONPATH=src python -m blueprint_pipeline.simready_assets --capture-root /path/to/capture-root
```

Evaluation prep build:

```bash
blueprint-build-evaluation-prep \
  --capture-root /path/to/capture-root \
  --provider manual
```

Both commands are local artifact writers. The package no longer installs a
top-level `blueprint-build-simready-assets` console script because this is a
legacy advisory path outside the current Capture App -> World Labs -> CPU
preflight -> simulation-manifest flow. They should not be treated as live
provider, simulator-execution, or robot-readiness proof.

## Review Notes

- `isaac_sim/site_scene.usda` is an OpenUSD/USDA review scene with box collision
  proxies and Blueprint claim-boundary metadata.
- `mujoco/site_scene.xml` is an MJCF review model with static box proxies.
- `pybullet/site_scene.urdf` is a URDF review model with fixed links for static
  scene proxies.
- These files intentionally use coarse review proxies when no simulator-owned
  robot assets or validated collision meshes are available.
