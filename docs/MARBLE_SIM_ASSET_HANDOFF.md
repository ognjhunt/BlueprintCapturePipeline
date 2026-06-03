# Marble Sim-Asset Handoff

Status: local deterministic review lane.

Owner repo: `BlueprintCapturePipeline`

## Purpose

The Marble sim-asset handoff turns a persisted World Labs / Marble world
manifest into local simulator-review packets for Isaac Sim, MuJoCo, and
PyBullet.

It does not call World Labs, download Marble assets, run Isaac Sim, run MuJoCo,
run PyBullet, convert SPZ/PLY/USD files, or claim robot readiness.

## Inputs

The lane reads persisted local artifacts:

- `pipeline/provider_run_manifest.json`
- `pipeline/worldlabs_request_manifest.json`
- `pipeline/worldlabs_operation_manifest.json`
- `pipeline/worldlabs_world_manifest.json`
- optional local conversion/export manifests such as
  `pipeline/marble_sim_assets/conversion_manifest.json` or
  `pipeline/worldlabs_export_manifest.json`

`pipeline/worldlabs_export_manifest.json` may describe assets downloaded or
generated from Marble outside the API response, including:

- exported splat PLY URLs or local paths
- exported collider mesh GLB URLs or local paths
- exported high-quality mesh GLB URLs or local paths

The default CLI is:

```bash
blueprint-build-marble-sim-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Use this when reviewing an explicit local world manifest:

```bash
blueprint-build-marble-sim-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --world-manifest /path/to/worldlabs_world_manifest.json
```

## Outputs

The lane writes:

```text
pipeline/marble_sim_assets/
  marble_asset_manifest.json
  marble_asset_validation.json
  marble_simready_bridge.json
  simulators/
    isaac_sim_review_manifest.json
    mujoco_review_manifest.json
    pybullet_review_manifest.json
```

When evaluation prep sees `pipeline/worldlabs_world_manifest.json`, it builds
the handoff and includes `marble_simready_bridge` and
`marble_asset_validation` in `evaluation_prep_manifest.json.artifacts`.

## World Labs Boundary

Current World Labs API world responses expose SPZ splat assets,
`assets.mesh.collider_mesh_url`, panorama imagery, and splat
`semantics_metadata` with `metric_scale_factor` and `ground_plane_offset`.

Marble's web/export workflow can produce PLY splats, collider mesh GLB, and
high-quality mesh GLB. The public API docs still say direct PLY retrieval via
API is not supported. This lane therefore treats PLY and high-quality GLB as
valid only when an explicit Marble export or local conversion manifest provides
them. Otherwise the Isaac Sim review manifest marks
`needs_conversion: spz_to_ply_or_usd`.

Collider mesh GLB is treated as a physics/collision review input. Missing
collider mesh blocks physics/collision readiness.

The robotics workflow described by World Labs and NVIDIA is:

```text
Marble world -> PLY splat + collider mesh GLB export ->
PLY-to-USD/USDZ conversion with NVIDIA tooling -> Isaac Sim import ->
robot assets and task setup -> simulator load/action/contact logs
```

MuJoCo / RoboSuite workflows similarly use the Marble collision mesh as scene
geometry, then add robot and task assets outside Marble.

## Simulator Boundaries

Allowed claim:

- Blueprint emitted local Marble simulator-review handoff artifacts from
  persisted World Labs manifests.

Blocked claims unless owner-system proof exists:

- Isaac Sim, MuJoCo, or PyBullet loaded the scene.
- Physics/contact behavior is valid.
- The collider mesh was converted into a valid MJCF, URDF, or USD scene.
- Robot policy execution succeeded.
- A real robot or accepted simulator trial passed.
- Articulated doors, drawers, fixtures, or tools are interaction-ready.

Robot readiness requires simulator load traces, action logs,
physics/contact validation logs, robot-team-owned robot assets, and accepted
simulator or real-robot trial evidence.

## Automatic Hook

`run_preview_provider()` builds the handoff after a World Labs provider run
persists a terminal `worldlabs_world_manifest.json` with a `world_id`.

The hook is local-only. It reads the just-written manifests and adds the
handoff paths into `provider_run_manifest.json.artifact_uris` without making a
new World Labs request.
