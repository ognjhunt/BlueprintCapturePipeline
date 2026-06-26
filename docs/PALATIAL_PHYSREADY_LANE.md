# Palatial PhysReady Lane

Status: optional provider-support lane, disabled by default.

Owner repo: `BlueprintCapturePipeline`

## Purpose

The Palatial PhysReady lane prepares and optionally materializes task-critical
object twins from captured site content. It is intended for objects such as a
microwave, tote, bin, door handle, cabinet, drawer, cart, or appliance control
that matter for contact-rich robot evaluation tasks.

The lane keeps Palatial behind a replaceable provider contract. Palatial outputs
are model-derived support assets, not raw capture truth and not rank-fidelity
proof.

## Default Behavior

Default command:

```bash
blueprint-build-palatial-physready \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Default behavior is local-only:

- reads capture/package object and task artifacts
- selects task-critical twin candidates
- creates prompts and Palatial request payload metadata
- estimates token cost
- records image/source lineage
- writes proof-boundary manifests
- does not call Palatial
- does not upload scan/capture images
- does not download remote assets
- does not run Isaac Sim, MuJoCo, PyBullet, or robot policies

## Easy On/Off Switch

Live Palatial calls require both an environment gate and a CLI flag:

```bash
BLUEPRINT_ENABLE_PALATIAL_PHYSREADY=true \
PALATIAL_API_KEY=<secret> \
blueprint-build-palatial-physready \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --allow-live-palatial
```

Disable by unsetting `BLUEPRINT_ENABLE_PALATIAL_PHYSREADY` or omitting
`--allow-live-palatial`.

Optional API overrides:

```bash
PALATIAL_GENERATE_URL=https://dashboard.palatial.cloud/api/v1/external/generate
PALATIAL_AUTH_MODE=x-api-key
```

Use `PALATIAL_AUTH_MODE=bearer` only after a real Palatial key smoke test proves
the workspace expects bearer auth.

## Inputs

The lane reads local capture/package artifacts:

- `capture_descriptor.json`
- `raw/manifest.json`
- `pipeline/evaluation_prep/object_geometry_manifest.json`
- `pipeline/evaluation_prep/task_anchor_manifest.json`

It can also take direct selection flags:

```bash
blueprint-build-palatial-physready \
  --capture-root /path/to/capture \
  --label microwave \
  --label tote
```

or:

```bash
blueprint-build-palatial-physready \
  --capture-root /path/to/capture \
  --object-id microwave_001 \
  --object-id blue_tote_001
```

Image references are taken from object-specific fields such as `reference_images`,
`image_paths`, `crop_paths`, `thumbnail_path`, `reference_frame_uri`,
`visual_replacement_masks`, and provenance fields. Use
`--include-capture-image-fallback` only when object-specific crop/image refs are
missing and the operator deliberately wants raw capture images included in the
request plan.

## Outputs

The lane writes:

```text
pipeline/palatial_physready/
  twin_candidate_manifest.json
  palatial_request_manifest.json
  palatial_physready_run_manifest.json
  materialization_manifest.json
  validation_manifest.json
  assets/<candidate_id>/*
```

`twin_candidate_manifest.json` records selected objects, prompts, source image
lineage, task context, scale hints, and desired articulation.

`palatial_request_manifest.json` records planned API request payloads, provider
endpoint/auth shape, cost estimate, and the live execution gate.

`palatial_physready_run_manifest.json` records whether live calls were allowed or
performed, whether remote downloads happened, and the claim boundary.

`materialization_manifest.json` records provider response export URLs or local
provider-response refs and local checksums when exports are materialized.

`validation_manifest.json` runs CPU metadata inspection for materialized USD,
OBJ, GLB/GLTF, MJCF/XML, and URDF files through the same inspection logic used
by scene-asset preflight. This is review evidence only.

## Provider Response Materialization

If Palatial returns a response out of band, materialize it without enabling live
API submission:

```bash
blueprint-build-palatial-physready \
  --capture-root /path/to/capture \
  --provider-response /path/to/palatial-response.json
```

Use `--download-exports` only when remote export URLs should be fetched:

```bash
blueprint-build-palatial-physready \
  --capture-root /path/to/capture \
  --provider-response /path/to/palatial-response.json \
  --download-exports
```

Supported response fields include nested `exports`, `download_url`,
`download_urls`, `export_url`, `export_urls`, `files`, `assets`, `url`, `uri`, or
`path` values that end in `.usd`, `.usda`, `.usdc`, `.mjcf`, `.xml`, `.urdf`,
`.obj`, `.glb`, `.gltf`, `.zip`, or `.json`.

## Claim Boundary

Allowed claim:

- Blueprint created or materialized Palatial PhysReady support artifacts for
  review from capture-derived object/task evidence.

Blocked claims unless owner-system proof exists:

- The Palatial asset is raw capture truth.
- The asset has loaded successfully in Isaac Sim, MuJoCo, or PyBullet.
- Physics/contact behavior is valid.
- Robot policy execution succeeded.
- A robot team can deploy against this asset without further validation.

Promotion into a Task Evaluation Run requires local checksums, rights/license
review, unit/scale sanity checks, collision/articulation metadata inspection,
and owner-system simulator proof before any rank-fidelity claim.
