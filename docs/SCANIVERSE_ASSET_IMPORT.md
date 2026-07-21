# Scaniverse Asset Import Lane

Status: local, proof-bounded support-asset lane.

`blueprint-import-scaniverse-assets` stages Scaniverse exports into an existing
Blueprint capture root. It is for capturer/operator workflows where a site was
also processed through Scaniverse Web from a 360 video capture and exported as
USDZ, PLY, SPZ, GLB/GLTF, FBX, OBJ, or USD. It is an optional support lane, not
a replacement for BlueprintCapture raw bundle authority.

## Contract

The lane writes:

```text
pipeline/scaniverse_assets/
  scaniverse_import_manifest.json
  scaniverse_import_proof_boundary.json
  assets/<staged exports>
```

The import also requires a Blueprint sidecar manifest that ties the Scaniverse
export back to the assignment, scene, capture, rights/provenance review, source
360 video, operator, and scale notes. Recommended fields are
`blueprint_assignment_id`, `blueprint_scene_id`, `blueprint_capture_id`,
`scaniverse_site_id`, `scaniverse_scan_id`, `capture_hardware`,
`source_video_filename`, `metric_scale_calibrated`, `export_created_at`,
`export_performed_by`, and `rights_scope`.

Supported import files:

- USDZ for Isaac-oriented visual/package review
- PLY or SPZ Gaussian splats
- GLB/GLTF/FBX/OBJ mesh exports
- USD/USDA/USDC scene assets

Formats already supported by the CPU scene-asset preflight lane are forwarded
to `pipeline/simulation_automation/` for local bounds/dependency inspection.
USDZ, FBX, and SPZ are accepted and checksummed, but they are not parsed by the
CPU preflight inspector unless a later parser is added.

## Usage

```bash
blueprint-import-scaniverse-assets \
  --capture-root /path/to/scenes/<scene_id>/captures/<capture_id> \
  --asset /path/to/scaniverse-export.usdz \
  --asset /path/to/scaniverse-splat.ply \
  --blueprint-sidecar /path/to/blueprint-scaniverse-sidecar.json
```

`--source-manifest` is accepted as an alias for `--blueprint-sidecar`.

## Claim Boundary

Allowed claim:

- Blueprint staged Scaniverse-derived support assets for review and downstream
  simulator handoff.

Blocked claims unless separate owner-system proof exists:

- The Scaniverse asset is raw Blueprint capture truth.
- The asset loaded successfully in Isaac Sim, MuJoCo, or another simulator.
- Collision, contact, scale, articulation, policy execution, task success, or
  deployment readiness has been validated.

As of this implementation, the importer is local-only. It does not call a
Niantic/Scaniverse API or prove that programmatic Scaniverse export access is
available. If Niantic grants Enterprise API access, implement it behind this
same provider boundary and keep the raw-capture proof hierarchy unchanged.

## Evaluation Admission

`scaniverse_import_manifest.json` is never an evaluation-admission artifact.
An assisted import may be considered only through the separately versioned
`evaluation_site_admission.v2` contract validated by
`validate_evaluation_site_admission`. That verifier requires all of the
following before it derives `evaluation_ready`:

- immutable site, scene, capture, source-bundle, and manifest identities and
  digests
- an independently produced verification report, from a verifier distinct from
  the importer and model backend, bound to the exact source manifest and
  source-artifact index
- active consent plus verified rights, privacy, provenance, and commercial
  `sim_evaluation` scope
- verified metric scale, named world/site/capture coordinate frames, up-axis,
  gravity alignment, and uncertainty evidence
- calibrated intrinsics, extrinsics, synchronized timestamps, and a passing
  bounded reprojection check
- calibrated static robot viewpoints tied to exact frames and trajectories
  from the moving capture
- matching robot, camera, and embodiment identities and digests
- unique scene-bound task objects, articulated parts, target zones, and an
  exact task-contract manifest plus canonical inline-row digest containing
  task/criterion/evidence/tolerance/evaluator mappings
- separate verified evidence for visual geometry, collision, contact, and
  dynamics truth
- passed site, task, and trajectory deduplication
- non-overlapping frozen train/dev/held-out-site splits and explicit OOD
  abstention behavior

Malformed nested rows, contradictory scene/capture/profile identities, missing
evidence digests, review-only scale or physics, and forced OOD decisions block
admission. Evaluation admission remains simulator evidence only; it does not
prove physical robot performance or real-world policy ordering.

## Pilot Test Plan

For the first controlled pilot, capture the same site once with BlueprintCapture
and once with the Scaniverse 360 workflow, preferably using Insta360 X5 or X4
hardware. Import the USDZ plus available PLY/SPZ/mesh exports with this command,
then verify:

- asset bounds, scale, up-axis, dependency references, and checksums
- visual fidelity of the splat/USDZ
- mesh usefulness for spawn, placement, and collision review
- whether metric scale survived the Scaniverse export
- manual workflow time and failure points
- buyer/PTDP readouts label Scaniverse assets separately from raw Blueprint
  capture evidence

## API And Plan Assumptions

Current public docs support manual Scaniverse Web upload/processing/download for
360 videos. Niantic documents NSDK/Sites APIs for authenticated site/asset
discovery and localization metadata, while Enterprise pricing mentions custom
API integration. This lane does not assume a public self-serve API for
programmatic 360 upload, asset-generation job creation, or USDZ export download.
Use Pro or Enterprise if commercial rights or API support matter; Free/Plus are
not enough evidence for serious commercial operation.

If Niantic later grants API access, wrap it behind a replaceable provider
adapter so Pipeline can swap to World Labs, Palatial, local SfM, or another
reconstruction backend without changing raw-capture or buyer-proof contracts.

Primary references:

- Niantic USDZ launch post: https://www.nianticspatial.com/blog/usdz-scaniverse
- Scaniverse 360 camera guide: https://www.nianticspatial.com/docs/scaniverse/360camera/
- Scaniverse quickstart: https://www.nianticspatial.com/docs/scaniverse/quickstart/
- Niantic pricing/API integration note: https://www.nianticspatial.com/pricing
- Niantic Sites API overview: https://www.nianticspatial.com/docs/nsdk/features/sites/
