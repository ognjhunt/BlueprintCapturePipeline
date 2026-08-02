# Scaniverse and Polycam External Asset Import Lane

Status: local, proof-bounded support-asset lanes. Remote provider execution is
disabled.

`blueprint-import-scaniverse-assets` stages Scaniverse exports into an existing
Blueprint capture root. It is for capturer/operator workflows where a site was
also processed through Scaniverse Web from a 360 video capture and exported as
USDZ, PLY, SPZ, GLB/GLTF, FBX, OBJ, or USD. It is an optional support lane, not
a replacement for BlueprintCapture raw bundle authority.

This command is the legacy operator-convenience lane. Its sidecar fields are
advisory and its manifest is not strict reconstruction admission. Task
Evaluation Supervisor imports use the separate, fail-closed contracts described
below.

## Strict Supervisor Import

The registered `import_external_reconstruction` tool accepts only the digest of
an `external_reconstruction_import_request.v1`. Trusted runtime state supplies
the validated request, source root, output root, and repository-owned local
importer; the agent receives no path, shell, network, database, or provider
handle.

The strict request requires exact source-capture identity and digest, immutable
asset paths and hashes, and an inline provider-matched provenance/rights
declaration. Scaniverse keeps its provider-specific
`niantic_scaniverse_provenance_rights_receipt.v1`; Polycam uses the neutral
`external_reconstruction_provenance_rights_receipt.v1`. A declaration for one
provider cannot be replayed against the other.
That declaration records product tier, terms version, provider scan or job
identity, ownership or license, commercial-use scope, consent/privacy status,
confidentiality, retention, deletion, model-training, competitive-use, resale,
and benchmarking terms. It attests that provider processing was performed by
the user and that Blueprint performed no remote upload.

Accepted files are `.usdz`, `.usd`, `.usda`, `.usdc`, `.ply`, `.spz`, and
`.glb`. These include the self-contained `GLB`, `USDZ`, and `PLY` exports useful
from Polycam. Multi-file `GLTF`/`OBJ` exports remain outside this strict lane
until dependency-set binding and reference rewriting are implemented; exporting
`GLB` avoids that ambiguity. The importer confines paths to the declared source
root, rejects symlinks, traversal, digest mismatch, excessive size/count, and
unsafe USDZ archives, then copies into a content-addressed local directory.
Untrusted source filenames are sanitized and never treated as instructions. It emits separate
`niantic_scaniverse_provenance_rights_receipt.v1` and
`external_reconstruction_import_receipt.v1` artifacts, and re-hashes assets on
replay.

The receipt proves only that exact, rights-reviewed provider exports were
admitted as derived support. Raw observation, metric scale, collision validity,
Isaac compatibility, simulator task evidence, physical success, deployment
readiness, and remote-upload authority all remain false until their independent
gates pass.

## Polycam route

Polycam can feed this pipeline in two distinct ways:

- A user-managed `GLB`, `USDZ`, or `PLY` export can enter the strict local
  external-import lane now. It is a derived appearance/mesh/point-cloud
  candidate and must still pass Blueprint scale, geometry, collider, and Isaac
  qualification.
- A LiDAR capture made after enabling Polycam Developer Mode can expose a raw
  data ZIP with cameras, confidence images, depth images, and mesh information.
  That is materially more useful for reconstruction research, but it is still a
  Polycam raw export, not Blueprint Raw Contract 3.2: it lacks Blueprint's
  encoder-attempt and retained-frame evidence and must enter through a separate
  source-profile adapter before any calibrated or metric claim.

Polycam also documents an Enterprise Content Management API with capture
listing, source `session.zip`, artifact download, export conversion jobs, and
webhooks. The API is a viable future provider adapter, not a shortcut around
governance. The current repository performs no Polycam network call; remote use
still requires exact terms review, workspace/API credentials, confidential
upload authority where applicable, immutable input/job binding, retention and
deletion handling, and the canonical paid-resource/provider seam.

Official current references (reviewed 2026-08-01):

- https://poly.cam/docs/api
- https://learn.poly.cam/hc/en-us/articles/27756102599572-What-File-Types-Can-Polycam-Export
- https://learn.poly.cam/hc/en-us/articles/34295907278996-How-to-Access-Developer-Mode
- https://learn.poly.cam/hc/en-us/articles/38276871185044-How-to-Extract-Raw-Data-and-What-Is-Included

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

The `invoke_authorized_reconstruction_provider` descriptor is registered as an
external side effect but intentionally has no live Scaniverse adapter. The
current deterministic `reconstruction_provider_admission.v1` result is blocked
unless a trusted, non-agent legal review accepts the exact commercial,
confidentiality, retention, deletion, model-training, competitive-use, resale,
and benchmarking terms; a programmatic upload/job/download API and the canonical
paid-allocation route are qualified; and provider credentials exist. The
official product workflow currently documented by Niantic is Scaniverse
Web/app upload and cloud processing, which is not sufficient evidence for those
API and teardown gates.

A future admitted request must also contain an exact operator authorization
receipt for confidential upload, provider execution, output download, deletion,
spend, TTL, retries, provider identity, and immutable input digests. Even then,
`reconstruction_provider_execution_receipt.v1` labels provider success
unqualified, and `reconstruction_provider_deletion_receipt.v1` cannot by itself
claim provider-zero.

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
- Task Evaluation Run and optional evidence-export readouts label Scaniverse assets separately from raw Blueprint
  capture evidence

### Finalizing a multi-asset pilot

Use the evidence finalizer after each lane has produced its import, Isaac,
policy-trace, Task Evaluation Run, and teardown artifacts:

```bash
python -m blueprint_pipeline.external_scene_pilot_evidence \
  --request /path/to/external_scene_pilot_finalization_request.v1.json \
  --artifact-root /path/to/BlueprintCapturePipeline \
  --repo-root /path/to/BlueprintCapturePipeline \
  --output-root /path/to/pilot-finalization
```

The request lists every gate with a status, deterministic blocker codes, and
one or more relative evidence paths plus their SHA-256 digests. The compiler
re-hashes every referenced regular file, refuses missing mandatory gates, and
rejects any `supported` claim whose declared gate set is not fully passed. It
emits a master run manifest, provider-neutral comparison, claim ledger,
terminal summary, reproduction record, and exact Git-state report.

A CPU qualification or formal Task Evaluation Run may complete with an
explicit abstention. Live stage load, nonblank rendering, collision/contact,
robot/camera evidence, two distinct candidate traces, digest linkage, and
remote provider-zero evidence may not be replaced by narrative claims.

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
