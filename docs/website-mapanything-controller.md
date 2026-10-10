# Optional MapAnything geometry controller

ADP-009B / public day 14. MapAnything is an explicit optional backend, selected
with `backend="mapanything"`, `BLUEPRINT_WEBSITE_GEOMETRY_BACKEND=mapanything`,
or the existing explicit `BLUEPRINT_MAPANYTHING_MODEL_PATH` or
`BLUEPRINT_WEBSITE_GEOMETRY_RESULT` compatibility routes. CPU frame preparation needs no MapAnything
installation, weights, or GPU allocation. The current production reconstruction
route uses Marble; Atlas access is unavailable and is not a default prerequisite. Geometry remains estimated; it never
becomes measured scale or physical evidence.
An unconfirmed task context must retain the typed `scene_preparation` purpose,
identity and digest. Preparation preserves `confirmed=false` and customer
criteria; it grants no spending authority.

Source sampling is independent of provider capacity. The preparation CLI accepts
`--maximum-frames`; Python callers accept `maximum_frames`, and the service may
set `BLUEPRINT_WEBSITE_SOURCE_MAXIMUM_FRAMES`. The effective default stays16.
The requested count is bound into new artifact identity; the unchanged default
can verify/reuse legacy inputs without decoding again. Hidden reconstruction
holdouts retain their existing split semantics. Source sample count, held-out
count, planned provider inputs, and actual provider inputs are separate fields.

The explicit Atlas component route records `atlas_pose_inputs.json` as **pending**,
with zero API mutation. It is preparation for a future available backend, not
the current production handoff. The admitted producer must supply project asset references, source
disclosure/spend authority, and the actual `images2PosedRGBD` operation. A
retained `atlas_posed_rgbd_result.json` uses `website_atlas_pose_result.v1`, a
canonical `digest`, the exact source/input `binding`, the documented pose task
`endpoint`, and real completed `operation`. The adapter validates the returned
frame count and RGB/depth/camera bundles and records `atlas_posed_rgbd_handoff.json`.
Explicit `target_cameras` and an optional `prompt` additionally prepare
`atlas_generation_inputs.json` for `atlasGenerate` without
resizing pixels, relabeling axes, assuming measured scale, or making a request.
Context views retain their returned grid; target cameras must use Atlas's
1280 by 720 grid. Returned pose frame count does not establish provider input
usage, which remains unknown until the admitted producer supplies its receipt.
It does not treat generated posed views as final scene assets. Admitted API
dispatch, final splat/collider assets, and exact native mask/pixel/geometry
binding remain required; these are typed pending states rather than a fallback
MapAnything rental. No nonexistent world/mesh endpoint is substituted.

The current Marble1.1Plus adapter retains its own8-image contract. That limit
does not constrain the Atlas context adapter, which has no invented100-image
cap. Atlas operation limits must be read separately from its current API schema.
See [images2PosedRGBD](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images),
[cameras and posed images](https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images),
and [API reference](https://atlas-beta.worldlabs.ai/docs/api-reference).

Visual delivery precedes that GPU job. CPU frame decoding, Gemini analysis,
hosted SAM tracking and image editing prepare Marble's input. The provider
publishes the finished visual world through the existing website callback.
Only an explicitly selected MapAnything handoff requests that worker and lifts the retained SAM
tracks into estimated geometry. Without retained source geometry or an explicitly selected geometry backend,
the production handoff first retains the verified Marble base scene and then
records `website_source_camera_depth_registration_required`. The native compiler
needs original-source aligned depth, intrinsics and camera poses to lift the task
masks and register the door/rack to the room; Marble's splat, coarse collider and
estimated scale do not establish that alignment. Missing measured scale stays
unknown. Atlas settings do not enable a current production handoff; its component
helpers remain separate future compatibility. A legacy weights path alone does
not select the production MapAnything controller. It invokes neither Atlas,
MapAnything nor deferred mask completion automatically. A missing geometry result
holds native construction, while the published visual world remains available. No second
tracking purchase is needed for this binding.

The service deployment supplies `BLUEPRINT_WEBSITE_MAPANYTHING_PROFILE`, the
path to one release-specific JSON profile. It is service configuration, never
an upload field or a per-capture command. The profile has:

- `schema_version`: `website_mapanything_runtime.v1`.
- `source_commit`: the admitted deployed Pipeline commit.
- `worker_image_digest`: the pinned container image reference, including digest.
- `maximum_cost_usd`, `max_hourly_rate_usd`, `hard_ttl_seconds`, and
  `minimum_gpu_ram_mb`: the bounded worker limits. The TTL is 120–3600 seconds;
  its maximum hourly cost must fit the cap. The TTL bounds the independent
  watchdog from arming, rounded down to whole minutes. The immutable worker
  TTL reserves 120 seconds within that budget for controller preflight; delayed
  allocation fails the existing watchdog deadline check.
- `runtime_files`: four objects with absolute `path` and `sha256:`-prefixed
  `digest`: the Pipeline wheel for that release, the pinned blueprint-contracts
  wheel, the pinned MapAnything wheel, and the hash-locked dependency file.
  The existing bootstrap installs exactly three wheels and one dependency file.

Canonical deployment builds this profile automatically when
`/etc/blueprint/website-mapanything-runtime.json` exists (or the deploy process
sets `BLUEPRINT_WEBSITE_MAPANYTHING_DEPLOYMENT_TEMPLATE`). This operator-owned
template uses `website_mapanything_deployment.v1`, the same worker image and
budget fields, and only the three pinned external runtime files: MapAnything,
blueprint-contracts, and the dependency lock. The release provisioner builds
the Pipeline wheel from the exact committed source tree without network
dependency resolution, copies and verifies the pinned inputs, and publishes
the profile under `system-runtimes/website-mapanything/<commit>/`. It reads the
files back as the service account and returns the profile binding through the
existing deployment environment. Subsequent captures reuse it; subsequent
releases rebuild it. Changed inputs or an unreadable runtime fail deployment.
This deployment artifact does not allocate a GPU or grant spending authority.

The existing object-store and Vast credentials remain service-owned. WebApp's
sponsorship configuration must include sufficient upstream allowance and current
Vast terms, as it already does for other preparation providers. The signed
reservation shares that allowance with SAM, image editing and Marble; no robot
team payment is required.

Controller sequence: validate the admitted release and runtime files; prepare
the input bundle; arm the existing independent watchdog for the exact resource
name; check capacity and provider inventory; obtain the signed budget grant;
stage the bundle; invoke the canonical allocator; retrieve and independently
validate outputs; verify teardown; close the watchdog and remove temporary
transport objects. Completed outputs replay without a new rental. A lost or
uncertain allocation response retains the watchdog and transport and refuses a
second rental until reconciliation. This is not a second job queue.

The ordinary website path does not accept `BLUEPRINT_WEBSITE_GEOMETRY_RESULT`
as a substitute for this controller handoff. That override remains only for
explicit component diagnostics without a confirmed website task.

Validation: focused tests cover the real bundle contract, exact-name watchdog,
controller reservation/dispatch, completed replay and uncertain allocation.
Deployment configuration and an actual controller-origin run are separate proof
requirements; a passing fixture or retained-output replay does not satisfy them.

The canonical control-plane deploy also installs the agent-run dispatcher
service and timer and restores the timer after deployment or reboot. The
service still requires `BLUEPRINT_AGENT_RUN_DISPATCH_ENABLED=true` and its
configured capture scope; timer installation alone does not authorize a run.
Its Python entrypoint is checked by the production startup guard.

CAD generation and repair share the pinned `gen_step()` shape-return contract:
the CAD CLI owns exports, and the agent returns geometry rather than a filename
or dictionary. This fixes the retained authoring failure in the reusable stage;
the failed standalone candidate remains unaccepted and is not controller proof.
