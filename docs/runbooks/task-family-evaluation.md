# Beta task-family evaluation

The 2026-10-27 owner request authorizes removing the two-profile implementation
limit. ADP's frozen physical proof remains separate. A family name never proves
that robots cannot perform a job, and registering a scorer does not supply
missing simulator physics.

`task_profile_contracts.py` routes the following families to the existing native
contract entrypoint and `task_evaluation_task_family_native_adapter.py`:

| Family | Independent simulator measurements |
| --- | --- |
| Pouring | Particle volumes in source/receiver frames, in-flight and spilled volume, volume balance |
| Wiping | Remaining residue area and cleaned fraction |
| Insertion | Peg depth, lateral/alignment error and contact force in the socket frame |
| Stacking | Support contact, center-of-mass support margin, release and motion |
| Button press | Directed joint travel, native activation sensor and contact force |
| Cloth folding | Fold keypoint error, polygon overlap, motion and the specified release condition |
| Multi-step | Ordered completion of separately typed stages and the final hold condition |

Seal explicit reset measurements, thresholds and hold durations with
`seal_task_family_contract`. Build the native scene contract through
`materialize_native_task_runtime_contract`; the family branch requires asset
digests and reset state, typed asset roles, frozen measurement geometry,
calibrated cameras, robot interfaces, engine identity and a scenario binding.
It supports multiple subjects and deformable state without applying the
single-rigid-object compiler to them.

Stacking support rings must be simple and convex. The validator checks
nonadjacent intersections/touches, unique vertices and all convex half-planes
in a translated unit-extent frame with a `1e-12` relative distance tolerance.
Forward collinear boundary vertices are supported; backtracking is refused
before runtime materialization. Cloth projections that are edge-on, concave or
self-crossing contribute zero overlap progress and retain a
`measurement_diagnostics` reason; nonfinite or malformed arrays invalidate the
episode. New telemetry and episode receipts identify
`polygon_validation_version=simple_convex_polygon.v2`. Historical receipts are
immutable; any rescore must produce a new version and digest.

The same materializer is available without a Python session:

```bash
python -m blueprint_pipeline.task_evaluation_task_family_native_adapter materialize \
  --inputs native-runtime-inputs.json --output native-runtime-contract.json
```

The input JSON supplies the materializer's typed scene, task, assets, robot,
cameras, state bindings and scenario arguments. This command validates and
writes a local contract; it does not launch a runtime or allocate resources.

Pouring freezes `transfer_workspace_bounds_world_m` as the admissible stream
region. Particles outside either container remain in flight within that region;
particles beyond it count as spill. The trace conserves source, receiver,
in-flight and spilled volume together. The task's explicit
`maximum_final_in_flight_volume_ml` limits liquid still in transit at completion;
an intermediate stream does not count as spilled liquid.

Website preparation still accepts retained internal
`descriptor.metadata.native_task_evaluation` for compatibility. When it is
absent, `task_family_button_authoring.py` can author one `button_press` or an
ordered composition of at most eight distinct buttons from the ordinary job,
retained operation reading, registered layout, exact admitted engineering model
and explicit criteria. It builds the existing `task_success_contract`,
`native_runtime_inputs`, asset source receipts and native measurement/reset
bindings, then validates them with the existing materializer and source checks.
It does not call a provider, start an engine or query a policy.

The bounded input is factual layout, not another finished native profile:

- When `registered_layout` is absent, the CPU-only
  `task_family_layout_authoring.py` adapter can compile existing engineering
  `task_masks.layout_source_records` into that contract. Its five exact file
  references are `capture_intake`, `calibration`, `frame_registration`,
  `target_poses` and `robot_registration`. Each uses the existing
  `{path, sha256, size_bytes}` evidence reference and is reopened with
  no-follow descriptors at preparation and native packet readback. These are
  internal engineering inputs, not JSON the customer must author.
  Calibration pins the admitted native capture envelope and its original
  intrinsics, pose, depth and confidence bytes. Camera metadata reads are
  bounded to 8 MiB; depth and confidence streams remain hash-only reads.
  The explicit `blueprint_raw_v3_2` camera-record format reads retained
  `session_intrinsics`/`T_world_camera` field shapes and converts the declared
  ARKit camera axes to OpenCV with a fixed axis transform. Unknown formats,
  inconsistent session/frame IDs and changed original bytes refuse; no pose
  fitting, geometry resizing or camera estimation occurs.
  The frame record supplies independent surveyed point correspondences, the
  proposed source-to-runtime transform and its explicit residual limit. The
  adapter checks rather than fits that transform, including metric scale and
  orientation. Separate target poses, robot/base/reset and camera references
  must already have measured or explicitly engineered evidence. Calibration
  or sensor declarations alone do not establish independent metric geometry.
  The adapter requires the existing metric source-geometry claim and never
  upgrades Atlas/estimated geometry. Missing records persist exact
  scene-preparation measurement requirements and preserve the accepted job.
  Layout and registration outputs use immutable, digest-bound authoring
  stages; no provider, policy or runtime is invoked or qualified.
- `task_masks.registered_layout` (`website_registered_task_layout.v1`) binds the
  source video, geometry, collider, metric registration receipt and transform.
  Each target binds its observed source pose, source frame IDs and engineering
  asset ID. The robot/base/reset and source-camera references must already be
  registered by engineering. Estimated visible bounds cannot supply these facts.
- Existing protected `BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH` rows may carry
  `authoring_assets`: exact licensed self-contained primitive URDF bytes,
  engineering evidence, contact parameters and explicit joint/sensor/reset
  mechanics. This producer supports two links and one prismatic joint; missing
  import support is a specific software dependency. No capture may select a
  catalog path, executable, runtime factory or arbitrary model dependency.
  An entry may supply explicit `button_evaluation_criteria` for minimum travel
  and maximum contact force, provided those exact values also occur in its
  pinned engineering-evidence JSON. The producer never infers either value
  from URDF effort/limits, appearance or a customer's prose. This supplies
  engineering criteria for the ordinary target/hold fact path; missing evidence
  remains an engineering requirement and creates no runtime qualification.
- The existing `operator_answers` carrier may retain
  `family_task_facts` (`website_family_task_facts.v1`), bound to capture, video,
  task-description and success-criteria digests. Confirmed operation IDs carry
  explicit thresholds and hold durations; target confirmation is needed only
  when the source quote does not select a unique captured button. No prose is
  converted to numbers. Missing joint-travel or force thresholds stay
  engineering/source dependencies; a missing job hold duration remains a normal
  task-fact request.
  New WebApp records declare `canonicalization: "rfc8785"`; their fact,
  description and success-criteria digests use the existing RFC8785 boundary.
  Unflagged retained records preserve their original Python canonicalization.
  The outer `website_site_task_context.v1` record declares the same flag when
  it carries these facts. All context readers honor it without changing raw
  artifact hashes or source/capture bindings; unknown or non-string explicit
  modes refuse. This changes byte compatibility, never measurement or spending
  authority. The v4 producer stays disabled until both deployed consumers are
  verified. The retained Node-generated context/vector fixture covers `1e-7`,
  integral `1.0`, UTF-16 key order and distinct composed/decomposed Unicode.

Repeated-family routing additionally requires exact operation
IDs/count/order and ordered, nonoverlapping verbatim source quotes. Missing IDs
or copying one occurrence to manufacture another operation keeps routing held.
Contract and packet readback check native stage identities as well as family
order, including explicit multi-step hints and aliases. Website multi-step
contracts require their retained source reading and original context. Standalone
manually authored native materialization retains its separate entrypoint.
A release/repress sequence on the same button remains a specific reset
integration gap, preserving the requested job.

Preparation also retains a source-bound authoring requirement generation under
`family_authoring/<request-digest>/requirements.json`. It separates owner task
facts, missing source measurements and runtime integration requirements, and
provides one plain-language next fact request. Source-bound operation order,
frame identities, target identities and already supplied contracts survive
restart. Changed inputs create a new immutable generation. Prose success
definitions remain prose; this precursor does not invent numeric pass marks or
renew authority.

`task_family_authoring_status.project_family_authoring_status` is the pure
projection hook for the existing preparation-status callback. Supply the
verified preparation and current context plus the trusted running Pipeline
commit; never obtain that commit from capture input. It emits the exact source
SHA/context/geometry/mask identities, description/criteria digests, at most eight
ordered operations, 32 captured target labels, 64 allowlisted requirements and
one next action that must be a member of those requirements. Its RFC8785 payload
is at most 48 KiB, leaving room in the v4 callback for the existing selectors.
It emits no local paths, reads no file, sends nothing and grants no authority.
The status owner must wire this into the current validated callback/outbox after
its source/context and ledger checks, retaining the final ledger recheck. A
missing or invalid projection is a delivery evidence gap, never task readiness.
That sender integration is pending; no parallel outbox or endpoint is added.

Incomplete or unrecognized source operations remain in their original order,
including their exact quotes and unresolved aspects. Clarifying the complete
ordered job comes before numeric criteria when that reading is incomplete or
conflicts with the declared atomic family or a retained contract. An atomic
family hint cannot discard a later operation; an unknown operation remains a
specific native-profile preparation gap.

Unimplemented lanes retain `precursor_only=true` and
`native_inputs_authored=false`; they stay `needs_input`. The bounded producer
sets `native_inputs_authored=true` only after materialization validates its
actual outputs. Its immutable source/operation/native stages adopt identical
completed bytes across restarts and version changed inputs. A supplied internal
contract is retained without asserting new owner confirmation. Missing or
expired runtime qualification may retain static native inputs, but it prevents
intake creation and remains an explicit dispatch blocker.
Family preparation promotes its shared `preparation.json` under the existing
handoff lease guard. Ownership loss at authoring boundaries stops continuation;
loss immediately before promotion refuses the current write. Completed private
immutable authoring stages remain available for later adoption.

Source-produced native inputs pin an immutable `evidence-inputs` stage under
`task_state_binding.source_authoring_evidence`. The packet pins the exact
requirements/producer receipt through `source_authoring_requirements`.
Materialization, packet verification and provider-bundle preflight reopen the
registration receipt, engineering evidence, operator catalog, retained RGB/depth
frames and every immutable authoring stage. Native outputs must match their
producer receipt. Missing or changed bytes, dropped closure fields and ancestor
swaps refuse through the existing packet boundary. Reopening uses no-follow
descriptors, checks ancestor linkage and bounds reads by the pinned size.

Only references/hashes enter the packet; raw frames and engineering evidence
are not copied into GPU assets. A validating host without already permitted
access to an exact reference remains blocked. These references grant no access
and there is no receipt-only fallback. Closure JSON is bounded to 8 MiB per
record and 4,096 retained geometry frames. Each frame ID is a nonblank string of
at most 1,024 UTF-8 bytes. Source frame identities must be unique before layout
or native authoring, and every selected source frame joins a distinct retained
native pose identity. A declared source-frame maximum remains an additional
bound; duplicate records are refused rather than deduplicated. The same frame
checks run again during native packet evidence readback. Provider context sampling follows its
independent configured contract. Supplied/manual native inputs without a source
producer claim remain a distinct compatibility route. Older source-produced
packets lacking the closure need local re-authoring from their retained inputs.

The ordinary production capture path does not yet emit the complete measured
record set or independently qualified metric geometry required by this layout
adapter, and WebApp does not yet persist these source-bound normal fact answers.
Those are engineering integration gaps, not requests for customer-authored
engine JSON or another approval stage. The development fixtures are not
registration, physics, runtime or policy qualification. No real button driver,
matching policy or native episode is admitted by this code; AUT-12 production
acceptance and AUT-13 execution remain incomplete. No new engine or candidate
configuration is enabled by these artifacts.

`execute_prepared_task_family` consumes the retained preparation and transfers
runtime ownership to execution: every preflight refusal closes the runtime,
and the episode executor closes it once after execution.

`execute_task_family_episode` consumes an actual compatible runtime and policy.
The runtime must apply and read back the sealed native scene contract, reset
the scene, expose the exact engine identity and native state, and supply the
policy observations. `native_runtime_contract_digest` must match before reset
and on every sample. Each native readback carries `native_provenance` with the
exact native runtime contract digest, scenario cell and instance digest, and
seed. The receipt and telemetry retain that identity so another scene cannot
inherit the same engine's result. The runtime is responsible for the actual particle, residue,
deformable or rigid material implementation. The adapter derives measurements
from that state, checks reset and clock, retains lossless policy frames and
their manifest/video, scores independently and closes the runtime on failure.
Missing media invalidates an episode even if its terminal state meets the
task threshold. Existing rights, runtime admission and resource allocation
checks continue to govern a production launch.

Rigid jobs retain their existing GPU screening path. Their robot-free probe
runs in construction's native-import session and travels inside the sealed
native-import result. Retention verifies construction document/asset hashes
and the exact final probe plan before binding the trace to the published
revision. This avoids an additional allocation or a manual trace-copy step.
The same compiled success contract must be used for probe and policy scoring.

Capture-derived container placement requires an observed floor, continuous
surrounding walls and a clear interior. Native scoring tests every oriented
collision-box corner against the interior volume, plus the placement support,
release and hold criteria. Carry clearance includes the container rim. A mask
box or a point inside the container cannot establish success.

A separately designed destination uses its qualified asset, local interior
geometry and live pose. Policy execution additionally needs the existing
full-scene placement qualification; standalone asset import does not establish
placement in the captured scene. `materialize_team_scene` accepts that receipt
through `destination_placement_qualification_path` (CLI:
`--destination-placement-qualification`). The digest-bound operator DROID profile
supplies `destination_execution_binding` with the expected `execution_commit`
and `runtime_identity`. A qualification cannot supply its own expected runtime.
The receipt stays outside the immutable scene revision because it binds that
revision's digest. Missing qualification is reported specifically. Existing
configured-controls progression remains the qualification execution route.

Focused regressions live in `test_task_family_button_authoring.py`,
`test_task_family_authoring.py`, `test_task_family_scoring.py`,
`test_task_evaluation_task_family_native_adapter.py`, the website family
preparation tests, `test_website_container_geometry.py`,
`test_task_evaluation_surface_target.py`,
`test_rigid_robot_free_construction_probe.py` and
`test_beta_policy_screening_rigid_scene.py`. These are hermetic software checks;
they do not establish live GPU compatibility or physical performance.
