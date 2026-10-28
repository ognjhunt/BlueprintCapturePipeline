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

Website preparation consumes `descriptor.metadata.native_task_evaluation` with
`task_success_contract` and `native_runtime_inputs`. The retained source scene
and frame registration must match. The handoff preserves the original job and
the compiled `task_evaluation_inputs`. Missing fields are named individually;
the preparation output does not claim that a simulator ran or authorize paid
execution. The existing package/status readers retain this route across
restarts.

Preparation also retains a source-bound authoring requirement generation under
`family_authoring/<request-digest>/requirements.json`. It separates owner task
facts, missing source measurements and runtime integration requirements, and
provides one plain-language next fact request. Source-bound operation order,
frame identities, target identities and already supplied contracts survive
restart. Changed inputs create a new immutable generation. Prose success
definitions remain prose; this precursor does not invent numeric pass marks or
renew authority.

Incomplete or unrecognized source operations remain in their original order,
including their exact quotes and unresolved aspects. Clarifying the complete
ordered job comes before numeric criteria when that reading is incomplete or
conflicts with the declared atomic family or a retained contract. An atomic
family hint cannot discard a later operation; an unknown operation remains a
specific native-profile preparation gap.

This is an authoring precursor, not a native-input producer. Its
`precursor_only=true` and `native_inputs_authored=false` fields are explicit;
the accepted job remains `needs_input` until the existing materializer validates
real inputs. A supplied internal contract is retained without asserting a new
owner confirmation. A bounded family producer, actual engine driver, matching
policy and observed native execution are still required to close AUT-12/AUT-13.
No new engine or candidate configuration is enabled by this artifact.

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

Focused regressions live in `test_task_family_scoring.py`,
`test_task_evaluation_task_family_native_adapter.py`, the website family
preparation tests, `test_website_container_geometry.py`,
`test_task_evaluation_surface_target.py`,
`test_rigid_robot_free_construction_probe.py` and
`test_beta_policy_screening_rigid_scene.py`. These are hermetic software checks;
they do not establish live GPU compatibility or physical performance.
