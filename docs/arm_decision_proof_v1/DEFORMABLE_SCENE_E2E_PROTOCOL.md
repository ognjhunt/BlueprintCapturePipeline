# Deformable Scene End-to-End Protocol

Status: outcome-blind freeze; terminal completion path B on external upload rights

Program: `arm-decision-proof-v1`

Run: `adp-deformable-scene-e2e-v1`

Scene: InteriorGS/SAGE `840873`

Task kind: `deformable_transfer`

Frozen candidates: `pi05_droid`, `groot_n17_droid`

This protocol defines one development-only public-dataset simulator rehearsal.
It does not qualify partner capture, real-site fidelity, physical towel material,
real Franka performance, deployment readiness, or customer value. No learned
policy outcome was inspected when the scene, task family, composed placements,
candidate pair, controls order, or initial powered subset were selected.

## Decision and bounded scope

The falsifiable deliverable is one digest-bound Task Evaluation Run in which:

1. the zero-action control fails;
2. a deterministic scripted control transfers the deformable through the same
   native action seam and genuine contact path used by policies;
3. exactly `pi05_droid` and `groot_n17_droid` run in identical admitted cells;
4. deterministic state scoring, policy-input frames, review videos, costs, and
   teardown evidence are sealed; and
5. the result either compares the two candidates or emits the smallest typed
   abstention.

The task is not folding, dressing, knotting, cable routing, liquid handling, a
general deformables platform, or a physical-performance claim.

## Published baseline and isolated worktree

The run is being developed on the isolated worktree
`/private/tmp/blueprint-adp-deformable-scene-e2e-20260810`, branch
`codex/adp-deformable-scene-e2e-20260810`. Its published baseline contains the
required generalized-harness commits:

- `f6804e2d6e83097152acea8c10ac653257d44c8b`, complete native Arena import
  dependency closure;
- `6590ca019258bf5e34906f4e813c2ad5da5d8731`, Arena asset-registry dependency
  packaging; and
- `b7d57b1c3d7c2c4d5c4d851a398d0c52e7b394c2`, robot-scoped Arena imports.

The published implementation head before the terminal evidence seal is
`1b04968b5c121032f33d6f07e0bda168022b39ca`. User-owned primary and sibling
worktrees remain untouched; only exact reviewed files from this isolated
worktree may be staged.

## Terminal prelaunch outcome

All locally decidable reusable construction gaps were exhausted through static
asset intake, clean pinned-PhysX preparation, native stage/readback contracts,
entity-keyed controls/cameras/policy sampling, trusted canary packaging, and
portable evidence indexing. The supplied Lightwheel job output may be inspected
locally, but no exact output license or service terms authorize copying it to
Vast. The frozen terminal blocker is
`lightwheel_simreadygen_job_output_rights_receipt_missing`.

No paid launch, provider upload, native control, or learned-policy episode is
permitted under this freeze. This is completion path B at an external-input
boundary, not a native capability result and not a policy failure. A later
rights resolution starts a new admission: fresh provider-zero, a single
zero-retry blank-stage canary, and backend re-freeze only after cook/contact,
reset, cameras, and both policy adapters pass.

## Outcome-blind scene selection

Selection criteria were frozen in
[`scene_selection_preregistration.v1.json`](deformable_scene/scene_selection_preregistration.v1.json).
The complete known local catalog was surveyed before target close-ups. The
catalog receipt is
[`scene_catalog_receipt.v1.json`](deformable_scene/scene_catalog_receipt.v1.json)
with digest
`sha256:2eeeaf123e2002542bd026e83bcb58d4bbb9f62f6ca4fef6d61eab975dd19443`.

| Scene | Disposition | Objective reason |
| --- | --- | --- |
| 840076 | Rejected | No admitted movable semantic |
| 840303 | Rejected | No admitted movable semantic |
| 840313 | Rejected | Previously used scene |
| 840411 | Rejected | No same-publisher-room compatible pair |
| 840796 | Rejected | Previously used scene |
| 840873 | Selected shortlist | Only known scene with rank-1 towel/basket candidates and both exact InteriorGS appearance and SAGE collision sources |
| 840874 | Rejected | No admitted movable semantic |
| 840920 | Rejected | No admitted movable semantic |
| 841151 | Rejected | No compatible destination semantic |
| 841193 | Rejected | No admitted movable semantic |
| 841229 | Rejected | No admitted movable semantic |
| 841244 | Rejected | No admitted movable semantic |
| 841757 | Rejected | No same-publisher-room compatible pair |

Scene `840873` is new to the evaluation harness and reuses none of the geometry,
placements, transforms, masks, destinations, camera assumptions, task-object
assumptions, or source colliders from `840313` or `840796`.

## Registered public sources and claim boundary

The selected source identities are:

- InteriorGS appearance: `0016_840873/3dgs_compressed.ply`, 30,670,612 bytes,
  `sha256:b2fba405...`; nonredistributable raw bytes may not be uploaded.
- InteriorGS labels: 198,732 bytes, `sha256:30007f1d...`; raw labels may not be
  uploaded.
- InteriorGS structure: 35,839 bytes, `sha256:a55eabf1...`; raw structure may
  not be uploaded.
- SAGE collision: `840873_collision.usd`, 20,030,048 bytes,
  `sha256:a3a2fb401...`; CC-BY-NC restrictions remain binding.

The final evidence package must retain the complete 64-hex digests, exact local
paths, revisions, license documents, attribution, disclosure classes, provider
terms, and output-rights receipts. Abbreviated digests above are descriptive,
not execution identities.

Independent label-to-connected-collider checks support common Z-up, metre-scale
world coordinates: instance 79 IoU `0.951837`, instance 87 IoU `0.963694`, wall
cabinet 71 IoU `0.986791`, and faucet 72 IoU `0.969110`. Registration remains a
simulator construction claim, not real-site truth.

## Observed design bases and composed task entities

The observed source instances stay unchanged as unscored background:

- towel instance `79`: a rolled towel within a visible towel stack, bounded by
  an observed envelope approximately `0.370025854 x 0.119487516 x 0.113052810 m`;
- basket instance `87`: a shallow rigid open rack/basket with an observed outer
  envelope approximately `0.284032895 x 0.466227452 x 0.123872766 m`.

The source basket is occupied by toiletries and its complete floor, wall
thickness, and empty interior are not observed. The current strict SAGE topology
receipt therefore reports `open_collision_cavity_passed=false` with 65
interior-prism obstructions. It is only a design basis for a separate engineered
open receptacle twin. The engineered empty interior, wall/floor construction,
material, and composed pose are generated simulator facts and must never be
reported as observed source truth.

The scored task uses stable task-neutral entity roles:

| Entity ID | Role | Physics | Source entity treatment |
| --- | --- | --- | --- |
| `840873_inserted_towel_twin` | `movable_deformable` | User-supplied Lightwheel-derived closed-surface candidate; pinned PhysX conversion and native qualification pending | Source towel 79 retained unchanged |
| `840873_inserted_basket_twin` | `destination_receptacle` | Engineered rigid open receptacle; native qualification required | Source basket 87 retained unchanged |
| registered support entity | `support_surface` | Registered static collision | No source removal |
| registered scene entities | `obstacle` | Registered scene collision | No obstacle moved or removed |
| frozen Franka entity | `robot` | Franka articulation | Robot-scoped Arena import only |

No Gaussian or source collider is removed for either source original. No
inpainting is authorized or required for the composed placements. Both inserted
entities have independent asset identities, poses, resets, contact roles,
scoring roles, provenance, and digests.

## Frozen prompt, task, and candidates

Prompt:

> Pick up the towel, place it inside the open basket, release it, and retreat.

The task starts with the inserted towel supported outside the inserted basket,
the basket stably supported, the Franka at its frozen reset, no penetration, and
all deformable nodes free after the bounded reset write. Direct object-state
writes, object teleportation, and hidden kinematic attachment are prohibited
after episode start.

Candidates are exactly:

1. `pi05_droid`
2. `groot_n17_droid`

Neither candidate may be silently substituted. A candidate can remain unevaluated
only through a typed admission blocker binding its exact source, checkpoint,
environment, preprocessing, prompt, adapter, and runtime evidence.

## Frozen composed placement cells

The placement manifest is
[`840873_composed_paired_entity_placement_manifest.v1.json`](deformable_scene/840873_composed_paired_entity_placement_manifest.v1.json),
digest
`sha256:baa01dc3322231be059d9c78067439432935778522b77a89bcbce4416e809a66`.
Both cells use frozen seed `2026081001` and the same entity/asset binding IDs.

### Canonical cell

`840873_canonical`:

- basket position `[-4.1, 1.9, 0.752873215]`, yaw 90 degrees;
- towel position `[-4.1, 2.2, 0.748305425]`, identity orientation;
- Franka base centre `[-3.5, 2.1, 0.375]`;
- placement receipt
  `sha256:c6554726e9fcb32f2a96267bb5daeecf337888bd7c0a7b76117686cb391b24d4`.

### Held-out composed-relocation cell

`840873_held_out_composed_relocation_seed_2026081001`:

- basket position `[0.9, 0.1, 0.061094195]`, yaw 90 degrees;
- towel position `[0.7, -0.2, 0.056526405]`, identity orientation;
- Franka base centre `[1.0, -0.7, 0.375]`;
- placement receipt
  `sha256:6fb7a985f0bf2834fc5ae54a9e5677a52c30d6cfbcf8927dc3786e9c175dc098`.

This cell is the one authorized randomized-composition test: admissible support
regions were enumerated first, then one was selected by the frozen seed. There
is no runtime resampling, outcome-conditioned relocation, or manual post-outcome
repositioning.

The placements are geometry-plausibility candidates only. Native support,
collision, full-phase IK, applied pose, camera, reset, and control readback must
still pass for each cell.

## Scenario suite and initial powered subset

The reusable suite contains these preregistered families:

- canonical;
- bounded placement/approach;
- illumination;
- camera/sensor;
- bounded deformable physics;
- admitted appearance/material cousin;
- held-out composed relocation; and
- held-out composed combinations.

The initial powered subset is deliberately limited to the canonical cell and
the single held-out composed-relocation cell above. No upper-bound cell is paid
or scored until all statically discoverable dependencies pass one-shot preflight
and both controls pass in every currently admitted cell. Every candidate receives
identical seed-resolved parameters, source bytes, assets, physics, cameras,
reset, scorer, and action seam. Native readback must prove every resolved value
was applied.

## SimReady asset contract

The remaining movable-asset insertion slot must bind exact geometry/topology,
materials/textures, metric scale, origin and pose, mass or area density,
thickness, stretch/bend/shear or truthful unresolved equivalents, damping,
friction, self-collision, discretization, solver iterations/substeps, maximum
strain, contact offsets, grasp representation, runtime import identity, reset,
and retained diagnostics.

A generated candidate is not automatically SimReady; a SimReady candidate is
not automatically native-qualified; visual alignment is not physical material
equivalence. The pinned Isaac Lab backend currently exposes volumetric FEM, not
an independently parameterized thin cloth shell. The selected development
candidate is the user-supplied Lightwheel-derived rolled-towel USD. Its inspected
visual mesh is a closed, oriented, watertight surface, but its source USD uses
experimental provider auto-cook schemas, contains empty tetrahedral meshes,
declares a static-rigid asset format, and is not directly compatible with the
pinned Isaac Lab deformable loader. A separate derived runtime USD must bake the
frozen metric scale into the surface points, preserve only bound visual
materials/textures, omit source lighting/provider physics declarations, and be
cooked once through the exact pinned PhysX API. The backend may be re-frozen only
after load/cook, contact, reset, camera, and both policy-adapter gates pass. No
service-generated or sample asset is treated as disclosure- or upload-admitted
without exact generated-output terms or other durable output-rights evidence.

## Construction and genuine manipulation gates

For every admitted cell, construction must retain native evidence for:

- exact entity-keyed asset composition and pose readback;
- stable support and no initial penetration;
- collision-free Franka base placement;
- workspace/IK reachability for pregrasp, grasp, lift, transport, deposit,
  release, retreat, and recovery;
- genuine two-sided gripper/deformable contact through a released native API;
- no hidden attachment and no post-start deformable state write;
- deformable lift, release, settling, finite state, and strain readback;
- receptacle stability and pose tolerance; and
- external, wrist, and overview camera gates for both task entities.

If the selected native runtime cannot expose qualified deformable contact or the
gripper cannot genuinely grasp the asset, the run emits the smallest typed
native capability abstention after all safe reusable work is complete.

## Controls and deterministic scoring

Controls run before either learned policy in each cell:

1. `zero_action_negative`: must fail the task;
2. `deterministic_scripted_positive`: must succeed through the same action seam,
   native reset, cameras, physics, and scorer used by policies.

Scripted success requires an initial towel state outside the destination,
same-sample qualified grasp contact, observed post-contact deformable motion,
ordered transport into the basket, release, minimum node fraction and centroid
inside the frozen destination OBB, a full settle window under the velocity
limit, finite/no-divergence state, maximum strain below the frozen bound,
gripper retreat, stable receptacle, and zero prohibited writes or attachments.

A failed scripted positive is a harness/task-construction blocker. It is never a
learned-policy failure. A learned `never_moved` trace is not interpretable until
action delivery and arm response are proven.

## Cameras and media

Policy inputs are exactly the calibrated external and wrist RGB streams.
Overview is review-only and is excluded from policy input and deterministic
scoring. Every episode must retain:

- exact lossless external and wrist policy-input frames;
- synchronized overview frames;
- camera transforms, intrinsics, timestamps, and renderer identity;
- a digest-bound frame manifest; and
- verifier-derived H.264 videos for external, wrist, and overview.

Video derivation must be replayed from the bound lossless frame sequences; codec
metadata or a self-asserted source-frame digest is insufficient.

## Paid execution and failure policy

Vast is the only authorized GPU provider. Before launch, the canonical allocator
must prove no unauthorized instances, bind an independent cap, TTL, watchdog,
teardown path, and admission receipt, and reserve spend within the goal-wide
`$20` ceiling. There are no automatic paid retries. Every paid null, exact cost,
provider input/output digest, and failure phase is retained. Final completion
requires API-confirmed global provider zero.

Raw nonredistributable InteriorGS PLY, labels, or structure bytes may never be
uploaded. Only rights-routed derived/runtime bytes may cross the provider seam.

## Evaluation and comparison

After both controls pass in an exact cell, `pi05_droid` and
`groot_n17_droid` run once each with the same prompt and resolved cell. The run
retains model/checkpoint/runtime identities, policy reset and RNG evidence,
preprocessing, exact policy inputs, source outputs, adapter replay, delivered
actions, observed arm/gripper response, deformable/receptacle/contact traces,
timings, deterministic score, and media.

Only episodes with trusted native authority, exact-cell control joins, complete
media, and interpretable action delivery enter the comparison. Ties, nulls, and
ambiguity remain explicit.

## Completion conditions

Completion is either:

- a sealed run with qualified construction, zero-action failure, scripted-positive
  success, and both frozen candidates evaluated; or
- a typed, evidence-backed abstention at the smallest genuinely missing external
  input or released native capability after the reusable local backlog is
  exhausted.

Artifact existence, scene startup, asset generation, one control, a nominal
rollout, or a learned-policy null is not completion.
