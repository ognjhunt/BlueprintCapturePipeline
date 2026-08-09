# Second-scene articulated SimReady branch results — 2026-08-09

Status: **the exact articulated refrigerator candidate exists and is statically
qualified; Franka placement, the fully bound door-state matrix, and the three
task cameras are resolved. Native simulator qualification is not reached and is
the smallest genuinely missing capability.**

Scope: ADP-009B/ADP-009D, InteriorGS/SAGE scene `840796`, refrigerator instance
`123`, frozen task "Open the upper refrigerator door to at least 45 degrees,
release it, and retreat." Every output is `development_only`. Nothing here is
physical-site metrology, physical equivalence, deployment readiness, or a
learned-policy result. Neither `pi05_droid` nor `groot_n17_droid` was
provisioned or queried.

Branch: `codex/adp-second-scene-simready-840796`, based on
`bf57b58cbad06164296673d25cac35009a7e2133`.

## The articulated candidate

The replacement is built by deterministic parametric derivation from the
digest-bound 28-component source asset
(`sha256:4e0986e79b9d358de7e3289c0179eeaa4ec94b035e2133d2ea309298b525a4c3`),
which is itself derived from the registered SAGE collision mesh. Every
partition threshold is a frozen observed value: the seam plane at
`z = 0.9399812489`, the door back/face planes, the hinge column, the recorded
pivot, and the `[0, pi/2]` limits. No model inference and no manual component
selection participate.

| Link | Components | Mass | Colliders |
| --- | ---: | ---: | ---: |
| `/Asset/cabinet` | 6 | 62.0 kg | 6 |
| `/Asset/upper_door` | 11 (2 slab, 3 handle, 6 hinge) | 11.0 kg | 11 |
| `/Asset/lower_door` | 11 (2 slab, 3 handle, 6 hinge) | 11.0 kg | 11 |

Two revolute joints (`upper_door_hinge`, `lower_door_hinge`) share the frozen
pivot and `[0, 90]` degree limits under exactly one articulation root. The
commanded task joint is the upper hinge only.

Retained candidate:
`BlueprintValidation/data/adp009a_tranche1_20260804/simready_candidate/840796_deterministic_v1/simready_candidate_deterministic.usda`,
digest `sha256:f626998bbb5bd48d57950c4b96d9a7452705c1b2a906df950e20ac9767e649c1`.
Manifest: `manifests/second_scene_840796_deterministic_simready_candidate.v1.json`.

The handles are the observed source components, not authored substitutes: the
three components per door that protrude past the recorded door-face plane are
bound as task contact geometry. Their placement is independently corroborated
by a deterministic observation of the retained closeups
(`manifests/second_scene_840796_handle_band_observation.v1.json`): both bars
measure about `0.272 m` long and `0.05 m` tall, upper at `z [1.005, 1.058]`,
lower at `z [0.802, 0.857]`, from the zero-parallax frontal view with threshold
sensitivity disclosed. Handle protrusion depth remains a recorded candidate
parameter, never an appearance-derived measurement.

The refrigerator interior is unobserved. It is authored as a labeled generated
inset block; every geometry prim carries an explicit
`observed_source_derived` or `generated_candidate_geometry` provenance tag, and
untagged geometry fails closed.

## Static qualification

Two fail-closed validators admit the candidate
(`src/blueprint_pipeline/articulated_simready_replacement.py`):

- topology: one articulation root, 1–4 revolute/prismatic joints, geometric
  task-joint resolution (`|axis dot| >= 0.99` against the frozen world Z,
  `>= 0.85` moving-link z-overlap with the frozen upper-member interval),
  frozen limits within tolerance, hinge pivot within `0.02 m` after the
  recorded world-to-asset transform;
- physics: per-link rigid body, mass/COM/inertia, inherited physics material
  with bounded friction and restitution, collider-to-link envelopes so no
  collider spans the door seam, reset-pose penetration ceiling, grounded
  support link, handle contact geometry on the task door, required generated
  interior, and mandatory provenance tagging.

## Door sweep, placement, and cameras

- **Twelve frozen door states** (0 through 55 degrees) are clear against the
  full 840796 SAGE static inventory by exact triangle-prism narrowphase. The
  prior sealed sweep only covered 0–45 degrees. Receipt
  `manifests/second_scene_840796_door_state_clearance.v1.json`.
- **Franka base**: 43 admissible cells from a screen over SAGE AABB obstacles,
  floorplan shell triangles, the target keepout, the z-aware swing corridor,
  the reach annulus across all twelve states, and approach-line occlusion.
  Best `(1.75, 1.99)`, worst-state reach margin `0.0131 m`, obstacle clearance
  `0.455 m`. Receipt `manifests/second_scene_840796_franka_base_placement.v1.json`.
- **Fully bound matrix**: binding that base plus the authored cabinet and the
  locked lower door, all twelve states stay clear with every obstacle class
  bound. Receipt
  `manifests/second_scene_840796_door_state_clearance_bound.v1.json`,
  digest `sha256:4303545a1cf5904c1badeef79c9a75fbfcb347119ec304f412babe4bcaa0d05b`.
  Scenario admission now requires exactly this receipt.
- **Cameras**: external policy view at `(1.70, 2.30, 1.45)` with at least
  `2011` handle pixels across all twelve states, wrist view tracking the moving
  handle with at least `2310`, review-only overview at `(3.05, 3.30, 1.90)`.
  A real constraint surfaced: past about 40 degrees a right-side external
  camera ends up behind the door face it is watching, so the external view must
  sit front-left. Receipt
  `manifests/second_scene_840796_task_camera_resolution.v1.json`.

No chair, table, or scene object was moved. The 840411 chair-contact rejection
is untouched and none of its geometry or assumptions were imported.

## Joint Agent comparison arm

The Joint Agent path is the independent comparison arm for the same topology,
not the gate for this branch. Five explicitly admitted zero-retry attempts ran
today; each returned a typed null and an encoded fix, and each torn down to
provider zero with staged objects absent.

| Run | Instance | Cost | Retained null and encoded fix |
| --- | --- | ---: | --- |
| v8 | 47282158 | `$0.061326` | Released `optimize_usd` has no local Scene Optimizer backend. Fix: ship the pinned Apache-2.0 `scene_optimizer_core_usd_25.11_py_3.12` v1.0.3 package in-bundle and export `WU_SO_PACKAGE_DIR`. Also registered `adp_joint_agent_result.json` as a runtime-result filename, which would have falsely blocked even a successful run. |
| v9 | 47283980 | `$0.126088` | The OvRTX daemon hung for the full readiness window with stderr drained invisibly. Fix: construct `ovrtx.Renderer()` directly before service start with retained diagnostics; add a machine avoidlist to the lane. Separately: watchdog close waited 45 s for process exit while absence was confirmed at +164 s, so a correct teardown was recorded as unclosed — close now polls the evidence (this also removes the same false blocker from the sibling excision lane). |
| v10 | (torn down) | `$0.082770` | Scene Optimizer subprocess died in 0.04 s and the released task quarantines every exception into the bare string "USD optimization failed". Fix: probe the optimizer directly and retain its real stdout/stderr. |
| v11 | (torn down) | `$0.118768` | The probe worked and named the cause: `libtbb.so.12: file too short`. `python -m zipfile` materializes the package's 35 shared-library symlinks as text stubs. Fix: extract with `unzip`, verify no sub-1 KB `.so` stubs, and treat a probe result carrying an error or zero executed operations as a failure. |
| v12 | (torn down) | `$0.058695` | Blocked earlier, at OVRTX dependency provisioning (`joint_agent_ovrtx_provision_failed`), a step that had succeeded in v8–v11. No new code defect is implicated; this reads as host-transient and has no retained diagnostic for that step. |

Retained Joint Agent spend today is `$0.447647`. Combined retained program
spend is about `$4.89` against the `$12` authority. Provider inventory was
`[]` before and after every launch, apart from the explicitly bound concurrent
sibling instance during v8.

## Claims

| Classification | Claims |
| --- | --- |
| Implemented | Deterministic source-to-topology derivation; articulated physics authoring with observed-handle group binding and labeled generated interior; two fail-closed static validators; twelve-state door clearance with bindable static obstacle classes; scenario admission requiring the bound matrix; Franka base placement search with typed rejection histogram; external/wrist/review camera resolution; digest-bound excision join seam with fail-closed inpainting-policy resolution; Joint Agent failure-path evidence retention, optimizer provisioning, on-host renderer probe, watchdog evidence polling, and machine avoidlist. |
| Generated SimReady candidate | One exact articulated refrigerator USD, `sha256:f626998b…`, statically admitted on topology and physics, with its Franka base, twelve-state clearance, and three cameras resolved. |
| Native-simulator qualified | **None.** No blank-stage Isaac diagnostic, joint/limit/lock readback, contact stability, penetration, reset replay, or deterministic final-state check has run. |
| Blocked or abstained | Joint Agent owned-core topology remains a typed abstention; its last attempt blocked at `joint_agent_ovrtx_provision_failed`. Zero-action and scripted-positive controls have not run. No learned-policy episode was prepared or launched. Gaussian excision/coverage join is implemented but unexecuted pending the sibling branch's owned-index result. |
| Physically unresolved | Physical equivalence; real refrigerator or Franka performance; hidden interior truth; partner-site fidelity; deployment readiness; any candidate ranking. |

## Smallest next action

Run the blank-stage Isaac/PhysX diagnostic against
`simready_candidate_deterministic.usda` and read back the articulation root,
joint count and types, upper/lower joint identity, axis and limits, the locked
lower joint, upper-door motion through 55 degrees, contact stability, absence
of initial penetration, and reset replay. That single native run converts the
statically admitted candidate into a native-qualified one or produces the exact
typed blocker. The Joint Agent comparison arm can retry independently; it does
not gate this path.
