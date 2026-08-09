# Second-scene articulated SimReady branch results — 2026-08-09

Status: **the exact SimReady articulated refrigerator exists and is statically
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
| Implemented | Suppression volumes with proven index, byte, and coverage equivalence to the sealed deletion, plus demonstrated reversibility; deterministic source-to-topology derivation; articulated physics authoring with observed-handle group binding and labeled generated interior; two fail-closed static validators; twelve-state door clearance with bindable static obstacle classes; scenario admission requiring the bound matrix; Franka base placement search with typed rejection histogram; external/wrist/review camera resolution; digest-bound excision join seam with fail-closed inpainting-policy resolution; Joint Agent failure-path evidence retention, optimizer provisioning, on-host renderer probe, watchdog evidence polling, and machine avoidlist. |
| Generated SimReady candidate | One exact SimReady articulated refrigerator USD, `sha256:a673d2e4…`, produced by the NVIDIA Physics Agent from the Blueprint-authored candidate `sha256:f626998b…`, statically re-admitted on topology and physics with the articulation and authored link masses intact, and with its Franka base, twelve-state clearance, and three cameras resolved. |
| Native-simulator qualified | **None.** The probe inputs are frozen and the readback is now a required gate, but no blank-stage Isaac diagnostic, joint/limit/lock readback, contact stability, penetration, reset replay, or deterministic final-state check has executed. |
| Blocked or abstained | The Content Agents Material and Texture agents returned typed nulls in the same run and the runner's terminal record is `blocked`, so this is a partial agent pass. Joint Agent owned-core topology remains a typed abstention; its last attempt blocked at `joint_agent_ovrtx_provision_failed`. Zero-action and scripted-positive controls have not run. No learned-policy episode was prepared or launched. Gaussian excision/coverage join is implemented but unexecuted pending the sibling branch's owned-index result. |
| Physically unresolved | Physical equivalence; real refrigerator or Franka performance; hidden interior truth; partner-site fidelity; deployment readiness; any candidate ranking. |

## SimReady authoring pass and native probe (later on 2026-08-09)

After the candidate was built, two further pieces landed.

**Content Agents SimReady pass.** The lane could previously only run the
840313 can. Four hardcodings were generalized so it accepts any admitted
variant, each with hermetic coverage:

- an `articulated_v1` input variant that keeps the scene-derived USD and its
  Blueprint reference render in the evidence root while the repo holds only
  the digest-bound manifest, and admits only a statically admitted candidate
  with both validator digests;
- three checked-in agent configs (material, texture, physics) that ship
  byte-identical and preserve the articulation by contract — `optimize_usd`
  disabled where present, the joint rigger never enabled, mass writing off
  because Blueprint already authored masses, inertias, colliders, and both
  joints;
- a remote-config contract that keeps every runtime-failure assertion but
  replaces the can's prim paths with internal consistency, an input
  normalizer that clears non-default mesh purposes and verifies the
  articulation survives, and a USD bbox probe derived from the bundle's own
  input;
- a bundle entry contract that requires exactly one input USD and one
  reference image by shape rather than by filename;
- a local preflight image admitted by recipe (Dockerfile digest, base image,
  pinned source tree) instead of one unreproducible build ID, which had made
  the gate unsatisfiable once that image was gone.

The local config preflight then **passed**: all three agent CLIs and the USD
bbox probe dry-ran in the pinned image against the articulated candidate.
The paid pass is bundle `sha256:224a9445…` (47.8 MB) bound to input USD
`sha256:f626998b…`. Two provider attempts were retained: one blocked before
allocation at the old entry contract (zero spend) and one whose host exited
70 s in before running the bundle (`$0.006258`, machine `46598`
auto-recorded to the avoidlist).

**Native articulation probe.** `materialize_articulated_native_probe` freezes
the inputs for the one run that would convert the statically admitted
candidate into a native-qualified one: a blank physics stage, an articulation
stage referencing the exact candidate bytes, and a spec listing eleven
required readbacks. For 840796 it is frozen at root `/Asset`, two revolute
joints, task axis Z with `[0, 90]` degree limits, twelve commanded states
through 55 degrees, both joints reset to zero, 40 settle samples at 15 Hz,
and a 0.001 rad locked-joint tolerance; spec digest
`sha256:52c6e567…`. It executes nothing, and a spec contradicting the
authored asset fails closed before any paid time.

`native_articulation_readback` is now a required construction gate, so no
scenario cell can materialize on static evidence alone.

## The SimReady asset

The NVIDIA Physics Agent pass completed successfully on the exact candidate
(instance run `840796_v4`, `$0.485139`, torn down to provider zero). The
enriched asset is retained as

`BlueprintValidation/data/adp009a_tranche1_20260804/simready_candidate/840796_simready_v1/simready_articulated_refrigerator_840796.usda`,
digest `sha256:a673d2e4797498db21597aadca998daa4ecf7cf509b7d6a62165adc4924c32be`,
manifest `manifests/second_scene_840796_simready_articulated_candidate.v1.json`.

What the agent added, on top of the Blueprint-authored candidate: six
inferred physics materials spanning distinct friction/restitution classes
(for example `sf0.90/df0.80/r0.80` and `sf0.40/df0.30/r0.40`), per-component
material bindings (1 to 27 prims), and per-component `PhysicsMassAPI`
(3 to 29 prims).

What it did not do, verified by readback: the articulation is intact - one
articulation root, both revolute hinges, task axis Z, `[0, 90]` degree
limits - and the Blueprint-authored link masses are unchanged at
`62 / 11 / 11 kg`. The 26 added component-level mass entries carry zero
values on non-rigid-body prims. Both static validators re-admit the enriched
asset unchanged, and a hermetic test now fails closed if any future agent
pass moves an authored link mass out of the admitted range.

Two agents in the same run returned typed nulls rather than output: the
Material Agent's pipeline failed, and the Texture Agent rejected its plan
with zero executable jobs because the configured material path is not bound
in the candidate. Neither blocks the physics result; both are retained. The
runner's terminal record is `blocked`, so the run is recorded as a partial
agent pass whose physics output is independently validated rather than as a
clean four-agent success.

Agent physics priors are advisory. They are not measured properties, and
they do not raise the claim ceiling.

## Suppression volumes: hiding the source object instead of deleting it

The sealed cutout answered "which splats does the twin's space own?" by
writing a new scene file with those rows removed. That works, but it forks a
140 MB scan per edit, cannot compose two task objects against one scene, and
edits bytes the capture contract protects. `gaussian_suppression_volume.v1`
records the same answer as geometry - the box the twin's body occupies, one
swept prism per articulated member taken to its *authored* limits rather than
the commanded maximum, and an optional annex of indices from an admitted
evidence process - and applies it against the untouched scan at render or
package time.

Three properties are proven against the real 840796 evidence rather than
asserted:

- **Index equality.** The body region alone reproduces the sealed geometric
  set exactly (3,791). Body plus annex reproduces the sealed deletion set
  index-for-index (4,422). Adding the door's swept wedge yields a strict
  superset (4,424) - two near-transparent halo splats at the door face that
  the deletion path had no region for.
- **Byte identity.** The payload built from the untouched canonical scan plus
  the receipt is byte-identical to the sealed deletion path's retained scene,
  `sha256:2c26029f…`. Retained rows are copied byte-for-byte in source order
  and verified, so this is a removal, never a rewrite.
- **Coverage equivalence by construction.** The sealed 96-cell hybrid review
  bound the exact splat it rendered, and that digest is the payload's digest.
  Every one of those eight cameras by twelve door states is therefore
  reproduced without re-execution, and the 2.9% worst-case residual carries
  over unchanged.

Reversibility was demonstrated end to end: rendering with the receipt absent,
applying it, then removing it returns a frame with **zero** pixel difference
from the original, and the canonical scan's digest `sha256:a8dd5bae…` is
unchanged throughout.

Two lifetimes share one materializer. A transient payload serves a single
render invocation and is deleted afterwards; a content-addressed cached
payload serves closed renderers such as the Isaac/NuRec lane. Neither is a
capture artifact - both are pure functions of (canonical scan, volume set)
and regenerable at any time.

The construction join now takes `suppression_mode` - `deletion`,
`render_time`, or `package_time` - and every coverage, collider-removal,
replacement-binding, and door-state gate runs identically in all three; a
test pins that raising residual pixels still blocks admission under
`render_time`. N task volumes bind against one canonical scan, so a
multi-object site carries one scan plus one small receipt per object rather
than a forked scene file per combination.

Seal: `manifests/second_scene_840796_suppression_volume_proof.v1.json`,
digest `sha256:ed65b20d…`.

What this does **not** change: the residual seam band is the same 2.9%, the
Gaussian-ownership question remains unsolved and unsolvable by deletion, and
nothing here is native qualification.

## Correcting the Content Agents account, and the one root cause behind it

Reading the retained 840796 logs closely changes what was recorded earlier.
The Material Agent did not fail at its work: it completed inference on 29
prims and created materials. It failed on a *preview render*, and one blank
render took the pipeline down with it. The Texture Agent rejected its plan for
a related reason. Both trace to a single fact - the candidate carried no bound
render material, so every OVRTX preview came back blank.

That had a second and worse consequence. With nothing to look at, the
classifier answered the question it was asked, and the prompts carried over
from the 840313 packet still said "classify the bright green beverage can". It
returned `Car_Paint_Green` for a pale off-white refrigerator. That is a config
defect on our side, not a model failure.

So the corrected agent ledger for run `840796_v4` is: Physics **succeeded**,
Validation **completed with zero failures**, Material **succeeded at inference
and was killed by a render**, Texture **rejected for a missing bound
material**, Joint **never reached inference** across five earlier attempts.

## The twin was sealed, and now it is not

`evaluate_interior_exposure` casts rays inward through the aperture the open
door leaves behind and reports what they hit first. On the twin as sealed:
**169 of 169 samples hit `/Asset/cabinet/component_008`**, exposed fraction
`0.0`. The interior prim existed at `y <= 0.142`, but the carcass front face at
`y = 0.178` stood in front of it. The asset had one articulation root, correct
joints and limits, clean colliders and a required generated interior - every
gate we had - and still showed a flat wall when the door opened.

`cut_support_link_aperture` clips the aperture rectangle out of the support
link's outward faces and retriangulates the remainder. Two invariants keep it
safe on a physics-qualified asset: existing points are never moved or dropped,
so a convex-hull collider is unchanged by construction, and only the support
link is ever cut. Applied to 840796: 27 faces removed, 67 added, 5.6% of
surface area, collider approximations unchanged, articulation preserved, and
**interior exposure moves to 169 of 169**. Both static validators still admit
the opened asset.

`ensure_render_material_scaffold` then authors what the texture stage needs:
four bound `UsdPreviewSurface` materials - door shell, cabinet shell, handle
metal, interior liner - seeded with the measured front-door albedo
`(0.810, 0.782, 0.762)` sampled from the splats the volume removed. Physics
bindings are proven unchanged, including the subtle case a test caught, where
an all-purpose render binding silently becomes the fallback for the physics
purpose.

A flat colour is not a texture pass and the receipts say so. The interior
remains generated candidate geometry; it was never observed.

Render-ready candidate `sha256:81b1796a...`, manifest resealed at
`sha256:68d78ba6...`. The local config preflight passes on the rebuilt bundle
`sha256:9d2d9221...`.

## Smallest next action

Run the blank-stage Isaac/PhysX diagnostic against
`simready_candidate_deterministic.usda` and read back the articulation root,
joint count and types, upper/lower joint identity, axis and limits, the locked
lower joint, upper-door motion through 55 degrees, contact stability, absence
of initial penetration, and reset replay. That single native run converts the
statically admitted candidate into a native-qualified one or produces the exact
typed blocker. The Joint Agent comparison arm can retry independently; it does
not gate this path.
