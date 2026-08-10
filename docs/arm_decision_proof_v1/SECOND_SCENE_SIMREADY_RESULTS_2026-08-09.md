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
not the gate for this branch. Multiple explicitly admitted zero-retry attempts ran
today; each returned a typed null and an encoded fix, and each torn down to
provider zero with staged objects absent.

| Run | Instance | Cost | Retained null and encoded fix |
| --- | --- | ---: | --- |
| v8 | 47282158 | `$0.061326` | Released `optimize_usd` has no local Scene Optimizer backend. Fix: ship the pinned Apache-2.0 `scene_optimizer_core_usd_25.11_py_3.12` v1.0.3 package in-bundle and export `WU_SO_PACKAGE_DIR`. Also registered `adp_joint_agent_result.json` as a runtime-result filename, which would have falsely blocked even a successful run. |
| v9 | 47283980 | `$0.126088` | The OvRTX daemon hung for the full readiness window with stderr drained invisibly. Fix: construct `ovrtx.Renderer()` directly before service start with retained diagnostics; add a machine avoidlist to the lane. Separately: watchdog close waited 45 s for process exit while absence was confirmed at +164 s, so a correct teardown was recorded as unclosed — close now polls the evidence (this also removes the same false blocker from the sibling excision lane). |
| v10 | (torn down) | `$0.082770` | Scene Optimizer subprocess died in 0.04 s and the released task quarantines every exception into the bare string "USD optimization failed". Fix: probe the optimizer directly and retain its real stdout/stderr. |
| v11 | (torn down) | `$0.118768` | The probe worked and named the cause: `libtbb.so.12: file too short`. `python -m zipfile` materializes the package's 35 shared-library symlinks as text stubs. Fix: extract with `unzip`, verify no sub-1 KB `.so` stubs, and treat a probe result carrying an error or zero executed operations as a failure. |
| v12 | (torn down) | `$0.058695` | The retained provider log proves this was not an unexplained transient: the launch requested 96 GB but native `df` readback exposed a 10 GB overlay with 9.5 GB free. Installing the 2.4 GB compressed OVRTX wheel exhausted it (`No space left on device`). Fix: paid offer selection now rejects advertised capacity below an explicit request, and the immutable Joint Agent bundle performs a native 32 GiB free-space gate before any large dependency install. |
| v13 | 47313565 | `$0.020477` | The disk fix worked: the selected offer reported 580.8 GB available, the instance delivered the requested 96 GB, native readback passed, and the optimizer package retained all 35 symlinks. The released Python editable install then failed before inference because uv bypassed the host's working package mirror and rejected `files.pythonhosted.org` with `UnknownIssuer`. Fix: freeze every local project's PEP 517 build requirements and pyproject digest, install that complete plan first, disable build isolation for the already-provisioned local projects, enable uv native TLS, and bridge the provider's configured `PIP_INDEX_URL` into `UV_DEFAULT_INDEX`. Instance destroyed, staged objects absent, provider-zero API count 0, retry cap 0. |
| v14 | 47314288 | `$0.025748` | The 96 GB native disk gate again passed and the bundle reached the new build-plan validator. Validation stopped before dependency installation because the launcher invoked the helper with the host system `python3`, which predates the helper's standard-library `tomllib` dependency, instead of the already-created pinned Python 3.12 environment. Fix: execute the digest and requirement validator through the bundle's exact `.venv/bin/python`; retain a shell contract test so no provider helper can silently fall back at this seam. Instance destroyed, staged objects absent, provider-zero API count 0, retry cap 0. |
| v15 | 47314835 | `$0.023332` | The pinned Python 3.12 build-plan validation and complete declared build-requirement install passed. NVIDIA's Joint Agent then failed while building an editable package because Hatchling's editable hook imports undeclared `editables`. This is a developer-install mode, not a runtime requirement. Fix: immutable provider bundles now build/install ordinary local wheels with build isolation disabled; editable source installs are forbidden by a focused launcher contract test. The Joint instance was destroyed and staged objects are absent; the exact concurrent SimReady instance remained authorized and untouched, retry cap 0. |
| v16 | 47315320 | `$0.023202` | Ordinary top-level wheel installation still honored NVIDIA's checked-in `[tool.uv.sources] world-understanding = { editable = true }` transitive source rule, so Hatchling again entered its editable hook and exposed the same undeclared `editables` import. Fix: the prelaunch build planner now scans every bound project's uv source declarations and materializes Hatchling's dynamic editable backend requirement before any local project build. This closes the full dependency family rather than special-casing a package name. The Joint instance was destroyed, staged objects are absent, and the authorized SimReady instance remained untouched, retry cap 0. |
| v17 | 47315803 | `$0.090825` | The complete dependency plan passed, OVRTX 0.3.0.312915 initialized on the RTX 4090, four construction previews rendered, and the released Joint Agent reached identification and structure analysis for the first time. Hosted NIM inference then returned authentication failures and no topology candidate. An independent one-token probe proved the root cause: `/v1/models` is public and had falsely qualified the stored key, while exact `chat/completions` returned HTTP 401. Fix: hosted-model admission now requires a one-token inference response; model backend/identity is a digest-bound replaceable input, and the already-proven OpenAI `gpt-4.1` path can be selected without changing topology review. The instance was destroyed, staged objects are absent, provider-zero API count 0, retry cap 0. |
| v19 | 47318232 | `$0.081835` | The OpenAI `gpt-4.1` model preflight, generic allocator binding, complete dependency plan, 96 GB disk gate, OVRTX install, renderer construction, and GPU warm-up all passed. The provider runner then stopped before its optimizer probe because it still required `NVIDIA_API_KEY` unconditionally even though the frozen model backend was OpenAI; its coarse catch retained only `joint_agent_runtime_exception:ValueError`. Fix: the runner now derives the credential environment from the frozen backend, returns the exact typed credential/backend blocker before GPU work, and the NIM compatibility resolver prefers the inference-specific key over the NGC registry key. Instance destroyed, staged objects absent, provider-zero API count 0, retry cap 0. |
| v21 | 47319708 | `$0.052768` | The backend-specific credential reached the released CLI, the optimizer probe passed, and a real `joint_agent_inference.log` was retained. Released config validation then found an unconverted `steps.identify_asset.vlm.backend='nim'`; the builder had rewritten only three named model steps. Fix: recursively rewrite every `llm`/`vlm` node while preserving unrelated options, including nested future steps. Local evidence closeout also exhausted the Mac disk after provider teardown; provider and object-store zero were already proven. Fix: require 2 GiB local evidence headroom before paid mutation, reserve 16 MiB for teardown receipts, close the watchdog before output extraction, and provide a fail-closed recovery receipt joined to teardown, object absence, watchdog cancel, and fresh API provider-zero. The local v21 closeout was recovered through that contract. |
| v22 | none | `$0.000000` | The allocator admitted the clean bundle, then the runtime correctly stopped before staging or provider mutation because a `$0.80` cap funds only 60 minutes at the admitted rate, below the 90-minute dependency-heavy minimum. The mutation-free dry run had failed to evaluate this same live-window constraint. Fix: local capacity and remaining live minutes are now evaluated identically in dry and execute modes. |
| v23 | 47321782 | `$0.077329` | The new admission checks passed with 9.34 GB local headroom and exactly 90 funded minutes. Python dependencies, 96 GB provider capacity, Scene Optimizer, OvRTX installation, renderer construction, three-minute warm-up, and released CLI dry run all passed. Execution validation still reported `steps.identify_asset.vlm.backend='nim'`: the released BYOA YAML omits that node, then NVIDIA's loader injects a NIM default, so recursive rewriting had no serialized node to rewrite. Fix: materialize every model consumer in the pinned released schema before recursively binding the frozen backend/model; the test fixture now omits `identify_asset.vlm` to reproduce the real default. Instance destroyed, staged objects absent, provider-zero API count 0, retry cap 0. |
| v24 | 47323204 | `$0.101801` | The implicit-backend fix worked: all four effective model nodes were explicitly OpenAI-bound, all eight released Joint Agent steps completed, 28/28 component predictions returned, and 11 articulation candidates were retained. Only 1 candidate was marked ready, 10 required review, and the bounded asset-level reconciliation response failed JSON parsing twice, so the model output is not admitted topology truth. Blueprint then failed before deterministic review because its postprocessor guessed `joint_agent_work/optimize_usd/...`; released v0.5.2 writes `joint_agent_work/optimized/...`. Fix: resolve released outputs by a strict non-symlink, nonempty, exactly-one role contract and test the real layout plus ambiguous-output rejection. Instance destroyed, staged objects absent, provider-zero API count 0, retry cap 0. |

Retained Joint Agent spend today is `$0.944964`. Combined retained program
spend is `$6.075253` against the `$12` authority. Provider inventory was
`[]` before and after every single-instance launch. v15 used the user's explicit
second-GPU authority: the allocator admitted exactly the already-running labeled
SimReady instance, destroyed only the Joint instance, and read back no unexpected
third resource.

## Claims

| Classification | Claims |
| --- | --- |
| Implemented | Suppression volumes with proven index, byte, and coverage equivalence to the sealed deletion, plus demonstrated reversibility; deterministic source-to-topology derivation; articulated physics authoring with observed-handle group binding and labeled generated interior; two fail-closed static validators; twelve-state door clearance with bindable static obstacle classes; scenario admission requiring the bound matrix; Franka base placement search with typed rejection histogram; external/wrist/review camera resolution; digest-bound excision join seam with fail-closed inpainting-policy resolution; Joint Agent failure-path evidence retention, optimizer provisioning, on-host renderer probe, watchdog evidence polling, and machine avoidlist. |
| Generated SimReady candidate | One exact SimReady articulated refrigerator USD, `sha256:a673d2e4…`, produced by the NVIDIA Physics Agent from the Blueprint-authored candidate `sha256:f626998b…`, statically re-admitted on topology and physics with the articulation and authored link masses intact, and with its Franka base, twelve-state clearance, and three cameras resolved. |
| Native-simulator qualified | **None.** The probe inputs are frozen and the readback is now a required gate, but no blank-stage Isaac diagnostic, joint/limit/lock readback, contact stability, penetration, reset replay, or deterministic final-state check has executed. |
| Blocked or abstained | The earlier Content Agents Material and Texture typed nulls were superseded by the retained v5 outputs documented below. Joint Agent inference completed in v24 and its 11-candidate document is retained, but owned-core topology remains a typed abstention: the candidate quality is review-heavy and Blueprint's guessed optimized-output path prevented deterministic bounds/review/publication. The strict output-role resolver is encoded and locally covered but has not yet been exercised by a fresh zero-retry run. Zero-action and scripted-positive controls have not run. No learned-policy episode was prepared or launched. Gaussian excision/coverage join is implemented but unexecuted pending the sibling branch's owned-index result. |
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

`accept_agent_enriched_asset` now decides whether any agent output may replace
the candidate. Nothing in the earlier runs checked this: one pass applied
per-component mass to twenty-six prims and another rebound every surface, with
no verification that the joint graph, the authored link masses, or the
reachable interior survived. Agents may add materials, textures and priors
freely; changing the joint graph, moving an authored link mass, sealing the
interior, or dropping a render binding rejects the enrichment. And
`interior_exposure` is now a required articulated construction gate, so a
sealed twin cannot reach scenario materialization at all.

Render-ready candidate `sha256:81b1796a...`, manifest resealed at
`sha256:68d78ba6...`. The local config preflight passes on the rebuilt bundle
`sha256:9d2d9221...`.

## The rerun: all four agents completed

With the aperture cut and four bound preview surfaces in place, the pass was
rerun (`840796_v5`, instance `47310948`, `$0.19232`, torn down to provider
zero). **Material, Texture, Physics and Validation all completed
successfully.** The single missing render material had been the root cause of
both earlier failures, and the stale 840313 prompt had been the reason the
classifier answered `Car_Paint_Green`.

The Texture Agent produced a real PBR set - albedo, roughness, metalness,
normal and ORM - and the albedo reads as a pale off-white panel with faint
vertical brushed grain, sampling to `(0.904, 0.879, 0.818)` against the
`(0.810, 0.782, 0.762)` measured from the removed splats.

`accept_agent_enriched_asset` admits both the textured and the physics output:
articulation preserved, authored link masses unchanged, interior still exposed,
render materials still bound. Textured candidate `sha256:5574c5f0...`, seal
`manifests/second_scene_840796_textured_simready.v1.json`.

One honest note about the runner: its terminal record is still `blocked`
without a runtime result even though all four agents completed, so the
acceptance evidence here comes from reading the retained outputs directly
rather than from the runner's own verdict.

## What the interior actually is

Worth stating precisely, because "there is an interior" overstates it. The
cavity is **one six-face box** with all normals facing outward - a solid inset
block, not a hollow liner. It has no shelves, bins, drawers or door pockets and
no child prims. Opening the door now reveals its recessed front face rather
than the carcass front, which is a genuine improvement, but a gripper could not
place anything inside it. Its appearance is a generated `interior_liner`
material at `(0.90, 0.91, 0.92)`; its physics material is the agent-assigned
`PhysMat_sf0_60_df0_45_r0_30`. The real interior was never observed - the
refrigerator is closed in every frame of the scan - so all of it stays tagged
`generated_candidate_geometry`.

## Smallest next action

Run the blank-stage Isaac/PhysX diagnostic against
`simready_candidate_deterministic.usda` and read back the articulation root,
joint count and types, upper/lower joint identity, axis and limits, the locked
lower joint, upper-door motion through 55 degrees, contact stability, absence
of initial penetration, and reset replay. That single native run converts the
statically admitted candidate into a native-qualified one or produces the exact
typed blocker. The Joint Agent comparison arm can retry independently; it does
not gate this path.
