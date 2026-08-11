# Deformable Scene End-to-End Results

Status: construction in progress; no learned-policy outcome exists

Run: `adp-deformable-scene-e2e-v1`

Scene: InteriorGS/SAGE `840873`

Task: “Pick up the towel, place it inside the open basket, release it, and retreat.”

This is a development-only public-dataset simulator rehearsal. It does not
qualify partner capture, physical towel fidelity, real Franka performance,
deployment readiness, or customer value.

## Current gate

The run has not reached a policy evaluation. Exact static intake of the
user-supplied Lightwheel-derived rolled-towel USD and a paid-free pinned-PhysX
preparation plan/package are complete. Independent review is now closing the
last local preparation boundaries: live cooked-topology/schema/configuration
readback, confined stage composition/output assets, bounded descriptor-relative
output snapshots, and parent-directory race protection. None requires GPU
execution or changes the frozen towel geometry.

The frozen Isaac Lab/Arena backend can load and read volumetric deformable
state, but its currently implemented evaluation seam does not yet expose
qualified rigid--deformable contact-pair attribution and normal force. The
blank-stage canary must test that native capability against the prepared towel;
if unavailable it will retain the typed blocker
`native_rigid_deformable_contact_attribution_unavailable`. That would be a
native capability abstention, not a learned-policy failure. Backend re-freeze
still requires cook/load, genuine Franka contact/lift/release, reset, cameras,
and both frozen policy adapters.

## Selected scene and objective reason

Scene `840873` was selected outcome-blind because it was the only scene in the
complete known local catalog with an observed rolled-towel design basis, an
observed rigid open-basket design basis, exact InteriorGS appearance, exact SAGE
collision, plausible Franka work areas, and admissible public-dataset rights.
Scenes `840313` and `840796` were rejected as previously used. The other ten
catalog scenes were rejected for missing an admitted movable semantic, missing
a compatible destination, or lacking a same-publisher-room pair. The complete
inventory and objective rejection reasons are frozen in
[`scene_catalog_receipt.v1.json`](deformable_scene/scene_catalog_receipt.v1.json).

## Frozen construction

- Source towel instance `79` and source basket instance `87` remain unchanged as
  unscored background.
- No source Gaussian or collider is removed, and no inpainting is used.
- The scored towel and basket are separate inserted task entities.
- The source basket is occupied. Its hidden floor and physical wall/floor
  thickness are unresolved and are not promoted into engineered truth.
- The strict source-basket topology receipt reports 65 clear-prism obstructions
  and `open_collision_cavity_passed=false`.
- The engineered receptacle twin has its own open-top geometry, collision,
  provenance, pose, and claim ceiling. Native stability/collision qualification
  remains pending.
- The towel slot is frozen to an observed metric envelope of
  `0.370025854 x 0.119487516 x 0.113052810 m`.
- The supplied candidate archive is `6,466,445` bytes with SHA-256
  `3f12482c7964e6c45692bd0907ed8d15d8ec04f895c96e15e06465e676612fe0`;
  its USD is `359,684` bytes with SHA-256
  `5c08de54062fbe5eb823ced1ab666c5409fc24006cc27aecc861f69447c6e7de`.
- Exact static intake replay is retained as
  `external_simready_deformable_asset_inspection.v2.json`, with receipt digest
  `sha256:cb20acbba9afd52deaabe8c1655c1900e7951091ef93df5a4a20f1fe84490709`
  and status `pending_pinned_physx_conversion_and_native_qualification`.
- The paid-free pinned-PhysX preparation plan has digest
  `sha256:0f1faa251b4148077e79702d8c75bc58542253086b2fe90d1c18e3485219f47b`;
  its immutable local source-package receipt has digest
  `sha256:06c4a678506cc25aec71f28d4e0e1be511fb59a895aca6d1d774c1dd0e33a6e5`.
  Packaging does not qualify the cook or authorize provider disclosure.
- The local download-provenance xattr points to Lightwheel job
  `job_msnornab_95503606da`; this identifies the delivery origin but is not an
  output-license or provider-terms grant. The six-file archive contains no
  license document.
- Its selected visual mesh is a one-component closed oriented manifold with
  3,002 vertices, 6,000 triangles, and observed source dimensions
  `0.2001516313 x 0.1229738749 x 0.1167533172 m`. The frozen metric bake scale
  is `[1.8487276451, 0.9716495971, 0.9683049070]`.
- That scale yields a derived rest volume of approximately
  `0.003645153206 m^3`; retaining the provider-authored development density of
  `220 kg/m^3` would yield a derived simulated mass of approximately
  `0.801933705 kg`. Neither value is an observed physical-towel measurement.
- The source USD is not yet a qualified SimReady deformable: its experimental
  auto-cook representation has empty simulation/collision TetMeshes, lacks the
  literal pinned PhysX deformable body/material schemas, declares a static-rigid
  format, and includes an embedded DomeLight. A clean derived standard-PhysX
  runtime asset and its native cook/readback identity remain pending.
- This candidate is a volumetric soft-solid approximation. Thin-cloth behavior,
  unrolling, independent bend/shear fidelity, and physical towel equivalence
  remain explicitly unresolved.

Two composed cells are frozen:

| Cell | Basket XYZ m | Towel XYZ m | Franka base XYZ m | Status |
| --- | --- | --- | --- | --- |
| Canonical | `[-4.1, 1.9, 0.752873215]` | `[-4.1, 2.2, 0.748305425]` | `[-3.5, 2.1, 0.375]` | Geometry-plausible; native gates pending |
| Held-out relocation | `[0.9, 0.1, 0.061094195]` | `[0.7, -0.2, 0.056526405]` | `[1.0, -0.7, 0.375]` | Geometry-plausible; native gates pending |

The held-out cell is the authorized randomized composition test. It was chosen
deterministically from enumerated admissible supports before policy outcomes;
there is no runtime resampling or outcome-conditioned repositioning.

## Provenance and rights

| Source | Revision | Local bytes | SHA-256 | Disclosure ceiling |
| --- | --- | ---: | --- | --- |
| InteriorGS repository | `334dfeea4e0241033b4e5de97c01bc7c9c080530` | n/a | revision-bound | Terms of Use; raw scene bytes nonredistributable |
| InteriorGS appearance PLY | same | 30,670,612 | `b2fba405...` | Never upload raw bytes |
| InteriorGS labels | same | 198,732 | `30007f1d...` | Never upload raw bytes |
| InteriorGS structure | same | 35,839 | `a55eabf1...` | Never upload raw bytes |
| SAGE collision | `3ba75cc7887b62bf84211d5db08adfa64d691597` | 20,030,048 | `a3a2fb401...` | CC-BY-NC-4.0; attribution and noncommercial boundary |
| User-supplied Lightwheel-derived towel ZIP | generated-job revision not embedded | 6,466,445 | `3f12482c7964e6c45692bd0907ed8d15d8ec04f895c96e15e06465e676612fe0` | Local inspection admitted; generated-output license and external-upload rights unresolved |

The sealed evidence package must carry the complete digests, exact paths,
license-document digests, provider terms, derived-upload disclosures, and output
rights. The abbreviated values above are human-readable summaries only.

The current rights audit is retained in
[`840873_towel_79_lightwheel_rights_receipt.v1.json`](deformable_scene/840873_towel_79_lightwheel_rights_receipt.v1.json).
Lightwheel's public CC-BY-NC terms apply to assets distributed through its asset
page and GitHub repository; no release identity joins this generated job output
to that library. The supplied archive contains no license or notice, and no
official SimReadyGen generated-output ownership, cloud-copying, redistribution,
retention, deletion, or training terms were discoverable. Vast requires the
uploader to possess all necessary rights. Consequently the exact typed blocker
is `lightwheel_simreadygen_job_output_rights_receipt_missing`, and no towel bytes
may be uploaded until a job-specific terms receipt or written authorization
closes it.

## Reusable harness progress

Implemented and covered by hermetic regressions:

- stable `task_entities[]` normalization with repeated semantic roles and legacy
  rigid/articulated compatibility;
- entity-keyed Arena packet, scene plan, spawn plan, reset recipes, and runtime
  bindings;
- task-neutral rigid, articulated, and deformable scoring dispatch;
- deterministic deformable transfer predicates with containment, settle,
  divergence, strain, release, retreat, receptacle stability, and prohibited
  write/attachment checks;
- frozen zero-action and scripted-positive control contracts;
- camera entity-ID canonicalization, integer semantic planes, visibility, and
  edge-clearance gates;
- trusted execution-envelope and provider-return trust boundaries;
- episode evidence that requires a numeric outside start state, same-sample
  grasp pair/force, ordered post-contact displacement, later containment and
  release, action/inference/frame joins, and video-to-lossless-frame replay;
- paired canonical/held-out composed placement receipts; and
- one-shot static deformable capability preflight and packaging contracts.

The original rigid fixture, articulated refrigerator fixture, and synthetic
deformable-plus-receptacle fixture remain compatibility fixtures. No
towel-specific execution fork is accepted.

## Controls and candidates

| Episode | Outcome | Interpretation |
| --- | --- | --- |
| Zero action | Not executed | Awaiting native-qualified towel/backend in each exact cell |
| Scripted positive | Not executed | Harness blocker until genuine native grasp/contact is available |
| `pi05_droid` | Not executed | Frozen candidate; no policy verdict |
| `groot_n17_droid` | Not executed | Frozen candidate; no policy verdict |

No `never_moved`, failure, success, tie, or ranking claim exists yet.

## Cost and provider lifecycle

- Deformable Vast/GPU spend: `$0.00`.
- Palatial/Newton local attempt 001: retained free null, 120 requested CPU
  frames, hard timeout at 300 seconds, `$0.00`, scratch removed, receipt file
  SHA-256 `a159530bca09fccf07cd42dc0e6a097e876621e47861e323ad6f779f647164fb`.
- Palatial/Newton local attempt 002: changed one-frame request, completed in
  `39.314083` seconds, 24,978 cloth particles, one VBD solver step, finite
  before/after state, `$0.00`, scratch removed; receipt digest
  `sha256:fde37439067fdba02256b288f26a474571e8e6cb4efa6895a16afb162a103d42`,
  receipt-file SHA-256
  `3c9914f2b759f50b573aa9b1a6c7bc39af0995664fbaff76a2dcd5f3b0f7673b`.
- Separate Pan-Chera/Luna CAD-agent attempt: retained paid null, exactly one API
  call, no retry, `$0.010374`, no CAD artifact.
- No deformable-owned Vast instance has been launched.
- A read-only Vast API inventory at `2026-08-11T02:29:18Z` returned `[]`;
  provider zero was observed and no canonical launch-lock holder was present.
  This is a status checkpoint, not the future canary admission receipt. The run
  will re-query immediately before launch and require a fresh provider-zero
  receipt after final teardown.

## Claim table

| Claim class | Current result |
| --- | --- |
| Implemented | Multi-entity contracts, paired placement, scoring/control, camera, trust, evidence, and preflight code exist with hermetic fixtures |
| Simulator-qualified | Static towel geometry only; neither the inserted towel nor engineered basket is yet natively qualified in the frozen scene |
| Blocked/abstained | Direct source-USD deformable loading is incompatible; local stage/preparation verifiers are being hardened; exact generated-output upload rights remain unresolved; native rigid--deformable pair/force attribution is a pending canary gate |
| Physically unresolved | Towel material equivalence, hidden source-basket interior/thickness, real Franka behavior, site fidelity, and sim-to-real transfer |

## Remaining bounded work

1. Finish the live native-stage and descriptor-relative preparation-verifier
   repairs, then land and publish them after the mandatory focused/sentinel gate
   and independent red-team reviews are green.
2. Bind exact generated-output rights and upload authority to the already
   materialized static-ingest replay and clean pinned-PhysX source package.
3. Materialize and seal the production scene/task/rights/entity/camera/scenario
   pre-insertion receipts with the supplied candidate bound to the sole movable
   deformable slot.
4. After provider zero and upload-rights admission, run one Vast blank-stage
   cook/load and Franka grasp/release canary with a cap, TTL, watchdog, no retry,
   and teardown. The canary owns contact/reset/camera/adapter gates and cannot be
   promoted from worker-authored JSON.
5. Re-freeze the backend only if loading, contact, reset, cameras, and both
   adapters pass; otherwise seal the typed native-capability abstention.
6. If re-frozen, insert the qualified towel, run zero action then scripted
   positive in both cells, and only then evaluate both frozen policies.

The single next action is to finish, independently verify, and publish the
native stage/preparation gates. The next external action is the one authorized
blank-stage towel/backend canary after generated-output rights are bound and a
fresh launch admission again proves provider zero.
