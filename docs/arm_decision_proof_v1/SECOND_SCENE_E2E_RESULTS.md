# Second-scene Arm Decision Proof v1 rehearsal results

Status: **terminal completion path B — typed, evidence-backed abstention before
SimReady construction**. Scene/task selection, registration, exact removal
inputs, two released Aura executions, one Joint Agent execution attempt, and
one frozen FlashSplat held-out ownership audit are sealed. The first Aura run used a reference-camera index inconsistent with the
released runtime's filename-sorted camera order. One explicitly authorized,
zero-retry correction used the right camera but produced severe multiview visual
artifacts and outside-mask scene damage. The Joint Agent run reached no model
inference because its bounded OVRTX warmup did not complete. Therefore no
The later replacement-first Gaussian audit also abstained: its safe owned set
left a large refrigerator shell, while deleting the ambiguous set would damage
protected kitchen appearance. Therefore no articulated SimReady replacement exists, the scripted-positive control is not
admissible, and neither frozen learned candidate was evaluated. This is an
upstream construction result, not a policy null.

The current terminal receipt is
`manifests/second_scene_840796_task_evaluation_abstention.v3.json`; it supersedes
v2 while retaining v2's digest as lineage. Its digest is
`sha256:d7b37673ab1bd3041e8d904e672b05213d718ae0e1f457c8538bca8564728ac1`.

## Frozen selection and task

The outcome-blind first-passing selection rule chose InteriorGS/SAGE scene
`840796` (`0498_840796`), refrigerator instance `123`. The task is: “Open the
upper refrigerator door to at least 45 degrees, release it, and retreat.” The
upper hinge is the only commanded/scored joint; the lower hinge is locked with
a 0.001 rad motion tolerance. Success is a stable 45–55 degree upper-door state
for 40 samples at 15 Hz, speed no greater than 0.05 rad/s, released task
contact, no limit/collision/containment failure, and completed retreat. The
freeze digest is
`sha256:0774b7072bb1a0fd0929fde81cd48aaff384af4608772f31d603c2324f97d03f`.
The exact candidates remain `pi05_droid` and `groot_n17_droid`.

| Scene(s) | Result | Evidence-backed reason |
|---|---|---|
| 840076 | Rejected | Best SAGE door collider AABB IoU 0.200712895, below the frozen 0.85 whole-object threshold. |
| 840411 | Rejected | Exact SAGE triangle sweep contacts chair 227 at 26.75 degrees, before the required 45 degrees. Moving the chair would change the public scene. |
| 840796 | Selected | Full six-room survey; observed two-door refrigerator/handle; unique SAGE collider IoU 0.992325475; exact upper-door sweep clear through 45 degrees. |
| 840873, 840874, 840920, 841151, 841193, 841757 | Not inspected | Frozen first-passing rule stopped at 840796; no learned or inpainting outcome was consulted. |

## Rights, provenance, registration, and visibility

InteriorGS is bound to revision
`334dfeea4e0241033b4e5de97c01bc7c9c080530` under its custom terms;
SAGE-3D Collision Mesh is bound to revision
`3ba75cc7887b62bf84211d5db08adfa64d691597` under CC-BY-NC-4.0. This
rehearsal is private, noncommercial research. Raw InteriorGS dataset bytes are
not redistributable and were not uploaded. Explicit v2 authority permits only
minimum scene-derived Aura inputs and derived SAGE articulation inputs, at most
two concurrent paid resources, and USD 12 combined spend. It does not alter
publisher rights or authorize public disclosure.

| Source | Bytes | SHA-256 |
|---|---:|---|
| `0498_840796/3dgs_compressed.ply` | 36,382,397 | `sha256:1b8d0ff463a9a68d83ad10b5a49c3c5be9819b692005162382eaf30c733a8fdf` |
| `0498_840796/labels.json` | 132,539 | `sha256:57aa8675dbf7938d0b7e0194b5c581b37d049594ba6037d0a77552e6693c1484` |
| `0498_840796/structure.json` | 70,572 | `sha256:6563f49fe1266d477a90ac0007c18feaceb3397531968a1a28a8864477d74e8d` |
| `840796_collision.usd` | 23,988,263 | `sha256:7bfc6e4b6909f7057fa1983ac228e5d7511d7053745d8b6f467aecfea0c7194e` |
| Aura input receipt | — | `sha256:14835d010e3f15b610ff33ee8ae56f0daf8fa01b96aefe17db5ec36852a46f29` |
| Joint Agent source USDA | — | `sha256:4e0986e79b9d358de7e3289c0179eeaa4ec94b035e2133d2ea309298b525a4c3` |

Registration proves publisher-frame synthetic consistency only: right-handed,
Z-up, 1 meter/unit, identity InteriorGS-to-SAGE transform, zero round-trip
error, and target AABB IoU 0.992325475. It is not physical-site metrology.

The full known topology survey retained all 593,665 splats, 24 nonblank views
across six publisher rooms, then nine nonblank target close-ups. Explicit gaps
are the refrigerator interior, surfaces behind the refrigerator, closed cabinet
interiors, surfaces outside publisher room profiles, and occluded backs or
undersides absent from source observations. These are reconnaissance renders,
not evaluation-authorized frames.

## Gaussian ownership excision audit

The replacement-first removal audit froze eight calibrated masks and withheld
`far_left` plus `far_right` before classification. Released FlashSplat commit
`3e3b14786333bf0163ba1b8541e86a3765112d7d` and rasterizer commit
`189c483ffa33dd6d5661343ce496df0c6eb80a0c` executed on one bounded GPU attempt
after two retained infrastructure nulls. Six calibration cameras produced two
repetitions of per-Gaussian front-to-back alpha-times-transmittance evidence.
The repetitions had no classification-label disagreement, but their
six-decimal contribution arrays were not byte-identical. The pre-heldout frozen
aggregation policy therefore forced every disputed Gaussian to `ambiguous`.

The exhaustive result over 593,665 source Gaussians is 2,436 `owned`, 587,007
`retained`, and 4,222 `ambiguous`, compared with the historical 3,791-Gaussian
OBB selection. Every retained PLY vertex row remains byte-identical to its
source row. Exact-camera black/white pairs independently recovered the rendered
alpha of the OBB, owned, and ambiguous layers. This closed a reusable renderer
gap: background RGB is now validated, passed to the renderer, and bound in its
manifest.

The held-out ownership gate failed on both cameras:

| Camera | Owned silhouette recall | Missing pixels / largest component | Protected significant pixels | Ambiguous component inside mask | OBB recall / protected pixels |
|---|---:|---:|---:|---:|---:|
| `far_left` | 0.880886288 | 140,057 / 101,147 | 4,256 | 1,089,831 | 0.915459430 / 3,080 |
| `far_right` | 0.854582679 | 183,463 / 110,319 | 4,618 | 1,145,765 | 0.909697844 / 1,422 |

This explains the visible refrigerator shell. The unanimously owned layer
captures most of the appliance, but a refrigerator-sized connected component
remains in the ambiguous set. Deleting all ambiguous Gaussians would also
delete 395,382 and 583,526 significant protected pixels on the two held-out
views. The new method is also worse than the OBB baseline on both frozen
comparison axes. The result is therefore a typed abstention at
`calibrated_gaussian_ownership_separation_without_protected_scene_deletion`.
The USD coverage sweep is not authorized by this audit, and no USD was inserted
to conceal the failure.

The checked-in abstention is
`manifests/second_scene_840796_gaussian_excision_heldout_abstention.v1.json`.
The runtime receipt digest is
`sha256:a4ab16cbbb6630dc3e9dcb4c8ceee4cbb90de298aba0dd416bff5241057e6336`
and its file digest is
`sha256:c2cc4f5b4faa14efc46ffaeb793c46197453c4e3f017d10137e06e808a2b92eb`.
All eight six-column contact sheets are under
`/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/gaussian_excision/840796_v2/heldout_audit_v1/evaluation/contact_sheets`.

## Object removal and articulation executions

AuraFusion360 is bound to source commit
`f23b26c44ba84608306ba952510533ebf4c7877d`, tree
`cc8447c66448b29bb4d39fec29c031df63d4b179`. Big-LaMa reference generation,
30k training, exact-camera rendering, removal, SAM2 masks, inpaint
initialization, SDEdit, and 10k fine-tuning all exited zero. The retained output
has 493,872 surfels, 126,432,821 bytes, digest
`sha256:4da6fd6ef384dc4b344cde6474b28d5faa66ffb9ef430bac5e6c6b5c611a7f57`,
plus eight exact-camera PNGs.

The first run does **not** qualify removal. Its adapter configured `front_medium` as
reference index 0, while the released Aura COLMAP loader sorts camera names and
places `front_medium` at index 2. The retained front view also visibly contains
large translucent/smeared geometry in the removed-refrigerator region. No
project-owner visual acceptance or technical admission was asserted.

Outside a 16-pixel-dilated target mask, the measured eight-view aggregate is
22.734145 dB PSNR, 0.96562979 windowed SSIM, 0.03185024 mean absolute error,
and 0.14463770 fraction of pixels with maximum-channel change above 20/255.
This retrospective measurement has no admission effect and no frozen quality
threshold.

The corrected adapter digest is
`sha256:bfdc2500d324b44e014c030b6c2bb3e6593e4f5d9797d70f7e637ae654e48893`.
It binds the released filename-sorted order and `front_medium` index 2 and
suppresses the unretained 240-frame trajectory. All eight released stages exited
zero. The final point cloud has 1,134,299 vertices, 290,382,134 bytes, digest
`sha256:0febc280ed539a2d1ffac11a00bf07ae62bfd09a81659caf614dc8cefe1ca965`.
This fixes the camera-index defect but **does not fix the Aura result**.

All eight final renders are unusable: four contain large black/rainbow Gaussian
explosions; the others contain large smears and semantic hallucinations,
including a person-like figure in the removed volume. The exact-camera locality
receipt measured 21.500983 dB mean outside-mask PSNR, 0.95173965 windowed SSIM,
0.03423921 mean absolute error, and 0.15807323 of preserved pixels changing by
more than 20/255. Per-view PSNR spans 19.013–23.318 dB. Because thresholds were
not frozen before this retry, those metrics have no success-admission effect;
the digest-bound visual rejection independently keeps the claim ceiling at
`rejected_visual_candidate_only`. The run did not retain inpaint-init renders or
SDEdit images, so the exact failing stage cannot be localized retrospectively.

The Joint Agent is bound to USD Content Agents v0.5.2, commit
`36dbf3f274f8e256637230a05a085853f65cc175`, with a 28-component source asset and packet digest
`sha256:53deb1ad7fa36484fca4081ed181299c594c48b2d9851f87de829f0dd6563153`.
The zero-retry paid attempt reached no inference and produced no owned-core
topology because the bounded local OVRTX PT/128 64x64 warmup did not complete.
Its smallest blocker is `joint_agent_local_ovrtx_renderer_not_ready`.

## Runs, cost, and teardown

| Run | Mutation | Result | Cost | Teardown |
|---|---:|---|---:|---|
| Local selection, topology survey, registration, exact SAGE sweep, masks, source extraction, packet construction, compilers, and hermetic regressions | Local | Retained by digest-bound manifests and tests | USD 0 | Local only |
| Aura `second_scene_840796_v4_execute`, Vast 47226054 | Paid | Released workflow completed; quality abstention at runtime reference-camera binding | USD 2.372684 adapter estimate | Instance destroyed; staged objects absent; provider inventory empty |
| Aura corrected `second_scene_840796_v5_retry_execute`, Vast 47244918 | Paid, explicitly authorized attempt 2 | Released workflow completed with correct reference binding; rejected for Gaussian explosions, hallucination, multiview inconsistency, and outside-mask damage | USD 1.774042 adapter estimate, below USD 3.50 attempt cap | Instance destroyed; staged objects absent; independent watchdog terminal; provider inventory `[]` |
| Joint Agent `840796_v7_execute`, Vast 47232529 | Paid | Typed null before inference: `joint_agent_local_ovrtx_renderer_not_ready` | USD 0.133936 exact retained cost | Instance destroyed; staged objects absent; continuing spend false |
| Gaussian contribution attempt 1, Vast 47281712 | Paid | Import closure null before scientific execution; retained and fixed generically | USD 0.068951 exact retained cost | Instance destroyed; provider-zero verified |
| Gaussian contribution attempt 2, Vast 47283773 | Paid | Previously known-bad host vanished before code; host avoidance gap fixed generically | USD 0.005121 exact retained cost | Instance destroyed; provider-zero verified |
| Gaussian contribution attempt 3, Vast 47284587 | Paid | Released FlashSplat executed; two repetitions and all outputs retained | USD 0.084794 exact retained cost | Instance destroyed; staged objects absent; watchdog terminal; strict provider-zero verified |

Corrected-retry local preflight retained two non-paid nulls rather than hiding
them. The first bundle build supplied the SAM2 package subdirectory instead of
the repository root and failed before upload or provider mutation. The corrected
build produced a 495 MB bundle, digest
`sha256:931f764210cd87d21428e8719c3978d260d98f157a0dbb17d04b5e4a97491581`.
The first allocator dry-run used a mistyped full commit SHA and failed closed
with zero provider mutations; the exact-SHA dry-run then admitted the immutable
commit, input digests, watchdog, TTL, cap, and attempt authority. Provider API
inventory was `[]` immediately before launch and after teardown.

Combined retained spend is USD 4.439528 using the two Aura adapter estimates
plus exact Joint Agent and Gaussian-excision costs, below the original USD 12
goal authority; the corrected Aura attempt also remained below its separate
USD 3.50 cap. Vast's terminal Aura budget ledger explicitly reports
`actual_cost_usd: null` and
`actual_cost_source: not_available_from_instance_probe_api`; a filtered invoice
query immediately after teardown returned no posted line. Therefore an exact
Aura billed amount is unresolved rather than fabricated. There was exactly one
freshly authorized paid Aura retry and no automatic retry. Final provider
inventory was `[]`.

Aura stage runtimes were 42.20 s reference generation, 3,868.85 s training,
1,144.42 s render, 1,080.78 s removal, 9.54 s SAM2, 1,373.16 s inpaint init,
155.01 s SDEdit, and 3,769.74 s fine-tuning. The run exposed four unretained
240-frame trajectory exports. Commit `e41483749` disables those optional exports
for future adapters while retaining all exact-camera evidence.

Corrected-run stage runtimes were 61.66 s reference generation, 3,746.67 s
training, 29.77 s rendering, 36.78 s removal, 7.84 s SAM2, 146.81 s inpaint
initialization, 140.02 s SDEdit, and 2,422.85 s fine-tuning. The terminal
fine-tune loss was 0.089397. Runtime completion is not visual qualification.

## Controls, candidates, media, and seal

No second-scene control or learned-policy episode ran. This is required by the
protocol: without qualified object removal and a retained Joint Agent topology,
there is no exact articulated SimReady replacement, native placement, or
scripted-positive control to admit. Candidate nulls, ties, or rankings are not
inferred.

Retained media and navigation:

- Before images and exact masks:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/inpainting_inputs/840796_refrigerator_v2/{images,masks}`
- Aura after frames, native render manifest, locality receipt, point cloud, and
  stage logs:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/aura_interiorgs/second_scene_840796_v4_execute/immutable_execution`
- Corrected Aura frames, point cloud, logs, exact-camera locality receipt, and
  corrected Big-LaMa reference:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/aura_interiorgs/second_scene_840796_v5_retry_execute`
- Gaussian ownership indices, retained scene, paired alpha renders, exact
  held-out receipt, and all eight contact sheets:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/gaussian_excision/840796_v2`
- Reconnaissance survey and target closeups:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/selection_inspection_second_scene/840796`
- Finder-friendly terminal evidence index:
  `evidence/second_scene_840796/OPEN_ME_episode_evidence_index.html`
- Authoritative index:
  `evidence/second_scene_840796/episode_evidence_index.v1.json`, digest
  `sha256:51c9cba8b4469fe60772f5609932c3c3aa1358d75e2f5b988f21af473bf08c92`

The index truthfully contains zero episode videos or scores and names the
pre-episode abstention. External, wrist, and overview videos do not exist
because no episode was admitted.

## Landed changes

| Commit | Change |
|---|---|
| `4d968c996` | Merge the latest published ADP-009D control harness. |
| `f3fa149b9` | Prevent absolute-position control starvation; cover 840313 and 840796. |
| `265373d5d`, `e1a18ba00`, `85aa92d6a`, `14733d48b`, `9637eb08e`, `e675edcff` | Retain, render, validate, and bundle Joint Agent construction evidence. |
| `51e5506d3` | Close watchdogs with explicitly authorized sibling GPUs. |
| `68b230ebc` | Freeze execution authority and the customer parallel-DAG contract. |
| `11a51f841` | Generalize Aura review bindings beyond scene 840313. |
| `acddc9ee2`, `41ed69ca3` | Join success or abstention from observed articulated construction execution. |
| `fc5efeadf`, `90109b879`, `109a6ae3f` | Produce portable episode/abstention evidence indexes and terminal Task Evaluation abstentions. |
| `ca8542786`, `46ba28d69`, `57876257d` | Generalize scorer, learned episodes, and native controls to articulated tasks while retaining rigid fixtures. |
| `8799d03e3` | Seal Joint Agent zero-retry execution abstentions. |
| `e41483749` | Skip unretained Aura trajectory rendering. |
| `a3eebc13e` | Derive Aura reference index from the released runtime camera order. |
| `117bcb06b` | Retain and propagate Aura runtime reference-binding abstentions. |
| `cf75480a3` | Seal terminal manifests, results, media index, and the corrected next action. |
| `1dc2877ff` | Bind the exact corrected Aura adapter, one-attempt authority, cap, TTL, and no-retry rule. |
| `141d54ca0` | Seal the corrected execution and visual rejection; retain future inpaint-init/SDEdit evidence; propagate the rejection through the terminal run. |
| `473235ba9`, `06d098812`, `bd51b42c4`, `3a17d0c37`, `9f0cce2ec` | Close the released FlashSplat runtime, host-routing, contribution-determinism, and conservative aggregation gaps with hermetic regressions. |
| `24cd42cbc` | Bind exact background RGB in every exact-camera splat render. |
| `adf24fb04` | Add paired-background alpha recovery, frozen held-out evaluation, OBB comparison, and portable six-column contact sheets with 840313 and 840796 fixture coverage. |

Before every commit, the required focused repository gate and Ruff gate passed.
The corrected-result gate passed 1,016 tests with 1 skipped and 9,151
deselected in 56.53 seconds; Ruff passed over `src/` and `tests/`.

## Claim ledger

| Classification | Claims |
|---|---|
| Implemented | Outcome-blind scene selection; topology-first survey; exact SAGE sweep admission; bounded 1–4-joint task scoring with one commanded joint; task-neutral controls and learned episode contracts; private paid-resource admission; zero-retry execution receipts; portable zero-episode abstention index; parallel customer DAG; runtime-sorted Aura reference binding; elimination of unretained trajectory renders; digest-bound visual rejection; future inpaint-init and SDEdit intermediate retention; full-resolution locality memory bounding; released FlashSplat runtime closure; three-way Gaussian ownership; byte-exact retained rows; conservative repetition aggregation; exact-background render binding; paired-background alpha audit; held-out OBB comparison and contact sheets. |
| Simulator-qualified | None for scene 840796. No native articulated replacement or control cell was admitted. |
| Blocked or abstained | Gaussian ownership abstained at `calibrated_gaussian_ownership_separation_without_protected_scene_deletion`: the owned layer leaves a large appliance shell, the ambiguous set contains substantial protected-scene contribution, exact contribution arrays were nondeterministic, and the method underperformed OBB on both held-out views. Corrected Aura output abstained at `aurafusion360_interiorgs_visual_artifact_rejection`; exact failure-stage localization is unavailable because that completed run predates intermediate retention. Joint Agent abstained at `joint_agent_local_ovrtx_renderer_not_ready`; SimReady replacement, Franka/IK/contact/camera gates, controls, matrix, both policies, episode media, comparison, and replay seal were not reached. |
| Physically unresolved | Appearance/collision registration as physical truth; hidden refrigerator/background truth; replacement physical equivalence; real refrigerator/Franka performance; partner fidelity; deployment readiness; physical candidate ranking. |

Exact claim ceiling: a private public-dataset construction rehearsal with two
retained upstream nulls. It does not qualify a partner capture, real-site
fidelity, deployment readiness, physical performance, or a learned-policy
comparison.

Single next action: freeze a new, outcome-unseen confirmation camera pair and
implement a calibration-only iterative ambiguous-promotion rule that may
promote a Gaussian only when it increases refrigerator-mask coverage without
increasing protected contribution. Confirm the frozen method once on the fresh
pair before authorizing replacement-depth coverage. Do not tune against the
already revealed `far_left` or `far_right` views and do not delete the whole
ambiguous set.
