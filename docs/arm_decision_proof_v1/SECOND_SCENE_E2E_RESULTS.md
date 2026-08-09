# Second-scene Arm Decision Proof v1 rehearsal results

Status: **terminal completion path B — typed, evidence-backed abstention before
SimReady construction**. Scene/task selection, registration, exact removal
inputs, one released Aura execution, and one Joint Agent execution attempt are
sealed. The Aura run used a reference-camera index inconsistent with the
released runtime's filename-sorted camera order; the Joint Agent run reached no
model inference because its bounded OVRTX warmup did not complete. Therefore no
articulated SimReady replacement exists, the scripted-positive control is not
admissible, and neither frozen learned candidate was evaluated. This is an
upstream construction result, not a policy null.

The terminal receipt is
`manifests/second_scene_840796_task_evaluation_abstention.v1.json`, digest
`sha256:a32fb5c762e4fbeb46bb24f8b1308e765dc18f972438d6dde55d4452e36c83fa`.

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

## Object removal and articulation executions

AuraFusion360 is bound to source commit
`f23b26c44ba84608306ba952510533ebf4c7877d`, tree
`cc8447c66448b29bb4d39fec29c031df63d4b179`. Big-LaMa reference generation,
30k training, exact-camera rendering, removal, SAM2 masks, inpaint
initialization, SDEdit, and 10k fine-tuning all exited zero. The retained output
has 493,872 surfels, 126,432,821 bytes, digest
`sha256:4da6fd6ef384dc4b344cde6474b28d5faa66ffb9ef430bac5e6c6b5c611a7f57`,
plus eight exact-camera PNGs.

The run does **not** qualify removal. Its adapter configured `front_medium` as
reference index 0, while the released Aura COLMAP loader sorts camera names and
places `front_medium` at index 2. The retained front view also visibly contains
large translucent/smeared geometry in the removed-refrigerator region. No
project-owner visual acceptance or technical admission was asserted.

Outside a 16-pixel-dilated target mask, the measured eight-view aggregate is
22.734145 dB PSNR, 0.96562979 windowed SSIM, 0.03185024 mean absolute error,
and 0.14463770 fraction of pixels with maximum-channel change above 20/255.
This retrospective measurement has no admission effect and no frozen quality
threshold.

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
| Joint Agent `840796_v7_execute`, Vast 47232529 | Paid | Typed null before inference: `joint_agent_local_ovrtx_renderer_not_ready` | USD 0.133936 exact retained cost | Instance destroyed; staged objects absent; continuing spend false |

Combined retained spend is USD 2.506620 using the Aura adapter estimate, below
the USD 12 authority. Vast's terminal budget ledger explicitly reports
`actual_cost_usd: null` and
`actual_cost_source: not_available_from_instance_probe_api`; a filtered invoice
query immediately after teardown returned no posted line. Therefore an exact
Aura billed amount is unresolved rather than fabricated. There were no paid
retries. Final provider inventory was `[]`.

Aura stage runtimes were 42.20 s reference generation, 3,868.85 s training,
1,144.42 s render, 1,080.78 s removal, 9.54 s SAM2, 1,373.16 s inpaint init,
155.01 s SDEdit, and 3,769.74 s fine-tuning. The run exposed four unretained
240-frame trajectory exports. Commit `e41483749` disables those optional exports
for future adapters while retaining all exact-camera evidence.

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
- Reconnaissance survey and target closeups:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/selection_inspection_second_scene/840796`
- Finder-friendly terminal evidence index:
  `evidence/second_scene_840796/OPEN_ME_episode_evidence_index.html`
- Authoritative index:
  `evidence/second_scene_840796/episode_evidence_index.v1.json`, digest
  `sha256:6f0a4b5a5f3c728e57e8997c34a8360d2e9598c3e1cc9f2f9395afaca6a93bc4`

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

Before every commit, the required focused repository gate and Ruff gate passed.
The terminal evidence gate passed 1,007 tests with 1 skipped and 9,150
deselected in 103.90 seconds; Ruff passed over `src/` and `tests/`.

## Claim ledger

| Classification | Claims |
|---|---|
| Implemented | Outcome-blind scene selection; topology-first survey; exact SAGE sweep admission; bounded 1–4-joint task scoring with one commanded joint; task-neutral controls and learned episode contracts; private paid-resource admission; zero-retry execution receipts; portable zero-episode abstention index; parallel customer DAG; runtime-sorted Aura reference binding; elimination of unretained trajectory renders. |
| Simulator-qualified | None for scene 840796. No native articulated replacement or control cell was admitted. |
| Blocked or abstained | Aura output abstained at `aurafusion360_runtime_reference_camera_binding_mismatch`; Joint Agent abstained at `joint_agent_local_ovrtx_renderer_not_ready`; SimReady replacement, Franka/IK/contact/camera gates, controls, matrix, both policies, episode media, comparison, and replay seal were not reached. |
| Physically unresolved | Appearance/collision registration as physical truth; hidden refrigerator/background truth; replacement physical equivalence; real refrigerator/Franka performance; partner fidelity; deployment readiness; physical candidate ranking. |

Exact claim ceiling: a private public-dataset construction rehearsal with two
retained upstream nulls. It does not qualify a partner capture, real-site
fidelity, deployment readiness, physical performance, or a learned-policy
comparison.

Single next action: regenerate the sealed Aura adapter using the now-landed
runtime-sorted camera order and unretained-media suppression; any new paid
execution is a retry and requires fresh explicit zero-retry authority.
