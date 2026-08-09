# Second-scene Arm Decision Proof v1 rehearsal results

Status: **in progress; not a completed Task Evaluation Run**. The scene/task is
frozen, Aura execution is live under its watchdog, and the first Joint Agent
attempt is a retained typed null. No articulated SimReady asset, native scripted
positive, or learned-policy result is claimed yet.

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
| 840411 | Rejected | Exact SAGE triangle sweep contacts chair 227 at 26.75 degrees, before the required 45 degrees. |
| 840796 | Selected | Full six-room survey; observed two-door refrigerator/handle; unique SAGE collider IoU 0.992325475; exact upper-door sweep clear through 45 degrees. |
| 840873, 840874, 840920, 841151, 841193, 841757 | Not inspected | Frozen first-passing rule stopped at 840796; no learned or inpainting outcome was consulted. |

## Rights, provenance, and retained source identity

InteriorGS is bound to revision
`334dfeea4e0241033b4e5de97c01bc7c9c080530` under its custom terms;
SAGE-3D Collision Mesh is bound to revision
`3ba75cc7887b62bf84211d5db08adfa64d691597` under CC-BY-NC-4.0. This
rehearsal is private, noncommercial research. Raw InteriorGS dataset bytes are
not redistributable and were not authorized for provider upload. Explicit v2
authority permits only minimum scene-derived Aura inputs and derived SAGE
articulation inputs, at most two concurrent paid resources, and USD 12 combined
spend. It does not change publisher rights or authorize public disclosure.

| Source | Bytes | SHA-256 |
|---|---:|---|
| `0498_840796/3dgs_compressed.ply` | 36,382,397 | `sha256:1b8d0ff463a9a68d83ad10b5a49c3c5be9819b692005162382eaf30c733a8fdf` |
| `0498_840796/labels.json` | 132,539 | `sha256:57aa8675dbf7938d0b7e0194b5c581b37d049594ba6037d0a77552e6693c1484` |
| `0498_840796/structure.json` | 70,572 | `sha256:6563f49fe1266d477a90ac0007c18feaceb3397531968a1a28a8864477d74e8d` |
| `840796_collision.usd` | 23,988,263 | `sha256:7bfc6e4b6909f7057fa1983ac228e5d7511d7053745d8b6f467aecfea0c7194e` |
| Aura input receipt | — | `sha256:14835d010e3f15b610ff33ee8ae56f0daf8fa01b96aefe17db5ec36852a46f29` |
| Joint Agent source USDA | — | `sha256:4e0986e79b9d358de7e3289c0179eeaa4ec94b035e2133d2ea309298b525a4c3` |

Registration claims only publisher-frame synthetic consistency: right-handed,
Z-up, 1 meter/unit, identity InteriorGS-to-SAGE transform, zero round-trip error,
and target AABB IoU 0.992325475. It is not physical-site metrology.

The complete known topology survey retained all 593,665 splats, 24 nonblank
views across six publisher rooms, then nine nonblank target close-ups. Explicit
gaps are the refrigerator interior, surfaces behind the refrigerator, closed
cabinet interiors, surfaces outside publisher room profiles, and occluded backs
or undersides absent from source observations. These are reconnaissance renders,
not evaluation-authorized frames.

## Executions, cost, and teardown

| Run | Paid mutation | Result | Exact cost | Teardown |
|---|---:|---|---:|---|
| Local scene survey, selection, masks, registration, source extraction, packet/build tests | No | Passed or retained rejection as described above | USD 0 | Local only |
| Aura InteriorGS `second_scene_840796_v4_execute`, Vast instance 47226054 | Yes | Live: 30k fit and render complete; removal/export in progress | Pending terminal receipt; USD 6 hard per-run cap | Watchdog armed; teardown pending terminal result |
| Joint Agent `840796_v7_execute`, Vast instance 47232529 | Yes | Typed null before inference: `joint_agent_local_ovrtx_renderer_not_ready` | USD 0.133936 | Instance destroyed; continuing spend false; staged objects absent |

The Joint Agent null performed no model inference and published no owned-core
asset. Its OVRTX daemon became responsive but the forced PT/128 64x64 warmup did
not complete within the bounded runtime. No retry was launched. Generic fixes
landed for a receipt-bound RT2/32 construction-render profile, required returned
construction artifacts, and sibling-aware watchdog closure.

## Landed changes

| Commit | Change |
|---|---|
| `4d968c996` | Merge the latest published ADP-009D control harness. |
| `f3fa149b9` | Prevent absolute-position control starvation with bounded command slew and setpoint lead; covers 840313 and 840796 fixtures. |
| `265373d5d` | Retain and digest-bind Joint Agent construction outputs outside ephemeral working storage. |
| `51e5506d3` | Close a run watchdog when only explicitly authorized sibling GPU IDs remain. |
| `e1a18ba00` | Bind the fast, construction-only OVRTX RT2/32 profile and separate it from evaluation media. |

Before every commit, the required focused repository gate and Ruff gate passed;
the latest result is 985 passed, 1 skipped, 9,136 deselected, followed by a
clean Ruff check over `src/` and `tests/`.

## Controls, candidates, media, and seal

No second-scene control or learned-policy episode has run. This is correct:
without a retained Joint Agent topology and independently physics-qualified
articulated replacement, exact Franka placement and the scripted-positive
control cannot qualify. Candidate nulls are therefore not inferred. External,
wrist, and overview episode videos, score manifests, the portable evidence
index, replay seal, and Task Evaluation Run seal do not yet exist.

## Claim ledger

| Classification | Claims |
|---|---|
| Implemented | Outcome-blind multi-scene selection; full topology-before-closeup survey; exact SAGE sweep gate; bounded 1–4-joint task scoring with one commanded joint; private paid-resource authority; zero-retry Joint Agent/Aura bundles; artifact retention; sibling-aware teardown; scene-neutral control fix. |
| Simulator-qualified | None for scene 840796 yet. |
| Blocked or abstained | Joint Agent v7 abstained at released-code local OVRTX warmup; Aura remains live; SimReady articulation, exact Franka/IK/contact/camera gates, controls, matrix, policies, media, and seal wait on construction. |
| Physically unresolved | Appearance/collision registration as physical truth; replacement physical equivalence; real Fridge/Franka performance; partner fidelity; deployment readiness; physical candidate ranking. |

Exact claim ceiling: this public-dataset rehearsal cannot qualify a partner
capture, real-site fidelity, deployment readiness, or physical performance.

Single next action: obtain and validate the terminal Aura receipt, output
digests, before/after renders, cost, teardown, and provider-zero evidence; then
join that result to the smallest remaining articulated-construction blocker.
