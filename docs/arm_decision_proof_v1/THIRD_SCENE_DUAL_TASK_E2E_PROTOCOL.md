# Third-Scene Dual-Task Generalization Rehearsal Protocol

Status: frozen development-only protocol. No learned-policy outcome was examined
when the scene, tasks, candidates, controls, or scenario suites were frozen.

## Program boundary and gate

This protocol exercises `ADP-009D`, the day-28 public-scene rehearsal and Gate 7
of the public evidence ladder. It is one bounded third-scene rehearsal with two
independent tasks. It does not change the Task Evaluation Run product, create a
dataset campaign, or promote simulator output to physical evidence.

Starting repository commit:
`229b785f69617be543cfc76dcd6944ee3b9f2e49`. The starting checkout was clean.
Work was isolated in the dedicated
`codex/adp-third-scene-dual-task-20260810` worktree. The 840796 worktrees,
evidence roots, launch locks, object-store prefixes, and provider resources are
external to this lane and must not be modified.

Before any paid construction run, the smallest local gate is a hermetic proof
that device ownership is coherent before the first Arena reset, followed by an
exact packet and runtime-source bundle dry-run. A Gaussian-contribution launch
additionally requires a digest-bound offline Python wheelhouse, an immutable
base image already containing the pinned Torch/CUDA runtime, exact-bundle
entrypoint rehearsal, and canonical paid-admission dry-run. These gates are now
implemented in the reusable runtime. They do not qualify scene 840920 without
the missing GPU contribution evidence and native replacement qualifications.

## Frozen scene selection

All admitted InteriorGS/SAGE candidates were surveyed under the criteria frozen
in
[`third_scene_dual_task_selection_preregistration.v1.json`](manifests/third_scene_dual_task_selection_preregistration.v1.json):
rights, full known-room topology, two distinct observed source objects, one
mechanically distinct task, one rigid task, support/destination geometry,
collision-free base candidates, policy and review camera coverage, appearance,
collision, and shared-frame evidence or abstention.

Scene `840920` is selected. It is a previously unused mixed-use room with a
front-loading washer region and a spatially separate desk/notebook region. The
complete admission/rejection ledger, exact source revisions, rights terms,
sizes, and digests are sealed in
[`third_scene_840920_dual_task_scene_freeze.v1.json`](manifests/third_scene_840920_dual_task_scene_freeze.v1.json).
This scene is genuinely unused because it does not occur in either the 840313
canned-beverage rehearsal or the ongoing 840796 refrigerator lane, and the
selection ledger records no prior method execution on it.

The InteriorGS source is a gated noncommercial research dataset. Raw
redistribution and raw private upload are forbidden. The SAGE collision source
is CC-BY-NC-4.0. This rehearsal may disclose only the minimum goal-authorized
derived bytes, with no training and bounded retention. Provider disclosure must
be admitted again for every launch.

## Registered appearance and collision

The full topology survey precedes task close-ups. InteriorGS is the appearance
authority; SAGE is collision/alignment evidence only. Exact renderer
qualification is required separately for every evaluation-authorized render.
Reconnaissance frames are never evaluation inputs.

The shared-frame receipt matches five object identities with maximum center
residual `0.002993334 m` and maximum extent residual `0.001011985 m`. This is
provider-declared alignment evidence, not an independent metric-scale or
handedness qualification. Those fields remain an explicit construction gate.
Unseen, uncaptured, occluded, and unrenderable regions must remain disclosed;
virtual camera motion cannot recover absent observations.

## Independent task freezes

Task A is `task_a_washer_door_open`: open and release the front-loading washer
door in the joint interval `[0.70, 0.95] rad`. It binds observed source instance
`165`, collider subtree `/Root/ZFAVSKZVAJTGUPTUKM888888`, a separate removal,
mask, collider-deletion, replacement, and qualification identity, and a general
five-joint articulation graph with target, dependent, passive, and locked roles.
Hidden hinge, latch, drum, and interior geometry are generated candidate content,
not observed truth.

Task B is `task_b_notebook_relocation`: relocate observed open notebook instance
`385` by `0.15 m` along its observed support, preserve its bounded orientation,
release, settle, and retreat. It binds collider subtree
`/Root/ZFUHSLMTQ3RUOPTUKM888888` and a second, disjoint set of removal, mask,
collider-deletion, replacement, and qualification identities. Its display hinge
is locked and generated-candidate mechanism content; the manipulation is scored
as rigid-body motion.

The two task definitions are sealed separately in
[`third_scene_840920_task_a_freeze.v1.json`](manifests/third_scene_840920_task_a_freeze.v1.json)
and
[`third_scene_840920_task_b_freeze.v1.json`](manifests/third_scene_840920_task_b_freeze.v1.json),
then joined without outcome leakage by
[`third_scene_840920_dual_task_freeze_join.v1.json`](manifests/third_scene_840920_dual_task_freeze_join.v1.json).
Exactly `pi05_droid` and `groot_n17_droid` are frozen for both tasks.

## Construction gates

Each task must independently produce calibrated, digest-bound masks and
Gaussian contribution evidence; owned/retained/ambiguous classifications;
byte-identical retained records; replacement-depth coverage over every relevant
camera and full motion/pose range; source-collider subtree deletion; and held-out
views. Bounding-box crops and scene-ID thresholds are forbidden.

Inpainting is decided only after the complete replacement is tested for
occlusion. Emit `inpainting_not_required` when no material residue is measurable.
Otherwise use only the admitted released-code seam backend inside the exact
residual mask, with cross-view geometric propagation and untouched-pixel
invariance. If that backend cannot qualify, abstain at the missing capability.

Both replacement assets must bind rights, geometry/material/texture/USD digests,
metric dimensions, pivot, placement, physics, collisions, import identity, and
observed-versus-generated labels. The shared scene must keep both replacements
co-present. A task selects its subject by stable asset ID; the inactive object
must still reset and read back correctly.

For each task, native construction must qualify its own Franka base, approach,
contact, path, release, retreat, recovery, support stability, penetration,
joint-limit, containment, and reset gates without moving unrelated geometry.
External and wrist cameras are policy inputs. Overview is review-only and is
forbidden from policy input and deterministic scoring.

## Frozen matrices and execution order

The task-neutral suite contract requires exactly the canonical,
placement/approach, illumination, camera/sensor, bounded-physics, admitted-object-
cousin, and held-out-composed families. Both candidates receive identical cells,
seeds, resets, observations, actions, and scorers. Native applied-parameter
readback is mandatory.

Initial execution is limited to:

1. Task A canonical, then Task A external-camera diagnostic.
2. Task B canonical, then Task B start-placement diagnostic.

For every admitted task/cell, run the zero-action negative first and require
failure. Then run the deterministic scripted positive through the same native
action seam and require stable success after release/settle. A failed positive is
a construction blocker. Only then may `pi05_droid` and `groot_n17_droid` run.
Harness-fault and undetermined episodes remain unranked; `never_moved` is not a
policy failure without action-delivery and robot-motion proof.

## Paid-compute and sealing rules

Any paid run requires a clean immutable pushed commit, exact input and bundle
digests, local construction packet, dependency closure, watchdog, TTL, `$12`
aggregate hard cap, zero retry, provider inventory, and an explicit allowlist of
every external instance. This lane may own at most two concurrent GPU instances.
It must tear down only its own instances and prove lane-owned provider zero.

Every episode must retain actions, delivery, robot/task state, contacts or an
explicit sensor gap, deterministic scoring, reset replay, timings, lossless
policy frames, frame manifest, terminal observations, and external/wrist/overview
videos. The portable evidence index is authoritative only for artifacts whose
digests it verifies.

Completion is either a sealed dual-task rehearsal satisfying all construction,
control, candidate, media, replay, spend, and teardown gates, or a typed
evidence-backed abstention at the smallest missing capability after safe local
implementation is exhausted. Neither outcome establishes partner capture,
real-site fidelity, deployment readiness, physical performance, customer value,
sim-to-real transfer, or truth for unseen generated geometry.
