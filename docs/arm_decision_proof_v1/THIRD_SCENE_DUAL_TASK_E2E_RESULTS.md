# Third-Scene Dual-Task Generalization Rehearsal Results

Status: **typed evidence-backed abstention (completion condition B)**.

The reusable harness work is implemented and locally qualified. Scene `840920`
and both tasks are preregistered, but neither source-object removal nor either
replacement asset exists as a qualified exact packet. Therefore no native
construction control, learned-policy episode, render-authorized evaluation
frame, or paid third-scene run was executed. This is not a policy result.

## Identity and isolation

- Backlog/gate: `ADP-009D`, public-scene day-28 rehearsal, Gate 7.
- Starting commit: `229b785f69617be543cfc76dcd6944ee3b9f2e49`.
- Landed reusable implementation commit: `bcabc56c0` on
  `codex/adp-third-scene-dual-task-20260810` (preceded by the pushed commits
  listed below).
- Starting worktree: clean; work took place in the dedicated third-scene
  worktree.
- External 840796 worktrees and evidence roots were not modified.
- The authority snapshot allowed external Vast instance `47373597`. A read-only
  refresh at `2026-08-10T16:19:08Z` found a different running external instance,
  `47385583`, owned by the ongoing 840796 native-microcheck lane. The third-scene
  lane owned zero instances and did not inspect its outputs, reuse it, terminate
  it, or charge it to this rehearsal.

## Selected scene and complete survey

Scene `840920` was selected because it is a previously unused mixed-use room
with two spatially and mechanically distinct observed sources: a front-loading
washer and a freestanding open notebook. It is materially different from both
the 840313 canned-beverage kitchen and the 840796 refrigerator scene.

Every admitted scene was considered before selection:

| Scene | Decision | Evidence-backed reason |
|---|---|---|
| 840076 | Rejected | No two independently removable feasible task sources observed. |
| 840313 | Rejected | Already used by the canned-beverage kitchen rehearsal. |
| 840411 | Rejected | Required articulated/rigid pair not observed. |
| 840796 | Rejected | Already owned by the refrigerator rehearsal lane. |
| 840873 | Rejected | Active deformable-task lane; not unused and out of scope. |
| 840874 | Rejected | Articulation observed, but no independent feasible rigid source/base pair. |
| 840920 | Selected | Unused mixed-use room with distinct washer and notebook task regions. |
| 841151 | Rejected | Only explicit articulation is a forbidden refrigerator. |
| 841193 | Rejected | No two independently removable feasible task sources observed. |
| 841757 | Rejected | No two independently removable feasible task sources observed. |

Within 840920, a stool was rejected as deformable; a chair was too large; a book
base candidate intersected `467` collision triangles; and a rice cooker was too
wide and lacked a qualified grasp. The notebook was selected instead. Full
survey receipts report `376` labels, `105` SAGE prims, `67` unambiguous matches,
`309` unmatched labels, and no ambiguous identity matches. Analytic base
candidates have zero overlap only; they are not native reach/contact proof.

## Rights, provenance, and registration

| Source | Exact revision/path | Bytes | SHA-256 | Rights ceiling |
|---|---|---:|---|---|
| InteriorGS | `334dfeea4e0241033b4e5de97c01bc7c9c080530` / `0492_840920/3dgs_compressed.ply` | 31,223,241 | `7de988a6cc1654dee5ed8c735bdc89ab0c5e67d10f6b76f8e65eec75522fc3b9` | Gated noncommercial research terms; no redistribution, raw private upload, or training. |
| InteriorGS labels | same revision | 247,565 | `24e8dc1abab348853da20fec39880999bdc84ed1eb479174bc949bed204fb444` | Same terms. |
| InteriorGS structure | same revision | 38,164 | `a85a5ecf3a828cd33e0656be297e5f25e1262e4534d3de91212c77f77bc316e4` | Same terms. |
| SAGE collision | `3ba75cc7887b62bf84211d5db08adfa64d691597` / `Collision_Mesh/840920/840920_collision.usd` | 17,929,371 | `9e51c7c4360b5071fdbbb9cebbf61a475cc88802d53b5b2b45df294ec2b7c8fd` | CC-BY-NC-4.0; raw upload not needed or authorized; no training. |

InteriorGS terms digest:
`508b0ec62ff1b5bee3c47524dd36f2402f5f70a09afbe62a4d0041078856356b`.
The appearance contains `509,477` Gaussian records. No raw dataset byte was
uploaded. Any future upload is limited to minimal goal-authorized derived bytes,
bounded retention, provider zero, and a disclosure receipt.

Topology receipt:
`sha256:87ec314cf6947fa51ac7dc819f618cfc779e2cebaafd6c74cf357d93148728a9`.
Reconnaissance-only render manifest:
`sha256:d807864a2f057fe505e519ba2a5dda317cf8b867bef6d1022362a6ab9afc9347`
with 21 frames. Shared-frame receipt:
`sha256:f49905f265b85fb7ead2f6525abeb836e03e2d379bf6f2b5f7ac91843c0f07aa`.
The five provider-declared correspondences have maximum center residual
`0.002993334 m` and extent residual `0.001011985 m`; independent metric scale,
axes, handedness, and transform truth remain unqualified. Reconnaissance is not
an evaluation-authorized renderer qualification.

## Frozen tasks and scenarios

| Field | Task A | Task B |
|---|---|---|
| Task | Open washer door | Relocate open notebook `0.15 m` on observed support |
| Stress | Articulated contact with target/dependent/passive/locked joints | Rigid grasp/place with position, orientation, support, release, and settle |
| Source instance | `165` | `385` |
| Collider subtree | `/Root/ZFAVSKZVAJTGUPTUKM888888` | `/Root/ZFUHSLMTQ3RUOPTUKM888888` |
| Task-freeze digest | `sha256:8290bfb4e4cc6ade4a79937157efda88c4ef6abdfc7a10489e3f854766bbd152` | `sha256:74adcf6f90701c0cdb73adb67b7a47c52c5feef3e6258d4a6e5602a8c09806d6` |
| Scenario-suite digest | `sha256:3dc59d61a659131921e6e279315d36765348eceb2a70d87d4c0733572124035a` | `sha256:b9e79b2bc3ddb9a24b066567ae4404f7289a707f27032424971624adecba86e4` |
| Initial cells | canonical, external-camera dx | canonical, object-start y |

Shared scene-freeze digest:
`sha256:08c7fcfa112574d4d3ad692d673563ff2d006271a28ae5898bbd66084fe6a10c`.
Dual-task join digest:
`sha256:d9ab18e48c2962f0fc4ba963000cad8bb3deafbd1a250468f278c907ff6fe0bb`.
Both suites freeze exactly `pi05_droid` and `groot_n17_droid`, all seven required
families, identical per-candidate cells/seeds, controls-first admission, and
native applied-parameter readback. No learned outcome influenced either freeze.

## Construction and evaluation outcomes

Task A removal: **blocked/not executed**. No calibrated mask set, Gaussian
ownership receipt, held-out deletion proof, collider-deletion receipt, or
replacement-depth sweep exists. Task B removal: **blocked/not executed** for the
same independently identified evidence fields. No shared mask, collider receipt,
replacement ID, or task-specific assumption was reused.

Task A inpainting decision: **not yet decidable**. The complete washer
replacement and articulated sweep do not exist, so replacement occlusion cannot
be measured. Task B inpainting decision: **not yet decidable**. The notebook pose
range and replacement-depth coverage do not exist. Neither task is labeled
`inpainting_not_required`, and no image edit was used as evidence.

Task A SimReady asset: **not authored or simulator-qualified**. Task B SimReady
asset: **not authored or simulator-qualified**. Consequently native import,
joint/physics behavior, appearance, placement, reach/contact, cameras, reset,
zero-action, scripted-positive, and policy execution all remain blocked.

| Task/cell | Zero action | Scripted positive | pi05_droid | groot_n17_droid |
|---|---|---|---|---|
| Task A canonical | Not run | Not run | Not admitted | Not admitted |
| Task A camera diagnostic | Not run | Not run | Not admitted | Not admitted |
| Task B canonical | Not run | Not run | Not admitted | Not admitted |
| Task B placement diagnostic | Not run | Not run | Not admitted | Not admitted |

There is no ranking, tie, null policy comparison, or media episode to interpret.
This abstention is a construction-input gap, not candidate failure.

## Reusable harness changes landed

The rehearsal found and fixed general contracts rather than adding task runtime
forks:

- `bb940942`: general articulation graph and dual-task preregistration.
- `4d1c583bf`: pre-first-reset CUDA/PhysX device ownership gate.
- `13c33e13c`: general SAGE collision, placement, and shared-frame evidence.
- `ff51d3447`: scene-neutral dual-task freeze and rights manifests.
- `70c8d1bfb`: repeatable replacement assets, active subject IDs, co-presence,
  and per-object reset/readback.
- `baa20c010`: independent removal/collider/replacement construction bindings.
- `661595133`: typed scenario-parameter application and native readback.
- `a29a3e144`: task-neutral rigid state and deterministic scoring.
- `1984fc8b5`: native task-neutral policy bundle, worker, and Vast adapter.
- `bcabc56c0`: shared-scene-bound task scenario suites and legacy adapter
  scene-neutrality cleanup.

Hermetic regressions preserve the 840313 rigid fixture, 840796 articulated
fixture, and dual-task shared-asset fixture. Scene-specific values remain in
manifests; execution uses stable task, subject-asset, reset, observation, action,
scenario, scoring, and evidence contracts.

## Run, cost, teardown, and media ledger

- Local scientific/contract runs: focused tests plus the mandated ADP/DROID/
  episode/NUREC/Aura selection; no native simulator execution.
- Paid third-scene runs: `0`.
- Third-scene paid cost: exactly `$0.00`.
- Third-scene instances allocated: `0`.
- Third-scene retries: `0`.
- Teardown: no third-scene provider resource existed to destroy; read-only
  inventory proves lane-owned provider zero. External instance `47385583`
  remained untouched.
- Episode receipts, frame manifests, reset replays, terminal observations, and
  videos: none exist because controls were not admitted. The evidence index
  represents this explicitly and contains no synthetic episode links.

Portable evidence index:
[`third_scene_dual_task_evidence/index.html`](third_scene_dual_task_evidence/index.html).

## Verification

Before the final reusable-code commit:

- `PYTHONPATH="$PWD/src" .venv/bin/pytest tests/ -q -k "adp009d or droid or episode or nurec or aura"`
  — `1123 passed, 9781 deselected`.
- `.venv/bin/ruff check src/ tests/` — passed.
- Focused scenario and scene-neutrality regressions — `45 passed`.

## Claim table

| Claim class | Result |
|---|---|
| Implemented | General articulation graphs; multi-replacement shared scenes; per-object resets; independent construction bindings; typed scenario application/readback; generic rigid scoring/state; task-neutral native policy lane; portable typed-abstention evidence index. |
| Simulator-qualified | No scene-840920 task, asset, control, or policy episode. Local contract behavior only. |
| Blocked/abstained | Two exact Gaussian removals, source-collider deletions, replacement assets, occlusion/inpainting decisions, native placement/camera/control gates, and both candidate matrices. |
| Physically unresolved | Partner capture, real-site fidelity, deployment readiness, physical manipulation, customer value, sim-to-real, and unseen generated mechanism truth. |

## Remaining ambiguity and single next action

Independent metric/handedness qualification, evaluation-authorized rendering,
native reach/contact, asset dynamics/appearance, and all episode evidence remain
unknown.

**Single next action:** materialize both exact replacement USD candidates and the
two independent calibrated removal/collider receipt sets, bind them into one
shared construction digest through the generalized replacement-construction
contract, and pass the scene-specific local packet dry-run. Do not launch paid
compute before that gate passes.
