# Third-Scene Dual-Task Generalization Rehearsal Results

Status: **in progress; construction evidence is not yet sufficient for controls**.

The reusable harness work is locally qualified and both replacement candidates
are authored. Scene `840920` and both tasks are preregistered, both exact source
collider deletions are materialized, and the appearance source is converted
locally to standard 3DGS without changing its Gaussian count. The two Gaussian
removals and native replacement qualifications remain open, so no control or
learned-policy episode is admitted. This is not a policy result.

## Identity and isolation

- Backlog/gate: `ADP-009D`, public-scene day-28 rehearsal, Gate 7.
- Starting commit: `229b785f69617be543cfc76dcd6944ee3b9f2e49`.
- Latest pushed reusable implementation commit: `3475aaa25` on
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

Local standard-format conversion retained all `509,477` records. Output:
`120,238,049` bytes, SHA-256
`07a94f9f48e0936cdf888b58e98533eec854d16b9b625aa0d4f33af3b5149e98`;
conversion receipt digest
`sha256:dbf105306d75aa840c19b73d6162a666ad7bff13db4332a3ac9e3c855192d060`.
This is format evidence only, not Gaussian ownership or render qualification.

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

Task A Gaussian removal: **blocked/not executed**. No calibrated mask set,
contribution/ownership receipt, held-out deletion proof, or replacement-depth
sweep exists. Task B Gaussian removal: **blocked/not executed** for the same
independently identified evidence fields. No shared mask, replacement ID, or
task-specific assumption was reused.

Both source collider deletions are materialized independently from the same
immutable SAGE source, plus one shared stage with both exact subtrees absent.
The batch receipt digest is
`sha256:17e4d236ea2bfa847b5aadfb9b9dc347aeea673dcb8a7a5bf1a1dc3319116c74`;
the washer and notebook child receipt digests are respectively
`sha256:a52aebfef5e3eba57ce2c04046d20ba3f5335b7649311e415e489a2872439a64`
and `sha256:edd05b4c240cb83108a484b9005a16136eaf88c799cfe037dc65262b3534036b`.
The shared collider stage is `61,346,958` bytes with SHA-256
`36bc6e6800f59c92b4f655ca7900e1041a5714447b82d4b7b7102d45ed4fa883`.
Source bytes and all unrelated composed prims are unchanged.

Task A inpainting decision: **not yet decidable**. The authored washer candidate
has not passed native import or the complete articulated depth sweep, so
replacement occlusion cannot be measured. Task B inpainting decision: **not yet
decidable**. The authored notebook candidate has not passed the rigid pose range
or replacement-depth coverage. Neither task is labeled
`inpainting_not_required`, and no image edit was used as evidence.

Task A SimReady candidate: **authored, not simulator-qualified**. Its general
graph compiler emitted six links, five joints, ten primitive colliders, complete
mass/COM/inertia and material parameters, joint limits/drives/reset metadata,
and an explicit runtime dependency-controller requirement. USD SHA-256:
`9c52d9fa3e9e52b1ce2bf8c166f7f65aaf325113ee41fcadeb5e72280c249c3a`;
authoring receipt digest:
`sha256:82397cc60d1c1f9375ed0e133967477f10b14498425643a2caab9c321f79453a`.

Task B SimReady candidate: **authored, not simulator-qualified**. It contains a
dynamic two-link body with a locked-open hinge and explicit rest frames. USD
SHA-256:
`21bb119a4ef2ec63741f7deaedece9ccff3312c0fc6da4cf57de61f23337c12a`;
authoring receipt digest:
`sha256:ee9673d9482f1c5a1cd2c7d8142e20d07ccf88dbdd1fdd01baed6492473c7f99`.
All dimensions derive from observed bounds or exterior evidence; hidden
mechanisms and all physics values remain labeled generated/authored estimates.
Native import, joint/physics behavior, appearance, placement, reach/contact,
cameras, reset, zero-action, scripted-positive, and policy execution remain
blocked.

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
- `6441ffe30`: independent source-collider batch removal with immutable OpenUSD
  working stages.
- `7fc382cd3` and `d2ae6d9d1`: rights-bound local standard-splat conversion and
  the exact 840920 conversion receipt.
- `3475aaa25`: variable-camera registered excision evidence, authorized-render
  bindings, file-backed construction joins, general graph USD authoring, and
  fail-closed non-servo task-joint semantics.

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
| Implemented | General articulation graphs and graph-driven USD authoring; multi-replacement shared scenes; per-object resets; independent collider deletion and file-backed construction bindings; registered variable-camera excision evidence; authorized-render receipts; typed scenario application/readback; generic rigid scoring/state; task-neutral native policy lane; portable evidence index. |
| Simulator-qualified | No scene-840920 task, asset, control, or policy episode. Local contract behavior only. |
| Blocked/abstained | Two exact Gaussian removals, replacement depth/appearance/native import qualifications, occlusion/inpainting decisions, native placement/camera/control gates, and both candidate matrices. |
| Physically unresolved | Partner capture, real-site fidelity, deployment readiness, physical manipulation, customer value, sim-to-real, and unseen generated mechanism truth. |

## Remaining ambiguity and single next action

Independent metric/handedness qualification, evaluation-authorized rendering,
native reach/contact, asset dynamics/appearance, and all episode evidence remain
unknown.

**Single next action:** materialize the two evaluation-authorized calibrated mask
sets and contribution/ownership receipts, then use the authored USD candidates
for complete joint-state/rigid-pose depth coverage. Do not launch native controls
or learned policies before both per-task construction joins pass.
