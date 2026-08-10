# Second-scene Arm Decision Proof v1 rehearsal results

Status: **terminal completion path B — typed, evidence-backed abstention at the
first native articulated-simulator gate**.

Scene selection, rights, full-topology survey, task freeze, coverage-conditioned
Gaussian cutout, exact source-collider removal, deterministic articulated USD,
and the construction join are complete. One zero-retry native diagnostic was
launched from immutable commit `a06116755`. The Vast container never produced
its heartbeat, so Isaac never opened the USD. Consequently native asset
qualification, Franka IK/contact/camera gates, controls, scenario cells, and both
learned candidates were not executed. This is an infrastructure null, not an
asset failure, control result, or learned-policy null.

The terminal receipt is
[`manifests/second_scene_840796_task_evaluation_abstention.v4.json`](manifests/second_scene_840796_task_evaluation_abstention.v4.json),
digest
`sha256:ab8c5256ccf4de64d75f7f7356064088df3bd1b7988f983bb0e7cd88e1229e9c`.
It supersedes v3's earlier Gaussian-ownership blocker while retaining the older
receipt as historical evidence.

## Selected scene and frozen task

The outcome-blind first-passing rule selected InteriorGS/SAGE scene `840796`
(`0498_840796`), refrigerator instance `123`. The exact task is: “Open the
upper refrigerator door to at least 45 degrees, release it, and retreat.” Only
`refrigerator_upper_door_hinge` is commanded and scored; the lower hinge is
locked at zero with a `0.001 rad` motion tolerance. Success requires 45–55
degrees for 40 samples at 15 Hz, target speed no greater than `0.05 rad/s`, task
contact released, no collision/limit/containment failure, and retreat complete.
The candidates remain exactly `pi05_droid` and `groot_n17_droid`. Freeze digest:
`sha256:0774b7072bb1a0fd0929fde81cd48aaff384af4608772f31d603c2324f97d03f`.

| Scene(s) | Decision | Evidence-backed reason |
|---|---|---|
| `840076` | Rejected | Best SAGE door collider AABB IoU `0.200712895`, below the frozen `0.85` whole-object threshold. |
| `840411` | Rejected | Exact SAGE triangle sweep contacts chair 227 at `26.75°`, before the required `45°`; moving the chair would change the public scene. |
| `840796` | Selected | Six-room survey, observed two-door refrigerator and handle, unique SAGE collider IoU `0.992325475`, exact upper-door sweep clear. |
| `840873`, `840874`, `840920`, `841151`, `841193`, `841757` | Not inspected | Frozen first-passing rule stopped at `840796`; no appearance or policy outcome was used. |

## Rights, provenance, registration, and visibility

InteriorGS is bound to revision
`334dfeea4e0241033b4e5de97c01bc7c9c080530` under its custom
nonredistribution terms. SAGE-3D Collision Mesh is bound to revision
`3ba75cc7887b62bf84211d5db08adfa64d691597` under CC-BY-NC-4.0. The
declared scope is private, noncommercial research. User authority allowed the
minimum private scene-derived processing inputs and at most two GPUs under a
combined USD 12 cap; it did not waive publisher rights. Raw InteriorGS PLY,
labels, and structure bytes were not uploaded or redistributed.

| Source | Bytes | SHA-256 |
|---|---:|---|
| `0498_840796/3dgs_compressed.ply` | 36,382,397 | `sha256:1b8d0ff463a9a68d83ad10b5a49c3c5be9819b692005162382eaf30c733a8fdf` |
| `0498_840796/labels.json` | 132,539 | `sha256:57aa8675dbf7938d0b7e0194b5c581b37d049594ba6037d0a77552e6693c1484` |
| `0498_840796/structure.json` | 70,572 | `sha256:6563f49fe1266d477a90ac0007c18feaceb3397531968a1a28a8864477d74e8d` |
| `840796_collision.usd` | 23,988,263 | `sha256:7bfc6e4b6909f7057fa1983ac228e5d7511d7053745d8b6f467aecfea0c7194e` |

Registration proves publisher-frame consistency only: right-handed, Z-up,
1 meter/unit, identity InteriorGS-to-SAGE transform, zero numerical round-trip
error, and target AABB IoU `0.992325475`. It is not physical-site metrology.

The survey retained all `593,665` source splats, 24 nonblank in-scene views over
all six publisher rooms, then nine target close-ups. Explicit source gaps are
the closed refrigerator interior, wall/floor behind it, closed cabinet
interiors, areas outside the publisher room profiles, and occluded backs and
undersides. Reconnaissance renders are separate from evaluation-authorized
policy frames.

## Removal, replacement, and construction join

The frozen FlashSplat ownership test is now terminally sealed by
[`second_scene_840796_gaussian_excision_heldout_abstention.v2.json`](manifests/second_scene_840796_gaussian_excision_heldout_abstention.v2.json),
digest
`sha256:1a13f112007f4a54bef96b12312b93a43df8ba0cd16f72143ca58b90d6a87de8`.
Six calibration cameras produced per-Gaussian front-to-back `alpha *
transmittance` totals for target-core, uncertain, and protected pixels. Geometry
and neighborhood consistency then produced exactly `2,436` owned, `587,007`
retained, and `4,222` ambiguous indices; the sets are exhaustive and disjoint.
Two independent ownership materializations produced byte-identical canonical
receipts, index sets, and artifact digests. All retained source records remained
byte-identical. The raw GPU accumulation arrays were not bit-identical, and that
fact remains reported; their classification label disagreement count was zero.

The scientific test nevertheless **failed on both preregistered held-out
cameras**. `far_left` owned-only recall was `0.8808863`, with `140,057` missing
mask pixels, `4,256` protected significant pixels, and protected alpha sum
`222.9882`; its 3,791-Gaussian OBB baseline missed `99,405` pixels with protected
alpha sum `59.2196`. `far_right` owned-only recall was `0.8545827`, with `183,463`
missing pixels, `4,618` protected significant pixels, and protected alpha sum
`150.9843`; its OBB baseline missed `113,928` pixels with protected alpha sum
`25.6941`. The conservative ambiguous-layer residue bounds contained connected
components of `1,089,831` and `1,145,765` pixels, versus the frozen maximum of
four. Thus silhouette, protected-spill, residual-alpha, residual-component, and
both OBB-comparison gates all failed on both cameras. The ownership test does
not authorize a replacement sweep.

The Finder-friendly eight-camera review index is
`/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/gaussian_excision/840796_v2/heldout_audit_v4/evaluation/OPEN_ME_gaussian_excision_contact_sheets.html`,
digest
`sha256:5091afa0dd4b7c2557b55715c3a7249ea2aa98fe5f3e192fde5d9111dd72faf3`.
Every lossless sheet shows `original | exact mask | OBB removed-only |
contribution removed-only | retained scene | ambiguity heatmap`.

The later replacement-first method is a separate, explicitly
coverage-conditioned construction claim. It tested a frozen target-only
expansion ladder against actual USD rendered depth; it does not convert the
ownership failure into a pass or claim factual ownership of every deleted
Gaussian.

Rung `rung_01_core95` deletes `4,422` indices and retains `589,243`; every
retained source PLY record is byte-identical and remains in original order. The
retained scene digest is
`sha256:2c26029f55b47efd9fc7c52286544bda81c8fbad49cb81eec5e7080dbcac9313`.
Across eight calibrated cameras and twelve upper-door states, actual USD depth
left a worst target-core uncovered fraction of `0.028953981964478046`, below
the frozen `0.05` limit. Worst uncovered target-core count was `2,041`; the
largest connected component was `989`, below the derived `1,609` bound; no
residual pixel escaped the target mask. Coverage receipt digest:
`sha256:3322d2b48905212e4852a6548e059c6a8b417019fbe3ed63538f1d04da70ab62`.
This admits narrow mask-contained seam repair only. It does not claim that all
deleted Gaussians were refrigerator-owned, that hidden background was
reconstructed, or that Aura qualified.

The exact SAGE subtree `/Root/ZC2DFJJVAIJFUPTUJQ888888` was removed. The output
collision layer digest is
`sha256:69fa0ebadf487a3e7f42e681ec5fbff69c9267502fcb4ede0cccadf5ee4a896d`;
all 72 unrelated composed prim inventories were unchanged. Receipt digest:
`sha256:085c04def2052b3c49f4bb047394c9a11ddecbdbd63d2bdbc0f08a471e69ba82`.

The replacement is a deterministic parametric articulated USDA derived from
the frozen source components, seam, hinge, limits, and handle observation:

- USD digest:
  `sha256:f626998bbb5bd48d57950c4b96d9a7452705c1b2a906df950e20ac9767e649c1`;
- fixed cabinet plus upper and lower door links; two 0–90° revolute joints;
- task joint `/Asset/joints/upper_door_hinge`; lower joint locked by task contract;
- cabinet mass `62 kg`; each door `11 kg`; static/dynamic friction `0.6/0.5`;
  restitution `0.05`;
- observed-source-derived collision components, explicit handle colliders, and
  a generated inset interior at `0.04 m`;
- approximate local envelope `0.714 × 0.699 × 1.631 m` from retained link bounds;
- authored fixed base and world translation `(1.9742142, 1.4792181, 0.0) m`.

Topology and physics receipt digests are respectively
`sha256:b4d0f35306e9cf337697cb7bf884b2cb684b9833e02dd52c329e8eb91c10c609`
and
`sha256:4a2303ada5b844d9fa13af91be35b0585a6ba3b6aeb19e3d6ee69a727bdfcde0`.
The asset is a **SimReady candidate, statically admitted**. It is not a Joint
Agent output, native-simulator-qualified asset, or physical equivalent.

The construction join is `join_admitted`, with receipt digest
`sha256:9e00db25948105db00ce2d0de63acd2fc0225c777d718330ea9a77dc7c18f8dc`.
Its inpainting policy is `narrow_mask_contained_seam_repair_only`.

## Franka, cameras, controls, and policies

Static geometry search found 43 collision-clear base candidates. The selected
candidate is `(1.75, 1.99) m`, with `0.4553 m` minimum obstacle clearance and
`0.013102 m` worst analytic reach margin over the 0–55° door sweep. The
12-state clearance matrix binds SAGE scene geometry, replacement cabinet,
locked lower door, and Franka base and reports clear. External, wrist, and
review-only overview poses were resolved statically.

These are construction candidates, not native readback. The required native
precontact/contact/travel/release/retreat/recovery IK, contact stability,
initial penetration, camera framing, synchronization, and applied-parameter
receipts were not observed because the runtime did not reach the USD. The
scenario matrix was therefore not materialized. Zero-action and scripted-
positive controls did not run. Neither `pi05_droid` nor `groot_n17_droid` ran.
There is no score, tie, learned null, candidate comparison, episode frame
manifest, or episode video to interpret.

The reusable harness now supports articulated joint-state scoring, task-neutral
controls and learned episodes, task-neutral canonical asset binding, and
joint-angle instance constraints while preserving the original 840313 rigid
fixture. A failed scripted positive would still block a cell; it could never be
reported as a policy failure.

## Executions, spend, and teardown

| Run | Result | Retained cost | Teardown |
|---|---|---:|---|
| Local selection, registration, topology survey, SAGE sweeps, masks, cutout ladder, exact renders, static asset, collider removal, compilers/tests | Completed locally | USD 0 | No provider resource |
| Aura `second_scene_840796_v4_execute`, Vast 47226054 | Released workflow completed; rejected reference-camera binding and visual result | USD 2.372684 estimate | Destroyed; objects absent; inventory empty |
| Corrected Aura `second_scene_840796_v5_retry_execute`, Vast 47244918 | Correct camera binding; rejected Gaussian explosions, hallucination, multiview inconsistency, outside-mask changes | USD 1.774042 estimate | Destroyed; objects absent; inventory empty |
| Joint Agent `840796_v7_execute`, Vast 47232529 | No inference; OVRTX warmup not ready | USD 0.133936 exact | Destroyed; objects absent; no continuing spend |
| FlashSplat contribution attempt 1, Vast 47281712 | Import-closure infrastructure null; generic fix landed | USD 0.068951 exact | Destroyed; provider zero |
| FlashSplat contribution attempt 2, Vast 47283773 | Known-bad host vanished before scientific code; host avoidance fixed | USD 0.005121 exact | Destroyed; provider zero |
| FlashSplat contribution attempt 3, Vast 47284587 | Released contribution evidence completed | USD 0.084794 exact | Destroyed; objects absent; provider zero |
| Content Agents v1/v2 | Local/admission preflight nulls; no provider mutation | USD 0 | Teardown not required |
| Content Agents v3, Vast 47292105 | Instance exited before returned asset | USD 0.006258 estimate | Destroyed; no continuing spend |
| Content Agents v4, Vast 47292293 | Runner failed without runtime result; no output asset | USD 0.485139 estimate | Destroyed; objects absent; no continuing spend |
| Native articulated v1/v2 | Local preflight/launch-lock nulls; no provider mutation | USD 0 | Teardown not required |
| Native articulated v3, Vast 47294329 | `vast_heartbeat_container_missing`; Isaac never opened asset | USD 0.007044 estimate, USD 0.80 cap | Destroy 200; objects absent; no continuing spend |
| Joint Agent v19, Vast 47318232 | OpenAI-bound bundle passed dependency/capacity/OVRTX gates; runner rejected the non-NVIDIA backend before inference due to a hard-coded credential check; generic fix landed | USD 0.081835 exact | Destroyed; staged objects absent; provider zero |
| Joint Agent v21, Vast 47319708 | Reached released CLI inference validation; one `identify_asset.vlm` node remained NIM because backend rewriting named only three steps; recursive model-node and local evidence-closeout fixes landed | USD 0.052768 exact | Destroyed; staged objects absent; provider zero; local closeout recovered from bound receipts |
| Joint Agent v22 local admission | Execute-mode admission correctly rejected a 60-minute funded window before staging or provider mutation; dry-run parity fix landed | USD 0.000000 exact | No provider mutation; provider remained zero |
| Joint Agent v23, Vast 47321782 | Dependencies, capacity, optimizer, OvRTX construction/warm-up, and released CLI dry run passed. Execution exposed NVIDIA's implicit `identify_asset.vlm` NIM default because the BYOA YAML omits the node; pinned-schema default materialization fix landed | USD 0.077329 exact | Destroyed; staged objects absent; provider zero |

Combined retained spend through Joint Agent v23 is **USD 5.973452**, below the USD 12 authority. Aura,
Content Agents, and the native diagnostic expose estimates where the provider
did not return a final billed line; those amounts are not relabeled as exact.
No automatic paid retry was performed. The post-native read-only API call
`vastai show instances --raw` returned `[]` at
`2026-08-09T18:49:22.474861+00:00`; provider-zero receipt digest:
`sha256:039c0a509390e1599012532aa003647545e077e8e9c6dbe7640b2b36790ecd3b`.

## Evidence and human-review media

- Full survey and refrigerator close-ups:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/selection_inspection_second_scene/840796`
- Exact masks and Aura before/after evidence:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/inpainting_inputs/840796_refrigerator_v2`
  and
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/aura_interiorgs/second_scene_840796_v5_retry_execute`
- Gaussian evidence and rung-1 retained scene:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/gaussian_excision/840796_v2/target_only_expansion_ladder_v1/rung_01_core95`
- Eight-camera × twelve-door-state hybrid frames and contact sheets:
  `.../rung_01_core95/reference_hybrid_review_v1`
- Static SimReady candidate and receipts:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/simready_candidate/840796_deterministic_v1`
- Construction join, source-collider removal, and native-attempt receipts:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/second_scene_840796_e2e`
- Finder-friendly index:
  `/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804/second_scene_840796_e2e/portable_evidence_index_v1/OPEN_ME_episode_evidence_index.html`
- Authoritative zero-episode index:
  `.../portable_evidence_index_v1/episode_evidence_index.v1.json`, index digest
  `sha256:8b7b4d5b94e835e8fc09ac9928f62dc8ed9899102ecc4826fc4cac55be33554d`.

The portable index intentionally contains zero episode links and names the
native-gate abstention. External, wrist, and overview episode videos do not
exist because no episode was admitted.

## Landed changes

| Commit(s) | Landed capability |
|---|---|
| `f3fa149b9`, `ca8542786`, `46ba28d69`, `57876257d` | Task-neutral action delivery, articulated scoring, learned episode, and controls with rigid/articulated fixtures. |
| `11a51f841`, `e41483749`, `a3eebc13e`, `117bcb06b`, `141d54ca0` | Aura input/order, retention, execution, and visual-rejection contracts. |
| `473235ba9`, `06d098812`, `bd51b42c4`, `3a17d0c37`, `9f0cce2ec`, `24cd42cbc`, `adf24fb04` | FlashSplat execution, deterministic aggregation, exact-background rendering, held-out audit, and contact sheets. |
| `e49959caf`, `7ef6217c9`, `2ba3cd670` | Direct-evidence expansion ladder, seam masks, and byte-exact cutout unions. |
| `51216f167` | Actual articulated USD depth sweep and 96-cell hybrid review. |
| `670411eee`, `bd3e345c6` | Native articulated diagnostic bundle and generic paid-provider routing/fallback. |
| `a06116755` | Coverage-conditioned cutout admission, exact source-collider subtree removal, and construction join. |
| `ee75b523a` | Task-neutral canonical asset and articulated joint-state scenario constraints. |
| `79780b3de` | API-derived provider-zero receipt and native-gate terminal abstention seal. |
| `5b878b79d` | Final protocol/results, checked-in construction manifest, and portable evidence bindings. |
| `fa1e4b025` | Fix the generic Vast startup state machine so `No such container` cannot terminate an instance still reported as `loading`; slow-lane regression covers loading-to-running recovery. |
| `d25cbf13b`, `d83221355`, `8faf757c3` | Verify two ownership materializations at the actual manifest/index seam, expose every held-out gate, require strict protected-alpha improvement over OBB, and generate the eight-camera visual index. |

Every reusable fix has a focused hermetic regression. Before each commit, the
required selector and Ruff gate passed. Latest pre-document code gate: `1,067`
passed, `1` skipped, `9,215` deselected; Ruff passed over `src/` and `tests/`.

## Final claim table

| Classification | Exact claims |
|---|---|
| Implemented | Outcome-blind selection; rights/provenance binding; topology-first survey; articulated task freeze; exact SAGE sweep; released-code diagnostic ownership audit; target-only byte-exact cutout ladder; actual-USD depth coverage audit; exact collider-subtree removal; deterministic articulated USD authoring/static validation; construction join; task-neutral scoring, episodes, controls, canonical assets, and scenario constraints; paid gates, teardown, provider zero; portable zero-episode index. |
| Simulator-qualified | None for scene `840796`. The GPU runtime did not open the USD, so no native articulation, contact, IK, camera, control, or policy qualification exists. |
| Blocked or abstained | Aura broad inpainting rejected; released Joint/Content Agent attempts returned no qualifying asset; native articulated diagnostic abstained at `native_articulated_asset_diagnostic_unobserved:vast_heartbeat_container_missing`; controls, matrix, both policy episodes, comparison, replay, and episode media remain unexecuted. |
| Physically unresolved | Hidden refrigerator/background truth; generated interior truth; replacement physical equivalence; real refrigerator/Franka interaction; partner capture fidelity; deployment readiness; physical policy performance and ranking. |

Exact claim ceiling: a private, development-only public-dataset construction
rehearsal with a statically admitted SimReady candidate and an unobserved native
diagnostic. It does not qualify a partner capture, real-site fidelity,
deployment readiness, physical performance, or a learned-policy comparison.

Single next action: **grant fresh retry-specific authority for exactly one new
zero-retry native articulated diagnostic on the same immutable construction
bytes.** The generic Vast loading-state defect exposed by v3 is fixed and
hermetically tested at `fa1e4b025`; the original no-automatic-retry rule remains
in force.
