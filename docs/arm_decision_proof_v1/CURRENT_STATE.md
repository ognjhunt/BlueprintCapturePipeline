# Arm Decision Proof v1 Current State

This table is the checkout audit required by Phase 0. Status vocabulary is
limited to `observed_complete`, `partial`, and `missing`. A locally passing
fixture does not promote a public-reference gate.

| Item | Status | Observed evidence |
|---|---|---|
| ADP-001 | `observed_complete` | [`north_star_contract.json`](north_star_contract.json), [`test_arm_decision_proof_focus.py`](../../tests/test_arm_decision_proof_focus.py) |
| ADP-002 | `observed_complete` | [`simpler_google_robot_pick_coke_can.v1.json`](manifests/simpler_google_robot_pick_coke_can.v1.json), [`public_reference_admission_receipt.json`](../../output/arm_decision_proof_v1/evidence/public_reference_admission_receipt.json), [`paid_runtime_canary_validation.json`](../../output/arm_decision_proof_v1/evidence/paid_runtime_canary_validation.json) |
| ADP-003 | `observed_complete` | [`adp_simpler_closed_loop_execution.json`](immutable_execution/adp_simpler_closed_loop_execution.json), [`execution_validation.json`](../../output/arm_decision_proof_v1/evidence/execution_validation.json); exactly two distinct genuine RT-1 checkpoint identities and six completed cells |
| ADP-004 | `observed_complete` | [`receipt_replay.json`](../../output/arm_decision_proof_v1/evidence/receipt_replay.json), six [`episode_receipts`](../../output/arm_decision_proof_v1/evidence/episode_receipts), and digest-bound [`traces`](immutable_execution/traces). The historical v1 execution predates the visual-evidence requirement and is explicitly labeled `legacy_execution_missing_required_media`; every newly executed v2 episode must retain lossless policy-input images, a frame manifest, a terminal image, and a derived review video or fail closed. |
| ADP-005 | `observed_complete` | [`decision_seal.json`](../../output/arm_decision_proof_v1/evidence/decision_seal.json) precedes [`physical_outcome_release_receipt.json`](../../output/arm_decision_proof_v1/evidence/physical_outcome_release_receipt.json); published outcomes are explicitly a software firebreak, not a genuinely unseen holdout |
| ADP-006 | `observed_complete` | [`bounded_development_decision.json`](../../output/arm_decision_proof_v1/evidence/bounded_development_decision.json) freezes the rule and correctly abstains because three trials per candidate are below the 99-trial conservative requirement |
| ADP-007 | `observed_complete` | [`evidence_matrix.json`](../../output/arm_decision_proof_v1/evidence/evidence_matrix.json) renders all six candidate-condition cells with source, reset, execution, trace, metric, physical outcome, version, digest, and qualification links |
| ADP-008 | `observed_complete` | [`REPLAY.md`](REPLAY.md), [`physical_outcome_join.json`](../../output/arm_decision_proof_v1/evidence/physical_outcome_join.json), [`bounded_verdict.json`](../../output/arm_decision_proof_v1/evidence/bounded_verdict.json), and [`artifact_index.json`](../../output/arm_decision_proof_v1/evidence/artifact_index.json); identical post-upgrade replays produced index digest `sha256:e009662e90c3d9966d31ccf56e209097c0df223b2332542b86e7823f15db48f2` without rerunning the historical simulator episodes. |

All entries are `retrospective_external_reference` and `development_only`.
No capture or reconstruction feature was added.

## Public Scene Qualification Phase

| Item | Status | Observed evidence |
|---|---|---|
| ADP-009A | `partial` | [`PUBLIC_EVIDENCE_LADDER.md`](PUBLIC_EVIDENCE_LADDER.md), [`public_scene_suite_manifest.v1.schema.json`](../schemas/public_scene_suite_manifest.v1.schema.json), [`public_scene_suite_index.v1.schema.json`](../schemas/public_scene_suite_index.v1.schema.json), and the fail-closed component/index tests define rights/revision/digest/frame/role/code-smoke admission, method-input allowlists, oracle/truth isolation, exact project-role binding, and aggregate matrix rules. Component receipts cannot claim matrix completion; the index rejects ARKitScenes, WildRGB-D, authored dataset substitutes, and SimReadyGen in the NVIDIA Content Agents role. No actual InteriorGS/SAGE pair, ScanNet++ scene, Inpaint360GS author-data control, InFusion adapter, AuraFusion360 challenger, NVIDIA Content Agents runtime, controlled background case, SimReady object, or physics control is yet admitted. |
| ADP-009B | `missing` | No unchanged Inpaint360GS author-data smoke, InFusion format/license/frame adapter, or AuraFusion360 representation/checkpoint adapter exists. No exact rights-admitted InteriorGS/SAGE-3D source object has been removed from both appearance and collision, completed, replaced by an exact SimReady USD, and qualified in Isaac. An authored positive control cannot substitute. NVIDIA USD Content Agents is audited as a candidate backend but has not been executed or admitted. |
| ADP-009C | `missing` | No exact ScanNet++ real measured scene or controlled known-background completion has passed the new metric/editing tests. |
| ADP-009D | `missing` | No deterministic InteriorGS/SAGE-to-ScanNet++ variation matrix or one-command public-data replacement rehearsal exists. ARKitScenes and WildRGB-D were explicitly removed from the required stack. |

An earlier development run retained local InteriorGS `0787_841244` PLY and
semantic sidecars and exercised import/render paths. Those bytes are useful
candidate inputs, but they do not satisfy ADP-009A: no current rights-authority
receipt, exactly matched SAGE collision binding, released-method runtime/license
smoke, or independently sourced SimReady USD admission exists yet. The
InteriorGS/SAGE release provides a metric frame, OBBs, and static collision
pairing, not a measurement-authoritative local surface mesh. Rendered cameras
and RGB will therefore be synthetic method inputs/self-consistency probes;
external metric depth remains a validation oracle unless a released method
explicitly accepts it through a preregistered adapter. No completion run has yet
demonstrated the required seal-before-clean-background-release firebreak.

ADP-009 is now the active engineering item. Every public artifact remains
`development_only`; this phase cannot qualify a fresh site, partner physics,
sim-to-real decision fidelity, or customer value.

## Prospective Partner Phase

| Item | Status | Observed evidence |
|---|---|---|
| Decision-design seam | `observed_complete` | [`adp_prospective_design.py`](../../src/blueprint_pipeline/adp_prospective_design.py) compiles one explicit baseline/alternative independent two-proportion design into an exact condition/reset/seed/repetition schedule and rejects an underpowered or altered schedule before execution; [`test_adp_prospective_design.py`](../../tests/test_adp_prospective_design.py) covers select, eliminate, equivalent/inconclusive, abstain, frozen denominators, owner-preregistered secondary metrics, and future terminal-media/grader provenance. Historical ADP-008 artifacts and digests are unchanged. |
| ADP-010 | `missing` | No named real partner, scored evidence packet, task owner, two partner candidate receipts, holdout custodian, or durable rights/authority receipt exists in the audited checkout. See [`ADP_010_BLOCKER_ACTION_PACKET.md`](ADP_010_BLOCKER_ACTION_PACKET.md). |
| ADP-011 | `missing` | Dependency-blocked by ADP-010; no partner stack exists to freeze without inventing one. |
| ADP-020 | `missing` | Dependency-blocked by ADP-010/011; no partner protocol exists and no task-owner, holdout-custodian, or Blueprint same-digest approvals exist. |

No prospective capture, reconstruction, simulator-scene construction,
production evaluation, physical holdout access, or physical trial was started.

## Founder Sim-Only Precursor

The founder explicitly narrowed the immediate stage to simulation-only harness
testing before partner or IRL work. This does not retroactively admit a partner
or complete ADP-010. The partner blocker packet remains the correct boundary for
a later physical phase.

| Item | Status | Observed evidence |
|---|---|---|
| Sim-only protocol | `observed_complete` | [`FOUNDER_SIM_ONLY_PROTOCOL.md`](FOUNDER_SIM_ONLY_PROTOCOL.md) freezes Isaac Lab-Arena's built-in Rubik's-cube-to-bowl task on Isaac Sim 6.0.1/PhysX, the DROID Franka-plus-Robotiq embodiment, explicit π0.5 baseline, Arena-supported GR00T N1.6-DROID alternative, 44 paired reset seeds per candidate, and an 88-episode schedule. A pre-spend audit superseded v1 before any candidate outcome or paid compute. Blueprint's founder approved exact v2 digest `sha256:05eb6f5c187fd69da6f40e7428634181b39fe7f02501bd7ccb9a4331801c01fc`; the immutable [v2 approval receipt](manifests/founder_sim_approval_receipt.v2.json) and [v2 execution admission](manifests/founder_sim_execution_admission.v2.json) remain simulation-only. [`adp_isaac_lab_arena_request.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_request.py) compiles its 88 frozen logical worker jobs without authorizing them before native controls. |
| GR00T adapter | `observed_complete` | [`groot_n16_arena_policy_runtime.py`](../../src/blueprint_pipeline/groot_n16_arena_policy_runtime.py) binds Arena's native NVIDIA ZMQ/DROID seam to the N1.6 source and checkpoint revisions with mandatory materialized-worker identity evidence. |
| Existing scene/assets | `partial` | Arena already registers the maple table, Rubik's cube, YCB bowl, home-office HDR, DROID embodiment, task, success metric, and OpenPI/GR00T remote-policy seams, so no generation or custom environment is required. [`adp_isaac_lab_arena_materialization.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_materialization.py) fails closed unless a native worker proves clean exact-revision source checkouts and byte-complete runtime, asset, embodiment, and checkpoint groups. Attempt 003 started only the paid native-control precursor and blocked before the Arena entrypoint, so no materialized worker or candidate evidence exists. |
| Local control preflight | `observed_complete` | [`franka_droid_control_preflight.py`](../../src/blueprint_pipeline/franka_droid_control_preflight.py) ran the pinned MuJoCo/Menagerie proxy and produced receipt digest `sha256:30cc9636bc4a90b023f89e7aca65c6ebd66229462562902e89ad4b5ef48c1106`: the scripted control succeeded, the stationary control failed, and both retained complete visual evidence. This is local control evidence only, not candidate or native-Isaac evidence. |
| Scenario cousins | `missing` | Deliberately excluded from the first digest. The 44 seeded placements per candidate are paired repetitions, not cousins. Future object, lighting, background, camera, mass, or friction cousins require a new protocol digest and should use Isaac Lab-Arena's frozen variation system; agentic generation is proposal-only. |
| Production simulation | `missing` | Founder v2 approval is complete. Paid attempt 003 reached Isaac Sim 6.0.1/Warp startup on one RTX A6000 but the fail-closed CUDA sanity classifier returned `cuda_runtime_incompatible` before the Arena entrypoint. [`adp_arena_vast_result.json`](../../output/arm_decision_proof_v1/arena_native_control_v2/attempts/attempt_003/adp_arena_vast_result.json) is `blocked`: no provider output ZIP, MP4, controller result, candidate policy result, or ranking evidence exists. The exact attempt ran for 741.886 seconds, cost `$0.097086`, used no retry, destroyed its instance, removed staged objects, and independent inventory verification found zero active Vast instances and no continuing spend. The contradictory GPU-observability probe is a diagnostic blocker, not an experiment result. |
