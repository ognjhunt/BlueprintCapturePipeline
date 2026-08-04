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
| Sim-only protocol | `blocked_pending_v3_approval` | [`FOUNDER_SIM_ONLY_PROTOCOL.md`](FOUNDER_SIM_ONLY_PROTOCOL.md) freezes Isaac Lab-Arena's built-in Rubik's-cube-to-bowl task on Isaac Sim 6.0.1/PhysX, the DROID Franka-plus-Robotiq embodiment, explicit π0.5 baseline, Arena-supported GR00T N1.6-DROID alternative, 44 paired reset seeds per candidate, and an 88-episode schedule. A pre-candidate audit found one stale N1.7 descriptive label in v2's shared-interface map; v3 corrects only that label and amendment provenance. No candidate policy was queried. Exact v3 digest `sha256:c9aac12d5643a788ef3195e5f959cc73677bd0f51f3583dd36dd4861d4e12924` requires founder reapproval; v2 receipts remain superseded provenance only. [`adp_isaac_lab_arena_request.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_request.py) compiles the unchanged 88-job schedule, and [`adp_arena_candidate_execution_gate.py`](../../src/blueprint_pipeline/adp_arena_candidate_execution_gate.py) blocks it until materialization, three native controls, and both media-complete policy dry-runs pass. |
| GR00T adapter | `observed_complete` | [`groot_n16_arena_policy_runtime.py`](../../src/blueprint_pipeline/groot_n16_arena_policy_runtime.py) binds Arena's native NVIDIA ZMQ/DROID seam to the N1.6 source and checkpoint revisions with mandatory materialized-worker identity evidence. |
| Existing scene/assets | `partial` | Arena already registers the maple table, Rubik's cube, YCB bowl, home-office HDR, DROID embodiment, task, success metric, and OpenPI/GR00T remote-policy seams, so no generation or custom environment is required. [`adp_isaac_lab_arena_materialization.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_materialization.py) now fails closed unless a future native worker proves clean exact-revision source checkouts and byte-complete runtime, asset, embodiment, and checkpoint groups. The actual worker receipt remains intentionally absent because no native/paid worker was started. |
| Local control preflight | `observed_complete` | [`franka_droid_control_preflight.py`](../../src/blueprint_pipeline/franka_droid_control_preflight.py) ran the pinned MuJoCo/Menagerie proxy and produced receipt digest `sha256:30cc9636bc4a90b023f89e7aca65c6ebd66229462562902e89ad4b5ef48c1106`: the scripted control succeeded, the stationary control failed, and both retained complete visual evidence. This is local control evidence only, not candidate or native-Isaac evidence. |
| Scenario cousins | `missing` | Deliberately excluded from the first digest. The 44 seeded placements per candidate are paired repetitions, not cousins. Future object, lighting, background, camera, mass, or friction cousins require a new protocol digest and should use Isaac Lab-Arena's frozen variation system; agentic generation is proposal-only. |
| Production simulation | `missing` | Founder v2 approval is complete. [`adp_isaac_lab_arena_vast.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_vast.py) provides the canonical exact-image, zero-retry, capped Vast lane for the native zero-action control with output return, watchdog, teardown, and provider-zero enforcement. Checkpoint/asset materialization, native positive/parity controls, and media-complete model-adapter dry runs remain prerequisites to the 88 candidate episodes. |
