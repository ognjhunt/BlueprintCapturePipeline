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
| ADP-009A | `partial` | The deterministic [`public_scene_suite_index.v1.json`](manifests/adp009a_materialized_suite/public_scene_suite_index.v1.json) now admits four of ten roles from observed bytes and execution receipts: selected InteriorGS scene `840313`, its exact SAGE-3D USDZ/collision companion, NVIDIA USD Content Agents v0.5.2, and AuraFusion360's unchanged author-data smoke. Scene `840313` was selected under preregistered criteria with target `ins160` (`canned_beverage`) mapped to removable collider `/Root/ZHQYGJJVAJYEYPTUKY888888`; rejected candidate `841244` and shortlist failures remain retained. Aura `live_v14` executed the publisher workflow with unmodified source, produced an 820,832-vertex point cloud, retained eight `960x540` same-camera comparison pairs, used zero retries, cost an estimated `$0.788692`, and proved provider/object-store zero. That author smoke is not an InteriorGS result. The current suite remains blocked on the Inpaint360GS author-data license authority, InFusion checkpoint license, controlled-background truth, dynamic exact-SimReady qualification, physics positive control, and ScanNet++ transfer. |
| ADP-009B | `partial` | The frozen, non-orbiting InteriorGS input request materializes eight lossless `2048x1536` RGB frames and eight OBB-plus-contained-Gaussian masks for selected scene `840313`, target `ins160`; the [input receipt](manifests/adp009b_interiorgs_edit_input_receipt.v1.json) binds actual source/render bytes, camera transforms, Git source identity, and executed commands. A deterministic [Inpaint360GS adapter receipt](manifests/adp009b_inpaint360_adapter_receipt.v1.json) additionally stages those frames as exact COLMAP PINHOLE cameras, uint8 instance masks, and an iteration-30000 publisher-splat seed; it is an experimental packet, not the frozen InteriorGS method lane. Retained run `live_v22_execute_attempt2` failed at the former total request-log deadline before a final artifact was returned. Its corrected successor `live_v28_target_execute` then completed the full released-source-with-adapters workflow on RTX 4090 with zero retries: exact-OBB removal, 30 target-centered virtual views, finite Big-LaMa color/depth, PLY fusion, and all 5,000 final refinement steps. It returned an actual 774,355-vertex, 241,600,681-byte point cloud (`sha256:e7fcae266988ffee1f203e89c2e5d6d4cdc987b504ae4233ea03355a6d34690c`), cost an estimated `$0.246182`, and proved provider/object-store zero. This is an executed edit but a rejected quality result: all 30 virtual masks covered 24.1–30.6% of their full frames and the method inserted 234,461 vertices beyond the post-removal count; the independent eight-camera `2048x1536` render manifest is `sha256:1f63bedf695ce403eef0aa5b977c57d72e2510008a1d1d9271f8e5a1fd30a3c3`, and the digest-bound outside-mask locality measurement is `sha256:fb35df9527a43619c8a74df80fd528cb7e9d6e8d8715a1756ad54b5d26b921a9` (mean PSNR `19.456037`, windowed SSIM `0.9336765`, LPIPS `0.1075777`, and `13.994182%` of outside-mask pixels changing by more than `20/255`). The result contains obvious translucent sheets and radial streaks and is not admitted as successful inpainting. Future packets now fail closed above 10% full-frame mask area or four times the largest frozen source-mask area, widen the target-camera standoff by five times, cap qualifying views at eight evenly distributed poses, and reject an added-Gaussian budget above the greater of twenty times the removal count or five percent of the baseline. The North Star reserves Inpaint360GS for unchanged author data and requires InFusion as the primary InteriorGS adapter, with AuraFusion360 as challenger. The [InFusion adapter receipt](manifests/adp009b_infusion_adapter_receipt.v1.json) now derives an exact blocked execution packet from observed bytes: it removes the 39 OBB-contained target Gaussians from a degree-3 publisher-Ply copy, preserves all 45 higher-order SH fields across the remaining 540,673 Gaussians, freezes `low_approach` by maximum preregistered target-mask coverage (79,681 pixels), binds the already admitted Apache-2.0 Big-LaMa checkpoint for single-view RGB completion, and prepares the native InFusion depth/render commands plus a tested SH-preserving compositor. This mechanical OBB partition is not yet semantic-completeness proof, no color/depth method or compositor has executed, no collider has been removed, and no SimReady object has been inserted. Execution remains fail-closed on `infusion_checkpoint_license_missing`; the public 6.86 GB checkpoint repository is accessible but still declares no license. NVIDIA USD Content Agents executed on the parametric can control, but that remains an authoring candidate rather than inpainting, measured geometry, dynamic simulation, or physical evidence. |
| ADP-009C | `missing` | No exact ScanNet++ real measured scene or controlled known-background completion has passed the new metric/editing tests. |
| ADP-009D | `missing` | No deterministic InteriorGS/SAGE-to-ScanNet++ variation matrix or one-command public-data replacement rehearsal exists. ARKitScenes and WildRGB-D were explicitly removed from the required stack. |

Candidate `0787_841244` remains a retained, rejected warm start rather than the
selected scene. Direct inspection could not establish a suitable target whose
InteriorGS instance identity mapped to a separately removable SAGE collider
without unrelated collision overlap. The expanded preregistered survey selected
scene `840313` instead and retained its whole-splat room survey and target
closeups. The InteriorGS/SAGE release provides a metric frame, OBBs, and static
collision pairing, not a measurement-authoritative local surface mesh. Rendered
cameras and RGB are therefore synthetic method inputs/self-consistency probes;
external metric depth remains a validation oracle unless a released method
explicitly accepts it through a preregistered adapter. No completion run has yet
demonstrated the required seal-before-clean-background-release firebreak.

The retained survey PNGs are lossless `1024x768` selection evidence, not the
final edit-quality render contract. A later ADP-009B edit must render the native
publisher 3DGS from source-calibrated deterministic virtual cameras at the
highest practical resolution, retain lossless RGB/masks/depth, and render the
replacement USD through the qualified NVIDIA/Omniverse path. Higher sampling
cannot reconstruct rooms or surfaces absent from the capture. Depth Anything 3
is not part of the unchanged controls: Inpaint360GS must keep its native author
workflow and AuraFusion360 pins Marigold. Any future Depth Anything 3 use must be
a separately preregistered ablation after a measured failure, and its predicted
depth remains non-metric supporting evidence.

The method prerequisite and suite replay commands are:

```bash
python -m blueprint_pipeline.public_scene_method_prerequisites \
  --request docs/arm_decision_proof_v1/manifests/adp009a_method_prerequisite_request.v1.json \
  --repo-root "$PWD" --data-root "$ADP009A_DATA_ROOT" \
  --method-root "$ADP009A_METHOD_ROOT" \
  --output "$ADP009A_DATA_ROOT/methods/adp009a_method_prerequisite_receipt.v2.json"
python -m blueprint_pipeline.adp_aura_author_smoke_vast \
  --materialize-author-data \
  --prerequisite-receipt "$ADP009A_DATA_ROOT/methods/adp009a_method_prerequisite_receipt.v2.json" \
  --job-dir "$ADP009A_DATA_ROOT/aura_author_smoke/author_data_v1"
python -m blueprint_pipeline.adp_aura_author_smoke_vast \
  --repo-root "$PWD" \
  --aura-root "$ADP009A_METHOD_ROOT/AuraFusion360_official" \
  --sam2-root "$ADP009A_METHOD_ROOT/sam2" \
  --wonderworld-root "$ADP009A_METHOD_ROOT/WonderWorld" \
  --prerequisite-receipt "$ADP009A_DATA_ROOT/methods/adp009a_method_prerequisite_receipt.v2.json" \
  --author-data-root "$ADP009A_DATA_ROOT/aura_author_smoke/author_data_v1/data" \
  --author-data-receipt "$ADP009A_DATA_ROOT/aura_author_smoke/author_data_v1/adp_aura_author_data_materialization_receipt.json" \
  --expected-output-ply "$ADP009A_DATA_ROOT/aura_author_smoke/expected_output_v1/360-USID/sunflower/point_cloud/iteration_object_inpaint_init/point_cloud.ply" \
  --job-dir "$ADP009A_DATA_ROOT/aura_author_smoke/bundle_v14"
python -m blueprint_pipeline.public_scene_suite_materializer \
  --request docs/arm_decision_proof_v1/manifests/adp009a_materialization_request.v1.json \
  --repo-root "$PWD" --data-root "$ADP009A_DATA_ROOT" \
  --method-root "$ADP009A_METHOD_ROOT" \
  --output-root docs/arm_decision_proof_v1/manifests/adp009a_materialized_suite
python -m blueprint_pipeline.public_scene_inpainting_inputs \
  --request docs/arm_decision_proof_v1/manifests/adp009b_interiorgs_edit_input_request.v1.json \
  --repo-root "$PWD" --data-root "$ADP009A_DATA_ROOT" \
  --output-root "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_v1" \
  --receipt-output docs/arm_decision_proof_v1/manifests/adp009b_interiorgs_edit_input_receipt.v1.json
python -m blueprint_pipeline.public_scene_inpaint360_adapter \
  --input-receipt docs/arm_decision_proof_v1/manifests/adp009b_interiorgs_edit_input_receipt.v1.json \
  --input-root "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_v1" \
  --repo-root "$PWD" --data-root "$ADP009A_DATA_ROOT" \
  --method-root "$ADP009A_METHOD_ROOT/Inpaint360GS" \
  --output-root "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_inpaint360_adapter_v1" \
  --receipt-output docs/arm_decision_proof_v1/manifests/adp009b_inpaint360_adapter_receipt.v1.json
python -m blueprint_pipeline.public_scene_infusion_adapter \
  --input-receipt docs/arm_decision_proof_v1/manifests/adp009b_interiorgs_edit_input_receipt.v1.json \
  --input-root "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_v1" \
  --prerequisite-receipt "$ADP009A_DATA_ROOT/methods/adp009a_method_prerequisite_receipt.v2.json" \
  --repo-root "$PWD" --data-root "$ADP009A_DATA_ROOT" \
  --infusion-root "$ADP009A_METHOD_ROOT/Infusion" \
  --lama-source-root "$ADP009A_METHOD_ROOT/Inpaint360GS" \
  --output-root "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_infusion_adapter_v1" \
  --receipt-output docs/arm_decision_proof_v1/manifests/adp009b_infusion_adapter_receipt.v1.json
python -m blueprint_pipeline.public_scene_inpaint360_author_archive \
  --source-root "$ADP009A_METHOD_ROOT/Inpaint360GS" \
  --output docs/arm_decision_proof_v1/manifests/adp009a_inpaint360_author_archive_probe.v1.json
```

The successful Content Agents execution used a deterministic NVIDIA-compatible
derivative of the canonical SimReady control: visual purpose was normalized to
USD `default` for v0.5.2 bounding-box discovery and the grasp-identifier extent
was recomputed from its curve width. The canonical source USD remains unchanged.
The external run cost `$0.100418`, used zero retries, destroyed Vast instance
`46835085`, removed all staged provider objects, and left zero active Vast
instances. A pre-allocation gate now rejects unsupported config fields, invalid
render modes, missing model access, changed container/source/bundle identities,
empty default-purpose bounds, failed native dry runs, failed Material input
validation, dirty orchestrator commits, or mutated preflight receipts before a
GPU can be allocated. Raw provider status is no longer used for monitoring;
the safe status path allowlists non-secret fields.

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
