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

The approved [public-scene transition decision](ADP_009_PUBLIC_SCENE_TRANSITION_DECISION.md)
selects AuraFusion360 for ADP v1, preserves Inpaint360GS and InFusion as
nonblocking typed abstentions, moves real metric transfer from ScanNet++ to the
mandatory fresh Raw V3.2 capture, and accepts the bounded construction
rehearsal without rewriting the historical `7/10` suite. The next active item is
ADP-009D: a development-only Franka policy rehearsal on the exact sealed
Aura/SAGE/SimReady scene before fresh capture.

| Item | Status | Observed evidence |
|---|---|---|
| ADP-009A | `partial` | The immutable historical [`public_scene_suite_index.v1.json`](manifests/adp009a_materialized_suite/public_scene_suite_index.v1.json) admits seven of ten roles from actual bytes and execution receipts: selected InteriorGS scene `840313`, its exact SAGE-3D companion, NVIDIA USD Content Agents v0.5.2, AuraFusion360's unchanged author smoke, the native Isaac/PhysX positive control, the exact approved match-v2 SimReady can, and a seal-before-truth-release controlled-background completion. Scene `840313` retains target `ins160` (`canned_beverage`) mapped to removable collider `/Root/ZHQYGJJVAJYEYPTUKY888888`; rejected candidate `841244` and shortlist failures remain retained. Exact SimReady `run_011` passed native Isaac Sim 6.0.1 drop/contact, slide, tip, and gripper probes. The index remains truthfully blocked on Inpaint360GS author-data license authority, InFusion checkpoint license, and ScanNet++ transfer. Under the 2026-08-06 transition these are historical nonblocking abstentions, not admissions and not a `10/10` claim. |
| ADP-009B | `observed_complete` | The frozen, non-orbiting InteriorGS input request materializes eight lossless `2048x1536` RGB frames and eight OBB-plus-contained-Gaussian masks for selected scene `840313`, target `ins160`; the [input receipt](manifests/adp009b_interiorgs_edit_input_receipt.v1.json) binds actual source/render bytes, camera transforms, Git source identity, and executed commands. Two retained Inpaint360GS scene experiments completed or failed honestly and were rejected for visual artifacts. AuraFusion360 `live_v12_openclip_execute` completed the exact released-source workflow and materially improved locality: mean PSNR `39.30477`, windowed SSIM `0.99231201`, LPIPS `0.01379092`, and `0.266055%` of outside-mask pixels changing by more than `20/255`. The project-owner [human-review receipt](manifests/adp009b_aura_human_review_receipt.v1.json) accepts it for the bounded internal hybrid control while retaining faint residue and representation limitations. The exact [hybrid replacement seal](manifests/adp009b_hybrid_replacement_seal_receipt.v1.json) binds Aura, matching SAGE collision, inactive source collider, preserved cabinet support, exact match-v2 SimReady can, official SimReady Foundation validation, and native Isaac drop/contact, slide, tip, and gripper lift/release results. Inpaint360GS author smoke and the prepared [InFusion adapter](manifests/adp009b_infusion_adapter_receipt.v1.json) remain nonblocking typed abstentions; neither is relabeled as executed or admitted. This completion is bounded to public-scene construction and does not establish hidden-background truth, real metrology, policy validity, or physical evidence. |
| ADP-009C | `observed_complete` | The Blueprint-authored known-background case passed its preregistered factual RGB/depth recovery checks: mean RGB mask PSNR `37.120493` dB, crop SSIM `0.983894`, boundary MAE `1.476368/255`, depth RMSE `0.006581` m, maximum p95 depth error `0.009391` m, and plane RMSE `0.006453` m. This is known synthetic truth, not InteriorGS hidden-background truth or real measurement. The retained [ScanNet++ access outcome](manifests/adp009c_scannetpp_access_outcome.v1.json) remains blocked and immutable; the required real metric transfer now occurs through the fresh Raw V3.2 capture rather than this public-data item. |
| ADP-009D | `partial` | The sealed Aura/SAGE/SimReady scene and exact native Isaac object probes are ready, but no Franka has yet been registered into that scene and no two-candidate learned-policy Task Evaluation Run has executed. The required next terminal is one basic pick-and-place rehearsal with exactly two frozen runnable policy candidates, complete policy-input media, deterministic non-policy scoring, replay, and provider teardown. Cosmos remains an auxiliary world-model/evaluator lane rather than a policy candidate unless an action-producing contract is separately proven. |

AuraFusion360 InteriorGS run `live_v12_openclip_execute` completed the exact
released source workflow through 10,000 inpaint-finetune steps with zero retries.
It retained a native 2D-Gaussian-surfels PLY with 415,265 vertices,
106,309,429 bytes, and SHA-256
`cbb05fc8e6da6ecdb72464f3b115f63e8747e2b67e97c309b4e40952b33000bd`,
plus eight native `2048x1536` renders at the frozen cameras. Estimated cost was
`$1.742443`; teardown and staged-object cleanup prove no continuing spend. The
[execution receipt](manifests/adp009b_aurafusion360_execution_receipt.v1.json)
is `executed_candidate`, not self-admitted. A deterministic native-2DGS render
manifest opens and rehashes the eight PNGs and derives their camera IDs from the
adapter inventory because the independent Spark verifier requires a third 3DGS
scale that Aura's native 2DGS PLY intentionally lacks. The resulting outside-mask
measurement (`sha256:eb3fb61eb5fe7da98728b3280bf832b49be0ff0684257b761d55599a88a5a4a4`)
is materially better than both Inpaint360GS runs: mean PSNR `39.30477`, windowed
SSIM `0.99231201`, LPIPS `0.01379092`, and `0.266055%` of outside-mask pixels
changing by more than `20/255`. Visual review of all eight before/mask/after
triplets finds no Inpaint360-style spikes or translucent sheets, but does find a
faint prismatic residue at the former can location and a white edge fringe in one
oblique view. Aura therefore remains a substantially improved visual candidate,
not admitted successful completion.

The exact SimReady can is now statically composed with the matching SAGE
collision proxy in a separate digest-bound USD layer. The deterministic
[replacement receipt](manifests/adp009b_simready_replacement_receipt.v1.json)
opens and hashes the actual collision USD, Aura PLY, and exact SimReady asset;
deactivates only target collider `/Root/ZHQYGJJVAJYEYPTUKY888888`; and preserves
the TV-cabinet support collider. Direct mesh inspection found four overlapping
horizontal support triangles at `z=0.5264650138348479` m with `0` m height span,
`0` degrees maximum tilt, and `0.785568074625` square meters of combined triangle
area. No mesh smoothing or generated support geometry was needed. Because the
SimReady asset uses a base-centered local datum, its placement is
`[3.4681748, -3.3100837, 0.5264650138348479]` m, a measured `+1.499910722` mm
support correction from the publisher OBB bottom. The project owner accepted
match-v2's eight-view identity, scale, and pose for native validation. Paid run
`vast_match_v2_native_run_007` then executed the same exact candidate on one
L40S with zero retries. All eight native OVRTX `2048x1536` RGB/depth renders
passed. The native OVPhysX 0.4.13 probe loaded the exact SAGE support mesh,
observed a `0.0493258238` m drop and 38 contact steps, and settled the can at
`z=0.5264592` m, only `0.0000057968` m below the expected support height,
with zero final motion and `0.00775` degrees upright rotation error. Material,
Texture, Physics, and Validation Agents also returned success. The run cost an
estimated `$0.196057`, destroyed Vast instance `46956437`, removed staged
objects, and a fresh independent inventory probe found zero active Vast
instances.

The digest-bound native visual review receipt is retained outside Git at
`simready/replacement_840313_match_v2/native_visual_review_run007/` with digest
`sha256:7b95577d254d9a1c4764f361362e7cd903b30e19c7c666f9d89e1b77a6357f51`.
It derives eight before/after pairs from the actual returned OVRTX arrays and
sealed Aura frames. These are native object-layer renders composited over Aura,
not native OVRTX renders of the 3DGS background. The project owner's explicit
decision to continue with this Aura result is now bound by the
[human-review receipt](manifests/adp009b_aura_human_review_receipt.v1.json) to
the Aura execution, the eight-camera locality measurement, and all 32 retained
review images. That receipt accepts a visual candidate for the bounded internal
hybrid replacement control; it does not manufacture hidden-background truth,
technical inpainting admission, or physical evidence. The exact unchanged
match-v2 USD subsequently passed the official SimReady Foundation
`Prop-Robotics-Physx 2.0.0` profile from an isolated byte-identical validation
input. Canonical Vast run `run_011` then loaded the exact SAGE collision layer
and match-v2 replacement in Isaac Sim 6.0.1 and passed four frozen 360-step
probes: drop/contact/settle, a `0.0041117726` m bounded slide, tip stability, and
gripper contact/lift/release with `0.0300313830` m observed lift. It confirmed
one replacement, the source target collider inactive, and the publisher support
collider active. The zero-retry run cost `$0.159906`, destroyed instance
`46976013`, removed both staged objects, and a fresh provider inventory returned
zero active Vast instances. This admits the exact-SimReady component. The
[hybrid replacement seal](manifests/adp009b_hybrid_replacement_seal_receipt.v1.json)
now fail-closed joins that admitted component and exact runtime digest to the
same Aura execution, project-owner visual decision, static replacement layer,
scene `840313`, target `160`, inactive source collider, and preserved support
collider. This completes the bounded internal hybrid replacement control, not
ADP-009B: the composite is not a native render of the 3DGS background, Aura has
no hidden-background truth, InFusion remains blocked, and all simulation
evidence remains distinct from physical truth.

Suite replay now fails closed on file identity as well as JSON shape. The
materializer opens the ten component manifests and receipts plus every referenced
artifact under explicit repository, data, and method roots; it recomputes their
sizes, SHA-256 values, canonical digests, roles, source-project identities,
statuses, and cross-bindings before matrix completion is possible. The current
blocked `7/10` replay opened and verified the referenced bytes, recomputed the
controlled-case scores from the actual outputs, and emitted index receipt
`sha256:4ac6de73e9d82ff881a7f662c58340700c76fa3762fa8ab25271d29cc77d9f43`.
It preserves the exact-SimReady component identity as the contract-required
`Blueprint-controlled` value and admits it only after the static profile, four
native Isaac probes, teardown, and staged-object cleanup all pass.

The exact 2026-08-02 Isaac development runtime, execution, teardown, and
provider-zero receipt bytes were recovered and retained outside Git under
`physics_positive_control/isaac_physx_attempt17_20260802/`. The materializer
recomputes all four file hashes and canonical digests, validates both nested
Isaac Sim 6.0.1 execution bundles, and cross-binds the request, image, instance,
teardown, and API-confirmed provider-zero evidence. This admits the separate
physics-positive-control role only; it does not establish exact-scene behavior,
physical truth, production readiness, or qualification of the replacement can.

Candidate `0787_841244` remains a retained, rejected warm start rather than the
selected scene. Direct inspection could not establish a suitable target whose
InteriorGS instance identity mapped to a separately removable SAGE collider
without unrelated collision overlap. The expanded preregistered survey selected
scene `840313` instead and retained its whole-splat room survey and target
closeups. The InteriorGS/SAGE release provides a metric frame, OBBs, and static
collision pairing, not a measurement-authoritative local surface mesh. Rendered
cameras and RGB are therefore synthetic method inputs/self-consistency probes;
external metric depth remains a validation oracle unless a released method
explicitly accepts it through a preregistered adapter. The separate Blueprint-
authored case now demonstrates the seal-before-clean-background-release
firebreak, but does not establish InteriorGS hidden-background truth.

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
python -m blueprint_pipeline.public_scene_simready_replacement \
  --request docs/arm_decision_proof_v1/manifests/adp009b_simready_replacement_request.v1.json \
  --repo-root "$PWD" --evidence-root "$ADP009A_DATA_ROOT" \
  --output-usda "$ADP009A_DATA_ROOT/simready/replacement_840313/collision_and_replacement.usda" \
  --output-receipt docs/arm_decision_proof_v1/manifests/adp009b_simready_replacement_receipt.v1.json
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
python -m blueprint_pipeline.public_scene_aura_execution \
  --adapter-receipt docs/arm_decision_proof_v1/manifests/adp009b_aurafusion360_adapter_receipt.v1.json \
  --runtime-result "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/immutable_execution/adp_aura_interiorgs_result.json" \
  --run-result "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/adp_aura_interiorgs_vast_result.json" \
  --evidence-root "$ADP009A_DATA_ROOT" --repo-root "$PWD" \
  --receipt-output docs/arm_decision_proof_v1/manifests/adp009b_aurafusion360_execution_receipt.v1.json
python -m blueprint_pipeline.public_scene_aura_native_render \
  --adapter-receipt docs/arm_decision_proof_v1/manifests/adp009b_aurafusion360_adapter_receipt.v1.json \
  --execution-receipt docs/arm_decision_proof_v1/manifests/adp009b_aurafusion360_execution_receipt.v1.json \
  --evidence-root "$ADP009A_DATA_ROOT" \
  --output "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/immutable_execution/aura_native_exact_camera_manifest.v1.json"
python -m blueprint_pipeline.public_scene_inpainting_locality \
  --before-dir "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_v1/images" \
  --mask-dir "$ADP009A_DATA_ROOT/inpainting_inputs/840313_ins160_v1/masks" \
  --after-render-manifest "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/immutable_execution/aura_native_exact_camera_manifest.v1.json" \
  --output "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/immutable_execution/aura_native_locality_measurement.v1.json" \
  --approved-root "$ADP009A_DATA_ROOT" --dilation-pixels 16 \
  --lpips-checkpoint-digest sha256:df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0 \
  --lpips-backbone-digest sha256:7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02
python -m blueprint_pipeline.public_scene_inpaint360_author_archive \
  --source-root "$ADP009A_METHOD_ROOT/Inpaint360GS" \
  --output docs/arm_decision_proof_v1/manifests/adp009a_inpaint360_author_archive_probe.v1.json
python -m blueprint_pipeline.public_scene_simready_visual_review \
  --native-provider-result "$ADP009A_DATA_ROOT/simready/content_agents/vast_match_v2_native_run_007/job_live/immutable_execution/adp_content_agents_vast_result.json" \
  --exact-camera-manifest "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/immutable_execution/aura_native_exact_camera_manifest.v1.json" \
  --frame-root "$ADP009A_DATA_ROOT/aura_interiorgs/live_v12_openclip_execute/immutable_execution/artifacts/final_frames" \
  --evidence-root "$ADP009A_DATA_ROOT" \
  --output-root "$ADP009A_DATA_ROOT/simready/replacement_840313_match_v2/native_visual_review_run007"
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
