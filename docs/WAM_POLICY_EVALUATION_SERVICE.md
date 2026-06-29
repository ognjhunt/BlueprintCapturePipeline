# WAM Policy Evaluation Service

## Purpose

Blueprint's robot-team service is capture-first and substrate-agnostic. A real
capture package produces Task Evaluation Run and Post-Training Data Package
artifacts; an evaluation substrate then generates support evidence for ranking
customer policies or checkpoints.

World-action-model evaluation is first-class as a support/evaluator substrate.
It is not the robot policy for Unitree G1. Classical simulation remains
supported as a fallback, cross-check, or stricter physics lane.

## Paper-Aligned Proof Target

The proof target is policy comparison inside a configured evaluator: policy A
outperforms policies B and C on the same scenario matrix, observation protocol,
and scoring protocol. This mirrors the evaluator framing in OSCAR
(`https://arxiv.org/html/2606.04463v2`) and SC3-Eval
(`https://arxiv.org/html/2606.18610v3`): generated or simulated rollouts can be
used to rank policies, and evaluator quality is measured with rank fidelity and
success-rate correlation when paired real-world anchors exist.

This repo should therefore treat `policy_ranking_scorecard.json` as an
evaluator-bounded comparison artifact. It may state which policy or checkpoint
ranked higher inside the configured evaluator only when every candidate policy
covers the same required `scenario_eval_run_id` set with one attempt per
required run, score ranges are valid, and the top-policy margin is outside the
scorecard tie band. The scorecard records `required_scenario_eval_run_ids`,
`per_policy_coverage`, `coverage_complete`, `missing_by_policy`,
`extra_by_policy`, `attempt_count_by_policy`, and `comparison_blockers` so a
partial or asymmetric evaluator run cannot look like a fair policy comparison.
If the top two policies tie or land inside `ranking_confidence.tie_band`, the
scorecard may keep the evaluator-ranked rows but must set an ambiguous status
such as `completed_ambiguous_ranking` and leave `top_policy_id` empty. It must
not be rewritten as a broader completion claim. The ranking claim is limited to
generated-world policy-evaluation rank fidelity when measured. MMRV, Spearman,
Pearson, and success-rate error belong to calibration against accepted anchor
rollouts; they remain `not_measured` until those anchors exist and are not
proven by WAM execution alone.

Accepted real-world calibration anchors use
`accepted_real_world_anchor.v1`. A row can support evaluator-vs-IRL accuracy
measurement only when the predicted row and actual row join exactly on
`scenario_eval_run_id`, `policy_id`, `task_id`, and
`scenario_variation_instance_id`. The actual row must also carry an actual
success/failure result, owner evidence, signed operator or owner attestation,
and an accepted reviewer/calibration decision. If the anchor requests physical
evidence, the physical run evidence refs must be present. Loose, fallback, or
inferred joins can be recorded for follow-up diagnostics, but they must not be
accepted for calibration.
Calibration reports must keep `sim_vs_real_calibration_score=null` until enough
accepted paired anchors exist.

When enough accepted anchors exist, `sim_vs_real_calibration_report.json`
computes Spearman rank correlation, Pearson success-rate correlation, MMRV
(`mean_maximum_rank_violation`), mean absolute success-rate error, and
confidence intervals. Until then, or when anchor quality fails, the report must
block external accuracy claims with explicit blockers:
`insufficient_anchor_count`, `unmatched_prediction_rows`,
`unmatched_actual_rows`, `stale_anchor_rows`, and
`conflicting_anchor_rows`. These blockers do not invalidate sim-only beta
 ranking or evaluator-bounded policy comparison, but they do block
customer-specific SRCC/Pearson claims and public external-accuracy
claims.

Some older artifact names and schema fields, including
`robot_team_grade_eval_closure_manifest.json`, remain for compatibility with
existing Pipeline/WebApp readers. Treat those names as legacy closure-package
surfaces, not as the current proof target. The current claim boundary is the
evaluator-bounded policy comparison recorded by `policy_ranking_scorecard.json`
and `wam_eval_claim_boundary.json`.

Forward/inverse consistency is a reliability signal for generated episodes. It
can help decide whether to stop or distrust an evaluator rollout, but it is not
itself a task-success label or policy-ranking outcome.

## Backend Strategy

The preferred new learned-WAM evaluator candidate is `cosmos3_wam`, modeled as
Cosmos3-Nano behind the same replaceable adapter contract as the older
OSCAR/Cosmos lanes. That preference is a backend strategy, not a permanent
company dependency and not a public accuracy claim. It still requires an
explicit adapter command, checkpoint or provider runtime, run gates, visual
smoke, external success labels when used, external episode-consistency scoring
when used, and accepted real-world anchors before any external rank-fidelity
metric is claimed.

The repo keeps the older lanes for baseline and compatibility:

- `oscar_wam`: OSCAR fine-tunes `Cosmos-Predict2.5-2B` on 180,657 filtered
  episodes: 94,830 robot episodes and 85,827 human egocentric episodes. Its
  skeleton-conditioned RoboArena policy-eval result is MMRV 0.571, Spearman
  0.750, Pearson 0.852, and SISR delta 1.73pp. Its GPT-5 generated-video
  success scorer matched 78/100 human labels, had specificity 0.90, and missed
  about one third of real successes, so Blueprint keeps generated-video success
  labels separate from consistency and rank-fidelity calibration.
- `cosmos_wam`: Cosmos-Predict2.5 remains a legacy/advisory baseline. NVIDIA's
  Cosmos-Predict2.5 repository says the line is no longer under active
  development and future releases, features, docs, and community support are
  focused on Cosmos 3.
- `cosmos3_wam`: Cosmos3-Nano is the preferred configured candidate for new
  learned-WAM evaluator work. SC3-Eval initializes from Cosmos3-Nano and reports
  headline closed-loop Pearson 0.929 and MMRV 0.119. Its in-distribution online
  split is Pearson 0.984 / MMRV 0.022 versus Cosmos-Predict2.5 at 0.897 / 0.090.
  On the out-of-distribution online split, SC3-Eval's Pearson is 0.870 versus
  Cosmos-Predict2.5 at 0.871, while MMRV is better at 0.171 versus 0.195. That
  supports a rank-fidelity preference, not universal grading.
- `cosmos3_super`: high-cost adjudication candidate for contested rankings after
  cheaper screens pass; not the default local path.
- `cosmos3_edge`: the Cosmos 3 technical report describes Edge as a later
  release, so Blueprint does not treat it as a released/default runtime unless a
  future session reverifies availability from primary sources.

SC3-Eval is a recipe, not just a checkpoint. Blueprint models its contribution
as forward/inverse dynamics consistency, cross-view consistency, and
uncertainty-driven early termination. These are reliability and abstention
signals only. SC3-Eval's published scope is also narrow relative to Blueprint's
goal: 381 hours in one physical table-bussing scene, 12 object categories, three
camera views, seven policy checkpoints, and at most 20-second rollouts.

## Robot Policy Versus WAM

Do not treat the evaluator WAM as the robot policy. For Unitree G1
manipulation, the preferred policy lane is Unitree-specific:

- `unitree_lerobot_policy` for G1 Dex1/Dex3/gripper manipulation policies.
- `unitree_unifolm_vla_policy` for Unitree-native VLA checkpoints and commands.
- `unitree_unifolm_wma_policy` for Unitree-native world-model-action commands.
- `unitree_groot_n17_sonic_policy` for GR00T N1.7 + `UNITREE_G1_SONIC`
  action chunks through SONIC whole-body control and simulator-only Sim2Sim.
- `unitree_g1_policy` for locomotion/control stacks such as Unitree RL Gym.

UnifoLM VLA readiness requires a VLA checkpoint through
`BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT` or the provider-facing alias
`BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT`, plus the VLM backbone
`BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT`.

`openvla_policy` remains a generic VLA candidate for comparison and non-Unitree
policy work, but it must not be selected as the G1 policy path. `oscar_wam`,
`cosmos_wam`, fixture WAM, and generated WAM outputs are future-world or scoring
support artifacts unless a separate Unitree-specific policy endpoint consumes
the observation and emits normalized G1 actions.

The WAM control-loop proof is true only when the same Unitree-specific policy,
for example `unitree_groot_n17_sonic_policy`, is called again on a
WAM-generated next observation. A single policy action, provider-smoke import,
or generated rollout does not prove closed-loop manipulation.

Downstream code should use the Unitree provider registry's explicit fields when
deciding whether the robot-policy side is in place:
`unitree_hand_manipulation_policy_in_place`,
`selected_unitree_manipulation_runtime`, `selected_unitree_action_command`,
`selected_unitree_hand_policy`, `openvla_selected_for_g1_policy`, and
`wam_selected_for_g1_policy`. `selected_provider` is retained only as a legacy
first-configured-provider field and may point at locomotion only.

For local Unitree G1 locomotion/control proof on this machine, source
`.env.unitree.local`. It sets the verified `BLUEPRINT_UNITREE_G1_POLICY_ROOT`,
`BLUEPRINT_UNITREE_G1_POLICY_SOURCE_ROOT`, `BLUEPRINT_UNITREE_RL_GYM_ROOT`, and
`BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT` values. The Unitree lane is documented in
[`UNITREE_G1_POLICY_ENDPOINT_LANE.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md).

When this service consumes a MuJoCo endpoint job that already used a fresh
Unitree hand/action endpoint, `wam_manipulation_loop_readiness_manifest.json`
must carry that source fact as `source_unitree_endpoint_hand_policy_used=true`
and `unitree_hand_manipulation_policy_scope=endpoint_action_command`. That does
not by itself make `policy_observes_wam_generated_next_observation` true. The
WAM loop needs a generated next-observation frame that is useful enough for
policy requery and the same Unitree-specific endpoint must be re-queried on that
observation.

The companion `policy_requery_endpoint_readiness_manifest.json` distinguishes a
prior source endpoint proof from a currently live requery endpoint by recording
`source_endpoint_proof_is_not_current_live_endpoint`,
`live_policy_requery_endpoint_ready`, the configured endpoint URL/auth envs, and
the remaining visual-quality or endpoint-auth blockers. The
`single_step_wam_policy_requery_visual_candidate.json` artifact intentionally
splits first-frame policy feedback from full-rollout success review: a
scene-preserving first generated frame can make
`single_step_wam_policy_requery_proven=true` only if the endpoint response is a
fresh Unitree-specific policy inference. A replay-backed Unitree-family output
must instead set `unitree_g1_hand_policy_output_observed=true`,
`policy_requery_provider_replay_used=true`, and
`single_step_wam_policy_requery_proven=false`, while
`full_rollout_visually_useful_for_success_review=false` still blocks
`wam_success_label_from_generated_video` and forward/inverse consistency proof.

The MuJoCo G1 policy/WAM closed-loop helper also has a repo-local default
OSCAR-style next-observation generator for runs without a configured live WAM
command. It writes action-conditioned JPEG support frames, short MP4 segments,
and action, simulated-proprioception, and projected-skeleton conditioning
metadata, then allows a fresh Unitree policy command to be re-queried on those
generated observations. Its artifacts must remain labeled as default local
support output; they do not prove a live learned OSCAR/Cosmos checkpoint,
external sensor feedback, success scoring, or generated-world rank fidelity.

## WAM-Derived Perception And Observation Harness

The closed-loop evaluator supplies whatever the selected policy declares it can
consume at each step. The WAM does not have to natively generate every
modality. The supported architecture is:

```text
policy action
  -> WAM/default generator emits next RGB/video/multiview observation
  -> WAM-derived perception harness derives support observations from generated media
  -> harness joins evaluator-controlled nominal state and calibration when available
  -> policy observation adapter passes only declared policy-interface fields
  -> diagnostics/scoring artifacts consume the harness outputs with proof boundaries
  -> loop repeats, or terminates early when reliability is too low
```

The reusable local harness writes these artifacts under the WAM loop/eval
directory:

- `wam_derived_observation_bundle.json`
- `wam_derived_observation_manifest.json`
- `wam_perception_harness_checks.json`
- `wam_policy_observation_adapter_report.json`
- `wam_derived_observation_steps.jsonl`
- `wam_perception_backend_request.json` and
  `wam_perception_backend_result.json` when an external backend is explicitly
  enabled
- `wam_perception_harness_validation_report.json`
- `wam_false_success_reduction_metrics.json`
- `wam_perception_harness_review_report.md`

The first implementation is deterministic and fixture-backed. It can use
existing `object_index` and `eval_ready_task_grounding` records for object
labels, boxes, masks, crops, and target prompts, then estimate derived masks or
boxes, tracks, relative depth, 2D pose, contact likelihood, visual cue status,
reviewability, and uncertainty. If a real detector/depth backend is added later
it must sit behind the same artifact contract and record its backend kind,
command/env gate, status, blockers, and whether a real model actually ran.
External perception backends are opt-in through:

```bash
BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_KIND=sam3 \
BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND="your-backend-command" \
BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND=true
```

The harness passes the request/output paths through
`BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT`,
`BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT`, and
`BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR`. If the env gate or command is
missing, the backend result is blocked and the fixture/object-index path remains
the only local deterministic path. Secrets must be provided through the backend
process environment or local secret files; raw credentials are not required in
Pipeline artifacts.

For a sim-only end-to-end provider/harness proof, use:

```bash
python -m blueprint_pipeline.wam_sim_provider_e2e \
  --provider-mode real \
  --generated-frame <gpt-image2-or-wam-generated-frame.jpg> \
  --target-prompt "robot arm" \
  --sam3-weights <sam3.pt> \
  --depth-provider v2 \
  --pose-model yolo11n-pose.pt
```

If `--generated-frame` is omitted, the runner first looks for an existing
generated WAM/GPT-image frame under `robot_eval_jobs/`; if none exists it writes
a local synthetic AI-style start frame. It then creates generated next-frame
steps, runs the WAM-derived harness on each step, adapts the observation back to
the declared policy schema, and writes `wam_sim_provider_e2e_manifest.json`,
`wam_sim_provider_e2e_trace.jsonl`, generated step frames, and the normal
`wam_derived_observation_harness/` artifact family. This proves the sim
architecture path only. The manifest records that optional truth-label
validation was not requested and keeps generated frames separate from capture
truth, inferred depth separate from sensor depth, SAM3 masks separate from
physical truth, and contact likelihood separate from physical contact proof.

Depth providers are replaceable. The default local smoke path uses the
Transformers Depth Anything V2 small model because it is lightweight enough for
repo-local provider proof. Depth Anything 3 can be selected for stronger
simulation geometry experiments with:

```bash
BLUEPRINT_WAM_DEPTH_PROVIDER_KIND=da3 \
BLUEPRINT_ALLOW_WAM_AUTO_DA3_PROVIDER=true \
BLUEPRINT_WAM_DA3_MODEL_ID=depth-anything/DA3-BASE \
python -m blueprint_pipeline.wam_sim_provider_e2e --provider-mode real --depth-provider da3
```

DA3 is optional and fail-closed. If the `depth_anything_3` package or selected
weights are unavailable, the backend records `da3_depth_provider_package_missing`
or the concrete provider error instead of falling back silently or implying that
metric sensor depth exists. Use `depth-anything/DA3METRIC-LARGE` only when the
runtime, license posture, and calibration needs fit the lane; generated-pixel
metric-looking estimates still remain support artifacts unless tied to a real
calibrated depth source.

For reusable GPU runs, build a dedicated WAM perception harness image context:

```bash
blueprint-build-wam-perception-harness-gpu-image \
  --image-ref docker.io/nijelhunt/blueprint-wam-perception-harness:20260626-cu126
```

The context writes `Dockerfile.wam-perception-harness-gpu`, build/push scripts,
`run_image_healthcheck.sh`, `prepare_model_mounts.sh`, and
`wam_perception_harness_gpu_image_manifest.json`. The Dockerfile bakes the
Blueprint harness code, CUDA PyTorch, `transformers`, `ultralytics`, Depth
Anything V2 model cache, YOLO pose cache, the real-provider probe, and the
sim-provider E2E runner. It does not bake raw Docker, DigitalOcean, object-store,
or Hugging Face credentials. SAM3 weights are not baked by default and are
expected at `/models/sam3/sam3.pt` through a mounted model directory or a
provider-side secret-gated fetch. This prevents per-run Python dependency
reinstalls while keeping model weights and credentials explicit.

For a one-command proof probe of the real-provider lane, use:

```bash
python -m blueprint_pipeline.wam_real_provider_validation_probe run \
  --generated-frame <wam-generated-frame.jpg> \
  [--validation-set <capture-backed-validation-rows.json>]
```

The probe writes
`robot_eval_jobs/wam_real_provider_validation_probe_<timestamp>/wam_real_provider_validation_proof_manifest.json`
and the normal `wam_derived_observation_harness/` artifact family. It records
whether SAM3 weights are present through `SAM3_WEIGHTS_PATH` or
`BLUEPRINT_SAM3_WEIGHTS_PATH`, whether depth and pose provider commands are
configured through `BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND` and
`BLUEPRINT_WAM_POSE_PROVIDER_COMMAND`, and whether labeled validation rows were
supplied. The validation set is optional for sim-only runs: missing or invalid
rows are recorded as `diagnostic_issues`, not as launch
or policy-ranking blockers. Provider setup can still fail its own probe, but
labeled validation absence does not block sim-only candidate selection.

When supplied for an external-accuracy diagnostic, the validation set path should
contain real/capture-backed labeled rows, not only fixture expectations. At
least one row should carry an accepted truth flag such as
`capture_backed=true`, `capture_truth=true`, `real_labeled_validation=true`, or
`accepted_real_world_anchor=true`; at least one validation label such as
`actual_success`, `capture_success`, `expected_target_visible`,
`expected_contact`, or `expected_object_id`; and a provenance reference such as
`source_capture_path`, `source_artifact_path`, `source_video_path`,
`source_frame_path`, `source_label_path`, `evidence_path`, or
`operator_attestation_path`. Accepted probe rows must also include a target
prompt, carry reviewer or label provenance, and match the probed frame when a
frame ID or frame path is supplied. Provider-only outputs such as SAM/depth/pose
result files can support reviewer inspection, but they are not accepted as the
validation source by themselves. Files that exist but do not meet that row
contract produce optional validation diagnostics and leave label-based accuracy
or false-success reduction as `not_measured`; they do not block generated-frame
provider runs by themselves.

Authoritative/evaluator-controlled channels stay separate from pixel-inferred
channels. Action command/history, gripper command, nominal joint state,
nominal/FK end-effector state, projected robot skeleton refs, action-conditioning
metadata, controller limits, and capture/sim camera calibration are labeled as
evaluator-controlled or nominal when present. Object masks, object IDs/tracks,
estimated depth, estimated object pose, visual gripper-object relation, contact
likelihood, success/failure visual cues, reviewability, and uncertainty are
labeled as derived from WAM-generated media.

The policy adapter is fail-closed against the declared policy observation
schema. RGB-only policies receive only the generated RGB/frame fields and any
declared nominal state fields; masks, depth, pose, contact likelihood, and
uncertainty stay available to diagnostics, scoring gates, and early termination
reports. Policies that declare RGB-D, mask, state, contact-likelihood, or
uncertainty support receive only those declared enriched fields, and
`wam_policy_observation_adapter_report.json` records requested, supplied, and
withheld fields.

Harness reliability checks can recommend early termination when a generated
frame is missing, too dark/flat, the target is offscreen, object identity is
lost, relative depth jumps unrealistically, or the action/robot trace no longer
matches the visual observation. Early termination blocks policy requery and
success scoring unless an explicit review path later accepts the artifact.
When multiple generated views are supplied, `multiview_consistent` records view
count, readable-view count, per-view frame quality, blockers, and confidence.
Single-view rollouts remain explicitly `not_evaluated` for multiview rather
than implying cross-view proof.

When camera calibration and calibration-quality metadata are available from
capture or sim artifacts, the harness can use them for calibrated projection
support such as metric-depth estimates from generated pixels and 3D camera-frame
pose estimates. Those fields still record `metric_depth_truth=false` and
`physical_pose_truth=false` unless a separate real calibrated depth/pose source
exists; calibration metadata does not turn WAM pixels into sensor measurements.

The review-acceptance path is explicit. A low-confidence step can unblock
generated-rollout success scoring only when a review acceptance payload marks
the step accepted for success scoring and includes a reviewer plus evidence
refs. Policy requery and ranking claims still follow the policy adapter and
generated-world rank fidelity gate.

Validation metrics are measured only against supplied labeled validation rows
from real/capture-backed clips or accepted anchors. The validation report can
measure object-id accuracy, target-visibility accuracy, contact-likelihood
accuracy, and false-success reduction versus plain generated-video scoring. If
no validation set is supplied, these metrics remain blocked/not measured; the
fixture harness contract alone does not prove an accuracy gain.

Critical claim boundaries:

- Harness outputs are derived observations, not real sensors.
- Inferred depth is not sensor depth.
- SAM/object masks or fixture boxes are not physical truth.
- Mask overlap, proximity, or contact likelihood is not stable grasp/contact proof.
- Generated rollout success, generated-video labels, and harness outputs are
  support artifacts; ranking claims require a scoped generated-world
  policy-evaluation rank-fidelity gate.
- WAM/harness outputs remain support artifacts for Task Evaluation Runs and
  Post-Training Data Packages grounded in capture truth.

Scene WAM episode packets can also prepare capture-derived robot POV seed
frames through the depth-splat synthesis path. For each task and robot profile,
the packet writes source QA, a coverage/quality report, a contact sheet, and
recapture guidance when no candidate clears the quality gate. A passing
depth-splat candidate can seed the initial WAM observation, but it remains a
synthetic/model-derived support artifact. It is not raw capture truth or
ranking evidence until the relevant generated-world policy-evaluation
rank-fidelity gate passes.

Synthetic fallback observations and synthetic 2D seeds are not allowed to unlock
live or review-quality WAM launch paths unless
`BLUEPRINT_ALLOW_SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT=true` is set for an
explicit experiment. Even then, provider success remains separate from
`visually_useful_rollout`, and artifacts must label `capture_truth=false` and
`geometry_truth=false`.

## Local Substrates

The substrate registry is written as `evaluation_substrate_registry.json` and
currently supports:

- `fixture_wam`: deterministic repo-local WAM fixture for tests and local demos.
- `cosmos3_wam`: live or owner-provided Cosmos-style WAM adapter, blocked until
  configured.
- `oscar_wam`: live or owner-provided OSCAR-style WAM adapter, blocked until
  configured.
- `classical_sim_mujoco`: MuJoCo or owner-command simulator path.
- `classical_sim_isaac`: Isaac Sim / Isaac Lab / owner GPU simulator path.
- `recorded_trace`: customer or owner recorded trace replay path.

Legacy simulator aliases such as `mujoco`, `isaac_sim`, and `fixture` are
accepted at the contract boundary and normalized into the registry.

## Fixture End-to-End Path

The local fixture path requires no GPU, secrets, provider calls, or live VLM.
It starts from an existing robot-eval job directory with
`scenario_eval_matrix.json`, `policy_package_manifest.json`, and `job_request.json`.

```bash
blueprint-run-robot-eval-job \
  --capture-root /path/to/<capture-root> \
  --job-request /path/to/robot_eval_job_request.json \
  --job-id <job_id> \
  --provisioner fixture_local \
  --simulator fixture \
  --evaluation-substrate fixture_wam
```

The WAM fixture can also be run directly against an existing job directory:

```bash
blueprint-run-wam-fixture-evaluator \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --evaluation-substrate fixture_wam
```

## Live Or Owner WAM Adapter Path

`cosmos3_wam` and `oscar_wam` are adapter contracts, not hardwired product
dependencies. They fail closed unless all of these are present:

- an explicit local run gate: `--allow-wam-provider` for robot-eval jobs or
  `--allow-live-provider` for the WAM evaluator CLI
- `BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true`
- a provider command such as `--wam-provider-command cosmos3_wam=/path/to/adapter`
  or `BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND`
- provider auth in env only: Cosmos accepts one of
  `BLUEPRINT_COSMOS3_WAM_API_KEY`, `COSMOS_API_KEY`, or `NVIDIA_API_KEY`;
  OSCAR accepts one of `BLUEPRINT_OSCAR_WAM_API_KEY` or `OSCAR_WAM_API_KEY`

Example job-level adapter run:

```bash
BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true \
BLUEPRINT_COSMOS3_WAM_API_KEY=<redacted> \
blueprint-run-robot-eval-job \
  --capture-root /path/to/<capture-root> \
  --job-request /path/to/robot_eval_job_request.json \
  --job-id <job_id> \
  --provisioner fixture_local \
  --simulator fixture \
  --evaluation-substrate cosmos3_wam \
  --allow-wam-provider \
  --wam-provider-command cosmos3_wam="/path/to/cosmos_adapter" \
  --wam-artifact-output-uri gs://customer-bucket/<job_id>/wam \
  --wam-provider-max-retries 1 \
  --wam-provider-timeout-seconds 120
```

The adapter receives `BLUEPRINT_WAM_PROVIDER_INPUT`,
`BLUEPRINT_WAM_PROVIDER_OUTPUT`, `BLUEPRINT_WAM_PROVIDER_SUBSTRATE`, and,
when supplied, `BLUEPRINT_WAM_PROVIDER_ARTIFACT_OUTPUT_URI`. It must write JSON
with `rollouts` or `wam_rollout_results.rollouts`. Secrets must stay in env and
must not be written into artifacts.

## WAM Artifact Contract

When WAM evaluation is requested, the job writes:

- `evaluation_substrate_registry.json`
- `wam_evaluation_request.json`
- `wam_provider_runtime_package.json`
- `wam_provider_execution_manifest.json`
- `wam_provider_cost_control_ledger.json`
- `wam_provider_artifact_upload_proof.json`
- `wam_policy_interface_binding.json`
- `wam_rollout_manifest.json`
- `wam_rollout_results.json`
- `vision_success_labels.json`
- `wam_vision_success_review_queue.json`
- `wam_episode_consistency_request.json`
- `wam_episode_consistency.command.json` when an external scorer command runs
- `wam_consistency_checks.json`
- `wam_derived_observation_bundle.json` when the derived observation harness runs
- `wam_derived_observation_manifest.json` when the derived observation harness runs
- `wam_perception_harness_checks.json` when the derived observation harness runs
- `wam_policy_observation_adapter_report.json` when the derived observation harness runs
- `wam_derived_observation_steps.jsonl` when the derived observation harness runs
- `wam_perception_backend_request.json` when an optional external perception
  backend is explicitly enabled
- `wam_perception_backend_result.json` when an optional external perception
  backend is explicitly enabled
- `wam_perception_harness_validation_report.json` when the derived observation
  harness writes validation metrics or records that validation labels are absent
- `wam_false_success_reduction_metrics.json` when labeled validation rows allow
  comparison against plain generated-video false-success labels
- `wam_perception_harness_review_report.md` when the derived observation harness
  writes its reader-facing reliability and claim-boundary report
- `normalized_attempt_trace.json`
- `failure_labels.json`
- `prediction_outcome_ledger.json`
- `calibration_report.json`
- `breakage_library.json`
- `policy_ranking_scorecard.json`
- `wam_eval_claim_boundary.json`
- `real_world_validation_followup_request.json`
- `srcc_validation_plan.json`
- `wam_real_world_validation_anchor_manifest.json`
- `wam_customer_validation_envelope.json`
- `wam_production_ops_manifest.json`
- `wam_classical_sim_cross_check_plan.json`
- `candidate_selection_report.json`
- `candidate_selection_report.md`
- `customer_handoff_report.json`
- `customer_handoff_report.md`

The fixture evaluator deterministically generates rollout support manifests,
fixture vision labels, normalized attempts, failure labels, a policy ranking
scorecard, a candidate selection report, a customer handoff report, and a
real-world validation follow-up request. Live providers are represented through
the same artifacts and remain blocked until adapter commands, auth envs, gates,
and output rollouts are present.

`failure_labels.json` is a review-backed diagnosis contract, not an authority
upgrade. Each failed-attempt label must carry `failure_mode_ids`,
`evidence_refs`, `source_trace_refs`, `frame_or_clip_refs`,
`visual_smoke_ref`, `confidence`, `review_status`,
`reviewer_acceptance_required`, `root_cause_category`,
`remediation_candidate`, and `unknown_when_evidence_weak`. Generated WAM
rollout labels must either point at reviewable media or be explicitly marked
`non_reviewable_failure_hypothesis`. Fixture or heuristic labels keep
`proof_effect=none_until_review_accepted_or_real_world_validation_supplied` and
remain non-authoritative until accepted review or real-world validation is
supplied.

`failure_diagnosis_coverage_complete` means every failed attempt has a label
with failure modes, evidence refs, and review status.
`failure_diagnosis_complete` additionally requires accepted or reviewable
status. Robot-eval closure blocks failure diagnosis completion when labels lack
evidence refs, failure modes, review status, or accepted/reviewable status.
Non-reviewable generated rollout hypotheses are preserved as blockers so
reports can explain why diagnosis remains incomplete.

`breakage_library.json` aggregates failed attempts by `policy_id`, `task_id`,
`scenario_id`, `failure_mode_id`, and `root_cause_category`. Dominant failure
modes include exemplar failed attempts plus available media and evidence refs,
so policy-improvement handoffs can point reviewers at exact traces without
treating generated/sim labels as real-world truth.

For a scorecard to support robot-team closure, it must be an evaluator-bounded,
non-overclaiming comparison with at least two policies, symmetric required-run
coverage, no missing required scenarios, no extra unknown scenarios, valid
score ranges, and no scorecard comparison blockers. High OOD or uncertainty
is a sim-ranking confidence issue: it downgrades
`ranking_confidence.confidence_level` and can require targeted reruns, but it
does not add an IRL-data blocker to candidate selection.

`candidate_selection_report.json` is the product handoff for the near-term
question: which policy performed best in this evaluator, and what broke. It
records `top_policy_id` only when the comparison is symmetric, decisive, outside
the tie band, not low-confidence, and not blocked by visual-review evidence.
Otherwise it keeps the diagnostic `evaluator_top_policy_id` visible and reports
a candidate shortlist instead of forcing a winner. It also carries the runner-up,
predicted success-rate margin, tie or ambiguity status, scenario matrix coverage,
decisive scenarios where candidates diverged, high-uncertainty scenarios, OOD
blockers, visual-review blockers, dominant failure modes, exemplar evidence
refs, failure clusters, and sim-ranking rerun recommendations.

Failure clusters are post-training data package hooks, not root-cause
certification. Each cluster should say what data to collect, which scenario
variants to add, and which policy adapter or checkpoint to retry. When failure
evidence is weak, the report must use `unknown_needs_review` instead of
inventing a root cause.

The report's boundary statement is explicit: it is a sim-ranking handoff, and
IRL validation artifacts are outside its pass/fail state.

Generated next-observation media cannot unlock a review-grade winner or
review-grade success/failure label from booleans alone. The scorecard gate must
carry a passed `persistent_wam_short_visual_sanity_manifest.json` or equivalent
inline short-sanity manifest with `visual_profile=review_quality`,
`short_visual_sanity_passed=true`, `visually_useful_rollout=true`, contact-sheet
or review-media refs, and provenance refs such as the source QA, visual-quality
report, video-review status, frame stats, or review video. Each generated-media
success/failure label used for ranking must also carry review-label refs. If
the media is fixture-only, visually weak, missing the short-sanity manifest, or
missing review-label refs, `policy_ranking_scorecard.json`,
`failure_labels.json`, `candidate_selection_report.json`, and
`visual_review_blocker_summary.json` must keep the review-grade ranking or
failure diagnosis blocked while preserving diagnostic evaluator rows and
blocker summaries. The gate remains a reviewability/support gate only; it does
not turn generated observations into raw capture, sensor truth, task-success
proof, or field evidence.

Forward/inverse episode consistency is intentionally separate from WAM/provider
execution and from the evaluator. The evaluator can prepare
`wam_episode_consistency_request.json` and normalize an external scorer command
into `wam_consistency_checks.json`, but it must not mark
`forward_inverse_consistency_proven=true` from generated rollout existence alone.
The external scorer contract is documented in
[`WAM_EPISODE_CONSISTENCY_SCORER.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/WAM_EPISODE_CONSISTENCY_SCORER.md).

## Policy Autoresearch

Policy autoresearch can call a WAM evaluator command through the same command
hook used for MuJoCo:

```bash
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --evaluation-substrate fixture_wam \
  --evaluator-command "python -m blueprint_pipeline.policy_autoresearch_wam_fixture_evaluator"
```

External evaluator commands receive both legacy and substrate-aware environment
variables:

- `BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE`
- `BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE`
- `BLUEPRINT_POLICY_AUTORESEARCH_MATRIX`
- `BLUEPRINT_POLICY_AUTORESEARCH_RECIPE`
- `BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT`

Promotion still depends on the frozen scenario/eval matrix, train/heldout split,
success and failure labels, safety/contact gates, and claim boundaries.

## Claim Boundaries

Generated WAM rollouts are model-derived support artifacts. They are not raw
capture evidence. Ranking claims require the scoped generated-world
policy-evaluation rank-fidelity gate, reported with MMRV, Spearman, and
Pearson when measured.

SC3-Eval-style and OSCAR/Cosmos-style results are credible research signals, but
they do not prove high SRCC for arbitrary customer hardware, policies, sites, or
task families. A customer-specific SRCC or Pearson claim requires paired real
world validation rollouts with exact `scenario_eval_run_id` joins, policy or
checkpoint IDs, and owner evidence or operator attestation.

Passing WAM heldout evaluation can support policy ranking, failure discovery,
and a real-world validation request. It cannot approve deployment by itself.
