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
(`https://arxiv.org/html/2606.18610v1`): generated or simulated rollouts can be
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
not be rewritten as deployment readiness, physical robot readiness, safety
approval, or a general external completion claim. MMRV, Spearman, Pearson, and
success-rate error belong to calibration against accepted real-world anchor
rollouts; they remain `not_measured` until those anchors exist and are not
proven by WAM execution alone.

Accepted real-world calibration anchors use
`accepted_real_world_anchor.v1`. A row can support evaluator-vs-IRL accuracy
measurement only when the predicted row and actual row join exactly on
`scenario_eval_run_id`, `policy_id`, `task_id`, and
`scenario_variation_instance_id`. The actual row must also carry an actual
success/failure result plus owner evidence or an operator attestation.
Calibration reports must keep `sim_vs_real_calibration_score=null` until enough
accepted paired anchors exist.

When enough accepted anchors exist, `sim_vs_real_calibration_report.json`
computes Spearman rank correlation, Pearson success-rate correlation, MMRV
(`mean_maximum_rank_violation`), mean absolute success-rate error, and
confidence intervals. Until then, or when anchor quality fails, the report must
block deployment/external accuracy claims with explicit blockers:
`insufficient_anchor_count`, `unmatched_prediction_rows`,
`unmatched_actual_rows`, `stale_anchor_rows`, and
`conflicting_anchor_rows`. These blockers do not invalidate sim-only beta
ranking or evaluator-bounded policy comparison, but they do block deployment
readiness, customer-specific SRCC/Pearson claims, and public external-accuracy
claims.

Some older artifact names and schema fields, including
`robot_team_grade_eval_closure_manifest.json`, remain for compatibility with
existing Pipeline/WebApp readers. Treat those names as legacy closure-package
surfaces, not as the current proof target. The current claim boundary is the
evaluator-bounded policy comparison recorded by `policy_ranking_scorecard.json`
and `wam_eval_claim_boundary.json`.

Forward/inverse consistency is a reliability signal for generated episodes. It
can help decide whether to stop or distrust an evaluator rollout, but it is not
itself a task-success label, deployment approval, or real-world outcome.

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
physical sensor feedback, success scoring, or deployment readiness.

Scene WAM episode packets can also prepare capture-derived robot POV seed
frames through the depth-splat synthesis path. For each task and robot profile,
the packet writes source QA, a coverage/quality report, a contact sheet, and
recapture guidance when no candidate clears the quality gate. A passing
depth-splat candidate can seed the initial WAM observation, but it remains a
synthetic/model-derived support artifact. It is not raw capture truth, physical
robot POV evidence, deployment approval, or safety validation.

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
does not become real-world failure proof; it downgrades
`ranking_confidence.confidence_level` and preserves the real-world calibration
boundary.

`candidate_selection_report.json` is the product handoff for the near-term
question: which policy performed best in this evaluator, and what broke. It
records the top policy only when the ranking is decisive; otherwise it reports
a candidate shortlist. It also carries the runner-up, predicted success-rate
margin, tie or ambiguity status, scenario matrix coverage, decisive scenarios
where candidates diverged, high-uncertainty scenarios, OOD blockers, dominant
failure modes, exemplar evidence refs, failure clusters, and real-world
validation follow-up requests.

Failure clusters are post-training data package hooks, not root-cause
certification. Each cluster should say what data to collect, which scenario
variants to add, and which policy adapter or checkpoint to retry. When failure
evidence is weak, the report must use `unknown_needs_review` instead of
inventing a root cause.

The report's boundary statement is explicit: do not use for deployment
approval.

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
capture evidence, real robot rollouts, deployment approval, safety approval, or
public-readiness proof.

SC3-Eval-style and OSCAR/Cosmos-style results are credible research signals, but
they do not prove high SRCC for arbitrary customer hardware, policies, sites, or
task families. A customer-specific SRCC or Pearson claim requires paired real
world validation rollouts with exact `scenario_eval_run_id` joins, policy or
checkpoint IDs, and owner evidence or operator attestation.

Passing WAM heldout evaluation can support policy ranking, failure discovery,
and a real-world validation request. It cannot approve deployment by itself.
