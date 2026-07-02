# SC3-Eval Fidelity + Robot/Policy-Agnostic Service Plan

Date: 2026-07-02

Status: slice 1 implemented and tested on `claude/sc3-eval-provider-agnostic-ex9wax`;
slices 2-5 are planned follow-on work with explicit gates.

## Linked Goals

1. **SC3-Eval-style evaluator fidelity.** Target 90%+ Pearson/Spearman correlation
   between Blueprint generated-evaluator results and real/owner outcomes — but only
   claimable after paired Blueprint predictions and accepted real/owner anchors exist
   and the metrics are actually computed from that join.
2. **Robot/policy-agnostic service contract.** Teams bring their own robot embodiment
   and policy API/container/trace. Unitree G1 is a default/reference embodiment, not a
   customer requirement. Cosmos3/SC3-style WAM is a preferred evaluator recipe
   candidate, not a permanent company dependency.

## Source Facts (re-verified 2026-07-02)

SC3-Eval (arXiv:2606.18610, v3, submitted 2026-06-17, revised 2026-06-26) reports
closed-loop Pearson 0.929 and MMRV 0.119 across seven policy checkpoints in one
table-bussing scene with three camera views and at most 20-second rollouts. Those are
the paper's numbers on its own setup. They are not Blueprint numbers and must never be
quoted as Blueprint accuracy. `sc3_eval_protocol.SC3_SOURCE_FACTS` pins the verified
facts; re-verify from the arXiv abstract before bumping the pinned version.

## Slice 1 — Protocol + Contracts (implemented on this branch)

- `src/blueprint_pipeline/sc3_eval_protocol.py`: declarative
  `sc3_eval_protocol.json` builder. Required data (synchronized multi-view cameras,
  robot camera profile, action chunks, initial observations, generated rollout frames,
  policy re-query trace, success criteria, failure taxonomy, accepted anchor joins) and
  required metrics (Pearson, Spearman/SRCC, MMRV, calibration error,
  confidence/abstention) with fail-closed statuses:
  - no accepted anchors → `correlation_not_measured`, never a failed sim-only ranking
  - missing symmetric coverage → `blocked_inconclusive_ranking` /
    `completed_ambiguous_ranking`, never a fabricated winner
- `scene_placement/robot_profile.py`: `RobotEmbodimentPack` contract
  (`robot_embodiment_pack.v1`) — robot_id, embodiment_type, kinematics,
  action_interface, camera_rigs, observation_schema, simulator_asset_refs,
  controller_constraints, calibration_requirements, claim_boundaries. Non-G1 robots
  load from JSON via `robot_profile_from_json_file` with no downstream G1 hardcoding.
- Policy adapter packs: all six modalities (`policy_api_endpoint`, `docker_container`,
  `recorded_action_trace`, `high_level_skill_trace`, `teleop_demo`,
  `sim_controller_plugin`) plus provider-worker HTTP workers flow through the same
  observation/action contract (`robot_team_policy_adapter_pack_contract.v1`).
  Customer-supplied endpoints/containers are `launch_reviewable_without_execution`;
  execution proof still requires `policy_execution_manifest.json`.
- Wiring: `robot_eval_job_orchestrator` writes `sc3_eval_protocol.json` per job and
  threads its status into `evaluation_result.json`, `proof_boundary.json`, the run
  manifest, and the data-package export map. `live_eval_closure_manifest.json` carries
  a non-gating `sc3_eval_protocol` summary block.
- Docs: `docs/SC3_EVAL_PROTOCOL.md`, README artifact list,
  `docs/WAM_EPISODE_CONSISTENCY_SCORER.md`.
- SC3 self-consistency signals (forward/inverse dynamics, cross-view consistency,
  uncertainty-driven early termination) stay reliability/abstention support only.

## Slice 2 — Accepted Anchor Accumulation (planned)

- Owner/real outcome intake keyed on `scenario_eval_run_id`, `policy_id`, `task_id`,
  `scenario_variation_instance_id`, joined into
  `wam_prediction_outcome_correlation_ledger.json` rows with acceptance provenance.
- Gate: an anchor counts only when a human/owner acceptance record exists; generated
  media, WAM labels, or consistency scores can never mint anchors.
- Exit: `accepted_anchor_count` grows past a declared minimum-N (pre-register the
  threshold before computing headline metrics; SC3 used 36-37 matched initial
  conditions per checkpoint as its reference density).

## Slice 3 — Computed Correlation Metrics (planned)

- Compute Pearson/Spearman/SRCC/MMRV/calibration error from the paired join only, in
  `sim_vs_real_calibration_report.json`, with per-metric sample counts and confidence
  intervals.
- Gate: metrics stay `correlation_not_measured` below minimum-N; a computed value below
  0.9 must be reported as-is. 90%+ language is unlocked only by computed values, and
  any public copy in Blueprint-WebApp routes through `robotPolicyEvaluationClaims.ts`
  plus `npm run claims:guard` there.

## Slice 4 — Embodiment Pack Registry (planned)

- A packs directory (JSON) loaded at intake through `robot_profile_from_json_file` +
  `register_robot_profile`, plus conformance checks (camera rig count vs. protocol
  multi-view requirement, action interface schema refs, calibration requirements).
- Gate: a pack failing conformance yields a blocked, reviewable status — never a
  silent fallback to G1.

## Slice 5 — Policy Adapter Conformance Harness (planned)

- Dry-run review checks per modality (endpoint reachability contract, container
  digest/manifest audit, trace timestamp alignment) that upgrade
  `launch_reviewable_without_execution` review quality without ever claiming
  execution, plus gated execution smoke via existing
  `BLUEPRINT_ALLOW_POLICY_EXECUTION` controls.

## Hard Boundaries (all slices)

- No paid provider launch by default; existing env/CLI gates stay mandatory.
- Never print/persist raw API keys, signed URLs, tokens, or customer endpoint secrets.
- Raw capture, rights/privacy/provenance truth, and proof boundaries are never
  rewritten by evaluator artifacts.
- Self-consistency, generated rollouts, and protocol readiness never become task
  success, deployment approval, physical readiness, or safety proof.

## Verification

```bash
python -m pytest tests/test_sc3_eval_protocol.py tests/test_robot_profile.py \
  tests/test_robot_eval_job_orchestrator.py tests/test_robot_eval_execution_coverage_edges.py \
  tests/test_provider_worker_contract.py tests/test_provider_worker_policy_command_adapter.py \
  tests/test_oscar_cosmos_wam_evaluator.py tests/test_wam_fixture_evaluator.py \
  tests/test_robot_initial_observation.py tests/test_live_robot_eval_closure_coverage_edges.py -q
```
