# SC3-Style Evaluation Protocol

Blueprint treats SC3-Eval as an evaluator recipe candidate, not a permanent
company dependency and not a public Blueprint accuracy claim.

The job-level `sc3_eval_protocol.json` artifact defines the data needed for an
SC3-compatible evaluator pass:

- synchronized multi-view robot cameras
- robot camera profile and calibration readiness
- action chunks and policy re-query trace
- initial observations and generated rollout frames
- explicit success criteria and failure taxonomy
- accepted real/owner anchor joins
- Pearson, Spearman/SRCC, MMRV, calibration error, confidence, and abstention
  metrics

If no matched accepted anchors exist, correlation status is
`correlation_not_measured`. Missing symmetric policy/scenario coverage stays
`blocked_inconclusive_ranking` or `completed_ambiguous_ranking`; Blueprint must
not fabricate a winner from incomplete coverage.

Forward/inverse dynamics consistency, cross-view consistency, and uncertainty
early termination are reliability and abstention support only. They do not
become task-success labels, policy success, deployment approval, physical robot
readiness, safety validation, real-world validation, or a 90%+ Blueprint
correlation claim.

The service contract is robot and policy agnostic. Teams may bring a
data-driven Robot Embodiment Pack plus one of the supported policy adapter
packs:

- `policy_api_endpoint`
- `docker_container`
- `recorded_action_trace`
- `high_level_skill_trace`
- `teleop_demo`
- `sim_controller_plugin`
- provider-worker HTTP workers through the same observation/action contract

Unitree G1 remains a default/reference embodiment for local smoke and historical
lanes. It is not required for customer robots. Cosmos3/SC3-style WAM remains a
preferred evaluator recipe candidate when configured and gated; it is not a
hardwired backend or universal grading proof.

`live_eval_closure_manifest.json` carries a non-gating `sc3_eval_protocol`
summary block; protocol readiness never becomes a closure gate. The phased
implementation plan (accepted-anchor accumulation, computed correlation
metrics, embodiment pack registry, policy adapter conformance) lives in
[`docs/goals/2026-07-02-sc3-eval-robot-policy-agnostic-service-plan.md`](goals/2026-07-02-sc3-eval-robot-policy-agnostic-service-plan.md).
