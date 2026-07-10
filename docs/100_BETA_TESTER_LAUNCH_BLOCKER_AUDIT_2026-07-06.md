# 100 Beta Tester Launch Blocker Audit - 2026-07-06

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

## Verdict

Do not launch this service to 100 external beta testers yet.

The current repo is locally testable, and the default Pipeline test lane is green, but the launch path is not operationally ready. The blockers are not one thing. They cluster into six hard areas:

1. The cross-repo paid beta contract gate is currently red.
2. Production WebApp to Pipeline forwarding and Pipeline intake are not configured/proven in the current environment.
3. The previously documented sim-only beta gate artifacts are not present in the current `output/` tree, so old evidence cannot be used as current proof.
4. Current real-policy-family artifacts complete scenario matrices, but closure still blocks on task metrics, full traces, delivery, WebApp lineage, policy-comparison confidence, live simulator execution, and live policy execution.
5. Live/operator evidence for real devices, Stripe money movement, payout readiness, finance ownership, buyer artifact access, legal/privacy ops, and KYC/background-check decisions is still missing.
6. Repo-wide quality gates are not all green: `pytest` passed, but broad `ruff` is red and the Pipeline/Capture worktrees are dirty.

This audit uses the current worktree and artifacts as of July 6, 2026. It does not treat the July 3 or July 4 remediation documents as current proof unless the current commands/artifacts confirm them.

## Evidence Collected

- `git status --short --branch`
  - Pipeline: dirty working tree on `main...origin/main`.
  - WebApp: clean on `main...origin/main`.
  - Capture: dirty working tree on `main...origin/main`.
- `git diff --check`: passed.
- `python -m pytest tests/test_lerobot_policy_family.py tests/test_lerobot_torch_policy_adapter.py tests/test_real_policy_family_eval_harness.py tests/test_post_training_data_package.py tests/test_buyer_package_readout.py -q`
  - `66 passed, 1 deselected`.
- `python -m pytest -q`
  - `2387 passed, 2 skipped, 1412 deselected, 9 warnings in 79.15s`.
- `python -m ruff check src/blueprint_pipeline scripts tests`
  - failed with 31 `E402` import-order errors in slow/integration test files.
- `python scripts/run_paid_marketplace_launch_gate.py`
  - `overall_status=automation_failed`.
  - Output: `output/paid_marketplace_launch_gate.json` and `output/paid_marketplace_launch_gate.md`.
- WebApp forwarding preflight artifact:
  - `$HOME/workspace/Blueprint-WebApp/output/pipeline/robot_eval_job_requests/forwarding_preflight.json`
  - `status=blocked`.
  - blockers: `missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL`, `missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`.
- `python -m blueprint_pipeline.live_pipeline_setup --no-load-env-files --output-path output/launch_audit_live_pipeline_setup_20260706.json`
  - `status=local_ready_live_external_blocked`.
  - 15 blockers.
- `python -m blueprint_pipeline.unitree_groot_sonic_provider_readiness --output-path output/launch_audit_unitree_groot_sonic_provider_readiness_20260706.json`
  - `status=ready_for_paid_provider_canaries`.
  - Important boundary: this is no-spend readiness only; next action is paid provider startup canaries.
- `python -m blueprint_pipeline.wam_real_provider_validation_probe run --output-dir output/launch_audit_wam_real_provider_probe_20260706`
  - `status=blocked`.
- Current real-policy artifacts inspected:
  - `output/real_policy_family_validation/scenes/demo-scene-1/captures/demo-capture-1/...`
  - `output/real_policy_family_validation_gpu/act-gpu-20260706T030741Z/job-real-policy-act-aloha-002/...`

## P0 Hard Blockers

These block a truthful 100-tester external beta.

1. **Paid beta automation is red.**
   - Evidence: `output/paid_marketplace_launch_gate.json` has `overall_status=automation_failed`.
   - The failing blocking check is `webapp_request_sync_contracts`.
   - Failure: `server/tests/stripe-native-parity.test.ts` expected bank-account disconnect endpoint status `200`, received `403`.
   - Impact: the paid marketplace gate cannot certify request/publication/sync/native Stripe parity.

2. **Current paid gate JSON loses source summaries and claims when automation fails.**
   - Evidence: `source_summaries=null`, `claims=null` in `output/paid_marketplace_launch_gate.json`.
   - Impact: the machine-readable closeout is weaker exactly when failure detail matters most.

3. **The paid gate markdown readout is internally misleading.**
   - Evidence: `output/paid_marketplace_launch_gate.md` says "Automated repository contracts passed" even though the overall status is `automation_failed` and one blocking WebApp check failed.
   - Impact: an operator could read the closeout as safer than the actual gate result.

4. **Production WebApp to Pipeline forwarding is not configured in the current environment.**
   - Evidence: WebApp `forwarding_preflight.json` has blockers `missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL` and `missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`.
   - Impact: WebApp can not prove it can forward robot-eval requests into Pipeline production intake.

5. **The forwarding probe did not run.**
   - Evidence: `forwarding_preflight.json` has `probe.requested=true`, `probe.attempted=false`, `probe.status=skipped`.
   - Impact: there is no current endpoint reachability or intake-staging proof.

6. **No current sim-only beta local gate artifact exists in `output/`.**
   - Evidence: `find output -type f -name 'sim_only_beta_local_gate_report.json'` returned no files.
   - Impact: previously documented sim-only gate results cannot be used as current launch evidence.

7. **No current sim-only beta release gate artifact exists in `output/`.**
   - Evidence: `find output -type f -name 'sim_only_beta_release_gate_report.json'` returned no files.
   - Impact: there is no current `ready_for_beta_release=true` or gate-blocker report for the active tree.

8. **No current sim-only production deployment proof artifact exists in `output/`.**
   - Evidence: `find output -type f -name 'sim_only_beta_production_deployment_proof.json'` returned no files.
   - Impact: production deployment parity, health, and intake readiness are unproven.

9. **Current real-policy sim-only core is not complete.**
   - Evidence: current `robot_team_grade_eval_closure_manifest.json` has `sim_only_beta_core_complete=false`.
   - Blocked required ids: `task_success_metrics`, `full_trace_package`, `closure_audit`.
   - Impact: even for sim-only policy comparison, the current run is not closeout-complete.

10. **Task success metrics are incomplete.**
    - Evidence: blockers include `missing_metric_clearance`, `missing_metric_path_deviation`, `batch_metrics_artifact_missing`.
    - Impact: beta users cannot receive a defensible task-success report.

11. **Full trace package is incomplete.**
    - Evidence: blockers include missing artifact checksums, batch trace package manifest, contact stream, control stream, metrics, planner state stream, and visual media coverage.
    - Impact: no replayable audit trail for customer-visible evaluation results.

12. **Closure audit is incomplete.**
    - Evidence: blockers include `task_metric_closure_incomplete` and `full_trace_package_incomplete`.
    - Impact: the system itself refuses to close the current run.

13. **Current policy comparison is inconclusive.**
    - Evidence: `policy_ranking_scorecard.json` has `status=blocked_inconclusive_ranking`.
    - CPU/demo run blocker: `policy_comparison_requires_at_least_two_candidates`.
    - GPU/ACT run blocker: `policy_comparison_policy_coverage_not_symmetric`.
    - Impact: the service cannot honestly tell a robot team which policy is better from these artifacts.

14. **No single best policy is claimed by current scorecards.**
    - Evidence: `top_policy_id=null`, `single_best_policy_claimed=false`.
    - Impact: the core policy-comparison value prop is not proven for the current evidence set.

15. **Policy behavior distinctness is not proven.**
    - Evidence: ranking confidence has `policy_behavior_distinctness_proven=false`.
    - Impact: equal or indistinguishable policy behavior makes ranking unreliable.

16. **Visual gate is not passed for policy ranking.**
    - Evidence: ranking confidence has `visual_gate_passed=false`.
    - Impact: generated/review media cannot support review-grade ranking claims.

17. **Current live eval closure is blocked.**
    - Evidence: `live_eval_closure_manifest.json` has `status=blocked`.
    - Impact: job-level closure does not support live external readiness.

18. **Task definitions are not complete enough for live closure.**
    - Evidence: live closure blocker `task_definitions_missing_standard_required_metrics`.
    - Impact: evaluation semantics are underspecified for external users.

19. **Scenario family library is missing/incomplete.**
    - Evidence: blockers `missing_scenario_family_library`, `scenario_family_library_missing_task_coverage`.
    - Impact: scenario coverage is not documented as a durable, reusable beta contract.

20. **WebApp upstream truth is missing from current real-policy artifacts.**
    - Evidence: blockers `missing_webapp_site_submission_id`, `missing_webapp_request_id`, `missing_webapp_buyer_request_id`, `missing_webapp_capture_job_id`.
    - Impact: current artifacts are not tied to the buyer/request/job lineage required for launch.

21. **Review acceptance evidence is missing.**
    - Evidence: blocker `review_acceptance_evidence_missing`.
    - Impact: no accepted human/operator review backs the package.

22. **Signed delivery access is missing.**
    - Evidence: blockers `signed_delivery_evidence_missing`, `signed_delivery_access_not_proven`.
    - Impact: buyers cannot be promised durable authenticated access from current artifacts.

23. **Live simulator execution is not proven in current closure.**
    - Evidence: blocker `live_simulator_execution:mujoco_g1_execution_not_proven`.
    - Impact: current local/support artifacts cannot be marketed as live simulator execution proof.

24. **Live policy execution is not proven in current closure.**
    - Evidence: blockers include `live_policy_execution_not_proven`, missing/non-completed policy execution traces, empty traces, and missing scenario-run coverage.
    - Impact: no launch-grade policy execution trail exists.

25. **Post-Training Data Package delivery handoff is not ready for current real-policy artifacts.**
    - Evidence: `package_delivery_handoff` blockers include `post_training_data_package_export_not_ready`, missing WebApp upstream ids, missing review acceptance, and missing signed delivery access.
    - Impact: current artifacts are not buyer-deliverable packages.

26. **Live pipeline setup is blocked on external/live inputs.**
    - Evidence: `output/launch_audit_live_pipeline_setup_20260706.json` has `status=local_ready_live_external_blocked` and 15 blockers.
    - Impact: the always-on pipeline is not launch-ready without additional configuration and commands.

27. **Real arena/simulator execution is not configured for the live control plane.**
    - Evidence: blockers `missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION`, `missing_simulator_command_or_arena_results_dir`, `simulator_command_executable_not_found`.
    - Impact: the live control plane cannot execute or ingest real arena/simulator results.

28. **Rollout vision labeling is not configured.**
    - Evidence: blockers `missing_env_BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING`, `missing_vision_labeling_command`, `vision_labeling_command_executable_not_found`.
    - Impact: beta review labeling cannot be automated through the live control plane.

29. **Delivery upload is not configured.**
    - Evidence: blockers `missing_env_BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD`, `missing_delivery_command`, `delivery_command_executable_not_found`.
    - Impact: no live package delivery upload path is proven.

30. **Live agent/Codex operator lanes are not configured.**
    - Evidence: blockers include missing `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS`, missing `OPENAI_API_KEY`, missing `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS`, missing Codex/OAuth gates.
    - Impact: if launch operations depend on those operators, the repo currently blocks them.

31. **Live control-plane WebApp upstream truth was not checked because no capture root was provided.**
    - Evidence: blocker `webapp_upstream_truth:capture_root_not_provided`.
    - Impact: launch audit lacks real request/job lineage evidence.

32. **WAM real-provider validation probe is blocked.**
    - Evidence: `output/launch_audit_wam_real_provider_probe_20260706/wam_real_provider_validation_proof_manifest.json` has `status=blocked`.
    - Impact: WAM/perception harness cannot claim real SAM3/depth/pose provider proof.

33. **SAM3 provider is not configured.**
    - Evidence: blocker `sam3_weights_path_missing`; SAM3 module unavailable; no model ref configured.
    - Impact: segmentation evidence is not backed by a real configured provider.

34. **Depth provider is not configured.**
    - Evidence: blocker `depth_provider_command_not_configured`.
    - Impact: inferred depth cannot be used as validated provider evidence.

35. **Pose provider is not configured.**
    - Evidence: blockers `pose_provider_command_not_configured`, `pose_model_path_not_configured`.
    - Impact: pose-conditioned observation proof is missing.

36. **No real SAM3/depth/pose provider ran.**
    - Evidence: blocker `no_real_sam3_depth_or_pose_provider_ran`.
    - Impact: harness outputs remain derived support artifacts, not provider validation.

37. **No target prompt was supplied to the WAM real-provider probe.**
    - Evidence: blocker `target_prompt_not_supplied`.
    - Impact: even a configured model would lack task/object-specific validation intent for this probe.

38. **Provider readiness has not crossed from no-spend readiness into paid canary proof.**
    - Evidence: `unitree_groot_sonic_provider_readiness` says `ready_for_paid_provider_canaries`, with next required action `run_paid_provider_startup_canaries_before_task_episode`.
    - Impact: credentials/image readiness is not startup, inference, task-episode, or semantic success proof.

39. **Object-store staging was not provided for the provider readiness audit.**
    - Evidence: `object_store_staging.status=not_provided`, `ready=false`.
    - Impact: provider output/input movement is not proven for the audited run.

40. **Remote build packet was not provided.**
    - Evidence: `remote_build_packet.status=not_provided`, `ready=false`.
    - Impact: remote build/reproducibility evidence is incomplete.

41. **Live provider task success is not proven.**
    - Evidence: provider readiness has `semantic_task_success_pass_proven=false` and `non_white_matte_g1_kitchen_task_success_proven=false`.
    - Impact: no launch claim should imply semantic task success.

42. **Real-device capture flows are not proven.**
    - Evidence: paid gate manual checks require `iphone_real_device_claim_flow`, `glasses_real_device_claim_flow`, and `android_real_device_claim_flow`.
    - Impact: 100 beta testers need actual device discovery, reservation, upload, and `capture_job_id` continuity.

43. **Live buyer payment settlement is not proven.**
    - Evidence: paid gate manual check `buyer_payment_settlement`.
    - Impact: paid beta cannot claim live marketplace payment readiness.

44. **Live capturer payout settlement is not proven.**
    - Evidence: paid gate manual check `capturer_payout_settlement`.
    - Impact: capturer compensation cannot be promised.

45. **Stripe connected account live readiness is not proven.**
    - Evidence: paid gate manual check `stripe_connected_account_live_readiness`.
    - Impact: payout onboarding/eligibility remains unverified.

46. **Payout exception monitoring is not proven.**
    - Evidence: paid gate manual check `payout_exception_monitor_live`.
    - Impact: failed/canceled payouts and overdue finance reviews may go unmanaged.

47. **Identity/KYC provider decision is missing.**
    - Evidence: paid gate manual check `identity_kyc_provider_decision`.
    - Impact: launch cannot safely claim identity/KYC readiness.

48. **Background-check provider decision is missing.**
    - Evidence: paid gate manual check `background_check_provider_decision`.
    - Impact: any screening claim would be unsupported.

49. **Human finance review owner is missing.**
    - Evidence: paid gate manual check `human_finance_review_owner`.
    - Impact: live payout exception handling lacks accountable ownership.

50. **Authenticated buyer artifact access after purchase is not proven.**
    - Evidence: paid gate manual check `buyer_artifact_access`.
    - Impact: payment fulfillment is not enough; the buyer must prove access to purchased artifacts.

51. **Android contract evidence is missing in this shell.**
    - Evidence: paid gate skipped `android_bundle_contracts` as `manual_required` because `ANDROID_HOME` / `ANDROID_SDK_ROOT` is not configured.
    - Impact: Android remains unverified by this automation run.

52. **Pipeline worktree is dirty.**
    - Evidence: modified files include `README.md`, `pyproject.toml`, `buyer_package_readout.py`, `post_training_data_package.py`, `real_policy_family_eval_harness.py`, LeRobot files, and tests; untracked Scaniverse import files exist.
    - Impact: the current state is not clean, published, or parity-proven for deployment.

53. **Capture worktree is dirty.**
    - Evidence: modified files include `ScanJob.swift`, `CaptureUploadMetadata.swift`, `JobDetailSheet.swift`, tests, README, and `CAPTURE_RAW_CONTRACT_V3.md`.
    - Impact: cross-repo launch evidence cannot be treated as a clean release snapshot.

54. **Repo-wide Ruff is red.**
    - Evidence: 31 `E402` errors in slow/integration tests after `pytest.importorskip("PIL")`.
    - Impact: CI/release quality is not clean if broad Ruff is required.

55. **Default pytest does not run all tests.**
    - Evidence: `2387 passed, 2 skipped, 1412 deselected`.
    - Impact: the default green suite is not the full integration/GPU/slow validation lane.

56. **Slow/integration/GPU lanes are not proven by this audit.**
    - Evidence: 1412 deselected tests were not run by `python -m pytest -q`.
    - Impact: launch claims that depend on slow/provider/runtime paths need separate evidence.

57. **Current real-policy artifacts are demo-local, not real beta-customer artifacts.**
    - Evidence: paths are under `output/real_policy_family_validation/scenes/demo-scene-1/captures/demo-capture-1` and GPU artifact capture root includes `/workspace/eval/...`.
    - Impact: they prove local/demo contracts, not 100-user production traffic.

58. **Current scenario matrices do not carry obvious policy/task ids in the inspected top-level fields.**
    - Evidence: inspected `scenario_eval_matrix.json` values for `policy_family_id`, `policy_id`, and `task_id` were `null`.
    - Impact: lineage may be present elsewhere, but the summary surface is too weak for buyer/operator triage.

59. **Digital-twin fidelity QA is missing for current real-policy closure.**
    - Evidence: blockers `digital_twin_fidelity_qa_artifact_missing`, `digital_twin_fidelity_qa_not_passed`.
    - Impact: not P0 for sim-only if you avoid fidelity claims, but blocks robot-team-grade claim upgrades.

60. **Remote/cloud execution path is not proven in current closure.**
    - Evidence: blocker `remote_prelaunch_spend_guard_not_passed`.
    - Impact: remote provider execution cannot be sold as proven.

61. **Real-world calibration is not available.**
    - Evidence: `sim_vs_real_calibration_path` is blocked with `sim_vs_real_calibration_not_required_for_sim_only_beta`.
    - Impact: not a sim-only blocker, but blocks any real-world calibration or deployment-readiness claim.

62. **Generated/WAM media remains support evidence only.**
    - Evidence: WAM probe claim boundaries say generated frames are not capture truth, inferred depth is not sensor depth, masks are not physical truth, and public claim upgrade is false.
    - Impact: cannot use generated media as external rank-fidelity or task-success proof.

63. **Live Pipeline DigitalOcean control-plane inventory was not configured/read in the no-load-env audit.**
    - Evidence: `digitalocean_control_plane.status=not_configured`, `api_token_present=false`, `api_read_allowed=false`.
    - Impact: optional, but no live CPU control-plane inventory proof was collected in this audit.

64. **The audit produced fresh output artifacts that are not release evidence by themselves.**
    - Evidence: new files under `output/launch_audit_*` are local no-spend/run artifacts.
    - Impact: they are diagnostic evidence, not deployment parity or live provider proof.

## P1 High Blockers

These may not all block a narrow internal sim-only test, but they block a reliable 100-person external beta.

65. **No load/soak test evidence for 100 testers.**
    - Need proof for concurrent uploads, queue pressure, Pipeline intake backpressure, artifact generation time, storage volume, and retry behavior.

66. **No capacity/cost model for 100 testers.**
    - Need expected capture count, average media size, provider runs per capture, GPU canary budget, object-store budget, and payout/payment cashflow exposure.

67. **No production observability packet was verified.**
    - Need dashboards/alerts for uploads, intake failures, provider failures, package failures, buyer access failures, payout exceptions, and spend.

68. **No incident response runbook was verified for beta operations.**
    - Need owner, escalation path, rollback path, takedown path, and customer communication templates.

69. **No data deletion/takedown drill was run in this audit.**
    - The repo has consent/takedown code, but a beta launch needs a current live drill with evidence.

70. **No privacy/legal signoff artifact was verified in this audit.**
    - Paid gate requires legal/EHS and operator DPA/data-processing terms before truthful launch.

71. **No real buyer session was verified against production artifact delivery.**
    - Need auth, entitlement, artifact link, expiration, revocation, and audit logging proof.

72. **No capturer onboarding run was verified end to end.**
    - Need account creation, job claim, capture upload, submission state, payout onboarding, and support recovery.

73. **No failed-upload recovery drill was verified.**
    - 100 beta testers will hit network dropouts, background-upload restarts, partial files, and duplicate submission attempts.

74. **No fraud/abuse controls were verified.**
    - Need fake captures, duplicate captures, malicious uploads, fake payout accounts, and buyer scraping/IDOR regression evidence.

75. **No support/admin queue readiness was verified.**
    - Need triage views for stuck captures, blocked packages, refund/payment exceptions, rejected packages, and review acceptance.

76. **No beta terms/consent acceptance flow was verified as current live behavior.**
    - Repo artifacts enforce gates, but launch needs exact tester-facing acceptance evidence.

77. **No versioned release artifact was proven for the Pipeline dirty tree.**
    - Need commit, tag or deployment SHA, environment config, and rollback target.

78. **No current WebApp production health proof was collected.**
    - The deployment parity proof artifact is absent; the paid gate only ran tests.

79. **No current Pipeline production health proof was collected.**
    - Forwarding and intake health are blocked/missing.

80. **No current Capture production build/archive proof was collected in this audit.**
    - Capture bridge tests passed, but real app archive/device evidence is separate.

81. **No iOS real-device screen recording was attached.**
    - Required by paid gate for iPhone claim flow.

82. **No Android real-device screen recording was attached.**
    - Required by paid gate for Android claim flow.

83. **No glasses real-device screen recording was attached.**
    - Required by paid gate for glasses claim flow; glasses should remain internal-only until proven.

84. **No live Stripe webhook reconciliation proof was attached.**
    - Payment and payout tests do not prove live webhook behavior.

85. **No live refund/dispute/failed-payment handling proof was attached.**
    - Required for 100 external users if money is involved.

86. **No signed delivery revocation proof was attached.**
    - Required to ensure buyers lose access when consent/revocation/payment state changes.

87. **No model/provider credential rotation proof was attached.**
    - Provider readiness found credentials, but launch needs rotation/ownership and secret-management proof.

88. **No active spend guard snapshot was attached.**
    - Provider readiness says paid canaries are allowed; launch needs current live spend guard evidence before and after canaries.

89. **No provider teardown proof exists for this audit.**
    - Needed after paid canaries or provider episodes.

90. **No package SLA was verified.**
    - Need target time from upload to package, retry policy, and customer-facing status semantics.

91. **No beta tester cohort controls were verified.**
    - Need invite list, roles, throttles, geo/site scope, and kill switch.

92. **No public/marketing claim review was verified against current artifacts.**
    - Current artifacts block physical readiness, deployment approval, rank fidelity, and live success claims.

93. **No current cross-repo rules parity proof was rerun in this audit.**
    - The July remediation doc says this was fixed before, but current launch should rerun it against current WebApp/Capture repos.

94. **No current storage rules parity proof was rerun.**
    - The July remediation doc listed this as a follow-up.

95. **No current WebApp full test/typecheck evidence was collected beyond the paid gate subset.**
    - One paid-gate WebApp test is failing; full WebApp CI state is not established here.

96. **No current Capture full test/build evidence was collected beyond bridge contracts.**
    - Capture dirty changes need their own build/test pass.

## P2 Medium Blockers / Quality Risks

97. **Launch evidence is split across markdown, JSON, output folders, and memory.**
    - Need one canonical launch-readiness packet.

98. **Older audit docs conflict with current command results.**
    - `docs/beta-launch-audit-2026-07-03/REMEDIATION-STATUS.md` says most code is fixed, but current paid gate is red and current forwarding is blocked.

99. **The current output tree lacks the older first-GPU walkthrough local-gate artifacts referenced by June docs.**
    - Need regenerate or remove stale references before launch review.

100. **The live setup no-load-env audit intentionally did not load `.env` files.**
    - This is safe for audit, but production readiness must be proven with configured deployment envs without leaking secrets.

101. **Provider readiness output records secret file paths.**
    - Values are redacted, but release docs should clarify whether local secret paths are acceptable in artifacts.

102. **The broad Ruff failure appears intentional/test-pattern related, but it is still a red gate.**
    - Either configure Ruff for `pytest.importorskip` slow tests or change the import pattern.

103. **Default `pytest` warnings about module re-execution should be reviewed.**
    - Not launch-blocking by themselves, but they indicate entrypoint tests may be importing modules before `runpy` execution.

104. **Pipeline paid-gate subcommand ran under Python 3.13 while the full local pytest run used Python 3.12.**
    - Need a documented canonical CI interpreter matrix for launch claims.

105. **The report surfaces `manual_required` Android evidence as blocking and also calls Android internal-only contract-ready in prose.**
    - Tighten wording so operators do not confuse toolchain evidence with device proof.

106. **`output/` is large (about 21 GB) and contains old/current proof artifacts.**
    - Need retention, pruning, and canonical artifact selection before operator handoff.

107. **`robot_eval_jobs/` is about 2.9 GB.**
    - Need retention and cost/storage plan for 100 testers.

108. **No beta data retention policy was verified in artifacts.**
    - Required for privacy and support operations.

109. **No subprocessor/access-audit terms were verified.**
    - Required by paid gate operator evidence.

110. **No customer-facing degraded-state copy was verified for every blocker class.**
    - Buyers/testers need clear status when packages are review-required, blocked, or degraded.

## Not P0 For A Sim-Only Beta Unless You Claim It

These are important claim-upgrade lanes, but they should not be turned into required blockers for a narrow sim-only/evaluator-bounded beta unless the launch messaging includes those claims.

- Physical robot readiness.
- Safety validation.
- Deployment approval.
- Real-world outcome proof.
- Sim-vs-real calibration with accepted real-world anchors.
- Generated-world rank fidelity.
- Public Google/Meta glasses site-faithful launch.
- Full real robot POV/action/timestamp evidence.

Missing these means the launch must not claim them. It does not automatically block a scoped internal sim-only policy-comparison beta.

## What Is Healthy

- Pipeline default test lane passed: `2387 passed, 2 skipped, 1412 deselected`.
- Focused changed-surface tests passed: `66 passed, 1 deselected`.
- `git diff --check` passed.
- Pipeline paid-gate subcheck passed: `55 passed`.
- Capture bridge contract check passed: `50 tests passed`.
- WebApp creator payout contract passed.
- WebApp marketplace fulfillment checkout contract passed.
- Current real-policy scenario matrices are `completed` with semantic spawn target coverage complete and no deterministic fallback spawn target runs.
- Provider no-spend readiness found the sealed Unitree/GROOT/SONIC image and credentials sufficient to proceed to paid canaries.
- Current claim boundaries are mostly conservative: they refuse deployment approval, physical readiness, safety validation, rank fidelity, and public claim upgrades.

## Recommended Fix Order

1. Fix the failing WebApp Stripe-native parity test and rerun `python scripts/run_paid_marketplace_launch_gate.py`.
2. Configure/prove production forwarding and Pipeline intake: URL, token, probe, intake audit, deployed commits, and health.
3. Regenerate current sim-only beta local, release, and deployment proof artifacts or explicitly retire that launch lane.
4. Close the current real-policy sim-only core blockers: task success metrics, batch metrics artifact, full trace package, and closure audit.
5. Produce a valid policy-comparison run with at least two candidates, symmetric scenario coverage, visual gate pass or explicit review grade, and no inconclusive ranking blockers.
6. Attach WebApp upstream truth ids to the package/eval artifacts.
7. Produce review acceptance and signed delivery access evidence.
8. Run paid provider startup canaries with spend guard, object-store staging, artifact upload, and teardown evidence.
9. Configure real SAM3/depth/pose providers or remove those claims from the launch lane.
10. Complete paid beta operator evidence: real devices, live Stripe payment, live payout, Connect readiness, payout monitor, KYC/background decision, finance owner, buyer access.
11. Clean or intentionally scope Ruff so repo-wide lint is green.
12. Stabilize and publish the dirty Pipeline and Capture worktrees, then prove deployment parity from exact commits.
13. Run full cross-repo CI/build evidence for WebApp and Capture, not only paid-gate subsets.
14. Build a single launch-readiness packet that links every artifact, command, owner, and remaining manual evidence id.
