# Policy-ranking research code (archived 2026-11-10)

The policy-ranking thesis experiments ran from 2026-07-26 to 2026-07-30
(RoboArena phases A and B, the powered DROID confirmation, Experiment 2, the
Cosmos causal screen). Their analysis and bundle-building code had no caller
left: no systemd unit, operator door script, deploy tool, workflow, console
script, GPU bundle list or other module imported it. It was removed from
`src/blueprint_pipeline/` under plan P-A02 of
[`../audits/2026-11-simplification-audit.md`](../audits/2026-11-simplification-audit.md).

## How to get the code back

- **Last commit that has every file below:**
  `4f2c74c84c5a68076ff1b71ae43a08cbadffdc5e` (main, 2026-11-09).
  `git show 4f2c74c84:src/blueprint_pipeline/<module>.py` prints one file;
  `git checkout 4f2c74c84 -- <paths>` restores them.
- **Commit at which the Experiment 2 manifest verifies:**
  `ba8fe1fb0e6339a2d3d40ca07dc960f94680b54d` (2026-08-16).
  `docs/experiments/policy_ranking_thesis_experiment_2_20260727/experiment_artifact_manifest.json`
  binds the sha256 of 13 source and test files. All 13 match at that commit.
  On main before this archive, 10 matched: `.gitignore`,
  `policy_ranking_evidence.py` and `tests/test_policy_ranking_evidence.py`
  had changed since (first at `c111f5805`). To reproduce the experiment's
  code exactly, check out `ba8fe1fb0`, not the commit above.

## What stays in the tree

- Every file under `docs/experiments/`. The evidence, protocols and verdicts
  are unchanged. `docs/experiments/policy_ranking_thesis_20260726/*.json` is
  read by every beta screening plan.
- The policy-ranking modules that production or a paid-call check still
  reaches, among them `policy_ranking_thesis`, `policy_ranking_evidence`,
  `policy_ranking_roboarena_calibration`, the `policy_ranking_successor_*`
  modules, and the four OpenAI judge and evaluator modules that
  `scripts/paid_api_call_sites.json` classifies.
- `judge_spend_governor` (a spend guard) and `roboworld_progress_judge`.

## Removed modules (22)

- `oscar_review_views.py`
- `policy_ranking_causal_conditioning.py`
- `policy_ranking_cosmos_causal_screen.py`
- `policy_ranking_disjoint_preflight.py`
- `policy_ranking_droid_kinematics.py`
- `policy_ranking_droid_reference.py`
- `policy_ranking_droid_reference_analysis.py`
- `policy_ranking_droid_reference_bundle.py`
- `policy_ranking_experiment_2.py`
- `policy_ranking_experiment_2_adjudication.py`
- `policy_ranking_label_free_chunk_selection.py`
- `policy_ranking_phase_b_cosmos_bundle.py`
- `policy_ranking_phase_b_high_motion_selection.py`
- `policy_ranking_phase_b_replay.py`
- `policy_ranking_power_analysis.py`
- `policy_ranking_powered_droid_analysis.py`
- `policy_ranking_powered_droid_bundle.py`
- `policy_ranking_roboarena_analysis.py`
- `policy_ranking_roboarena_collapse_audit.py`
- `policy_ranking_roboarena_crop_audit.py`
- `policy_ranking_task_instruction_extraction.py`
- `policy_ranking_wam_validity.py`

## Removed tests (20)

- `test_oscar_review_views.py`
- `test_policy_ranking_causal_conditioning.py`
- `test_policy_ranking_cosmos_causal_screen.py`
- `test_policy_ranking_disjoint_preflight.py`
- `test_policy_ranking_droid_kinematics.py`
- `test_policy_ranking_droid_reference.py`
- `test_policy_ranking_droid_reference_bundle.py`
- `test_policy_ranking_experiment_2.py`
- `test_policy_ranking_experiment_2_adjudication.py`
- `test_policy_ranking_label_free_chunk_selection.py`
- `test_policy_ranking_phase_b_high_motion_selection.py`
- `test_policy_ranking_phase_b_replay.py`
- `test_policy_ranking_power_analysis.py`
- `test_policy_ranking_powered_droid_analysis.py`
- `test_policy_ranking_powered_droid_bundle.py`
- `test_policy_ranking_roboarena_analysis.py`
- `test_policy_ranking_roboarena_collapse_audit.py`
- `test_policy_ranking_roboarena_crop_audit.py`
- `test_policy_ranking_task_instruction_extraction.py`
- `test_policy_ranking_wam_validity.py`
