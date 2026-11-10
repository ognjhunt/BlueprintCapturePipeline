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
- The policy-ranking modules that production still reaches, among them
  `policy_ranking_thesis`, `policy_ranking_roboarena_calibration` and the
  `policy_ranking_successor_*` modules.
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

## Second removal (2026-11-10, later the same day)

The owner approved removing the four OpenAI judge and evaluator modules and
the evidence store only they used. Each could spend OpenAI money behind an
environment switch alone, outside the paid-resource admission;
`scripts/paid_api_call_sites.json` listed five of their call sites as
`unadmitted`. Those entries left the record and the verifier's list of known
exceptions with the code.

- **Last commit that has these files:**
  `4835a6df1` (main, 2026-11-10).
- Modules: `policy_ranking_evaluator_diagnostic_openai.py`,
  `policy_ranking_evaluator_diagnostic_openai_batch.py`,
  `policy_ranking_roboarena_evaluator_openai.py`,
  `policy_ranking_thesis_judge_openai.py`, `policy_ranking_evidence.py`.
- Tests: the `tests/test_<module>.py` file of each.
- Three of them (`policy_ranking_thesis_judge_openai.py`,
  `policy_ranking_evidence.py` and their tests) are bound by the Experiment 2
  manifest; use `ba8fe1fb0` to reproduce that experiment, as above.
