# Arm Decision Proof v1 Current State

This table is the checkout audit required by Phase 0. Status vocabulary is
limited to `observed_complete`, `partial`, and `missing`. A locally passing
fixture does not promote a public-reference gate.

| Item | Status | Observed evidence |
|---|---|---|
| ADP-001 | `observed_complete` | [`north_star_contract.json`](north_star_contract.json), [`test_arm_decision_proof_focus.py`](../../tests/test_arm_decision_proof_focus.py) |
| ADP-002 | `partial` | [`simpler_google_robot_pick_coke_can.v1.json`](manifests/simpler_google_robot_pick_coke_can.v1.json), [`public_reference_admission.py`](../../src/blueprint_pipeline/public_reference_admission.py); exact runtime lock awaits the admitted Vast execution |
| ADP-003 | `partial` | [`arm_decision_proof.py`](../../src/blueprint_pipeline/arm_decision_proof.py), [`evaluation_run_contract.py`](../../src/blueprint_pipeline/evaluation_run_contract.py), [`test_arm_decision_proof.py`](../../tests/test_arm_decision_proof.py); real two-checkpoint execution awaits ADP-002 runtime completion |
| ADP-004 | `partial` | Digest-bound receipt/replay implementation and focused tests exist; real episode traces await the admitted execution |
| ADP-005 | `partial` | Decision seal, separate outcome loader, release receipt, mismatch rejection, and firebreak tests exist; integrated public run awaits execution |
| ADP-006 | `partial` | Frozen deterministic rule exposes uncertainty, invalid region, coverage, abstention, and next measurement; integrated result awaits execution |
| ADP-007 | `partial` | Per-cell matrix links source/runtime/reset/trace/metric/outcome/qualification digests; integrated result awaits execution |
| ADP-008 | `missing` | One-command reconstruction is implemented and fail-closed, but no admitted immutable execution package has completed yet |

All entries are `retrospective_external_reference` and `development_only`.
No capture or reconstruction feature was added.
