> Historical audit snapshot before implementation. See the regression tests and the restart-reuse implementation note for the repaired behavior.

# Overnight production fixes: independent audit

Audited release: `ad019946ac60273a94bcd21877e24540815fea88` (PRs #1896–#1902). The production active link still selected this commit at **2026-09-13 12:38:22 UTC**. Audit checkout is detached and isolated from the primary checkout and production.

**Result: three confirmed remaining defects.** Existing focused tests passed (119); three additional offline regressions reproduce failures. These demonstrate reachable defects, not proof that each defect has already affected the current production attempt. No GPU, model call, deployment, merge, restart, or production mutation was performed by this audit.

## Findings

### 1. P1 — Settlement prevents automatic recovery of its own predecessor

Location: `src/blueprint_pipeline/task_evaluation_scene_progression.py:393–400`, with `task_evaluation_scene_intake.py:262`.

The actual `_recover` path settles the failed source row before reserving its successor. Settlement becomes a validated cancellation. Intake then rejects recovery from that same source with `scene_intake_recovery_prior_attempt_cancelled`. This occurs despite valid retained failure and ownership reconciliation evidence. It can strand a hands-off retry until an external intervention changes the path.

Reproduction: `test_settling_retired_source_must_not_disable_its_authorized_recovery` drives the real recovery, settlement, and reservation functions, substituting only captured external observations from the existing recovery fixture. The expected successful recovery instead raises the cancellation error.

Repair requirement: distinguish a terminal, settled predecessor from a never-started cancellation and admit its authorized recovery while retaining lineage, ownership, retry-budget, and idempotence checks. Merely moving settlement after reservation is insufficient without considering budget admission and crash recovery.

### 2. P1 — Terminal settlement releases the entire reservation without reconciling incurred cost

Location: `src/blueprint_pipeline/task_evaluation_terminal_scene_attempt_settlement.py:114–119`; consumer `task_evaluation_scene_intake.py:303–323`.

Dependent launch status `completed`, `failed`, or `blocked` is accepted as settlement evidence without actual-cost reconciliation or proof of no paid execution. Intake then excludes the entire settled row from cumulative exposure and the ordinary paid-attempt count. A stopped resource is not proof of zero cost; a blocked launch may already have paid for image edits.

Reproduction: `test_paid_terminal_launch_must_not_be_refunded_without_cost_reconciliation` uses a $26 intent, existing source/dependent reservations totaling $24.72, and a completed launch with no billing reconciliation. Settlement allows another full $26 reservation. No actual paid work is performed in the test; it demonstrates that missing reconciliation does not prevent full budget reuse.

Repair requirement: retain reconciled incurred spend (or a conservative unreconciled debit), release only the unspent balance, and preserve executed-attempt counts. Proven never-started reservations may be released in full. This finding concerns the per-intent accounting path; it does not establish that separate fleet-level spend guards are absent or ineffective.

### 3. P1 — Workspace bundle cleanup can delete a live pinned bundle

Location: `src/blueprint_pipeline/control_plane_storage_gc.py:874–877`, and the workspace-manifest integration in `run_storage_gc`.

The new semantic-pretraining workspace cleanup decides that a workspace is idle from modification time. It does not use the storage pins and active-reference protection available to other cleanup branches. Reading a bundle does not update its modification time.

Reproduction: `test_workspace_gc_must_honor_a_live_pin_even_when_files_are_old` creates a tiny local bundle whose files are seven hours old, writes and verifies a fresh real activation storage pin, and invokes the actual GC entry point. GC removes the bundle despite that live pin. Only temporary test files were deleted.

Repair requirement: apply shared pin, active queue/reference, and lease protection when building and applying the deletion manifest, with synchronization or rechecking sufficient to protect concurrent use. Age alone cannot establish that a bundle has no live consumer.

## Coverage of the six reported fixes

| Reported fix | Audit assessment |
|---|---|
| Image-edit cost projection (#1896) | No additional defect reproduced. Default projection follows the sealed registry per-request maximum; explicit estimates and downstream budget checks remain. This does not independently certify actual provider billing. |
| View coverage (#1897) | No additional defect reproduced. This deliberately relaxes angular coverage from 30 to 45 degrees; minimum approved-view and neighboring-view requirements remain. It changes acceptance policy and is not proof of appearance quality. |
| Dead-attempt reservation accounting (#1898) | Two confirmed defects: automatic recovery and unreconciled spend release, above. |
| Multiple canceled controls launches (#1899) | Per-launch grouping fixes the reported global-grouping problem; strict source evidence remains. No separate defect reproduced beyond its interaction with settlement. |
| Runtime bundle cleanup and systemd permissions (#1900–#1901) | Permission path is present, but cleanup does not honor live pins, above. |
| Partial teacher views in background initialization (#1902) | No additional defect reproduced. Rejected teacher images are excluded, selection metadata survives the handoff, and the downstream consumer validates the selection. GPU training and resulting image quality remain unproven by these CPU tests. |

## Verification

Existing focused contracts: **119 passed in 14.09 seconds**.

```bash
python -m pytest tests/test_task_evaluation_scene_configuration_artifixer_driver.py tests/test_semantic_target_training_selection.py tests/test_artifixer_background_initialization.py tests/test_terminal_scene_attempt_settlement.py tests/test_task_evaluation_scene_recovery.py tests/test_task_evaluation_controls_terminal_adoption.py tests/test_control_plane_storage_gc.py tests/test_task_evaluation_scene_configuration_activation_automation.py -q -p no:cacheprovider --tb=short --junitxml=audit_results/existing-tests.xml
```

This covers the changed image-cost, selection, background initialization, settlement, recovery, controls adoption, cleanup, and activation surfaces.

New audit contracts: **3 failed in 0.56 seconds**, each demonstrating one finding above.

```bash
python -m pytest tests/test_overnight_fix_audit_regressions.py -q -p no:cacheprovider --tb=short --junitxml=audit_results/audit-regressions.xml
ruff check tests/test_overnight_fix_audit_regressions.py
```

Changed-file lint passed. Logs and JUnit results are retained alongside this report. The regression tests intentionally remain red against the deployed baseline; this audit has not implemented the repairs.

## Deployment and replay conclusions

Main advanced during the audit through #1903, #1904, and #1905 to `ff1bd7a3262027aa11a795795bcb5ae84cd63ee1`, while the production active link remained at `ad019946`. The inspected deployment path checks out an explicitly supplied commit. Thus moving main did not itself change this running release. An explicit deployment can still cause release transition and revalidation. None was initiated here.

The newer retained-edit and verdict-cache changes are outside the six-fix audit and have **not** received equivalent regression review. Their diff does not modify the three defective paths identified above. They must not be described as audited clean on the strength of this report.

The statement that replay necessarily costs another $1.90 in edits is too broad. Selection and background-seeding consumers can be exercised offline using retained images and review receipts, with external edges stubbed. The existing stage replay refuses paid phases by default, and its isolated path disables networking. However, the existing parent replay does not rehearse the entire ArtiFixer training path, and invoking the full driver without the correct retained checkpoints or stubs may still request edits. The appropriate repair loop is a retained-input consumer rehearsal, not another paid end-to-end attempt merely to check these CPU contracts.

Passing the existing tests did not establish clean hands-off recovery or budget/cleanup safety. The three missing cross-module contracts above are the immediate repair scope; broad architectural work is not required to demonstrate them.
