# Restart reuse repairs — 2026-09-13

ADP-009D dependency for the day-21 sealed/joined rehearsal result: retry retained stages without stale approvals, repeated image generation, lost workspace inputs, or invented budget refunds. The observed completion artifacts for this change are focused regression results and merged code. Deployment, GPU execution, and a measured restart speed-up are separate proofs.

The adjacent `*-before-fixes.md` files preserve the two original audit reports. Their failing-test and not-implemented statements describe those earlier snapshots. Both audits' regression contracts are now included in the maintained tests.

## Repaired contracts

- Validator entries bind executed code, imported data construction, and nested verdict dependencies. Nested persistent and in-operation cache hits propagate their code and file dependencies into enclosing entries. Unprovable concurrent computation is not persisted. Entries written before dependency version 2 are rejected once, then rebuilt under the corrected contract. Successful null results are distinguishable from misses.
- Imported-data dependency discovery follows initialization and the helper functions it invokes, rather than unrelated local imports in every function in a module. AST metadata is cached by file identity, and data closure is reused within a validation operation. The existing fewer-than-40-module regression remains enforced.
- Capsule extraction uses a fresh directory bound to the capsule digest. Automatically discovered missing/corrupt candidates are recorded as ineligible and skipped, allowing intact history to supply candidates. Explicitly supplied authoritative inputs remain strict. Every retained raw image still needs current review and locality processing.
- A terminal predecessor can recover through the existing authority/lineage checks. Terminal settlement does not erase unreconciled spend or executed-attempt counts. Proven never-queued dependent reservations can be released. Source or executed rows retain a conservative reservation until cost reconciliation exists; downstream holds of a blocked launch are released only when they could not become eligible; no historical receipts are rewritten. Status no longer calls terminal-settled execution an unstarted cancellation.
- Workspace GC rechecks pins, queued references, and active-process references at apply time. Semantic preparation holds a shared workspace lock; reclamation requires an exclusive lock. Lock ownership/group permit the production reaper and service user to cooperate. Legacy workspaces without a producer lock are retained, as are workspaces whose process inventory cannot be inspected. Outputs and receipts remain retained when an eligible bundle is removed.
- A canonical optional deduplication CLI uses Linux `FIDEDUPERANGE`, which compares extents in the kernel and preserves copy-on-write behavior with distinct inodes. Unsupported platforms/filesystems are skipped; there is no pathname replacement or hard-link fallback. The result reports bytes deduplicated, not a claim about physically freed bytes.

## Deployment and operations

These PRs do not deploy or restart services. Claude retains production ownership. The canonical `scripts/control_plane_hardlink_dedup.py` also delegates to the safe extent-sharing implementation, preserving its quiescence and filtering arguments. The old host-local `/root/dedup_apply.py` is not changed by merging this repository; stop invoking that hard-link script. The maintained alternative, on a release containing this change, is:

```bash
python -m blueprint_pipeline.control_plane_file_dedup --groups /root/dedup_groups.json
python -m blueprint_pipeline.control_plane_file_dedup --groups /root/dedup_groups.json --apply
```

The first command is a candidate-only dry run. The second attempts only kernel-verified extent sharing. Some filesystems cannot perform this operation and will retain the copies. The UAPI layout comes from Linux `include/uapi/linux/fs.h`; the exact kernel behavior must be observed on the deployed host before reporting reclaimed capacity.

Existing v1 cache entries need one conservative refresh. Do not promise a 10x speed-up or an exact 83-to-45-minute reduction from these fixes. Record deploy completion, factory completion, activation, dispatch, and actual GPU start on the next authorized attempt. Bundle materialization, reviews, and release admission still have work to perform; raw-candidate reuse is not complete stage resumption.

## Verification

The combined local focused set passed 137 tests in 18.19 seconds before the final legacy-workspace protection refinement. The final GC/workspace/pretraining set passed 29 tests. Cache/reuse checks passed 20 tests in 1.59 seconds after narrowing dependency collection. Changed-file lint passes. Hosted impacted checks cover each focused PR independently; no broad-lane exemption is added.

The retained tests are:

- `tests/test_speedup_audit_regressions.py`: stale imported/nested approvals, cache-hit dependency propagation, null verdict reuse, capsule collisions, damaged candidates, and safe deduplication responses.
- `tests/test_validation_code_dependencies.py`: derived initialization and unused local imports.
- `tests/test_overnight_fix_audit_regressions.py`: actual recovery and unreconciled cost/attempt accounting.
- `tests/test_workspace_reclamation_safety.py`: live pins, pins created after the dry run, and the shared producer lock.

The original fixes landed through #1906 and #1911. #1909 preserves those changes and adds the remaining cache, deduplication, and historical recovery-grant protections. #1913 layers synchronized workspace protection over #1911 and retains both historical audit reports. Duplicate accounting work in #1910 was superseded rather than replacing the merged implementation.

The current-main integration checks for the cache/controls/training-source path passed 62 tests. Historical recovery-grant compatibility passed 8 tests, retaining the default ordinary-retry treatment for grants issued before the budget field existed. The workspace follow-up passed 36 integrated GC/pretraining/overnight-audit tests before the final queue/legacy-protocol cases. The full-suite diagnostics on #1911 also exposed this legacy-grant issue (fixed by #1909) and an unchanged `vast_provider_adapter.py` line-budget violation; no full-suite-green claim is made.
