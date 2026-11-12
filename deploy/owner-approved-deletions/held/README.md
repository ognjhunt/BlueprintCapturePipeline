Manifests here are never planned or applied: the units read only the `*.json` directly in
`deploy/owner-approved-deletions/`. Their approvals stand and their receipts on the host are untouched.
A manifest is moved here once it has been applied, and is brought back only deliberately.

- `2026-11-04-authority-ended-scenes.json`: held aside on 2026-11-12. The owner has not yet answered
  whether to re-run this privacy deletion for the one workspace it skipped on 2026-11-04. Move it back
  (before its approval expires 2026-11-18T15:00Z) only on the owner's word.
- `2026-11-04-arena-scratch.json`, `2026-11-04-controlled-policy-integration.json`,
  `2026-11-12-old-development-folders.json`: applied on 2026-11-04 and 2026-11-12, held on 2026-11-12 so
  that a later run deletes only what the operator saw named in that run's plan. A run would re-check
  every entry it skipped before from scratch, and one skipped only as `in_use` could then be deleted.
- `2026-11-12-host-work-volume-reserve.json`: item 3 of the owner's approval of 2026-11-12 (the
  children of `configured-controls`), to be used only if items 1 and 2 free less than expected. Its
  newest entry (2026-09-09) is older than the 27 September the owner was told for the whole folder, and
  `autoprovision` could not be measured: reconcile that before it leaves `held/`.
