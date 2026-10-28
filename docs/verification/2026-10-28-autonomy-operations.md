# Autonomy operations verification

This implements the bounded reporting portions of AUT-17 through AUT-20.
It unblocks beta job-to-evidence diagnosis. It does not establish production
completion, concurrency, scene fidelity or physical performance.

AUT-20 requires an explicitly passed canonical package audit, the exact
capture/package bindings, the complete audit artifact dependency closure and
current proof-boundary checks. Each local capability additionally identifies
its own retained artifacts. Missing, contradictory or changed evidence leaves
the capability `not_proven_in_current_control_plane`. The cross-file regression
uses the canonical auditor: changing attempt count from one to two with only
one clip invalidates the prior pass.

AUT-17 extends the existing operator status with beta triggers, current versus
historical unit state, safe delivery ages and a screening-ledger projection.
The existing ledger/outbox remains authoritative. The worker status contains
counts, observation times and fixed transport failure categories; it contains
no site/campaign IDs, remote bodies or credentials. The projection grants no
launch, retry, funding or publication authority.

The existing preflight writes a schema-allowlisted summary at
`preflight/public/summary.json`. Raw `latest.json` and history remain mode 0600.
The preflight directory gets execute-only traversal (0711), its dedicated
public directory is 0755 and only the safe summary is 0644. No secret/config
directory permissions change. Publication refuses symlinks, unsafe ownership,
an unknown reader or a nontraversable ancestor. The door independently refuses
invalid schemas/digests, stale/future summaries, release drift and
credential-shaped content. The raw report is never opened by this reader.

AUT-19 separates `survey_coverage_fraction` from `classified_fraction`, retains
the historical `attributed_fraction` alias, and explicitly reports unvisited
and unknown unreadable bytes. Hourly capacity surveys use four immediate-root
partitions per configured mount, with a bounded discovery budget and
identity/digest-bound continuation. Changes to root inode, directory/link
inventory or mount identity refuse the old cursor. Known active producer root
names are considered first. Each report remains a partial current snapshot;
successive byte totals must never be summed because of hardlinks and changing
files. A single exceptionally large partition may still be truncated; this
change improves root fairness rather than claiming a full arbitrary-depth
filesystem census.

Loose scratch remains a review-candidate heuristic. A fresh complete existing
reference census can label a root active/referenced. Unknown/incomplete
inventory, an empty reference set or an owner guess never creates retirement
eligibility. Existing owner, restore, pin, lease and retention controls remain.
The survey issues no deletion authority, performs no retirement, and records
no fabricated actual reclaimed bytes or measured concurrent footprint.

AUT-18 adds an opt-in `--lane beta_rigid` shadow report. The existing natural
systemd definition and full diagnostic remain unchanged. The maintained route
checks its exact entrypoint, including a retirement supervisor's worker, and
explicit read/write path requirements. Lazy-import and template candidates
remain traceable; they cannot establish a confirmed selected-route write
requirement. Required output access and required disabled triggers still
refuse. A successful inactive oneshot and an intentional hold remain distinct.
Recognized path coverage is explicitly limited to maintained plumbing
requirements, rather than universal runtime readiness.

Focused checks run without provider calls:

- Proof/readiness regression suite: 31 passed.
- Existing preflight, operator-status and config suites: 65 passed.
- Safe-summary/route/screening-status regressions: 18 passed.
- Changed-file Ruff checks passed.
- Disk, capacity and new partition/lifecycle regressions: 140 passed.

Production safe-summary readback, natural worker/door upgrade, shadow comparison
and actual root-owned/blueprint-user readback remain unverified. Production
access is blocked by the retained network denial; this work does not use an
alternate route. No deployment, paid execution, hold release or production
storage mutation was performed. Two overlapping normal jobs, measured
baseline/peak/final allocated bytes, a complete current production reference
census and normal retention observation remain necessary for AUT-19 acceptance.
