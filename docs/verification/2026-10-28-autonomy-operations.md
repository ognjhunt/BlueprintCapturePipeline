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
identity/digest-bound continuation. The bounded v2 cursor preserves a rotating
queue of child identity digests when new job roots appear. New entries join the
tail, so additions cannot repeatedly restart the first four partitions. Changes
to a root or child inode/type, or mount identity, refuse the corresponding stale
identity. Known active producer root names are considered first when discovered.
Each report remains a partial current snapshot;
successive byte totals must never be summed because of hardlinks and changing
files. A single exceptionally large partition may still be truncated; this
change improves root fairness rather than claiming a full arbitrary-depth
filesystem census.

Partition discovery and every selected/nested directory walk now use retained
directory descriptors, component-by-component nofollow opens and device/inode/
type checks. Linkage is checked before and after observation. A cursor's cached
stat is an expected identity, never authority to scan a path. A replaced or
unverifiable directory yields `partition_identity_refused`, incomplete inventory
and unknown unreadable bytes; outside targets are not traversed or attributed.

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
and actual root-owned/blueprint-user readback remain unverified by this lane.
This lane performs only source changes and offline checks; root coordinates any
separately authorized existing production route. No alternate route, deployment,
paid execution, hold release or production
storage mutation was performed. Two overlapping normal jobs, measured
baseline/peak/final allocated bytes, a complete current production reference
census and normal retention observation remain necessary for AUT-19 acceptance.

The independent AUT-19 review reproduced starvation when a new job root was
added before each four-partition pass. Its scratch regression failed on
`f1ff10df6`; the queue correction passes it together with seven local
continuation tests, including arrival of an earlier-sorting priority root.
The independent selected-directory replacement regression also failed on that
head and now passes. Six local identity regressions additionally cover a swap
before open, before and after scandir, a nested swap, root replacement during
inventory and same-device replacement by a real directory. The two existing
path-based mock fixtures now inject by descriptor identity, retaining their
unreadable-directory and non-UTF8 assertions.

Final AUT-19 source-only review checks: 162 passed in 24.85 seconds across
`tests/test_control_plane_disk_usage_identity.py`,
`tests/test_control_plane_disk_usage.py`,
`tests/test_control_plane_disk_usage_continuation.py`,
`tests/test_control_plane_capacity_controller.py`,
`tests/test_lane_scratch_census.py` and the exact two independent probes at
`/tmp/autonomy-coverage-operations-review-20261028/test_aut19_independent_review.py`.
Changed-file Ruff and whitespace checks pass. A separate broader run including
the lane-scratch contract tests passed 176 tests before the final root-file
error classification change; the final 162-test run includes that change.
