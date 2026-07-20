# Public Launch and SC3 Remediation Status

Updated: 2026-07-20

This is a proof-bounded summary of the current worktree remediation state for
[`PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md`](PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md).
The authoritative row-by-row source is the v3 machine ledger:
[`public_launch_sc3_quality_gap_ledger_2026-07-09.json`](public_launch_sc3_quality_gap_ledger_2026-07-09.json).

## Current verdict

- The audit contains 107 rows and 107 authored acceptance criteria. Each row
  maps its authored exit/acceptance block to a criterion ID, exact audit line,
  criterion-text digest, scope, evidence records, command result, freshness,
  binding state, supersession set, and explicit remaining work.
- The normalized 107-criterion mapping from acceptance-text digest to artifact
  path/role/support flags and applicable command is locked by its own SHA-256.
  That regression lock detects an unrelated-file substitution, but remains a
  control check rather than closure evidence.
- 94 rows and criteria are `partial`: at least one current, digest-valid,
  non-control remediation artifact exists, but the authored criterion is not
  fully proven and is not commit/release bound.
- 13 rows and criteria remain `open`: `REL-02`, `EVID-02` through `EVID-10`,
  plus `EVID-12` through `EVID-14`. Their audit definition
  is digest-bound, but it is explicitly definition-only and cannot derive
  remediation progress. `SC3-22`, `EVID-01`, and `EVID-11` are now `partial`
  because candidate-registry, ranking, and runtime-rights contracts exist;
  none has external closure proof.
- No row or criterion is `closed`. In particular, all 107 `commit` and
  `release_id` bindings remain `null`; this worktree has not fabricated a
  commit or release binding before an actual commit exists.
- Closure derivation is disabled in this v3 snapshot. It cannot be enabled by
  filling in ledger fields: a future schema and verifier must validate an
  externally signed release attestation, the actual Git `HEAD`, and a retained
  release artifact before any criterion can derive `closed`.
- The ledger currently binds 266 criterion-evidence records covering 164
  unique Git-tracked repository artifacts by SHA-256. Of those records, 252
  are remediation records and 14 are definition-only
  references to the source audit for the open rows. They support `partial` or
  `open` status only. An untracked file cannot derive status, and every current
  record is unbound and cannot support closure.
- There are 94 applicable criterion command slots, and all 94 are
  `not_recorded`; the other 13 are `not_applicable`. There are zero claimed
  passing command results because no trusted, retained command-output artifact
  is present. Recorded command attestations are disabled in v3: a console
  summary, self-computed text digest, caller-written artifact, or caller-chosen
  authority string is not proof. A future schema must add a cryptographic
  verifier before any recorded command result can be accepted.
- A final canonical full-suite rerun is still pending after the active
  remediation repairs. This status is not full-suite-green or release proof.

| Family | Partial | Open | Closed | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `REL` | 14 | 1 | 0 | Local release controls exist; GitHub branch protection remains externally unproven. |
| `DATA` | 24 | 0 | 0 | Local truth, package, and output contracts have focused evidence but are not commit-bound. |
| `SC3` | 22 | 0 | 0 | Local evaluator and external-study intake contracts are hardened; a frozen independently accepted Blueprint study remains absent. |
| `RUN` | 24 | 0 | 0 | Local runtime/deployment controls exist; live readback and operator evidence remain separate blockers. |
| `P2` | 8 | 0 | 0 | Local governance and defense-in-depth work has evidence but is not commit-bound. |
| `EVID` | 2 | 12 | 0 | Anchor and runtime-rights intake contracts exist; required external, live, paid, legal, or physical evidence remains absent. |
| **Total** | **94** | **13** | **0** | No launch or scientific claim is upgraded by this document. |

## Launch scope

- The active profile is evaluator-bounded sim policy comparison with live
  runtime and buyer delivery. `BASE`, `SIM`, `SC3`, and `LIVE` criteria are
  launch-scoped; buyer delivery and rights rows `EVID-10` and `EVID-11` are
  explicitly enabled.
- 99 criteria are launch-scoped. 97 are launch-blocking: 87 are `partial`, 10
  are `open`, and none is `closed`.
- `SC3-22` and `EVID-01` remain launch-scoped but nonblocking while the public
  claim mode is exactly `correlation_not_measured`. Enabling a real-world
  ordering claim makes their external study proof blocking.
- PTDP, payment, payout, unsupported-device, and physical-robot-only rows are
  nonblocking when those features or claims are disabled or unmarketed.
  `EVID-14` remains open and cannot be converted into a sim-only launch claim.

## How status is derived

Criterion status is recomputed rather than accepted from prose:

1. `open` means there is no digest-valid, authoritative, independent
   remediation artifact for the criterion.
2. `partial` requires at least one such artifact, while any acceptance,
   command, freshness, commit, release, or remaining-work closure condition is
   still unsatisfied.
3. `closed` additionally requires the closure-authority policy to be enabled by
   a schema with a real external verifier, the acceptance check to pass, every
   applicable command to have a trusted retained result, current closure
   evidence to match the actual `HEAD` and release artifact, and
   `remaining_work` to be empty. The current policy is intentionally disabled.
4. `reopened` wins when a previously closed criterion no longer derives
   `closed`.

Gap status is then aggregated from its criteria. A digest mismatch, stale
record, circular control artifact, missing command result, or absent binding
cannot be papered over by an asserted row status.

## P2-04 is non-circular

P2-04 has no self-evidence exemption. The ledger, this status document, and
`tests/test_quality_gap_ledger.py` are declared control artifacts and are
forbidden from deriving remediation status for P2-04 or any other row.

P2-04 instead derives `partial` from the independently inspectable superseded
banner in `docs/specs/launch-audit-2026-07-02/README.md`, bound in the ledger to
SHA-256
`b0d325d6050728390497e6385526e7ee37b613c7b0940c4dd0211ca47afc8470`.
That artifact is Git-tracked and is neither the ledger, this status document,
nor its validator test. Its acceptance criterion remains unproven because the
current ledger is not commit/release bound, its focused command slot is
`not_recorded`, and durable release evidence does not yet exist.

## Explicit open boundaries

- `REL-02` stays open until GitHub proves required checks, direct-push and
  force-push denial, and an audited break-glass path on the actual repository.
- `SC3-22` and `EVID-01` remain externally unproven even though their local
  protocol/intake work now derives `partial`. They require a frozen,
  independently accepted study before any real-world correlation claim.
- `EVID-02` through `EVID-10` and `EVID-12` through `EVID-14` stay open.
  `EVID-11` remains `partial`; its runtime license contract is not counsel or
  privacy-owner signoff. Repository declarations, fixtures, local contract
  tests, and artifact templates do not satisfy those evidence rows.
- `EVID-14` is physical-robot proof. It is explicitly **nonblocking for the
  evaluator-bounded sim-only scope** and remains blocking only for physical
  deployment or physical-safety claims.

## Remaining closeout

Every criterion contains its authored acceptance text in `remaining_work`,
plus the missing commit/release binding. Every applicable command is currently
`not_recorded`; the ledger never turns an absent, future-dated, self-attested,
or unbound command result into success. Closure still requires the final
canonical suite, static/dependency/package/container gates, commit parity, a
trusted retained result artifact, and the applicable external evidence for the
launch scope being claimed.
