# SPEC-11: Launch-gate hardening — exit codes, skip-to-green, self-attested proofs

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

- Status: Proposed
- Priority: **P1 — major** (P0 for item 1 if any automation gates on harness exit codes)
- Area: `src/blueprint_pipeline/city_launch_autonomy_harness.py`, `scripts/run_external_alpha_launch_gate.py`, `scripts/run_paid_marketplace_launch_gate.py`, `scripts/run_sim_only_beta_release_gate.py`
- Doctrine: no fake readiness; evidence checklist (`autonomous-loop-evidence-checklist-2026-05-03.md`)

## Problem

The gate machinery is mostly honest in its artifacts, but several green signals can be
produced without any real verification — exactly how a team talks itself into a launch
the evidence doesn't support:

1. **Exit code 0 on blocked runs:** `city_launch_autonomy_harness.py:1382` returns
   non-zero only for `blocked_repo_*`; a run blocked on external dependencies (e.g.
   Austin 2026-05-06 with 33 blockers) exits `0`, indistinguishable from success to CI or
   scripts.
2. **External alpha gate is trivially satisfiable:**
   `run_external_alpha_launch_gate.py` lets every leg be skipped (`:201-204`); Android is
   silently `manual_required` on any host without the SDK (`:267-273`); the iOS leg
   targets a simulator (`:239-265`); the pipeline leg runs hermetic contract tests under
   `contract_test_env()` (`:277-282`); no artifact/hash/live-service checks; ends
   `print("passed"); return 0` (`:284`).
3. **Paid-marketplace gate skip-to-green:** `run_paid_marketplace_launch_gate.py:242-249`
   skips missing iOS/Android toolchains to `manual_required` (`:101-112`), and overall
   failure counts only `blocking and status=="failed"` (`:413-421`) — so
   "automated_contracts_passed" can ride entirely on mocked JS/Python suites
   (see `ops/paid-marketplace-launch-gate-2026-05-07.json`: Firebase dev fallback, Stripe mocked).
4. **Sim-only release gate trusts self-attested proof JSON:**
   `run_sim_only_beta_release_gate.py:317-414` reads `production_deployment_proven`,
   `git_parity_proven`, `pipeline_intake.accepted` straight from files with no
   independent probe/signature at gate time; `ready_for_beta_release = not blockers` (`:467`).
5. **Green-by-default proof fields:** `city_launch_autonomy_harness.py:176-192` pre-sets
   eight payout/ops/claim fields to `true` in `default_proof`, so they can never surface
   as blockers.
6. **Non-reproducible readiness artifact:** `beta_launch_readiness_deep_audit_current.json`
   has no committed generator, records a stale HEAD, and is already out of date
   (`docs/last_24h_launch_audit_2026-06-26.md:44,67,190`).
7. **Misreadable lane status:** the city-harness pipeline lane reports `succeeded` off
   four local marker files from a placeholder capture root while the load-bearing
   `pipeline.pubsub_handoff_succeeded` is false (`city_launch_autonomy_harness.py:1036-1050`;
   Austin `lane-results/pipeline.capture-root-evidence.json`).

## Proposed fix

1. **Exit-code contract:** harness exits 0 only for `ready_*`; distinct non-zero codes
   for `blocked_repo_*` (2) vs `blocked_external_dependency` (3). Update callers/CI.
2. **Skip ≠ green anywhere:** gates that skip a leg must (a) exit non-zero unless an
   explicit `--allow-skip <leg> --reason <text>` waiver is passed, and (b) write the
   waiver (with reason, operator, expiry date) into the gate artifact. Expired waivers
   fail the gate.
3. **Verification-at-gate-time:** the sim-only release gate probes live health endpoints
   and re-derives git parity itself (or verifies a signature over the upstream proof
   emitted by the producing job) instead of trusting bare booleans.
4. **No green-by-default proof fields:** every `default_proof` field starts false/absent;
   policy/claim-boundary fields become a separate `policy_attestations` block that is
   visibly attested (who/when), not silently true.
5. **Reproducible readiness JSON:** commit the generator for the deep-audit artifact
   (script + inputs), stamp it with HEAD + timestamp, and fail the release gate when the
   artifact is older than a configurable TTL (e.g. 7 days) or built from a different HEAD.
6. **Rename lane truth:** the pipeline lane's local-marker check becomes
   `local_contract_markers_present`; `succeeded` requires the live-evidence field
   (`pubsub_handoff_succeeded`) so a skim can't misread placeholder evidence as real.

## Acceptance criteria

- [ ] Blocked-external runs exit non-zero; CI treats them as failures.
- [ ] Any skipped leg without an in-date waiver fails the gate; waivers appear in artifacts with expiry.
- [ ] Sim-only gate performs at least one live probe per claimed-live capability, or verifies producer signatures.
- [ ] `default_proof` contains no pre-set `true` capability fields.
- [ ] Deep-audit JSON regenerates from a committed script and carries HEAD/timestamp; stale artifacts block release.
