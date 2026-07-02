# SPEC-13: Re-run city-launch evidence; fix Austin backend quota; unblock the evidence trail

- Status: Proposed
- Priority: **P1 — major** (operational; blocks any evidence-based launch decision)
- Area: `ops/city-launch-runs/`, city-launch harness runs, WebApp city-status backend (cross-repo)

## Problem

The newest launch evidence on disk cannot support a go decision for any city:

1. **Everything is blocked and stale.** Latest runs: Austin `2026-05-06T015927`
   (`blocked_external_dependency`, 33 blockers), Durham `2026-05-11`
   (`blocked_repo_or_contract_failure`, 36 blockers), audit-city `2026-05-05`
   (38 blockers). No `ready_to_market_*` proof exists anywhere; artifacts are ~2 months
   old (today: 2026-07-02). The latest Austin proof shows every live capability false:
   `capture.real_device_capture_uploaded`, `pipeline.pubsub_handoff_succeeded`,
   `retrieval.dense_index_exists`, `hosted_session.*`, `payouts.*`, all `ops.*` monitors.
2. **Austin's gating data source was down during its runs:**
   `lane-results/city_backend.webapp-status-route.json` records
   `webapp_status_route_source_unavailable` with `RESOURCE_EXHAUSTED: Quota exceeded` on
   the `cityLaunchActivations` / `cityLaunchCandidateSignals` Firestore collections and
   `supported_city_count=0`. Austin readiness is currently *unknowable*, not merely
   unproven.
3. **Sim-only robot-eval path equally blocked** (`docs/last_24h_launch_audit_2026-06-26.md`,
   `READINESS_MATRIX.md:29`): release gate blocked, production intake token/health
   unproven, and no live GPU frame has ever been produced.

(Durham's `backend_supported=true` with zero live supply is a WebApp-side fix — see
`Blueprint-WebApp` spec WSPEC-09 in the companion audit.)

## Why this blocks beta

A beta decision needs current evidence. Right now the honest summary is: "as of 8 weeks
ago, every lane was blocked, and one city's gate data source was quota-exhausted." Running
a beta on this trail would be launching blind — and the blocker lists are the actionable
work queue we should be burning down.

## Proposed fix

1. **Fix the Firestore quota/indexing issue** on `cityLaunchActivations` /
   `cityLaunchCandidateSignals` (quota increase, query pagination, or cached status
   endpoint) so the city-status route can actually answer.
2. **Re-run the city-launch harness (`--execute-local` where applicable) weekly** for
   each candidate city during the launch push; wire it to the SPEC-11 exit-code contract
   so CI surfaces blocked-external distinctly. Keep artifacts under
   `ops/city-launch-runs/` with a `latest` symlink/manifest per city.
3. **Triage the blocker lists** from the newest Austin/Durham runs into an ordered
   burn-down (each blocker mapped to an owner and, where code-level, to one of
   SPEC-01…12). The harness blocker taxonomy already names the gaps — treat it as the
   beta checklist.
4. **Evidence freshness TTL:** launch/beta decisions require harness artifacts newer than
   a configurable TTL (7 days suggested), enforced by the release gate (shared with
   SPEC-11 item 5).

## Acceptance criteria

- [ ] City-status route returns real data for Austin (no quota errors) and the harness city-backend lane succeeds or reports true blockers.
- [ ] Fresh (≤7 days) harness runs exist for every beta-candidate city, with blocker deltas tracked run-over-run.
- [ ] Beta go/no-go references only in-TTL artifacts; the release gate enforces the TTL.
