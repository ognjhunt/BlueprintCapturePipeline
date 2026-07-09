# Provider Reliability Manifest (`provider_reliability_manifest.v1`)

Ops-facing, fail-closed ledger for every paid GPU provider run (RunPod, Vast,
Lambda, DigitalOcean, fixture). One JSON file per run answers, without hunting
old sessions or replaying logs:

- Which phase did the run reach, and which phase failed first?
- What exact blocker stopped it?
- Is the billing allocation proven torn down, or is there open billing risk?

Module: `src/blueprint_pipeline/provider_reliability_manifest.py`
Tests: `tests/test_provider_reliability_manifest.py`
First integrated writer: `isaac_particlefield_render_job` writes
`<out_dir>/provider_reliability_manifest.json` on every paid attempt (including
fail-before-spend attempts).

## Run phases (ordered)

| Phase | Meaning | Typical blockers |
| --- | --- | --- |
| `pre_spend_preflight` | Contracts validated BEFORE any billable request | `capacity_unavailable:*`, `worker_image_contract_invalid:*`, `runtime_contract_invalid:*`, `provider_credential_missing:*`, `spend_gate_closed:*` |
| `provider_launch` | Provider accepted the create request; billing started | `provider_launch_failed`, `capacity_unavailable:race_lost` |
| `container_startup` | Worker booted and wrote its startup marker | `startup_marker_timeout:*` |
| `runtime_execution` | Isaac/WAM work is emitting progress markers | `post_marker_no_progress:*`, `runner_failed:*` |
| `artifact_collection` | Outputs retrieved locally, fresh, checksummed | `required_artifact_missing:*`, `stale_artifact_rejected:*`, `artifact_collection_failed:*` |
| `artifact_quality` | Collected media is decodable/non-degenerate | delegated to `media_validity_contract.v1` |
| `task_evaluation` | Task-level judgment | delegated to `success_claim_ledger.v1` |
| `teardown` | Billing allocation proven terminal | `teardown_unproven:*` |

`failed_phase` is always the FIRST failing phase; later phases stay recorded but
cannot rewrite the run's story. Phases a lane genuinely does not perform (e.g. a
render-only lane never does `task_evaluation`) are declared in
`not_applicable_phases` — recorded as not-applicable, never as PASS. `teardown`
and `pre_spend_preflight` can never be declared not applicable.

## Top-level fields

```json
{
  "schema_version": "provider_reliability_manifest.v1",
  "run_id": "…",                       // launch session nonce or instance id
  "provider": "runpod",
  "session_dir": "output/…",           // where the run's artifacts live
  "run_completed": false,
  "furthest_phase_reached": "runtime_execution",
  "failed_phase": "runtime_execution",
  "failure_blockers": ["post_marker_no_progress:…"],
  "teardown_proven": true,
  "open_billing_risk": false,
  "not_applicable_phases": ["artifact_quality", "task_evaluation"],
  "phases": { "<phase>": {"present": true, "passed": true, "not_applicable": false, "blockers": []} },
  "phase_contracts": { "<phase>": { …full contract… } },
  "blockers": [],
  "claim_boundary": "…"
}
```

## Component contracts

### `pre_spend_preflight.v1` — fail before spend

`build_pre_spend_preflight(...)` must PASS before any billable provider request:
capacity evidence (`available` strict boolean from a read-only offer query),
worker image contract (`image_ref` present and `pinned=True`), runtime contract
(startup + progress marker names and positive timeouts — an unmonitorable run
must not spend), credential presence, and the explicit spend gate. Missing or
non-boolean evidence fails closed.

### `post_marker_stall_policy.v1` — terminate stalls

`evaluate_post_marker_stall(...)` distinguishes the two stall modes:

- startup-marker timeout → launch succeeded, container startup failed;
- post-marker no-progress → startup succeeded, runtime execution stalled. A
  worker that never writes a progress marker after boot uses the startup marker
  age as its stall clock — silence is a stall, never patience.

In `isaac_particlefield_render_job`, the post-marker watchdog is now ON by
default (`BLUEPRINT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS`, default 900s) and
a stalled pod is terminated with `teardown_reason=post_marker_progress_timeout_terminated`.
The render bootstrap also carries pod-side backstops:
`BLUEPRINT_RENDER_POD_HARD_TTL_SECONDS` (default 7200s) kills the worker command
even if the host collector dies before teardown, and
`BLUEPRINT_RENDER_POD_IDLE_TTL_SECONDS` (default 1800s) exits the keep-alive loop
after `runner_done` so result-collection grace cannot become an infinite billable
container. These pod-side exits are only runtime-cost containment; provider API
teardown proof still comes from the host watch loop or scheduled spend guard.

### `provider_teardown_proof.v1` — billing must end in evidence

A terminate *request* is not proof. Teardown passes only with a
provider-reported terminal status (`terminated`, `destroyed`, `deleted`,
`exited`, `not_found`) plus a verification timestamp. RunPod `STOPPED` is NOT
terminal (stopped pods bill disk). Intentional keep-alive (warm pools,
`keep_on_success`) is recorded as an OPEN billing allocation with a reason —
never silence.

### `provider_artifact_collection.v1` — stale artifacts are rejected

Required artifacts must be present AND proven fresh
(`artifact_freshness_evidence.v1` — run-id match, generated-at, or mtime vs run
start). Present-but-stale output from a previous warm-pool run is rejected, not
accepted. Optional artifacts (e.g. review media) may be skipped only with an
explicit `skip_reason`; skips are recorded, never silent.

## Related hardening

- Local MP4 repair (`isaac_g1_kitchen_parity_job._repair_collected_review_mp4s`)
  now takes the run's `expected_frame_count`; a repair over fewer frames is
  labeled `repaired_truncated` with blocker
  `mp4_repair_truncated_frames:<video>:<n><<expected>` instead of `repaired` —
  a locally assembled MP4 can no longer mask a partially-uploaded provider render.
- `scripts/gpu_spend_guard.py --json-report <path>` persists a
  `gpu_spend_guard.v1` snapshot (live allocations, burn/hr, protected ids, reap
  candidates, reap results) so watchdog runs leave durable teardown evidence.
  Use `--max-live-instances` and `--max-burn-usd-per-hour` to persist the
  `gpu_fleet_budget_guard.v1` aggregate ceiling in the same snapshot; a blocked
  fleet budget exits with code 2 and must not be treated as launch-ready.
  Booted orphan allocations are eligible only after
  `--max-booted-orphan-seconds`, while expected warm workers remain protected by
  live `warm_serve_pod.json` markers rather than a static provider id allowlist.
- `blueprint-run-lambda-provider-adapter --mode terminate-instances` now follows
  the terminate request with bounded `GET /instances` verification. The teardown
  manifest is `completed` only when every requested id is absent from the
  provider list or reports a terminal status; otherwise it remains
  `termination_unverified` with `open_billing_risk=true`.
- Customer robot-eval provider failover is wired through
  `blueprint-run-robot-eval-provider-race` as live-gated serial adapter failover:
  the handoff requires at least two runnable provider adapter commands, and live
  execution additionally requires `BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH=true`
  plus `--allow-live-provider-race`. This is not parallel provider racing and a
  completed launcher result still does not prove simulator execution, artifact
  quality, task success, or generated-world/rank fidelity.

## Claim boundaries

Every contract in this manifest is infrastructure/reliability evidence only.
Provider launch, container startup, runtime execution, and artifact collection
NEVER imply artifact quality or task success — those claims live exclusively in
`media_validity_contract.v1` and `success_claim_ledger.v1`
(`success_claim_contracts.py`) and keep their own blockers.
