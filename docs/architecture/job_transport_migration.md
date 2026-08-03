# Job transport migration (C8) — strangler, not rewrite

The 10.5k-line `robot_eval_job_orchestrator.py` is not "a filesystem queue";
its inbox runner is one region among routing, evaluation, packaging, claim
boundaries, and execution logic. Only transport/retry mechanics move to
managed services; domain state machines (admission, evidence, gates,
terminal commits) stay custom.

## Division of ownership

| Layer | Owner |
| --- | --- |
| Capture/handoff events, fan-out | Google Pub/Sub (existing base dependency; `pubsub_handoff_listener.py` already implements leases, ack heartbeats, permanent-invalid, DLQ evidence) |
| Explicit job dispatch (dedup, schedule, rate, bounded retries) | Google Cloud Tasks (`cloud_tasks_dispatch.py`; deploy precedent: `functions/storage_trigger.py` mode switch) |
| Job identity, idempotent claims, attempts, evidence, budgets, terminal results, provider-zero proof | Blueprint artifacts/state (unchanged) |
| Hard TTL, spend protection, teardown on consumer death | Independent watchdogs (`paid_lane_guard`, `gpu_spend_guard`, stall watchdog) — transport-independent, pinned by `tests/test_job_transport.py::test_watchdogs_remain_transport_independent` |

Google provides durable delivery only. GPU compute stays provider-neutral:
the subscriber runs in Blueprint's control plane and invokes
`paid_resource_allocator`; Vast/RunPod/Lambda need no queue awareness, and
credentials live in the allocator — `blueprint.job_envelope.v1` refuses
credential-shaped payload keys at build time. If Pub/Sub is unavailable, new
dispatch pauses; nothing about teardown or spend safety depends on it.

## The immutable envelope

`job_transport_envelope.py` wraps the existing `robot_eval_job_request.v1`
contract without changing it: content-derived `envelope_id`
(job id + canonical payload sha256, timestamp-independent), payload digest,
`execution_authority: filesystem` until promotion. Pub/Sub is at-least-once
and can redeliver acked messages; Cloud Tasks task-name dedup tombstones
expire (~1h) — so the envelope id feeds Blueprint's own durable idempotency
(processed markers, per-capture ledger), which remains the truth.

## Migration steps (in order, each gated on evidence)

1. **Envelope** around the existing request contract — done
   (`job_transport_envelope.py`).
2. **Shadow-publish and compare, never execute twice** — done, default off:
   `BLUEPRINT_JOB_TRANSPORT_SHADOW=1` publishes each admitted job's envelope
   (topic via `BLUEPRINT_JOB_TRANSPORT_SHADOW_TOPIC`, else local evidence
   ledger) from the inbox admission point; `compare_shadow_parity` reports
   published/delivered/missing/duplicates. Publish failures are contained —
   admission never breaks.
3. **Move a non-paid fixture lane first**, filesystem fallback retained:
   Cloud Tasks dispatch is allowlisted to `{"fixture"}`
   (`CLOUD_TASKS_ALLOWED_LANES`) — widening it is a deliberate code change.
4. **Prove behavior before promotion**: duplicate delivery idempotence,
   consumer crash, lease expiry, DLQ routing, replay, terminal-commit — the
   existing listener already implements the lease/ack/DLQ machinery; parity
   ledgers supply the evidence.
5. **Paid lanes last**, only with independent-watchdog and provider-zero
   proofs intact.
6. **Delete only superseded polling/transport code** — never evidence
   artifacts or teardown ownership.

## Retry and breaker policy

`transport_retry_policy.py` (shared with the C4 provider transport):

- Reads: `bounded_read_retry` — mandatory exception allowlist, attempts AND
  total-delay stops, jittered exponential backoff, evidence hook. Tenacity's
  unbounded defaults are unreachable.
- Mutations: `mutation_single_attempt` (exactly one attempt) or
  `reconcile_then_retry_mutation` (retry only after provider inventory
  proves absence; unproven absence raises `MutationRetryForbidden`). This
  preserves the `allocation_created is False` vs
  `allocation_outcome_ambiguous` paid-lane discipline.
- `optional_circuit_breaker` (pybreaker) is optional, process-local, and
  advisory at provider integration points; it never replaces
  `provider_race.ProviderCircuitBreaker` or paid-resource circuit state.
  pybreaker is not a base dependency; requesting it without the package
  raises rather than degrading silently.
