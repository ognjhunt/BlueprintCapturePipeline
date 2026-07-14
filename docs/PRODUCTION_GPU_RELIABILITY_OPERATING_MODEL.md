# Production GPU reliability operating model

This is the one production path for customer-facing G1 kitchen evaluation.
Legacy cold-launch commands remain diagnostic and release-engineering tools;
they are not customer-serving entrypoints and cannot promote a release.

## One golden path

```text
clean protected main
  -> sealed release-candidate manifest
  -> immutable host image + exact cached worker digest
  -> canary and reliability qualification
  -> promoted warm-worker release tuple
  -> atomic worker bind (zero provider calls)
  -> smoke seed 1000
  -> only if smoke passes: episode seeds 1001, 1002, 1003
  -> incremental hash-verified artifacts
  -> terminal evidence validation and customer readout
  -> healthy worker release or quarantine
```

The release tuple is source SHA + build-input fingerprint + host image ID +
full OCI digest + model/asset revisions. Changing any member creates a new
release candidate and resets qualification.

## The ten reliability controls

| Requirement | Enforced implementation | Promotion evidence |
|---|---|---|
| 1. One golden path | `production_gpu_release_candidate.py`, this document, and the production CLI/service units | Sealed closure and exact digest |
| 2. Remove customer cold start | `production_gpu_worker_pool.py` atomically binds only a fully warmed worker; no provider adapter is imported | Warm-bind p95 probe |
| 3. Control image size | `production_gpu_image_contract.py` classifies the 47 GB image as `active_worker_only`; Packer caches it before readiness | Registry size/layer diagnostic and baked-host proof |
| 4. One authoritative control plane | `production_gpu_campaign_control_plane.py` owns campaign/attempt/artifact truth; only the separate autoscaler owns provider mutation | API contract tests and provider-call count zero |
| 5. Explicit startup/episode states | Durable SQLite transitions reject illegal skips and require terminal reasons | Event log plus terminal snapshot |
| 6. Canonical episode contract | One smoke plus three unique seeds, exact closure, dynamic completion, 30-minute cap, and at least 640x480 | Spec digest and attempt manifests |
| 7. Resumable artifacts | Offset-checked chunks, per-chunk hashes, final hashes, atomic finalization, per-attempt isolation | Complete required artifact set |
| 8. Continuous reliability | `production-gpu-reliability.yml` runs contract qualification and an opt-in scheduled private warm-bind canary | Repeated campaigns, p95 metrics, rollback drill |
| 9. Honest asynchronous UX | Customer status is accepted/queued/starting/running/processing/completed/failed; no invented ETA or provider internals | Status API contract |
| 10. Engineering operating model | Ownership, SLOs, incident response, and promotion rules below | Reviewable release qualification |

## Ownership and mutation boundaries

- Release engineering owns source closure, OCI build, SBOM/provenance, host
  image bake, and candidate qualification.
- The warm-pool registry owns ready-worker truth and atomic leases.
- The campaign control plane owns customer job, attempt, artifact, terminal,
  and semantic-success truth. It accepts no provider credentials.
- The asynchronous autoscaler is the only production component authorized to
  create paid capacity. It must reserve the campaign budget, use the paid-lane
  admission lock, and open the teardown ledger. A customer request never calls
  it synchronously.
- The independent hard-TTL watchdog is the second provider mutation owner, but
  termination-only: it may delete an allocation and close its teardown ledger
  if the normal owner crashes or misses the deadline. It cannot create capacity.
- A worker owns simulator execution and evidence emission, never release
  promotion.

Direct provider adapters remain available for bounded diagnostics. Calling one
does not update production promotion state and cannot satisfy a customer job.

## SLOs and failure behavior

- Warm bind target: p95 at or below 10 seconds; hard request contract 30 seconds.
- Customer request cold provisioning: forbidden.
- Asynchronous cold replenishment target: p95 at or below 1,800 seconds.
- Episode runtime: dynamic task completion with a 1,800-second emergency cap;
  there is no fixed frame count.
- Review output: at least 640x480 and validated for decode, duration, frames,
  nonblank content, and attempt provenance.
- Smoke failure terminates the campaign as `smoke_blocked`; full episodes stay
  `planned` and cannot transition to running.
- Missing/corrupt artifacts prevent an attempt from passing.
- A stale or unhealthy worker is quarantined, never silently returned to ready.
- Ambiguous provider allocation or teardown keeps the paid ledger open and
  blocks further production mutation until reconciled.
- Open budget reservations retain their worst-case USD and wall-time charge;
  controller crashes therefore cannot return unproven budget to the pool.

## Release promotion and rollback

Static tests establish contract correctness only. Promotion requires fresh,
exact-release live evidence: minimum ready capacity, repeated terminal
campaigns, required attempt pass rate, warm-bind p95, cold-replenishment p95,
provider inventory, rollback drill, and teardown/absence proof. The
qualification result is either `promoted` or `quarantined`; there is no partial
success label.

`production_gpu_launch_qualification.py` assembles current registrations,
provider inventory, a minimum 20-sample warm-bind probe, three asynchronous
replenishment cycles, the rollback drill, and teardown proof into the existing
fail-closed startup gate. Historical evidence and local benchmarks are rejected.

Release qualification and current deployment are separate claims. A rehearsal
may prove warm startup, binding, rollback, and teardown, but
`customer_launch_ready` additionally requires the configured minimum (two for
the current active-worker image) to be ready in the pool and simultaneously
present in fresh provider inventory. Tearing down the rehearsal therefore
closes the customer launch gate until production capacity is deployed again.

Keep the prior promoted tuple ready until the candidate passes. Rollback stops
new candidate leases, drains/quarantines candidate workers, restores the prior
tuple, proves ready capacity, then deletes retired capacity with provider-side
absence evidence.

## Incident response

1. Stop new leases for the affected release fingerprint.
2. Preserve the control-plane snapshot, event log, partial artifacts, paid
   teardown ledger, provider phase trace, and exact release manifest.
3. Quarantine affected workers and reconcile any ambiguous allocation before
   another provider mutation.
4. Classify the defect as release closure, host/runtime, capacity, simulator,
   policy, renderer, transport, or evaluator.
5. Configuration-only repairs may retry the immutable release. A build-closure
   change requires protected-main merge, a new digest, and qualification from
   smoke zero.
6. Publish an honest customer terminal status; never report task success from
   startup, artifact arrival, or video validity alone.

## Current proof boundary

The local implementation and focused tests prove the contracts and fail-closed
transitions. They do not prove that a provider host image is currently baked,
that a production warm pool is deployed, or that three live kitchen episodes
completed semantically. Those claims remain blocked until the exact release
tuple passes the live qualification workflow.
