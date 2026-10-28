# Retained handoff recovery

This is an optional pass of the existing Pub/Sub listener job. It creates no
publisher, service, credentials or spending authority. The existing retained
DLQ subscription and per-capture lease/authority/budget mechanisms remain in
charge. Code and offline tests do not prove source-to-DLQ forwarding or live
acknowledgement.

The default is observe and classify. Configure only the already-authorized
retained subscription as `BLUEPRINT_PUBSUB_HANDOFF_DLQ_SUBSCRIPTION`. The normal
listener reads at most one retained delivery each tick, persists exact private
bytes and broker provenance under `<storage-root>/.handoff_recovery`, and neither
acknowledges nor executes it in this mode. Broker source-subscription attributes
must match the actual source subscription; otherwise it stays unacknowledged
and reports an orphan. Oversize/unprovable input is never acknowledged.

After independent review and observation of the actual retained subscription,
the release owner may explicitly set
`BLUEPRINT_PUBSUB_HANDOFF_RECOVERY_SAFE_REPLAY=true`. Only this mode acknowledges
a retained message, after its recoverable local successor has been fsynced.
An acknowledgement receipt is written only after the broker call returns. An
uncertain acknowledgement can redeliver the same immutable event; it does not
reset recovery state or per-capture effect history.

Automatic execution is limited to source-listener evidence naming an owner
transport or generation-pinned membership/staging failure before a job lease.
The original payload is passed to the canonical processor with
`require_unattempted_delivery=True`; it reopens current owner, rights, membership
and grants. The claim refuses every existing ledger. A concurrent source worker
therefore wins the same existing lease boundary rather than starting another
attempt. Recovery uses deterministic jitter, exponential backoff up to one hour,
and one wake after an observed release/dependency generation change. A denial,
changed source, deliberate hold or unclassified error is never made retryable
by elapsed time or deployment.

Completed canonical output is adopted. Ended authority remains ended. Explicit
assessment/reconstruction waits retain their existing wait reconcilers.
Unknown provider outcomes retain the original operation and an exact named
next action; there is no generic paid replay or provider-zero inference.
Recovery records bind source bytes, finalization/membership selectors, and any
existing ledger's attempt/revision/token/delivery key. They always say
`retry_exhaustion_proven=false`: five broker deliveries or long silence do not
prove terminal preparation failure or renew funding.

Use the existing CLI for a safe summary:

```bash
python -m blueprint_pipeline.pubsub_handoff_listener --storage-root /var/lib/blueprint/pubsub-handoffs --recovery-status
```

The same summary is retained at `.handoff_recovery/status.json`. It includes
pending/due counts, oldest observed broker age, next actions by safe cause,
retention deadline and orphan/reconciliation alerts. Raw payloads, broker ack
credentials, customer content and remote error bodies are not in the summary.
Missing broker publish time is an explicit unknown-retention alert. No recovery
record is deleted by this pass.

Production acceptance still needs a permitted non-customer source message held
through all broker retries, actual forwarding into the retained subscription,
dependency recovery, one logical execution, and source/DLQ/recovery
acknowledgement readback. Keep injected failure receipts and verify crash
boundaries, expired/withdrawn authority, unknown provider outcomes and retention
alerts against that permitted subscription. Do not infer that proof from IAM
configuration or these local tests.
