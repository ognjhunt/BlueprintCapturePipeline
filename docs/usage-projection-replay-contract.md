# Recover immutable placement usage delivery

Scope: ADP-081, partner-proof day-42 economics evidence; this is a prerequisite
repair, not completion of that gate. The configured-controls placement path
already checkpoints inference and projects usage into WebApp `agentRuns`.
Estimates remain estimates, separate from official billing and spend authority.

Starting source: Pipeline `dd2d404fbcd914f7c371849d31b8a86fcc8e84a2`, clean
`work` checkout; implementation uses isolated `codex/usage-replay-20261005`.
Current-main, open-PR, branch-name and file-history reads found no owner of this
specific projection retry. Research/contact and scene settlement work remain
outside this change. Other machines' uncommitted changes are not observable.

## Before

Saved placement -> build timestamped packet -> immutable write -> signed sync
-> bound ingest receipt -> controls continuation.

After a timeout: saved placement -> new timestamp -> immutable conflict ->
operator intervention before the existing idempotent consumer can reconcile.

## After / implementation contract

- Trigger: the existing configured-controls call or its existing recovery replay.
- Input: existing placement receipt, run ID, launch ID, source commit and output
  directory. No additional model, provider, scheduler, credentials or permissions.
- First writer atomically publishes the complete immutable packet. A concurrent
  writer or restarted caller reads that packet, validates its digest and compares
  the complete projection to the current input, preserving its original timestamp.
  Changed run, launch, source or usage data fails before transport.
- Code signs and resends the same serialized packet through the existing adapter.
  WebApp's transaction retains one call record per `call_id`, requires the same
  packet digest, and returns `created` or `replayed`. A lost acknowledgement is
  reconciled by replay, never by repeating inference.
- Transport results are separate immutable content-addressed artifacts. A failed
  or skipped attempt cannot occupy the immutable location of a later successful
  receipt. Existing packet and legacy sync artifacts remain untouched/readable.
- States: packet absent -> retained -> delivery failed/skipped or acknowledged.
  `require_sync` still blocks continuation until a bound success. Existing caller
  owns retry timing/deadline; one bounded transport attempt per call, no busy loop.
- Exceptions retain existing `openai_inference_usage_*` errors. Invalid retained
  bytes fail closed; transport failures retain their typed result before raising.
- Completion: downstream ingest acknowledges exact run/launch/source/digest/count,
  and existing result artifact validation permits controls continuation. A local
  test is not deployed, runtime-observed or customer-outcome proof.
- Tests: lost response after consumer acceptance, exact body/digest replay,
  optional failure then required recovery, mismatched bindings, corrupt/symlink
  packet, simultaneous writers, crash before packet publication and after it.
- Rollout: normal reviewed Pipeline release, fresh ownership and idle-state reads,
  required CI and existing main-ancestry deploy. No paid or external-send canary.
- Rollback: preserve packet and all attempt files. Old code can read saved result
  descriptors but retains the old rematerialization defect; do not delete evidence
  or rerun inference as a rollback workaround.

Mechanism: deterministic file/transport reliability. The managed Agents API
already owns the model/tool loop; neither a new agent nor Agents SDK is needed.
Official architecture documentation checked on 2026-10-05:
https://developers.openai.com/api/docs/guides/agents-api/architecture
