# Verified handoff

Recovery branch: `codex/private-provider-eval-recovery-20260930`, based on
`1f771e61`. Final independent GPT6.1Sol review found no remaining blockers in
the reviewed fixes for local commit. The last canonical-journal alias fix was
independently replayed: 14 focused HTTP tests passed, one mocked dispatch,
no replacement journal after retargeting the output alias.

Local final suite: **42 tests passed**, changed-file Ruff and diff checks passed.
Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest experiments.provider_eval_recovery.test_harness experiments.provider_eval_recovery.test_live_http experiments.provider_eval_recovery.test_public_inputs -q
.venv/bin/ruff check experiments/provider_eval_recovery
git diff --check
```

The repository's six sentinel targets produced **9 passes**: success-claim
freshness, runtime URL security, shared paid admission (four parameter cases),
unmanifested mutation rejection, bounded release verification and lane-writer
governance. No production code or runner2486/snapshot2487 was modified.

The supplied actual public-file bytes match
`cac8d7a31aea2ad1c2e5a47ea37e434abcaae4b4910ba4afc8f81dd11c404ff3`.
`public_inputs.py` prepared **20 actual cases / 80 envelopes**, zero provider
calls, from the explicitly labeled parent-message route. Actual oracle remains
parent-side; Library ZIP transfer failed twice via the supported helper and its
ZIP hash remains unverified in this executor.

Artifacts:

- `/workspace/provider-eval-real-input-evidence-20260930/`: private actual-input
  provenance, mapped cases and80 request envelopes.
- `/workspace/provider-eval-recovery-final-evidence-20260930/`: separate synthetic
  lifecycle rehearsal, pilot8 plus remaining72 cells, one cumulative journal,
  80 separate reviewer receipts, 160 synthetic claims checked. No live evidence.
- `SECURE_SETUP.md`: exact hosts/headers, approved budget and current-process
  binding booleans. The parent owns user key setup/publication. Absence in this
  old running process does not describe the Personal vault inventory.

Spend observed in this implementation task: **0 provider calls, $0 external
experiment spend**. Sole live owner is fresh task
`01a0f3b8-6abe-775b-bfea-5102185b80ce`, new catalog version
`6abd5bd3b9f081a18f7946316d12ad95`; parent reports all3 bindings present there.
No duplicate key setup or paid call is allowed from this old task.

Limits of handoff: `live_http.py` is a guarded callable HTTP seam, not a complete
live runner CLI. It requires a pinned tokenizer callback, immutable live plan,
configured-access receipt and canonical paid admission capability. Parent-side
real semantic grading and invoice reconciliation are still needed. The adapter
tests do not certify Network-secret substitution of Parallel's `x-api-key`.
Authenticated pilot validation belongs exclusively to the fresh execution owner.
No provider winner can be reported from synthetic fixtures.

## Runnable live CLI, final narrow review

`live_runner.py` now completes a mocked pilot and remaining18 with actual20 public
inputs, supported `/v1/responses/input_tokens`, exact source/count/synthesis
receipts, canonical journal, current owner/catalog/access/approval gates and no
provider calls from this implementation task. Missing count metadata blocks before
search. Empty or altered pilot receipts cannot admit remaining18. Token count
failures retain exposure and prevent inference. All52 focused tests passed;
changed-directory Ruff and `git diff --check` passed. Earlier nine repository
sentinel checks remain valid: no production file changed.

Independent `gpt-6.1-sol` review reproduced two CLI findings; both were fixed and
reviewed with23 focused runner+HTTP tests passing. Mocked pilot reservations:
$0.49284; full80 cells:$4.92840. Real provider behavior/billing and count endpoint
pricing remain unverified; actual semantic grades await isolated parent oracle.
The optional admitted pilot header probe adds no unmetered auth request or proxy
bypass and preserves a known-unsupported header capability refusal.

## Case10 retained public-query recovery

Parent reports37 live results and a held Parallel Advanced case10 HTTP200 response
at$2.28551 reserved exposure. This implementation task has no access to the fresh
executor's live artifacts; it received the sanitized public BMW article URL and
language parameter/ten-result shape through the parent. No live calls were made.

The query exception is limited to a public BMW article locale parameter, with
unknown query keys/credential values/userinfo/fragments/redirects still refused.
Offline reconciliation verifies the whole retained envelope digest supplied by
the sole owner, original frozen request and binding grant, scope, reservation and
uncertain state before appending completion. The pinned recovery receipt records
the original d12d1ab8 code fingerprint and exact patched fingerprint; it permits
only that code change, preserving original request keys and cumulative exposure.

58 focused hermetic tests pass, including a reproduced37-cell pause at$2.285510,
zero-call/idempotent adoption with no secret reads or new reservations, and mocked
completion to80 cells with exactly240 total dispatches and$4.92840 reservation.
Wrong owner/hash/request/cell/grant, missing or tampered raw responses, credential
query values, altered scope and subsequent code changes fail closed. Ruff and
diff checks pass. Independent GPT6.1Sol review reports no must-fix findings in
the recovery implementation; all28 then-current runner/HTTP tests passed.
Two additional negative reconciliation regressions cover tamper/binding and
missing/private-query evidence. Paid resume awaits the parent's separate pilot
evidence audit, not this implementation task.
