# Parallel FindAll daily research caller

The existing daily research session can use `blueprint_findall_create`,
`blueprint_findall_status` and `blueprint_findall_result`. They share the current
owner ledger, fenced lease, immutable artifact namespace and application-tool
receipts. No additional worker, schedule or automatic first contact is created.

## Production binding and admission

The fenced provider's existing factory installs the typed FindAll handler only
when the actual worker has a valid `PARALLEL_API_KEY` binding. The value goes
straight to the pinned Parallel client; metadata inspects only its name and
presence. Missing or invalid bindings leave ordinary research available and do
not advertise FindAll tools. A local binding does not establish worker readiness.

A binding alone does not admit a paid create. `allocation.py` requires the
owner-directed, verified paid expansion direction to name `findall`. Each daily
row freezes that direction, source commit, expiry and combined per-run allowance.
Exa and FindAll claims debit the same allowance. A smaller live allowance lowers
headroom; a larger one never enlarges a previously admitted row. Each request
reserves its whole `maximum_cost_usd`, subject to the current per-start bound.
Uncertain, completed or cancelled requests retain their reservations until
verified provider billing supports a different amount; cancellation is not a refund.

The canonical exact-request issuer binds the prepared body, pricing version,
operation identity and exposure. The owner journal checks the actual held lease,
phase, stop state, original deadline, source commit and live direction before the
claim. The existing Firestore transaction checks the same paid fences while
committing the claim. Immediately after that commit, before the one POST, the
journal freshly checks authority again, excluding only its exact matching claim
from prior debits. A failed final check consumes no provider request and retains
the nonreplayable claim conservatively.

`maximum_cost_usd` is a local admission exposure, not a native FindAll billing-cap
field. Prices are versioned estimates from the official pricing contract. The
owner approves disclosure of the request objective and evidence to Parallel
through the existing paid direction; historical receipts or available credentials
do not create additional authority. No paid readiness canary is required.

## Session and evidence contracts

Preflight advertises the three functions only with the typed handler installed.
The exact definitions digest and `parallel-findall-v1` profile are frozen in the
session intent and checked during research, QA and repair. Installation cannot
add tools to an already running unpinned session. For broad site discovery, enumerate with FindAll early before deep individual
Perplexity investigation; retain the sourced backlog, then assess a subset.
Requested matches are not returned, deduplicated or qualified counts. Creates are restricted to the
original research phase before final output and QA; free reads are restricted to
provider IDs retained by the same daily owner.

A durable claim precedes the sole `POST /v1beta/findall/runs`. Ambiguous outcomes
never permit a replacement create; known provider IDs survive receipt failures.
The runtime cancels potentially active runs when research ends, the original
deadline expires, or fresh live authority is lost through a brake, removed source,
changed source commit, expiry or lowered allowance. Cancellation attempts are
durably claimed and bounded; unknown IDs and unconfirmed outcomes remain visible
with their reservations held.

Every status/result snapshot is retained as ONE immutable file
(`blueprint.findall-snapshot.v2`): one write and one readback, with its SHA-256
recorded. A snapshot above 7 MiB is refused before any write with
`findall_snapshot_too_large`; the run and its reservation stay recorded. Only the
provider GET and its encoding run under the 120 s read alarm; store calls never run
under an alarm, so a timeout cannot break the store connection. Small snapshots
return the complete raw object. Large ones are served as 24,000-character pages
sliced from the retained file: the response returns `json_fragment`, `page`,
`page_count`, `next_page`, the whole-snapshot SHA256 and the receipt. Continue with
the same `findall_id`, `receipt_sha256` and `page`. Continuations re-read that file
without a new provider request. Concatenating fragments in page order reconstructs
the exact JSON. Each response stays under the tool-output ceiling, and the session's
tool budget bounds how many pages a run can read. Export verifies the file's hash
and byte count. An old multi-part receipt is refused.

Raw candidates, duplicate/conditional rows, citations, basis, reasoning, provider
status and unknown fields remain discovery evidence. `matched` does not imply a
verified or qualified lead. Independent `blueprint.lead-verification.v1` assessments
and the existing review/promotion gate determine qualification.

## Release and worker readback

Build the immutable standalone archive from the final reviewed Pipeline commit.
It projects the canonical seven-file standard-library closure under
`tools/daily_research/pipeline_runtime/blueprint_pipeline/`; no SDK installation is
needed for that optional import. Preserve current merged research and WebApp
changes when updating the exact archive and documentation pins.

Before deployment and control repin, obtain a fresh atomic snapshot showing no
active research, QA, repair or publication rows under the existing held lease.
The fence manifest and Firestore control must identify the same source commit.
Read back the installed archive SHA256, every allowlisted file hash, actual import
paths, worker binding name/presence, typed handler, frozen registry digest and
saved-agent definition. Record private operational evidence in the company's
existing evidence store. Do not infer readiness from a local environment or a
source merge. New credentials or security changes require separate authorization.

The non-secret local metadata command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m tools.daily_research.findall
```

Without an actual API owner object this truthfully reports no installed callable
handler and `production_binding_verified: false`. It never reads credential
values or contacts a provider. Production proof and business yield require actual
worker readback and the next authorized natural run respectively.

Official contracts: [create](https://docs.parallel.ai/api-reference/findall/create-findall-run),
[result](https://docs.parallel.ai/api-reference/findall/findall-run-result),
[cancel](https://docs.parallel.ai/api-reference/findall/cancel-findall-run),
[pricing](https://docs.parallel.ai/getting-started/pricing).
