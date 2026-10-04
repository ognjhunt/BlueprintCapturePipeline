# Optional Parallel FindAll daily caller

This integration supplies three application tools to the existing daily research
session: `blueprint_findall_create`, `blueprint_findall_status` and
`blueprint_findall_result`. It reuses the existing owner ledger, artifact namespace,
lease, caller receipts and canonical paid-resource admission contract. There is no
new server, store, scheduler, automatic provider job or credential reader.

The implementation is a reviewed integration seam. A source merge or portable
archive does **not** activate the production worker. The default factory has no
FindAll handler. Existing Exa expansion remains independently configured and keeps
its original registry and transport.

## Owner installation boundary

After the sole release owner confirms the current source, release window and
required authority, their existing API factory may attach a typed handler:

```python
from tools.daily_research.findall import FindAllApplicationTools

# All dependencies are supplied by the trusted existing owner path. The client
# is already privately configured; this code does not obtain its credential.
api.findall_application_tools = FindAllApplicationTools(
    ledger=ledger,                         # SAME object passed to Runner/Consumer
    client=preconfigured_admitted_client,  # AdmittedFindAllClient
    grant_provider=existing_exact_request_admission_broker,
    current_authority=existing_fresh_authority_check,
    assert_current_lease=existing_fresh_lease_assertion,
)
```

`assert_current_lease` must assert the held backend fence and return literal
`True`; it must never acquire another lock. For Firestore, wrap the existing
`ledger.bridge.call("assert_lease")` operation, returning `True` only after its
success. The outer daily caller owns acquisition and release.

The admission broker receives the prepared exact request, including its objective,
conditions, generator, full body, operation identity, stated maximum exposure and
allocation binding digest. It must use the existing authorized broker and current
shared allocation; historical approval references are insufficient. This package
adds no issuer allowlist entry or broker implementation. If that owner path cannot
supply a valid grant for `parallel_findall`, create stays blocked.

The authority callback must freshly check stop state, current task/phase scope,
original deadline, provider disclosure permission, pricing/exposure evidence and
shared remaining allowance, including model/tool/provider charges and unknown
holds. It runs before the broker and again before POST. Read scopes carry their
operation and exact retained run ID. A callback returning anything except `True`
fails closed. Do not infer a hard billing cap from `maximum_cost_usd`: it is a
locally bound admission estimate, not a native FindAll request field.

Preflight advertises the optional definitions only when that typed handler is
installed. The exact definitions digest and `parallel-findall-v1` profile are
frozen into session intent and checked during research, QA and repair. Installing
a handler cannot enlarge an already running unpinned session. Ordinary preflight
and tools remain unchanged when the handler is absent.

## Durable and evidence behavior

Before any injected caller lifecycle write, the same held lease and owner row's
durable call, FindAll and Exa claim fields must match. Stale rows are refused before
error handling can persist them. Creation consumes one exact durable claim before
the only `POST https://api.parallel.ai/v1beta/findall/runs`; uncertain replies never
trigger another start. Known IDs survive receipt/store failures. Status/result
GETs are restricted to IDs retained in this daily owner's FindAll journal.

Result snapshots retain raw candidates, duplicate/conditional rows, citations,
basis, reasoning, provider status and unknown fields. Immutable artifact hashes
and read dates accompany them. `matched` remains discovery input; independent
`blueprint.lead-verification.v1` assessment and the existing review/promotion gate
decide qualification. No snapshot is rewritten to imply verified evidence.

## Exact live handoff and readback

1. The sole existing release owner merges owner-ledger PR #2570 at its reviewed
   head, then reviews/merges this compatible runtime port and builds the immutable
   standalone archive through the existing release channel. Recheck current head
   and active leases before promotion; preserve concurrently merged Exa/workflow
   changes. Older pinned operator scripts describe their historical packages and
   must not be used as a new package's approval or verifier.
2. The user/owner privately confirms or configures `PARALLEL_API_KEY` on the actual
   worker's supported secret binding. Inspect only binding name and presence.
   Local Mac presence proves nothing about production. Never copy saved Task MCP
   credentials, extract a vault value, paste a key into chat, or create a new key.
   Credential configuration requires its own action-time authorization.
3. Obtain explicit approval for these three application functions and outside-
   sandbox Parallel API disclosure of their objective/evidence. Sandbox networking
   remains disabled. Any necessary security/configuration change is a separate
   reviewable owner action; this setup performs none.
4. Supply the existing fresh exact-request admission broker/current-authority
   callbacks only after the shared remaining research allocation is verified.
   Exa's allocation, a prior console allowance or an available credential does not
   authorize Parallel API spending. Setup and readiness readback require no paid
   provider or model call. A paid live proof needs a separately approved exact
   request and fresh grant; do not invent an extra allowance.
5. Read back installed source commit, archive SHA256, allowlisted file hashes and
   actual imported module paths; then inspect `runtime_status(api)` on that actual
   worker and its newly admitted preflight/session registry. Record the three tool
   names and frozen definitions digest, identical ledger/lease binding and worker
   credential NAME/PRESENCE only. Do not call the grant broker or start a job as a
   readiness probe. Bind this readback to the actual release/process, since local
   metadata intentionally reports `production_binding_verified: false`.

The non-secret local metadata command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m tools.daily_research.findall
```

It imports the handler and checks environment membership only; standalone use
without `api` truthfully reports that no callable handler is installed. It never
reads a credential value or contacts a provider. The portable archive projects
the canonical six-file stdlib import closure under `blueprint_pipeline/`; no
SDK install or provider credential is needed to import the optional handler.

Official contract: [Create FindAll run](https://docs.parallel.ai/api-reference/findall/create-findall-run).
