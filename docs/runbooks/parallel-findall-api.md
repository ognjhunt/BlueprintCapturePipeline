# Parallel FindAll API setup for Blueprint agents

This optional utility prepares the official FindAll request offline, checks
authentication with a read-only list endpoint, and reads an existing run's status
or result. A separate Python adapter supports future grant-dependent creation
and explicitly selected cancellation. It adds no dependency, server, scheduled
task, deployment, issued grant, or pipeline hook; no provider operation was run
during setup. It uses Blueprint's existing
`safe_outbound_http` boundary with the Parallel HTTPS host pinned, redirects
refused, a timeout, and a response-size cap.

Scope: the user's explicit “Setup parallel find all api” instruction authorizes
this non-secret agent utility. The example relates to ADP-010 partner discovery,
but no observed ADP gate blocker or day-gate completion is claimed. Candidate
organizations require human admission; provider findings and citations are
research leads, never capture, rights, task, or physical-outcome truth. The
separate active provider benchmark and console comparison remain owner-managed.

## Verified local setup (2026-10-03)

- The local repo virtualenv reports Python 3.12.11; default `python3` on the
  inspected Mac reports 3.14.6, outside the repo's `>=3.10,<3.13` range. Use a
  supported interpreter without changing the active owner's environment.
- `PARALLEL_API_KEY` was present by **name only** in the local Mac command
  process environment, observed with both login and non-login shells. No value was
  inspected, imported, reused, or sent. Validity, organization, scope, and the
  user's permission to use that binding remain unverified. This is not evidence
  of a saved config entry or a production service binding: neither was inspected.
- Neither `parallel-web` nor `parallel-web-tools` was installed in the inspected
  repo runtime. No dedicated pipeline FindAll API binding was found.

## Run without a key or network

From a checkout with the utility and a supported local virtualenv:

```bash
export BLUEPRINT_FINDALL_PYTHON="$PWD/.venv/bin/python"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m blueprint_pipeline.parallel_findall check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m blueprint_pipeline.parallel_findall prepare docs/examples/parallel_findall_spec.json
```

`check` tests environment membership only; it does not validate authentication.
`prepare` validates the five required JSON fields plus optional official scalar
`metadata`, preserves that metadata, and returns a request envelope
with `body_json`, `POST https://api.parallel.ai/v1beta/findall/runs`, required
`x-api-key` header name, and `execution_authorized: false`. It makes zero calls
and does not read credentials. The example is reviewable input, not a launched
search. `preview` is also a real run and must not be treated as offline setup.

## Secure user authentication check (no run ID)

An authorized user can check their own existing API key privately:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m blueprint_pipeline.parallel_findall auth-check --prompt-key
```

This makes exactly one `GET https://api.parallel.ai/v1/monitors?limit=1`, using
the documented `x-api-key` authentication. It creates no monitor or run, starts
no research, and reports only authentication facts; monitor contents and IDs are
not returned to the caller. A successful response proves access to this list
endpoint, not FindAll entitlement, billing readiness, or a spend grant. A 401/403
or other failure remains a typed error. No authentication probe was executed
against Parallel during setup.

The Account API's apps/balance reads require an account OAuth access token,
not a standard API key; this utility does not request that wider grant. The
standard API docs do not list a FindAll run-list endpoint, so this documented
Monitor GET avoids inventing one or requiring an existing FindAll run ID.

## Secure user handoff for an existing run

An authorized user may enter their own existing API key privately in a terminal:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m blueprint_pipeline.parallel_findall status findall_REPLACE_WITH_OWN_RUN --prompt-key
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m blueprint_pipeline.parallel_findall result findall_REPLACE_WITH_OWN_RUN --prompt-key
```

The prompt refuses non-terminal input, does not echo or save the key, and uses it
only for the selected GET. These commands are for the user's own terminal,
not an agent credential-retrieval workflow. Do not paste a key into chat, a command argument, a
tracked file, or an agent tool call. Do not unlock the user's screen or read a
saved Task MCP credential. A saved MCP credential does not authorize API reuse.

After the user explicitly authorizes their existing runtime binding for the
selected API read, `--use-runtime-key` is the alternative to private terminal entry.
The client never discovers or loads a credential automatically. Results preserve
the provider run, pending and non-matching candidates, reasoning, and citations;
they are snapshots, not proof of a completed search or admitted partner.

## Guarded future creation (Python controller seam)

`parallel_findall_execution.AdmittedFindAllClient.create()` supplies the missing
POST transport. It accepts an explicit key, reviewed spec, stable Blueprint
`operation_id`, decimal USD ceiling, opaque `PaidResourceAdmissionGrant` for
`parallel_findall`, and the owning controller's durable submission journal.
There is no paid CLI, grant issuer, or production caller in this change. Adding
the resource-class name to the canonical admission vocabulary does not issue a
grant or authorize a run. The existing issuer allowlists stay unchanged.

Offline review is usable without credentials:

```python
from blueprint_pipeline.parallel_findall_execution import prepare_submission

prepared = prepare_submission(
    reviewed_spec,
    operation_id="BLUEPRINT_OPERATION_REPLACE_WITH_APPROVED_ID",
    maximum_cost_usd="1.00",
)
```

This returns the exact request body, versioned pricing facts, maximum-cost
estimate and `allocation_binding_digest`, with `execution_authorized: false`.
The digest binds objective, conditions, generator, match limit, metadata,
operation ID, ceiling and pricing snapshot. The client requires that same digest
on an opaque grant from the **existing** admission chokepoint. Missing, forged,
wrong-class, unbound or mismatched grants fail before claiming or calling HTTP.
The controller must check current user authority, organization, disclosure,
expiry and remaining shared allowance, and reverify pricing before issuing that
exact grant; a key or an `allow_paid` boolean never supplies authority.

The checked 2026-10-03 pricing snapshot is fixed + per-match: preview
$0.10 + $0; base $0.25 + $0.03; core $2 + $0.15; pro $10 + $1.
The estimate uses the requested match limit, with no enrichment/extension/ingest
calls. It is an estimate under that pricing snapshot, **not observed billing or
a provider-enforced dollar cap**. A ceiling below it is refused. This creates no
budget authority and does not borrow the active console comparison's allocation.

`FindAllSubmissionJournal` is an interface to the owner's existing one-start
control, not a new database or approval framework:

1. `claim_submission(prepared)` must atomically check current authority and
   durably record `submission_unresolved` under the operation ID before returning
   literal `True`. Reject every already-claimed operation ID, including changed
   bodies, restarts and prior failures. Store no credential.
2. The adapter makes **one** POST to `/v1beta/findall/runs`; it never retries.
3. `record_created(operation_id, allocation_binding_digest, run)` must durably
   retain the raw returned run receipt and provider ID. The adapter records a
   syntactically valid provider ID even if later status/generator validation fails.
   On receipt-write failure, `FindAllSubmissionUnresolved.findall_id` retains that
   ID so the owner can recover it. No uncertain claim is released automatically.

No create idempotency key or FindAll run-list endpoint appears in the inspected
standard API contract. A timeout, HTTP failure or malformed receipt therefore
leaves an unresolved claim requiring owner/provider reconciliation. Do not retry
based on a guessed absence or treat a Monitor read as FindAll reconciliation.

The Python call after **separate action-time authorization and controller wiring**
is `client.create(reviewed_spec, operation_id=..., maximum_cost_usd=...,
paid_resource_admission_grant=exact_grant, journal=owner_journal)`. This setup did
not wire or execute it. Production integration/deployment and any later run remain
with the sole release owner, distinct from the owner-managed console comparison.

`AdmittedFindAllClient.cancel(findall_id)` sends one empty-body POST to
`/v1beta/findall/runs/{findall_id}/cancel`, requiring the documented 204 response.
Use the explicit status read afterward to reconcile `status.is_active` and
terminal state. A 409 means already terminated and is not a cancellation receipt;
read status. Cancellation does not refund completed work. The owner retains
responsibility for monitoring the admitted operation and cancellation when its
approved deadline or allowance closes. No background watchdog, implicit
cancellation, extension, or enrichment is installed by setup.

## Remaining runtime handoff

The code and synthetic checks do not establish a production binding, key
validity, FindAll entitlement, budget grant, deployed caller or paid success.
The authorized user first performs the private authentication check described
above. A Monitor success remains only Monitor-read scope. A later authorized
read of the user's existing FindAll run can verify that run's read scope; it
still does not prove create authority. No live or paid proof was performed here.
The owner then wires the existing authority/one-start controller to this journal
interface and binds the runtime credential through the supported private flow.
No new API key, storage grant, spending or infrastructure was created.

## Verification/scoring consumer boundary

`FindAllClient.result(findall_id)` returns the unchanged provider snapshot shape
(`run`, `candidates`, `last_event_id` and unknown fields), including basis,
reasoning, citations and provider statuses. Consumers may separately translate
candidate evidence for `tools/daily_research/verification.py`; keep the raw
snapshot immutable. Provider `matched` does not mean verified, qualified,
commercially admitted, rights-cleared, or robot-ready. Verification/scoring,
full-cohort comparisons and `review.lead_verification` are separate consumer
contracts, not client outputs.

Official alternatives are the Python SDK (`parallel-web`,
`Parallel().beta.findall.create(**body_json)`) and CLI (`parallel-web-tools`,
`parallel-cli`). They are not installed by this change. The supported CLI's
`parallel-cli login --device` is a user-completed authorization path for headless
environments; use it only after action-time approval of authentication/storage
effects. Its `findall run --dry-run` **calls ingest**; it is not an offline
verification command and was not run here.

## Verification

```bash
env -u PARALLEL_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m pytest -q tests/test_parallel_findall.py tests/test_parallel_findall_execution.py
```

Tests block networking, replace credential input with synthetic fixtures, verify
name/presence-only checks, offline validation, exact GET paths and headers,
transport policy, response identity/shape, preservation of candidates/citations,
secret-free failures, explicit credential-source selection, and no-run-ID
authentication that does not expose monitor data. Execution fixtures check exact
grant binding, claim-before-POST ordering, restart/changed-body duplicate refusal,
ambiguous outcomes, persisted IDs and cancellation responses. No scientific,
production-authentication, or paid-execution claim follows from these tests.

Official documentation checked on 2026-10-03:

- [FindAll quickstart](https://docs.parallel.ai/findall-api/findall-quickstart)
- [Create request contract](https://docs.parallel.ai/api-reference/findall/create-findall-run)
- [Parallel CLI, authentication and dry-run semantics](https://docs.parallel.ai/integrations/cli)
- [Monitor list read and API-key contract](https://docs.parallel.ai/api-reference/monitor/list-monitors)
- [Account API token boundary](https://docs.parallel.ai/integrations/account-api)
- [Versioned price reference](https://docs.parallel.ai/getting-started/pricing)
- [Cancel request and 204 response](https://docs.parallel.ai/api-reference/findall/cancel-findall-run)
- [Run lifecycle and cancellation accounting](https://docs.parallel.ai/findall-api/core-concepts/findall-lifecycle)
