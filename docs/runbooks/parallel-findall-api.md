# Parallel FindAll API setup for Blueprint agents

This optional utility prepares the official FindAll request offline, checks
authentication with a read-only list endpoint, and reads an existing run's status
or result. It adds no dependency, server, scheduled task,
deployment, paid operation, or pipeline hook. It uses Blueprint's existing
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
`prepare` validates a minimal five-field JSON spec and returns a request envelope
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

## Paid creation is a separate action

This utility deliberately has **no create/ingest/extend/enrich/cancel method**.
No new API keys, spending, infrastructure, permissions, or config grants were
created. No live authentication or paid proof was performed.

For a later approved run, review the prepared `body_json`, exact objective,
generator, match limit, permitted disclosures, organization/billing context,
spend ceiling, and cancellation responsibility with the sole local owner. Use
the existing canonical paid-resource admission rather than turning this reader
into a spend-gate bypass. This setup grants no permission to start another
benchmark or reproduce the existing console job. Production integration or
deployment must be coordinated with that owner first.

Creation is intentionally prepare-only because the inspected canonical allocator
and `paid_resource_admission` have no Parallel FindAll resource/grant binding,
pricing envelope, or cancellation lifecycle. A boolean such as `allow_paid=True`
would not supply those contracts. Adding a runnable POST here would create a new
paid path outside the existing admission boundary. The minimum future execution
change is an owner-approved Parallel grant-gated adapter that accepts the exact
prepared `body_json`, budget/disclosure authority and runtime key capability,
performs a single create attempt without automatic retries, reconciles ambiguous
creation, records the returned `findall_id`, and owns cancellation/terminal
reconciliation. Writing that guarded adapter is possible once its grant contract
is agreed; this setup neither invents nor grants the missing authority.

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
env -u PARALLEL_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$BLUEPRINT_FINDALL_PYTHON" -m pytest -q tests/test_parallel_findall.py
```

Tests block networking, replace credential input with synthetic fixtures, verify
name/presence-only checks, offline validation, exact GET paths and headers,
transport policy, response identity/shape, preservation of candidates/citations,
secret-free failures, explicit credential-source selection, and no-run-ID
authentication that does not expose monitor data. No scientific,
production-authentication, or paid-execution claim follows from these tests.

Official documentation checked on 2026-10-03:

- [FindAll quickstart](https://docs.parallel.ai/findall-api/findall-quickstart)
- [Create request contract](https://docs.parallel.ai/api-reference/findall/create-findall-run)
- [Parallel CLI, authentication and dry-run semantics](https://docs.parallel.ai/integrations/cli)
- [Monitor list read and API-key contract](https://docs.parallel.ai/api-reference/monitor/list-monitors)
- [Account API token boundary](https://docs.parallel.ai/integrations/account-api)
