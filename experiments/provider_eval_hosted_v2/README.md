# Private hosted research cohort v2

This is a separate, offline-tested comparison protocol, `hosted_agent_research_v2`.
It uses the actual Agents API managed harness with `gpt-6.1-sol` in the existing
Default project `proj_F2tFJuxLaovJru8RrtXRaqNj`. It does not change the production
research agent, create persisted agent definitions, or relabel earlier scores.
No hosted session or provider request has been executed from this implementation.

`bounded_adaptive_v1` used direct Responses calls: `/v1/responses/input_tokens`
and `/v1/responses`, with a Python search/assessment/optional-follow-up/synthesis
loop. Its evidence prefixes were bounded at normalization and model-input
construction. The managed agent in this protocol chooses its own research loop;
complete accepted evidence remains available for relevant-section inspection.

## Hosted architecture and evidence

The existing repository Agents runtime supplies durable session ownership,
required-action reconciliation, terminal turn/item validation and cancellation.
This experiment subclasses it without editing production files. Its wire routes
are `POST /v1/agents/sessions`, session/turn/item GETs, and
`POST /v1/agents/sessions/{session_id}/events` with
`agent.session.input.tool_result`, the original `turn_id`, and `call_id`.
Authentication uses the existing Bearer binding, `OpenAI-Project`, and
`OpenAI-Beta: agents=v1`. Session creation uses an inline agent definition,
`openai_hosted`, a small container, explicit disabled sandbox networking, and
disabled multi-agent delegation.

The initial workspace contains the frozen public case, trusted UTC run date,
the common prompt and full source files with provenance. The research cutoff
remains September 30, 2026. A frozen file snapshot keeps restart requests stable.
`list_evidence`, `find_evidence` and offset-based `read_evidence` expose complete
later tool evidence; the managed agent can inspect files using its hosted shell.
No fixed text prefix hides a retained tail. File/API size limits produce explicit
gaps rather than silent truncation. Untrusted retrieved text cannot authorize
actions or change the clock.

Each of the four arms has its own evidence and operations directory. Only
completed, corrected native v1 searches from that same case and arm are reused,
after validating the original request and completed raw-envelope hashes.
Truncated diagnostic searches, warnings and other arms' sources are excluded.
Valid reused searches count against the same three-search opportunity. Importing
does not modify the original aggregate journal or reserve their cost again.

The public 20-case input remains `../provider_eval_recovery/real_public/inputs.parent-message.json`,
SHA256 `cac8d7a31aea2ad1c2e5a47ea37e434abcaae4b4910ba4afc8f81dd11c404ff3`.
Its provenance is parent-message transfer; this does not claim successful Library
materialization. The reviewer oracle remains parent-side and unavailable to this
agent or the providers. The subsequently approved one-case soft-budget pilot has
a separate receipt and executable entry point; see [SOFT_PILOT.md](SOFT_PILOT.md).

## Comparable application tools

The agent may choose at most three native searches and three public source
fetches per arm. Parallel Fast/Advanced use short printable entity-first keywords
separate from their task objective. Perplexity Fast/standard use their native
Search API request contracts. Ordinary product names such as `Scan&Sand` are
valid typed JSON; query arguments never enter a shell. Both providers have the
same task criteria and tool opportunities, with no alternate provider or native
OpenAI search fallback.

Provider HTTP uses existing bindings only: Parallel `x-api-key` to
`api.parallel.ai/v1/search`, Perplexity Bearer to `api.perplexity.ai/search`.
The main $10 ledger reserves fees plus the existing extras allowance before
dispatch, retains measured HTTP latency, and never automatically retries an
uncertain request. Completed replay checks the original aggregate journal and
envelope independently of mutable operation sidecars. Source manifests and file
identities are verified before reuse. Warning or invalid retained responses stop
the operation without claiming a refusal before its already completed dispatch.

Public-source fetching uses an application route, without keys, cookies or
redirects. It pins a public DNS address, preserves HTTP versus HTTPS and source
authority, and accepts complete UTF-8 HTML/plain/JSON bodies up to 2 MB. Unsafe
URLs are quarantined; known benign display normalization never upgrades a
lookalike domain into a verified vendor. PDFs, other encodings, redirected pages,
binary content and oversized pages currently remain explicit inspection gaps.
No fetching route is live-authorized by this draft.

## Budget and exact live blocker

The execution owner's reconciled STOP checkpoint is 235 completed calls, zero
inflight/uncertain calls, and **$3.934105 reserved**, leaving **$6.065895** of the
original $10 ceiling. This report is supplied evidence, not a claim that the
owner's filesystem is shared here. Its count/extras allowances remain held.

These are planning targets using cumulative 12,000 input / 2,500 output tokens
per root session, short-context maximum cache-write rates, one 20-minute small
container period per session, three searches per arm, search extras, and separate
independent review plus a counting allowance:

| Scope | Hosted sessions | Incremental target | Prior reserve plus target |
| --- | ---: | ---: | ---: |
| One case, four arms | 4 | $0.468980 | $4.403085 |
| Two cases, four arms | 8 | $0.937960 | $4.872065 |
| Twenty cases, four arms | 80 | $9.379600 | $13.313705 |

The smallest proposed hosted pilot is **one case across all four arms**, followed
by reconciliation before considering another case. The full matrix does not fit
the existing cap. No reduced matrix is silently selected or executed. Valid reuse
may save searches but is not credited before verifying the actual receipts.

These are **not hard bounds**. The documented hosted API exposes no per-session
token/dollar limit that caps the managed model loop. Task token targets and
application deadlines/cancellation are soft controls. Usage is best effort and
does not expose separate cache-write counts; final billing, count fees, extra
container periods, long-context pricing and taxes require reconciliation. Three
container periods would add $0.24 to the one-case planning target; that still does
not establish an upper bound.

The original `live_admission` and `hosted.py` CLI `--execute` still fail closed before key access,
session creation, model calls, or research requests. There is no override flag or
receipt claiming a fictitious API hard cap. An existing saved-agent GET200 proves
read access only. The official hosted quickstart requires application-key scopes
`api.agents.read`, `api.agents.write`, and `api.responses.write`; inference/write
access remains unverified here. Reuse existing authorized bindings only; this
draft neither creates a key nor requests broader permissions. The soft-pilot
entry point requires the tested public SDK pair OpenAI 3.22.1 / Agents SDK 0.22.3,
uses the real SDK for its read-only access check, and reuses the documented raw
REST transport for durable execution. The initial 2.45.0 SDK lacks `beta.agents`.

## Runnable offline commands

Run from the repository root after fetching this branch:

```bash
PYTHONPATH=src .venv/bin/python -m pytest experiments/provider_eval_hosted_v2/test_hosted.py tests/test_agent_execution_sessions.py -q
.venv/bin/ruff check experiments/provider_eval_hosted_v2
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.hosted --aggregate-root /workspace/provider-eval-private-live-20260930
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.hosted --aggregate-root /workspace/provider-eval-private-live-20260930 --prepare-case 1 --mode parallel_fast
```

Preflight reads the existing journal without resetting it. Preparation writes a
separate cohort's public evidence files and imports eligible retained sources,
with zero HTTP or paid calls. Repeat preparation for another explicitly chosen
arm if needed. `--execute` intentionally returns exit 2 and a precise blocker.
This is an offline integration handoff, not an authenticated hosted launch CLI.

`prepare_task` accepts an independently trusted admission; it never manufactures
one. Its file/input digests, exact source commit, project/model/tool identities
and deadline must be bound by the caller. Synthetic admission in tests is not
authorization. A live integration must account for outer Sol, managed model
usage, containers, all search attempts/extras and independent output review in
the same ledger. The separately approved soft-pilot route supplies that admission
through an explicit, digest-bound receipt and monitoring, without claiming a hard
provider cap or enabling arbitrary cases/sessions.

The managed final JSON and session/turn IDs are retained by the existing runtime.
It instructs the agent to write/read back `/workspace/outputs/answer.json`; direct
artifact-content downloading is not wired or claimed verified in this draft.
Permanent session deletion is explicitly blocked. No new answers have been
graded, and neither provider can be declared the winner from these mocks.

## Primary documentation

- [Agents quickstart and application-key scopes](https://developers.openai.com/api/docs/guides/agents-api/quickstart)
- [Hosted environment](https://developers.openai.com/api/docs/guides/agents-api/environments/openai-hosted)
- [Workspace files](https://developers.openai.com/api/docs/guides/agents-api/environments/files)
- [Application function tools](https://developers.openai.com/api/docs/guides/agents-api/tools/functions)
- [Sessions and durable events](https://developers.openai.com/api/docs/guides/agents-api/sessions)
- [Usage and billing limitations](https://developers.openai.com/api/docs/guides/agents-api/observability)
- [Session create reference](https://developers.openai.com/api/reference/resources/beta/subresources/agents/subresources/sessions/methods/create)
- [OpenAI model/container pricing](https://developers.openai.com/api/docs/pricing)
- [Parallel search contracts](https://docs.parallel.ai/api-reference/search/search) and [pricing](https://docs.parallel.ai/getting-started/pricing)
- [Perplexity search contracts](https://docs.perplexity.ai/api-reference/search-post) and [pricing](https://docs.perplexity.ai/docs/getting-started/pricing)
