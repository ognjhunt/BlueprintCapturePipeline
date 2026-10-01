# Perplexity Fast application tools

The selected production default is `search_provider=perplexity-fast-v1` with
`discovery_profile=adaptive-sites-v1`. The disabled example is
`perplexity-daily.config.example.json`. This is an agent-owned discovery loop:
the existing GPT6.1Sol researcher chooses queries, contradictions, source reads
and follow-ups across a defined task/industry/region hypothesis scope.
Recurring activation is disabled and its soft target is pending an explicit owner
budget decision (`soft_target_usd=null`, pending budget-authority reference).
One-time test authorization is separate from recurring authorization.
There is no count-based stopping rule or outer-runner prospect/query shortlist.
Ten findings do not mean done; retain fifteen or fifty defensible findings when
the evidence supports them. Coverage and diminishing returns govern completion.
Budget/time interruption and access blockage remain explicitly incomplete.
Existing CRM duplicates and
robot deployments never count toward the target. Incomplete coverage is honest.

## Exact bindings and activation gate

- Application worker only: owner-managed `PERPLEXITY_API_KEY`
- API: `POST https://api.perplexity.ai/search`, `search_type=fast`,
  `search_context_size=high`, `max_results=1..20` (default 10)
- Existing `OPENAI_API_KEY`, Default project
  `proj_F2tFJuxLaovJru8RrtXRaqNj`, saved agent
  `agent_5a01ec367d1042ef8632bb5f2e6af8b4919909d2abed48ed95`,
  `gpt-6.1-sol`, medium reasoning, subagents off, standard service tier
- Existing hosted template and four digest-checked instruction files; sandbox
  network remains disabled
- Existing canonical Firestore control and durable lease; config adds only the
  selected search provider and already reviewed adaptive envelope

Preflight checks secret presence without exposing its value and refuses before
session creation if it is absent. Presence does not prove provider access.
Binding presence alone does not prove a successful provider request or profile
activation. Never copy a comparison-executor key,
create a credential, alter grants, or mount a key in the hosted environment.
Any new persistent credential binding needs the owner's separate approval.

## Agents API integration

New session requests retain `agent_id` and use a session-only inline agent
override: exact `blueprint_search` and `blueprint_read_source` function definitions,
standard service tier, and the pinned base instructions with the selected tool
instructions appended. Tools replace the saved tool list for that session. The
saved agent/template is unchanged. Existing sessions cannot have tools changed.
The retained failed October 1 intent, original artifact, cleanup guard and
disabled same-session adaptive test are untouched. They cannot be converted into
this profile, reset or silently restarted.

Both research and same-session QA respond only to currently pending
`function_call` required actions with exact turn/call/name/argument bindings.
Before executing a call, persist its immutable request and attempted state in
the dated row. Before returning a result, persist exact output/error in an
immutable `DATE-tool-CALLID.json` file and its digest/length/pointer in the row.
This keeps full source evidence from inflating lifecycle rows. The profile's QA
input is likewise saved as immutable `DATE-qa-input.json`; exports retain and
verify these bytes. A file saved before a row-pointer crash is recovered without
another search. Compact row receipts preserve requests, attempts and accounting.
A lost search POST reply is not retried. After restart, submit the saved result
with the original tool-result idempotency key while that action is still pending.
Unknown execution becomes a visible no-replay error. Render rechecks the lease,
enabled flag and pinned search profile before execution and result submission.
Stop/deadline checks preserve the pinned shared research/QA watchdog.

No native web-search fallback is enabled. API errors, missing credentials,
unsupported sources and evidence size ceilings are visible gaps. This avoids an
unobserved second search provider or search fee. An application HTTPS source
reader handles primary static pages without granting sandbox network access:
DNS must resolve exclusively to public addresses, connections pin the validated
address while verifying the original TLS hostname, and every HTTPS redirect is
revalidated. No credentials, cookies or authentication headers go to sources.

## Evidence and cost

Search results retain the entire usable provider response including passages,
URLs, title, dates, last_updated, request and raw response digest. They are
provider-extracted passages, not proof of complete primary-page reading. Source
reads retain full extracted static text, metadata, links, checked time,
Last-Modified and raw-byte digest. PDFs, unsupported encoding, JS-only pages,
HTTP failures and oversized responses remain explicit gaps. No content is
silently truncated; absent dates are never invented. The already reviewed
evidence skill and QA/publishing gates remain authoritative.

An absolute 15-second application-call alarm covers DNS, slow-drip responses,
redirects and parsing, bounded further by the original phase deadline. Deadline
and stop checks are repeated after every lease/control read before mutations.

Transport response/tool-result ceiling: 500,000 bytes; cumulative immutable tool
evidence ceiling: 5,000,000 bytes; compact call records and initial intent each
have a 1,000,000-byte ceiling, reserving room under the existing 8 MiB store
boundary for later packets and delivery plans. Prospective row size is checked
before persistence; the shared application call ceiling is 500.
The concise new-profile publication packet is limited to 500,000 bytes; an
oversized packet blocks before QA/publication while retaining the complete raw
artifact. Every subsequent profile row also refuses a prospective 7,000,000-byte
overflow before durable state changes or external publication. The existing
native lanes are unchanged.
These are
resource safety ceilings, not research quality/search quotas. If reached, the
research must report searched scope, source coverage, duplicates/rejections,
unresolved promising branches and evidence-based or interrupted stopping reason.
No exhaustive global-market claim is supported. Standard search costs
are never substituted: Fast is estimated at $0.001 per successful request, with
attempts counted conservatively and unsettled billing explicit. Model, search
and hosted-environment costs are additional. The legacy native profile retains
its historical $1 soft planning target. The new profile requires a chosen positive
soft target and non-pending recurring-budget authority before activation; changed
budget authority fences further tool calls. No hard total-dollar cap is claimed.

Sources: [Perplexity Fast Search](https://docs.perplexity.ai/docs/search/fast-search),
[Search request/response schema](https://docs.perplexity.ai/api-reference/search-post),
[Agents functions](https://developers.openai.com/api/docs/guides/agents-api/tools/functions),
[session overrides](https://developers.openai.com/api/docs/guides/agents-api/configuration#override-settings-for-one-session)
