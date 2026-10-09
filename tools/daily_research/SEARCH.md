# Perplexity Fast application tools

> Owner direction, 2026-10-09: application dollar budgets and arbitrary call-count gates are retired. Historical dollar fields and capped requests remain immutable accounting and request-binding data. New research keeps the original source, scope, expiry, release, idempotency, byte and time constraints, and actual provider limits. This source change grants no activation, paid run or send authority. Older numeric allowance descriptions below are historical and do not govern admission.


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

Employer job postings are another source of site-linked physical-duty evidence,
not a prerequisite or proof of automation interest. The runtime and
[`saved-agent-discovery-policy.md`](saved-agent-discovery-policy.md) direct the
agent to retain posting identity, quoted duties, original dates and observed
application status, with unknown currentness explicit. Use existing evidence
fields for supported claims and concise findings for requisition/status/source
details and unresolved site joins; no new candidate schema or automatic job
scraper is introduced. Posting aliases/reposts and facility/task duplicates are
separate identities. All original evidence remains in the packet/duplicate
cohort; hiring-led conversation or evaluation improvement remains a hypothesis
until actual downstream outcomes are recorded.

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
  selected search provider and the owner-approved adaptive envelope: at most
  3600 seconds total (`runner.MAX_ADAPTIVE_RUNTIME_SECONDS`, owner decision
  2026-10-04). Production uses 3600 seconds with 900 reserved for QA, so
  research has 2700 seconds. Each row keeps the envelope it was admitted with;
  see [RENDER.md](RENDER.md#runtime-envelope)

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
Stop/deadline checks preserve the pinned shared research/QA watchdog: research
tools stop at the row's research deadline. For an ordinary daily row, QA, repair
and publication tools stop at its total deadline.

Known public-source tool failures expose JSON output `{ok:false,error:{code}}`
and a JSON error with the unchanged stable code, so a caller can inspect the
failure without parsing a bare error string as JSON. These failures provide no
source text, publication dates or successful-read evidence. Optional Exa skips
preserve their explicit reason and continuation guidance in both output and
error; an unverified allocation remains a skip with no native start. Existing
immutable tool-result receipts are returned byte-for-byte, never reformatted.

The October 4 disposition `BP-RESEARCH-IMPROVE-20261004-TOOL-ERRORS` corrects
future result formatting, preserving the original valid accounting skip and
failed source results. Its company-owned proposal and five original results are
retained at `gs://blueprint-8c1ca.appspot.com/operations/research/improvements/2026-10-04/BP-RESEARCH-IMPROVE-20261004-TOOL-ERRORS/cd1b2be2b3041fa35a7d849edb5ef86705d8cb1605a61878935b53b63be883b4/proposal-and-original-evidence.json`,
generation `1791119633053384`, 8,838 bytes, SHA256
`cd1b2be2b3041fa35a7d849edb5ef86705d8cb1605a61878935b53b63be883b4`.
This evidence reference changes no allocation, native-start authority or access.

The explicit `publication_profile=agent-owned-v1` additionally advertises
`blueprint_inspect_publication` and `blueprint_publish_research` when creating a
new session. After QA validates the research, that same saved session receives
the complete validated results and retained history. The agent inspects approved
destinations, selects full or concise presentation, uploads to each destination,
and receives structured transport errors and exact readback receipts. The worker
executes these requests and retains original results, decisions, claims and
receipts. A definitive initial Notion 400 validation rejection permits a revised
agent choice only after complete readback proves absence; unknown writes retain
their claims and remain observation-only. The original absolute deadline retains
its admitted cancellation request once, preserving an unknown reply.
Stopped or changed-authority observers use GET-only terminal reconciliation;
unknown attempts stay pending and cannot resume inference, tool calls or uploads.
This profile grants no additional time, spending, destinations or sends. Deploying
source alone cannot add tools to an existing session; activation requires this
explicit profile in a newly admitted session. The example remains disabled.

`history_profile=agent-history-v1` adds `search_company_history` and
`fetch_company_history_record` to a newly admitted session. The agent chooses
queries, optional city/industry/task/company/kind filters, pages and exact record
IDs across research, QA and publication. It receives full record content,
provenance, coverage, semantic-search availability and correctable field errors
through the same saved tool-result loop. This profile uses the trusted company
access binding and omits the legacy preselected history preload. Query arguments
cannot grant access or choose another company scope. Original requests/results
remain immutable and exportable; a repeated pending call reuses its saved result.
The legacy learning-context path and saved tool definitions remain unchanged.

`mcp_profile=owner-readonly-mcp-v1` explicitly preserves the owner's existing
official Sheets, Slack, Notion and Firestore service connections in a new session, alongside the
selected search, publication and history tools. Preflight retains their exact
non-secret configuration and digest before creation; session/QA recovery checks
that frozen binding. Credential references and optional initialization remain
as configured by the owner; metadata-only SDK GETs resolve each exact credential
to its existing active project vault and approved MCP endpoint. Only matching
singleton vaults are attached through `vault_ids`; a vault containing another
credential, an ambiguous match or a missing credential is refused before intent
or create. The non-secret credential/vault binding and digest are frozen in the
original create payload and verified against session attachments during recovery.
An omitted/null static-bearer credential URL remains unknown in its retained auth
metadata, separately from the owner's configured endpoint; a reported mismatch
is refused. OAuth credential URLs must match the configured endpoint.
Older charged payloads keep their original omitted vault attachment and never
resolve or acquire new vaults. No vault, token, credential or grant is created.
Session allowlists expose only Sheets `get_values`/`get_spreadsheet` and Slack
public/channel search and channel/thread reads, plus Notion
`notion-get-tool-access`/`notion-search`/`notion-fetch`, intersected with any existing
owner allowlist. Firebase's official remote Firestore endpoint admits only
`get_database` database metadata. Native document reads, queries and collection
lists are excluded because they cannot enforce the existing subject/prospect
history grants; business records continue through bounded inputs and the scoped
company-history search/full-record fetch tools. All writes and sends stay with the existing authorized canonical
transports. Saved-agent configuration and previously charged sessions are not
changed. Source/catalog validation is not proof of authentication or successful
MCP reads; unavailable optional connections remain explicit gaps.
Tool-name sources are the [official Sheets catalog](https://developers.google.com/workspace/sheets/api/reference/mcp)
and [Slack's own tool guidance](https://github.com/slackapi/slack-skills-plugin/blob/main/skills/slack-search/SKILL.md),
plus the [official Notion catalog](https://developers.notion.com/guides/mcp/mcp-supported-tools)
and [Firestore database metadata tool](https://docs.cloud.google.com/firestore/docs/reference/mcp/tools_list/get_database).
Notion access metadata is checked once when available before content search;
dropped filters, truncation and unavailable tools remain explicit coverage gaps.
Adding an owner connection affects only fresh preflight/create bindings;
charged sessions continue validating their original configuration and instructions.

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

### Prospective delegated research over MCP

`mcp_profile=owner-delegated-research-mcp-v1` is a new, explicitly selected
profile. It preserves the old read-only context allowlists and adds optional
owner-authenticated research connections without altering saved agents or any
charged `owner-readonly-mcp-v1` intent. The lead model remains `gpt-6.1-sol`;
Perplexity fast and scoped history functions remain available. The lead agent
chooses queries, delegation, comparison, source verification and publication.

The default read-only profile validates known owner research connections but
excludes them from its frozen connection/vault bindings and session tools. Adding
Exa, Parallel or Blueprint to the saved agent therefore does not block ordinary
read-only research or activate paid delegation. Unknown or malformed connections
still refuse before a new intent; charged sessions retain their original scope.

| Saved server label | Frozen endpoint | Session tools |
| --- | --- | --- |
| `exa` | `https://mcp.exa.ai/mcp` | `agent_run` |
| `blueprint` | `https://tryblueprint.io/api/blueprint-work/mcp` | `start_gemini_deep_research`, `get_gemini_deep_research` |
| `parallel_task` | `https://task-mcp.parallel.ai/mcp` | `createDeepResearch`, `getStatus`, `getResultMarkdown` |

The new profile also admits Exa's documented `?login` (blank value) and
`?tools=...` selectors for its four documented catalog names, alone or together.
The complete original URL is frozen unchanged in connection, credential/vault
identity and digests. Secret/unknown query keys, duplicate keys, other origins or
paths, userinfo and fragments are refused; old profiles do not gain URL aliases.

These are paid research tools, not read-only context. An actual authenticated
`tools/list`, existing singleton credential/vault metadata and retained spending
allocation are needed before live use. Inline keys, credential copying, extra
vault attachments and inferred catalog fields are not introduced. The existing
Work MCP path is defined by WebApp's `WORK_MCP_PATH`; Blueprint provides the
Gemini research adapter. Google's own Deep Research MCP support is not an
official hosted server exposing research start/get tools.

Exa reports a running `id`; `agent_run(runId=...)` observes that same run.
`previousRunId` creates a new follow-up. Use `effort=ultra` only when advertised
by the current authenticated tool schema. Gemini and Parallel starts similarly
retain their returned identifiers for observation; missing acknowledgment is
never permission to create a duplicate. Parallel `ultra8x` is documented for its
API, but must be present in the MCP schema before use; its long runtime does not
extend the existing research deadline. No Find All MCP tool was found in the
verified Parallel catalog, so no guessed tool or API fallback is added.

The original non-secret connection and singleton vault binding are frozen in the
create payload and metadata. QA/recovery use the frozen profile, never the
owner's later saved connection changes. Profile changes are refused at durable
create admission. Provider MCP items are retained in the existing exact-turn
evidence export. Native external charges are **not measured** by Perplexity or
OpenAI usage; prospective preflight explicitly retains
`native_research_cost_status=unknown_not_metered_by_host`. Missing provider cost
receipts remain unknown, never zero or an all-provider total. An advertised cost
limit may be set within a retained allocation, but this source does not pretend
an optional limit or the existing soft target is a measured hard spending cap.
It does not activate tools, amend grants, lengthen deadlines or run a comparison.

Catalog sources: [Exa MCP](https://exa.ai/docs/get-started/exa-mcp),
[Parallel Task MCP](https://docs.parallel.ai/integrations/mcp/task-mcp).
Find All and native authentication/results require separate actual evidence.

## Guarded optional daily Exa expansion

`expansion_profile=exa-guarded-v1` exposes only
`blueprint_start_exa_expansion` and `blueprint_read_exa_expansion` through the
existing daily function boundary. It cannot be combined with the unrestricted
paid research MCP profile. The lead chooses a US physical-task hypothesis after
initial discovery and before the original final artifact/QA; returned findings
remain unqualified, use the same source checks and exact site/task deduplication,
and enter the existing company publication and learning path.

A new start explicitly selects `effort: "ultra"`; the hosted tool defaults to
low effort, where the budget is not an applicable metered cap. Admission requires
an advertised string effort enum containing `ultra`, a numeric
`budget.maxCostDollars` of at least $1 (Exa's
[documented Ultra minimum](https://exa.ai/docs/agent/agent-ultra)) within the live
schema's bounds, the existing worker's `EXA_API_KEY`, and headroom in this run's
frozen paid expansion grant (below). A live schema maximum still binds; without
one, the host's single $50 per-start ceiling applies. The host rechecks the grant,
brake and release before submission. Native Exa caps are not a claim that
OpenAI/hosting or the daily total is hard-capped. Unknown headroom means skip,
never zero. No key or verified live tool schema is created by selecting the profile.

The owner's per-run amount is the only binding number. The start tool's schema
accepts `max_cost_micros` from 1,000,000 (Exa Ultra's $1 minimum) to 50,000,000
(the code's per-start ceiling); the host admits at most half of this run's
allowance per start and only within what remains ($10 allows 5,000,000, $20
10,000,000, $30 15,000,000). A smaller request returns
`expansion_cap_below_ultra_minimum` before any credential or allocation lookup. A
cap above the remaining allowance or per-start maximum returns
`expansion_cap_exceeds_remaining_allocation` or
`expansion_cap_exceeds_per_start_maximum`; any other grant gap keeps
`expansion_remaining_all_in_allocation_unverified` and names `allocation_reason`.
These skips name the exact `max_start_micros` and `remaining_micros`; none starts
Exa or consumes the claim. Stable `ExpansionError` codes reach the agent unchanged
instead of a generic unavailable error. With one Exa start per run, Exa alone can
use at most half of the allowance; the rest stays combined headroom for FindAll.

The 2026-10-04 release raised the schema maximum from 5,000,000 once, at an idle
release boundary: session checks compare live tools with `search.tools()`, so a
schema change must never land while a daily session is in flight, and tools are
never built per run.

### Owner-directed paid expansion allowance

Paid expansion sources (Exa now; FindAll later) share one combined per-run
allowance that the owner sets as data; the 2026-10-04 owner decision is $10 per
daily research run. It is host-reserved and separate from `soft_target_usd`, which
stays $5 and covers the rest of the run; neither is extra authority. Changing
$10 to $20 or $30 needs no code change, redeploy or new package; run the packaged
owner command as in [operators/README.md](operators/README.md#owner-paid-expansion-allowance):

```bash
paid-expansion-direction.py show
paid-expansion-direction.py set --per-run-usd 20.00 --approval-reference REF [--apply]
paid-expansion-direction.py disable [--apply]
```

- `allocation.py` is the single producer and arbiter. An owner direction
  (`blueprint.research-paid-expansion-direction.v1`) has version, supersedes,
  `per_run_limit_usd` (format `^[1-9]\d{0,2}(\.\d{2})?$` within $1.00–$100.00; the
  code ceiling refuses typos such as `200` or `1000`), sources, scope (project,
  agent, Firestore root, run-key prefix, America/Chicago), effective/expiry times,
  approval reference, approver, issue time and reason. The SHA-256 of its canonical
  JSON names a create-only record at
  `gs://blueprint-8c1ca.appspot.com/operations/research/paid-expansion/<sha256>/direction.json`.
- Company control carries it top-level, `control.paid_expansion = {enabled,
  current: {sha256, version, uri, direction}}`. Config keys stay allowlisted, so
  older packages ignore it and keep today's skip. Only the fenced bridge op
  `paid_expansion_set` writes it: a compare-and-swap on the caller's expected
  previous SHA that validates format, ceiling, scope, digest and version chain and
  appends `paidExpansionDirections/<sha256>` create-only. `configure` keeps the
  current value; neither `init` nor `configure` can write a different one.
- Under its lease and before the durable intent, the runner freezes one
  `paid_expansion_grant` per daily row: `grant_id = sha256([direction_sha256,
  run_key])`, version, `limit_micros`, `per_start_max_micros` (half the limit,
  clamped to $1–$50: $10→$5, $20→$10, $30→$15), `source_commit` and `valid_until`
  (the earlier of the direction expiry and the research deadline). The worker
  verifies the direction digest in control and never reads object storage. A
  refusal such as `paid_expansion_disabled`, `paid_expansion_direction_invalid`,
  `paid_expansion_limit_invalid` or `paid_expansion_expired` is recorded and
  ordinary research continues without expansion; so is a grant the company store
  refuses at the intent (`paid_expansion_grant_not_admitted`). A crash or retry
  reuses the row.
- The frozen grant is the run's upper bound, and live control can only tighten
  it. A raised amount applies from the next run. The brake, a lower amount, a
  removed source or a changed release applies at once, so a mistaken amount can
  be braked and corrected for the run already in progress.
- `allocation.problem` admits each start. Remaining is the effective limit minus
  the reserved caps of this run's durable claims; an unknown cost holds the whole
  reservation, and nothing is released without a provider-reported terminal cost
  (none is pinned yet). It refuses a start above the remaining allowance or
  per-start maximum, after `valid_until`, while the owner's brake is applied, or
  after a `source_commit` change. The put transaction for an Exa claim also
  refuses a reservation beyond the frozen grant, a grant not backed by its audited
  direction or not naming Exa, a cap that differs from the native request budget,
  a braked control, a changed release and a row whose manifest an older bridge
  rewrote. Time stays with the worker, which rechecks `valid_until` after the
  claim and before the POST.
- `set` and `disable` are dry runs unless `--apply`. `set` builds version+1
  superseding the current SHA (or, after a rollback dropped control's copy, the
  audit head), writes the object create-only, swaps control and reads both back;
  it waits for an active run unless `--during-active-run`. `disable` is the
  emergency brake: it stops new paid starts at once, including in an active run,
  needs no object write, and only a new `set` re-enables expansion. Both poll for
  the worker's next lease release instead of displacing it. A direction lasts 90
  days unless `--expires-at` sets up to 366 days.
- FindAll (PR 2) joins `allocation.SOURCES`, the bridge's `PAID_SOURCES` and
  `allocation.claims()`, marked `TODO(FindAll)`. Sending stays off.

The worker consumes one durable whole-run claim before the native POST, retains
complete private MCP receipts and provider records in the existing company
ledger, and permits subsequent reads of only the acknowledged original ID.
Unknown submissions retain their claim/reservation and prevent another optional
Exa start; they do not discard evidence or stop ordinary research.
Claims made before explicit Ultra selection keep their original request bytes;
only their retained ACK or original-ID reads may recover them, never a new start.
The original research deadline is preserved; no extension, enrichment, alternate provider,
`previousRunId`, automatic POST retry, outreach or provider deletion is allowed.
Saved OpenAI OAuth credentials remain in their URL-bound vault and are never
extracted into the worker. The MCP transport uses the documented `x-api-key`
header at `https://mcp.exa.ai/mcp?tools=agent_run`, with credentials excluded from
company receipts and logs.
