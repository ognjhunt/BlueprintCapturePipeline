# Adaptive commercial site discovery

The intended daily target is **at least ten new distinct site/task opportunities**.
An exact operating location, sourced recurring physical work, plausible labeled
robot/task hypothesis, incumbent check, counterevidence and one useful first
question make a discovery row useful. Unknown interest, budget, support or pilot
readiness stay explicit; they are not alone reasons to reject a discovery row.
Nothing is pilot-qualified. Existing deployments are separate learning notes;
existing CRM matches are evidence refreshes. Neither fills the new-opportunity
quota. Keep the supported subset when fewer than ten survive and explain the
coverage and shortfall. Never pad or invent addresses, residual manual work,
service eligibility, willingness or performance.

`adaptive-daily.config.example.json` is disabled and requires an explicit
`discovery_profile=adaptive-sites-v1` opt-in. It retains the daily **$1 soft TOTAL
target**, Default project, saved Sol agent, hosted template, reviewed mounted
skills, native search and sandbox network restriction. It removes the legacy
two-search/two-open/three-candidate and five-observed-activity quality caps for
newly admitted adaptive rows. Up to 100 records is only an input resource limit.
The v3 schema accepts coverage and ten or more rows; v1/v2 stay unchanged.
Legacy v3 artifacts without coverage remain replayable. Adaptive collection
requires actual query/page counts, explored branches, rejections, stopping reason
and a shortfall reason where needed. Counts are agent reports, not billing
receipts. Exact duplicates are excluded locally; agent QA checks semantic
matches, existing deployments, source support and final accepted-count shortfall.
The report and CRM publication remain under existing one-use, digest-bound
claims and approvals; no outreach or stage promotion occurs.

The disabled example admits 30 minutes total, reserving ten for same-session QA.
The research deadline is 20 minutes and the total QA deadline 30 minutes from
the original intent start. Those durations are pinned in a newly created row
before provider creation. Outer observers accommodate those admitted phases;
they do not extend an older row when configuration changes. Existing rows keep
their original deadline/tool guard. This example does not modify live control,
erase the Oct 1 failed intent or authorize another create for that date. The
cleanup guard still blocks a new daily session until exact session/environment
cleanup is approved and verified.

## Separately authorized Oct 1 test preparation

`adaptive-test.config.example.json` records the one-time **$25 total ceiling**,
an early **$8 model-token estimate stop**, a 30-minute watchdog/ten-minute QA
reservation and the exact existing Oct 1 session. Its enabled flag is false.
It is deliberately a different schema from recurring configuration: feeding
$25 into the daily runner refuses. New session creation, daily state reset,
publication and permanent deletion are all forbidden by this prepared profile.

`adaptive.py` prepares an event and intent locally using the actual SDK
`agent.session.input.message` shape. It performs zero provider requests and zero
Firestore writes. It requires the retained failed-row digest and raw artifact
digest, exact context/policy bindings and a complete fresh canonical CRM snapshot.
It omits CRM contact/email fields from the event. The test output has a separate
artifact path. Optional exact-session/environment/root-turn GET receipts must
show idle/no actions, a connected small hosted environment, standard service
tier and exactly the original completed root turn. Ambiguity or expiry remains
an admission blocker, not permission to replace/reset/delete the session.

Example, from an immutable reviewed portable package with private local input
materializations and a new output path:

```bash
python -m tools.daily_research.adaptive \
  --status /private/retained-status.json --crm /private/fresh-crm.json \
  --source-commit FULL_REVIEWED_40_CHARACTER_SHA \
  --session /private/exact-session-get.json \
  --environment /private/exact-environment-get.json \
  --turns /private/complete-exact-session-root-turns.json \
  --output /private/new-disabled-test-intent.json
```

This command is preparation and does not establish launch readiness.
`adaptive_runtime.py` is the single-test adapter to the existing Store, Provider,
Runner collector and Consumer QA, not another scheduler. An explicit `execute`
uses fresh GETs and complete canonical CRM reads before admission, then saves
the complete intent and original daily blob binding under
`blueprintDailyResearch/sites-first/adaptiveTests/adaptive-discovery-20261001`.
The existing shared lease, immutable chunked blobs and heartbeat fence a single
research event attempt and a separate single QA event attempt. The original
daily row, file manifests and work item remain untouched. The test artifact is
`/workspace/outputs/adaptive-discovery-20261001.json`, bound to exactly one new
root turn; the original root/artifact cannot satisfy collection. Test files are
in a separate subcollection. An uncertain reply or restart reconciles by GETs;
research input is never repeated. Same-session agent QA follows within the pinned
30-minute total deadline and produces a durable accepted count. There is no
session create, publication, deletion or dot review step in this adapter.

The example profile still has pending admission references. **Do not run paid
execute until exact fresh session and total-spend admission are reconciled.**
Replacing a pending reference records an owner-resolved admission; it does not
prove that an unavailable provider dollar cap exists. `status` makes no provider
request. `reconcile` never starts research input; it may send the once-admitted
QA event for an existing test. Use an immutable reviewed package:

```bash
python -m tools.daily_research.adaptive_runtime status
python -m tools.daily_research.adaptive_runtime reconcile --profile /private/admitted-test.json
```

The observer stops on disabled control, interruption, unknown/invalid turn
usage, $8 cumulative model estimate, research deadline or total QA deadline. It
cancels only the exact bound session. The observer loop stops at the original total
deadline plus 60 seconds of cancellation/collection grace. Provider reads can
paginate, so the operator must also bound the entire process group with GNU
`timeout --signal=TERM --kill-after=60s 1860s` when invoking execute/reconcile;
this is an independent wall-time watchdog. A forced stop cannot prove remote
cancellation: retain the durable unresolved record and reconcile its exact
session/turns after the lease expires, without another research input. Unverified terminal
cancellation remains unresolved with cleanup held. Usage can lag; the test may
stop without sufficient output. A successful input POST proves neither settled
total spend nor ten accepted prospects.

## Spend admission limits

The token estimate charges all input at $9/M (the sum of long-context input
and cache-write rates), all output including reasoning at $15/M, then includes the documented
10% regional premium. This deliberately reserves both input and cache-write
charges because SDK usage does not expose separate cache writes; it does not
claim both are always billed. Reasoning is not counted twice. Standard service tier is
required; no cache discount is assumed. This is a conservative token estimate,
**not settled total spend**: it excludes unreported/lagged usage, web-search tool
fees and hosted-environment charges. Null/invalid usage is unknown. Preparation
never runs a spend watcher. The single-test adapter applies the $8 stop to
complete new-turn usage, retaining the estimate and exclusions in its record.

Official project monthly hard limits provide a useful additional boundary but
enforcement is delayed and spend can exceed the configured threshold. The
shared Default project also serves communications; changing its limit affects
that workload and needs coordinated owner action. Neither a prompt instruction,
a token estimate, cancellation nor that limit proves a strict per-test $25
TOTAL ceiling. No configurable Agents API per-session dollar ceiling was found
in the supported SDK 3.22.1 session/event APIs. Pending spend/session admission
therefore remains explicit; no paid test is launched by this change.

References: [Sol pricing/limits](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[project spend enforcement](https://developers.openai.com/api/docs/guides/spend-limits),
[best-effort Agents usage](https://developers.openai.com/api/docs/guides/agents-api/observability),
[session/event API](https://developers.openai.com/api/reference/resources/beta/subresources/agents/subresources/sessions/methods/create).
