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

This command is **preparation, not a test controller or launch readiness**.
Before any POST, the exact intent/event/input hashes need their own Blueprint
durable test record and one-use fenced claim, separate from the immutable daily
record. A supported execution/observation contract must retain the exact new
root-turn/artifact bindings and cancel on watchdog, disabled control, unknown
usage or exhausted admission. Repeating an uncertain event is forbidden. Agent
QA must be separately admitted within the same total time/spend envelope. No
new provider framework or dot review step is required or supplied here.

## Spend admission limits

The token estimate charges all input at $5/M (long-context cache-write upper
rate), all output including reasoning at $15/M, then includes the documented
10% regional premium. Reasoning is not counted twice. Standard service tier is
required; no cache discount is assumed. This is a conservative token estimate,
**not settled total spend**: it excludes unreported/lagged usage, web-search tool
fees and hosted-environment charges. Null/invalid usage is unknown. The $8 stop
is prepared policy only; this preparation command never runs a spend watcher.

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
