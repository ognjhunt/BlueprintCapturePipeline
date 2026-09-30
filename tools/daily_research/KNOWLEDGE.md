# Reviewed knowledge snapshot and research v2

This change is authorized by the owner's explicit 2026-09-30 request to implement
Blueprint's versioned knowledge snapshot and research runner integration. It is a
bounded scope exception to the repository's ADP-only default; no ADP backlog item
or physical-proof gate is invented. It changes offline export, validation and the
existing research runner. It does not authorize a canary, access provisioning,
production deployment, saved-agent changes, or schedule cutover.

Notion is the editable authority for reviewed claims, limits, conflicts, unknowns
and evidence. The [Knowledge page](https://app.notion.com/p/3eb80154161d8116858ed5f376b4b7a9)
and [capability register](https://app.notion.com/p/3eb80154161d8107a342d99b00534164)
are reviewed destinations. GitHub owns the export/schema/loader/runner contract
and tests. Sheets remains the canonical CRM; its complete 26-hour snapshot gate
is unchanged. A generated mirror is read-only background, never live operational
state. No private prospects or contacts belong in this mirror.

## Two layers and provenance

The snapshot schema separates companies with extensible team roles from records
for exact products/robots/policies/software/verticals/world models and versions.
A software-only company needs no hardware specification. Every specification
fact requires its typed value/unit/conditions object; other fields cannot carry
a specification object. Each record contains a
small set of typed facts: task claims, specifications, supported hardware,
availability, deployment, geography, integration requirements, limits, unknowns
and later qualification details such as cycle time, reliability and supervision.
Specifications carry units and conditions; a rated payload does not prove a task.
Unknown fields always remain gaps, even when reviewed. Facts carry individual confidence, status, freshness days, relevance tags and
source evidence. No giant fleet or vendor directory is required.

Evidence levels distinguish `vendor_claim`, `demonstrated_capability`,
`named_deployment`, and `current_availability`. Levels are explicit reviewed
labels, never inferred from a vendor's claim. An unnamed shipment is not a named
or current deployment. Structural validation cannot prove a label or statement
is factually supported; the parent must check source support before any write.

`schema_version`, `exported_at`, `source_pages` and `content_hash` bind an export.
Native revision IDs may be stored when exposed. Otherwise record
`revision.kind=page_last_edited_at` and the exact returned timestamp as a **revision
surrogate**, never as a native revision ID. The SHA-256 covers canonical ASCII
JSON with sorted keys and compact separators, excluding only `content_hash`.
Hashing proves integrity/binding, not reviewer identity or authenticity.

Every source preserves publication date (null when unknown), original
`source_checked_at`, and separately `revalidated_at` (null unless actually
reviewed again). Date-only source review dates remain date-only; exact timezone
instants remain exact. Dates must use YYYY-MM-DD and timestamps must use
RFC3339-compatible timezone forms; noncanonical or normalized-overflow offsets
are refused. Freshness arithmetic conservatively uses the start of a
date-only review day in America/Chicago without changing stored provenance.
`exported_at` and `snapshot_loaded_at` are exact timezone instants and never imply
a source was checked again. Quotes are exact reviewed source excerpts, or null
when the reviewed register has only a paraphrase. Cached output cannot invent a
quote. The same source URL with a newly checked date requires actual live review.

## Local export and opt-in integration

An authorized reviewer prepares a JSON object with the schema's source pages,
companies and records. The exporter only validates and serializes that input;
it has no Notion/CRM connector, no model calls and no inferred claims:

```bash
python -m tools.daily_research.knowledge reviewed-input.json knowledge.json
```

The output path is a local generated artifact. Review the input and resulting
hash before placing it at the configured private host path. No real bootstrap
snapshot is shipped. `tests/fixtures/daily_research/knowledge.synthetic.v1.json`
is clearly synthetic and must never become operational reviewed evidence.

Use the disabled `knowledge.config.example.json` for deliberate opt-in
`research_contract_version=2`. v1 remains the default and still requires every
`evidence.checked_date` to equal the run date. Supplying a snapshot to v1 is
refused. v2 requires a snapshot and validates it **before any provider access or
create**. Missing, malformed, unknown-version, tampered, future-dated or oversized
snapshots fail closed. Limits: 256 KiB export, 100 companies, 100 records, 20 facts
per record, four sources per fact, and 30 source-page references. Sources/pages,
foreign keys, duplicate IDs/JSON keys, timestamp ordering and condition-bearing
specifications are validated.

Optional filters are `company_ids`, `record_ids`, `task_tags`, `geography_tags`.
Filters combine by intersection, tags match exact reviewed labels. Fact tags
fall back to record tags when empty. Unknown geography remains an explicit gap
for a relevant task, never a positive service eligibility claim. The selected
context is capped at 12 records, 60 facts and 32 KiB; a larger set requires tighter
filters. The small selected context is supplied as a JSON string in the session
input. It is explicitly untrusted data, never executable instructions. No new
MCP tool, hosted handler, networking, credential or agent definition is needed.

The runner persists exact filtered context and its digest in the durable ledger
before its single create attempt. Resume/collection uses that immutable context
and rejects ledger tampering even if the configured export changes or disappears.
Cache freshness is recomputed at collection time; a fact that expires after load
cannot satisfy a candidate, and the saved context is never refreshed or rewritten.
The current complete CRM snapshot is revalidated at collection; the runner does
not query live Sheets itself. Chicago-date idempotency, cancellation,
cleanup, cost/usage semantics and saved provider bindings remain unchanged.

## Research output v2

`daily-research.v2.schema.json` documents the output shape. Runtime validation
also enforces the semantic bindings and candidate role coverage. Evidence adds
`origin`, `evidence_level`, `source_checked_at`, `snapshot_loaded_at`,
`revalidated_at`, `snapshot_record_id`, `snapshot_fact_id` to v1's fields.

Task and geography evidence must be live and checked on the run's Chicago date.
Their `evidence_level` is null: an ordinary operator task or location fact is not
a robot demonstration. Non-null supported evidence levels are required only for
capability evidence; classification and claim_kind apply to every role.
Capability evidence may reuse a reviewed, supported, nonconflicted stable fact
within its field freshness envelope, with the exact selected record/fact,
statement, source, dates, evidence level and preserved/null quote. Cached review
dates remain original dates. The run's `checked_date` is not a freshness claim
for each cached source. Live evidence must be checked by collection time; future
instants on the same run date are rejected.

Availability, geography, deployment, integration, safety, support, price and
supervision fields and `current_availability` evidence always require live
sources. Conflicted/unsupported/unknown/stale facts remain research gaps.
Reviewed labels and field freshness thresholds are configured in the reviewed
export (1–90 days); loading never renews them. Cached limits and specifications
can guide a hypothesis; the parent still checks the proposed task match and
consequential live facts. Prompt rules forbid inferring task performance from
hardware maximums or eligibility from unknown geography.

v2 focuses research on gaps, conflicts, staleness, discoveries and consequential
facts. It returns at most ten `proposed_knowledge_deltas`, each with selected
record/fact IDs (null for a discovery), a reason, proposed statement, unknowns and
fresh live evidence. These are evidence-backed proposals for separate parent
review. Candidate review never auto-accepts or writes knowledge deltas. The v2
packet and reviewed candidate outbox retain the Knowledge-page destination from
the immutable packet. The runner itself never writes Notion or CRM.

Daily discoveries, weekly active availability review, and monthly broad refresh
are possible later review cadences, not installed schedules. No timers or live
units change in this integration.

## Offline verification

CLI preflight shares the local snapshot validation/filtering boundary with run
admission and refuses invalid v2 inputs before provider construction or reads.

The lightweight daily-research workflow runs the existing lifecycle suite plus
`tests/test_daily_research_knowledge.py`, schema/fixture checks and changed-file
lint. Its test-only jsonschema format extras activate date-time validation; tests
assert that the checker is installed and reject malformed dates through both the
schema and loader. It rejects skipped/failed/errored cases. Tests cover missing snapshots before
provider access, bounds, hashing, revisions, date granularity, stale/conflicted/
unsupported facts, stable cached capability reuse, live operational boundaries,
filtered relevance/unknown geography, fabricated quotes, ledger resume/tampering,
proposed deltas and serialization of malicious text as untrusted data. Prompt
isolation and schema validation are defense layers; they do not establish that a
model is immune to injection or that reviewed labels are true.

## Prerequisites before live verification

The existing README's approval and lifecycle gates remain authoritative. Before
any separately authorized v2 canary, prepare reviewed real Notion input with
honest revision surrogates and source-check granularity; approve its private host
handoff and bounded filter config. Verify the exact saved agent/template, installed
skills, supplied v2 input, resulting output/artifact provenance and run-date live
evidence. Parent source review, delivery readback receipts, usage reconciliation
and hosted cleanup must succeed before coordinated single-trigger cutover. This
PR supplies offline verification only; none of these live actions is performed.
