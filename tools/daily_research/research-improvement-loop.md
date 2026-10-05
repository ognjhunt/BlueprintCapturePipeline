# Daily research improvement handoff

Before writing its research output artifact, the agent records useful improvement
proposals in the existing fields. After the run is terminal, a separate engineering
follow-up consumes retained research/QA evidence, implements, independently reviews
and releases accepted changes. It never edits terminal artifacts to add proposals.
This policy adds no model call, mandatory
output field, live self-editor, provider start or business-worker scheduler.
An optional Codex follow-up or another existing authorized engineering agent may
perform the work; canonical records and recovery must work without Codex or a
provider session. Scheduling such a follow-up does not make it the business
runtime or permit a second release owner.

Scope is the owner's existing eight workstreams, with AWS skipped. Research
discovery supports [ADP-010 partner selection, partner-phase day 7](../../docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md);
findings do not constitute partner admission or physical proof. Recurring changes
stay limited to research strategy, output presentation, source/date handling,
exact facility/task deduplication and ordinary-error recovery. Do not widen the
general AutoAgent allowlist or introduce services, credentials or access.

Optional paid list expansion (Exa now; FindAll later) draws on a separate,
host-reserved per-run allowance that the owner sets as data with
`operators/paid-expansion-direction.py` (2026-10-04 owner decision: $10 combined
per run); the $5/day research soft target is unchanged. The actual session must
expose existing `agent_run` with a supported `maxCostDollars` cap, and the run's
frozen grant must have remaining allowance for the native start (see
[SEARCH.md](SEARCH.md#owner-directed-paid-expansion-allowance)). Otherwise skip
expansion with a specific reason. Retain one-use intent before the single
start and use the original acknowledged `runId` for observation; uncertain writes
are reconciled without replay or `previousRunId`. Merge raw discoveries before
final QA and exact site/task/history deduplication; no quotas or enrichment.

Activation is separate from policy publication: the current
`owner-readonly-mcp-v1` session profile does not expose Exa, and unverified cost
headroom cannot authorize an expansion. The remaining-allocation route is the
frozen per-run grant; it enables nothing until a released package selects
`exa-guarded-v1` and the owner applies a direction. Do not report expansion as
active before a real daily session exposes the tool and records a granted row.

## Capture and choose

1. Read the terminal run, retained source/QA evidence and relevant learning
   history. A pending, inaccessible or missing record stays unknown. Preserve
   the original run and source dates. Do not force future dates, run paid canaries
   or replay completed provider jobs/benchmark inputs to manufacture improvement
   evidence.
2. Read proposals in existing `proposed_next_actions` and supporting `findings`.
   Each should identify the actual weakness, exact run/source/evidence refs and
   hashes where available, current instruction/source version, smallest proposed
   change, expected benefit, acceptance/regression examples and rollback version.
   Missing detail means fetch the existing evidence or narrow the claim; it is
   not permission to invent it. Robot-capability `proposed_knowledge_deltas`
   remain separate factual proposals, never automatically approved facts.
3. Select only new actionable evidence. Reuse the same stable Blueprint proposal
   identity across retries and bind it to the source evidence and candidate.
   Resolve repeated proposals against prior dispositions. If nothing is useful,
   retain a truthful no-change disposition without edits or duplicate follow-ups.

## Implement and release

1. Preserve active/dirty checkouts. Use an isolated branch for the smallest
   candidate. Retain a concise disposition with proposal ID, original run and
   evidence refs/hashes, baseline and candidate versions, owner, review/check
   receipts, rollback, current state and next action. GitHub owns source and
   instructions; private evidence stays in existing authorized Blueprint storage
   as portable JSON/Markdown with hashes and an export/recovery route. Provider
   IDs are provenance, not the only business identity or retained copy.
2. Independently review the actual diff and replay useful retained cases offline.
   Acceptance examples must cover the observed weakness and plausible regression;
   do not write tests that merely assert the new prose or spend hours on broad
   unrelated suites. Code changes run the affected checks; required exact-head
   CI gates the release. Ordinary repairs within existing authority require no
   additional human approval.
3. The sole release owner merges/publishes the reviewed exact source identity.
   Runtime changes use the existing exact-SHA deployment path. Instruction-only
   changes preserve the full previous definition, verify unchanged tools/model/
   access, publish the reviewed text once, verify readback, and align only the
   expected instruction hash through the existing leased/fenced control update.
   Source merge, saved text, installed runtime and live configuration are distinct
   receipts; claim only the stages actually verified. Keep active runs bound to
   their original inputs and instruction version.
4. Preserve the current US-only multisector discovery scope, separate research
   and communications budgets, unknown-usage holds, grants and expiry, opt-outs,
   draft-only/no-send policy and Gmail-draft setting. A candidate cannot change
   these, qualify a buyer, delete provider data, or authorize another paid test.

## Observe and recover

Use the next real authorized run to assess the recorded acceptance conditions.
Retain its version and evidence, separate descriptive improvement from causal
proof, and keep healthy output usable if one claim fails. An improvement may be
proposed, rejected, released or reverted; none of these means a new prospect is
qualified. Do not force edits on days without new actionable evidence.

For a regression, the release owner restores the exact retained instruction
version and matching expected hash, or releases a reviewed source revert through
the existing deployment route. Append the correction and verification receipt;
preserve both histories. An uncertain write is reconciled by exact readback before
any replay. Missing release access or genuinely broader spending/access/sending/
expiry/deletion authority is reported specifically; ordinary formatting or
metadata errors do not become a new approval framework.
