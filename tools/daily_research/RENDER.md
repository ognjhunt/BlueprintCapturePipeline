# Existing Render worker integration

Selected new-row search profile: [Perplexity Fast application tools](SEARCH.md).
The existing worker passes `PERPLEXITY_API_KEY` only to its application-side
Python child; no credential enters the hosted sandbox or private Node pipe.
The profile remains disabled until the owner-managed production binding and
existing intent/cleanup/release gates are verified. Old admitted rows remain
native-search rows; the failed October 1 intent is never reset or converted.

Owner direction: 2026-09-30. Blueprint owns the trigger, durable execution state,
reports, QA and publication. Dot can report outcomes; it is not a workflow step.
This isolated research package uses no Pipeline/GPU application installation.
Existing operator hold `20260930T132640Z-hold-354d4ae6` remains held.

## Runtime and controls

Use existing worker `srv-d9t8gg1t0dsc73am9q70` and Node 24. Python 3.11, 3.12 and
3.14 are tested; the verified disabled release uses Python 3.14.3.
The WebApp build validates and extracts the pinned portable archive, then creates
`dist/daily-research/venv` with only `openai==3.22.1`. Firebase Admin and Google
auth resolve from the existing WebApp dependencies. Python accesses Firestore
through a private Node pipe; credentials stay in memory/environment, never in
the archive, inputs, argv, logs or provider sandbox.

The saved template discovers the exact mounted instruction files through
capability directories, with empty attached-skills/plugins lists. See
[SKILLS.md](SKILLS.md) for reviewed hashes, session byte binding and the limits
of a read-only template GET. Do not re-register skills to repair preflight.

Two independent controls must permit creation:

- `BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED=true` starts only the research clock.
  It defaults false, including in the Render manifest.
- Firestore `blueprintDailyResearch/sites-first.enabled=true` permits new work.
  Its config requires the reviewed release SHA, reconciled legacy-attempt
  reference, scheduler authority, and saved-agent instruction SHA256.

`BLUEPRINT_DAILY_RESEARCH_PACKAGE_BUILD=true` installs the research-only package
during the existing worker build. The web service has no Python build requirement.
Keep all existing ops flags false, autonomous outbound false, launch-only true,
launch-forward true, candidate-outbox true, and ADP engineering unset. Do not add
local env bootstrap files or change paid-execution authority.

The Python clock schedules 07:00 America/Chicago using zoneinfo, including DST.
Boot reconciles an unfinished older date first, or the latest eligible date;
it never creates a paid catch-up backlog. Disabled idle minute ticks read only the small
control document; enabled workflow ticks also query bounded pending projections.
Recovery errors retry after five minutes; active QA is observed at three-second
intervals and its cancellation is attempted before returning an observation error. Deploy overlaps use
a transactional, heartbeat-renewed, fenced lease. Immutable compressed chunks,
complete input snapshots, exact create payload and dated intent are committed
before the separate one-use create claim and provider POST. A lost reply or
restart reconciles exact metadata; it never repeats create for that date.
Timeout poisons the pipe so late replies cannot authorize subsequent requests.

The worker requests process-group SIGTERM, allows 25 seconds, then bounds shutdown
with SIGKILL. A host with a shorter termination grace may kill it earlier;
the durable intent remains for recovery. A local process exit does not prove
provider cancellation, success or cleanup.

## Runtime envelope

Owner decision 2026-10-04: one daily research run can use 60 minutes in total.

- `runner.MAX_ADAPTIVE_RUNTIME_SECONDS = 3600` is the only runtime bound for an
  `adaptive-sites-v1` row. Config admission, the pinned phase windows, the
  validation-repair authority, the paid expansion grant and the Exa expansion
  deadline all use it. The non-adaptive cap stays 180 seconds.
- The approved production setting is `config.max_runtime_seconds=3600` with
  `config.qa_reserved_seconds=900`. After the operator applies it (below), a new
  row pins `total_runtime_seconds=3600` and `research_runtime_seconds=2700`
  (45 minutes).
- Research ends at `started_at + research_runtime_seconds`. The paid expansion
  grant and Exa reads end no later than that. Repair, QA and publication share
  the total deadline `started_at + total_runtime_seconds`. A repair authority
  must last exactly the row's `total_runtime_seconds`.
- Each row keeps the envelope it was admitted with. A row admitted at 1800
  seconds keeps 1800 seconds. A later config change cannot extend or shorten it.
- The soft target stays $5; `expansion_profile` requires that value. A longer
  run can use more model, search and hosted-environment cost. The soft target
  is not a hard cap.
- One-time paths keep their own pinned windows: the 600-second recovered-QA and
  QA-retry continuations, the Oct 1 adaptive test, and the baseline canary. The
  canary admission still requires the Oct 2 profile (1800/600) and its
  `timeout ... 1860s` watchdog, so it refuses a 3600 profile until it has its
  own authority.

Apply the change in this order. A package without this change refuses a 3600
config on every scheduler tick (`approved_envelope_mismatch`).

1. Run `status` and make sure that no row is unfinished (every row is
   `completed`, `failed` or `cancelled`). Then vendor and deploy a release that
   contains this change. A worker deploy sends SIGTERM, and a stopped observer
   cancels the active session. A normal run starts at 07:00 America/Chicago and
   can last until about 08:01; a catch-up run lasts 60 minutes from its start.
2. Export the complete current control document with the read-only command in
   [operators/README.md](operators/README.md#daily-research-runtime-envelope).
   Change only `config.max_runtime_seconds` to 3600 and
   `config.qa_reserved_seconds` to 900. `source_commit` must name the installed
   release; if it does not, set it in the same input.
3. From `/opt/render/project/src` in the worker shell, run:

   ```bash
   PYTHONPATH=dist/daily-research/release dist/daily-research/venv/bin/python \
     -m tools.daily_research.render configure --input /PRIVATE/control.json
   ```

   `configure` checks the root bindings and `config` with the installed release.
   It does not check the other sections. It replaces the whole document except
   the lease, `cleanup_observation_required`, `paid_expansion` and `site_universe`, so a partial
   file silently removes sections. For example, a missing `workflow` stops QA
   and publication. A held lease refuses with `runner_overlap`; try again after
   the worker releases it.
4. Read back the control, compare it with the export, and run `preflight`. The
   next row admitted after the change pins the new envelope. A control change
   wakes the scheduler: if the due day has no row yet, its run starts at once.

To roll back, first wait until no 3600-second row is unfinished. Then configure
1800/600 while this release is still installed, and only then install an older
release. An older release refuses a 3600 control on every tick. It also refuses
a 3600-second row with `pinned_phase_envelope_invalid`, and it cancels such a
row that is still in its research phase.

## Site universe slice

Optional, owner-pinned prioritization for ADP-010 partner discovery; off by default.
`control.site_universe` is top-level like `paid_expansion`: config keys stay
allowlisted and older packages ignore it. An older release's `configure` keeps only
what its input holds, so an input without the key turns the slice off. The disk
`Ledger` has no company control, so the feature is always off there.

- Off (key absent or `enabled=false`): the create payload, metadata,
  instructions and row are byte-identical to today, and nothing else is read. The
  pin comes from the control read the run start already makes under the lease
  when the history profile is on (the production config); otherwise the run makes
  one extra control read.
- On, under the lease after `preflight`, `site_universe.attach` reads the pinned
  object through `site_universe_object_get` (exact generation, size and SHA-256;
  its own 20 s bound inside the 35 s pipe deadline), validates the export and
  selects up to `slice_size` sites: sites with an outcome in the last
  `reoffer_after_days`, CRM rows and prior formal candidates are removed; one seed
  per policy capability, then rank order with at most half per lead capability and
  one site per group, then each cap relaxed. A prior row in the window whose packet
  no longer matches its `packet_digest`, or whose outcomes are unavailable, is
  skipped; every site it names stays out for the window, and the selection counts
  it with its code. Only such a row that names no readable site refuses the slice.
  The export must list only reviewed licenses (`CC0-1.0`, `ODbL-1.0`, `US-Gov-Work`, `US-PD`).
- The slice goes into `/workspace/inputs/blueprint-site-universe-slice.json` as an
  inline file, `metadata.site_universe_slice_digest` holds its SHA-256, one
  trusted paragraph follows the CRM prefix, and the prompt's inventory disposition
  list gains `screened`. No tool is added, so the session's tool schemas,
  `search.tools()` and `check_agent` are unchanged. `put`, `observe` and
  `create_check` compare metadata, so the slice cannot change after the intent.
- The slice never stops an intent that fits without it. Over `search.MAX_INTENT`
  the run keeps today's payload, shrinks `row.site_universe` to `{state, code}`
  (`site_universe_intent_resource_ceiling` for a slice that no longer fits), and
  drops even that when it does not fit.
- Inventory validation is exactly the previous rules without a slice. With one, a
  record may carry a `site_universe_id` from the frozen slice (no other id),
  `screened` is valid, and only a record with such an id may have empty
  `source_urls`. The v3 JSON Schema accepts this shape; the host is stricter.
- Recovery, repair and QA use the frozen row only; `frozen_slice` re-checks the
  SHA-256 before any use. `Runner.prepare_output` writes `packet.site_universe`
  (one outcome per slice site, link issues and the funnel, under 16 KB and bound
  by `packet_digest`) on every path. Explain, repair, QA and publication text gain
  one sentence only when a slice is attached.
- Any slice failure records `row.site_universe = {state: "refused", code}` (or
  `exhausted` for an empty slice) and the run continues without the slice. A lost
  store or lease stops the run as today.

Deploy a release with this module only while the worker is idle, with the key
absent, and confirm that the next create payload is unchanged before pinning. The
owner commands are in [operators/README.md](operators/README.md#site-universe-backlog-slice).

## Stable private consumer contract

Contract `blueprint.research-snapshot.v1`, root
`blueprintDailyResearch/sites-first`:

| Record | Contents |
| --- | --- |
| `runs/YYYY-MM-DD` | Immutable row blob pointer, date/state, metadata binding, cleanup guard, actual session/turn/environment IDs, one-use create claim |
| `blobs/SHA256` + `chunks/N` | Exact uncompressed SHA256/length, gzip metadata, immutable chunks at most 256 KiB each; read verifies all bytes |
| `files/YYYY-MM-DD-{artifact,evidence,output,review,qa,qa-evidence}.json` | Blob pointers; downloaded raw artifact is immutable once bound |
| `files/{crm,knowledge,refresh-policy}.json` | Private checked input pointers, not public prospects |
| `workItems/YYYY-MM-DD` | `owner=blueprint-research-qa-publication-agent`, `stage=validation_repair_pending`, `agent_qa_pending` or `publication_pending`, run key, row blob, packet digest, `observer_receipt_required=false`, no-outreach scope; repeated nonprogress becomes `validation_repair_blocked` |

When a completed research artifact fails validation, the same workflow agent
receives precise field/reason/allowed-semantics feedback, retained source receipt
bindings, knowledge context and fresh CRM identity projection in its existing
session. It may revise or honestly quarantine unsupported claims, choosing its
own configured tools and depth. Complete correction requests and versioned raw
artifacts are durable before effects; uncertain input replies reconcile through
GET only. Corrections share the original total runtime and soft budget. The
original root/report/clock remains immutable. Repeated failures and exhausted
authority produce actionable status, never invented grades or duplicate roots.
Validated corrections proceed through the existing agent QA and publication
readbacks; no Dot or parent review is required in the daily operating workflow.
Exports include `DATE-repair-N-{input,artifact}.json` with verified hashes.

The communications agent can import `Store` from the pinned
`tools/daily_research/firestore_bridge.mjs` and use an existing authorized
Firestore Admin instance. `await store.snapshot(date)` returns `{schema_version,
row, files, missing_files}`. Files are exact bytes encoded as base64. The method
checks chunk integrity and raw artifact binding. Python `render export` also
checks canonical evidence and review packet digests before writing a private
export. Missing files are explicit; corruption refuses. Do not consume a work
item as completed research until the exact root turn is completed, raw artifact
is saved and verified, and row is `awaiting_review`, `reviewed` or `completed`.

## Executable agent QA and publication

The same packaged scheduler executes `render.consume_workflow` → `Consumer.step`
for the durable queue. The `run`/`reconcile` CLI also consumes an eligible result.
It is independently disabled until root `enabled=true` **and**
`workflow.enabled=true`, with non-pending `qa_authority_reference` and
`publication_authority_reference`. The example disables both. These references
record actual scoped owner authority; they do not grant access by themselves.

QA sends one input event to the existing saved-agent session, after persisting
its exact input/CRM digest and obtaining a transactional one-use claim. It creates
no second session or new agent. The same saved Sol agent checks original source
support, semantic CRM duplicates, claim scope and explicit unknowns, writing an
exact-turn QA artifact. The consumer validates every candidate's check and does
another complete CRM read before selecting rows. Credentials and CRM contact
fields are excluded from the QA input. This is an agent attestation backed by
saved artifacts, not a claim that schema validation proves truth. In-time
terminal results can be collected after restart. Unknown/active QA is observed
and cancellation attempted on disable, observation failure, search-limit breach
or the **shared** research+QA deadline: 180 seconds for a legacy row, and the
row's own pinned total for an adaptive row (see [Runtime envelope](#runtime-envelope)).
Cold disabled recovery never
admits another input or publication. A cancel request is not terminal proof.
New ordinary intents also bind the original QA authority, exact message/key,
deadline, and complete pre-submission item/artifact inventory. A typed submission
HTTP503 permits at most two durable same-key attempts after 5/15-second minimum
backoff, honoring a longer Retry-After within that original deadline. Each attempt
rechecks the full inventory and fresh lease/control/stop/deadline before POST.
An accepted turn/message/effect, unknown failure, lost reply, changed authority,
or legacy intent without that binding permits observation only. A Retry-After
beyond the deadline suppresses replay while preserving GET/cancel recovery.

A readable but malformed QA artifact returns precise affected-field diagnostics
to that same saved session, with at most two corrective messages inside the
original QA deadline and total soft target. Corrections must reconsider actual
source/CRM evidence; strings are never coerced into boolean acceptances. The
original artifact, each corrective input and artifact, and the final evidence
for every attempt keep separate filenames and hashes in portable exports.
A lost or ambiguous submission reply is observed without another POST. Missing
terminal artifacts remain explicit collection failures. Disable, cancellation,
changed authority, expired lease/deadline and exhausted correction attempts
prevent another corrective message or publication. This behavior also applies
to the selected search profile and explicitly admitted recovered QA; a read-only
terminal collector cannot submit corrections.

QA prose is retained losslessly under the existing 2 MB artifact ceiling rather
than arbitrary summary/reason lengths. Notion uses the existing 1800-character
blocks, native request limits and exact full-content readback; Sheets never receives
the QA prose. Accepted evidence remains an unqualified discovery, not proof of
buying intent or pilot readiness.

Reports that exceed one Notion request use a digest-bound paginated plan in the
same company-owned ledger and one page under the existing research parent. Each
create/append request has at most 90 blocks and 450,000 UTF-8 JSON bytes; text
chunks preserve Unicode character boundaries. One durable claim precedes each
request, with a fresh workflow-authority and lease check before the write. Each
worker pass may advance one new batch. Ordered readback traverses all page
children and must match the complete original report before a receipt is issued.
After an uncertain reply, the same batch is never sent again: a later complete
ordered batch readback may admit the next batch. An unobserved write remains
pending; partial batches, conflicting content, repeated block IDs or duplicate
pages return explicit diagnostics without deleting, replacing or truncating
the source report. Existing single-request plans and receipts replay unchanged.
Portable exports include a separate hash-bound `publication-manifest.json` with
the exact plan JSON and consumed ordered claims; historical row and artifact
bytes are unchanged. Importing an identical row preserves its live company
manifest and unknown-write claims. The generic import command cannot resume
unfinished paginated publication into a new database without its verified claim
history; it refuses rather than inventing write authority. Completed records
with both acknowledged readbacks may be imported for archival inspection.

A first cancellation records its actual reason,
time, deadline and key before submission. Only an in-time completed result proven
to precede a later deadline-only cancellation may be collected with that history
intact; early, disabled, interrupted and unknown-timing cancellations stay blocked.
Legacy records keep their original five-activity search limit, counted across
research, QA and corrections. The selected adaptive search profile follows its
recorded coverage instructions instead of that legacy count stop. The recorded
soft TOTAL target includes every phase; model/search/environment usage and QA
turn evidence remain in the dated row.
An exhausted budget/time envelope blocks publication rather than adding a run.

Those are the legacy scan guards. The disabled
`adaptive-daily.config.example.json` opts newly admitted v3 rows into ten or more
new site/task opportunities, honest coverage/shortfall and a pinned research
phase within a shared research+QA total. That example keeps the original
20-minute research phase within a 30-minute total for the Oct 1 test; the
current production envelope is 60 minutes (see [Runtime envelope](#runtime-envelope)).
Agent QA excludes CRM
matches and existing deployments from that quota. The daily $1 soft TOTAL
target stays unchanged; old rows retain their original guards. See
[ADAPTIVE.md](ADAPTIVE.md) for the separate disabled $25 test preparation and
its unresolved total-spend/session admission. Installing this package does not
change live control or recover/retry the failed Oct 1 intent.

`publisher.mjs` handles only the two digest-bound deliveries. It preserves the
exact request plan and one-use attempt claim before each service POST. A timeout
or restart uses GET-only reconciliation; absence never authorizes a second POST.
Failed or conflicting readback blocks the result for Blueprint review.

- Sheets uses the existing Firebase service identity and the exact canonical CRM,
  `Prospects` first 19 columns. It checks the observed row-5 schema, complete
  identities, fresh snapshot equality and plain empty destination cells, then
  appends RAW values with unique `BP-######` IDs and a delivery marker. Existing
  rows, duplicate historical columns, formatting and contacts are preserved.
  New rows stay `Research` / `Needs recheck` / `Unverified`. No contact drafting
  or outreach occurs. Human concurrent edits can cause an ID/readback conflict;
  the consumer refuses success and never repeats an uncertain append.
  Existing nonblank rows must retain string values for organization, site,
  task evidence URL and task, matching the runner's identity-completeness gate.
  Incomplete identities also refuse a publication readback receipt.
- Notion creates one marked report child under Knowledge page
  `3eb80154161d8116858ed5f376b4b7a9`. It verifies the exact parent and report
  paragraphs before recording its receipt. The worker needs an authorized
  existing Blueprint integration in `NOTION_API_TOKEN` or `NOTION_API_KEY`, with
  **Read content + Insert content** and this page shared to it. Update content,
  workspace-wide access and new keys are not required by this publisher.
  Reconciliation scans the whole parent until `has_more=false`, bounded by
  100 pages of 100 children and a shared 25-second read budget including exact
  report verification. Missing/repeated cursors or an exhausted bound refuse
  publication; an incomplete scan never proves that a report is absent.

The existing service account's complete canonical Sheet read was verified
(HTTP200, complete 11-row CRM at 2026-10-01 01:19:46 UTC), followed by an
owner-approved, independently read-back **Editor** grant on that exact file.
The owner saved a narrowly scoped Notion worker key; its runtime page read is
still pending. Both services' actual publication writes/readbacks remain
unverified. These are live activation gates, not code prerequisites. The native MCP setup preparation is
separate; ChatGPT connections do not supply worker credentials. No live write,
QA event, canary or automatic consumer execution has yet been verified.

Both exact readback receipts complete publication. Cleanup still requires
Blueprint action-time approval naming the exact session/environment; unresolved
cleanup blocks another date. Communications consumes this snapshot and existing
approval gates; dot/parent is never a required runtime reviewer or publisher.

## Operator build/canary/cutover contract

The parent operator performs live actions; these are instructions for the
selected Blueprint service, not the user's local computer. No secrets are shown.
From `/opt/render/project/src`, all commands have the common prefix:

```bash
PYTHONPATH=dist/daily-research/release dist/daily-research/venv/bin/python -m tools.daily_research.render
```

1. Deploy the reviewed WebApp SHA with package-build true and worker-enable
   false. Read back the extracted `manifest.json` and vendored receipt SHA. GET
   binding was previously observed for the Default-project saved agent; Firestore
   writes/create and actual canary remain unproved until these steps succeed.
2. Prepare a private copy of `render.control.example.json` with exact packaged
   Pipeline `source_commit`. Keep `enabled=false`; initialize once with
   `init --input /PRIVATE/control.json`. Existing control refuses replacement;
   use `configure --input /PRIVATE/control.json` for a validated update.
   `configure` keeps the top-level `paid_expansion` owner direction, which only
   `operators/paid-expansion-direction.py` changes ([SEARCH.md](SEARCH.md#owner-directed-paid-expansion-allowance)),
   and the `site_universe` pin, which only `operators/site-universe-backlog.py` changes
   (see [Site universe slice](#site-universe-slice)).
   Deploy a release that changes a session tool schema only while the worker is
   idle. The 2026-10-04 release raises the Exa `max_cost_micros` maximum from
   5,000,000 to 50,000,000, and an in-flight session's tool check would refuse it.
3. Publish reviewed `knowledge.json` and approved `refresh-policy.json` with
   `publish-input --name NAME --input /PRIVATE/FILE`. The same command can import
   the latest reviewed `crm.json`; each new run refreshes the full canonical
   `Prospects` tab with the existing service account and readonly Sheets scope.
   `canonical_crm_read_unavailable` means this account cannot complete that exact
   read; report it, do not create a key or grant. Reconciliation bypasses fresh
   CRM reads so an older run can still be observed/cancelled.
4. Inventory and reconcile old automation `6abc4ffae84881919154bba45f749074` and
   prior sessions/dated intents. The exact prior smoke session
   `sess_0259dcecd393a414006abc7bc3444481939257e0f1dfc82829` was verified idle,
   one completed root turn, purpose `one_public_web_smoke_test`, no dated run key
   and **no hosted/self-hosted environment**. Preserve that receipt; it needs no
   environment deletion and does not clear other unknown sessions. The historical
   validation session cleanup receipt remains described in SKILLS.md. Import an existing runner state directory with
   `import-state --input /PRIVATE/STATE` while control is disabled. Conflicting
   dates refuse. Missing/ambiguous legacy history must be resolved before canary;
   never erase it by initializing another root. Record the actual reconciliation
   reference in control. Reserve one canary date and prevent the old trigger from
   creating that date. Do not enable both triggers.
5. Run `preflight`; require the full fresh CRM, field-specific knowledge policy,
   exact saved agent/template/skills and instruction pin
   `84aeea7cec9eb6e20d0d2fba10dcb269a615174e48ed91a60ff7a6a83ca37938`.
6. Put the real canary authority and reserved date in control, set `enabled=true`
   through `configure`, and manually invoke `run` after that date's Chicago 07:00.
   Keep the worker-enable flag false. Subsequent inspection uses `reconcile`,
   `status`, and `export --date DATE --output /PRIVATE/NEW_EXPORT_DIR`.
7. Require actual session/root-turn/environment IDs, terminal turn state,
   verified raw artifact/digest, agent QA, canonical duplicate checks, both
   publication readbacks and working Blueprint-owned retrieval. The $1 soft
   target covers model/search/hosted environment; it is not a hard cap.
8. Preserve artifacts. An authorized agent/operator requests action-time human
   approval for permanent deletion naming the exact session/environment through
   Blueprint's review surface. After the separately approved action, submit
   `record-cleanup` with IDs and `action_time_approval_reference`. Authenticated
   session/environment 404s must verify absence; billing-stop timing remains
   unproved. Unresolved cleanup blocks the next paid date.
9. At verified cutover, disable the old dot automation and read back disabled
   state. Record sole-trigger authority, enable the research worker flag, then
   verify the first real unattended 07:00 wake. Startup catches the latest due
   date, so account for that during activation. Daily operation is not claimed
   until that wake, agent publication and cleanup are observed.

Rollback: disable the research worker flag and Firestore control, preserve all
dated state and artifacts, and return to the previous WebApp release if needed.
Never restore an empty ledger or an older snapshot of run state.

## Same-session agent repair (Oct 2)

Owner request `Sentinel_3b6171ff167c8191b378202c5f0c54c0` authorizes an agent
repair loop for the already retained baseline. `repair-output --attempt 1
--date 2026-10-01` uses the same installed/isolated package and archive arguments
in [preserved baseline recovery](operators/README.md#preserved-baseline-output-recovery-and-corrected-cost-reporting),
under `timeout --signal=TERM --kill-after=60s 1860s`. It pins one immutable
correction-plus-QA window equal to the row's own `total_runtime_seconds` (30
minutes for that baseline row) against the exact original session/root/raw
hash and existing shared $25 soft baseline authority. Invoking it again resumes
that window; it never resets the root clock or creates another research session.
No additional `recover-output` or ten-minute QA receipt is required for this path.

The agent receives precise validation failures and retained evidence/context,
fresh CRM identity keys only, and writes a complete corrected report. It chooses
its strategy and configured tools, preserving supported work and marking gaps
honestly. Ordinary operator background requires no robot maturity grade;
employer-hosted job boards require semantic affiliation checks in agent QA.
Original bytes, failure and every correction input/artifact remain immutable in
durable state. Expected pending usage does not block correction or QA. Duplicate
effect claims, disabled/revoked authority, late cancellation and genuine lack of
progress remain bounded. An accepted request with a lost reply is observed by GET
and never resubmitted. A validated correction proceeds through existing QA and
canonical publication. Restart after QA/publication starts resumes that phase.

For the native fixture owner, `recovery.replay_saved_artifact(row, raw_artifact,
tool_files, known, observed_at)` is a pure offline full-contract replay. Supply
the complete retained row, exact raw report bytes, immutable source/tool files
keyed by filename, original CRM identity set and an aware observation time. It
verifies raw/context hashes, collects all independent feedback, derives only
evidence-bound precision and optional proposal quarantine, then revalidates the
entire report/coverage. The sanitized return includes remaining errors, derived
hash, counts and exact normalization receipts; it makes no provider/database or
publication call and cannot claim QA/newness. Keep the private fixture outside
the repository. After verified live QA/publication use `export-recovered` from
this overlay; it includes all hash-verified versioned repair inputs/artifacts.

Ordinary daily runs use this same loop automatically within their existing
workflow authorization, original total deadline and daily soft budget. They do
not require this one-time baseline command or any Dot/parent runtime step.

**Feedback is the gate itself.** `validation_feedback` reports every failure at
once from the same ordered rules the strict gate raises (`runner.output_issues`,
verified against a frozen copy of the previous gate). Each failure is located
to the exact field, and one defect never cascades into a second report.

**No progress keeps valid work instead of blocking it.** When a correction ends
in `no_progress` (an identical repeat, a failed or late turn, or no artifact),
the loop takes the best eligible revision: the original or an in-window
diagnosed correction. It excludes only items whose every failure is located
inside them (a candidate, a knowledge proposal or a summary line) and sends the
rest through the unchanged strict gate to the same QA and publication. Late or
unread corrections are retained but never used. Excluded items stay in
`validation_repair_outcome` and the packet's `research_exclusions`, and agent QA
reports them as rejected. Global failures, and corrections an operator or
deadline cancelled, stay blocked with the complete feedback.

**Approvals are data, not code.** The core window check binds a correction's
recorded approval to the row's own admitted baseline (id, soft total, budget
reference). A new approval needs `repair-output --authority-reference
EXISTING_APPROVAL`, never a release. The helper's default remains the reference
recorded above.

If a baseline correction expires without a confirmed turn, preserve its input,
claim, deadline and cancellation evidence. The existing provider-free
`recover-output` path may prepare the retained original report only after full
evidence-bound normalization and validation. Its separately authorized,
one-use 600-second recovered-QA continuation governs QA and QA tools; it never
extends the repair or research clock. Repeated arming cannot renew that window.
Before its first QA message, the same session must be idle without pending
actions and its complete turn inventory must match the retained completed turns.
No correction resend, new research session or deletion is required for this path.
Submission failures retain sanitized error metadata without exception prose or
request contents, and an uncertain QA input is observed without resubmission.

The isolated reviewed overlay's `recover-original-and-qa` command combines those
existing recovery, model-observation and QA steps for the explicitly authorized
baseline. It checks the saved agent, original session and complete completed-turn
inventory, derives and validates the original packet without inference, then pins
the separate 600-second QA receipt and runs existing QA/publication. It checks the
session and turn inventory again after consuming the one-use QA claim, before
submission. Restart follows that same receipt and claim; it cannot renew the
window, resend the expired correction, create another session or delete one.
Use the existing exact `timeout --signal=TERM --kill-after=60s 1860s` process
watchdog and the installed/overlay archive, source and file-hash arguments.


## Adjust the next run's duration

On the installed, source-pinned worker, the owner can change total duration without
building another package:

```bash
cd /opt/render/project/src
PYTHONPATH=dist/daily-research/release dist/daily-research/venv/bin/python -m tools.daily_research.render set-runtime --minutes 120
```

Use `--minutes 60`, `120`, `180`, `240`, or another whole-minute value from 2 to 240.
The command requires the adaptive profile and preserves the current QA reserve.
For example, with 15 minutes reserved for QA, 120 minutes means 105 minutes of
research plus 15 minutes of QA. Set both explicitly when needed:

```bash
PYTHONPATH=dist/daily-research/release dist/daily-research/venv/bin/python -m tools.daily_research.render set-runtime --minutes 120 --qa-minutes 20
```

The operation asserts the installed/control source pin, holds
the existing lease, and atomically requires no active research, QA, repair or
publication rows. It compare-and-swaps the current configuration and source.
Only the next run's duration and QA reserve change; existing rows keep their
original deadlines. It neither starts a provider nor changes the paid expansion
allowance, model/search budget, schedule, credentials or sending settings. The
current intended release setting is 60 minutes total with 15 minutes reserved for QA.
