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

## Stable private consumer contract

Contract `blueprint.research-snapshot.v1`, root
`blueprintDailyResearch/sites-first`:

| Record | Contents |
| --- | --- |
| `runs/YYYY-MM-DD` | Immutable row blob pointer, date/state, metadata binding, cleanup guard, actual session/turn/environment IDs, one-use create claim |
| `blobs/SHA256` + `chunks/N` | Exact uncompressed SHA256/length, gzip metadata, immutable chunks at most 256 KiB each; read verifies all bytes |
| `files/YYYY-MM-DD-{artifact,evidence,output,review,qa,qa-evidence}.json` | Blob pointers; downloaded raw artifact is immutable once bound |
| `files/{crm,knowledge,refresh-policy}.json` | Private checked input pointers, not public prospects |
| `workItems/YYYY-MM-DD` | `owner=blueprint-research-qa-publication-agent`, `stage=agent_qa_pending` or `publication_pending`, run key, row blob, packet digest, `observer_receipt_required=false`, no-outreach scope |

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
or the **shared** research+QA 180-second deadline. Cold disabled recovery never
admits another input or publication. A cancel request is not terminal proof.
The five-activity search limit and $1 soft TOTAL target include both phases;
model/search/environment usage and QA turn evidence remain in the dated row.
An exhausted budget/time envelope blocks publication rather than adding a run.

Those are the legacy scan guards. The disabled
`adaptive-daily.config.example.json` opts newly admitted v3 rows into ten or more
new site/task opportunities, honest coverage/shortfall and a pinned 20-minute
research phase within a 30-minute research+QA watchdog. Agent QA excludes CRM
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

## Shared learning caller

The existing research clock can run deterministic business analysis at **06:45
America/Chicago**, before the unchanged 07:00 research date. The WebApp bundles
`server/research-learning/research-worker-host.ts` as `dist/research-learning/research-worker-host.js`
and passes its local path through the existing private Python/Node bridge.
This creates no daemon, network endpoint, model-analysis call or ops-scheduler
dependency. `server/worker.ts` and communications scheduling are unchanged.

Activation uses optional, host-owned `control.learning`, validated by WebApp's
`researchLearningControlSchema`: version `blueprint.research-learning-worker.v1`,
`enabled: true`, `startDate`, existing consumer `binding` and `selection`,
`businessScope`, `learningGrant`, and `terminalSubjectKey`. Omission or
`{enabled:false}` retains the prior runner. The research worker flag still owns
process startup. An enabled learning binding can aggregate while provider
research is disabled; it does not authorize provider creation.

The native source owner supplies the reconciled source-snapshot hash, exact
CRM/prospect/capability IDs, authorized business subject keys, principal and
actual grant expirations. All five outcome sections must already be authorized;
the caller neither broadens nor renews grants. Grant renewal/reconciliation
belongs to Blueprint's authorized control plane and review surface, with no
required Dot task. Original business messages must be captured with original
IDs/times/source hashes by their existing owner; this caller seeds no summaries
as historical evidence. No new key, OAuth grant or external permission is needed
for these Firestore-only hooks.

Daily identity is `research-learning:YYYY-MM-DD:0645`, with `asOf` bound to that
date's Chicago 06:45 instant. Restart/retry reuses the same retained overview
receipt. The scheduler reads only the small control on idle ticks; aggregation
and bounded native-terminal reconciliation occur at startup, the learning date
change or recovery, rather than every minute. At most 100 native projections
from `startDate` are reconciled before an explicit scoped export/reconciliation
is required. Private baseline/canary roots are excluded.

Before a new create, the real overview and relevant business, outcome, native
research and site history are captured from one scoped consumer closure. The
input has exact UTF-8 JSON bytes and SHA-256, original source dates, explicit
unknowns, up to 500 events per selected history, and a 600 KB resource limit.
Any remaining pages are explicit; cached directory coverage stays partial.
An enabled binding with no verified overview refuses creation. Hypotheses are
provisional and never automatic prospect filters. The input, hash and full
create payload are persisted in the dated immutable intent before the provider
call. The final create claim rechecks the learning binding and expiry. Recovery
keeps the original captured context and does not reread it or create again.

After a native terminal state, the caller invokes the existing deterministic
`recordTerminalRun` against the exact `runs/YYYY-MM-DD` projection/hash. Stable
Blueprint observation IDs retain each source state; counts, finish times and
delivery remain unknown when unsupported. Retry is idempotent and changes no
native source or approval state. `render export` retains the full context in
`status.json`; canonical history/overviews and their linked snapshots live
under `blueprintResearchLearning/default`, exportable as standard JSON with
source hashes to existing company storage. Derived review payloads are not
connector-write receipts.

Release only after the active baseline settles: preserve research disabled,
review/verify the new archive and compiled host, reconcile the actual binding
and original-message inputs, then exercise scoped offline aggregation/readback
without provider calls. Verify the captured context before a separately
authorized daily cutover. Never replace a running test's installed package.
