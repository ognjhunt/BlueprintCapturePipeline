# Scoped research operator commands

Code and instructions are maintained in Blueprint's GitHub repository. Firestore
is the durable research ledger; provider IDs are provenance. Operator commands
must use an exact reviewed Git commit and the exact installed standalone package.
They have no ChatGPT Library dependency and require no Pipeline application,
GPU installation, new key, service, OAuth connection or security setting.

## Stopped terminal Sheets recovery

The existing private `firestore_bridge.mjs` JSON-line CLI accepts
`recover_terminal_sheets` under its ordinary acquired lease. Use the existing
worker credentials in place and literal `BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED=false`;
root and config `enabled` must both remain false. The request has exactly these
fields: `op`, `day`, `source_row_blob`, `payload_digest`, `rejected_call_id`, and
`publication_authority_reference`. Obtain the hashes and call ID from the retained
normal run and its exact failed Sheets/concise tool result. The authority must
match the original publication authority still stored in current control.

This operation uses no model/session calls. It verifies the original QA artifact,
accepted candidates, publication evidence and completed-before-deadline turn,
then publishes only the unchanged canonical Sheets payload. Existing claims are
readback-only; absent readback never authorizes another append. No Notion write,
presentation change, deadline extension, scheduling or control enable is allowed.
The original agent publication state and tool results remain unchanged.
Actual recovery time, bindings, plan digest and readback receipt are retained at
`blueprintDailyResearch/sites-first/terminalSheetsRecoveries/{source_row_blob}`.
Repeat the exact request to observe an uncertain receipt. Export that company
record with the original source blob and normal run snapshot for handoff.

## Disabled October 2 migration

`research-oct2-control.py plan` performs Firestore GETs and writes a new private
local plan (0600, exclusive create, fsync). It makes no provider calls and does
not take a lease. `apply-disabled` is a Firestore-writing operator action. It
uses the existing fenced lease, compares the full current control and retained
Oct1 row/raw with the plan, then configures and verifies exact readback. A stale
plan refuses; generate a new file rather than editing or bypassing its digests.

This helper is pinned to Pipeline `35f5c9ad43f84aa053aa7616a63a9aa4f6e32a61`,
archive SHA256 `1aa932767fe9ec73c06ece6b5ba1e573027a636a3249363d62df7bf6415a3651`
(409600 bytes, 41 source files). It verifies the archive, manifest, all installed
source hashes and actual imported module. WebApp
`b0cbd5e4c84120cf79ca704554433b8723a08b01` vendors this same receipt; operator
must verify the currently deployed installation rather than infer it from Git.

Run from `/opt/render/project/src` after materializing the unchanged reviewed
helper into a private directory using the existing authenticated operator route:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-oct2-control.py plan \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --output /tmp/blueprint-oct2-reviewed/live-disabled-plan.json
```

After scoped migration authorization, apply the same fresh plan:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-oct2-control.py apply-disabled \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --input /tmp/blueprint-oct2-reviewed/live-disabled-plan.json
```

Expected readback: root and config enabled are false, source is the exact pinned
Pipeline source, contract v3, search `perplexity-fast-v1`, discovery
`adaptive-sites-v1`, total runtime 1800 seconds, QA reserve 600 (research 1200),
soft total target $5 and recurring budget reference exactly
`Sentinel_c30352f247c88191bbd695cc2bd99de1`. That receipt approves model, search
and hosted environment total as a soft target; it is not a hard API billing cap.
This pinned helper still writes 1800/600. The 2026-10-04 owner decision later
raised the daily envelope to 3600 seconds total with 900 reserved for QA; see
[Daily research runtime envelope](#daily-research-runtime-envelope).

Preserve first_date, existing workflow settings/authorities, approval/scheduler
references, unrelated control fields, history, failed intent, raw bytes and
cleanup state. Existing lease bookkeeping is preserved through configure and
released normally. No provider session, run, input, publication, outreach,
deployment or deletion occurs. Root false makes the preserved workflow dormant.
The retained Oct1 date remains occupied, so an early restart cannot recreate it;
October 2 07:00 America/Chicago is 12:00 UTC. Enabled cutover is a separate action
after test, publication and cleanup evidence; do not simply flip a Boolean now.

## Private artifacts and portable handoff

Stable normal identity: `blueprint-researcher:YYYY-MM-DD`, with canonical document
`blueprintDailyResearch/sites-first/runs/YYYY-MM-DD`. Complete requests, raw
artifacts and tool evidence already survive restart in Firestore immutable blobs
and chunks. The installed `render export` command verifies bindings and exports
ordinary JSON status/artifact/evidence/review/QA/tool files. Export is read-only
for Firestore and writes a new private local directory; it is not publication.

Existing company object storage candidate is
`gs://blueprint-8c1ca.appspot.com`. The native runtime owner independently
verified a US Standard bucket, existing identity create/get/list, and no public
IAM bindings. Exact destination approval for private backup writes is still
required before any transfer. Do not provision credentials, grant access, make
objects public, mint Firebase download tokens, or delete existing copies.

Use a Blueprint-owned prefix such as
`blueprint-workflows/daily-research/sites-first/YYYY-MM-DD/exports/SHA256/`.
Publish a standard JSON manifest containing schema version, Blueprint workflow
and run IDs, file names/media types/byte counts/SHA256, source commit, object
generations and provider/session provenance. Save ordinary JSON/JSONL files
optionally gzip compressed, create-only (`ifGenerationMatch: 0`), with existing
private bucket/object access and authenticated exact-object hash readback.
Same-name retries read and compare, never overwrite. The manifest references
Blueprint IDs; no Library ID or provider session ID is a canonical identity or
required consumer lookup. Retain local/Library copies as secondary copies only.

## Fresh Perplexity canary

The private one-time adapter reuses the reviewed Store, leases, Runner, Consumer
and publisher, mapping only to
`blueprintDailyResearch/sites-first/canaries/perplexity-fast-20261001`.
It must leave normal control, the failed Oct1 run and Oct2 slot unchanged.
Stable test identity is `blueprint-research-canary:perplexity-fast-20261001`;
provider session, turn and environment IDs are provenance. Date-scoped Store
internals retain the reviewed `blueprint-researcher:2026-10-01` key. The test
input explicitly labels the brief as a test and prioritizes publicly verified
named decision contacts without expanding the strict output schema.

Required admission: exact existing $25 one-time approval reference, freshly
checked package/agent/template/tool/instruction/skills bindings, complete CRM,
checked knowledge/refresh policy, reconciled original cleanup receipt, canonical
publication authorities, normal root/config disabled. Test uses the production
$5 soft target and a separate one-time $25 soft total testing allowance; the
legacy `ceiling_usd` admission field records that allowance, not a hard provider
billing cap. It never replaces recurring authority or creates a timer.

Research deadline is 1200 seconds and total research+QA 1800 seconds, with 600
seconds reserved for QA. These are the canary's own pinned values, not the daily
envelope: `inspect` still requires the 1800/600 production profile, so after the
daily envelope changes to 3600/900 a new canary attempt refuses with
`canary_production_profile_not_migrated` until the canary has its own authority.
Existing independent Linux watchdog convention is
`timeout --signal=TERM --kill-after=60s 1860s`. Process stop/cancellation does not
guarantee provider teardown or a total-dollar cap. Model usage is observed for
baseline measurement; an arbitrary $8 stop and cancellation for absent usage
are removed. Selected tools retain existing resource ceilings, and actual
billing still needs reconciliation.
No fixed prospect quota; use defined scope, coverage and diminishing returns.

One durable create claim only, persisted full immutable intent first; an unknown
POST reconciles by GET and is never resent. One bounded QA turn in that session;
publication only after terminal evidence-backed agent QA and dedup. Recovery
must verify exact terminal turns, downloaded immutable artifacts, canonical
Sheets/Notion receipts and no duplicate publication. No Dot review/publication
dependency, outreach or permanent deletion. Cleanup needs its own exact
action-time approval; $25 test approval does not authorize deletion.

Use the exact reviewed GitHub commit containing `research-perplexity-canary.py`,
its sibling `.mjs`, and the unchanged `research-oct2-control.py`. Materialize all
three ordinary source files into `/tmp/blueprint-oct2-reviewed` through the
existing authenticated operator route and compare their reviewed SHA256 before
execution. Do not install the Pipeline application or deploy to a Pipeline host.
The adapter verifies the exact installed package receipt before every command.
Keep the production worker settings and normal research control disabled.

Create a private approval JSON file from the **existing** one-time $25 receipt;
use its exact evidence reference, never invent one or reuse recurring/deletion
authority. Required fields (the example reference must be replaced by the
actual already-approved receipt) are:

```json
{
  "schema_version": "blueprint.perplexity-canary-admission.v1",
  "test_id": "perplexity-fast-20261001",
  "authority_reference": "PENDING-actual-existing-one-time-25-receipt",
  "ceiling_usd": 25,
  "scope": "one-time-fresh-research-agent-qa-canonical-publication-no-outreach"
}
```

The fixed admission expires at `2026-10-02T10:00:00Z`; starting a new test after
that time refuses. Existing uncertain intent remains reconcilable and never
allows another create. The code does not change a template, account connection,
credentials, network policy, or production scheduler.

From `/opt/render/project/src`, run the complete read-only admission first:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-perplexity-canary.py inspect \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --approval /tmp/blueprint-oct2-reviewed/canary-approval.json \
  --output /tmp/blueprint-oct2-reviewed/canary-plan.json
```

`inspect` reads the complete canonical CRM, checks its digest and identity keys,
validates current knowledge/refresh policy and saved-agent/template/instruction/
four inline skill-file bindings using the pinned package's real preflight.
It verifies original cleanup and migrated controls. It makes no Firestore
writes or paid provider calls; only the new private plan file is written.
Read access alone does not prove publication writes or final report quality.
The authenticated runtime owner must also retain current exact canonical
Sheets Editor and Notion parent/integration permission evidence before execution.

Stage the same unedited plan (Firestore writes to the private canary namespace,
no provider inference or changes to normal control):

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-perplexity-canary.py stage \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --plan /tmp/blueprint-oct2-reviewed/canary-plan.json
```

Under the existing one-time test authorization, execute once with the exact
independent watchdog. The direct parent must be `timeout` with these arguments;
an unbounded invocation refuses before provider creation:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  timeout --signal=TERM --kill-after=60s 1860s \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-perplexity-canary.py execute \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar
```

For lost replies, interruption or restart, use the same bounded command with
`reconcile` in place of `execute`. It never creates a new session; it may finish
the already-authorized research tool responses, one QA turn, canonical
publication and authenticated readback. This recovery command is not read-only.
Use `status` with the same package/archive arguments for read-only sanitized
state, exact root/QA turn IDs and terminal states, artifact SHA256, cleanup
guard and destination readback receipts. Do not treat an idle session or process
exit as successful research.

For a new private standard-file export use `export` with
`--output /tmp/blueprint-oct2-reviewed/canary-export`. It is read-only for
Firestore and writes an exclusive private directory. It reuses the package's
artifact/evidence/QA/tool binding verification. Require complete terminal root
and QA artifacts, acknowledged canonical Sheets and Notion readbacks, and
reported missing-file review before calling the test successful.

Archive that export with a standard hash manifest to the authorized existing
company storage destination, read back the exact generation and hashes, then
obtain separate action-time approval for the exact test session's permanent
deletion. Record cleanup only after authenticated session and environment GETs
prove absence: use `record-cleanup` with the same package/archive arguments and
`--receipt /tmp/blueprint-oct2-reviewed/canary-cleanup-receipt.json`. The receipt
must name the exact test `session_id`, `environment_id`, and separate actual
`action_time_approval_reference`; retain stable Blueprint ID and private backup
manifest/URI/generation/hash evidence alongside it. This command reuses
`Runner.record_cleanup`, makes authenticated absence GETs, verifies retained
artifact/QA/evidence bindings, then writes only the private test cleanup record.
It never deletes a provider resource. The normal `render record-cleanup` CLI
addresses normal history and must not be used for this isolated test.

Before daily cutover, verify this private canary's cleanup guard
as well as normal dated history; normal production Store does not query the
private test namespace. Verify Oct2 remains unopened, the old Dot trigger stays
disabled, and only the standalone 07:00 America/Chicago trigger is enabled.
No canonical publication through Dot or ChatGPT Library is required.

## Baseline measurement and pending usage

The public [Agents API usage contract](https://developers.openai.com/api/docs/guides/agents-api/observability)
is best-effort: turn/session usage can be null, recorded counts may change, and
counts are not a final bill. Missing counts are neither zero nor evidence of
overspend. The measurement adapter records `usage_state=pending` and
`estimate_usd=null`; if some turns have counts, `reported_estimate_usd` describes
only that reported subset. A failed telemetry GET is explicitly `unavailable`.
Reported counts are labeled `reported_best_effort` and retain exclusions for
unreported usage, tools and hosted compute. Root and QA work may proceed under
the already-authorized soft-total scope while counts are pending; exact current
status, immutable phase deadlines, independent watchdog, one-create/one-QA
claims and resource ceilings continue to apply. No prospect-count stopping
rule or billing cap is introduced. Terminal reconciliation refreshes accounting
by GET without repeating research, QA or publication.

The first accepted test was cancelled by the historical missing-usage guard.
Its create has been consumed. This code correction preserves cancellation and
never clears an intent, resets its start/deadline, resubmits a create, or opens
another test identity. The runtime owner has retained its terminal/usage
diagnostics and completed separately approved cleanup; the baseline section
below records that newer receipt and renewed testing authority. Do not repeat
deletion or observation of the absent provider resources. Unknown charges stay
explicit. This fix alone is not paid-session or permanent-deletion authority.
Production remains disabled until real coverage/QA/publication, cleanup and
cutover are verified.

## Approved baseline with sequential retries

The consumed `perplexity-fast-20261001` test is terminal cancelled, deleted under
separate action-time approval, and durably recorded with cleanup false. Both
provider resources returned authenticated 404. Preserve its original intent,
row and private diagnostic backup; do not recreate it or reconcile a missing
provider session. This is a historical failed test, not a successful baseline.

The renewed approval `Sentinel_c2046c5f146c81918921eba1ed7f6caa` permits **another
$25 soft TOTAL across the baseline and retries**, including model, search and
hosted environment, with possible in-flight overshoot. The old test belongs to
the previous scope and is not charged against this new allowance. Recurring
production authority and permanent deletion remain separate.

The same reviewed helper now accepts `--attempt N --date YYYY-MM-DD`. It derives
`baseline-20261002-attempt-0001`, `...0002`, etc. Each attempt has its own
`blueprintDailyResearch/sites-first/canaries/TEST_ID` immutable intent and one
create claim. A shared coordinator at
`blueprintDailyResearch/sites-first/baselines/baseline-20261002` retains the one
approved allowance and ordered attempt references. There is no retry count
quota. Repeating `execute` or `reconcile` for the same identity never creates
again; a new numbered attempt is admitted only after its predecessor is
terminal, separately cleaned up, and has no unacknowledged publication. A
baseline with complete coverage, terminal QA and both publication readbacks
closes further attempts; a completed partial/interrupted report can be retried
after cleanup, without resetting its publication history. Staging compares the predecessor
blob and shared coordinator transactionally; overlapping callers cannot stage
different concurrent attempts or replenish the allowance. Normal production
control/history are unchanged.

The internal reviewed date key remains intact. Publication markers and the
bound Notion report provenance use `blueprint-research-canary:TEST_ID`, so even
identical same-date partial reports have distinct report titles/readback
bindings. Each retry refreshes the full canonical CRM before research and QA;
previously published prospects are deduplicated instead of appended again.

Create a private approval JSON from the actual renewed receipt, naming the
chosen attempt (this is not a request for another grant):

```json
{
  "schema_version": "blueprint.perplexity-canary-admission.v1",
  "test_id": "baseline-20261002-attempt-0001",
  "authority_reference": "Sentinel_c2046c5f146c81918921eba1ed7f6caa",
  "ceiling_usd": 25,
  "scope": "baseline-research-agent-qa-canonical-publication-with-retries-no-outreach"
}
```

Materialize the exact reviewed `.py`, `.mjs` and unchanged control helper into
a new private directory through the existing authenticated GitHub/operator
route and verify all three hashes. This is an operator source update only:
the installed standalone package remains pinned to `35f5c9ad...`, with its
manifest/archive/import receipt checked by each command. No worker restart,
deployment, package rewrite, normal control write, key or security change is
needed. Keep research disabled.

For October 2 before 12:00 UTC, the Chicago research due date remains
`2026-10-01`. Use that same bound date for this attempt's recovery even after
the calendar advances. For a genuinely new numbered attempt, choose the current
Chicago due date; admission rejects a stale date before a fresh create.
From `/opt/render/project/src`, after hash verification:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-baseline-reviewed/research-perplexity-canary.py inspect \
  --attempt 1 --date 2026-10-01 \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --approval /tmp/blueprint-baseline-reviewed/baseline-attempt-1-approval.json \
  --output /tmp/blueprint-baseline-reviewed/baseline-attempt-1-plan.json
```

`inspect` is read-only except its exclusive private plan file. Repeat the same
command with `stage`, replacing approval/output with
`--plan /tmp/blueprint-baseline-reviewed/baseline-attempt-1-plan.json`; this writes
only the new private attempt control/inputs and shared coordinator. After the
native owner clears the current deployment window, execute the admitted attempt
with the existing independent watchdog:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  timeout --signal=TERM --kill-after=60s 1860s \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-baseline-reviewed/research-perplexity-canary.py execute \
  --attempt 1 --date 2026-10-01 \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar
```

Use `reconcile` under the same watchdog for that exact attempt; it never creates
a replacement. `status` with the same attempt/date/package/archive is read-only,
reports all admitted baseline attempts and cumulative best-effort model/search
estimates, and keeps pending accounting and unverified total billing explicit.
There is no missing-usage cancellation or artificial dollar/prospect cutoff.
The $25 allowance is not multiplied by attempt count, estimated charges are not
final billed totals, and unknown tools/environment charges are not zero.

`export` and `record-cleanup` accept the same attempt/date arguments and retain
existing artifact/QA/readback/absence guards. Each new session's permanent
deletion requires its own exact action-time approval; the renewed $25 receipt
does not authorize it. A retry uses N+1 and a fresh approval JSON/immutable plan
under the SAME renewed budget reference, preserving all preceding attempts and
reported exposure. No automatic fresh retry, outreach, scheduler activation or
additional paid analysis is introduced by installing the helper.

If an attempt was staged but never acquired a durable run intent and its due
date advances, `abandon-unstarted` with that exact attempt/date/package/archive
atomically proves no run document/create claim exists and no active lease,
disables only its private control, and records an abandoned admission in the
shared coordinator. It makes no provider calls and fabricates no terminal run
or cleanup receipt. The next numbered attempt can then use the current due
date. An existing intent, uncertain POST or active lease refuses abandonment;
those retain their existing reconciliation and separately approved cleanup
requirements. Repeated abandonment is idempotent and never re-enables a slot.

## Preserved baseline output recovery and corrected cost reporting

This repair addresses the retained Oct 2 baseline artifact
`011c09c6e5900e910c65c71a7852a8a8aefb4b394a98abd91e8c4c9498085a36`.
The observed third knowledge proposal has operator sources and null evidence
levels at `/proposed_knowledge_deltas/2/evidence/{0,1,2}/evidence_level`.
The candidate site-evidence contract permits null; the robot-knowledge proposal
contract does not. Never substitute a demonstration/deployment grade. The
explicit recovery quarantines that entire optional proposal, preserves raw and
normalized output, and strictly validates every remaining field/candidate before
preparing QA. Quarantine does not establish source support or newness.

The CRM snapshot was retained at admission outside the research payload. The
adaptive prompt replaces the original task envelope, so the basic CRM note is
not necessarily present. `diagnose-output` inspects the exact saved payload and
context hashes, verifies raw bytes, and replays both original and quarantined
validation with no provider calls or Firestore writes. Search/page counts alone
cannot explain the one-candidate result; QA must examine actual scope, findings,
blockers, stopping reason and duplicates. It receives fresh canonical CRM
identities through the existing reader. The future prompt explicitly separates
nullable candidate evidence from nonnull robot-knowledge proposal levels.

Use the final reviewed repair archive in a **new temporary directory**, leaving
`dist/daily-research` unchanged. The repair archive includes the operator helpers.
Use its `release` root on `PYTHONPATH`, with the installed package's existing
Python interpreter/SDK. Every repair command verifies both the old installed
archive/manifest and the new repair archive/source/file/import hashes. Supply:

```bash
PYTHONPATH=/tmp/blueprint-research-repair/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-research-repair/release/tools/daily_research/operators/research-perplexity-canary.py \
  diagnose-output --attempt 1 --date 2026-10-01 \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --repair-package /tmp/blueprint-research-repair/release \
  --repair-archive /tmp/blueprint-research-repair/blueprint-research.tar \
  --repair-source FULL_REVIEWED_COMMIT --repair-sha256 REVIEWED_ARCHIVE_SHA256
```

Run this read-only diagnosis first. Require the exact artifact SHA above,
original error `knowledge_delta_evidence_invalid`, three exact invalid pointers,
matching immutable knowledge/policy context, and successful derived validation.
Any other error is a blocker. No assertion of verified coverage/newness is made.

Use the same verified command arguments with `reprice` next. It makes only GETs
for this existing session's turns and writes the corrected private estimate,
retaining the unversioned estimate in `canary_model_estimate_history`. Version2
uses mutually exclusive input/cache-read/cache-write rates. For the recorded
root usage (3,363,036 input, 3,224,626 cached, 15,233 output including reasoning),
standard/global recorded-token scenarios span $0.7516126–$1.5654702. Unknown
per-request context/cache writes are ranges, regional premium/hosting/tool fees
stay separate, and billed dollars remain null. The old $33.5454009 estimate is
excluded from current baseline totals; neither amount is an invoice.

`recover-original-and-qa` is the explicitly paid, combined recovery path for the
existing approved baseline. It derives the retained original report, records fresh
model usage, and starts the already authorized one-use 600-second QA continuation
in the same session before existing publication. The expired correction and its
claim/cancellation evidence remain unchanged. The command requires the isolated
reviewed overlay arguments and exact existing process watchdog; it cannot create
or delete a session or resend a correction. Restart observes its existing QA
receipt and input claim without renewing either. Fresh session/complete-turn
checks run both before packet recovery and after claiming the QA input.

`recover-output --receipt PRIVATE_JSON` is provider-free but writes the private
run and immutable `DATE-recovery.json`. Its exact receipt shape is:

```json
{
  "session_id": "sess_06ea8f997fa27202006abf0b37b9f4819aacfaa2cb1414eb14",
  "turn_id": "EXACT_RETAINED_ROOT_TURN_ID",
  "raw_output_sha256": "011c09c6e5900e910c65c71a7852a8a8aefb4b394a98abd91e8c4c9498085a36",
  "approval_reference": "Sentinel_dac3e21091cc819196cb4e5799b7229d",
  "scope": "quarantine-null-operator-deltas-no-inference-no-publication"
}
```

Require `awaiting_review`, one quarantined proposal, unchanged root session/turn,
unchanged raw SHA and zero provider/publication mutations. This action does not
start QA. Original failure and excluded evidence remain in the durable recovery
receipt and portable export, without modifying the original output files.

The original research window can already be exhausted. Do not reset its
`started_at`, runtime fields, completion time or original cost history.
`authorize-recovered-qa --receipt PRIVATE_JSON` instead pins one new ten-minute
QA continuation under the existing shared $25 baseline allowance. It performs
no provider calls; repeat invocations never extend the pinned window. Its receipt
contains exactly: `authority_reference` (the approval above),
`scope`=`same-session-recovered-qa-and-existing-publication-no-new-research`,
`baseline_id`=`baseline-20261002`, `soft_total_usd`=25, exact `session_id`,
`root_turn_id`, `raw_output_sha256`, current recovered `packet_digest` and the
canonical SHA256 `model_observation_digest` of the corrected private estimate.
It refuses unknown/unversioned model usage and mismatched session/artifact/cost
bindings. Budget authority remains a shared soft target, not a newly claimed cap.

Only after the runtime owner assesses those receipts may it invoke `resume-qa`
with the same pinned repair arguments and the existing reviewed external process
watchdog. This is an explicitly paid command. It requires an existing recovered,
authorized session, has no root-create path, sends at most the already-fenced
single QA input, and uses the original source/duplicate checks and publication
readback gates. Unknown replies are observed, not blindly resubmitted. Expiry
blocks new QA and still observes/cancels an existing QA. Verified publication,
canonical export, cleanup approval and dated-ledger reconciliation remain
required before enabling the normal schedule. No permanent deletion is authorized
by recovery or the renewed baseline allowance.

After terminal QA/publication, use `export-recovered --output NEW_PRIVATE_DIRECTORY`
with the same repair arguments. This provider-free overlay export verifies and
includes `DATE-recovery.json` alongside original artifact/output, QA, evidence and
tool receipts. The old installed export omits the new receipt and is insufficient
for this repaired run's canonical backup. The read-only diagnosis also returns
actual declared coverage, stopping reason and bounded root query/source trace;
those observations, rather than search counts, support the scope audit.

`retry-qa-submission` is the explicitly authorized recovery of the retained
HTTP503 QA input (`req_c88b962feb724dbaa72643721e4745be`), not new research.
It binds the exact final row digest, full baseline-only session/turn/items/artifact
GETs and unchanged immutable QA input before pinning one new 600-second phase.
The expired `qa_continuation` and exact prior QA/cancellation record remain
unchanged in `qa_retry_continuation.previous_qa`; their clocks are never renewed.
The native operator's completed observer and read-only reconciliation are the
source receipt; flags alone do not prove cancellation completion, and the new
receipt explicitly records the previous cancel outcome as `not_inferred`.
An explicitly unresolved cancel reply refuses admission.

At most two durable transactional slots resend exactly the original message,
session and idempotency key. Each checks complete saved work before and after
its claim, then rechecks private lease, normal origin, controls, stop and deadline
immediately before POST. Backoff floors are 5s/15s and honor HTTP `Retry-After`;
long hints stop the phase. SDK retries stay disabled. Accepted work is observed;
a crash, unknown reply, changed error or lost reply-persistence receipt never
grants the next slot. The original input claim remains consumed. The retry phase
has a separate stable cancellation-operation key so the earlier cancel response
cannot deduplicate its own deadline/disable cancellation. Restart cannot renew
the phase. Existing QA, deduplication, publication and readback gates apply;
successful command exit alone never establishes QA or publication completion.

Official contracts: [retry transient failures](https://developers.openai.com/api/docs/guides/agents-api/errors#retry-transient-failures)
and [same-key session input recovery](https://developers.openai.com/api/docs/guides/agents-api/sessions).
No new session, model change, grants, keys, outreach or deletion is included.

## Collect the retained in-time QA result without inference

`collect-completed-qa` is scoped to baseline attempt 1 and its exact native
source-blob, QA-turn and raw-artifact hashes. It GETs the three exact immutable
Firestore snapshots and verifies their raw bytes, native creation times and
pre-cancel/intent/reply flags. The authenticated native receipt records completion
at 09:38:27 UTC, before the 09:39:14.788518 deadline and the cancellation dispatch
interval 09:39:28.549726–09:39:29.639687 UTC. The exact request time remains unknown.
The receipt references the existing private GCS export and native metadata;
it does not invent a `cancel_record`, erase cancellation flags, resend the QA
input, create another phase, extend a clock, or admit ordinary unknown-timing
cancellations.

Under the existing baseline QA/publication authority, the command performs full
GETs, checks the same completed turn, idle/no-action session, retained raw bytes,
immutable input, exact evidence and complete inventory. It runs the existing
source/claim/identity attestation and fresh CRM check, preserving full QA prose.
The original QA/cancellation history is retained inside an immutable recovery
receipt. Fresh normal-origin, private control, workflow-authority and lease
checks precede validation and the existing canonical publication. Readback must
match the full Notion report and exact Sheets rows; accepted discoveries remain
unqualified. Restart may finish those same claimed publications and never
starts provider work. A changed source or receipt refuses instead of rebinding.

Use the verified isolated portable archive and existing SDK/dependencies,
with `collect-completed-qa` in the standard watchdog invocation above. This is
a Firestore-writing validation and canonical-publication action, not a read-only
probe. Keep the production scheduler disabled until actual QA validation,
both publication receipts, and `export-recovered` are verified. No worker
deployment, Pipeline host deployment, paid GPU job, outreach or deletion is
part of this command.

## Owner paid expansion allowance

`paid-expansion-direction.py` sets the combined per-run allowance for paid
expansion sources (Exa now; FindAll later). The 2026-10-04 owner decision is $10
per daily research run, separate from the unchanged $5 research soft target.
The amount is data; a new value needs no code change, redeploy or package. It is
the only binding number: each start may use up to half of it ($10 allows a $5
Exa start, $20 a $10 start, $30 a $15 start; never above $50). A raised amount
applies from the next run. A lower amount or the brake applies to the run in
progress.

Run it from the installed release with the existing worker environment
(`FIREBASE_SERVICE_ACCOUNT_JSON`, `node`); it is packaged with the release.

```bash
RELEASE=/opt/render/project/src/dist/daily-research/release
COMMAND="/opt/render/project/src/dist/daily-research/venv/bin/python $RELEASE/tools/daily_research/operators/paid-expansion-direction.py"
PYTHONPATH=$RELEASE $COMMAND show
PYTHONPATH=$RELEASE $COMMAND set --per-run-usd 20.00 --approval-reference REF
PYTHONPATH=$RELEASE $COMMAND set --per-run-usd 20.00 --approval-reference REF --apply
PYTHONPATH=$RELEASE $COMMAND set --per-run-usd 10.00 --sources findall --approval-reference REF --apply
PYTHONPATH=$RELEASE $COMMAND set --per-run-usd 10.00 --sources exa --approval-reference REF --apply
PYTHONPATH=$RELEASE $COMMAND disable --apply
```

- `show` reads control and the `paidExpansionDirections` audit chain and verifies
  the current object. It writes nothing.
- `set` is a dry run without `--apply`. It prints the next direction: version+1,
  superseding the current SHA-256. Amounts must match `^[1-9]\d{0,2}(\.\d{2})?$`
  within $1.00–$100.00 (`20` becomes `20.00`). Optional flags: `--approved-by`,
  `--reason`, `--expires-at` (UTC, at most 366 days; default 90 days) and
  `--expect-current SHA|none` to pin the transition shown by a dry run.
- `set --apply` writes `gs://blueprint-8c1ca.appspot.com/operations/research/paid-expansion/<sha256>/direction.json`
  create-only through the existing bridge identity and reads it back. It then
  takes the fenced lease only for the `paid_expansion_set` compare-and-swap and
  reads control back. A concurrent change refuses with
  `paid_expansion_direction_conflict`; run `show` and plan again. While a run or
  QA is active it refuses with `paid_expansion_run_active_apply_after_run`;
  `--during-active-run` overrides this (a lower amount then also tightens the run
  in progress). The new direction supersedes control's current one, or the audit
  head if a rollback dropped control's copy. A current direction this package
  cannot verify is reported as `current_problem` and can still be replaced.
- `--sources` chooses which paid sources the direction admits: `findall`, `exa`
  or `exa,findall`. Without it, `set` admits every supported source. A start of
  a source that the current direction does not name is refused with
  `paid_expansion_source_not_directed`, and research continues. Switching
  sources is the same `set` command with another `--sources` value. Adding a
  source applies from the next run, like a higher amount. Removing a source, like
  a lower amount or the brake, applies at once: new starts of that source are
  refused, and active FindAll runs are cancelled at the next observation.
- FindAll has no provider-side dollar cap. Its create request carries
  `match_limit`, and the host reserves the generator's listed fixed price plus
  per-match price times `match_limit` (pricing version in
  `src/blueprint_pipeline/parallel_findall_execution.py`). The limit holds at
  Parallel's list price with Parallel enforcing `match_limit`. Exa's
  `budget.maxCostDollars` is a provider-enforced cap. The FindAll instructions
  start with base requests of about 50 matches ($1.75 each). Larger or repeated
  requests can use most of a $10 allowance and leave no room for an Exa start, so
  set `--per-run-usd 20.00` when both sources should run every day.
- `show` and `set` report `source_readiness`: for each source, its credential
  binding name (`EXA_API_KEY` or `PARALLEL_API_KEY`), whether that binding is
  present in this worker process (presence only; no value is read or printed)
  and whether the current direction names it. `set` adds the warning
  `paid_expansion_source_binding_missing:<source>` when it names a source whose
  binding is missing, because every run skips that source until the key exists.
- `disable --apply` is the emergency brake. It stops new paid starts at once,
  including in an active run, and needs no object write. Re-enable with a new
  `set`; to correct a mistaken amount mid-run, brake and then `set` the lower
  amount.
- Both commands poll every 0.25 s, for up to 200 s, for the worker's next lease
  release. They hold the lease only for the swap, so they never displace an
  active worker.

All I/O goes through injectable adapters; the hermetic tests use the real bridge
with in-memory Firestore and a fake object store. No provider, model, session,
CRM write or send.

## Site universe backlog slice

`site-universe-backlog.py` pins one reviewed site universe export as optional
prioritization for the daily run (ADP-010 partner discovery). It is off until the
owner pins an export; with `control.site_universe` absent or `enabled=false` the
run reads nothing more and its create payload, metadata, instructions and row are
byte-identical to a release without this feature. The export stays internal
(ODbL): the slice goes into the hosted sandbox for processing, never into
findings or publication.

```bash
RELEASE=/opt/render/project/src/dist/daily-research/release
COMMAND="/opt/render/project/src/dist/daily-research/venv/bin/python $RELEASE/tools/daily_research/operators/site-universe-backlog.py"
PYTHONPATH=$RELEASE $COMMAND show
PYTHONPATH=$RELEASE $COMMAND publish --file /PRIVATE/backlog.v1.json.gz
PYTHONPATH=$RELEASE $COMMAND publish --file /PRIVATE/backlog.v1.json.gz --apply
PYTHONPATH=$RELEASE $COMMAND pin --sha256 SHA --generation GEN --approval-reference REF --slice-size 20
PYTHONPATH=$RELEASE $COMMAND pin --sha256 SHA --generation GEN --approval-reference REF --slice-size 20 --apply
PYTHONPATH=$RELEASE $COMMAND disable --apply
PYTHONPATH=$RELEASE $COMMAND funnel --days 7
```

- Every command except `show` and `funnel` is a dry run unless `--apply`. Output
  shows counts, SHA-256s, generations and site ids, never site names.
- `publish` validates the local file with the runtime loader
  (`site_universe.load_export`: canonical bytes, `rows_sha256`, at most 2 MiB gzip,
  6 MiB raw and 5,000 rows) and prints the URI and SHA-256. `--apply` writes
  `gs://blueprint-8c1ca.appspot.com/operations/research/site-universe/<sha256>/backlog.v1.json.gz`
  create-only through the existing bridge identity, reads it back and prints the
  object generation. A published export grants nothing until it is pinned.
- `pin` reads exactly that generation, runs the loader and a dry-run selection
  for the next run date with current history and the canonical CRM, then takes the
  fenced lease only for the `site_universe_set` compare-and-swap and reads control
  back. `--slice-size` is 5–30 (default 20) and at most the research window
  divided by 90 seconds (2700 s allows 30). `--reoffer-after-days` is 1–365
  (default 90). `--approval-reference` must not be `PENDING`. An empty dry-run
  slice refuses with `site_universe_slice_empty`. `--expect-current SHA|none` pins
  the transition a dry run showed; a concurrent change refuses with
  `site_universe_control_conflict`. While a run or QA is active it refuses with
  `site_universe_run_active_apply_after_run`; `--during-active-run` overrides this.
  A pin applies from the next create; a running row keeps what it froze.
- `show` reads control, verifies the pinned object and prints the dry-run
  selection. `ready: true` means the next run would attach a slice;
  `enabled_unusable` gives the code the run would record.
- `disable --apply` keeps the pin with `enabled=false`, so the next run reads
  nothing more. Re-enable with `pin`. Like `pin`, it refuses with
  `site_universe_run_active_apply_after_run` while a run or QA is active;
  `--during-active-run` overrides this. A running row keeps the slice it froze
  either way.
- `funnel --days 7` lists each run date's state and code (`attached`, `refused`,
  `exhausted` or `off`) and its funnel, and sums the counts: selection, agent work
  (touched, screened, researched with a gap, inventory and formal candidates,
  rejected, learning, duplicate, agent-added), QA (eligible for promotion and
  accepted, from the slice and for the whole run) and run level (time, calls,
  research evidence bytes, paid reservations). Contacted, replied and
  conversations are null because the run sends nothing; per-site cost is
  `not_measured`.

The run records `row.site_universe` and `packet.site_universe`; status shows the
state and code. Any slice failure (`site_universe_pin_invalid`,
`site_universe_slice_exceeds_research_window`, `site_universe_object_*`,
`site_universe_export_*`, `site_universe_history_binding_invalid`,
`site_universe_intent_resource_ceiling`, `site_universe_profile_unsupported`,
`site_universe_attach_unavailable`) records `{state: "refused", code}` and research
continues exactly as without the slice; an empty selection records `exhausted`.
Near the intent ceiling the record shrinks to `{state, code}` or is dropped, so a
pinned run never fails where an unpinned one would succeed. A lost store or lease
still stops the run. The hermetic tests use the real bridge
with in-memory Firestore and a fake object store. No provider, model, session, CRM
write or send.

## Daily research runtime envelope

The 2026-10-04 owner decision gives each daily research run 60 minutes in total:
`config.max_runtime_seconds=3600` with `config.qa_reserved_seconds=900`, so
research has 2700 seconds and QA keeps 900. The $5 soft target is unchanged.
`runner.MAX_ADAPTIVE_RUNTIME_SECONDS` (3600) is the only adaptive bound; see
[RENDER.md](../RENDER.md#runtime-envelope) for every value that derives from it.
Each row keeps the envelope it was admitted with, so a row admitted at 1800
seconds is never extended.

The installed release must contain this bound before the control changes. An
older release refuses a 3600 config on every scheduler tick with
`approved_envelope_mismatch`. Deploy the worker while research is idle, outside
07:00–08:05 America/Chicago. Then read the complete current control document
and change only the two config fields. `source_commit` must name the installed
release; if it does not, set it in the same input. Apply the document with the
existing fenced `configure` command from `/opt/render/project/src`:

```bash
PYTHONPATH=dist/daily-research/release dist/daily-research/venv/bin/python \
  -m tools.daily_research.render configure --input /PRIVATE/control.json
```

`configure` validates the input with the installed release and replaces the
whole control document except the lease, `cleanup_observation_required` and
`paid_expansion` and `site_universe`. Never apply a partial document. Read the control back and run
`preflight` with the same prefix before the next 07:00 run.
