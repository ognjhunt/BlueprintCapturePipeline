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
create payload, metadata, instructions and row are byte-identical to a release
without this feature. The pin comes from the control read the run start already
makes when the history profile is on; otherwise the run makes one extra control
read and nothing else. The export stays internal
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
  6 MiB raw and 5,000 rows, and only the reviewed licenses `CC0-1.0`,
  `ODbL-1.0`, `US-Gov-Work` and `US-PD`) and prints the URI and SHA-256. `--apply` writes
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
pinned run never fails where an unpinned one would succeed. A prior row offered a
slice only if its intent bound one (`metadata.site_universe_slice_digest` and the
slice file). Without a trusted outcome block, every site that slice offered stays out
for the window (the funnel counts it as `history_rows_untrusted` with
`history_codes`); a bound slice file that no longer verifies refuses with
`site_universe_history_binding_invalid`. A lost store or lease still stops the run. The hermetic tests use the real bridge
with in-memory Firestore and a fake object store. No provider, model, session, CRM
write or send.

## Outreach-ready hypothesis direction

`outreach-ready-direction.py` is the only switch for outreach-ready hypotheses in
daily QA (ADP-010 partner discovery; owner decision 2026-10-05; design v1.1, rule v1.2). Without
an enabled `daily_qa` direction every run is in shadow mode: the row, create payload,
packet, QA input, review, publication payloads and status output are byte-identical
to a release without the feature. After publication completes, the tier is recorded
only in the ledger file `<date>-outreach-ready-shadow.json`, and nothing is admitted.

```bash
RELEASE=/opt/render/project/src/dist/daily-research/release
COMMAND="/opt/render/project/src/dist/daily-research/venv/bin/python $RELEASE/tools/daily_research/operators/outreach-ready-direction.py"
PYTHONPATH=$RELEASE $COMMAND show
PYTHONPATH=$RELEASE $COMMAND set --approval-reference REF
PYTHONPATH=$RELEASE $COMMAND set --approval-reference REF --apply
PYTHONPATH=$RELEASE $COMMAND disable --apply
```

- `show` reads control, verifies the pinned object generation and prints what the
  next run would freeze (`shadow`, `refused` with a code, or `enabled`; a screen-only
  direction shows `shadow` with `outreach_ready_daily_qa_not_directed`). It writes
  nothing.
- `set` is a dry run without `--apply`. It prints the next direction: version+1,
  superseding the current SHA-256, with scope `paths` (default `daily_qa`;
  `daily_qa,site_screen` also names the screen admission once it ships), label
  `hypothesis`, `--max-rows-per-batch` (1-50, default 50) and
  `sends_authorized:false`. `--approval-reference` names the owner record and must
  not be `PENDING`. `--expires-at` is UTC, at most 366 days (default 30 days).
  `--expect-current SHA|none` pins the transition a dry run showed.
- `set --apply` writes
  `gs://blueprint-8c1ca.appspot.com/operations/research/outreach-ready/<sha256>/direction.json`
  create-only and reads that generation back. It then takes the fenced lease only
  for the `outreach_ready_set` compare-and-swap, which checks the pinned
  generation's bytes and the version chain, and reads control back. A concurrent
  change refuses with `outreach_ready_direction_conflict`. While a run or QA is
  active it refuses with `outreach_ready_run_active_apply_after_run`;
  `--during-active-run` overrides this.
- A run freezes the direction under its lease at the durable intent, and the row
  can never change or drop that record later. An enabled direction that names
  `daily_qa` pins lead-verification result v3 for the row and asks QA for
  `outreach_ready_keys`. An unusable, expired or not-yet-effective `daily_qa`
  direction records `{state: "refused", code}` and the run stays in shadow mode. A
  screen-only direction freezes nothing, so daily rows and their manifests stay
  exactly as without it. If a rollback's older bridge drops the record's manifest
  digest, the row is not stranded: it binds the record again, stays
  `outreach_ready_unbound` and publishes no hypothesis.
  Saved plans without a write claim are rebuilt with verified rows only; consumed
  plans keep exact GET-only readback without any further Notion batch claims.
- Admission at the QA decision also needs the live pin. `disable --apply` is the
  brake: it keeps the pin with `enabled=false` and applies at once, including to a
  run in progress. **The brake does not stop hypotheses that review already bound:**
  once the protected review has written them into the publication payloads, that
  day's publication still writes them (labelled, draft only). To keep them out,
  apply the brake before that day's QA decision. A new direction can lower a running
  row's row limit or remove its path, never widen it. Re-enable with a new `set`.
- Admitted keys publish only through the row's existing publication path
  (agent-owned in production): Sheets rows with Verification `Hypothesis` and
  Notion entries labelled "Hypothesis, not verified", each with exactly one
  question (template S, M, A or U). A hypothesis that is malformed or no longer current
  at publication is left out on its own reason; it never blocks that day's verified
  rows. Nothing authorizes a send.
- Enable a direction only after the WebApp release that accepts result v3 and the
  v1.2 `hypotheses` payloads (#855, #862) is deployed. An older WebApp refuses every day
  whose payload contains hypotheses, including that day's verified rows.
- A direction names the rule of the release that set it (`rule_version`). After a
  release that changes the rule (v1.1 to v1.2), run `set --apply` from that release:
  until then `show` reports the pinned direction as unverified and every run is in
  shadow mode with `outreach_ready_direction_invalid`. `disable --apply` still brakes it.

The hermetic tests use the real bridge with in-memory Firestore and a fake object
store. No provider, model, session, CRM write or send.

## Per-site research line (site screen)

`site-screen.py` runs the per-site research line for ADP-010 partner discovery. The
owner approved it on 2026-10-05 after a successful 50-site pilot. Each site gets one
Parallel Task run (processor `core`, $0.025 per completed run; failed runs are not
billed). The run fills the `blueprint.site-screen.v3` form, with a URL and an exact
quote for each answer (design v1.1: the facility type and operator and the
variability signals are added). An optional second stage, the contact screen
(`blueprint.site-contact.v2`), runs only for sites whose screen is outreach-ready
(owner decision 2026-10-05, company GCS
`operations/recovery/2026-10-05/owner-decisions/owner-decision-contact-sources-20261005.json`).
The code is `tools/daily_research/site_screen.py`, standard library only, and the
standalone release ships it. Nothing sends, drafts or writes a CRM.

```bash
RELEASE=/opt/render/project/src/dist/daily-research/release
COMMAND="/opt/render/project/src/dist/daily-research/venv/bin/python $RELEASE/tools/daily_research/operators/site-screen.py"
OUT=/Users/Shared/blueprint-private/site-screen-20261005  # Durable, outside every Git work tree, never /tmp.
SPEND="--owner-reference REF --ceiling-usd 15 --max-runs 700"
BATCH="--batch-size N --seed site-screen-calibration-v1"  # Below the eligible count, at most --max-runs.
PYTHONPATH=$RELEASE $COMMAND plan --input /PRIVATE/backlog.v1.json.gz $BATCH
PYTHONPATH=$RELEASE $COMMAND run --input /PRIVATE/backlog.v1.json.gz $BATCH --out $OUT $SPEND
PYTHONPATH=$RELEASE $COMMAND run --input /PRIVATE/backlog.v1.json.gz $BATCH --out $OUT $SPEND --apply
PYTHONPATH=$RELEASE $COMMAND collect --out $OUT
PYTHONPATH=$RELEASE $COMMAND verify --out $OUT
PYTHONPATH=$RELEASE $COMMAND contact --out $OUT $SPEND --apply
PYTHONPATH=$RELEASE $COMMAND collect --out $OUT
PYTHONPATH=$RELEASE $COMMAND verify --out $OUT
PYTHONPATH=$RELEASE $COMMAND summary --out $OUT
```

These are manual, owner-approved runs on the owner's machine; `run` and `contact`
refuse on the daily worker (see the spend gate below). From a checkout, run
`PYTHONPATH=. python tools/daily_research/operators/site-screen.py` from the
repository root, with `--key-file` set to a private env file. Start with
`--max-runs 1`.

- `--input` is a site universe export (`backlog.v1.json.gz`, checked by
  `site_universe.load_export`), a discovery inventory page
  (`blueprint.discovery-inventory.v1`), or a JSON list of inventory records and
  export rows. Sites are screened in file order, which is rank order for an export.
  Inventory records with disposition `rejected`, `learning` or `duplicate` are
  refused, and a site listed twice is refused the second time. `plan` counts each
  refusal by code and estimates the cost of the batch. It reads nothing else.
- `--batch-size N` (on `plan` and `run`) screens N sites: the first ones in file
  order, with about one in ten taken instead from below that cut and flagged
  `calibration: true`, so the ranking's yield can be measured. Calibration sites are
  those with the lowest `sha256(seed:site_key)`, so one input, size and `--seed` give
  the same batch on any host; one follows each nine ranked sites. A batch larger than
  `--max-runs` is refused (`site_screen_batch_exceeds_max_runs`). When every site
  fits, there is no cut and no calibration site, so choose N below the eligible count
  that `plan` shows. The flag is never sent to the provider; `summary` counts tiers
  for ranked and calibration sites separately.
- `--task-focus CAPABILITY` (on `plan` and `run`) aims every site's question at one capability. Only
  site universe rows that list it are screened; other rows are refused with
  `site_screen_input_outside_focus`, and inventory records with
  `site_screen_input_focus_needs_site_universe`. The provider's task hint becomes that capability's
  plain description (`FOCUS_HINTS`). A row in a JSON input list may carry its own `screen_focus`,
  which wins, so one batch can be stratified across task families for demand discovery (owner
  decision 2026-10-05, company GCS
  `operations/recovery/2026-10-05/owner-decisions/owner-decision-demand-discovery-20261005.json`).
  `plan` counts the sites `by_focus`, and each run intent records its `task_focus`.
- On a site universe row with an `osha_ita` or `epa_frs` source and a street, city
  and state, that government record is the primary source for the exact site
  address, kept as `{source: "government_record", source_ids, site_id, answer}`.
  The operator always needs its own quote: the export does not record which source
  supplied the name or operator (an OpenStreetMap record can), so `source_ids` lists
  every source of the row. Any other row, and each web-found inventory site, must
  prove the address with its own quote too.
- `run` and `contact` are dry runs unless `--apply`. A dry run makes the same
  admission, writes nothing and calls nothing.
- The first `--apply` (of either stage) pins `--ceiling-usd`, `--max-runs` and
  `--owner-reference` in `owner_ceiling.json` (`ceiling_usd`, `max_runs`,
  `created_at`, `owner_reference`), which is created once. A later invocation of
  either stage must give the same owner reference and the same or a lower ceiling
  and run limit; otherwise it refuses with `site_screen_owner_reference_mismatch`,
  `site_screen_ceiling_above_pin` or `site_screen_max_runs_above_pin`.
- Every event is fsynced first to the out dir's spend journal, `spend.jsonl` (its
  first line is the pin), then to the stage's ledger, `<stage>/runs.jsonl`. An
  `intent` line precedes each create; it records the code's Git commit and whether
  tracked files were dirty (or a release's manifest commit), and `run` prints the same.
  Then `created` (with the run id), `refused` (no run exists; a later run tries the
  site again) or `uncertain` (the outcome is unknown) follows. A site with a run id,
  or with an unknown outcome, is never submitted again. An interrupted create counts
  as an unknown outcome.
- Each command first checks that the pin, the journal and the ledgers agree, and that
  every kept result and page read has its created run in the ledger. Damage to any
  one of those files at a time refuses (`site_screen_spend_journal_missing`,
  `site_screen_spend_journal_mismatch`, `site_screen_owner_ceiling_missing`,
  `site_screen_owner_ceiling_mismatch`). Matching edits to two of them, or the loss
  of a whole out dir before any result is kept, are not caught and can reset spend:
  keep the out dir durable and never edit it by hand. The only repairs are the two
  crash windows: a pin file whose journal line was written, and a ledger's last
  event whose journal copy was written; both are completed from the journal.
- Before each create, the price of each run in either stage that may be billed
  (completed, in flight, cancelled or of unknown outcome), plus the new run, must be
  at most the ceiling (above 0, at most $100). All runs of both stages must be at
  most the run limit (1–5,000). Otherwise the create is refused with
  `site_screen_spend_ceiling_reached` or `site_screen_max_runs_reached`. Both bound
  the out dir's total, not one invocation. Only a run observed `failed` frees its
  price, and it still counts as a run. `--processor` accepts only a processor with a
  reviewed price (`core`).
- One out dir holds one owner allowance. A new out dir starts again from zero, so keep
  the whole campaign in one out dir on durable storage outside every Git work tree,
  for example `/Users/Shared/blueprint-private/...`. An out dir under `/tmp`,
  `/private/tmp`, `/var/tmp` or `/var/folders` is refused
  (`site_screen_out_dir_volatile`): the system prunes them, and a lost ledger means
  paying again.
- A create answered 401, 402, 403 or 429, a redirect, or an unreachable provider
  stops the run (`site_screen_provider_*`, `site_screen_processor_refused`). Other
  4xx answers refuse that site only (`site_screen_create_rejected`). A 5xx answer, a
  success without a valid run id, or a connection lost after the request was sent is
  an unknown outcome: it stops the run, and its price stays committed.
- `collect` polls each created run every 15 s for up to `--wait-seconds` (default
  1800, at most 7200). Status and result reads are not billed. It stores each
  terminal response unchanged in `<stage>/results/<site_key>.json` and records an
  `observed` line with its SHA-256. A 401 stops it; other read errors are counted and
  tried again on the next pass.
- `verify` reads each cited page once with the daily agent's reader (`search.source`,
  45 s alarm) and keeps the text in `<stage>/evidence/<site_key>.json` with its date.
  A quote needs at least five words. It is `verified_on_page` when our read of its
  own URL holds every word of it, in order and as whole words (only case, spacing and
  punctuation may differ; there is no near match); else `in_citation_excerpt` when a
  provider citation excerpt cited for that same URL (scheme, `www.`, a trailing slash
  and the fragment aside) holds it; else `unverified`, or
  `unverified_page_unreachable` with the read's code. Other levels are `no_quote`,
  `quote_too_short` and `source_not_allowed`.
- LinkedIn is never read and never evidence: any URL whose host is `linkedin.com`,
  `lnkd.in` or a subdomain, or that contains either name (an archive, translation or
  redirect wrapper), is refused before any connection, on the first request and on
  every redirect, and its excerpts are ignored. Both forms send
  `source_policy: {exclude_domains: ["linkedin.com", "lnkd.in"]}`.
- The input location gives whatever it holds: street, city and state, as in
  "12 Main St, Springfield, IL 62701", or only the city for the backlog's
  "Springfield, United States; state not individually established". Its site anchors
  (`site_anchors`) are a street with a house number; the city followed directly by
  its state code in capitals or its name, on one line ("Mission, TX"); and the
  site's distinctive name words with the city as a capitalized place name in one
  sentence. A city alone never counts, so "our mission" or "e-commerce" never tie a
  text to Mission, TX or Commerce, CA. `plan` counts the sites with each anchor kind
  and with none.
- A proven quote must also name its answer: every significant word of the operator's
  name (legal forms aside), and that name must match the input's operator (one name's
  distinctive words, without words such as Manufacturing or Holdings, all in the
  other), else `operator_mismatch`; for the site, a site anchor, or the street of the
  provider's own address when that quote holds it and it lies in the input's city
  and state (only when the input has no street); and a word of the task phrase. The
  task quote, its page or a same-URL excerpt must name a site anchor (or the proven
  provider address); otherwise the task is `company_level_task` and does not
  qualify.
- `outreach_ready` (rule `blueprint.site-screen-rule.v3`, design v1.1) needs the
  operator, the exact site and the site task each proven that way, and nothing
  contradicted: an operator the input does not name (`operator_mismatch`), the site
  shown closed (`operating_now` `no`, or an answer that is not yes, no or unknown),
  and, each with a proven quote, an `office` or `mailing`
  facility, a `contractor` or `tenant` site the input attributes to another operator,
  `manual_today` `no`, or `existing_automation` `full` (this task at this site fully
  automated). Partial automation, or automation of other tasks or sites, keeps the
  site eligible. Every other site is `screened`, with its `blockers`.
- Each record asks exactly one question, `verification.outreach_question` under
  outreach-ready rule v1.2 (see [RENDER.md](../RENDER.md#outreach-ready-tier)): S
  (site link open) "Is <task> done at <site>, or somewhere else in the company?"; M
  (manual workflow open) "Which parts of <task> at <site> still need people, and what
  has kept them from being automated?"; else U "Is any of <task>
  at <site> automated today, or is it all done by hand?". `<site>` is "your <City>
  site" from the address city, else "your <name> site" for a short input site name,
  else "this site". The provider's "partial" choice also includes automation of
  other tasks/sites, so it cannot supply the exact task/site premise of template A.
  The shared builder retains A only for explicitly validated exact partial scope;
  screen records keep that premise open with U. Every other open check (existing
  automation, freshness, fit, interest) is recorded in `open_checks` and not asked.
  `variability_signals` is recorded and never required.
- Records are recomputed from the stored raw results and page reads, under each
  stage's current rule (`screen_gates` gives the lead-verification gate shape and
  `evidence_index` the URL-keyed evidence). `verify` writes
  `<stage>/records/<site_key>.<rule>.json`, and `summary` and `contact` recompute in
  memory, so a new rule needs no paid run, no page read and no deletion.
- `contact` creates one run per outreach-ready screen record (recomputed under the
  current screen rule), in screen order, under the same pinned ceiling and run limit.
  The form asks for the deciding role, a named current person from a reputable public
  source, a business email address published verbatim, the channel type, and a
  contact form or phone URL when no address is published.
- A person counts only when the person quote (five words or more) contains the name
  and stands whole-word on our read of the person's page or in a provider excerpt
  cited for that same URL.
- An email counts only when all of these hold (rule `blueprint.site-contact-rule.v3`):
  it is one plain address on the operator's own domain or a subdomain of it, never a
  free-mail domain; its quote holds the exact address; and our own read of the cited
  page holds the quote and the whole address. A provider excerpt alone never counts.
  The operator's domain is that of the page whose proven quote names the operator in
  the screen (the operator quote's URL; basis `operator_quote`), or (new in v3) the
  registrable domain of the screen's `website` answer when our own read of a page on
  that domain or a subdomain, cited by the screen or the contact stage, names the
  operator in its prose (basis `website`; an address or link that spells the name does
  not count). The bare `website` answer alone never counts, and a page elsewhere that
  names the operator (news, a directory, a government record) sets no domain. Neither
  basis is ever a directory, data broker, job board or applicant tracking, social, map,
  newswire, government, free-mail or LinkedIn host; the provider is told only the
  operator-quote website. Also new in v3: a quote of fewer than five words still proves
  an address that stands as a whole token on our own read of its cited page on the
  operator's domain; on any other page the five-word minimum stands, and person quotes
  always keep it. The codes are `site_screen_email_free_mail`,
  `site_screen_operator_domain_unproven`, `site_screen_email_off_operator_domain`,
  `site_screen_quote_lacks_address` and `site_screen_address_not_on_source`.
- A rule change recomputes records from stored results and page reads, but `collect`
  removed every address a rule of its day discarded (below), so a later rule never
  recovers one: those sites keep their discarded decision. `summary` counts
  `email_reasons` and `operator_domain_basis`.
- `collect` checks a completed contact result's email on our own read of its page
  before anything is stored. Then every address but a verified one is replaced by
  `[redacted-email]` in the stored result and page reads, so a discarded address is
  never kept; the decision (level and reason) stays with the page reads.
- LinkedIn is never read or evidence for a person or an email, including inside an
  archive or redirect wrapper (`person_source_not_allowed`).
- The recipient comes from the address itself; the provider's channel label is
  recorded and ignored. `person_email`: the local part holds a word of at least three
  letters from a verified person's name. `team_inbox` or `general_inbox`: a role word
  such as sales or operations (team), or info, contact or press (general). A careers,
  legal, support or similar inbox, or anyone else's address, gives `none`. The order
  is `person_email`, `team_inbox`, `general_inbox`, then `none`. A title alone never
  proves remit, so `decision_remit` is always an open question for a named person.
  `person_current` (no dated source within 18 months) and `recipient` are added when
  they are unproven.
- `summary` prints and writes `summary.json`: for each stage, run counts, answer and
  quote-level counts by field, claim states, tier counts (by origin and for ranked and
  calibration sites), blockers, questions, open checks, recipient, person and email
  levels, and cost. `estimated_cost_usd` counts completed runs; `committed_usd`
  counts each run that may be billed.
- The key comes from `PARALLEL_API_KEY`, or from `--key-file` (KEY=VALUE lines,
  `export` allowed) when one is given. It is never printed or written. `--out` must
  be outside this repository and outside any Git work tree
  (`site_screen_out_dir_inside_repository`). A new out dir gets mode 0700, and every
  file 0600. Only one command at a time can use an out dir (`site_screen_out_dir_busy`).
- Spend gate. The pinned ceiling and the journal hold for one out dir on one machine.
  Scheduled use needs the shared `paid_resource_admission` seam first: a
  `parallel_task` resource class in `PAID_RESOURCE_CLASSES`, an issuer in `src/` (the
  FindAll `admit_exact_request` pattern), a grant before each create,
  `scripts/verify_paid_resource_allocator.py` extended to scan
  `tools/daily_research/`, and ceilings taken from the owner-pinned direction under
  the worker lease, with one allowance across stages, out dirs and hosts. Until then,
  `run` and `contact` refuse whenever `BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED` is
  set, to any value (`site_screen_worker_needs_paid_admission`).
  Output is counts and stable `site_screen_*` codes only, never site names, people
  or addresses. The hermetic tests use a fake Task API transport and a fake page
  reader with synthetic sites.

## Site-screen admission (host-owned, design section 3)

Contact lookup rule v2 qualifies each person against the target facility after replaying the shared company search.
A different provider city/state holds a local manager unless a retained person/role/facility quote proves responsibility.
Matching location alone proves no site responsibility. Current corporate contacts and other approved people with unknown
scope remain explicitly labelled corporate referrals; the first-email question asks who covers the named facility.
FullEnrich DELIVERABLE/current-employment and corroborated/uncorroborated labels retain their original meanings.
Older sealed journals are requalified without buying another search. New searches retain at most five minimal current-role
candidates so another facility can select its own proven person. This repair never authorizes sends or fresh spend.

Outreach-ready site-screen records move into the CRM as rows labelled Hypothesis and on
to WebApp drafting (owner decisions 2026-10-05, company GCS
`operations/recovery/2026-10-05/owner-decisions/owner-decision-screen-to-crm-and-contacts-20261005.json`
and `owner-decision-first-outreach-batch-20261005.json`). Nothing sends: every bundle,
row, work item and draft is `sends_authorized: false`, and every WebApp send path
refuses a screen hypothesis. The code is `tools/daily_research/screen_admission.py`.

```bash
# 1. Render worker shell: a direction that names site_screen (until the expiry you set).
PYTHONPATH=$RELEASE $PY $RELEASE/tools/daily_research/operators/outreach-ready-direction.py set --paths daily_qa,site_screen --approval-reference REF --apply
PYTHONPATH=$RELEASE $PY $RELEASE/tools/daily_research/operators/outreach-ready-direction.py show   # sha256 and generation
# 2. Owner's machine, from a checkout (the out dir is local): build, then upload with your own gcloud login.
PYTHONPATH=. python tools/daily_research/operators/site-screen.py admit --out $OUT --direction-sha256 SHA --direction-generation GEN
PYTHONPATH=. python tools/daily_research/operators/site-screen.py admit --out $OUT --direction-sha256 SHA --direction-generation GEN --apply
# 3. Render worker shell, while the daily work is idle: pin the uploaded generation.
PYTHONPATH=$RELEASE $PY $RELEASE/tools/daily_research/operators/site-screen.py admission-pin --admission-id ID --generation G --approval-reference REF
PYTHONPATH=$RELEASE $PY $RELEASE/tools/daily_research/operators/site-screen.py admission-pin --admission-id ID --generation G --approval-reference REF --apply
PYTHONPATH=$RELEASE $PY $RELEASE/tools/daily_research/operators/site-screen.py admission-show
PYTHONPATH=$RELEASE $PY $RELEASE/tools/daily_research/operators/site-screen.py admission-disable --apply   # the brake
```

- `admit` selects outreach-ready screen records: at most `--per-focus` (default 2) per
  task family, taking sites with a verified recipient first in the owner's order
  (published person email; quoted person with a provider-verified email;
  provider-sourced person, corroborated, then not; published team inbox; published
  general inbox), then screen order; or exactly `--keys`. Each screen record, and its
  contact record, is recomputed from the stored results and page reads and must equal
  its derived record file (`screen_admission_record_mismatch`,
  `screen_admission_contact_record_mismatch`); run `verify` after a rule change.
- The bundle is canonical JSON: the direction (`sha256`, `generation`, `uri`), a
  manifest (rules, the out dir's owner-ceiling digest, the selection) and at most 50
  results. Each result has `result_digest` over its input, answers and checks, the
  Parallel run id, each proving quote with the SHA-256 of the page text or excerpt that
  holds it, the record's one question, and the chosen recipient with its label and
  provenance (the published page and the operator-domain proof, or the provider's
  `status`, `score`, `checked_at` and `request_digest`). `admission_id` is the SHA-256
  of the bundle's bytes. Provider lookups use `contact_lookup.load` to recompute the current
  records from the durable journal, then `choose_recipient` against the current contact.
  FullEnrich's original `DELIVERABLE` status, unknown score, request/record/lookup digests,
  and provider employment/source proof are retained; cached recipient JSON grants nothing.
- `admit --apply` keeps the bundle write-once in `<out>/admissions/<id>.json`, uploads it
  with `gcloud storage cp --if-generation-match=0` to
  `gs://blueprint-8c1ca.appspot.com/operations/research/screen-admission/<id>/bundle.json`
  and reads that generation back. No credential is created or moved: the upload uses
  the owner's existing gcloud login, and the worker reads the object with its own
  identity. Output is counts and digests only.
- `admission-pin` reads exactly that generation through the bridge, runs this package's
  loader (`load_bundle`: canonical bytes, digests, rules, questions, recipients) and the
  live direction gate (enabled, the bundle's exact `sha256` and `generation`, naming
  `site_screen`, effective and unexpired, `max_rows_per_batch` at least the batch), then
  takes the fenced lease only for the `screen_admission_set` compare-and-swap and reads
  control back. A pinned admission whose claimed write was never read back blocks a new
  pin (`screen_admission_current_write_uncertain`) unless `--supersede-uncertain`.
- The worker processes the pin from its scheduler only when no daily row, QA, repair or
  publication is active or queued (`screen_admission_waiting`), under its lease. It
  refreshes the canonical CRM, drops sites the CRM holds (`runner.keys` and the
  publisher's structural rule) or an earlier acknowledged admission wrote, and the
  bridge op `screen_admission_publish` plans the rows create-only, claims, writes once
  and reads back. Each row is labelled `Hypothesis` (G), `Outreach-ready: operator,
  site, task proven` (Q), carries the recipient with its label in E, F and H, and ends
  M with `First email asks: <question>` and its marker `[screen:<id>;<result_digest>]`.
  A CRM change before the claim makes a fresh plan (`screen_admission_replan_required`).
  A claimed write the readback does not show stays `screen_admission_write_uncertain`
  and is only read back, never written again.
- The acknowledged admission writes `screenWorkItems/<id>`; the WebApp reads it with
  `Store.screenSnapshot` (work item, acknowledged state with the plan and receipt, and
  the bundle bytes from the blob store) and verifies every digest itself. The hermetic
  tests use the real bridge with in-memory Firestore, a fake object store and a fake
  CRM sheet; `tests/fixtures/daily_research/screen-admission-snapshot.json` is the
  golden snapshot the WebApp tests copy.
## Contact lookup (FullEnrich)

`contact-lookup.py` looks up a provider-verified work email for a real person at each
contact site of a site-screen out dir. Owner decisions 2026-10-05, company GCS
`operations/recovery/2026-10-05/owner-decisions/`:
`owner-decision-contact-provider-lookup-20261005.json` (amends
`owner-decision-contact-sources-20261005.json`) and
`owner-decision-provider-sourced-person-20261005.json`. One provider: FullEnrich API v2
(`app.fullenrich.com`). The code is `tools/daily_research/contact_lookup.py`, standard
library only. Nothing sends, drafts or writes a CRM; the founder sends every email himself.

```bash
OUT=/PRIVATE/site-screen-out-dir   # The site-screen out dir: durable, outside every Git work tree, never /tmp.
KEYS=/PRIVATE/fullenrich.env       # One line: FULLENRICH_API_KEY=...
SPEND="--owner-reference owner-decision-contact-provider-lookup-20261005 --max-credits 50 --max-calls 100"
SEARCH="--person-search owner-decision-provider-sourced-person-20261005"  # Optional: provider_sourced people.
PYTHONPATH=. .venv/bin/python tools/daily_research/operators/contact-lookup.py lookup --out "$OUT" $SPEND --key-file "$KEYS" $SEARCH
PYTHONPATH=. .venv/bin/python tools/daily_research/operators/contact-lookup.py lookup --out "$OUT" $SPEND --key-file "$KEYS" $SEARCH --apply
PYTHONPATH=. .venv/bin/python tools/daily_research/operators/contact-lookup.py summary --out "$OUT"
PYTHONPATH=. .venv/bin/python tools/daily_research/operators/contact-lookup.py balance --key-file "$KEYS"
```

Run from the repository root after the site screen's `contact`, `collect` and `verify`, with an
interpreter that has no `tools` package of its own in site-packages (one shadows this
repository's `tools/` even with `PYTHONPATH=.`); the repository's `.venv` has none.

- Who is looked up, per contact record (recomputed under the current contact rule), in
  contact order: a site with a published, verified person email is left alone
  (`published_person_email`); a site without an operator domain proven by the screen is
  skipped (`operator_domain_unproven`). A named person the contact stage proved by a
  quote, with a dated source at most 18 months old, gets one work-email enrichment
  (`quoted_person`); an undated or older one is `person_not_current`. Otherwise, only
  with `--person-search` naming the decision above (anything else refuses with
  `contact_lookup_person_search_reference_invalid`), one people search on the operator's
  domain for the listed roles (`TITLES`: owner, president, general manager, plant
  manager, operations manager or director, engineering or automation manager) keeps minimal
  candidates FullEnrich places at that domain now (`is_current` true, or no end date
  in `employment.current`; historical `employment.all` entries need `is_current: true`)
  in a listed role. Each facility selects its proven site contact first, otherwise an
  explicitly unknown corporate referral; a location-mismatched local manager is held.
  The selected person gets one enrichment (`provider_sourced`). The
  employment field relied on is recorded; a past employer never counts. The pages the
  site screen already read and kept are checked for the person with their title and the
  operator (`corroboration`, true or false; no new read, never LinkedIn).
- An email counts only when FullEnrich marks it `DELIVERABLE` (`HIGH_PROBABILITY` is its
  catch-all estimate, and `CATCH_ALL` and `INVALID` do not count either), it is on the
  operator's domain or a subdomain, never free mail, its local part names the person
  (never a role inbox), and the enrichment's own profile, where it gives one, names the
  same person at the operator now. Only `contact.work_emails` is requested; personal
  emails and phones are never asked for or kept, and a rejected address is never written.
- Spend. The first `--apply` pins `--owner-reference`, `--max-credits` (FullEnrich
  credits) and `--max-calls` in `lookup/owner_ceiling.json`, once; later runs may only
  lower them (`contact_lookup_credits_above_pin`, `contact_lookup_calls_above_pin`,
  `contact_lookup_owner_reference_mismatch`). Pin the whole allowance for this out dir.
  An `intent` is fsynced to `lookup/spend.jsonl` before every call. Admission counts
  answered calls at their reported credits and calls of unknown outcome or unread result
  at their most (a search 1.25: five people at 0.25; an enrichment 1), and stops with
  `contact_lookup_credit_ceiling_reached` or `contact_lookup_max_calls_reached`. One
  search per domain and one enrichment per person and domain are never sent twice. A
  401, 402, 403, 429, a redirect or no connection stops the run unbilled, and so does
  an enrichment FullEnrich ends unbilled (out of credits, rate limited, cancelled); a
  later run may send those again. A 5xx, a lost connection or an unreadable answer may be
  billed, so it is never sent again and its most stays committed.
- Results. An enrichment is asynchronous: `--apply` reads started ones every 10 s for up
  to `--wait-seconds` (default 120; reads are not billed), and a later `--apply` reads
  the rest (`state: pending` until then). Without `--apply` the run admits the same calls
  and sends and writes nothing; it counts the searches, as their enrichments depend on
  the answers.
- Records. `lookup/records/<site_key>.contact-lookup-rule.v2.json` holds each site's
  lookup (`source: provider_lookup`, `label: looked_up`, FullEnrich's own `status`, its
  `valid` or `not_valid` mapping, the person with `sourcing` and proof, and the address
  only when usable) and its `recipient` from `contact_lookup.choose_recipient`, a pure
  function for the admission step: 1 published person email, 2 looked-up person email,
  3 published team inbox, 4 published general inbox, 5 none, each with its source and
  labels. `contact_lookup.load(workspace, states=states)` recomputes the records from
  the durable journal and current site evidence under the admission caller's out dir
  lock. Cached record files do not authorize recipients. A quoted source must still be
  within 18 months at lookup time and its verified quote must support the claimed title.
- `summary` recomputes every record from the journal alone and writes
  `lookup/summary.json`. Output is counts and stable `contact_lookup_*` codes only, never
  names or addresses. Every file is 0600 in 0700 folders. The key comes only from
  `--key-file`; it is never printed or written. The command refuses on the daily worker
  (`contact_lookup_worker_needs_paid_admission`). The hermetic tests use a fake FullEnrich
  transport and synthetic sites.
- `balance` makes one unbilled `GET /api/v2/account/credits` and prints only the finite
  remaining credit count and check time. It creates no artifact or lookup intent.
  Read the actual balance before an owner-authorized first pass; a reported free tier
  is not evidence of the remaining allocation or permission for a paid subscription.

Provider contract references: [data dictionary](https://docs.fullenrich.com/api/v2/general/data-dictionary),
[enrichment result](https://docs.fullenrich.com/api/v2/contact/enrich/bulk/get),
[credit balance](https://docs.fullenrich.com/api/v2/account/credits/get).

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
whole control document except the lease, `cleanup_observation_required`,
`paid_expansion`, `site_universe` and `outreach_ready`. Never apply a partial document. Read the control back and run
`preflight` with the same prefix before the next 07:00 run.
