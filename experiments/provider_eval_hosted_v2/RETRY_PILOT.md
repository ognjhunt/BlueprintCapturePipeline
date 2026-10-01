# Authorized $2 hosted retry

Approval: assistant `Sentinel_f8a3006241c88191bcb27ec7e2121aba` proposed retrying
the four-mode Chef Robotics pilot with a **$2 soft total hosted target**, including
the first failed attempt, still against the original $10 aggregate ledger. User
`Sentinel_c4c056bcbf408191850ec0dd9e86f6c5` answered **yes**. Only execution owner
`01a0f3b8-6abe-775b-bfea-5102185b80ce` may dispatch. Implementation/review make no
paid calls. This authorizes four new sessions; a 404 alone never does.

The new cohort is `hosted_agent_research_v2_retry1`, with a separate immutable
receipt, task/session identities, AgentJournal, evidence and output directories.
The old deleted session, original scope, failure trace and $4.765920 accounting
checkpoint remain intact. Old unused reservations are not released or relabeled.
Raw evidence is reused only from valid corrected searches for the same provider
arm. No controller outputs or other arms' sources are pooled.

## Exact commands

Use the reviewed clean commit in the owner's experiment checkout. The already
tested SDK pair remains OpenAI 3.22.1 / Agents SDK 0.22.3; no new key, project,
permission or dependency is needed if the prior pilot environment has that pair.
The existing hosted Sol/Default binding has already accepted the original session.

First ensure the prior approved cleanup was adopted offline using the existing
[cleanup command](SOFT_PILOT.md#already-approved-deleted-session). If already adopted,
do not rerun it. Prepare checks the anchored deletion proof before admitting retry.

To print the original approval SHA without reading any key or making HTTP calls:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from experiments.provider_eval_recovery.harness import digest, read_json
root = Path('/workspace/provider-eval-private-live-20260930')
print(digest(read_json(root / 'protocols/hosted_agent_research_v2/soft_pilot_approval.json')))
PY
```

Use that exact value as `ORIGINAL_APPROVAL_SHA`:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.retry_pilot --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --prior-approval-sha256 ORIGINAL_APPROVAL_SHA --prepare-retry
```

Prepare makes no HTTP calls. Inspect the printed receipt; then supply its exact
`retry_receipt_sha256` below:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.retry_pilot --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --retry-receipt-sha256 ACTUAL_RETRY_RECEIPT_SHA --execute
```

The same execute command resumes the same durable identity and completed outputs;
it cannot silently mint a second retry. Replace `--execute` with `--status` for
local accounting/cleanup reporting without HTTP. Once stopped, new paid work stays
blocked. No automatic paid retries, old-session reopening, or 80-case matrix exists.

## Accounting and bounds

The original $10 ledger retains every prior allowance, count fee, extra and unknown.
The **separate $2 soft hosted target** counts the failed session's $0.251335 model
high-water plus its retained $0.09 cleanup buffer: **$0.341335**. Old unstarted-arm
allocations remain reserved against $10; they are not treated as billed usage or
charged again as prior usage in the soft target. Billing/async teardown remain
unreconciled. The old raw-search diagnostic cohort remains in the original $10
accounting, outside this hosted retry's incremental soft target.

Each new arm reserves $0.26 model planning exposure (raised from $0.055 after the
49,484-input/261-output trace) and $0.09 retained-container cleanup carry. Four arms
plus isolated review/count allowance $0.05548 reserve **$1.455480** before creation.
At most three native searches per arm, including valid reused searches, add at most
$0.0735 of fees/extras. Projection is **$1.870315 hosted soft total** and
**$6.294900 shared aggregate reserve**. No prior hold is released.

The application stops admission near **$1.90**. Unknown fresh usage, uncertain
creation/tool outcomes, security/protocol warnings and five-minute deadlines stop
the whole pilot. Usage can exceed targets before the next GET, so this is not a
hard cap or guarantee of four answers. Managed token targets are planning only:
50,000 input / 2,048 output per arm, 48 application functions, three searches and
three public fetches, with one active root session at a time. Cancellation is
bounded and reconciled; permanent deletion of new sessions needs exact action-time
approval. Container carry continues until an approved deletion receipt is adopted.

Every new paid gate validates the immutable retry source/receipt, prior deletion
and cumulative cost proof. Source, approval, baseline, target, model/project or
prior-cost changes are refused. The original worker lock is shared across cohorts.

All four agents receive the same revised instructions: inspect index metadata,
select relevant source IDs and narrow ranges, and retain compact extracts/findings.
Full text remains available at all offsets; no prefix clipping or inaccessible
evidence is added. The task criteria, source verification and parent-isolated
ground-truth review remain identical. The controller does not grade itself.

Outputs live under
`protocols/hosted_agent_research_v2_retry1/soft_pilot/reports/`; corresponding task,
session, usage, evidence and cleanup artifacts are retained alongside them. Review
and final billing still precede any winner claim.

## Usage-lag recovery of the existing retry

The first retry session was cancelled on an empty usage snapshot 0.33 seconds
after creation. Supplied timestamps show usage arriving after 14.7 seconds:
8,594 input / 84 output tokens. Missing usage is unknown, not zero. The repaired
monitor waits at most **30 seconds**, using only session GETs; model/container
holds stay counted, and application tools/message replies wait for valid usage.
Malformed, decreasing, stale observations and grace expiry refuse paid work.
Fresh unchanged counts remain valid best-effort observations, not a settled bill.
Neither the usage counts nor the $2 soft target constitute a hard cap.

The user has authorized finishing the existing retry. Its exact known session is
`sess_09118c760a810004006abdc1869c9c8194844ab4a80b431ad5`, with cancelled turn
`turn_09118c760a810004006abdc18be18081948811782f9c45e2ef`. The old task, cancelled
turn, failure, receipt, allowance and ledger remain intact. One child task ending
`_usage_resume1` sends one durable message event to this session; it never creates
a replacement session or automatically resends an uncertain message. Its fixed
five-minute follow-up deadline is anchored at preparation, within original receipt
expiry. Original task deadline stays unchanged. Function opportunity is reduced by
prior calls, search/fetch counts remain durable, and model exposure is cumulative
for this same session. The other three original arms retain their existing slots.

Use the current reviewed commit, SDK pair and **original retry receipt SHA**:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.usage_recovery --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --retry-receipt-sha256 ACTUAL_ORIGINAL_RETRY_SHA --prepare-continuation
```

Preparation makes GETs only. It verifies exact owned session/model/network,
settled usage, idle session, cancelled root, immutable old source/approval, ledger
prefix and remaining soft headroom. It retains the fresh API responses and the
original lag failure under `soft_pilot/usage_recovery.json` and
`soft_pilot/retained_usage_lag_stop.json`; no scope or budget is reset. Only that
exact historical lag stop is superseded. A later stop remains effective.

The available actual retained lifecycle fixtures pass the offline regressions.
After the read-only preparation succeeds:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_hosted_v2.usage_recovery --aggregate-root /workspace/provider-eval-private-live-20260930 --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce --retry-receipt-sha256 ACTUAL_ORIGINAL_RETRY_SHA --execute
```

The recovery uses the existing $6.221400 aggregate reserve, with all prior cost
uncertainty retained. The prior cancelled retry's observed model usage is included
within its cumulative $0.26 arm hold, never zeroed or charged twice. Container age
continues to accrue. If time, uncertainty or updated exposure reaches the original
limits, admission stops; this command does not renew approval or increase budget.

The recovery packet `libfile_f0f9e052fc388191949d408941075d2d` could not be downloaded
through supported Library materialization, and its gzip has no readable Library
projection. No proxy denial was bypassed. The parent then supplied exact retained
creation/first-GET/settled usage observations and a later real session projection
with its complete turn-list response. Those available fixtures pass offline
regressions, including actual null/default fields and absent session turn fields.
The initial full session bodies were never saved and cannot be recovered; their
status/network values are unknown, not reconstructed. This retention gap is
documented in `fixtures/usage_lag_retained.json` and does not block the reviewed
preparation, which retains current full responses before continuation. No material
API-contract difference was found. Implementation made no live calls.

## Future event sends; existing uncertain input stays unresolved

The already attempted follow-up had no idempotency key. Its `sent_unknown` journal
state is retained. Exhausted paginated item/turn reads and an idle session do not
prove nonacceptance. Do not resend that payload, assign it a retroactive key,
create a replacement session, or adopt this commit over the used recovery proof.
The original proof binds both its source commit and runtime hash.

For future admitted event sends only, `event_transport.py` uses the official
OpenAI SDK 3.22.1 `beta.agents.sessions.events.with_raw_response.create`. Before
dispatch it persists the exact canonical payload, scope-bound deterministic
`Idempotency-Key`, and transmission intent in the journal and private sidecar.
SDK automatic retries are disabled. An explicit retry retains the original
payload and key, permits at most two transmissions within 60 seconds of the first
intent, and still passes the existing admission, usage, deadline and budget gates.
An acknowledged event is not sent again; acknowledgement is not turn completion.
Safe receipts retain exception class/cause class, HTTP status, request ID and
latency without exception prose, credential values or response-body dumps. The
endpoint remains pinned to `api.openai.com/v1`, the existing Default project is
retained, and authorization redirects are refused.

No new cohort, receipt migration, paid retry or launch command is added. In
particular, these future safeguards cannot resolve the current unkeyed timeout.

The smallest supported resource-close option is permanent deletion of the exact
retained session after archiving all available conversation items, turns, source
files, outputs and local uncertainty receipts. Current session:
`sess_09118c760a810004006abdc1869c9c8194844ab4a80b431ad5`.
Its deletion has **not** been approved. The execution coordinator must obtain
action-time approval naming that exact session. Existing budget approval and
permission to delete the earlier, different session do not authorize this delete.

After exact approval, use the existing SDK/binding and `sessions.delete` on only
that ID with automatic retries disabled, retain the deletion response, and make
bounded read-only checks of that session and its known environment. A conflict or
uncertain DELETE remains unresolved; do not blindly resend uncertain mutations.
Physical cleanup is asynchronous. Resource retirement does not establish whether
the earlier message was accepted, settle billing, release retained allowances or
rewrite the historical `sent_unknown` state. Cancellation alone is not permanent
resource retirement. Leaving the session idle does not prove zero container cost.
