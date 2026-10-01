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

After the actual retained lifecycle fixtures are checked:

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
projection. No proxy denial was bypassed. Parent-supplied timing evidence and
mocked lifecycle regressions are verified; exact historical initial/settled API
fixtures remain pending readable JSON or execution-owner local replay. The
read-only preparation step is available now; implementation made no live calls.
