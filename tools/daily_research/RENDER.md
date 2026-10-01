# Existing Render worker integration

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
it never creates a paid catch-up backlog. Idle minute ticks read only the small
control document. Recovery errors retry after five minutes. Deploy overlaps use
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
| `files/YYYY-MM-DD-{artifact,evidence,output,review}.json` | Blob pointers; downloaded raw artifact is immutable once bound |
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

QA/publication agents perform the workflow, rather than asking dot to relay it:

1. Read the durable packet, original inputs and exact-turn evidence. Verify
   source support and freshness, distinguish operator/vendor/independent claims,
   check commercial/industrial robotics relevance, and preserve unknown interest.
   Do not autoqualify the broad directory or home/data-only/generic-3D entries.
2. Re-read the canonical CRM and check semantic site/task duplicates. Submit
   `review --date DATE --input DECISION.json` with `packet_digest`, agent
   `reviewer_reference`, `source_support_verified=true`, `crm_rechecked=true`,
   `accepted_keys` and bounded `summary`. These attestations require actual agent
   checks; the clock does not manufacture them from schema validation.
3. Consume only the resulting digest-bound `sheets`/`notion` delivery outbox.
   Use existing authorized app bindings and existing Blueprint review/approval
   surfaces where human authority is required. Preserve manual Sheet edits;
   record exact target records/columns and source-backed Notion links.
4. After each real write/readback, submit `receipt` with destination, delivery
   key, payload digest, `readback_verified=true`, and the actual reference.
   Duplicate receipts are idempotent; conflicting receipts refuse. Both receipts
   complete publication. A dot/observer receipt is never required, including for
   legacy `parent_status` entries. No prospect outreach is authorized here.

The queue is a durable handoff, not proof the consumer is deployed. The separate
communications owner must bind its agent to this contract and verify QA and both
publication readbacks. Missing app bindings produce an explicit pending/blocker
record, never fabricated success or new credentials/grants.

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
   prior sessions/dated intents. Import an existing runner state directory with
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
