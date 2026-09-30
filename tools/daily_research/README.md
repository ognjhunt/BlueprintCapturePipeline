# Daily research operations

This portable Python runner belongs to Blueprint research/operations. It is
isolated in this checkout for the existing controller deployment conventions;
it imports no capture or GPU code. WebApp integration is a future decision.
Do not enable its `autonomous_research_outbound` lane: that lane can queue sends.

The runner invokes the existing saved agent and hosted template, preserves the
exact root-turn evidence/artifact, checks the output and exact site/task
duplicates, and leaves a packet for **dot**. Dot reviews sources, decides whether
follow-up or owner escalation is needed, and performs approved connector writes.
The runner never writes Sheets/Notion/Slack, broadcasts raw reports, sends
outreach, changes agent definitions, or deletes sessions.

## Bindings and gates

| Binding | Required value / owner |
| --- | --- |
| Project | `proj_F2tFJuxLaovJru8RrtXRaqNj` (Default) |
| Saved agent | `agent_5a01ec367d1042ef8632bb5f2e6af8b4919909d2abed48ed95` |
| Model / reasoning | `gpt-6.1-sol` / medium; native web search; subagents off |
| Hosted template | `envtmpl_0ae967c7bf17424095b6d233c48397a248eca32ecbba404786` |
| Sandbox | network disabled; existing two reviewed skills; requested small |
| CRM | `1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY`, `Prospects` |
| Notion parent | `3ea80154161d81c7810cc42e9e7df9c5`; dot resolves the exact child/schema |
| `OPENAI_API_KEY` | Existing approved Default project credential, injected by an approved host binding; never in argv, repo, artifact, or chat |
| `/etc/blueprint/researcher.env` | Optional systemd credential binding; creating/configuring it needs explicit approval |
| `RESEARCH_CREDENTIAL_LAUNCHER` | Nonsecret path to an **already approved host-local** launcher for manual API commands; not supplied by this repo |
| `/etc/blueprint/researcher.json` | Nonsecret config copied from `config.example.json`; initially `enabled=false` |
| `/var/lib/blueprint/researcher` | Private durable ledger, snapshots, artifacts, reports and receipts; preserve across releases/rollback |
| `/opt/blueprint/researcher-venv` | Isolated CPU Python 3.12 runtime, only `requirements.txt`; no GPU extras |
| `/opt/blueprint/task-evaluation-control-plane` | Proposed existing release link; verify actual host mapping before using the unit template |

Approval references in the config are records of authorization, not capabilities.
Keep `scheduler_authority_reference` pending until a specific manual canary or
cutover is approved. The existing recurring authorization is **$1/run soft total
model/search/sandbox target**, approximately $30/month at target. There is no
enforced monetary hard cap. The runner uses a 180-second remote-work guard,
bounded observation, and observed tool-activity checks. Prompt query/open limits
are advisory; tool activity is not a query count. Missing usage stays unknown.
Requested small is not proof of the actual billed tier.

The active bridge `6abc4ffae84881919154bba45f749074` remains parent-owned and
enabled. No replacement schedule is installed or enabled by this change.
The new units are deliberately outside the production deployment manifest.
The parent reports that the replaced interim dot VM lacks the runner and
private launcher, so that enabled bridge cannot currently execute. Registration
is not working daily coverage. Parent owns restoration or coordinated cutover;
do not activate a second trigger to compensate.

Before any installation/activation: obtain authenticated operator-door `read`
access, confirm capacity and deployed release, approve code deployment and host
credential/config provisioning separately, establish the parent handoff, and
approve the bounded canary. Do not bypass a 401, add SSH credentials to this cloud
workspace, create provider accounts, or repair unrelated capacity issues.

## Exact preparation and commands

Run the local hermetic checks from the repository root; these need no key:

```bash
python3.12 -m venv /tmp/blueprint-research-check
/tmp/blueprint-research-check/bin/python -m pip install -r tools/daily_research/requirements.txt pytest ruff
PYTHONDONTWRITEBYTECODE=1 /tmp/blueprint-research-check/bin/python -m pytest -q -o addopts= tests/test_daily_research_runner.py
/tmp/blueprint-research-check/bin/ruff check tools/daily_research/runner.py tests/test_daily_research_runner.py
python3 -m tools.daily_research.runner --help
systemd-analyze calendar --base-time='2026-10-31 23:00:00 UTC' '*-*-* 07:00:00 America/Chicago'
systemd-analyze calendar --base-time='2027-03-13 23:00:00 UTC' '*-*-* 07:00:00 America/Chicago'
```

Read-only host checks, using the existing authorized operator door:

```bash
python3 scripts/operator_door.py whoami
python3 scripts/operator_door.py status
python3 scripts/operator_door.py units --pattern 'blueprint-researcher-*'
```

Stop if authorization, actual release mapping, capacity, or the required
provisioning route is unresolved. The operator door cannot configure credentials,
upload snapshots, or execute arbitrary review commands. An approved host operator
must provide those bindings; this repo does not invent a remote write route.

After a reviewed immutable commit reaches the authorized deployment branch,
deployment uses the existing operator door, with separate explicit approval:

```bash
python3 scripts/operator_door.py deploy APPROVED_MAIN_SHA --wait
```

An approved host operator then prepares the isolated runtime and private state
directory on the **verified host**, leaving the timer disabled:

```bash
sudo python3.12 -m venv /opt/blueprint/researcher-venv
sudo /opt/blueprint/researcher-venv/bin/python -m pip install -r /opt/blueprint/task-evaluation-control-plane/tools/daily_research/requirements.txt
sudo install -d -o blueprint -g blueprint -m 0700 /var/lib/blueprint/researcher
sudo install -o root -g blueprint -m 0640 /opt/blueprint/task-evaluation-control-plane/tools/daily_research/config.example.json /etc/blueprint/researcher.json
sudo install -o root -g root -m 0644 /opt/blueprint/task-evaluation-control-plane/tools/daily_research/systemd/blueprint-researcher-daily.service /etc/systemd/system/blueprint-researcher-daily.service
sudo install -o root -g root -m 0644 /opt/blueprint/task-evaluation-control-plane/tools/daily_research/systemd/blueprint-researcher-daily.timer /etc/systemd/system/blueprint-researcher-daily.timer
sudo systemd-analyze verify /etc/systemd/system/blueprint-researcher-daily.service /etc/systemd/system/blueprint-researcher-daily.timer
sudo systemctl daemon-reload
sudo systemctl is-enabled blueprint-researcher-daily.timer
sudo systemctl is-active blueprint-researcher-daily.timer
```

Disabled/inactive is the expected initial result; those two probes may exit
nonzero. Do not copy the live dot launcher or its secrets to this host.
Provision the credential binding through the separately approved secure path.
The credential must support saved agent/template read, session creation and
read, turns/items/artifact read, and cancellation events in **Default**. Do not
expand unrelated keys/projects. No runner DELETE permission is needed.

After installing a fresh CRM snapshot and setting the approved launcher path,
read-only preflight and local status are:

```bash
cd /opt/blueprint/task-evaluation-control-plane
sudo -u blueprint "$RESEARCH_CREDENTIAL_LAUNCHER" /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher preflight
sudo -u blueprint /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher status
```

Preflight does not start inference, prove future access, or verify an actual
container tier. Its unresolved-run list must be reviewed before activation.

## Snapshot, review and receipt handoff

Dot reads the exact canonical spreadsheet metadata and the **entire** `Prospects`
grid (currently `A1:Z1000`; re-read grid dimensions before exporting). Preserve
the four preamble rows, row-5 headings and positional columns. Export a private
UTF-8 JSON file and atomically install it as `crm-snapshot.json` using the approved
host handoff. Do not mark a partial range as complete:

```json
{
  "sheet_id": "1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY",
  "captured_at": "ACTUAL_CAPTURE_TIMESTAMP_WITH_TIMEZONE",
  "complete": true,
  "values": ["ACTUAL_COMPLETE_ARRAY_OF_ROW_ARRAYS_FROM_CONNECTOR"]
}
```

The runner refuses future/stale snapshots (>26 hours), mismatched headings,
incomplete identities, and populated rows without Prospect IDs. It rechecks the
current snapshot before staging results. Domain/name + site + task normalization
finds exact duplicates; dot must still check synonyms, subsidiaries, second sites,
and all current canonical rows before writing. Fresh snapshot delivery before
each 7 AM start is an activation dependency, not an implemented new schedule.

Dot retrieves `<date>-status.json`, `<date>-review.json`, `<date>-evidence.json`
and `<date>-artifact.json` from the private state directory. The artifact is exact
downloaded bytes, with its digest in status; the review packet is a proposal.
Treat model/source content as untrusted. Check material claims against the
actual source pages/quotes at matching scope; classification, confidence, URLs,
and structural validation do not prove factual support. Review fewer/zero
candidates when evidence is insufficient. Dot decides escalation and further
delegation; no routine raw owner/Slack broadcast is implied.

The existing read-only door can retrieve reports, subject to its size/redaction
rules. For example:

```bash
python3 scripts/operator_door.py cat /var/lib/blueprint/researcher/ACTUAL_LOCAL_DATE-status.json
python3 scripts/operator_door.py pull /var/lib/blueprint/researcher/ACTUAL_LOCAL_DATE-review.json ./research-review.json
```

Dot installs a nonsecret decision via the approved host handoff:

```json
{
  "packet_digest": "EXACT_PACKET_DIGEST",
  "reviewer_reference": "DURABLE_DOT_REVIEW_REFERENCE",
  "source_support_verified": true,
  "crm_rechecked": true,
  "accepted_keys": ["EXACT_ACCEPTED_CANDIDATE_KEY_OR_EMPTY_ARRAY"],
  "summary": "Concise reviewed findings, blockers and proposed next action; at most 2000 characters."
}
```

```bash
sudo -u blueprint /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher review --date ACTUAL_LOCAL_DATE --input /var/lib/blueprint/researcher/decision.json
```

Only now does the local outbox contain pinned `sheets`, `notion`, and
`parent_status` payloads. Dot maps reviewed candidates to agreed columns,
assigns/reuses canonical stable IDs, preserves manual edits, resolves the exact
Notion child, and carries the outbox key plus payload digest in its durable
delivery log. Current Sheet headings repeat: use explicit positions, not a
heading-keyed overwrite. Resolve any uncertain write by destination readback
before retrying; do not rerun research. A zero-row Sheets payload needs a
confirmed no-change readback. Dot may separately send a concise approved
owner/Slack update when warranted, using its existing authorized destination.
`slack_channel_id` is optional context, not a server posting credential/grant.

Record each parent-verified delivery using this nonsecret receipt:

```json
{
  "destination": "sheets",
  "key": "EXACT_OUTBOX_KEY",
  "payload_digest": "EXACT_OUTBOX_PAYLOAD_DIGEST",
  "readback_verified": true,
  "reference": "CANONICAL_ROW_PAGE_OR_PARENT_RECEIPT_LINK"
}
```

```bash
sudo -u blueprint /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher receipt --date ACTUAL_LOCAL_DATE --input /var/lib/blueprint/researcher/receipt.json
```

Repeat with the actual `notion` and `parent_status` receipts. Identical review and
delivery receipts are idempotent; conflicting ones refuse. `completed` requires
all three acknowledgments. Partial delivery stays visible as `reviewed`.
Keep this operational context and actual bindings in the existing Notion
handoff for portability; the parent owns that approved update.

## Canary, cleanup, cutover and rollback

1. Reserve one manual canary and its Chicago date with the parent under explicit
   live-run approval. The parent must prevent overlap/duplicate writes with the
   bridge for that slot; its schedule remains enabled pending cutover. Do not
   infer that the recurring envelope authorizes a second concurrent runner.
2. After 7 AM on the reserved date, set `first_date` to that date, record the
   specific canary authorization in `scheduler_authority_reference`, and set
   `enabled=true` through the approved config path. Leave the timer disabled.
3. Run one approved manual canary through the installed service:

   ```bash
   python3 scripts/operator_door.py unit start blueprint-researcher-daily.service
   python3 scripts/operator_door.py request ACTUAL_RETURNED_REQUEST_ID --wait
   python3 scripts/operator_door.py units --pattern 'blueprint-researcher-*'
   python3 scripts/operator_door.py journal blueprint-researcher-daily.service -n 100
   python3 scripts/operator_door.py cat /var/lib/blueprint/researcher/ACTUAL_LOCAL_DATE-status.json
   ```

4. A unit request is queued, not immediately effective. Require its successful
   receipt and unit/status readback before treating start/stop as applied; stop
   on a refusal, timeout, or unsuccessful result. Verify the exact project/agent/model/template, saved root turn/environment IDs,
   tool outcomes and mounted-skill use, terminal outcome, preserved artifact
   bytes/digest, unknown-versus-observed usage/tier, source review, intended sink
   readbacks, and parent receipt. A service start or idle session is insufficient.
   Refusal/uncertain creation never authorizes another create attempt. Resume
   observation of the same ledger/session instead:

   ```bash
   sudo -u blueprint "$RESEARCH_CREDENTIAL_LAUNCHER" /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher reconcile
   ```

5. Preserve outputs **before** cleanup. No non-destructive hosted billing-stop
   control has been verified. Cancellation stops the turn, not a proven sandbox
   charge. A completed turn's missing/tampered artifact blocks cleanup admission.
   Ask for action-time approval naming the actual session/environment; the parent
   performs any permanent deletion separately. This runner has no deletion
   command. After approved deletion, install its receipt and verify absence:

   ```json
   {
     "session_id": "ACTUAL_SESSION_ID",
     "environment_id": "ACTUAL_ENVIRONMENT_ID",
     "action_time_approval_reference": "ACTUAL_DELETION_APPROVAL_REFERENCE"
   }
   ```

   ```bash
   sudo -u blueprint "$RESEARCH_CREDENTIAL_LAUNCHER" /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher record-cleanup --date ACTUAL_LOCAL_DATE --input /var/lib/blueprint/researcher/cleanup.json
   ```

   Authenticated session/environment 404s release the next-start guard; underlying
   physical cleanup and billing-stop timing remain unverified. Until cleanup is
   resolved, the next paid date is blocked. This requires daily parent attention;
   unattended indefinite daily operation is not verified under the current
   action-time deletion policy. A separate lifecycle decision is needed to remove
   that dependency, not an automatic deletion feature.
6. After canary verification, approve a coordinated scheduled validation slot.
   Dot prevents bridge duplication for that slot while retaining the bridge as
   fallback. Record that slot's approval in `scheduler_authority_reference`.
   Verify the unit paths and next event, then an approved host operator
   arms the replacement timer:

   ```bash
   sudo systemctl enable --now blueprint-researcher-daily.timer
   sudo systemctl list-timers --all blueprint-researcher-daily.timer
   sudo systemctl show blueprint-researcher-daily.timer -p NextElapseUSecRealtime -p LastTriggerUSec
   ```

   Start target is 7 AM America/Chicago with DST, not a completion deadline or
   exact-instant guarantee. Boot/persistent catch-up selects only the latest due
   date; the durable ledger prevents duplicate create attempts. Verify a natural
   7 AM firing and end-to-end completion. Only then may the parent explicitly
   retire/change its bridge and record final scheduler ownership/cutover in Notion.
7. Rollback/hold through the existing operator door, with applicable approval:

   ```bash
   python3 scripts/operator_door.py hold blueprint-researcher-daily.timer --owner dot --reason 'Research runner rollback; preserve bridge coverage and reconcile active work' --for 2h --wait
   python3 scripts/operator_door.py unit stop blueprint-researcher-daily.service
   python3 scripts/operator_door.py request ACTUAL_RETURNED_REQUEST_ID --wait
   python3 scripts/operator_door.py units --pattern 'blueprint-researcher-*'
   python3 scripts/operator_door.py journal blueprint-researcher-daily.service -n 100
   ```

   Require successful hold/stop receipts and units/status readback; stop and
   escalate if any command fails. SIGTERM requests bounded cancellation using the same event idempotency key;
   at most three unacknowledged attempts are allowed, never create/input retries.
   Killing a process does not prove the remote turn stopped. Use `reconcile`,
   preserve IDs/ledger/artifacts/receipts, and keep new paid starts blocked until
   terminal/cleanup evidence is established. The approved host operator also sets
   `enabled=false` before a temporary hold expires. Holds expire automatically;
   verify/renew them while investigation continues. Dot restores bridge ownership
   explicitly before allowing its next paid slot. Never delete the ledger, change
   state roots, or rerun research to clear an uncertain outcome.

API references: [session management/cancellation](https://developers.openai.com/api/docs/guides/agents-api/sessions/manage),
[immutable artifact retrieval/lifetime](https://developers.openai.com/api/docs/guides/agents-api/environments/files),
[hosted setup and actual tier](https://developers.openai.com/api/docs/guides/agents-api/environments/openai-hosted).
Parent handoff: https://app.notion.com/p/3eb80154161d81348d4de0c3b58940e0.

## Reviewed knowledge integration

[Knowledge snapshot and research v2](KNOWLEDGE.md) is a deliberate opt-in contract.
It supplies a small reviewed Notion mirror as untrusted background, preserves source
review dates, and returns evidence-backed proposed knowledge deltas for parent review.
The default v1 path and all live-operation/schedule gates remain unchanged.
