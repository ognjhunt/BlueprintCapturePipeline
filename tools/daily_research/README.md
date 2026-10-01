# Standalone daily research

The intended adaptive target is ten or more NEW commercial site/task opportunities.
See [ADAPTIVE.md](ADAPTIVE.md) for the disabled daily opt-in and separate test preparation.
Daily $1 soft TOTAL remains unchanged; old rows retain their admitted guards.

**2026-09-30 Render route:** the selected deployment is the existing Render
worker plus existing Firestore, using the recovered adapter. See
[RENDER.md](RENDER.md) for the current deployment and agent-owned QA/publication
contract. The disk/systemd instructions below remain a portable alternative.
Dot is an observer; it is not a required trigger, reviewer, publisher or result
store. An agent supplies source-support review and publication readbacks through
the durable workflow. Permanent session deletion still needs action-time human
approval naming the actual session/environment.

Blueprint research/operations owns this runner. This directory is its source
home; capture/GPU, WebApp outbound, Paperclip, dot and Codex task scheduling are
not runtime dependencies. A Blueprint-owned systemd timer invokes this isolated
Python launcher; OpenAI runs the existing hosted agent; private Blueprint-owned
disk retains the results for any authorized reviewer. Parent review is supported
but the parent is neither the trigger nor the only result store.

This runbook supersedes the earlier controller/operator-door deployment route.
The package exporter copies only an explicit research-file allowlist from an
immutable commit. It never installs units, enables a schedule, transfers a key,
calls a provider or copies private CRM/knowledge inputs. No GPU packages are
required. The recurring authorization is a **$1/run soft total model/search/sandbox
target**, approximately $30/month at target; no monetary hard cap is enforced.

## Route and current readiness

Use an isolated timer on an **existing persistent Blueprint host**, after the
owner verifies that host and its state disk. The shipped timer schedules 07:00
America/Chicago, follows DST, uses one-second timer accuracy and no randomized
delay. OS scheduling or a host outage can still delay execution. Persistent=true
recovers a missed event; the runner processes only the latest eligible Chicago
date and never replays a paid backlog. OnBootSec also reconciles on boot. The
existing lock and durable per-date intent prevent another create after overlap,
a restart or an uncertain POST response.

GitHub Actions is the alternative if delayed execution is acceptable: its
schedule now supports `cron: '0 7 * * *'` plus `timezone: America/Chicago`, but
scheduled jobs can be delayed or dropped. An ephemeral Actions workspace plus
end-of-job artifact upload cannot preserve the before-POST intent through a
crash. A transactional durable ledger binding would be necessary; none is
verified, so this change does not add an active research workflow.

Sources: [GitHub schedule semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
[systemd timer semantics](https://github.com/systemd/systemd/blob/main/man/systemd.timer.xml).

**Not operational yet:** no standalone persistent host or Default-project binding
has been verified. Codex Cloud's successful provider comparison proves access in
that development environment only. The production host's documented binding is a
different project. Do not reuse it or move a development key. An owner must
securely configure/approve the Default binding on the chosen existing host.
No new paid service or capacity is proposed.

## Exact nonsecret bindings

| Name/path | Required value |
| --- | --- |
| `OPENAI_API_KEY` | Owner-managed credential for Default project; never in argv, repo, bundle, report or chat |
| Project | `proj_F2tFJuxLaovJru8RrtXRaqNj` |
| Saved agent | `agent_5a01ec367d1042ef8632bb5f2e6af8b4919909d2abed48ed95` |
| Model / reasoning | `gpt-6.1-sol` / medium, native search, subagents off |
| Hosted template | `envtmpl_0ae967c7bf17424095b6d233c48397a248eca32ecbba404786`, sandbox network disabled |
| `/opt/blueprint/researcher/releases/FULL_SHA` | Immutable isolated bundle, not the full Pipeline checkout |
| `/opt/blueprint/researcher/current` | Link to the reviewed research-only release |
| `/opt/blueprint/researcher-venv` | CPU Python 3.12 and `requirements.txt` (`openai==3.22.1`) |
| `/etc/blueprint/researcher.json` | Initially disabled `standalone.config.example.json` |
| `/etc/blueprint/researcher.env` | Proposed root-managed systemd binding file; owner secure setup/approval required |
| `blueprint` user/group | Existing host account; verify before install, do not silently create |
| `/var/lib/blueprint/researcher` | Private persistent SQLite ledger, receipts, snapshots and exact output bytes; preserve across reboot/releases/rollback |
| `crm.json` | Complete canonical Sheets `Prospects` snapshot, maximum 26 hours old; fresh export before each research start |
| `knowledge.json` / `refresh-policy.approved.v1.json` | Reviewed knowledge snapshot and approved v3 90/30/7 refresh policy; see [KNOWLEDGE.md](KNOWLEDGE.md) |

Freshness eligibility is field-specific; export time does not renew source
checks. The example applies no task/geography filter, so Austin-only or
portioning-only scope is not silently introduced. Missing/conflicted/unsupported
knowledge remains a gap, and consequential availability/geography claims need
live evidence. The adaptive profile records coverage and an honest shortfall when fewer than ten new prospects withstand research. Legacy examples retain their narrow scan guards until explicitly replaced.

CRM authority: [existing Sheet](https://docs.google.com/spreadsheets/d/1n95Ih0Swc-q-kZyUaDHoZh6SVzxvf_zt-CRR7i39bWY/edit).
Reviewed knowledge is private input, not bundled evidence. The reviewed Library
bundle is `libfile_a9fff47b9b908191b69316ae1a438310`, **version 1**; expected
archive SHA256 `c5ad0be3c7034bebd5f2751b713adfa5f93476eb3966a96acbce741fcdde7591`.
Its snapshot content hash is
`965da67a6f9776661c244e9fd64962b5270c1e06c9f970b960023458cbf105f4`.
Use supported consumer materialization and verify bytes; no guessed private URLs.
The refresh policy must be independently owner-reviewed and installed too.

## Build and install while disabled

Local source review/export requires no API key. Substitute the independently
reviewed **full 40-character commit**; never use moving `main` or a branch name:

```bash
RESEARCH_RELEASE_SHA=REPLACE_WITH_REVIEWED_FULL_SHA
python3 -m tools.daily_research.standalone --revision "$RESEARCH_RELEASE_SHA" --output /tmp/blueprint-research-export
cat /tmp/blueprint-research-export/receipt.json
sha256sum /tmp/blueprint-research-export/blueprint-research.tar
```

The output directory must not already exist. Compare its digest/commit with the
reviewed handoff. Copy this **nonsecret** archive/receipt via the existing approved
host file-transfer route. This task has not established such a route; do not
create SSH credentials or use the broad controller deploy endpoint.

On the owner-selected host, establish these facts before installation:

```bash
id blueprint
python3.12 --version
systemctl --version
findmnt -T /var/lib/blueprint
df -h /var/lib/blueprint
systemctl is-enabled blueprint-researcher-daily.timer
systemctl is-active blueprint-researcher-daily.service
systemctl list-timers --all blueprint-researcher-daily.timer
```

Require persistent storage, working flock/SQLite fsync, disk capacity, accurate
host time/tzdata and no active replacement timer. Inventory any existing state;
**do not start with a new empty ledger to erase prior uncertain attempts**.
Preserve old launcher receipts and reconcile all saved session/turn/environment
IDs through authenticated GETs. The parent owns the old schedule/ledger handoff.
If that history is absent or ambiguous, resolve it before a paid canary.

Only after host installation and the narrow credential-binding setup are approved,
with a verified archive in the current directory:

```bash
RESEARCH_RELEASE_SHA=REPLACE_WITH_SAME_REVIEWED_FULL_SHA
sha256sum blueprint-research.tar
sudo install -d -m 0755 "/opt/blueprint/researcher/releases/$RESEARCH_RELEASE_SHA"
sudo tar --no-same-owner -xf blueprint-research.tar -C "/opt/blueprint/researcher/releases/$RESEARCH_RELEASE_SHA"
sudo ln -sfn "/opt/blueprint/researcher/releases/$RESEARCH_RELEASE_SHA" /opt/blueprint/researcher/current
sudo python3.12 -m venv /opt/blueprint/researcher-venv
sudo /opt/blueprint/researcher-venv/bin/python -m pip install -r /opt/blueprint/researcher/current/tools/daily_research/requirements.txt
sudo install -d -m 0700 -o blueprint -g blueprint /var/lib/blueprint/researcher
sudo install -d -m 0755 /etc/blueprint
sudo install -m 0640 -o root -g blueprint /opt/blueprint/researcher/current/tools/daily_research/standalone.config.example.json /etc/blueprint/researcher.json
sudo install -m 0644 /opt/blueprint/researcher/current/tools/daily_research/systemd/blueprint-researcher-daily.service /etc/systemd/system/
sudo install -m 0644 /opt/blueprint/researcher/current/tools/daily_research/systemd/blueprint-researcher-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload
systemd-analyze calendar '*-*-* 07:00:00 America/Chicago'
sudo systemctl list-timers --all blueprint-researcher-daily.timer
```

For an upgrade, retain existing config/state/venv instead of overwriting them.
Do not run `enable` or `start` yet. The owner securely provides the binding file
without displaying its contents and installs reviewed CRM/knowledge/policy inputs
through its approved route. The bundled config remains `enabled=false`.

Verify the real service account/binding using a GET-only transient command:

```bash
sudo systemd-run --wait --collect --pipe --uid=blueprint --working-directory=/opt/blueprint/researcher/current --property=EnvironmentFile=/etc/blueprint/researcher.env /opt/blueprint/researcher-venv/bin/python -m tools.daily_research.runner --config /etc/blueprint/researcher.json --state-dir /var/lib/blueprint/researcher preflight
```

Require exact saved IDs/config/skills, fresh input digests and reconciled existing
runs. GET success does not prove inference, report transport, physical cleanup,
actual sandbox size, or the eventual unattended wake.

## Canary and one-trigger cutover

1. Parent/operator reserve one explicit canary date and reconcile the old
   automation's attempt ledger. Prevent it from issuing that date's create.
   Leave the new timer disabled. Canary spend remains separately gated.
2. After 07:00 on that Chicago date, the approved operator sets `first_date` to
   the reserved date, records its real canary authorization in
   `scheduler_authority_reference`, and sets `enabled=true` in the private config.
3. One manual service start exercises the exact installed credential/runtime:

   ```bash
   sudo systemctl start blueprint-researcher-daily.service
   sudo systemctl status blueprint-researcher-daily.service --no-pager
   sudo journalctl -u blueprint-researcher-daily.service -n 100 --no-pager
   ```

4. Require the actual session/root-turn/environment IDs, terminal result, exact
   downloaded artifact and digest, source review, duplicate checks and working
   owner-controlled report retrieval. Never automatically repeat an uncertain
   POST. To observe the same run, use the transient command above with
   `reconcile` replacing `preflight`; it cannot create a new session.
5. Close out that hosted session under the lifecycle gate below. A successful
   service exit or idle sandbox is not sufficient.
6. Only after all these facts are verified, parent disables old automation
   `6abc4ffae84881919154bba45f749074` and records readback. The owner changes the
   scheduler reference to the verified standalone cutover record and arms the
   sole independent timer:

   ```bash
   sudo systemctl enable --now blueprint-researcher-daily.timer
   sudo systemctl list-timers --all blueprint-researcher-daily.timer
   sudo systemctl show blueprint-researcher-daily.timer -p NextElapseUSecRealtime -p LastTriggerUSec
   ```

   Persistent/on-boot catch-up can fire when armed; coordinate its due date as
   part of cutover. Verify the first real unattended 07:00 wake and saved output
   before claiming daily operation. The parent must not activate both triggers.

## Owned reports and review

The private state directory is the canonical store: `<date>-status.json`,
`<date>-artifact.json` (exact downloaded bytes), `<date>-evidence.json` and
`<date>-review.json`. Findings, source URLs/dates/classification, confidence,
unknowns/blockers and proposed next actions are for review. Validation proves
structure/binding, not that a source supports a factual claim.

Before activation, the operator demonstrates one authorized reviewer can retrieve
these files through an existing Blueprint-controlled read/export route, and
records that route, immutable release/hash, schedule, lifecycle constraints and
report links in the [Notion handoff](https://app.notion.com/p/3eb80154161d81348d4de0c3b58940e0)
and [operations hub](https://app.notion.com/p/3ea80154161d81c7810cc42e9e7df9c5).
Notion is a linked review surface; owned disk remains the execution/result store.
Do not publish raw prospects or add a new public bucket/transport credential.

Any authorized reviewer may submit the existing `review --date DATE --input FILE`
command with packet digest, reviewer reference, source-support confirmation,
canonical CRM recheck, accepted keys and concise summary. The resulting pinned
local outbox has `sheets` and `notion` proposals. An approved agent connector
operator maps explicit Sheet columns, preserves manual edits and records actual
readback using `receipt --date DATE --input FILE`. Identical receipts are
idempotent; conflicting ones refuse. `completed` requires both publication receipts.
No sink write, prospect email, raw-report blast or Slack send is automatic. These
local review/receipt commands require no API key and may run on an approved
private copy of state for inspection; mutations belong in the canonical ledger.

## Lifecycle and rollback

The runner never deletes a session. Cancel stops a turn; no non-destructive
hosted sandbox billing-stop control is verified. Idle is not documented billing
shutdown. Preserve the artifact before any approved cleanup. Permanent deletion
still needs **action-time approval naming the actual session/environment**;
an authorized operator performs it separately. Install that approval receipt and
use `record-cleanup --date DATE --input FILE` through the same secure GET binding.
The runner requires authenticated session/environment 404s before clearing the
next-run guard. Physical reclamation/billing timing remains unverified.

**A timer alone cannot make indefinite daily operation ready:** after the canary,
unresolved hosted cleanup blocks the next paid date. Current policy needs an
owner/operator closeout process each day or a separately approved documented
lifecycle solution. This package does not introduce automatic irreversible
cleanup or broaden deletion authority.
[OpenAI session lifecycle](https://developers.openai.com/api/docs/guides/agents-api/sessions/manage).

Rollback uses the independent host's ordinary service controls, not Pipeline
app health or an operator-door hold action:

```bash
sudo systemctl disable --now blueprint-researcher-daily.timer
sudo systemctl stop blueprint-researcher-daily.service
sudo systemctl is-active blueprint-researcher-daily.timer
sudo systemctl is-active blueprint-researcher-daily.service
```

Read back both stopped states; stop requests bounded cancellation and may leave a
remote turn unresolved. Set config `enabled=false`, reconcile that same session
with `reconcile`, preserve every ledger/output/receipt, and restore only the
previous reviewed research release link if needed. Never delete/reinitialize the
state directory, switch projects/models, reactivate dot as the permanent trigger,
or assume a local service stop closes the hosted sandbox.
