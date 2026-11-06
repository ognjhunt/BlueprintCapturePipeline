# Operator door

The fixed [`install-public-franka-policy` operation](research/openwam-franka-integration-2026-10-30.md)
installs OpenWAM Alpha on the distinct single-FR3 + nominal Robotiq2F85 simulation
body from retained imported assets, with pinned public license evidence and no compute.

The fixed [`install-scripted-robot-baselines` operation](scripted_robot_baseline_registration.md)
registers all supported scripted configurations from one exact retained public
robot preparation batch, with individual unavailable reasons and no compute.

The fixed [`install-public-yam-policy` operation](research/molmoact2-yam-registration-2026-10-29.md)
installs the public MolmoAct2 YAM screening registration from retained prepared
assets and verified official license evidence, without launching compute.

The fixed [`install-public-piper-policy` operation](research/openwam-piper-registration-2026-10-29.md)
installs the public OpenWAM PiPER screening registration from retained prepared
assets, with a registry digest check and scoped public-license evidence. It
launches no compute and accepts no arbitrary paths, commands or endpoints.

The operator door is a token-gated HTTPS window onto the control-plane host
for agent sessions that cannot SSH to it. Claude Code cloud sessions send all
egress through an HTTPS proxy, so `ssh root@<host>` has no path from them. The
door gives those sessions the host access a website scene run actually used,
measured from 11,749 SSH commands across the September 2026 runs: run-state
reads, directory listings, unit state and journals, the deploy cycle, starting
controller oneshots, pausing timers around deploys, and downloading receipts
and artifacts. It deliberately offers no shell, and it never runs unreviewed
code as root: only commits already on `origin/main` can be deployed, and there
is no candidate-code stage replay. Replay a failed stage in the cloud session
instead, against inputs pulled through the door (see "Using it").

Code: `deploy/operator-door/` (standard library only), units in
`deploy/systemd/blueprint-operator-door*`, client `scripts/operator_door.py`,
edge route in `deploy/caddy/Caddyfile`.

## How it is put together

```
cloud session ──HTTPS──▶ Caddy /api/live-pipeline/operator/* ──▶ 127.0.0.1:8767
                         (TLS; bearer added by the              blueprint-operator-door.service
                          cloud proxy, never in the VM)          User=blueprint, read-only sandbox
                                                                    │ writes validated request
                                                                    ▼
                                             /var/lib/blueprint-operator-door/requests/pending/
                                             (root:blueprint-door 2770; only the door has that group)
                                                                    │ PathExistsGlob
                                                                    ▼
                                             blueprint-operator-door-runner.service (root oneshot,
                                             no network, no capabilities): revalidates, then
                                             `systemctl --no-block …` or `systemd-run` of a fixed script
```

- **Independent of releases.** The door runs from `/opt/blueprint/operator-door`
  on the system `python3`, not from a pipeline release or virtualenv, so a broken
  release cannot lock cloud sessions out of the host that would fix it.
- **Door process.** `User=blueprint` (the pipeline writes state with
  `UMask=0077`, so only that account can read it) plus `systemd-journal` for
  journal reads. `ProtectSystem=strict` with only its state directory writable,
  `IPAddressAllow=localhost`, no capabilities, and `InaccessiblePaths=` for every
  known secret location. `systemd-analyze security` exposure 1.4.
- **Runner.** Root oneshot with no capabilities, `PrivateNetwork=true`, writable
  spool only, and `StartLimitIntervalSec=0` so a burst of requests cannot stop
  the path unit. It trusts nothing in `pending/`: no symlinks or FIFOs, bounded
  size, id must match the file name, the door's schemas are applied again, and
  anything unreadable is quarantined rather than retried forever. `processing/`,
  `completed/` and `results/` are root-owned, so no service-account process can
  plant a link where root writes.
- **Transient scripts.** Deploys and door upgrades run as transient units
  (`blueprint-operator-door-{deploy,upgrade}-…`) that receive only validated
  values as `DOOR_*` environment variables, and write
  `results/<id>.outcome.json` and `results/<id>.log`.

## Secrets never leave

Secrets on this host are not confined to `/etc/blueprint/provider-secrets`: env
files and their backups under `/etc/blueprint`, `.env` files, a service-account
key and forwarded-token JSON under `/var/lib/blueprint/pipeline-control-plane`
are readable by the service account. Three layers keep them out of responses:

1. systemd makes the known secret directories and files inaccessible to the
   door process;
2. the file API refuses credential-looking names (`*.env` and backups,
   `*secret*`, `*token*`, `*credential*`, `*service-account*`, `api_key`,
   `id_rsa`-style keys, key and certificate extensions);
3. every byte range served, every archive member and every journal line is
   scanned for credential-shaped content (private keys, service-account and
   OAuth fields, provider key formats, JWTs, signed URLs, bearer headers and
   `NAME_KEY=`/`_TOKEN=`/`_SECRET=` assignments).

Two more rules close what patterns alone would miss. Under `/etc/blueprint`
only `*.json` files can be read at all. Compressed files (gzip, zip, xz,
bzip2) are scanned after decompression, with the verdict cached per file
version; zip members are also judged by name, nested archives are refused, and
formats the standard library cannot decompress (zstd, 7z, rar) are refused.
Anything under a `.git` directory is refused, since packed objects hide content.
Every pattern is bounded, so pathological input scans in linear time.

Over-refusal is the accepted failure mode. A refused file is reported by name
and rule, never by content; an archive lists what it skipped and why in a
`.operator-door-manifest.json` member.

## API

Base: `https://<control-plane host>/api/live-pipeline/operator/v1`. Every route
but `healthz` needs `Authorization: Bearer <token>`. Scopes are `read`,
`operate` and `deploy`.

| Route | Scope | Returns |
|---|---|---|
| `GET /healthz` | none | `{"ok": true, "version": …}` |
| `GET /whoami` | read | token name and scopes |
| `GET /status` | read | deployed commit and blockers (loopback `/version`), active release, deploy units in flight and recent receipts, paid-launch lock holders from `/proc/locks`, spend guard, failed and controller units, disk, `capacity` (the capacity controller's `summary.json`: level, alerts, per-mount admission and the usage survey by storage class, root and owner), load, `memory` (see below), door queue, owned timer holds (including overdue ones), and unreported break-glass note count/latest summary |
| `GET /fs/list?path=&sort=name\|mtime&match=` | read | directory entries (capped) or a file's metadata |
| `GET /fs/read?path=&offset=&length=` | read | bytes; `X-Door-Size`, `X-Door-Offset`, `X-Door-Length`, `X-Door-Eof` headers for paging |
| `GET /fs/archive?path=` | read | tar.gz of a directory (512 MiB cap, two at a time) |
| `GET /journal?unit=&lines=&since=` | read | `journalctl -u` text, redacted |
| `GET /units?pattern=&state=` · `GET /units/show?unit=a,b` | read | `blueprint-*` unit rows or properties (never `Environment`) |
| `POST /requests` | per kind | `202 {"id": …}` |
| `GET /requests` · `GET /requests/<id>` | read | queue, or one request with result, outcome, redacted log tail and live unit state |

`status.memory` answers "what holds host RAM?" read-only. `meminfo` gives
`MemTotal`, `MemAvailable`, `SwapTotal`, `SwapFree` and `Shmem` in bytes from
`/proc/meminfo`; `slices` gives cgroup v2 `memory.current`/`memory.peak` for
`system.slice`, `user.slice` and `init.scope`; `system_units.top` lists the 20
largest direct children of `system.slice` by `memory.current` (every unit, not
only `blueprint-*`; a `system-*.slice` row sums its template instances), with
`count` and `scan_truncated`; `tmpfs` gives total and used bytes (statvfs) for
`/dev/shm`, `/run` and `/tmp` when each is a tmpfs in the door's view;
`oom_reports` parses the last 4 kernel OOM-killer reports of the past 24 h from
`journalctl -k` (newest 20000 lines, 8 MiB, 10 s; `count` and `log_truncated`
say what the window held) into each event's UTC time, invoking task, constraint
and unit-name cgroups, the killed task (pid, uid, `anon/file/shmem-rss` kB,
`oom_score_adj`) and the task table's 15 largest rows by rss (pages, with
`page_size_bytes`) plus per-uid totals, or only row counts when the table's
header is not recognised. Only numbers, uids and unit or task names are
returned, every read is capped, and anything missing or unreadable is `null`
rather than an error.

`status.notifier_binding` is a fixed-target view of the live control-plane
postcheck binding. It reports safe unit/drop-in paths, an opaque command hash,
known postcheck script paths and whether the release-selector key occurs.
It never returns raw commands or `Environment`, and does not grant file access
to `/etc/systemd`. This metadata alone does not prove the imported notifier:
the naturally emitted alert audit owns its source identity and full payload.

The legacy `blueprint-pipeline-control-plane` loop was retired on 2026-11-06
(P-RT-06): the repository no longer deploys or installs it, and `notifier_binding`
reads `unknown` once an operator removes the host's unit. Until then,
`unit repair-notifier-binding blueprint-pipeline-control-plane.service` is a
fixed migration of the legacy `95-blueprint-active-release.conf` postcheck
invocation. Supply `--expected-postcheck-sha256` from fresh status and
`--expected-source-commit` from the proven active release. It requires the
explicit dispatcher stop, preserves all other directives/hooks, retains a
root-only original backup, and reloads systemd without starting the service.
Unknown hook shapes, symlinks or identity drift are refused. No general path
or command setter is exposed; judge recovery from the next natural alert.

Read roots: `/var/lib/blueprint`, `/opt/blueprint`, `/workspace`,
`/mnt/blueprint-work`, `/etc/blueprint` (names screened as above) and the
door's own state. Paths resolve to their real location and must stay inside a
root and outside every hidden path.

Request kinds:

`public-embodiment-preparation` (`operate`) submits one explicitly funded batch
of public robot asset imports. It accepts only `kind` and a nested `request`
using `embodiment_asset_preparation_request.v1`. The sealed request includes:

- A stable `authorization_id`, exact installed 40-character `release_commit`,
  and ordered, unique `embodiment_ids` from the public runtime-profile allowlist.
- `max_spend_usd` at most $4, `max_hourly_rate_usd` at most $0.80, and
  `ttl_seconds` from 1,800 to 14,400. The hourly ceiling times the full TTL must
  fit the spend cap.
- `expires_at_epoch` between now plus TTL plus five minutes and now plus seven
  days, and a printable `authorization_reference` describing the authorization.
- `request_digest`, computed by the pipeline's `canonical_digest(request,
  digest_field="request_digest")`. The ordered candidate list and every grant
  field are bound to this digest.

Submit the reviewed, sealed JSON file explicitly:

```bash
python3 scripts/operator_door.py public-embodiment-preparation --request-file grant.json --wait
```

The client derives a stable operation key from the authorization ID. Retry the
same file after a lost response; do not create a new authorization ID to recover
an uncertain allocation. Different request bytes under an existing ID are
refused. The door transient has no network, reads no provider credentials, and
invokes only the installed controller's `enqueue` as `blueprint`. The fixed
`blueprint-embodiment-preparation.path` activates its allocator-backed service
after durable publication under
`/var/lib/blueprint/pipeline-control-plane/embodiment-preparation`.
`--wait` confirms the enqueue receipt, not successful asset imports or policy
episodes; inspect the retained batch results and service journal for those
outcomes. Installing the lane creates no grant. This request cannot select
paths, URLs, images, executable code, policy runs, or task execution.

The existing `/var/lib/blueprint` read root includes `latest.json`,
`authorizations/<authorization_id>.json`, and `jobs/<authorization_id>/` below
that preparation directory. Published model receipts are at
`/var/lib/blueprint/robot-models/batches/<authorization_id>.json`; each retained
model has `binding-inputs.json` and `body-result.json` beneath its embodiment
and version directory. Use the ordinary `cat`, `ls`, or checked `pull` commands.
Both the door and preparation service run as `blueprint`, so private owner-only
records remain readable without granting broader filesystem permissions.
Existing secret-path and credential-content refusals still apply.

Deploys bind fallback `TMPDIR` to `/mnt/blueprint-work/control-plane-deploy-staging`.
The installer creates only an absent root-owned0755 leaf and then runs the
launcher's same guard before swapping code. Every path component must be a real
root-owned directory, nonwritable by group/others and readable/traversable by
the service account's actual group or others. The parent must share the work
volume's device rather than root's. Unsafe paths refuse without repair.
The original full-copy disk reservation remains unchanged. Verified vendor
hardlink reuse retains its existing staging override and discount. The launch
guard creates or removes no files and exposes no caller-selected path.

The existing `selected-handoff` operation also supports `atlas-status` and
`atlas-retry` for one retained original website browser capture. Both keep the
immutable GCS generation/hash/size selectors, current original-owner and source
membership checks, and retirement admission. Status reads the existing job and
the signed WebApp `atlas-configuration` operation with `create:false`; an older
server or an absent/expired grant refuses without creating or extending one.
It returns source identities, job state, grant cap/expiry and separate Atlas and
native preparation ledger summaries. Settlements are retained ledger debits,
not independently verified provider billing. The optional
`capture_inference_accounting` projection supplies retained usage pricing and
explicit unknowns outside preparation from the exact capture's call ledger and
verified linked assessment. Its linked assessment cost can overlap its call
rows and is never added to them. Privacy's historical cost remains unknown;
an absent projection is `null`, not zero cost. These facts enter the reviewed
snapshot digest without creating a new spending grant or unknown-cost gate.
The two lane caps are never a combined allowance or the remaining authorization
for an entire run.
Only the noncreating Atlas configuration read allows up to 200,000 response
bytes, covering the 128 preparation and 64 inference row limits. Other control
reads retain 100,000 bytes. Oversized responses fail closed without truncation;
unbounded configuration prose does not make the transport unbounded.

Use `selected-handoff --selector-file SELECTOR --atlas-status --wait` first.
After reviewing that result against the run's authorized cumulative ceiling,
use `--atlas-retry --atlas-status-digest sha256:... --wait` with the same selector.
The existing preview job lock protects a fresh source/rights/grant/accounting
check and refuses any snapshot drift. Only `website_atlas_preview.retry_failed_job`
may requeue that failed job. It preserves its original seven-day expiry, source,
grant and bought receipts; the later timer can make paid calls through existing
admission. Status/retry never advance the job or alter timer holds. Native resume
options, paths, commands, profiles and spending overrides are refused. Both run
as `blueprint`; retry can write only the selected capture's `pipeline/atlas`
directory and the door's result directory. This recovery supports the beta scene
journey and the ADP-009D development rehearsal without raising evidence claims.

`--reissue-preparation --atlas-status-digest sha256:...` (mode
`preparation-reissue`) re-issues a completed native preparation as a new job
after its sponsor renewed the expired grant in the WebApp. It takes the same
review, Atlas lock and ignored blockers as `--resume-preparation`, but requires
the native job to be `completed`. It never invokes the native handler or buys
anything. `website_preparation_renewal` refuses unless the old intent was revoked or
expired with every attempt cancelled or settled, no sibling of the capture is
live, the current task equals the stored one and the fresh grant renews exactly
the old one. It then writes the renewed preparation under
`renewals/<authority hex>/`, recomputes its CPU native inputs, registers it and
enqueues it at the WebApp. The intake then charges every sibling's holds against the
capture's original caps. A WebApp refusal (for example
`website_scene_preparation_budget_exhausted`) is a typed `blocked` outcome
recorded in `reissue.json`. It runs as `blueprint` under the production runtime guard
and writes under `/var/lib/blueprint`. The owner approved it for the
dispatcher-hold-only profile on 2026-11-02 ("approve adding preparation-reissue
to the operator door's restricted profile"): it acts only on a grant the owner
renewed and buys nothing.

The same profile admits `confirm-owner-success-contract` (below) as it admits
the budget grant: a preview, then an apply bound to the preview's digest, that
writes one sealed record naming the owner as approver. It changes no
threshold, starts no compute and moves no money. It was added to the profile
in its own commit for the owner's explicit sign-off.

The profile also admits `restore-g1-runtime-source` (below). The owner approved
it on 2026-11-06 ("build a door operation for it next", then "do all this"):
it restores one pinned, digest-verified packet and its receipt from the
retained object-store copy to the paths the shared G1 registry already names,
overwrites nothing, starts no compute and moves no money. It was added to the
profile in its own commit.

| Kind | Scope | Body | What the runner does |
|---|---|---|---|
| `deploy` | deploy | `commit` on `origin/main`, `wait_for_idle` | `door-deploy.sh`. While any `blueprint-*deploy*` unit is active, an iteration deploy is parked until it ends (see below); a deploy with release provenance is refused |
| `unit` | operate | `unit` (`blueprint-*`), `action` `start`\|`reset-failed`\|`stop`\|`restart` | `systemctl --no-block <action> -- <unit>`; bare `stop` of a `.timer`/`.path` refuses with `unit_stop_requires_hold`, and `restart` remains limited to safe triggers. The door's own and safety-critical triggers cannot be paused or restarted. |
| `hold` | operate | `.timer`/`.path` `unit`, `owner`, printable `reason`, `expires_in_seconds` (60–86400) | stops and disables the unit, writes a root-owned hold record with its prior boot policy, and schedules an expiry timer; a different owner cannot replace an active hold |
| `release-hold` | operate | `.timer`/`.path` `unit` | restores the prior boot policy, starts the held unit, and records who released it; refuses if no active hold exists |
| `door-upgrade` | deploy | `commit` on `origin/main` | `door-upgrade.sh`: that commit's `install.sh --upgrade`, rolled back on a failed health check |
| `retire-scene-workspace` | operate | `scene_id`, optional `bucket`, `apply` (default `false`) | `door-retire-scene-workspace.sh`: the active release's `website_scene_workspace_retention retire` for that scene; without `apply` it only plans. Outcome `planned`, `retained` (code = first reason), `retired` or `failed`; the module's full result is `results/<id>.retirement.json` |
| `restore-scene-workspace` | operate | `scene_id`, required `bucket` | `door-restore-scene-workspace.sh`: replays the retired receipt to the canonical workspace path, checks every byte and moves the historical receipt aside. Outcome `restored` or `failed`; the module's full result is `results/<id>.restore.json` |
| `provider-output-resume` | operate | `run` (a canary dispatch directory), `attempt` (1–999), `ingest` (default `false`) | `door-provider-output-resume.sh`: the active release's `provider_output_promotion resume --attempt-root <run>/allocator/attempts/attempt_NNN [--ingest]` for one streamed policy-canary attempt, run as `blueprint` with umask 0077 (the attempt tree is the service user's): promotion to B2, the gated cleanup, the staged-object absence proof and, with `ingest`, the needed members' ingestion, which short-circuits once materialized. It refuses, touching nothing, until the attempt's teardown manifest records `continuing_spend_from_this_run: false` (`provider_output_resume_attempt_not_torn_down`). Outcome `completed`, or `blocked` with the first blocker as its code; the module's summary is `results/<id>.provider-output-resume.json` |
| `restore-g1-runtime-source` | operate | `apply` (default `false`) and nothing else: every path, digest and object identity is pinned in the release module | `door-restore-g1-runtime-source.sh`: the active release's `native_g1_runtime_source_restore [--apply]`, as `blueprint` with umask 0077. It keeps the network and the provider secrets the WAM object store needs; every other secret is hidden, and only `/mnt/blueprint-work`, the disk reservation ledger and the door's results are writable. It checks the build record `/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926-build.json` (pinned size, sha256 and `receipt_digest`), HEADs the content-addressed copy the 2026-09-27 run retained (exact size and recorded sha256), checks the disk headroom (the packet's 4.64 GB plus 256 MiB) and reports both target paths and whether they exist. Without `apply` it writes nothing but its result. With `apply` it refuses a target that holds other bytes and returns `already_restored` when both verify. Otherwise it holds the disk in the ledger, streams the object into a temporary file (sha256 and size checked as it lands, the file removed on any mismatch), links it to `native_task_runtime_sources.zip` and writes `native_task_runtime_source_packet.v1.json` as the build record's exact bytes (directory 0750, files 0640, nothing replaced). Last it runs `verify_native_task_runtime_source_packet`, removing what it created if that fails. It starts no compute and spends nothing but one download. Outcome `planned`, `restored` or `already_restored` (the last two carry `receipt_sha256` and `receipt_size_bytes`, the install request's `runtime_source`), `refused` (a fixed `g1_runtime_source_*` code) or `failed`; the full result is `results/<id>.g1-runtime-source.json` |
| `extend-scene-execution-window` | operate | `intent_id` (`scene-<64 hex>`), `intent_digest`, `expires_at` (UTC `YYYY-MM-DDTHH:MM:SSZ`), `authorization_reference` (names the owner's approval, ≤ 500 printable chars), `ack` = `extend-scene-execution-window`, `apply` | `door-extend-scene-execution-window.sh`: the active release's `task_evaluation_scene_execution_window_door`, as `blueprint`. It reads the capture's current rights from the WebApp and refuses a revoked intent, a digest mismatch, a withdrawn capture or revoked consent, and any expiry not after now, more than 7 days ahead, or more than 7 days past the intent's original expiry. Without `apply` it only plans. With `apply` it writes one sealed `authorized_via: operator_door` extension (door caller, door request id, the owner's reference, every execution bound unchanged) and sends it to the WebApp's signed internal endpoint, retrying a transport failure twice. An equal or earlier expiry is `already_covered`; with `apply` it re-sends the covering extension, and a plan never writes or POSTs. Outcome `planned`, `extended`, `already_covered`, `refused` (code) or `failed` (`webapp_notice_failed`); the full result is `results/<id>.scene-window.json` |
| `extend-scene-execution-budget` | operate | `intent_id` (`scene-<64 hex>`), `owner_user_id`, `owner_organization_id`, `approved_by` (must be the owner user), `approval_evidence` (≤ 500 printable chars), `max_total_spend_usd` (0 < x ≤ 1000), `max_paid_attempts`, `apply`, `apply_ack` (the preview's `apply_ack` digest with `apply`, else empty), `ack` = `extend-scene-execution-budget` | `door-extend-scene-execution-budget.sh`: the active release's `task_evaluation_scene_execution_budget_door`, as `blueprint`, with no network and every secret hidden. Without `apply` it only previews the current and requested ceilings and their `apply_ack` digest. With `apply` and that digest it appends one owner grant through `extend_scene_execution_budget` (monotonic, bound to the intent's issuer and owner; the reference names the approving owner, the evidence, `channel=operator_door`, the door caller and request id). It refunds, settles and rewrites no hold. Outcome `planned`, `extended`, `refused` (code) or `failed`; the full result is `results/<id>.scene-budget.json` |
| `confirm-owner-success-contract` | operate | `intent_id` (`scene-<64 hex>`), `owner_user_id`, `owner_organization_id`, `approved_by` (must be the owner user), `approval_evidence` (≤ 500 printable chars), `apply`, `apply_ack` (the preview's `apply_ack` digest with `apply`, else empty), `ack` = `confirm-owner-success-contract`; no threshold is a field | `door-confirm-owner-success-contract.sh`: the active release's `task_evaluation_owner_success_confirmation_door`, as `blueprint`, with no network and every secret hidden (the budget grant's sandbox). It finds the intent's published job from retained host records only: the progression's completion names the launch, the launch's validated result names the revision (its bytes must be the progression's published result), and the success criteria and task template are the copies the completion join retained under the launch's `robot-free-readiness/documents/`, each checked against the revision. Without `apply` it only previews: the revision digest, the success document, the thresholds exactly as built, the `proposal_only` authority they carry, any existing confirmation, and an `apply_ack` digest of the binding, owner, approver and evidence. With `apply` and that digest it writes one sealed `task_evaluation_owner_success_contract_confirmation.v1` record under `<intent>/owner-success-confirmations/<digest>.json` (exclusive; a matching record already there is returned as `already_confirmed` and nothing is written). It refuses another owner, an approver who is not the owner, a revoked intent, a job not yet published, missing or changed retained documents and an authority that is not `proposal_only`. It changes no threshold, starts no compute and moves no money. Outcome `planned`, `confirmed`, `already_confirmed`, `refused` (code) or `failed`; the full result is `results/<id>.owner-success.json` |

**`deploy` is root-equivalent.** Whoever holds it can get code that is on
`origin/main` running as root, which is what a deploy is. Give it only to
tokens that should be able to ship; `read` and `operate` never run code.

Every script runs in a transient unit named
`blueprint-operator-door-{deploy,upgrade}-<sha12>-<id8>.service` or
`blueprint-operator-door-retire-<sha256(scene_id)[:12]>-<id8>.service` (a hash, so
a scene id can never match `blueprint-*deploy*`), or
`blueprint-operator-door-output-resume-<sha256(run/attempt)[:12]>-<id8>.service`, or
`blueprint-operator-door-g1-runtime-source-<sha256(plan|apply)[:12]>-<id8>.service`, and the
request view shows its live state. Each unit's `RuntimeMaxSec` equals its start timeout,
because `TimeoutStartSec` does not bound an `exec` unit once it has started:
deploy 3 h, door upgrade 30 min, retirement, provider-output resume and the G1 runtime
source restore 2 h. A retirement unit is further
sandboxed (`ProtectSystem=strict`, `PrivateTmp`, `PrivateDevices`, `ProtectHome`,
`ProtectKernelTunables`, `ProtectControlGroups`, `NoNewPrivileges`, and a narrow
capability set including `CAP_SYS_PTRACE` for the `/proc` reference check) and may write
only `/var/lib/blueprint/pubsub-handoffs`, the disk reservation ledger, the
storage-pin lock, and the door's results. Deploy and door-upgrade also have
`RuntimeMaxSec` limits and receive SIGTERM when they run past them. It reads
`/etc/blueprint/pipeline-control-plane.env` as data, never
as shell (only `KEY=VALUE` lines, matching quotes stripped, nothing expanded or run,
and only `BLUEPRINT_*`, `GOOGLE_*` and `GCLOUD_PROJECT` exported), never echoes it,
and archives to the same artifact store as the reclaim timer. See
"Scene workspace retirement" in `docs/CONTROL_PLANE_STORAGE.md` for what it checks.

### What a door deploy does

`door-deploy.sh` repeats the deploy the agent lanes ran by hand over SSH:

1. fetch `main` into one long-lived source clone,
   `/opt/blueprint/control-plane-config-tools/operator-door-source` (origin is
   GitHub, so the deploy tool's pushed-commit checks see real refs), with the
   door's read-only deploy key, and forget any other remote branch an older door
   fetched;
2. require the commit to be an ancestor of `origin/main`;
3. with `wait_for_idle` (the default), wait for scene progression and
   configured-controls progression: a target tool that supports `--drain-unit`
   stops only those services' own triggers and lets runs already in flight
   finish (up to 30 minutes) before it takes the paid-launch gate, so nothing
   new starts before the release moves while the spend guard and every other
   watcher stay live; an older tool gets the original idle poll;
4. run the **target commit's** `scripts/deploy_control_plane_commit.py` from a
   throwaway worktree with `--iteration --preserve-configured-controls-state`,
   receipt `deploy-receipts/iteration_<sha12>_door.json`.

The request log prints `phase=` timestamps for each of these steps, ending at
`phase=tool_start`; the receipt's stage timings cover the tool from there.

The deploy builds the commit's release checkout and runtime trees first, with
every timer, path watcher and paid launch still live: those trees are the
commit's own and nothing reads them until the release link moves. Only then
does it drain (step 3), take the paid-launch gate and stop every watcher, for
the steps that change what a running unit or a launch reads, the switch, the
intake restart and the re-arm. The receipt's `trigger_pause_seconds` says how
long drained triggers, all other watchers and new paid launches were held;
`stage_timings_seconds` splits the whole deploy by stage.

**Request a deploy right after merging.** Run `deploy <merge sha> --wait` at once;
the build overlaps main's CI. Before anything pauses, the tool waits for that CI
to list the commit's source attestation (`--attestation-wait-seconds`, default one
hour). It paces its requests, because this host reads GitHub unauthenticated at 60
requests an hour. The wait grants nothing: the signed proof is still verified
after the pause, before the privileged installer runs. If the bound expires, or
main's CI for that commit finishes without an attestation, the deploy fails with
`deploy_scene_retirement_runtime_unproven` and nothing has been paused. The
receipt's `runtime_diagnostic` then has phase `await_attestation` and reason
`deadline` or `ci_finished_without_attestation`. A successful receipt records
`attestation_wait_seconds`.

**Deploys requested during a deploy wait their turn.** An iteration deploy
requested while another deploy runs gets the result `parked`, not a refusal. Up to
16 wait. Within about a minute of the running deploy ending, the newest request
starts: the hold sweep's timer wakes the runner, so this also survives a reboot.
That deploy fetches main, then `parked_settle.py` decides what it can from what is
true now:

| Situation | Result |
|---|---|
| The active release, proven by that deploy's own `deployed` receipt newer than the release link, contains the commit | `superseded`, with `superseded_by_commit` |
| It was already inside what was running when it was requested | `failed`, `parked_rollback_refused`. To roll back, request it when no deploy runs |
| What runs now is older than what ran when it was requested: someone rolled back while it waited | `failed`, `parked_release_rolled_back` |
| It is ahead of the active release | It can deploy. The commit that goes out contains the starting request's own and the most others |
| It is the active commit, and no receipt proves that deploy finished | It can deploy again |
| It is neither ahead of nor inside the active release, or inside an unproven one | It stays parked while another request can deploy. A later commit may contain it. If none does, `failed`, `parked_commit_not_contained` or `parked_active_release_not_proven` |

**Nothing is reported as satisfied before a receipt proves it.** A request a deploy
covers stays parked until that deploy finishes, then ends `superseded` on the next
wake-up.

**A deploy runs for the request that named its commit.** Its log and outcome are that
request's, whichever request happened to start. If the commit fails, only that request
fails, so a commit whose CI went red cannot take older good ones with it: each of the
others gets its own try.

A failed fetch, a failed launch, an unreadable release link, a refused staging guard
or a held deploy lock says nothing about a request, so all of them stay parked and try
again on the next wake-up. A file in `parked/` nothing can use is dropped, and the rest
go on. A deploy killed outright ends `refused`, code `parked_deploy_interrupted`, even
if it had finished and only its outcome was lost: read the receipt.

A request 8 hours old ends `refused`, code `deploy_parked_expired`. That is as long as
a client waits, and longer than two deploys at their limit, so nobody is waiting for
it and it may no longer be wanted. A new request deploys normally.

A deploy requested directly when nothing is running starts at once, even with others
parked, and keeps the power to roll back. So a queue that cannot start never stands in
the way of the deploy that repairs it. To replace a parked commit, merge the fix and
request that: the newer commit wins. `status` counts them under `door_requests.parked`,
including the one running. The runner parks and starts them with no network or git, as
sandboxed as before.

`deploy --latest-attested --wait` resolves the newest `origin/main` commit whose
"SBOM, license, and provenance contract" job succeeded, from GitHub's public API.
It refuses unless that commit is ahead of the active release, or is the active
release with no `deployed` receipt. Naming a commit deploys it regardless.

`--preserve-configured-controls-state` keeps the owner's configured-controls
pause across the deploy; `deploy_control_plane_iteration.sh` and
`deploy_control_plane_canary.sh` do not pass it, which is why the door does
not call them. The deploy tool's own guards still apply (paid-launch locks,
dirty or unpushed sources, surface agreement, disk budget). An active door hold
also takes precedence over automatic timer/path arming: deploy leaves that
unit stopped and records its owner, reason and expiry in the receipt. If the
holds directory cannot be read, the receipt alerts `door_holds_unreadable`.
The signed hotfix overlay path of the iteration wrapper is not offered through
the door. The Atlas provider poll (`blueprint-website-atlas-provider-poll.timer`
and `.path`) is the one exception to the deploy's quiesce: it only reads
operations already bought, so it keeps running through a deploy and is
restarted with the release; see "Provider poll and completion webhooks" in
`docs/runbooks/atlas-beta.md`.

Pause a safe trigger through the door with an owner and automatic expiry:

```bash
python3 scripts/operator_door.py hold blueprint-task-evaluation-scene-progression.timer \
  --owner alice --reason "inspect capture" --for 2h --wait
python3 scripts/operator_door.py release-hold blueprint-task-evaluation-scene-progression.timer --wait
```

The root runner keeps `requests/holds/<unit>.json` at mode 0644. It accepts a
hold only when `systemctl cat` shows the matching `ConditionPathExists` guard
in the unit and systemd reports `NeedDaemonReload=no`. An unguarded or stale
host-installed unit is refused with `hold_unit_guard_missing`. A hold waits for
the trigger to stop and confirms it is inactive, publishes the guard record,
then stops and confirms again before disabling it. A crash before the record
exists leaves no acknowledged hold; a crash after publication leaves the guard
for reboot recovery. The hold reports success only after the trigger is inactive.
A renewed hold gets a new request id without shortening the deadline; an older
expiry timer cannot release it. Holds disable the unit until release, then
restore its earlier boot policy. Release writes a durable intent under
`requests/holds/releasing/`, removes the guard, starts the unit, and archives
the record under `requests/holds/history/`. The boot sweep completes any
interrupted release. Holdable timer and path units start after the installed
boot sweep, which waits for each stop to finish; a minute timer repeats the
sweep thereafter. Status
shows `remaining_seconds` and flags an overdue record, so a failed expiry is
visible for an operator to release. Safety-critical teardown, spend-guard,
capacity, storage-GC, replay-cache-GC and preflight triggers cannot be held.

### Founder stop of paid agent dispatch

The dispatcher-only `--until-released` option records
`require_explicit_release: true`. Its bounded `--for` deadline requests review;
expiry and the boot sweep keep the dispatcher disabled and stopped. Only an
explicit `release-hold`, after the founder authorizes resumption, removes this
guard. Ordinary operational holds retain automatic expiry. An overdue explicit
stop retains its owner and cannot be replaced by an automatic-expiry renewal.

```bash
python3 scripts/operator_door.py hold blueprint-agent-run-dispatcher.timer \
  --owner <current-hold-owner> --reason "Founder stop: explicit resumption required" \
  --for 24h --until-released --wait
```

Upgrade the door to the reviewed commit first and verify its terminal upgrade
receipt before requesting this policy. Older installed code does not enforce it.
Then verify `/status` reports `require_explicit_release: true` and the dispatcher
timer remains inactive and disabled. This stops agent-run dispatch only; it
does not claim every host controller or already running provider job is stopped.

Lanes that deploy by hand should keep naming their transient units
`blueprint-<label>-deploy-<sha>` so the door's in-progress check and `status`
see them.

## Install, tokens, upgrade, removal

Install from a commit on `main`. The repository is private and, until the
door's own deploy key exists, the host has no GitHub credential, so the first
install ships a bundle from the operator's machine:

```bash
# on the operator's machine
git fetch origin main && git bundle create /tmp/operator-door.bundle origin/main
scp /tmp/operator-door.bundle root@<host>:/tmp/
# as root on the host
git init --quiet /tmp/operator-door-install
git -C /tmp/operator-door-install fetch --quiet /tmp/operator-door.bundle 'refs/remotes/origin/main:refs/remotes/origin/main'
git -C /tmp/operator-door-install checkout --quiet --detach <commit>
bash /tmp/operator-door-install/deploy/operator-door/install.sh
rm -rf /tmp/operator-door-install /tmp/operator-door.bundle
```

The installer stages and import-checks the code, backs up the current unit
files, swaps the code into place (keeping `.previous`), creates the
`blueprint-door` group, the state directories and an empty token store (never
replacing an existing one), installs and starts the units, and health-checks
the door: `/healthz` must answer and `self-test` must pass as the `blueprint`
user, which proves the door can read its token file. Any failure after the swap
restores the previous code and units and restarts the previous door, or
disables a first install. It then patches `/etc/caddy/Caddyfile` in place (the
live file names the host literally and differs from the repository copy),
validates it, reloads Caddy and confirms the route through Caddy's admin API,
restoring the backup on any failure.

The installer also creates the door's GitHub deploy key,
`/etc/blueprint-operator-door/deploy-key/github`: generated once and never
replaced, root-only, in a directory the door process cannot enter. It pins
GitHub's SSH host keys from `api.github.com/meta` beside it, and prints the
public half. Register that once, read-only:

```bash
gh repo deploy-key add <public key file> --repo ognjhunt/BlueprintCapturePipeline \
  --title "operator-door <host> (read-only)"
```

Door deploys and upgrades fetch with it. The source clone's `core.sshCommand`
names the key, and a `url.git@github.com:.insteadOf` rewrite keeps origin's
HTTPS URL, so the deploy tool's own `git fetch origin main` works too. Without
the key a deploy is refused with `github_deploy_key_missing`. Unregistered, it
fails with `fetch_failed`. Revoke it under the repository's Settings → Deploy
keys.

Issue a token on the operator's own machine, so the plaintext never reaches the
host or any transcript:

```bash
umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.blueprint-secrets/operator_door_token
python3 -c 'import hashlib,pathlib; t=pathlib.Path("~/.blueprint-secrets/operator_door_token").expanduser().read_text().strip(); print("sha256:"+hashlib.sha256(t.encode()).hexdigest())'
```

Register only the hash on the host:

```bash
PYTHONPATH=/opt/blueprint/operator-door python3 -m operator_door token add \
  --name cloud-scene-runs --sha256 sha256:<hex> --scopes read,operate,deploy
```

Put the plaintext into the cloud environment's API credential for the host
(header `Authorization`, prefix `Bearer`), for example with
`pbcopy < ~/.blueprint-secrets/operator_door_token`. Revoke with
`python3 -m operator_door token revoke --name <name>`; the change is picked up
on the next request.

Upgrade from a session with the `deploy` scope:
`python3 scripts/operator_door.py upgrade-door <main commit> --wait`, or rerun
`install.sh --upgrade` as root.

An upgrade from recorded baseline `d78ee479368c2df99370a9dc4a61dd328d07d215`
automatically preserves that door's `deploy`, `unit`, and `door-upgrade`
request set and adds only `hold`/`release-hold` for
`blueprint-agent-run-dispatcher.timer`. A host operator can explicitly select
the same profile with `install.sh --upgrade --dispatcher-hold-only`.
This profile refuses missing existing credentials before staging, preserves
credential bytes/permissions and Caddy, and skips unrelated owner-store
provisioning. It persists across subsequent upgrades. `/status` reports
`door.request_kinds` and `door.hold_target` so operators can verify the fence.
An ordinary full code upgrade from the old baseline would otherwise admit
additional privileged operations under existing `operate` scopes.

Remove: `systemctl disable --now
blueprint-operator-door.service blueprint-operator-door-runner.path`, delete the
`handle /api/live-pipeline/operator/*` block from the live Caddyfile and reload
Caddy, then remove `/opt/blueprint/operator-door*`,
`/var/lib/blueprint-operator-door` and `/etc/blueprint-operator-door`, and delete
the deploy key from the repository's Settings → Deploy keys.

## Using it

```bash
python3 scripts/operator_door.py whoami
python3 scripts/operator_door.py status
python3 scripts/operator_door.py usage
python3 scripts/operator_door.py ls /var/lib/blueprint/pipeline-control-plane/deploy-receipts --sort mtime
python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/storage-gc/summary.json
python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/storage-gc/latest.json
python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents/<intent>/progression.json
python3 scripts/operator_door.py pull /var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/<launch> ./launch
python3 scripts/operator_door.py journal blueprint-task-evaluation-scene-progression.service -n 200 --since -1h
python3 scripts/operator_door.py unit start blueprint-task-evaluation-scene-progression.service
python3 scripts/operator_door.py deploy <sha on main> --wait
python3 scripts/operator_door.py retire-scene-workspace <scene_id> --wait           # plan only
python3 scripts/operator_door.py retire-scene-workspace <scene_id> --apply --wait   # retire behind a receipt
python3 scripts/operator_door.py restore-scene-workspace <scene_id> --bucket <bucket> --wait
python3 scripts/operator_door.py unit start blueprint-owner-approved-deletion-plan.service  # plan the deployed owner-approved deletions
python3 scripts/operator_door.py unit start blueprint-owner-approved-deletion.service       # delete what that plan found eligible
python3 scripts/operator_door.py provider-output-resume <run> <attempt> --ingest --wait
python3 scripts/operator_door.py restore-g1-runtime-source --wait             # plan only
python3 scripts/operator_door.py restore-g1-runtime-source --apply --wait     # restore the G1 runtime source
python3 scripts/operator_door.py extend-scene-execution-window <intent_id> --intent-digest sha256:<hex> \
    --expires-at <UTC instant> --authorization-reference "<owner approval>" --wait                    # plan only
python3 scripts/operator_door.py extend-scene-execution-window <intent_id> --intent-digest sha256:<hex> \
    --expires-at <UTC instant> --authorization-reference "<owner approval>" \
    --apply --ack extend-scene-execution-window --wait                                             # record + tell WebApp
python3 scripts/operator_door.py confirm-owner-success-contract <intent_id> --owner-user-id <uid> \
    --owner-organization-id <org> --approved-by <uid> --approval-evidence "<owner approval>" --wait   # preview only
python3 scripts/operator_door.py confirm-owner-success-contract <intent_id> --owner-user-id <uid> \
    --owner-organization-id <org> --approved-by <uid> --approval-evidence "<owner approval>" \
    --apply --apply-ack sha256:<preview apply_ack> --ack confirm-owner-success-contract --wait     # record
```

### Owner confirmation of a website job's success thresholds

A website job is built with Blueprint's fixed development controls as its
success thresholds, marked `proposal_only` (`website_native_submission`), so
the team packet refuses it with
`team_scene_owner_contract_invalid:configured_owner_success_contract_authority_missing`
and screening refuses an unconfirmed owner contract. Once the job is published,
the owner can accept those exact thresholds: run the preview, show the owner
the `preview.configured_success_criteria` it returns, and apply with its
`apply_ack` and the owner's verbatim approval as `--approval-evidence`. The
confirmation changes no threshold.

The team packet worker reads the record from the request's own intent
(`team_scene_packet_worker.owner_success_confirmation_scope`: the intent must be
the request's, by id and digest, and its progression must have queued that
exact request). It recomputes the binding (revision digest, success document
reference, thresholds digest, proposal digest, owner) from what it compiled,
and only an exact match makes the owner contract `task_owner`/`confirmed` with
author `owner-success-confirmation:<confirmation_digest>`; anything else refuses
as before. A request blocked on an owner-contract refusal returns to `pending/`
once per new confirmation of its intent, on the worker's own pass or the
screening timer's (`requeue_transient_blocked`); to run it now,
`python3 scripts/operator_door.py unit start blueprint-team-scene-packet.service`.
Screening then reads the contract the team packet sealed.

The owner can also confirm on the website job page, with no operator
(`website_owner_success_confirmation`). On the screening timer's pass,
`requeue_transient_blocked` sends each request blocked only on that refusal
(at most four a pass, rotating) to the WebApp over the signed Pipeline sync
transport: `POST /api/internal/pipeline/task-evaluation-scene-intents/<intent>/owner-success-proposal`
carries the published thresholds, their revision and a `proposal_digest`
sealing the whole binding. The WebApp shows them to the job's owner link only,
in plain words, with Confirm and "this is wrong" (a typed reason, no numbers),
and answers with the owner's decision on exactly that proposal. A confirmation
is written by the door's own writer and validators, with `issuer_channel`
`website`, and the same pass returns the request to `pending/`. A "wrong"
answer writes nothing under the intent and leaves the request blocked; its
status keeps the reason under `website_owner_success`, as it keeps `pending`
or a `failed` exchange. Each request is asked again only at its kept
`next_check_at`, or at once when its intent's progression changes and builds a
new proposal; the wait doubles while the answer repeats, from 2 min (pending,
up to 1 h), 30 min (held, up to 1 day), 15 min (a WebApp 4xx, up to 1 day) or
2 min (a transport failure, up to 6 h). Any error in one job's exchange is that
job's `failed` exchange and never stops the screening pass. A republished job
has a new proposal digest and needs a new decision. The door path above is
unchanged. The WebApp alerts ops (Slack, then ops email) when an owner holds a
job's thresholds.

Every team-packet status names the release that wrote it (`release_commit`,
the commit of the release checkout the worker or timer runs from). Any other
deterministic refusal (neither transient nor owner-contract) returns to
`pending/` once per newly deployed release, on the same two passes, so a deploy
that fixes its cause re-drives it with no operator step; a status without
`release_commit` counts as an older release. The status keeps the releases
already tried under `release_requeued` (written before the move), so a refusal
the new release does not fix stays blocked until the next deploy. The pass
reports it as `{"reason": "new_release", ...}`. When the running release cannot
be read, nothing is requeued on that basis. Completed and skipped requests never
move, except as follows.

A completed rigid half whose recorded carry search found no stance the Franka
carries from (`robot_station_search.cleared` false, or `reach_shortfall`
`out_of_reach`: capture-447ae4f4's eight stances, all `unreachable:approach`)
is compiled again once per stance-search version, on the same two passes and
the CPU, when its station was chosen by an older stance search than the running
release's (`team_embodiment_stance.STANCE_SEARCH_VERSION`, recorded in every
stance as `search.stance_search_version`; a station without one is version 1).
The status keeps the versions tried under `stance_search_version_requeued`
(written before the move); the pass reports `{"reason": "stance_search_version", ...}`.
A release that leaves the stance search unchanged recompiles nothing, and a
half the new search still cannot carry from is kept until the next version.

A provider-output resume unit may write only the canary dispatch root, the disk
reservation ledger and its result (`ProtectSystem=strict` and the retirement
unit's other sandboxing), for at most 2 h. It reads the control-plane
environment file as data, as the retirement does, and defaults the dedicated B2
store's settings to the dispatcher unit's.

A retirement that plans or retires exits 0; one the scene does not qualify for
(`retained`) exits 1 and prints the outcome with its first reason, for example
`acknowledgement_unproven:<capture>` or `open_scene_intent:<intent>`.

To replay a failed stage against candidate code, pull the child's retained job
and inputs (`pull` of the `sam31-preparation-executions/<state>/<child>` job and
the prepared-reference directory it names), then run
`python -m blueprint_pipeline.task_evaluation_stage_replay --child <id>` in the
cloud session with `--queue-root`, `--input-root`, `--replay-root` and
`--approved-root` pointing at the pulled copy. It runs your branch's code with
no host secrets and no paid calls, and nothing unreviewed runs on the host.

An Atlas preview job replays the same way before a fix is deployed: `pull` its
capture root's `pipeline/atlas/preview_job.json`, `configuration.json`,
`pipeline/website_task_context.json` and `raw/` video to the same absolute paths,
then run `python -m blueprint_pipeline.atlas_preview_replay --job <that job>`
(add `--configuration <snapshot>` for the current allowance). It advances the job
on a scratch copy until its first stop and reports what it would buy there; a
`missing_input` names the next file to pull, and `unpulled_before` lists records
the code looked for just before stopping, which a partial pull can turn into a
false purchase or refusal. The pulled files are never written.

### Replay the post-completion chain before deploy

A website scene's post-completion chain (completion join, robot-free readiness,
team scene packet, rigid probe enqueue, screening plan) replays the same way,
with `task_evaluation_downstream_replay`. Pull these to their own absolute paths,
because the launch receipt, its result and the registry name files by absolute
path (`<bucket>` is the capture partition under `pubsub-handoffs`):

```bash
L=/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/<launch>
I=/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents/<intent>
C=/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/<scene>/captures/<capture>
R=/etc/blueprint/team-evaluation-registry.json
door() { python3 scripts/operator_door.py pull "$1" "$1"; }
for f in launch_profile.json launch_receipt.json webapp_sync_succeeded.json \
         post_teardown_provider_zero_receipt.json allocator/result.json; do door "$L/$f"; done
# Present only once the probe or an earlier join ran; a refused pull just means it is absent.
for f in rigid_robot_free_probe_trace.v1.json robot_free_readiness.v1.json robot-free-readiness; do door "$L/$f"; done
# The published revision and the configured bundle the result names (the bundle can be large).
REV=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["configured_scene_revision_path"])' \
      "$L/allocator/result.json")
door "$REV"; door "$(dirname "$REV")/configured_scene_bundle.v1.zip"
door "$I"                                   # intent.json, attempts/, progression
for f in capture_descriptor.json pipeline/website_task_context.json pipeline/website_scene_sponsorship.json \
         pipeline/atlas/configuration.json pipeline/website_candidate_ranking.json; do door "$C/$f"; done
door "$R"
```

Pull only these: the replay copies and digests everything under each input, so
a whole launch run with its provider output costs minutes. Also pull, to their
own paths, the DROID profile and runtime-source receipt the registry's
`derived_profiles.packet_inputs` names, and for a scene whose probe has not passed
the packet that receipt names. The scored-matrix salt key is never pulled: the
replay uses a placeholder. Save the WebApp's current answers for the capture as
`webapp/task-context.json` and the non-creating `webapp/atlas-configuration.json`
(the one carrying `job_ceiling`); without them the replay answers from the
capture's retained copies and `webapp_answers` says so. Then, on the fix branch:

```bash
python -m blueprint_pipeline.task_evaluation_downstream_replay --launch-run "$L" --intent-dir "$I" \
    --capture-dir "$C" --registry "$R" --webapp-snapshots ./webapp --mirror ./mirror \
    --until screening_plan --replay-root ./replays
```

It runs this tree's join, readiness scorer, team compile, probe lane and
screening planner on scratch copies and prints `downstream_replay_report.v1.json`
(also written under `./replays`). `first_stop.kind` is `completed` (`--until`
reached), `would_buy` (the probe's paid allocator, named, never started),
`external_effect` (a WebApp write other than the recorded job offer, the network,
a subprocess or a write outside the replay root), `missing_input` (`code` is the
next file to pull, or a published object to add at `./mirror/<bucket>/<key>`;
objects the launch retained byte for byte are found without it), `refusal` (the
code's typed blocker, with `fired_predicates` and `unpulled_before`) or `error`.
`evidence.unchanged` confirms the pulled files' digests did not change; nothing
is sent, bought or written outside `./replays`. Exit codes: 0 completed,
would_buy or external_effect; 2 error; 3 refusal; 4 missing_input.

`usage` answers "what uses the disk?" in one call. It prints `capacity.usage` from
`status` as tables: each surveyed filesystem's used, surveyed and classified bytes
and the attributed fraction, then bytes by storage class, the top roots, the top
owners (`scene:`, `run:`, `release:`, `store:` or a root's child) and the 20
largest unclassified roots (fewer if the bounded summary must shrink). Bytes
are counted once per inode, so hardlinked stores are not double counted. The
survey is at most hourly; the header gives its age. See
[`CONTROL_PLANE_STORAGE.md`](CONTROL_PLANE_STORAGE.md#usage-attribution).

```text
usage survey: complete, 1200 s old

mount                used       surveyed   classified  attributed
/                    150.0 GiB  145.0 GiB  130.0 GiB   96.7%

owner                        class  allocated  root
store:prepared-references    cache  40.0 GiB   /var/lib/blueprint/task-evaluation-inputs/prepared-references
scene:site-capture-ae539f2c  work   12.0 GiB   /var/lib/blueprint/pubsub-handoffs
```

Exit codes: 0 success; 1 a waited-for request did not succeed, or `usage` found
no usage survey; 2 refused (rule on stderr); 3 unauthorized or missing scope; 4
network; 5 server error. The audit log is
`/var/lib/blueprint-operator-door/audit/audit.jsonl`.


### Checked retained-file pulls

Use paired `--expected-sha256 sha256:<64 lowercase hex digits>` and
`--expected-size <canonical decimal bytes>` on `pull` when retaining a census
artifact for owner annotations. Obtain both values from an independently
retained, trusted publication receipt for the privileged artifact. Do not derive
the expected identity from the download or a separate unanchored fetch of a
mutable `latest.json`. This client consumes that anchor; it does not publish or
authenticate it.

```bash
python3 scripts/operator_door.py pull <published-census-path> retained-census.json \
  --expected-sha256 sha256:<digest-from-trusted-receipt> \
  --expected-size <bytes-from-trusted-receipt>
```

Checked pull is file-only through the existing binary `/fs/read` route. It
requires exact range headers, truthful EOF and an incrementally checked raw
SHA-256 before atomic local publication. The policy bounds files at 16 MiB,
requests at 4096 and each requested chunk at 1 MiB; a smaller server cap remains
supported. Zero-byte files require one truthful empty EOF response. Unique
operation-owned temporary files prevent truncating an existing destination;
failures before publication preserve it. Checked local destinations require
parents without symlinks or traversal. Retained directory/file identities bind
temporary creation, publication and cleanup; observed substitutions refuse
verified success and preserve foreign entries. These checks do not provide an
atomic compare-and-rename against arbitrary concurrent writers, or rollback
after another writer changes the path during publication. If initial temporary
identity acquisition fails, the descriptor closes and the unproven empty entry
is retained rather than deleted. The existing per-request timeout
remains; no new whole-transfer deadline is claimed. Ordinary file and directory
pull behavior is unchanged.

Success adds only `verified_digest` and `verified_bytes` to the usual path/saved/
bytes output. Checked errors use small fixed codes without server text, URLs or
argument contents. HTTP error bodies are capped at 4 KiB plus one sentinel;
exit codes remain 2 for generic refusal/invalid transfer or local I/O, 3 for
unauthorized/missing scope, 4 for network failures and 5 for server failures.
Only a complete bounded JSON `scope_missing` error makes HTTP 403 an exit 3.

A verified file proves byte identity with the supplied anchor, not privileged
collection, complete inventory, fresh references, owner approval or deletion
authority. Obtain the reviewed census and publication identity separately, then
run the [retained annotation validator](runbooks/control-plane-capacity.md#review-retained-experiment-folder-decisions).
Collection, Plan12b validation and cleanup approval remain separate gates; this
client does not create a collector, new route/service or cleanup capability.
