# Operator door

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
| `GET /status` | read | deployed commit and blockers (loopback `/version`), active release, deploy units in flight and recent receipts, paid-launch lock holders from `/proc/locks`, spend guard, failed and controller units, disk, load, door queue |
| `GET /fs/list?path=&sort=name\|mtime&match=` | read | directory entries (capped) or a file's metadata |
| `GET /fs/read?path=&offset=&length=` | read | bytes; `X-Door-Size`, `X-Door-Offset`, `X-Door-Eof` headers for paging |
| `GET /fs/archive?path=` | read | tar.gz of a directory (512 MiB cap, two at a time) |
| `GET /journal?unit=&lines=&since=` | read | `journalctl -u` text, redacted |
| `GET /units?pattern=&state=` · `GET /units/show?unit=a,b` | read | `blueprint-*` unit rows or properties (never `Environment`) |
| `POST /requests` | per kind | `202 {"id": …}` |
| `GET /requests` · `GET /requests/<id>` | read | queue, or one request with result, outcome, redacted log tail and live unit state |

Read roots: `/var/lib/blueprint`, `/opt/blueprint`, `/workspace`,
`/mnt/blueprint-work`, `/etc/blueprint` (names screened as above) and the
door's own state. Paths resolve to their real location and must stay inside a
root and outside every hidden path.

Request kinds:

| Kind | Scope | Body | What the runner does |
|---|---|---|---|
| `deploy` | deploy | `commit` on `origin/main`, `wait_for_idle` | refuses while any `blueprint-*deploy*` unit is active; otherwise `door-deploy.sh` |
| `unit` | operate | `unit` (`blueprint-*`), `action` `start`\|`reset-failed`\|`stop`\|`restart` | `systemctl --no-block <action> -- <unit>`; `stop`/`restart` only for `.timer`/`.path`, never the door's own units or a spend-guard, watchdog, teardown or reaper trigger |
| `door-upgrade` | deploy | `commit` on `origin/main` | `door-upgrade.sh`: that commit's `install.sh --upgrade`, rolled back on a failed health check |

**`deploy` is root-equivalent.** Whoever holds it can get code that is on
`origin/main` running as root, which is what a deploy is. Give it only to
tokens that should be able to ship; `read` and `operate` never run code.

### What a door deploy does

`door-deploy.sh` repeats the deploy the agent lanes ran by hand over SSH:

1. optionally wait (up to 30 minutes) until scene progression and
   configured-controls progression are idle;
2. fetch every branch into one long-lived source clone,
   `/opt/blueprint/control-plane-config-tools/operator-door-source` (origin is
   GitHub, so the deploy tool's pushed-commit checks see real refs);
3. require the commit to be an ancestor of `origin/main`;
4. run the **target commit's** `scripts/deploy_control_plane_commit.py` from a
   throwaway worktree with `--iteration --preserve-configured-controls-state`,
   receipt `deploy-receipts/iteration_<sha12>_door.json`.

`--preserve-configured-controls-state` keeps the owner's configured-controls
pause across the deploy; `deploy_control_plane_iteration.sh` and
`deploy_control_plane_canary.sh` do not pass it, which is why the door does
not call them. The deploy tool's own guards still apply (paid-launch locks,
dirty or unpushed sources, surface agreement, disk budget). The signed hotfix
overlay path of the iteration wrapper is not offered through the door.

Lanes that deploy by hand should keep naming their transient units
`blueprint-<label>-deploy-<sha>` so the door's in-progress check and `status`
see them.

## Install, tokens, upgrade, removal

Install from a pushed commit, as root on the host:

```bash
git clone --quiet https://github.com/ognjhunt/BlueprintCapturePipeline.git /tmp/operator-door-install
git -C /tmp/operator-door-install checkout --quiet <commit>
bash /tmp/operator-door-install/deploy/operator-door/install.sh
rm -rf /tmp/operator-door-install
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
`install.sh --upgrade` as root. Remove: `systemctl disable --now
blueprint-operator-door.service blueprint-operator-door-runner.path`, delete the
`handle /api/live-pipeline/operator/*` block from the live Caddyfile and reload
Caddy, then remove `/opt/blueprint/operator-door*`,
`/var/lib/blueprint-operator-door` and `/etc/blueprint-operator-door`.

## Using it

```bash
python3 scripts/operator_door.py whoami
python3 scripts/operator_door.py status
python3 scripts/operator_door.py ls /var/lib/blueprint/pipeline-control-plane/deploy-receipts --sort mtime
python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents/<intent>/progression.json
python3 scripts/operator_door.py pull /var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/<launch> ./launch
python3 scripts/operator_door.py journal blueprint-task-evaluation-scene-progression.service -n 200 --since -1h
python3 scripts/operator_door.py unit start blueprint-task-evaluation-scene-progression.service
python3 scripts/operator_door.py deploy <sha on main> --wait
```

To replay a failed stage against candidate code, pull the child's retained job
and inputs (`pull` of the `sam31-preparation-executions/<state>/<child>` job and
the prepared-reference directory it names), then run
`python -m blueprint_pipeline.task_evaluation_stage_replay --child <id>` in the
cloud session with `--queue-root`, `--input-root`, `--replay-root` and
`--approved-root` pointing at the pulled copy. It runs your branch's code with
no host secrets and no paid calls, and nothing unreviewed runs on the host.

Exit codes: 0 success; 1 a waited-for request did not succeed; 2 refused (rule
on stderr); 3 unauthorized or missing scope; 4 network; 5 server error. The
audit log is `/var/lib/blueprint-operator-door/audit/audit.jsonl`.
