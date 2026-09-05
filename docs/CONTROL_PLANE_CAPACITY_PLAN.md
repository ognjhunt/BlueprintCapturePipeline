# Control-plane capacity: from a floor to a plan

Status: plan of record, 2026-09-05. Owner: control-plane storage.

## The problem, measured

The production control-plane host (one DigitalOcean droplet, 154 GB root disk) filled
for the fourth time on 2026-09-05: 144 GB used, 9.7 GB free, and the disk-admission
floor (8 GiB plus 2 GiB per stage) refused every stage. The WebApp saw a `503`. That
`503` was the first capacity signal anyone received.

| What was on the disk | Size | Why it could not be reclaimed |
|---|---|---|
| 4 sealed Quick-10 canary run directories (one 4.2 GB provider archive plus its extracted contents each) | 31 GB | evidence offload disabled; 14-day hot window |
| 23 unsealed run directories from superseded or torn-down workers | part of 4.2 GB | no terminal receipt, retained forever as "active" |
| 20 release worktrees and 36 runtime trees | 6 GB | 34 `pending` queue rows bound to dead commits pinned 33 releases; 5 orphan trees blocked the whole retention apply |
| engineering, render probes, diagnostic checkouts, release builds, backfills | 12 GB | no storage class, so the reaper ignored them |
| caches (prepared references, compiled episodes) | 13 GB | referenced by the same dead rows |

A full end-to-end run leaves 12 to 15 GB behind, so the disk held about seven runs of
history. Every deploy multiplied inputs because bindings were keyed by commit.

## Target shape

| Today | Launch-ready |
|---|---|
| Control plane holds archives, caches, scratch and state on one root disk | Control plane holds state. Bulk bytes stream provider → object store (Backblaze B2, `blueprint-task-evaluation-artifacts-prod`), content-addressed. Local copies live on a work volume and are archived behind a pointer within 48 h |
| One 154 GB root disk | Small root for OS, releases and state. Resizable block volume bind-mounted under the bulk roots so unit sandboxes do not change |
| Per-commit copies of runtimes and sources | Content-identity bindings; deploy retires superseded releases as its last stage |
| Floor guard, `503` on breach | Measured every ten minutes, forecast, alerted at 70 % and 85 %, volume grown one step when critical |
| Offload disabled, 14-day window | Object store is the system of record; sealed runs archived after 48 h, abandoned runs after 72 h; `restore` reverses any migration |
| Pet host | Terraform, Caddyfile and installer in the repo; volume mount procedure in `deploy/host/` |

## What has landed

- **#1652** storage GC: stranded queue rows (pending rows bound to a superseded release move to `stranded/` beside a receipt), `scratch` storage class reaped by idle age, abandoned-run sealing in the offload lane, orphan release trees removable by retention, offload window and abandonment window wired through the unit, hourly timer.
- **#1653** installed publisher sources are bound by content identity; one installation serves every release.
- **#1655** production-chain preflight: every unit probed under its own sandbox before anything is submitted or paid for.
- **#1656** capacity controller: measure, history, forecast, alert, grow.

## Host procedure

1. Deploy a release at or after the merges above. After the deploy, start
   `blueprint-control-plane-storage-gc.service` once; the hourly timer takes over.
2. Opt in to offload in `/etc/blueprint/pipeline-control-plane.env`:
   `BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD=1` (done on the production host).
3. Attach a block volume, then `deploy/host/mount_work_volume.sh --device
   /dev/disk/by-id/<volume> --plan`, review, and `--apply --ack move-work-roots-to-volume`.
   All bulk roots move in one rsync (hardlinks preserved) and come back as bind mounts
   recorded in `/etc/fstab`.
4. Point the capacity controller at the volume in the environment file:
   `BLUEPRINT_CAPACITY_MOUNTS=/var/lib/blueprint:/mnt/blueprint-work`,
   `BLUEPRINT_CAPACITY_VOLUME_ID`, `_MOUNT`, `_DEVICE`, `_MAX_GIB`, and
   `BLUEPRINT_CAPACITY_AUTORESIZE_ACK=grow-control-plane-volume` once growth is wanted.
5. Set `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` so capacity alerts reach an operator.

## Capacity statement after the procedure

Disk per run on the root disk is flat at megabytes of state. Bulk bytes are bounded to
about 48 hours of runs on the volume, which grows on demand up to the configured
maximum. Concurrency is then governed by paid-launch lock slots and provider
availability, which is where the limit belongs.

## Remaining work before launch

- **Stream, do not download.** Every lane adapter still downloads
  `vast_provider_runtime_output.zip` (4.2 GB for a Quick-10) to the host before
  sealing. The lanes should seal from the object store manifest and fetch only
  what interpretation needs. Sixteen adapters share the pattern; the offload lane
  bounds the exposure to 48 hours in the meantime.
- **Queue with an ETA instead of a `503`.** Intake refuses when a stage cannot be
  admitted; the WebApp contract should accept a queued state with the capacity
  controller's forecast.
- **Content-addressed runtime trees.** Deploy still publishes a runtime tree per commit
  (hardlinked); keying by recipe digest removes the copy entirely.
- **One-command rebuild drill.** Terraform, Caddyfile, installer and the volume
  procedure exist; the drill that proves a fresh host comes up from them has not been run.
