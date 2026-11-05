# Control-plane host profile and sizing, 2026-10-31

Host `paperclip-prod-01` is a DigitalOcean droplet with 4 vCPUs. The root disk is 165 GB and holds `/` and `/var/lib/blueprint`. The growable volume is 316 GB at `/mnt/blueprint-work`, and its subtrees are bind-mounted at `/var/lib/blueprint/<rel>`. Every number below came from read-only operator-door calls (`status`, `usage`, `units`, `show`, `ls`, `journal`) between 23:00 and 23:30 UTC. Per-unit CPU is systemd's own `Consumed … CPU time` line from each unit's last 2,000 journal lines, so each unit's window is between 1 and 24 hours long.

## Where the CPU goes

At sample time the load average was 5.4–6.0, against about 8 earlier in the day. The systemd-accounted CPU of the periodic units adds up to about **3.3 cores on average** on a 4-vCPU host. The rest of the load is runnable and IO-waiting tasks.

| unit | avg cores | CPU/run | cadence | note |
|---|---:|---:|---|---|
| gpu-spend-guard | 0.42 | ~104 s | 1 min after exit (~4 min) | 8 full `**` globs of `pipeline-control-plane` per run, `CPUWeight=1000` (fixed in PR #3205) |
| agent-stage-replay | 0.42 | ~9 s | **5 s** after exit | 500 runs in 3 h. It polls an idle replay queue |
| task-evaluation-launch-activation | 0.33 | ~25 min | per activation | ~1 h wall, bundle staging and zipping (bundle reuse is handled in another session) |
| provider-billing-reconciler | 0.28 | ~33 s | 10 min | 1.4 GB peak |
| pubsub-handoff-listener | 0.26 | ~44 s | 1 min after exit | each run acknowledges 0 messages. The `ExecStartPre` runtime-env guard takes ~52 s |
| episode-compilation-remote | 0.20 | ~23 s | 60 s + path | |
| task-evaluation-scene-progression | 0.19 | 25 s normal, 200–336 s on succession | 1 min after exit | succession passes took 756–1,698 s wall (fixed in PR #3205) |
| pipeline-control-plane | 0.14 | ~44 s | 5 min | `ExecStartPre` runtime-env guard takes ~57 s |
| scene-project-spend-refresh | 0.11 | ~34 s | 5 min | |

### Wasted periodic work

1. **The runtime-env guard runs before every one-minute Pub/Sub poll.** `production-runtime-env-guard` ran 483 times across three units, for **7.2 hours of wall time in about 15 hours**. Its slow steps are `spend_authority_ledger` (20–40 s) and `launch_profile_catalog` (22–36 s). Only the import verdict is reused between runs. Proposed fix: key the ledger and catalog reconciliation on a digest of their inputs, and reuse the previous verdict while that digest is unchanged, as the import verdict already does. The guard protects money, so this change needs its own PR and an owner review. It is not in #3205.
2. **`blueprint-agent-stage-replay.timer` fires 5 s after the unit exits.** That is 0.42 cores spent polling an idle queue. Proposed fix: trigger it with a `.path` unit on the replay queue, plus a 5-minute fallback timer.
3. **Spend-guard and succession tree walks.** Fixed in PR #3205: one walk per root, and unchanged directories are not re-listed.
4. **Progression slots go to dead intents.** In the slowest pass (1,698 s), 13 of the 16 `maximum_intents_per_pass` slots went to intents blocked on `scene_intake_authority_revoked` or `…_expired`. The cursor rotates, so all 41 intents are reached eventually, but live intents wait about 3 passes. Each pass still runs the terminal-result join for the dead intents. Proposed fix: do not count an intent against the pass budget when it is authority-blocked and has no pending terminal join. The authority stop itself must stay (it is the rights hard stop).

## Disk

| mount | used | free |
|---|---:|---:|
| `/` and `/var/lib/blueprint` (same filesystem) | 88.8 % | 18.5 GB |
| `/mnt/blueprint-work` | 81–93 % during the evening | 21–59 GB; it dropped ~16 GB/h during staging and recovered afterwards |

**Why the survey attributes about 0 % of `/mnt/blueprint-work`.** `survey_usage` visits only `partitions_per_mount=4` top-level partitions per hourly pass, each with an equal share of a fixed entry and time budget, and each pass replaces the last one ("partial snapshots, never cumulative"). The large partitions are `task-evaluation-inputs` (642 children), `pipeline-control-plane` (capture-reconstruction runs, native-G1 campaign work), the `g1-841757-*` runs and checkpoint cache, deploy-source archives and `embodiment-preparation`. They exhaust their share and are truncated. The report then attributes only what the 4 visited partitions held, which tonight was mostly `gcp-billing-reconciliation-20261024` (196 MiB). Proposed fix: keep each partition's last *complete* measurement with its age, and report the attributed fraction as the sum of those, not as the current pass alone. Let a partition that is too large continue by cursor at the next level down, instead of restarting.

**Why reclaim is ineffective** (from the storage-GC and capacity journals):
- `result_registry` keeps **17.1 GB in 29 runs** marked `active_or_unsealed`. The residue offload keeps 16 runs because of `dispatch_receipt_missing` (Oct 26 operator-policy and paired-policy runs), and 13 runs are already offloaded. Runs that never got a dispatch receipt will never be sealed, so they are held forever. Proposed fix: reclassify a run as historical once no queue envelope names it and it has had no write for N days. That makes it an offload candidate; it does not delete it.
- `sam31-preparations` is a plan-only GC root (dry run). It is never reclaimed by design.
- GC derived roots cover only `prepared-references`, `compiled-episodes`, `launch-activations`, `profile-install-staging` and `policy-canary-presubmission`. **No GC root covers** `g1-841757-blueprint-runs`, `g1-841757-checkpoint-cache`, `archived-deploy-sources-20260926`, `control-plane-retired-source-archives`, `control-plane-deploy-sources` or the `embodiment-preparation/public-robot-imports-*` batches. Neither do the 1,100 per-commit state directories under `pipeline-control-plane`, though those are small (3 files each).
- On the root disk, `cold-engineering-relocated` (1.7 GB) and `retained-scene-render-job` are unclassified.

Deleting any retained data is an owner decision. The candidates for the owner to review, from largest expected size down:
1. the September G1 841757 runs and checkpoint cache;
2. the deploy-source archives;
3. the public-robot-import batches that have been superseded;
4. the 16 receipt-less policy runs, offloaded to cold storage rather than deleted.

## Sizing recommendation

Demand today: about 3.3 cores of accounted CPU, plus bursts from launch activation (`CPUQuota=200%` for about an hour) and succession passes. The peak resident memory of a single unit is 4.2 GB (activation and embodiment preparation). Progression reached 3.2 GB and the billing reconciler 1.4 GB, and these can overlap at about 10 GB.

After PR #3205 and fixes 1 and 2 above, I expect about 1.2–1.5 cores of the 3.3 to go away:
- spend guard: about 0.35;
- stage replay: about 0.4;
- env guard: most of 0.4.

That would bring average demand to about 2 cores. On the current 4 vCPUs, that is enough for steady state, but not for an activation running alongside a succession.

| option | spec | approx. list price* | effect |
|---|---|---:|---|
| A. stay, fix only | current 4 vCPU (shared), volume as is | about $0 extra | load drops to about 2 cores. Activation and succession still contend |
| **B. recommended** | **CPU-Optimized (dedicated) 8 vCPU / 16 GB**, volume grown 316 → 500 GB | about $165–190/mo droplet + about $50/mo volume | no steal time. Activation's 200 % quota no longer starves the timers. 16 GB covers about 10 GB of overlapping peaks with page cache to spare. Succession and spend-guard walks hit warm dentry and inode caches |
| C. headroom | General Purpose (dedicated) 8 vCPU / 32 GB, 500 GB volume | about $250–300/mo + about $50/mo | as B, plus room for concurrent activations and a much larger page cache for the 270 GB working set |

\*DigitalOcean list prices from memory, not checked tonight. Confirm them before any decision. Infrastructure spend is the owner's decision. It is not covered by the per-site budget or by the USD 100 Google Cloud ceiling.

Dedicated rather than shared: the workload is a steady mix of CPU and IO from about 30 timers, plus hour-long CPU bursts. On a shared plan, steal time adds to the load average that is already above the core count. Grow the disk only after the retention fixes: today about 17 GB is held only for want of a seal, and the G1 and archive trees are outside every GC root. More disk alone would just postpone the same alert.

Order: deploy #3205 → fix the env-guard and stage-replay cadence → fix the retention rules and get the owner's review of the deletion candidates → re-measure → then resize if still needed. B is the size I would move to if the owner wants headroom tonight, independent of the fixes.
