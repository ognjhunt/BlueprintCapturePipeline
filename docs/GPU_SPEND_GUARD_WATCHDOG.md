# GPU spend guard: scheduled reaper + pod-side watchdog (R055 / R056)

This closes two GPU-cost gaps left after the aggregate spend/GPU-ceiling work
(R041, `src/blueprint_pipeline/fleet_spend_ledger.py`). Those enforce budgets
*before* launch; the items here reclaim spend *after* something has already gone
wrong (a stuck allocation, or a launching host that died mid-run).

## R055 — the guard is now scheduled and enforced

`scripts/gpu_spend_guard.py` was a manual, dry-run-by-default tool. It is now run
on a timer as a systemd oneshot, in `--reap` mode, writing a durable JSON
snapshot as teardown evidence.

Units (mirror the control-plane unit family under `deploy/systemd/`):

- `blueprint-gpu-spend-guard.service` — `Type=oneshot`, reuses
  `EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env` and the file-based
  `~/.blueprint-secrets` convention (no new secrets). `ExecStart` runs
  `gpu_spend_guard.py --reap --max-boot-seconds <n> --orphan-booted-max-age-seconds <n>
  --json-report <path>`; `ExecStartPost` runs the post-check.
- `blueprint-gpu-spend-guard.timer` — `OnBootSec=3min`, `OnUnitActiveSec=5min`,
  `Persistent=true`. A short cadence so runaway cost is caught within minutes.
- `blueprint-gpu-spend-guard-postchecks.sh` — fails the unit unless a fresh,
  well-formed `gpu_spend_guard.v1` snapshot produced in `--reap` mode exists, so a
  silently-broken watchdog surfaces instead of leaving cost unwatched.

Install/enable (via the existing installer):

```bash
sudo scripts/install_live_pipeline_control_plane.sh --enable-now
# or, later:
sudo systemctl enable --now blueprint-gpu-spend-guard.timer
```

Marking it *deployed* is an ops step: the unit files + installer wiring + env
knobs are implemented here; enabling the timer on the droplet is the remaining
action.

### Snapshot (durable teardown evidence)

Written to `BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_PATH`
(`/var/lib/blueprint/gpu-spend-guard/gpu_spend_guard_snapshot.json`). Schema
`gpu_spend_guard.v1`: every live allocation, the burn estimate, protected vs.
reap-candidate ids, and — when reaping ran — per-instance termination results.
New fields: `orphan_booted_max_age_seconds`, `booted_orphan_reaping_enabled`,
`credentials_available`. The snapshot is written even with no credentials
configured, so the post-check can tell "ran, nothing to do" from "never ran".

## R056 — reap booted orphans + a pod-side self-terminating watchdog

### (a) Reap booted orphans (`gpu_spend_guard.py`)

Previously only never-booted boot-timeout duds were reaped; a *booted* pod whose
launching host died billed forever. Now, with
`--orphan-booted-max-age-seconds` (env `BLUEPRINT_GPU_ORPHAN_BOOTED_MAX_AGE_SECONDS`,
`0` = disabled, recommended `21600` = 6h), `is_reapable` also reaps a **booted**
pod that is orphaned and older than the hard age ceiling.

Safety guards (fail-safe toward *keep*):

- Warm-candidate ids (`DEFAULT_WARM_CANDIDATE_IDS`) are never reaped.
- Live warm-serve pods are protected via their `warm_serve_pod.json` marker
  (`status == "serving"`), unioned into `protected_ids` — never reaped.
- A pod with a live owning process (`started_pod_id.txt` etc. referenced by a
  running cmdline) is never reaped.
- Only a booted pod that is unowned **and** past the hard ceiling is reaped;
  freshly-booted pods (under the ceiling) are kept.

The 6h ceiling is far beyond any healthy render/eval window, so only a genuinely
leaked booted pod trips it.

### (b) Pod-side hard-TTL self-kill (`isaac_particlefield_render_job.py`)

The host-side watch loop (`max_seconds`) and stall watchdog (post-marker
no-progress, 900s) can only tear a pod down while the host process is alive. If
the host itself dies, a booted render pod would bill until reaped. The render
bootstrap (`docker_start_cmd` / `BOOTSTRAP`) now bakes an env-gated hard-TTL
self-kill thread that `os._exit`s the container when
`BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS` elapses — mirroring the
eval worker's TTL env pattern.

- Set by `_env_for` / `build_render_launch_spec` with a sane default
  (`DEFAULT_EXTERNAL_WATCHDOG_TTL_SECONDS = 7200` = 2h), which is far above the
  host watch/stall windows, so a healthy monitored render is never self-killed —
  only a host-orphaned pod is.
- Additive/opt-out-safe: `watchdog_ttl_seconds=0` omits the env, disabling the
  self-kill; existing render launches/tests are unaffected.

## Tests

- `tests/test_gpu_spend_guard.py` — booted-orphan reap matrix (old+orphaned
  reaped; within-ceiling, warm-serve-protected, warm-candidate, fresh-owner, and
  unknown-age all kept), main() end-to-end reaping only the leaked booted orphan,
  env-driven enablement, and the no-credentials snapshot.
- `tests/test_isaac_particlefield_render_job.py` — the bootstrap carries the TTL
  self-kill and stays valid Python; `_env_for` / `build_render_launch_spec` set
  the TTL env by default and omit it when disabled.
- `tests/test_gpu_spend_guard_deploy.py` — the systemd unit files parse and carry
  the required sections/keys, the installer wires them, and the post-check
  passes/fails correctly (fresh reap snapshot vs. missing/dry-run/wrong-schema/
  stale).
