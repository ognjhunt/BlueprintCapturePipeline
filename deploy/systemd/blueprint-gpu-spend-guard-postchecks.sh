#!/usr/bin/env bash
# Post-check for the scheduled GPU spend guard (R055): fail the unit unless the
# --reap pass persisted a fresh, well-formed gpu_spend_guard.v1 snapshot, so a
# silently-broken watchdog surfaces instead of leaving cost unwatched.
set -u

repo="${BLUEPRINT_PIPELINE_REPO:-/opt/blueprint/BlueprintCapturePipeline}"
snapshot="${BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_PATH:-/var/lib/blueprint/gpu-spend-guard/gpu_spend_guard_snapshot.json}"
# Snapshot older than this many seconds is stale (default 2x the 5min timer).
max_age="${BLUEPRINT_GPU_SPEND_GUARD_SNAPSHOT_MAX_AGE_SECONDS:-900}"

cd "${repo}" || exit 1

if [ -x .venv/bin/python ]; then
  py=(.venv/bin/python)
else
  py=(python3)
fi

if [ ! -f "${snapshot}" ]; then
  echo "gpu-spend-guard: snapshot not written: ${snapshot}" >&2
  exit 1
fi

"${py[@]}" - "${snapshot}" "${max_age}" <<'PYCHECK'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
max_age = float(sys.argv[2])
try:
    snap = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001
    print(f"gpu-spend-guard: snapshot not valid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)
if snap.get("schema_version") != "gpu_spend_guard.v1":
    print(
        f"gpu-spend-guard: unexpected schema_version {snap.get('schema_version')!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
if snap.get("reap_mode") is not True:
    print("gpu-spend-guard: snapshot not produced in --reap mode", file=sys.stderr)
    raise SystemExit(1)
age = time.time() - path.stat().st_mtime
if age > max_age:
    print(
        f"gpu-spend-guard: snapshot stale ({int(age)}s > {int(max_age)}s)",
        file=sys.stderr,
    )
    raise SystemExit(1)
reaped = len(snap.get("reap_results") or [])
print(
    "gpu-spend-guard: snapshot ok "
    f"(live={snap.get('live_instance_count')}, "
    f"burn=${snap.get('total_burn_per_hour_usd')}/hr, reaped={reaped}, "
    f"booted_orphan_reaping={snap.get('booted_orphan_reaping_enabled')})"
)
PYCHECK
