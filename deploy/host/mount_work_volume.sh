#!/usr/bin/env bash
# Move the control plane's bulk roots onto a resizable block volume and bind-mount
# them back at their original paths.
#
# Why: the control-plane root disk filled because run evidence, caches, scratch,
# scene workspaces and the handoff spool shared one disk with the host's state.
# Bulk bytes belong on a volume that can grow online; state, queues and ledgers
# stay on the root disk so a cache flood can never starve them.  Bind mounts keep
# every recorded path and every unit's ReadWritePaths valid, so nothing else
# changes.
#
# Plan by default.  --apply requires the acknowledgement and root, stops the worker
# units for the duration of the copy (one rsync per base directory, so hardlinks
# between roots are preserved), verifies the copy, swaps each root for a bind
# mount recorded in /etc/fstab, and only then removes the originals.
set -euo pipefail

ACK_REQUIRED="move-work-roots-to-volume"
MOUNT_DEFAULT="/mnt/blueprint-work"
STATE_ROOT_DEFAULT="/var/lib/blueprint"

# Bulk roots, relative to --state-root: every cache, evidence_cold and scratch
# root, the handoff spool and native run work.  Never queues, ledgers, spend
# guard, intents or the control-plane manifest.  tests/test_mount_work_volume_script.py
# ties this list to control_plane_storage_roots.STORAGE_ROOTS.
#
# task-evaluation-inputs is ONE root on purpose.  Its stores hardlink into each
# other (prepared references into compiled episodes, runtime prerequisites into
# runtime trees), and link(2) fails with EXDEV across mount points even on one
# filesystem, so a bind per store turns every one of those links into a copy.
ROOTS=(
  task-evaluation-inputs                                # one tree: its stores hardlink into each other
  pubsub-handoffs                                       # scene workspaces and the handoff spool
  production-gpu-artifacts
  pipeline-control-plane/task-evaluation-launch-runs
  pipeline-control-plane/task-evaluation-policy-canaries
  pipeline-control-plane/capture-reconstruction-runs
  pipeline-control-plane/capture-reconstruction-derived
  pipeline-control-plane/episode-interpretation-backfills
  pipeline-control-plane/policy-canary-preprovider-audits
  pipeline-control-plane/scene-configuration-diagnostics
  pipeline-control-plane/result-artifact-cache
  pipeline-control-plane/profile-install-staging
  pipeline-control-plane/policy-canary-presubmission
  pipeline-control-plane/native-g1-team-campaign-work
  pipeline-control-plane/engineering
  pipeline-control-plane/render-probes
  pipeline-control-plane/diagnostic-checkouts
  pipeline-control-plane/release-builds
)

# Roots outside --state-root, each bound to the same path under the mount.
ABSOLUTE_ROOTS=(/workspace)                              # bound to ${MOUNT}/workspace

# Units that write under the moved roots, with the timers and path units that
# would start them again (tests/test_mount_work_volume_script.py derives the set
# from deploy/systemd).  They are all stopped for the move, and only the ones that
# were running start again.  Intake stays up: it writes queues, which never move,
# and the reproducible result artifact cache, where a write that lands during the
# copy shows up as drift and the swap is refused.
WORKER_UNITS=(
  blueprint-task-evaluation-launch-preparation.path
  blueprint-task-evaluation-launch-preparation.timer
  blueprint-task-evaluation-sam31-preparation-execution.path
  blueprint-task-evaluation-sam31-preparation-execution.timer
  blueprint-task-evaluation-episode-compilation.path
  blueprint-task-evaluation-launch-activation.path
  blueprint-task-evaluation-launch-dispatcher.path
  blueprint-task-evaluation-policy-canary-dispatcher.path
  blueprint-native-g1-team-campaign-dispatcher.timer
  blueprint-native-g1-team-campaign-settlement.timer
  blueprint-task-evaluation-configured-controls-progression.timer
  blueprint-task-evaluation-configured-controls-progression.path
  blueprint-task-evaluation-terminal-resource-release.path
  blueprint-control-plane-storage-gc.timer
  blueprint-pubsub-handoff-listener.timer
  blueprint-task-evaluation-scene-progression.timer
  blueprint-capture-reconstruction-dispatcher.path
  blueprint-capture-reconstruction-dispatcher.timer
  blueprint-agent-stage-replay.timer
  blueprint-completed-replay-cache-gc.timer
  blueprint-scene-object-discovery.path
  blueprint-task-evaluation-launch-reconciler.timer
  blueprint-task-evaluation-launch-preparation.service
  blueprint-task-evaluation-sam31-preparation-execution.service
  blueprint-task-evaluation-episode-compilation.service
  blueprint-task-evaluation-launch-activation.service
  blueprint-task-evaluation-launch-dispatcher.service
  blueprint-task-evaluation-policy-canary-dispatcher.service
  blueprint-native-g1-team-campaign-dispatcher.service
  blueprint-native-g1-team-campaign-settlement.service
  blueprint-task-evaluation-configured-controls-progression.service
  blueprint-task-evaluation-terminal-resource-release.service
  blueprint-control-plane-storage-gc.service
  blueprint-pubsub-handoff-listener.service
  blueprint-task-evaluation-scene-progression.service
  blueprint-capture-reconstruction-dispatcher.service
  blueprint-agent-stage-replay.service
  blueprint-completed-replay-cache-gc.service
  blueprint-scene-object-discovery.service
  blueprint-task-evaluation-launch-reconciler.service
  blueprint-production-gpu-campaign-control-plane.service
)

usage() {
  cat <<USAGE
usage: $0 --device /dev/disk/by-id/<volume> [--mount ${MOUNT_DEFAULT}] [--plan | --apply --ack ${ACK_REQUIRED}]
          [--state-root ${STATE_ROOT_DEFAULT}] [--root-prefix DIR]

  --plan        (default) print what would move, with sizes; changes nothing
  --apply       perform the migration; requires root and --ack ${ACK_REQUIRED}
  --root-prefix prefix every host path with DIR (hermetic tests; no mounts are made)
USAGE
}

DEVICE=""
MOUNT="${MOUNT_DEFAULT}"
STATE_ROOT="${STATE_ROOT_DEFAULT}"
ROOT_PREFIX=""
MODE="plan"
ACK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --mount) MOUNT="$2"; shift 2 ;;
    --state-root) STATE_ROOT="$2"; shift 2 ;;
    --root-prefix) ROOT_PREFIX="$2"; shift 2 ;;
    --plan) MODE="plan"; shift ;;
    --apply) MODE="apply"; shift ;;
    --ack) ACK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "${DEVICE}" ] || { echo "--device is required" >&2; exit 2; }
HOST_STATE="${ROOT_PREFIX}${STATE_ROOT}"
HOST_MOUNT="${ROOT_PREFIX}${MOUNT}"

# Every root as its host path, its path relative to the mount (the same path
# relative to its base directory), and that base directory.
ROOT_HOST=()
ROOT_VREL=()
ROOT_BASE=()
for rel in "${ROOTS[@]}"; do
  ROOT_HOST+=("${STATE_ROOT}/${rel}"); ROOT_VREL+=("${rel}"); ROOT_BASE+=("${HOST_STATE}")
done
for abs in "${ABSOLUTE_ROOTS[@]}"; do
  ROOT_HOST+=("${abs}"); ROOT_VREL+=("${abs#/}"); ROOT_BASE+=("${ROOT_PREFIX}/")
done

size_mib() {
  if [ -d "$1" ]; then du -xsm "$1" 2>/dev/null | cut -f1; else echo 0; fi
}

is_bound() {
  # A root already served by the volume is a mountpoint whose source lives under the mount.
  [ -z "${ROOT_PREFIX}" ] && mountpoint -q "$1" 2>/dev/null
}

mounts_below() {
  # Mount points strictly below a host path.  Moving a root with mounts below it
  # would copy them onto themselves and then delete through them, so such a root
  # is blocked.  The hermetic prefix mounts nothing.
  [ -z "${ROOT_PREFIX}" ] || return 0
  findmnt -rn -o TARGET | awk -v prefix="$1/" 'index($0, prefix) == 1'
}

plan() {
  echo "device: ${DEVICE}"
  echo "mount:  ${HOST_MOUNT}"
  if [ -z "${ROOT_PREFIX}" ] && command -v blkid >/dev/null 2>&1; then
    echo "filesystem: $(blkid -o value -s TYPE "${DEVICE}" 2>/dev/null || echo none)"
  fi
  local total=0 i=0 root dest mib below
  while [ "${i}" -lt "${#ROOT_HOST[@]}" ]; do
    root="${ROOT_PREFIX}${ROOT_HOST[$i]}"
    dest="${HOST_MOUNT}/${ROOT_VREL[$i]}"
    mib="$(size_mib "${root}")"
    if is_bound "${root}"; then
      echo "bound    ${root} (${mib} MiB)"
    elif [ -d "${root}" ]; then
      below="$(mounts_below "${ROOT_HOST[$i]}")"
      if [ -n "${below}" ]; then
        echo "blocked  ${root} (mount points below it: $(printf '%s' "${below}" | tr '\n' ' '))"
      else
        total=$((total + mib))
        echo "move     ${root} -> ${dest} (${mib} MiB)"
      fi
    else
      echo "missing  ${root}"
    fi
    i=$((i + 1))
  done
  echo "total to move: ${total} MiB"
  echo "mode: ${MODE}; nothing changed"
}

# rsync the pending roots below each base directory in one relative invocation,
# so hardlinks between them survive.  Arguments go before the source paths.
rsync_pending_relative() {
  local base i rels
  for base in "${HOST_STATE}" "${ROOT_PREFIX}/"; do
    rels=()
    for i in "${PENDING[@]}"; do
      if [ "${ROOT_BASE[$i]}" = "${base}" ]; then rels+=("${ROOT_VREL[$i]}"); fi
    done
    [ "${#rels[@]}" -gt 0 ] || continue
    (cd "${base}" && rsync "${RSYNC_FLAGS[@]}" --relative "$@" "${rels[@]}" "${HOST_MOUNT}/") || return $?
  done
}

PENDING=()
RSYNC_FLAGS=(-a)

# Worker units that were running when the move began.  Only these start again,
# so a unit an operator had stopped stays stopped.
RUNNING_UNITS=()

restart_units() {
  [ "${#RUNNING_UNITS[@]}" -gt 0 ] || return 0
  echo "restarting worker units that were running: ${RUNNING_UNITS[*]}"
  systemctl start "${RUNNING_UNITS[@]}" || true
}

stop_units() {
  local unit state still=()
  for unit in "${WORKER_UNITS[@]}"; do
    if systemctl is-active --quiet "${unit}"; then RUNNING_UNITS+=("${unit}"); fi
  done
  echo "stopping worker units for the move"
  trap restart_units EXIT
  systemctl stop "${WORKER_UNITS[@]}" || true
  for unit in "${WORKER_UNITS[@]}"; do
    state="$(systemctl is-active "${unit}" 2>/dev/null || true)"
    case "${state}" in
      inactive|failed|unknown|"") ;;
      *) still+=("${unit}:${state}") ;;
    esac
  done
  if [ "${#still[@]}" -gt 0 ]; then
    echo "refusing: worker units still running after the stop: ${still[*]}" >&2
    exit 2
  fi
}

apply() {
  [ "${ACK}" = "${ACK_REQUIRED}" ] || { echo "refusing: --ack ${ACK_REQUIRED} is required to move production roots" >&2; exit 2; }
  [ -n "${ROOT_PREFIX}" ] || [ "$(id -u)" = "0" ] || { echo "refusing: --apply must run as root" >&2; exit 2; }
  if [ -z "${ROOT_PREFIX}" ]; then
    [ -b "${DEVICE}" ] || { echo "refusing: ${DEVICE} is not a block device" >&2; exit 2; }
    if ! blkid -o value -s TYPE "${DEVICE}" >/dev/null 2>&1; then
      echo "formatting ${DEVICE} as ext4 (no filesystem present)"
      mkfs.ext4 -F -L blueprint-work "${DEVICE}"
    fi
    mkdir -p "${HOST_MOUNT}"
    local uuid
    uuid="$(blkid -o value -s UUID "${DEVICE}")"
    if ! grep -q " ${HOST_MOUNT} " /etc/fstab; then
      echo "UUID=${uuid} ${HOST_MOUNT} ext4 defaults,nofail,noatime,discard 0 2" >> /etc/fstab
    fi
    mountpoint -q "${HOST_MOUNT}" || mount "${HOST_MOUNT}"
    systemctl daemon-reload
    stop_units
  else
    mkdir -p "${HOST_MOUNT}"
  fi

  local i=0 below
  while [ "${i}" -lt "${#ROOT_HOST[@]}" ]; do
    if [ -d "${ROOT_PREFIX}${ROOT_HOST[$i]}" ] && ! is_bound "${ROOT_PREFIX}${ROOT_HOST[$i]}"; then
      below="$(mounts_below "${ROOT_HOST[$i]}")"
      if [ -n "${below}" ]; then
        echo "refusing: ${ROOT_PREFIX}${ROOT_HOST[$i]} has mount points below it" >&2
        exit 2
      fi
      PENDING+=("${i}")
    fi
    i=$((i + 1))
  done
  if [ ${#PENDING[@]} -eq 0 ]; then
    echo "nothing to move"
    return 0
  fi

  # One rsync per base directory keeps hardlinks that span roots (deduplicated run
  # artifacts) as hardlinks on the volume.  GNU rsync (the production host) takes
  # the full flag set and copies all roots in one relative invocation; a minimal
  # rsync (macOS openrsync in the hermetic test) copies root by root with what it
  # supports, and the copy is verified with diff.
  local help relative=""
  help="$(rsync --help 2>&1 || true)"
  for opt in --hard-links --acls --xattrs --numeric-ids; do
    if printf '%s' "${help}" | grep -q -- "${opt}"; then RSYNC_FLAGS+=("${opt}"); fi
  done
  if printf '%s' "${help}" | grep -q -- "--relative" && printf '%s' "${help}" | grep -q -- "--itemize-changes"; then
    relative="yes"
  fi
  echo "copying ${#PENDING[@]} roots to ${HOST_MOUNT}"
  local root dest
  if [ -n "${relative}" ]; then
    rsync_pending_relative
  else
    for i in "${PENDING[@]}"; do
      root="${ROOT_PREFIX}${ROOT_HOST[$i]}"
      dest="${HOST_MOUNT}/${ROOT_VREL[$i]}"
      mkdir -p "${dest}"
      rsync "${RSYNC_FLAGS[@]}" "${root}/" "${dest}/"
    done
  fi
  echo "verifying the copy"
  local drift="" itemized
  if [ -n "${relative}" ]; then
    itemized="$(rsync_pending_relative -n --itemize-changes)" || {
      echo "refusing to swap: the verification rsync failed" >&2
      exit 3
    }
    drift="$(printf '%s\n' "${itemized}" | grep -v -e '^\.d' -e '^$' || true)"
  else
    for i in "${PENDING[@]}"; do
      drift="${drift}$(diff -rq "${ROOT_PREFIX}${ROOT_HOST[$i]}" "${HOST_MOUNT}/${ROOT_VREL[$i]}" || true)"
    done
  fi
  if [ -n "${drift}" ]; then
    echo "refusing to swap: copy differs from source" >&2
    echo "${drift}" | head -20 >&2
    exit 3
  fi

  for i in "${PENDING[@]}"; do
    local host="${ROOT_HOST[$i]}"
    root="${ROOT_PREFIX}${host}"
    dest="${HOST_MOUNT}/${ROOT_VREL[$i]}"
    local owner mode
    if stat -c '%u' / >/dev/null 2>&1; then
      owner="$(stat -c '%u:%g' "${root}")"
      mode="$(stat -c '%a' "${root}")"
    else
      owner="$(stat -f '%u:%g' "${root}")"
      mode="$(stat -f '%OLp' "${root}")"
    fi
    mv "${root}" "${root}.migrated-to-volume"
    mkdir -p "${root}"
    chown "${owner}" "${root}"
    chmod "${mode}" "${root}"
    if [ -z "${ROOT_PREFIX}" ]; then
      if ! grep -q " ${root} " /etc/fstab; then
        echo "${dest} ${root} none bind 0 0" >> /etc/fstab
      fi
      mount --bind "${dest}" "${root}"
    fi
    rm -rf "${root}.migrated-to-volume"
    echo "bound    ${root} <- ${dest}"
  done
  if [ -z "${ROOT_PREFIX}" ]; then systemctl daemon-reload; fi
  echo "done"
}

case "${MODE}" in
  plan) plan ;;
  apply) apply ;;
esac
