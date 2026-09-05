#!/usr/bin/env bash
# Move the control plane's bulk roots onto a resizable block volume and bind-mount
# them back at their original paths.
#
# Why: the control-plane root disk filled four times because run evidence, caches
# and scratch shared one 154 GB disk with the host's state.  Bulk bytes belong on a
# volume that can grow online; state, queues and ledgers stay on the root disk so a
# cache flood can never starve them.  Bind mounts keep every recorded path and every
# unit's ReadWritePaths valid, so nothing else changes.
#
# Plan by default.  --apply requires the acknowledgement and root, stops the worker
# units for the duration of one rsync (hardlinks across roots are preserved because
# all roots move in a single invocation), verifies the copy, swaps each root for a
# bind mount recorded in /etc/fstab, and only then removes the originals.
set -euo pipefail

ACK_REQUIRED="move-work-roots-to-volume"
MOUNT_DEFAULT="/mnt/blueprint-work"
STATE_ROOT_DEFAULT="/var/lib/blueprint"

# Bulk roots by storage class: cache, evidence_cold and scratch.  Never queues,
# ledgers, spend guard, intents or the control-plane manifest.
ROOTS=(
  task-evaluation-inputs/prepared-references
  task-evaluation-inputs/compiled-episodes
  task-evaluation-inputs/sam31-preparations
  task-evaluation-inputs/launch-activations
  task-evaluation-inputs/render-probes
  pipeline-control-plane/task-evaluation-launch-runs
  pipeline-control-plane/task-evaluation-policy-canaries
  pipeline-control-plane/episode-interpretation-backfills
  pipeline-control-plane/scene-configuration-diagnostics
  pipeline-control-plane/engineering
  pipeline-control-plane/render-probes
)

# Units that write under the moved roots.  Intake stays up: it writes queues only.
WORKER_UNITS=(
  blueprint-task-evaluation-launch-preparation.path
  blueprint-task-evaluation-sam31-preparation-execution.path
  blueprint-task-evaluation-sam31-preparation-execution.timer
  blueprint-task-evaluation-episode-compilation.path
  blueprint-task-evaluation-launch-activation.path
  blueprint-task-evaluation-launch-dispatcher.path
  blueprint-task-evaluation-policy-canary-dispatcher.path
  blueprint-task-evaluation-configured-controls-progression.timer
  blueprint-task-evaluation-configured-controls-progression.path
  blueprint-task-evaluation-terminal-resource-release.path
  blueprint-control-plane-storage-gc.timer
  blueprint-task-evaluation-launch-preparation.service
  blueprint-task-evaluation-sam31-preparation-execution.service
  blueprint-task-evaluation-episode-compilation.service
  blueprint-task-evaluation-launch-activation.service
  blueprint-task-evaluation-launch-dispatcher.service
  blueprint-task-evaluation-policy-canary-dispatcher.service
  blueprint-task-evaluation-configured-controls-progression.service
  blueprint-task-evaluation-terminal-resource-release.service
  blueprint-control-plane-storage-gc.service
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

size_mib() {
  if [ -d "$1" ]; then du -xsm "$1" 2>/dev/null | cut -f1; else echo 0; fi
}

is_bound() {
  # A root already served by the volume is a mountpoint whose source lives under the mount.
  [ -z "${ROOT_PREFIX}" ] && mountpoint -q "$1" 2>/dev/null
}

plan() {
  echo "device: ${DEVICE}"
  echo "mount:  ${HOST_MOUNT}"
  if [ -z "${ROOT_PREFIX}" ] && command -v blkid >/dev/null 2>&1; then
    echo "filesystem: $(blkid -o value -s TYPE "${DEVICE}" 2>/dev/null || echo none)"
  fi
  local total=0
  for rel in "${ROOTS[@]}"; do
    local root="${HOST_STATE}/${rel}"
    local mib
    mib="$(size_mib "${root}")"
    total=$((total + mib))
    if is_bound "${root}"; then
      echo "bound    ${root} (${mib} MiB)"
    elif [ -d "${root}" ]; then
      echo "move     ${root} -> ${HOST_MOUNT}/${rel} (${mib} MiB)"
    else
      echo "missing  ${root}"
    fi
  done
  echo "total to move: ${total} MiB"
  echo "mode: ${MODE}; nothing changed"
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
    echo "stopping worker units for the move"
    systemctl stop "${WORKER_UNITS[@]}" || true
    trap 'echo "restarting worker units"; systemctl start "${WORKER_UNITS[@]}" || true' EXIT
  else
    mkdir -p "${HOST_MOUNT}"
  fi

  local pending=()
  for rel in "${ROOTS[@]}"; do
    local root="${HOST_STATE}/${rel}"
    [ -d "${root}" ] || continue
    is_bound "${root}" && continue
    pending+=("${rel}")
  done
  if [ ${#pending[@]} -eq 0 ]; then
    echo "nothing to move"
    return 0
  fi

  # One rsync for every root keeps hardlinks that span roots (deduplicated run
  # artifacts) as hardlinks on the volume.  GNU rsync (the production host) takes
  # the full flag set and copies all roots in one relative invocation; a minimal
  # rsync (macOS openrsync in the hermetic test) copies root by root with what it
  # supports, and the copy is verified with diff.
  local help flags=(-a)
  help="$(rsync --help 2>&1 || true)"
  for opt in --hard-links --acls --xattrs --numeric-ids; do
    if printf '%s' "${help}" | grep -q -- "${opt}"; then flags+=("${opt}"); fi
  done
  echo "copying ${#pending[@]} roots to ${HOST_MOUNT}"
  if printf '%s' "${help}" | grep -q -- "--relative" && printf '%s' "${help}" | grep -q -- "--itemize-changes"; then
    (cd "${HOST_STATE}" && rsync "${flags[@]}" --relative "${pending[@]}" "${HOST_MOUNT}/")
  else
    for rel in "${pending[@]}"; do
      mkdir -p "${HOST_MOUNT}/${rel}"
      rsync "${flags[@]}" "${HOST_STATE}/${rel}/" "${HOST_MOUNT}/${rel}/"
    done
  fi
  echo "verifying the copy"
  local drift=""
  if printf '%s' "${help}" | grep -q -- "--relative" && printf '%s' "${help}" | grep -q -- "--itemize-changes"; then
    drift="$(cd "${HOST_STATE}" && rsync "${flags[@]}" --relative -n --itemize-changes "${pending[@]}" "${HOST_MOUNT}/" | grep -v '^\.d' || true)"
  else
    for rel in "${pending[@]}"; do
      drift="${drift}$(diff -rq "${HOST_STATE}/${rel}" "${HOST_MOUNT}/${rel}" || true)"
    done
  fi
  if [ -n "${drift}" ]; then
    echo "refusing to swap: copy differs from source" >&2
    echo "${drift}" | head -20 >&2
    exit 3
  fi

  for rel in "${pending[@]}"; do
    local root="${HOST_STATE}/${rel}"
    local dest="${HOST_MOUNT}/${rel}"
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
  [ -z "${ROOT_PREFIX}" ] && systemctl daemon-reload
  echo "done"
}

case "${MODE}" in
  plan) plan ;;
  apply) apply ;;
esac
