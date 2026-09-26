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
# One bind per root.  A host that still carries the September per-store binds
# below task-evaluation-inputs is consolidated: the tree is copied around them,
# they are unmounted and dropped from /etc/fstab, and the tree is bound whole.
#
# Plan by default.  --apply requires the acknowledgement and root, and refuses
# before it moves anything when the layout is in doubt.  It stops the worker units
# for the duration of the copy (one rsync per base directory, so hardlinks between
# roots are preserved), verifies the copy, swaps each root for a bind mount
# recorded in /etc/fstab (backed up first), and removes an original only once it
# matches the volume copy again; otherwise it keeps the original and stops.
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

# Hot evidence that rides inside a bulk root by design: the volume is durable
# block storage, and splitting task-evaluation-inputs would break the hardlinks
# that keep it small.  The plan names these so the owner sees them, and the
# governance test refuses hot evidence on the volume that is not listed here.
# Paths are relative to --state-root; quote a glob.
EVIDENCE_HOT_ON_VOLUME=(
  task-evaluation-inputs/sam31-profile-registry
  task-evaluation-inputs/task-evaluation-terminal-results
  task-evaluation-inputs/g1-team-campaign-registry.json
)

# Units whose sandbox can write under the moved roots, with the timers and path
# units that would start them again.  They are all stopped for the move, and only
# the ones that were running start again.  The spend guard stops too: its orphan
# scan reads pod owner files under the moved trees, and one scan while a root is
# between its old and new mounts could reap a live pod.  Every unit that can write
# under /var/lib/blueprint is either here or in UNITS_LEFT_RUNNING
# (tests/test_mount_work_volume_script.py derives the set from deploy/systemd).
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
  blueprint-pipeline-control-plane.timer
  blueprint-agent-run-dispatcher.timer
  blueprint-existing-policy-canary-continuation.timer
  blueprint-gpu-spend-guard.timer
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
  blueprint-pipeline-control-plane.service
  blueprint-agent-run-dispatcher.service
  blueprint-existing-policy-canary-continuation.service
  blueprint-existing-policy-canary-watchdog.service
  blueprint-gpu-spend-guard.service
)

# Units whose sandbox can write under /var/lib/blueprint but that keep running
# during the move, one per line with the reason that is safe.
# shellcheck disable=SC2034  # read by tests/test_mount_work_volume_script.py
UNITS_LEFT_RUNNING=(
  blueprint-pipeline-intake.service                # by design; queues never move, and a result-cache write mid-move shows up as drift or a stale copy
  blueprint-provider-billing-reconciler.service    # writes billing evidence under gpu_spend_guard only, which never moves
  blueprint-production-gpu-worker-pool.service     # its only state is production-gpu-worker-pool.sqlite, which never moves
  blueprint-production-gpu-worker-agent.service    # writes host, cache and warm evidence under /var/lib/blueprint/evidence, which never moves
)

usage() {
  cat <<USAGE
usage: $0 --device /dev/disk/by-id/<volume> [--mount ${MOUNT_DEFAULT}] [--plan | --apply --ack ${ACK_REQUIRED}]
          [--state-root ${STATE_ROOT_DEFAULT}] [--root-prefix DIR [--bound-roots-file FILE]]

  --plan              (default) print what would move, with sizes; changes nothing
  --apply             perform the migration; requires root and --ack ${ACK_REQUIRED}
  --root-prefix       prefix every host path with DIR (hermetic tests; no mounts are
                      made, and DIR/etc/fstab is edited only when it exists)
  --bound-roots-file  with --root-prefix: the mount points to assume, one host path
                      per line as findmnt lists them; apply edits it in place of
                      mount and umount.  A path followed by "busy" will not
                      unmount (the stand-in for EBUSY)
USAGE
}

refuse() {  # exit code, reason, then detail lines
  local code="$1" detail
  shift
  echo "refusing: $1" >&2
  shift
  for detail in "$@"; do
    if [ -n "${detail}" ]; then printf '%s\n' "${detail}" | sed 's/^/  /' >&2; fi
  done
  exit "${code}"
}

DEVICE=""
MOUNT="${MOUNT_DEFAULT}"
STATE_ROOT="${STATE_ROOT_DEFAULT}"
ROOT_PREFIX=""
BOUND_ROOTS_FILE=""
MODE="plan"
ACK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --mount) MOUNT="$2"; shift 2 ;;
    --state-root) STATE_ROOT="$2"; shift 2 ;;
    --root-prefix) ROOT_PREFIX="$2"; shift 2 ;;
    --bound-roots-file) BOUND_ROOTS_FILE="$2"; shift 2 ;;
    --plan) MODE="plan"; shift ;;
    --apply) MODE="apply"; shift ;;
    --ack) ACK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "${DEVICE}" ] || { echo "--device is required" >&2; exit 2; }

# Paths are compared as strings with the mount table and /etc/fstab, so they must
# be absolute and clean.
clean_absolute() {
  case "$1" in /?*) ;; *) return 1 ;; esac
  case "$1/" in *//*|*/./*|*/../*) return 1 ;; esac
}
MOUNT="${MOUNT%/}"
STATE_ROOT="${STATE_ROOT%/}"
ROOT_PREFIX="${ROOT_PREFIX%/}"
clean_absolute "${MOUNT}" || refuse 2 "--mount must be a clean absolute path"
clean_absolute "${STATE_ROOT}" || refuse 2 "--state-root must be a clean absolute path"
if [ -n "${ROOT_PREFIX}" ]; then
  clean_absolute "${ROOT_PREFIX}" || refuse 2 "--root-prefix must be a clean absolute path"
fi
if [ -n "${BOUND_ROOTS_FILE}" ]; then
  # A hermetic-test hook: on a host the mount table is the only authority.
  [ -n "${ROOT_PREFIX}" ] || refuse 2 "--bound-roots-file needs --root-prefix"
  [ -f "${BOUND_ROOTS_FILE}" ] || refuse 2 "--bound-roots-file does not exist"
fi

HOST_STATE="${ROOT_PREFIX}${STATE_ROOT}"
HOST_MOUNT="${ROOT_PREFIX}${MOUNT}"
FSTAB="${ROOT_PREFIX}/etc/fstab"
EPOCH="$(date +%s)"

# Every root as its host path (what the mount table and /etc/fstab name), its path
# relative to the mount (the same path relative to its base directory), and that
# base directory.
ROOT_HOST=()
ROOT_VREL=()
ROOT_BASE=()
for rel in "${ROOTS[@]}"; do
  ROOT_HOST+=("${STATE_ROOT}/${rel}"); ROOT_VREL+=("${rel}"); ROOT_BASE+=("${HOST_STATE}")
done
for abs in "${ABSOLUTE_ROOTS[@]}"; do
  ROOT_HOST+=("${abs}"); ROOT_VREL+=("${abs#/}"); ROOT_BASE+=("${ROOT_PREFIX}/")
done

within() {  # $1 is $2 or lies below it
  [ "$1" = "$2" ] || [ "${1#"$2"/}" != "$1" ]
}

for host in "${ROOT_HOST[@]}"; do
  if within "${MOUNT}" "${host}" || within "${host}" "${MOUNT}"; then
    refuse 2 "--mount overlaps a root it would move" "${host}"
  fi
done

exists() { [ -e "$1" ] || [ -L "$1" ]; }

size_mib() {
  # du still prints a total when a file vanishes under it, and exits 1 for that.
  local mib=""
  if [ -d "$1" ]; then mib="$(du -xsm "$1" 2>/dev/null | cut -f1)" || true; fi
  echo "${mib:-0}"
}

inode_of() {
  if stat -c '%d:%i' / >/dev/null 2>&1; then stat -c '%d:%i' "$1"; else stat -f '%d:%i' "$1"; fi
}

# --- the mount table ---------------------------------------------------------
# Targets as host paths, with each mount's device and filesystem root on a host.
# Under --root-prefix the table is --bound-roots-file (targets only), or empty.
MT_TARGET=()
MT_DEVICE=()
MT_FSROOT=()
VOLUME_DEVICE=""
VOLUME_FSROOT=""

load_mount_table() {
  local table target device fsroot i=0 rows=0
  MT_TARGET=()
  MT_DEVICE=()
  MT_FSROOT=()
  if [ -n "${ROOT_PREFIX}" ]; then
    table=""
    if [ -n "${BOUND_ROOTS_FILE}" ]; then table="$(cat "${BOUND_ROOTS_FILE}")"; fi
  else
    table="$(findmnt -rn -o TARGET,MAJ:MIN,FSROOT)"
  fi
  while read -r target device fsroot; do
    [ -n "${target}" ] || continue
    MT_TARGET+=("${target}")
    MT_DEVICE+=("${device:-}")
    MT_FSROOT+=("${fsroot:-}")
  done <<< "${table}"
  if [ -z "${ROOT_PREFIX}" ] && ! is_mount_point /; then
    refuse 2 "could not read the mount table"
  fi
  # The volume, when exactly one mount serves the mount path.
  VOLUME_DEVICE=""
  VOLUME_FSROOT=""
  while [ "${i}" -lt "${#MT_TARGET[@]}" ]; do
    if [ "${MT_TARGET[i]}" = "${MOUNT}" ]; then
      rows=$((rows + 1))
      VOLUME_DEVICE="${MT_DEVICE[i]}"
      VOLUME_FSROOT="${MT_FSROOT[i]}"
    fi
    i=$((i + 1))
  done
  if [ "${rows}" -ne 1 ]; then
    VOLUME_DEVICE=""
    VOLUME_FSROOT=""
  fi
}

is_mount_point() {  # host path
  local i=0
  while [ "${i}" -lt "${#MT_TARGET[@]}" ]; do
    if [ "${MT_TARGET[i]}" = "$1" ]; then return 0; fi
    i=$((i + 1))
  done
  return 1
}

mounts_below() {  # host path; prints the mount targets strictly below it, deepest first
  local i=0
  while [ "${i}" -lt "${#MT_TARGET[@]}" ]; do
    case "${MT_TARGET[i]}" in "$1"/*) printf '%s\n' "${MT_TARGET[i]}" ;; esac
    i=$((i + 1))
  done | LC_ALL=C sort -ru
}

is_volume_bind() {  # child host path, the filesystem root it must bind from the volume
  # The hermetic table lists targets only; its children are volume binds by
  # construction, and their volume copies are checked like a host's.
  [ -z "${ROOT_PREFIX}" ] || return 0
  [ -n "${VOLUME_DEVICE}" ] || return 1
  local i=0 rows=0
  while [ "${i}" -lt "${#MT_TARGET[@]}" ]; do
    if [ "${MT_TARGET[i]}" = "$1" ]; then
      rows=$((rows + 1))
      if [ "${MT_DEVICE[i]}" != "${VOLUME_DEVICE}" ] || [ "${MT_FSROOT[i]}" != "$2" ]; then return 1; fi
    fi
    i=$((i + 1))
  done
  [ "${rows}" -eq 1 ]
}

# --- /etc/fstab ----------------------------------------------------------------
FSTAB_BACKUP=""
FSTAB_NEXT=""     # a rewritten fstab beside the real one, not yet renamed into place
FSTAB_NOTE=""
FSTAB_CHANGED=""

fstab_enabled() { [ -z "${ROOT_PREFIX}" ] || [ -f "${FSTAB}" ]; }

fstab_entries() {  # live entries as "target source type options"
  awk '!/^[[:space:]]*(#|$)/ && NF >= 2 { print $2, $1, $3, $4 }' "${FSTAB}"
}

fstab_has_target() {  # host path [file]
  TARGET="$1" awk '!/^[[:space:]]*#/ && $2 == ENVIRON["TARGET"] { found = 1 } END { exit !found }' "${2:-${FSTAB}}"
}

has_bind_option() {
  case ",$1," in *,bind,*|*,rbind,*) return 0 ;; *) return 1 ;; esac
}

backup_fstab() {  # once per run, before the first edit
  [ -z "${FSTAB_BACKUP}" ] || return 0
  FSTAB_BACKUP="${FSTAB}.blueprint-${EPOCH}.bak"
  if exists "${FSTAB_BACKUP}"; then FSTAB_BACKUP="${FSTAB}.blueprint-${EPOCH}-$$.bak"; fi
  cp -p "${FSTAB}" "${FSTAB_BACKUP}"
  echo "backed up ${FSTAB} to ${FSTAB_BACKUP}"
}

# Write the rewritten /etc/fstab beside it, keeping its mode: the entries whose
# targets $1 lists (one per line) dropped, and entry $2 appended when its target
# has none.  Nothing changes until fstab_install renames it into place, so the
# write (and the backup) can come before the first change of a swap.
fstab_prepare() {  # targets to drop, entry to add (may be empty)
  local drop="$1" line="$2" add=0 target tmp before hits after
  local begin='BEGIN { count = split(ENVIRON["DROP"], names, "\n"); for (k = 1; k <= count; k++) if (names[k] != "") drop[names[k]] = 1 }'
  # shellcheck disable=SC2016  # awk code: $2 is awk's second field
  local match='!/^[[:space:]]*#/ && ($2 in drop)'
  FSTAB_NEXT=""
  FSTAB_NOTE=""
  if [ -n "${line}" ]; then
    target="${line#* }"
    target="${target%% *}"
    fstab_has_target "${target}" || add=1
  fi
  if [ -z "${drop}" ] && [ "${add}" -eq 0 ]; then return 0; fi
  backup_fstab
  tmp="${FSTAB}.blueprint-edit.$$"
  before="$(awk 'END { print NR }' "${FSTAB}")"
  hits="$(DROP="${drop}" awk "${begin} ${match} { hits++ } END { print hits + 0 }" "${FSTAB}")"
  cp -p "${FSTAB}" "${tmp}"
  FSTAB_NEXT="${tmp}"
  DROP="${drop}" awk "${begin} ${match} { next } { print }" "${FSTAB}" > "${tmp}"
  if [ "${add}" -eq 1 ]; then printf '%s\n' "${line}" >> "${tmp}"; fi
  after="$(awk 'END { print NR }' "${tmp}")"
  if [ "${after}" -ne $((before - hits + add)) ]; then
    refuse 2 "the rewritten ${FSTAB} has ${after} lines, not $((before - hits + add)); left it unchanged"
  fi
  if [ "${add}" -eq 1 ]; then FSTAB_NOTE="recorded ${line} in ${FSTAB}"; else FSTAB_NOTE="rewrote ${FSTAB}"; fi
  FSTAB_NOTE="${FSTAB_NOTE}${drop:+, dropped $(printf '%s' "${drop%$'\n'}" | tr '\n' ' ')}"
}

fstab_prepare_root() {  # index: drop the bound children's entries, record the root's bind
  fstab_enabled || return 0
  local i="$1" host="${ROOT_HOST[$1]}" rel drop=""
  # shellcheck disable=SC2086  # child names were validated: no whitespace or glob characters
  for rel in ${CHILDREN[i]}; do drop="${drop}${host}/${rel}"$'\n'; done
  fstab_prepare "${drop}" "${MOUNT}/${ROOT_VREL[$1]} ${host} none bind 0 0"
}

fstab_install() {  # one atomic rename
  [ -n "${FSTAB_NEXT}" ] || return 0
  mv -- "${FSTAB_NEXT}" "${FSTAB}"
  FSTAB_NEXT=""
  FSTAB_CHANGED="yes"
  echo "${FSTAB_NOTE}"
}

# --- classification ------------------------------------------------------------
# bound:       the root itself is a mount point; left alone.
# move:        a plain directory; copied, then bound.
# consolidate: old per-store binds of the volume lie below it; the tree is copied
#              around them, they are unmounted, and the tree is bound whole.
# missing:     nothing to move.
# blocked:     in doubt; --apply refuses before it moves anything.
STATUS=()
CHILDREN=()
REASON=()

block() { STATUS[$1]=blocked; REASON[$1]="$2"; }

classify_roots() {
  local i=0 host root dest below
  while [ "${i}" -lt "${#ROOT_HOST[@]}" ]; do
    host="${ROOT_HOST[i]}"
    root="${ROOT_PREFIX}${host}"
    dest="${HOST_MOUNT}/${ROOT_VREL[i]}"
    STATUS[i]=""
    CHILDREN[i]=""
    REASON[i]=""
    if is_mount_point "${host}"; then
      STATUS[i]=bound
    elif exists "${root}.migrated-to-volume"; then
      block "${i}" "an earlier move kept ${root}.migrated-to-volume; reconcile it with ${dest} first"
    elif exists "${root}.new-mount-point"; then
      block "${i}" "an earlier move left ${root}.new-mount-point; check it and remove it first"
    elif [ -L "${root}" ]; then
      block "${i}" "the root is a symlink"
    elif ! exists "${root}"; then
      STATUS[i]=missing
    elif [ ! -d "${root}" ]; then
      block "${i}" "the root is not a directory"
    elif [ -z "${ROOT_PREFIX}" ] && [ "$(readlink -f -- "${root}")" != "${root}" ]; then
      block "${i}" "the path runs through a symlink"
    elif [ -L "${dest}" ] || { exists "${dest}" && [ ! -d "${dest}" ]; }; then
      block "${i}" "${dest} is not a directory"
    elif [ -d "${dest}" ] && [ "$(inode_of "${root}")" = "$(inode_of "${dest}")" ]; then
      block "${i}" "it already is ${dest}"
    else
      STATUS[i]=move
      below="$(mounts_below "${host}")"
      if [ -n "${below}" ]; then classify_children "${i}" "${below}"; fi
      if [ "${STATUS[i]}" != blocked ]; then check_fstab "${i}"; fi
    fi
    i=$((i + 1))
  done
}

classify_children() {  # index, mount targets below the root, deepest first
  local i="$1" host="${ROOT_HOST[$1]}" vrel="${ROOT_VREL[$1]}" child rel list=""
  while IFS= read -r child; do
    rel="${child#"${host}"/}"
    case "${rel}" in
      *[[:space:]]*|*'*'*|*'?'*|*'['*|*\\*)
        block "${i}" "cannot consolidate ${child}: unsupported name"
        return 0 ;;
    esac
    if [ ! -d "${HOST_MOUNT}/${vrel}/${rel}" ] || ! is_volume_bind "${child}" "${VOLUME_FSROOT%/}/${vrel}/${rel}"; then
      block "${i}" "${child} is mounted, but not as a bind of ${MOUNT}/${vrel}/${rel}"
      return 0
    fi
    list="${list}${list:+ }${rel}"
  done <<< "$2"
  STATUS[i]=consolidate
  CHILDREN[i]="${list}"
}

# Every /etc/fstab entry at or below a root must be one this script wrote: the
# root's own bind, or a bound child's.  Anything else would mount inside the tree
# at the next boot.
check_fstab() {  # index
  fstab_enabled || return 0
  local i="$1" host="${ROOT_HOST[$1]}" vrel="${ROOT_VREL[$1]}" entries target source fstype options rel expected
  entries="$(fstab_entries)"
  while read -r target source fstype options; do
    [ -n "${target}" ] || continue
    if [ "${target}" = "${host}" ]; then
      expected="${MOUNT}/${vrel}"
    elif within "${target}" "${host}"; then
      rel="${target#"${host}"/}"
      case " ${CHILDREN[i]} " in
        *" ${rel} "*) expected="${MOUNT}/${vrel}/${rel}" ;;
        *) block "${i}" "${FSTAB} mounts ${target} inside the root"; return 0 ;;
      esac
    else
      continue
    fi
    if [ "${source}" != "${expected}" ] || [ "${fstype}" != none ] || ! has_bind_option "${options}"; then
      block "${i}" "${FSTAB} has an unexpected entry for ${target}"
      return 0
    fi
  done <<< "${entries}"
}

plan() {
  echo "device: ${DEVICE}"
  echo "mount:  ${HOST_MOUNT}"
  if [ -z "${ROOT_PREFIX}" ] && command -v blkid >/dev/null 2>&1; then
    echo "filesystem: $(blkid -o value -s TYPE "${DEVICE}" 2>/dev/null || echo none)"
  fi
  load_mount_table
  classify_roots
  local total=0 i=0 root dest mib note rel
  while [ "${i}" -lt "${#ROOT_HOST[@]}" ]; do
    root="${ROOT_PREFIX}${ROOT_HOST[i]}"
    dest="${HOST_MOUNT}/${ROOT_VREL[i]}"
    case "${STATUS[i]}" in
      bound)
        note=""
        if fstab_enabled && ! fstab_has_target "${ROOT_HOST[i]}"; then
          note="; not in ${FSTAB}, so the bind does not survive a reboot"
        fi
        echo "bound    ${root} ($(size_mib "${root}") MiB${note})"
        if exists "${root}.migrated-to-volume"; then
          echo "kept     ${root}.migrated-to-volume (an earlier move kept it; reconcile it with ${dest})"
        fi
        ;;
      move)
        mib="$(size_mib "${root}")"
        total=$((total + mib))
        note=""
        if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}")" ]; then
          note="; ${dest} is not empty, and apply refuses anything there the root lacks"
        fi
        echo "move     ${root} -> ${dest} (${mib} MiB${note})"
        ;;
      consolidate)
        mib="$(size_mib "${root}")"
        total=$((total + mib))
        echo "consolidate ${root} -> ${dest} (${mib} MiB; bound children: ${CHILDREN[i]})"
        ;;
      blocked) echo "blocked  ${root} (${REASON[i]})" ;;
      *) echo "missing  ${root}" ;;
    esac
    i=$((i + 1))
  done
  for rel in "${EVIDENCE_HOT_ON_VOLUME[@]}"; do
    echo "evidence_hot on volume: ${HOST_STATE}/${rel}"
  done
  echo "total to move: ${total} MiB"
  echo "mode: ${MODE}; nothing changed"
}

# --- copy and compare ------------------------------------------------------------
PENDING=()
RSYNC_FLAGS=(-a)
RELATIVE=""

# One rsync per base directory keeps hardlinks that span roots (deduplicated run
# artifacts) as hardlinks on the volume.  GNU rsync (the production host) takes
# the full flag set and copies all roots in one relative invocation; a minimal
# rsync (macOS openrsync in the hermetic test) copies root by root with what it
# supports, and the copy is compared file by file.
probe_rsync() {
  local help opt
  help="$(rsync --help 2>&1 || true)"
  for opt in --hard-links --acls --xattrs --numeric-ids; do
    if printf '%s' "${help}" | grep -q -- "${opt}"; then RSYNC_FLAGS+=("${opt}"); fi
  done
  if printf '%s' "${help}" | grep -q -- "--relative" && printf '%s' "${help}" | grep -q -- "--itemize-changes"; then
    RELATIVE="yes"
  fi
  # On a host a missing flag is not a smaller copy but a wrong one: without
  # --hard-links the content stores would be copied apart, and the verification
  # runs with the same flags, so it would not notice.
  if [ -z "${ROOT_PREFIX}" ]; then
    local missing=()
    for opt in --hard-links --acls --xattrs --numeric-ids; do
      case " ${RSYNC_FLAGS[*]} " in *" ${opt} "*) ;; *) missing+=("${opt}") ;; esac
    done
    [ -n "${RELATIVE}" ] || missing+=("--relative with --itemize-changes")
    if [ "${#missing[@]}" -gt 0 ]; then
      refuse 2 "this host's rsync lacks what the move needs" "${missing[@]}"
    fi
  fi
}

# rsync the pending roots below each base directory in one relative invocation.
# A consolidating root's bound children are excluded: their bytes are already on
# the volume, served by the very mounts being replaced.  Arguments go before the
# source paths.
rsync_pending_relative() {
  local base i rel rels excludes
  for base in "${HOST_STATE}" "${ROOT_PREFIX}/"; do
    rels=()
    excludes=()
    for i in "${PENDING[@]}"; do
      [ "${ROOT_BASE[i]}" = "${base}" ] || continue
      rels+=("${ROOT_VREL[i]}")
      # shellcheck disable=SC2086  # child names were validated: no whitespace or glob characters
      for rel in ${CHILDREN[i]}; do excludes+=("--exclude=/${ROOT_VREL[i]}/${rel}/"); done
    done
    [ "${#rels[@]}" -gt 0 ] || continue
    (cd "${base}" && rsync "${RSYNC_FLAGS[@]}" --relative ${excludes[@]+"${excludes[@]}"} "$@" "${rels[@]}" "${HOST_MOUNT}/") || return $?
  done
}

# Entries of $1, minus the relative paths that follow, that are missing from $2
# or differ there.  Content is compared byte for byte; $2 may hold more.
tree_drift() {
  local src="$1" dest="$2" rel entry prune=()
  shift 2
  for rel in "$@"; do prune+=(-path "./${rel}" -prune -o); done
  (cd "${src}" && find . ${prune[@]+"${prune[@]}"} -print) | while IFS= read -r entry; do
    if [ -L "${src}/${entry}" ]; then
      if [ ! -L "${dest}/${entry}" ] || [ "$(readlink "${src}/${entry}")" != "$(readlink "${dest}/${entry}")" ]; then
        echo "differs  ${entry}"
      fi
    elif [ -d "${src}/${entry}" ]; then
      if [ -L "${dest}/${entry}" ] || [ ! -d "${dest}/${entry}" ]; then echo "missing  ${entry}"; fi
    elif [ -f "${src}/${entry}" ]; then
      if [ -L "${dest}/${entry}" ] || [ ! -f "${dest}/${entry}" ] || ! cmp -s "${src}/${entry}" "${dest}/${entry}"; then
        echo "differs  ${entry}"
      fi
    else
      echo "cannot compare  ${entry}"
    fi
  done
}

# What of $1 the volume copy $2 lacks, leaving out the relative paths that follow.
drift_between() {
  local src="$1" dest="$2" rel itemized excludes=()
  shift 2
  if [ -n "${RELATIVE}" ]; then
    for rel in "$@"; do excludes+=("--exclude=/${rel}/"); done
    itemized="$(rsync "${RSYNC_FLAGS[@]}" -n --itemize-changes ${excludes[@]+"${excludes[@]}"} "${src}/" "${dest}/")" || return $?
    printf '%s\n' "${itemized}" | grep -v -e '^\.d' -e '^$' || true
  else
    tree_drift "${src}" "${dest}" "$@"
  fi
}

# --- mounts ----------------------------------------------------------------------
# On a host these mount and unmount; under --root-prefix they edit the bound-roots
# file instead, and never touch a mount.
bind_at() {  # volume path, host path of the mount point
  if [ -z "${ROOT_PREFIX}" ]; then
    mount --bind "$1" "$2"
  elif [ -n "${BOUND_ROOTS_FILE}" ]; then
    printf '%s\n' "$2" >> "${BOUND_ROOTS_FILE}"
  fi
}

held_busy() {  # hermetic: the bound-roots file marks a mount point that will not unmount
  local i=0
  while [ "${i}" -lt "${#MT_TARGET[@]}" ]; do
    if [ "${MT_TARGET[i]}" = "$1" ] && [ "${MT_DEVICE[i]}" = busy ]; then return 0; fi
    i=$((i + 1))
  done
  return 1
}

unbind_at() {  # host path of a mount point
  if [ -z "${ROOT_PREFIX}" ]; then
    umount "$1"
  else
    if held_busy "$1"; then return 1; fi
    local tmp="${BOUND_ROOTS_FILE}.edit.$$"
    cp -p "${BOUND_ROOTS_FILE}" "${tmp}" &&
      LINE="$1" awk '$1 != ENVIRON["LINE"]' "${BOUND_ROOTS_FILE}" > "${tmp}" &&
      mv "${tmp}" "${BOUND_ROOTS_FILE}"
  fi
}

# Rename $1 to $2, which must not exist, so it can never land inside a directory
# that appeared there meanwhile.  GNU mv -T renames onto the name itself (at most
# replacing an empty directory); the hermetic path checks first.
rename_to() {
  if [ -z "${ROOT_PREFIX}" ]; then
    mv -T -- "$1" "$2"
  elif exists "$2"; then
    echo "$2 appeared during the swap" >&2
    return 1
  else
    mv -- "$1" "$2"
  fi
}

# A root between its old and new mounts.  The worker units stay stopped while one
# is set, so nothing writes into a half-swapped tree.
SWAPPING=""
# A prepared mount point that has not been renamed into place yet.
STAGED=""

rebind_children() {  # index, then the children unbound so far (deepest first)
  local i="$1" host="${ROOT_HOST[$1]}" vrel="${ROOT_VREL[$1]}" k
  shift
  local undone=("$@")
  k=$((${#undone[@]} - 1))
  while [ "${k}" -ge 0 ]; do
    if ! bind_at "${HOST_MOUNT}/${vrel}/${undone[k]}" "${host}/${undone[k]}"; then
      echo "could not bind ${host}/${undone[k]} back; ${host} is half consolidated" >&2
      return 0
    fi
    echo "bound back ${ROOT_PREFIX}${host}/${undone[k]}"
    k=$((k - 1))
  done
  SWAPPING=""
}

# Unmount a consolidating root's bound children, deepest first.  If one will not
# unmount, or a mount is left at or below the root, bind back the ones already
# undone so the host keeps its old layout, and refuse.
unbind_children() {  # index
  local i="$1" host="${ROOT_HOST[$1]}" rel below
  local undone=()
  # shellcheck disable=SC2086  # child names were validated: no whitespace or glob characters
  for rel in ${CHILDREN[i]}; do
    if ! unbind_at "${host}/${rel}"; then
      rebind_children "${i}" ${undone[@]+"${undone[@]}"}
      refuse 2 "could not unmount ${host}/${rel}" "fuser -vm ${host}/${rel} shows what holds it"
    fi
    undone+=("${rel}")
    echo "unbound  ${ROOT_PREFIX}${host}/${rel} (consolidating into ${ROOT_PREFIX}${host})"
  done
  load_mount_table
  below="$(mounts_below "${host}")"
  if [ -n "${below}" ] || is_mount_point "${host}"; then
    rebind_children "${i}" ${undone[@]+"${undone[@]}"}
    refuse 2 "a mount is left at or below ${host} after its binds were undone" "${below}"
  fi
}

swap_root() {  # index
  local i="$1" host="${ROOT_HOST[$1]}"
  local root="${ROOT_PREFIX}${ROOT_HOST[$1]}" dest="${HOST_MOUNT}/${ROOT_VREL[$1]}"
  local kept="${ROOT_PREFIX}${ROOT_HOST[$1]}.migrated-to-volume" owner mode below drift
  if stat -c '%u' / >/dev/null 2>&1; then
    owner="$(stat -c '%u:%g' "${root}")"
    mode="$(stat -c '%a' "${root}")"
  else
    owner="$(stat -f '%u:%g' "${root}")"
    mode="$(stat -f '%OLp' "${root}")"
  fi
  if [ -z "${ROOT_PREFIX}" ]; then
    require_units_stopped
    require_no_door_requests
  fi
  # Everything the swap writes to the root disk comes first (the new mount point
  # and the rewritten fstab), so a full disk refuses here and never halfway.
  STAGED="${root}.new-mount-point"
  mkdir "${STAGED}"
  chown "${owner}" "${STAGED}"
  chmod "${mode}" "${STAGED}"
  fstab_prepare_root "${i}"
  if [ "${STATUS[i]}" = consolidate ]; then
    SWAPPING="${root}"
    unbind_children "${i}"
  else
    load_mount_table
    below="$(mounts_below "${host}")"
    if [ -n "${below}" ] || is_mount_point "${host}"; then
      refuse 2 "${host} gained a mount during the copy; it was not moved" "${below}"
    fi
    SWAPPING="${root}"
  fi
  # Two renames and a mount.  Intake stays up and may recreate a cache root it
  # writes; if it does so in between, the second rename fails instead of landing
  # inside it.
  rename_to "${root}" "${kept}"
  rename_to "${STAGED}" "${root}"
  STAGED=""
  bind_at "${dest}" "${host}"
  fstab_install
  SWAPPING=""
  echo "bound    ${root} <- ${dest}"

  # The original goes only once it matches the volume copy again, so nothing
  # written after the verification, and nothing an old bind hid, is lost.
  drift="$(drift_between "${kept}" "${dest}")" || refuse 3 "could not compare ${kept} with ${dest}; kept it"
  if [ -n "${drift}" ]; then
    echo "refusing to remove ${kept}: it holds bytes the volume copy lacks" >&2
    # A here-string, not a pipe: under pipefail a writer cut off by head would
    # turn this refusal into a SIGPIPE exit.
    head -20 <<< "${drift}" >&2
    exit 3
  fi
  load_mount_table
  below="$(mounts_below "${host}.migrated-to-volume")"
  if [ -n "${below}" ] || is_mount_point "${host}.migrated-to-volume"; then
    refuse 3 "a mount lies below ${kept}; kept it" "${below}"
  fi
  if [ -z "${ROOT_PREFIX}" ]; then rm -rf --one-file-system "${kept}"; else rm -rf "${kept}"; fi
}

# --- worker units ----------------------------------------------------------------
# Worker units that were running when the move began.  Only these start again, so
# a unit an operator had stopped stays stopped.
RUNNING_UNITS=()

restart_units() {
  [ "${#RUNNING_UNITS[@]}" -gt 0 ] || return 0
  echo "restarting worker units that were running: ${RUNNING_UNITS[*]}"
  systemctl start "${RUNNING_UNITS[@]}" || true
}

# On every exit of an apply: reload systemd once /etc/fstab changed, so the old
# children's generated mount units cannot come back over the tree; then, unless a
# root is half swapped, drop staging that never went live and start again the
# units that were running.
finish() {
  if [ -z "${ROOT_PREFIX}" ] && [ -n "${FSTAB_CHANGED}" ]; then systemctl daemon-reload || true; fi
  if [ -n "${SWAPPING}" ]; then
    echo "leaving the worker units stopped: ${SWAPPING} is between its old and new mounts${FSTAB_NEXT:+, and its rewritten fstab waits at ${FSTAB_NEXT}}; finish or undo the swap by hand (docs/CONTROL_PLANE_STORAGE.md, Volume layout), then start: ${RUNNING_UNITS[*]:-nothing}" >&2
    return 0
  fi
  if [ -n "${STAGED}" ]; then rmdir -- "${STAGED}" 2>/dev/null || true; fi
  if [ -n "${FSTAB_NEXT}" ]; then rm -f -- "${FSTAB_NEXT}"; fi
  restart_units
}

stop_units() {
  local unit
  for unit in "${WORKER_UNITS[@]}"; do
    if systemctl is-active --quiet "${unit}"; then RUNNING_UNITS+=("${unit}"); fi
  done
  echo "stopping worker units for the move"
  systemctl stop "${WORKER_UNITS[@]}" || true
  require_units_stopped
}

# Checked after the stop and again right before every swap: a unit that came
# back (started by hand, or by something outside the stop list) would write into
# a tree that is about to change mounts.
require_units_stopped() {
  local unit state still=()
  for unit in "${WORKER_UNITS[@]}"; do
    state="$(systemctl is-active "${unit}" 2>/dev/null || true)"
    case "${state}" in
      inactive|failed|unknown|"") ;;
      *) still+=("${unit}:${state}") ;;
    esac
  done
  if [ "${#still[@]}" -gt 0 ]; then
    refuse 2 "worker units are running; the move needs them stopped" "${still[@]}"
  fi
}

# The operator door runs each request (a deploy, an upgrade, a scene-workspace
# retirement) as a transient blueprint-operator-door-* unit that may write under
# the roots.  Its own runner units are static and always present, so only
# transient units count.
require_no_door_requests() {
  local unit rest listed requests=()
  listed="$(systemctl list-units --plain --no-legend --state=active,activating,deactivating,reloading 'blueprint-operator-door-*' 2>/dev/null || true)"
  while read -r unit rest; do
    [ -n "${unit}" ] || continue
    if [ "$(systemctl show -p Transient --value "${unit}" 2>/dev/null || true)" = yes ]; then requests+=("${unit}"); fi
  done <<< "${listed}"
  if [ "${#requests[@]}" -gt 0 ]; then
    refuse 2 "operator door requests are running; let them finish first" "${requests[@]}"
  fi
}

apply() {
  [ "${ACK}" = "${ACK_REQUIRED}" ] || { echo "refusing: --ack ${ACK_REQUIRED} is required to move production roots" >&2; exit 2; }
  [ -n "${ROOT_PREFIX}" ] || [ "$(id -u)" = "0" ] || { echo "refusing: --apply must run as root" >&2; exit 2; }
  trap finish EXIT
  if [ -z "${ROOT_PREFIX}" ]; then
    # One run at a time: a second one could rename a root into the first one's
    # kept original.
    exec 9>/run/lock/blueprint-mount-work-volume.lock
    flock -n 9 || refuse 2 "another run holds /run/lock/blueprint-mount-work-volume.lock"
    [ -b "${DEVICE}" ] || { echo "refusing: ${DEVICE} is not a block device" >&2; exit 2; }
    # Format only a device blkid reports as blank (exit 2); any other probe
    # failure, an ambivalent result included, refuses instead.
    local probed=0
    blkid -o value -s TYPE "${DEVICE}" >/dev/null 2>&1 || probed=$?
    case "${probed}" in
      0) ;;
      2)
        echo "formatting ${DEVICE} as ext4 (no filesystem present)"
        mkfs.ext4 -F -L blueprint-work "${DEVICE}"
        ;;
      *) refuse 2 "blkid could not probe ${DEVICE} (exit ${probed}); it was not formatted" ;;
    esac
    mkdir -p "${HOST_MOUNT}"
    local uuid fstype
    uuid="$(blkid -o value -s UUID "${DEVICE}")"
    fstype="$(blkid -o value -s TYPE "${DEVICE}")"
    fstab_prepare "" "UUID=${uuid} ${MOUNT} ${fstype} defaults,nofail,noatime,discard 0 2"
    fstab_install
    mountpoint -q "${HOST_MOUNT}" || mount "${HOST_MOUNT}"
  else
    mkdir -p "${HOST_MOUNT}"
  fi

  load_mount_table
  classify_roots
  local i=0 blocked=()
  while [ "${i}" -lt "${#ROOT_HOST[@]}" ]; do
    case "${STATUS[i]}" in
      blocked) blocked+=("${ROOT_PREFIX}${ROOT_HOST[i]}: ${REASON[i]}") ;;
      move|consolidate) PENDING+=("${i}") ;;
    esac
    i=$((i + 1))
  done
  if [ "${#blocked[@]}" -gt 0 ]; then
    refuse 2 "the layout is in doubt; no root was moved" "${blocked[@]}"
  fi
  if [ ${#PENDING[@]} -eq 0 ]; then
    echo "nothing to move"
    return 0
  fi

  probe_rsync
  if [ -z "${ROOT_PREFIX}" ]; then
    require_no_door_requests
    systemctl daemon-reload
    stop_units
  fi

  echo "copying ${#PENDING[@]} roots to ${HOST_MOUNT}"
  local root dest rel excludes
  if [ -n "${RELATIVE}" ]; then
    rsync_pending_relative
  else
    for i in "${PENDING[@]}"; do
      root="${ROOT_PREFIX}${ROOT_HOST[i]}"
      dest="${HOST_MOUNT}/${ROOT_VREL[i]}"
      excludes=()
      # shellcheck disable=SC2086  # child names were validated: no whitespace or glob characters
      for rel in ${CHILDREN[i]}; do excludes+=("--exclude=/${rel}/"); done
      mkdir -p "${dest}"
      rsync "${RSYNC_FLAGS[@]}" ${excludes[@]+"${excludes[@]}"} "${root}/" "${dest}/"
    done
  fi
  echo "verifying the copy"
  local drift="" itemized one
  if [ -n "${RELATIVE}" ]; then
    itemized="$(rsync_pending_relative -n --itemize-changes)" || {
      echo "refusing to swap: the verification rsync failed" >&2
      exit 3
    }
    drift="$(printf '%s\n' "${itemized}" | grep -v -e '^\.d' -e '^$' || true)"
  else
    for i in "${PENDING[@]}"; do
      # shellcheck disable=SC2086  # child names were validated: no whitespace or glob characters
      one="$(drift_between "${ROOT_PREFIX}${ROOT_HOST[i]}" "${HOST_MOUNT}/${ROOT_VREL[i]}" ${CHILDREN[i]})" || {
        echo "refusing to swap: could not compare ${ROOT_PREFIX}${ROOT_HOST[i]} with its copy" >&2
        exit 3
      }
      drift="${drift}${one:+${one}$'\n'}"
    done
  fi
  if [ -n "${drift}" ]; then
    echo "refusing to swap: copy differs from source" >&2
    head -20 <<< "${drift}" >&2
    exit 3
  fi
  # The bind exposes the whole volume copy, and the copy never deletes, so
  # anything already there that the root lacks (an earlier run's copy of what
  # was since removed, or a store copy that is no longer bound) would go live.
  local stale=""
  for i in "${PENDING[@]}"; do
    # shellcheck disable=SC2086  # child names were validated: no whitespace or glob characters
    one="$(drift_between "${HOST_MOUNT}/${ROOT_VREL[i]}" "${ROOT_PREFIX}${ROOT_HOST[i]}" ${CHILDREN[i]})" || {
      echo "refusing to swap: could not compare ${HOST_MOUNT}/${ROOT_VREL[i]} with its root" >&2
      exit 3
    }
    if [ -n "${one}" ]; then stale="${stale}${ROOT_VREL[i]}:"$'\n'"${one}"$'\n'; fi
  done
  if [ -n "${stale}" ]; then
    echo "refusing to swap: the volume copy holds entries its root lacks; check them and move them aside, then rerun" >&2
    head -20 <<< "${stale}" >&2
    exit 3
  fi

  for i in "${PENDING[@]}"; do
    swap_root "${i}"
  done
  echo "done"
}

case "${MODE}" in
  plan) plan ;;
  apply) apply ;;
esac
