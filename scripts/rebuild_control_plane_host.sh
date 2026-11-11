#!/usr/bin/env bash
set -euo pipefail

# Take a fresh Ubuntu host to a working control plane from this repository, the state backup and the
# secrets the owner supplies. Idempotent: every step can run again on a host it already rebuilt.
#
#   1. Preflight: every owner secret in deploy/host/rebuild-owner-secrets.txt is present, in
#      --secrets-from or already under /etc/blueprint. A missing one stops the rebuild and is named.
#      Nothing is ever invented for a secret, a money limit or an approval; --waive-secret NAME lets the
#      owner go on without one, and says so (the units that read it then fail until it is supplied).
#   2. The service account (as the installer creates it).
#   3. scripts/bootstrap_control_plane_host.sh: packages, Caddy, the venv, the import proof.
#   4. The work volume. On a fresh host the bulk roots do not exist, and mount_work_volume.sh binds only
#      roots that exist (a missing root is "nothing to move"), so the rebuild first creates each missing
#      root empty, with its owner from control_plane_storage_roots (mode 0750, group blueprint), then runs
#      deploy/host/mount_work_volume.sh --device DEV --apply, which binds them. It then checks that every
#      bulk root is a mount point on the work volume and refuses to go on otherwise, so nothing below
#      writes into the root disk (--verify-volume runs that check alone).
#   5. The owner's secrets into /etc/blueprint (0640 root:blueprint; the backup key and bucket 0600 root).
#   6. The state restore: python -m blueprint_pipeline.control_plane_state_backup restore. It verifies every
#      file, restores the standing approvals exactly as they were, and writes the money hold that keeps
#      paid admission closed until the spend guard's reap and the billing reconciler have caught up (it
#      clears by itself). A generation already restored is not restored again.
#   7. scripts/install_live_pipeline_control_plane.sh: units, directories, the Caddy edge.
#   8. deploy/operator-door/install.sh: the door (its deploy key is new: the owner registers it).
#   9. The active release: the normal deploy tool (scripts/deploy_control_plane_commit.py) deploys this
#      checkout's commit with the operator door's own arguments. It is what publishes
#      /opt/blueprint/task-evaluation-control-plane, which every unit runs from, and what installs and
#      arms the deployed units. Nothing here publishes a release any other way.
#  10. The units that were enabled when the backup was taken (from its enabled_units.json), after the
#      release exists.
#
# --dry-run checks the preflight and prints every step; it changes nothing and needs no root, so CI runs
# it (tests/test_rebuild_control_plane_host.py). --root DIR reads and would write under DIR instead of /.
#
# Usage:
#   scripts/rebuild_control_plane_host.sh [--dry-run] [--root DIR] --secrets-from DIR \
#       --work-volume-device /dev/disk/by-id/<volume> [--generation latest|<id>] [--waive-secret NAME]...

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROOT="/"
DRY_RUN=false
SECRETS_FROM=""
DEVICE=""
GENERATION="latest"
WAIVED=()
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --root) ROOT="$2"; shift 2 ;;
    --secrets-from) SECRETS_FROM="$2"; shift 2 ;;
    --work-volume-device) DEVICE="$2"; shift 2 ;;
    --generation) GENERATION="$2"; shift 2 ;;
    --waive-secret) WAIVED+=("$2"); shift 2 ;;
    --verify-volume) VERIFY_ONLY=true; shift ;;
    -h|--help) sed -n '3,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="${ROOT%/}"
ETC="${ROOT}/etc/blueprint"
SECRETS_LIST="${REPO_ROOT}/deploy/host/rebuild-owner-secrets.txt"
VENV="${VENV:-${REPO_ROOT}/.venv}"
BACKUP_ETC="${ETC}/control-plane-backup"
REPORT_ROOT="${ROOT}/var/lib/blueprint/pipeline-control-plane/control-plane-backup"
MOUNT_SCRIPT="${REPO_ROOT}/deploy/host/mount_work_volume.sh"
MOUNTINFO="${BLUEPRINT_REBUILD_MOUNTINFO:-/proc/self/mountinfo}"
STATE_DIR="${ROOT}/var/lib/blueprint/pipeline-control-plane"

step() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'plan: %s\n' "$*"
  else
    printf 'run: %s\n' "$*"
    "$@"
  fi
}

waived() {
  local name="$1" item
  for item in "${WAIVED[@]+"${WAIVED[@]}"}"; do
    [[ "${item}" == "${name}" ]] && return 0
  done
  return 1
}

bulk_roots() {
  bash "${MOUNT_SCRIPT}" --list-roots
}

# Every bulk root must be its own mount point, on the same device as the work volume's mount.
verify_volume() {
  local volume="${ROOT}/mnt/blueprint-work" device="" root found unbound=()
  device="$(awk -v target="${volume}" '$5 == target { print $3 }' "${MOUNTINFO}" | tail -n 1)"
  if [[ -z "${device}" ]]; then
    echo "refusing: ${volume} is not mounted; the work volume must be mounted before anything is restored" >&2
    return 4
  fi
  while IFS= read -r root; do
    found="$(awk -v target="${ROOT}${root}" '$5 == target { print $3 }' "${MOUNTINFO}" | tail -n 1)"
    [[ "${found}" == "${device}" ]] || unbound+=("${root}")
  done < <(bulk_roots)
  if [[ "${#unbound[@]}" -gt 0 ]]; then
    for root in "${unbound[@]}"; do
      echo "not bound to the work volume: ${root}" >&2
    done
    echo "refusing: ${#unbound[@]} bulk root(s) are not on the work volume; nothing is restored onto the root disk" >&2
    return 4
  fi
  echo "work volume: every bulk root is bound to ${volume} (${device})"
}

if [[ "${VERIFY_ONLY}" == "true" ]]; then
  verify_volume
  exit $?
fi

# ---------------------------------------------------------------- 1. preflight
missing=()
[[ -n "${SECRETS_FROM}" ]] || echo "note: no --secrets-from; every owner secret must already be under ${ETC}" >&2
while IFS= read -r name; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  if [[ -e "${ETC}/${name}" ]] || [[ -n "${SECRETS_FROM}" && -e "${SECRETS_FROM}/${name}" ]]; then
    continue
  fi
  if waived "${name}"; then
    echo "waived by the owner: ${name} (the units that read it fail until it is supplied)" >&2
    continue
  fi
  missing+=("${name}")
done < "${SECRETS_LIST}"
if [[ "${#missing[@]}" -gt 0 ]]; then
  for name in "${missing[@]}"; do
    echo "missing owner secret: ${name}" >&2
  done
  echo "refusing: ${#missing[@]} owner secret(s) missing; supply them under --secrets-from (or /etc/blueprint)," \
    "or name each with --waive-secret" >&2
  exit 3
fi
if [[ -z "${DEVICE}" ]]; then
  echo "refusing: --work-volume-device is required (the owner attaches the work volume first)" >&2
  exit 3
fi
if [[ "${DRY_RUN}" != "true" && "${EUID}" -ne 0 ]]; then
  echo "refusing: a rebuild runs as root (or pass --dry-run)" >&2
  exit 2
fi
echo "preflight: every owner secret present (${#WAIVED[@]} waived)"

# ---------------------------------------------------------------- 2. the service account
if ! getent group blueprint >/dev/null 2>&1; then
  step groupadd --system blueprint
fi
if ! getent passwd blueprint >/dev/null 2>&1; then
  step useradd --system --gid blueprint --home-dir /nonexistent --shell /usr/sbin/nologin blueprint
fi

# ---------------------------------------------------------------- 3. packages and venv
step "${REPO_ROOT}/scripts/bootstrap_control_plane_host.sh" --repo-root "${REPO_ROOT}"

# ---------------------------------------------------------------- 4. the work volume
# 4a. Each missing bulk root, created empty so the volume script binds it (it skips a missing root).
mapfile -t roots < <(bulk_roots)
mapfile -t owners < <(python3 -I -c '
import sys
sys.path.insert(0, sys.argv[1])
from blueprint_pipeline.control_plane_storage_roots import classify_path
for path in sys.argv[2:]:
    root = classify_path(path)
    print(root.owner if root is not None else "blueprint")
' "${REPO_ROOT}/src" "${roots[@]}")
for index in "${!roots[@]}"; do
  [[ -e "${ROOT}${roots[index]}" ]] && continue
  step install -d -m 0750 -o "${owners[index]}" -g blueprint "${ROOT}${roots[index]}"
done
# 4b. The volume script binds them (it formats only a blank device and refuses when in doubt).
mount_args=(--device "${DEVICE}" --mount /mnt/blueprint-work --apply --ack move-work-roots-to-volume)
[[ -z "${ROOT}" ]] || mount_args+=(--root-prefix "${ROOT}")
step bash "${MOUNT_SCRIPT}" "${mount_args[@]}"
# 4c. Nothing is restored or installed until every bulk root is on the volume.
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "plan: verify every bulk root is a mount point on the work volume (refuse otherwise)"
else
  verify_volume
fi

# ---------------------------------------------------------------- 5. the owner's secrets
if [[ -n "${SECRETS_FROM}" ]]; then
  step install -d -m 0750 -o root -g blueprint "${ETC}" "${ETC}/provider-secrets" "${ETC}/credentials"
  step install -d -m 0700 -o root -g root "${BACKUP_ETC}"
  while IFS= read -r name; do
    [[ -z "${name}" || "${name}" == \#* || ! -e "${SECRETS_FROM}/${name}" ]] && continue
    if [[ -d "${SECRETS_FROM}/${name}" ]]; then
      step cp -a "${SECRETS_FROM}/${name}/." "${ETC}/${name}/"
    elif [[ "${name}" == control-plane-backup/* ]]; then
      step install -m 0600 -o root -g root "${SECRETS_FROM}/${name}" "${ETC}/${name}"
    else
      step install -m 0640 -o root -g blueprint "${SECRETS_FROM}/${name}" "${ETC}/${name}"
    fi
  done < "${SECRETS_LIST}"
fi

# ---------------------------------------------------------------- 6. the state restore
restore=(env PYTHONPATH="${REPO_ROOT}/src"
  BLUEPRINT_CONTROL_PLANE_BACKUP_ENCRYPTION_KEY_FILE="${BACKUP_ETC}/encryption_key"
  BLUEPRINT_CONTROL_PLANE_BACKUP_ACCESS_KEY_ID_FILE="${BACKUP_ETC}/b2_key_id"
  BLUEPRINT_CONTROL_PLANE_BACKUP_SECRET_ACCESS_KEY_FILE="${BACKUP_ETC}/b2_application_key"
  BLUEPRINT_CONTROL_PLANE_BACKUP_BUCKET_FILE="${BACKUP_ETC}/b2_bucket"
  BLUEPRINT_CONTROL_PLANE_BACKUP_ENDPOINT_URL_FILE="${BACKUP_ETC}/b2_s3_endpoint_url"
  BLUEPRINT_CONTROL_PLANE_BACKUP_REGION_FILE="${BACKUP_ETC}/b2_region"
  BLUEPRINT_CONTROL_PLANE_BACKUP_REPORT_ROOT="${REPORT_ROOT}"
  "${VENV}/bin/python" -m blueprint_pipeline.control_plane_state_backup restore
  --fs-root "${ROOT:-/}" --generation "${GENERATION}" --ack restore-control-plane-state)
step "${restore[@]}"

# ---------------------------------------------------------------- 7. units, directories, edge
step env SYSTEMD_DIR="${ROOT}/etc/systemd/system" ENV_DIR="${ETC}" \
  STATE_DIR="${ROOT}/var/lib/blueprint/pipeline-control-plane" HANDOFF_DIR="${ROOT}/var/lib/blueprint/pubsub-handoffs" \
  TASK_EVALUATION_INPUT_ROOT="${ROOT}/var/lib/blueprint/task-evaluation-inputs" WORK_VOLUME_ROOT="${ROOT}/mnt/blueprint-work" \
  "${REPO_ROOT}/scripts/install_live_pipeline_control_plane.sh"

# ---------------------------------------------------------------- 8. the operator door
step bash "${REPO_ROOT}/deploy/operator-door/install.sh"

# ---------------------------------------------------------------- 9. the active release
# The deploy the operator door runs (deploy/operator-door/door-deploy.sh), with its arguments, for the
# commit this checkout is at: it publishes the active release link the units run from.
commit="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${commit}" ]]; then
  echo "refusing: ${REPO_ROOT} is not a git checkout; the deploy tool needs the commit it deploys" >&2
  exit 2
fi
step env PYTHONDONTWRITEBYTECODE=1 "${VENV}/bin/python" "${REPO_ROOT}/scripts/deploy_control_plane_commit.py" \
  --source-repo "${REPO_ROOT}" --source-commit "${commit}" \
  --release-root "${ROOT}/opt/blueprint/task-evaluation-control-plane-releases" \
  --state-root "${STATE_DIR}" --active-link "${ROOT}/opt/blueprint/task-evaluation-control-plane" \
  --iteration --preserve-configured-controls-state \
  --receipt-out "${STATE_DIR}/deploy-receipts/iteration_${commit:0:12}_rebuild.json"

# ---------------------------------------------------------------- 10. the units that were enabled
units_file="${REPORT_ROOT}/enabled_units.json"
if [[ -f "${units_file}" ]]; then
  mapfile -t units < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["units"]))' "${units_file}")
  for unit in "${units[@]}"; do
    [[ -n "${unit}" && -e "${ROOT}/etc/systemd/system/${unit}" ]] || { echo "not installed, not enabled: ${unit}" >&2; continue; }
    step systemctl enable --now "${unit}"
  done
elif [[ "${DRY_RUN}" == "true" ]]; then
  echo "plan: systemctl enable --now <each unit in ${units_file}, restored in step 6>"
else
  echo "flag: the backup recorded no enabled units (${units_file} absent); enable units through the door" >&2
fi

echo "rebuild $([[ "${DRY_RUN}" == "true" ]] && echo planned || echo complete)"
echo "next: register the door's new deploy key, issue door tokens, and check 'door status'; paid admission" \
  "opens by itself once the spend guard's reap and the billing reconciler have run after the restore"
