#!/bin/bash
# Shared helpers for the operator door's transient-unit scripts.
#
# The root runner starts these scripts with `systemd-run`, passing only values
# it has already validated as DOOR_* environment variables. The scripts check
# the formats again, write their log and outcome next to the runner's result,
# and never read the spool themselves.

door_init() {
  local kind="$1"
  : "${DOOR_REQUEST_ID:?}" "${DOOR_RESULTS_DIR:?}"
  # The id names the log and outcome files, so it is checked before any path use.
  if ! [[ "$DOOR_REQUEST_ID" =~ ^[0-9]{8}T[0-9]{6}Z-${kind}-[0-9a-f]{8}$ ]]; then
    echo "refused: request_id_invalid" >&2
    exit 2
  fi
  DOOR_LOG="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.log"
  DOOR_OUTCOME="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.outcome.json"
  DOOR_TOOL=""
  exec >>"$DOOR_LOG" 2>&1
  trap 'door_on_exit $?' EXIT
  : "${DOOR_COMMIT:?}" "${DOOR_SOURCE_CLONE:?}" "${DOOR_UPSTREAM_URL:?}" "${DOOR_VENV_PYTHON:?}"
  [[ "$DOOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || door_fail commit_invalid
  echo "[$(date -u +%FT%TZ)] ${kind} ${DOOR_COMMIT} (request ${DOOR_REQUEST_ID})"
}

# door_outcome STATUS CODE EXIT_CODE [KEY VALUE]... -> results/<id>.outcome.json
door_outcome() {
  local status="$1" code="$2" rc="$3"
  shift 3
  python3 - "$DOOR_OUTCOME" "$status" "$code" "$rc" "$@" <<'PY'
import datetime, json, os, sys
path, status, code, rc, *pairs = sys.argv[1:]
document = {
    "schema": "blueprint_operator_door_outcome.v1",
    "status": status,
    "code": code or None,
    "exit_code": int(rc),
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
document.update(dict(zip(pairs[0::2], pairs[1::2])))
tmp = "%s.tmp.%d" % (path, os.getpid())
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(document, stream, sort_keys=True)
os.replace(tmp, path)
PY
  DOOR_OUTCOME_WRITTEN=1
}

door_fail() {
  echo "refused: $1"
  if [ -n "${DOOR_OUTCOME:-}" ]; then
    door_outcome refused "$1" 2
  fi
  exit 2
}

door_on_exit() {
  local rc="$1"
  door_remove_tool
  if [ -z "${DOOR_OUTCOME_WRITTEN:-}" ] && [ -n "${DOOR_OUTCOME:-}" ]; then
    door_outcome failed "script_exited_early" "$rc"
  fi
}

# Keep one long-lived source clone whose origin is GitHub, so every deploy's
# release worktree has a stable parent and pushed-commit checks see real refs.
#
# The repository is private and the host has no other GitHub credential, so a
# GitHub origin is fetched over SSH with the door's read-only deploy key. The
# settings live in the source clone's own config, where the deploy tool's
# `git fetch origin main` finds them too; origin itself stays the HTTPS URL.
door_git_auth() {
  DOOR_GIT_SSH=""
  case "$DOOR_UPSTREAM_URL" in
    https://github.com/*) ;;
    *) return 0 ;;
  esac
  [ -r "${DOOR_GITHUB_KEY:-}" ] || door_fail github_deploy_key_missing
  [ -s "${DOOR_GITHUB_KNOWN_HOSTS:-}" ] || door_fail github_known_hosts_missing
  DOOR_GIT_SSH="ssh -i $DOOR_GITHUB_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$DOOR_GITHUB_KNOWN_HOSTS"
}

door_prepare_source() {
  local src="$DOOR_SOURCE_CLONE"
  local auth=()
  door_git_auth
  if [ -n "$DOOR_GIT_SSH" ]; then
    auth=(--config "core.sshCommand=$DOOR_GIT_SSH" --config "url.git@github.com:.insteadOf=https://github.com/")
  fi
  if [ ! -d "$src/.git" ]; then
    echo "creating source clone $src"
    mkdir -p "$(dirname "$src")"
    if [ -d "${DOOR_REFERENCE_REPO:-}/.git" ]; then
      git clone --quiet --no-checkout ${auth[@]+"${auth[@]}"} --reference "$DOOR_REFERENCE_REPO" --dissociate \
        "$DOOR_UPSTREAM_URL" "$src" || door_fail clone_failed
    else
      git clone --quiet --no-checkout ${auth[@]+"${auth[@]}"} "$DOOR_UPSTREAM_URL" "$src" || door_fail clone_failed
    fi
    # Nothing runs from a brand-new clone yet, so checking it out here is safe.
    git -C "$src" checkout --quiet --detach "$DOOR_COMMIT" || door_fail commit_not_found
  fi
  git -C "$src" remote set-url origin "$DOOR_UPSTREAM_URL"
  if [ -n "$DOOR_GIT_SSH" ]; then
    git -C "$src" config core.sshCommand "$DOOR_GIT_SSH"
    git -C "$src" config url.git@github.com:.insteadOf https://github.com/
  fi
  git -C "$src" fetch --quiet --prune origin '+refs/heads/*:refs/remotes/origin/*' || door_fail fetch_failed
  git -C "$src" cat-file -e "${DOOR_COMMIT}^{commit}" 2>/dev/null || door_fail commit_not_found
}

door_require_on_main() {
  git -C "$DOOR_SOURCE_CLONE" merge-base --is-ancestor "$DOOR_COMMIT" origin/main || door_fail commit_not_on_main
}

# A throwaway worktree at the target commit, so scripts run from the new code
# while the source clone itself is moved only by the deploy tool.
door_add_tool() {
  local parent="${DOOR_TOOL_PARENT:-$(dirname "$DOOR_SOURCE_CLONE")}"
  DOOR_TOOL="$(mktemp -d "$parent/operator-door-tool.XXXXXX")"
  rmdir "$DOOR_TOOL"
  git -C "$DOOR_SOURCE_CLONE" worktree add --quiet --detach "$DOOR_TOOL" "$DOOR_COMMIT" || door_fail tool_worktree_failed
}

door_remove_tool() {
  if [ -n "${DOOR_TOOL:-}" ]; then
    git -C "$DOOR_SOURCE_CLONE" worktree remove --force "$DOOR_TOOL" >/dev/null 2>&1 || rm -rf "$DOOR_TOOL"
    DOOR_TOOL=""
  fi
}
