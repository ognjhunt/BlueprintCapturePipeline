#!/bin/bash
# Shared helpers for the operator door's transient-unit scripts.
#
# The root runner starts these scripts with `systemd-run`, passing only values
# it has already validated as DOOR_* environment variables. The scripts check
# the formats again, write their log and outcome next to the runner's result,
# and never read the spool themselves.

# door_init_request KIND: check the request id, then send everything to the
# request's log and make sure an outcome is written however the script ends.
door_init_request() {
  local kind="$1"
  : "${DOOR_REQUEST_ID:?}" "${DOOR_RESULTS_DIR:?}"
  # The id names the log and outcome files, so it is checked before any path use.
  if ! [[ "$DOOR_REQUEST_ID" =~ ^[0-9]{8}T[0-9]{6}Z-${kind}-[0-9a-f]{8}$ ]]; then
    echo "refused: request_id_invalid" >&2
    exit 2
  fi
  DOOR_KIND="$kind"
  DOOR_LOG="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.log"
  DOOR_OUTCOME="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.outcome.json"
  DOOR_TOOL=""
  exec >>"$DOOR_LOG" 2>&1
  trap 'door_on_exit $?' EXIT
}

# door_init_git: the commit and source settings of a kind that fetches and runs a commit.
door_init_git() {
  : "${DOOR_COMMIT:?}" "${DOOR_SOURCE_CLONE:?}" "${DOOR_UPSTREAM_URL:?}" "${DOOR_VENV_PYTHON:?}"
  [[ "$DOOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || door_fail commit_invalid
  echo "[$(date -u +%FT%TZ)] ${DOOR_KIND} ${DOOR_COMMIT} (request ${DOOR_REQUEST_ID})"
}

# door_phase NAME: a timestamp in the request log. The deploy tool's receipt times
# its own stages; this measures what the door does before the tool starts.
door_phase() {
  echo "[$(date -u +%FT%TZ)] phase=$1"
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
  door_settle_older_parked "" "$rc"
  if [ -z "${DOOR_OUTCOME_WRITTEN:-}" ] && [ -n "${DOOR_OUTCOME:-}" ]; then
    door_outcome failed "script_exited_early" "$rc"
  fi
}

# door_outcome_for ID STATUS CODE EXIT_CODE [KEY VALUE]...: an outcome for another
# request this deploy replaced, leaving this request's own outcome state alone.
door_outcome_for() {
  local request_id="$1" own="$DOOR_OUTCOME" own_written="${DOOR_OUTCOME_WRITTEN:-}"
  shift
  DOOR_OUTCOME="$DOOR_RESULTS_DIR/$request_id.outcome.json"
  door_outcome "$@"
  DOOR_OUTCOME="$own"
  DOOR_OUTCOME_WRITTEN="$own_written"
}

# door_settle_older_parked [CONTAINER] [RC]: settle each older deploy parked behind
# this one, exactly once. With a container commit, a request it contains is
# superseded and any other fails, because deploying it later would drop this one.
# With none (the script is exiting before it could tell), they all fail.
door_settle_older_parked() {
  local container="${1:-}" rc="${2:-0}" entry older_id older_commit by
  [ -n "${DOOR_PARKED_OLDER:-}" ] && [ -z "${DOOR_PARKED_SETTLED:-}" ] || return 0
  DOOR_PARKED_SETTLED=1
  local entries=()
  IFS=, read -r -a entries <<<"$DOOR_PARKED_OLDER"
  for entry in ${entries[@]+"${entries[@]}"}; do
    older_id="${entry%%=*}"
    older_commit="${entry#*=}"
    [[ "$older_id" =~ ^[0-9]{8}T[0-9]{6}Z-deploy-[0-9a-f]{8}$ && "$older_commit" =~ ^[0-9a-f]{40}$ ]] || continue
    if [ -z "$container" ]; then
      door_outcome_for "$older_id" failed parked_successor_failed "$rc" newest_request "$DOOR_REQUEST_ID"
    elif ! git -C "$DOOR_SOURCE_CLONE" merge-base --is-ancestor "$older_commit" "$container" 2>/dev/null; then
      door_outcome_for "$older_id" failed parked_commit_not_contained 1 newest_commit "$container"
    elif [ "$container" = "$DOOR_COMMIT" ]; then
      door_outcome_for "$older_id" superseded "" 0 superseded_by "$DOOR_REQUEST_ID" superseded_by_commit "$container"
    else
      door_outcome_for "$older_id" superseded "" 0 superseded_by_commit "$container"
    fi
  done
}

# A parked request waited behind another deploy, possibly of a newer commit. If what
# is now running already contains it, deploying it would roll production back. A
# deploy requested directly keeps that power: an explicit rollback is a real action.
door_settle_parked() {
  [ -n "${DOOR_PARKED:-}" ] || return 0
  local active
  active="$(basename "$(readlink -f "${DOOR_ACTIVE_RELEASE_LINK:?}")")"
  if [[ "$active" =~ ^[0-9a-f]{40}$ ]] && git -C "$DOOR_SOURCE_CLONE" merge-base --is-ancestor "$DOOR_COMMIT" "$active" 2>/dev/null; then
    echo "not deploying: the active release $active already contains $DOOR_COMMIT"
    door_settle_older_parked "$active"
    door_outcome superseded "" 0 superseded_by_commit "$active"
    exit 0
  fi
  door_settle_older_parked "$DOOR_COMMIT"
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
  # Only main decides what the door may deploy, and nothing on the door's path
  # reads another remote branch: this script and the tool never pass --canary.
  door_phase fetch_main_start
  git -C "$src" fetch --quiet origin '+refs/heads/main:refs/remotes/origin/main' || door_fail fetch_failed
  door_phase fetch_main_done
  door_forget_other_remote_branches "$src"
  if ! git -C "$src" cat-file -e "${DOOR_COMMIT}^{commit}" 2>/dev/null; then
    # A pushed commit that main does not contain still refuses as commit_not_on_main.
    git -C "$src" fetch --quiet origin "$DOOR_COMMIT" 2>/dev/null || true
  fi
  git -C "$src" cat-file -e "${DOOR_COMMIT}^{commit}" 2>/dev/null || door_fail commit_not_found
}

# A narrow refspec cannot prune what an older door fetched. Those tracking refs
# only slow every `for-each-ref --contains` the release staging runs.
door_forget_other_remote_branches() {
  local stale
  stale="$(git -C "$1" for-each-ref --format='delete %(refname)' refs/remotes/origin \
    | { grep -vE '^delete refs/remotes/origin/(main|HEAD)$' || true; })"
  if [ -n "$stale" ]; then
    printf '%s\n' "$stale" | git -C "$1" update-ref --stdin || echo "keeping stale remote branches: update-ref failed"
  fi
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
