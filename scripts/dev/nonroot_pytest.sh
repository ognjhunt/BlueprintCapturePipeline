#!/usr/bin/env bash
# Run test paths as an unprivileged user in a throwaway home, the way CI runs them.
#
# Development machines here run as root; CI does not. Root ignores file modes, so a
# test that writes over a read-only sealed record passes as root and fails in CI, and
# a test that writes under the real /var/lib/blueprint succeeds as root and is refused
# in CI (both happened on 2026-11-10). This runs pytest as `nobody` (or NONROOT_USER)
# with a clean environment, a temporary HOME and TMPDIR, and no bytecode or pytest
# cache writes, so those tests fail here too.
#
# Usage: scripts/dev/nonroot_pytest.sh <test paths or node ids> [pytest options]
#   PY=<python>          interpreter (default: the cloud Pipeline venv, else python3)
#   NONROOT_USER=<user>  unprivileged user (default: nobody)
#
# Limit: the checkout still belongs to root. A test that checks a file's owner, or
# writes into the checkout itself, fails here for that reason alone. Tell those apart
# by running the same command on a clean checkout of main: a failure on both is this
# limit, not your change. CI's runner owns its checkout, so they pass there.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/opt/blueprint-cloud/pipeline-venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
if [ "$#" -eq 0 ]; then
  echo "usage: $0 <test paths or node ids> [pytest options]" >&2
  exit 2
fi
cd "$ROOT" || exit 2
if [ "$(id -u)" != 0 ]; then
  # Already unprivileged, as CI is.
  exec "$PY" -m pytest -q -p no:cacheprovider "$@"
fi

RUN_AS="${NONROOT_USER:-nobody}"
RUN_GROUP="$(id -gn "$RUN_AS")" || exit 2
if ! runuser -u "$RUN_AS" -- test -r "$ROOT/pyproject.toml" -a -x "$ROOT/tests"; then
  echo "$RUN_AS cannot read $ROOT: every directory above it needs o+x and the checkout o+rX" >&2
  exit 2
fi
SCRATCH="$(mktemp -d /tmp/nonroot-pytest-XXXXXX)" || exit 2
trap 'rm -rf -- "${SCRATCH:?}"' EXIT
mkdir -p "$SCRATCH/home" "$SCRATCH/tmp"
chown -R "$RUN_AS:$RUN_GROUP" "$SCRATCH"
chmod 700 "$SCRATCH"

runuser -u "$RUN_AS" -- env -i \
  PATH="/usr/local/bin:/usr/bin:/bin" LANG="${LANG:-C.UTF-8}" \
  HOME="$SCRATCH/home" TMPDIR="$SCRATCH/tmp" PYTHONDONTWRITEBYTECODE=1 \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*' \
  "$PY" -m pytest -q -p no:cacheprovider --basetemp="$SCRATCH/basetemp" "$@"
