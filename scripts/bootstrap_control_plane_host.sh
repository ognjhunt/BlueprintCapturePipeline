#!/usr/bin/env bash
set -euo pipefail

# Bring a bare Ubuntu host to the point where the Blueprint control plane can
# be installed, then prove the result.
#
# This exists because a rebuilt host is not usable after `pip install -e .`:
#
#   * the canonical allocator reaches `cv2` transitively through the excision
#     audit, and `cv2` lives in the `runtime` extra; without it the only
#     provider-mutation entrypoint cannot be imported at all; and
#   * installing that extra pulls a non-headless OpenCV whose `cv2` needs
#     `libGL.so.1`, which a server image does not carry.
#
# Both were found the expensive way: the host served traffic and passed its
# environment checks while the allocator was unimportable, and it surfaced only
# when a stranded provider record could not be released. The runtime guard now
# detects that state, but detection is not installation, so the steps live here.
#
# Idempotent. Safe to re-run against an already provisioned host.
#
# Usage:
#   scripts/bootstrap_control_plane_host.sh [--repo-root DIR] [--skip-apt]
#
# After this, run scripts/install_live_pipeline_control_plane.sh to install the
# systemd units, the environment file, and the Caddy edge.

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SKIP_APT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --skip-apt) SKIP_APT=true; shift ;;
    -h|--help) sed -n '3,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

VENV="${VENV:-${REPO_ROOT}/.venv}"

if [[ "${SKIP_APT}" != "true" ]]; then
  if [[ "${EUID}" -ne 0 ]]; then
    echo "apt steps need root; re-run as root or pass --skip-apt" >&2
    exit 1
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # git/python build the checkout and venv; libgl1 and libglib2.0-0 are what
  # OpenCV needs on a headless server image. docker.io is what the Content
  # Agents config preflight shells out to: that preflight is a
  # network-disabled dry run of the agent configs, and it must run where the
  # bundle it describes is going to run. Without it the lane can be staged,
  # profiled, and published, and is then refused at the paid boundary for
  # evidence that only exists on somebody's workstation. docker-buildx is the
  # canonical immutable worker-image publisher used by the SAM 3.1 lane; keep
  # it in the host bootstrap so image publication is not a remembered patch.
  apt-get install -y -qq \
    git \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    ca-certificates \
    docker.io \
    docker-buildx \
    libgl1 \
    libglib2.0-0
  systemctl enable --now docker >/dev/null 2>&1 || true
  # Caddy terminates TLS for the public hostname. The intake service binds
  # 127.0.0.1 only, so without an edge the control plane is unreachable.
  if ! command -v caddy >/dev/null 2>&1; then
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https gnupg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
      | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
      > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq
    apt-get install -y -qq caddy
  fi
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q --upgrade pip setuptools wheel

# The `runtime` extra is not optional for a control-plane host: the canonical
# allocator cannot be imported without it.
"${VENV}/bin/pip" install -q -e "${REPO_ROOT}[runtime]"

# Installing is not proof. The guard imports every control-plane entrypoint, so
# a missing dependency fails the rebuild here rather than during the first paid
# operation. Only the entrypoint result is asserted; the production fail-closed
# flags belong to the deployed environment file, not to bootstrap.
echo "verifying control-plane entrypoints..."
PYTHONPATH="${REPO_ROOT}/src" "${VENV}/bin/python" - <<'PY'
import sys

from blueprint_pipeline.production_runtime_env_guard import (
    build_production_runtime_env_guard,
)

report = build_production_runtime_env_guard({})
detail = report["control_plane_entrypoints"]
for failure in detail["failed"]:
    print(f"  UNIMPORTABLE {failure['module']}: {failure['error']}", file=sys.stderr)
if detail["failed"]:
    print(detail["remediation"], file=sys.stderr)
    raise SystemExit(1)
print(f"  all {len(detail['checked'])} control-plane entrypoints import")

# The STEP extractor imports build123d lazily only after it has verified the
# mesh packet. Importing allocator entrypoints cannot prove this dependency,
# which is why a production host passed bootstrap and then failed at the first
# fresh-scene extraction. Probe the exact callable the materializer uses.
try:
    from build123d import import_step
except Exception as exc:
    print(f"  UNIMPORTABLE build123d.import_step: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
if not callable(import_step):
    print("  UNUSABLE build123d.import_step: not callable", file=sys.stderr)
    raise SystemExit(1)
print("  build123d.import_step available")
PY

echo "host bootstrap complete"
echo "next: scripts/install_live_pipeline_control_plane.sh"
