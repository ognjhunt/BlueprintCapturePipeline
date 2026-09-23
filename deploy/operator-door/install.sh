#!/bin/bash
# Install or upgrade the operator door on the control-plane host (run as root).
#
#   bash deploy/operator-door/install.sh [--upgrade] [--no-caddy]
#
# Installs the door code to /opt/blueprint/operator-door (independent of pipeline
# releases), its three units, its state directories and an empty token store
# (an existing store is never touched), then adds one Caddy route for
# /api/live-pipeline/operator/* to the live Caddyfile. --upgrade keeps the
# previous code and rolls back if the new door fails its health check.
set -euo pipefail
umask 022

upgrade=0
caddy=1
for argument in "$@"; do
  case "$argument" in
    --upgrade) upgrade=1 ;;
    --no-caddy) caddy=0 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done

source_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$source_dir/../.." && pwd)"
units_dir="$repo_root/deploy/systemd"
install_root="${DOOR_INSTALL_ROOT:-/opt/blueprint/operator-door}"
state_root="${DOOR_STATE_ROOT_DIR:-/var/lib/blueprint-operator-door}"
config_dir="${DOOR_CONFIG_DIR:-/etc/blueprint-operator-door}"
systemd_dir="${DOOR_SYSTEMD_DIR:-/etc/systemd/system}"
caddyfile="${DOOR_CADDYFILE:-/etc/caddy/Caddyfile}"
health_url="http://127.0.0.1:8767/api/live-pipeline/operator/v1/healthz"
units=(blueprint-operator-door.service blueprint-operator-door-runner.service blueprint-operator-door-runner.path)

[ "$(id -u)" -eq 0 ] || { echo "install.sh must run as root" >&2; exit 1; }
id blueprint >/dev/null
getent group systemd-journal >/dev/null
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'
for unit in "${units[@]}"; do [ -f "$units_dir/$unit" ] || { echo "missing $units_dir/$unit" >&2; exit 1; }; done

# 1. Stage and check the code before touching the running door.
stage="$install_root.new"
rm -rf "$stage"
mkdir -p "$stage"
cp -R "$source_dir/operator_door" "$stage/"
cp "$source_dir"/door-common.sh "$source_dir"/door-deploy.sh "$source_dir"/door-replay.sh \
  "$source_dir"/door-upgrade.sh "$source_dir"/install.sh "$stage/"
git -C "$repo_root" rev-parse HEAD >"$stage/INSTALLED_COMMIT" 2>/dev/null || echo unknown >"$stage/INSTALLED_COMMIT"
find "$stage" -name '__pycache__' -prune -exec rm -rf {} +
chown -R root:root "$stage"
chmod -R u=rwX,go=rX "$stage"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$stage" python3 -c 'import operator_door.server, operator_door.runner'

# 2. Swap code, keeping the previous version for rollback.
if [ -d "$install_root" ]; then
  rm -rf "$install_root.previous"
  mv "$install_root" "$install_root.previous"
fi
mv "$stage" "$install_root"

# 3. State directories and the token store (never overwritten).
install -d -o blueprint -g blueprint -m 0750 "$state_root" "$state_root/audit"
install -d -o blueprint -g blueprint -m 2770 "$state_root/requests"
for sub in pending processing completed results; do
  install -d -o blueprint -g blueprint -m 2770 "$state_root/requests/$sub"
done
install -d -o root -g blueprint -m 0750 "$config_dir"
if [ ! -e "$config_dir/tokens.json" ]; then
  printf '{"schema": "blueprint_operator_door_tokens.v1", "tokens": []}\n' >"$config_dir/tokens.json"
  chown root:blueprint "$config_dir/tokens.json"
  chmod 0640 "$config_dir/tokens.json"
fi

# 4. Units.
for unit in "${units[@]}"; do
  install -o root -g root -m 0644 "$units_dir/$unit" "$systemd_dir/$unit"
done
systemctl daemon-reload
systemctl enable --now blueprint-operator-door-runner.path
systemctl enable blueprint-operator-door.service
systemctl restart blueprint-operator-door.service

healthy=0
for _ in $(seq 1 40); do
  if curl -fsS --max-time 2 "$health_url" >/dev/null 2>&1; then healthy=1; break; fi
  sleep 0.5
done
if [ "$healthy" -ne 1 ]; then
  echo "operator door failed its health check" >&2
  if [ "$upgrade" -eq 1 ] && [ -d "$install_root.previous" ]; then
    rm -rf "$install_root.failed"
    mv "$install_root" "$install_root.failed"
    mv "$install_root.previous" "$install_root"
    systemctl restart blueprint-operator-door.service
    echo "rolled back to the previous door" >&2
  fi
  exit 1
fi

# 5. Caddy route (patched into the live file; the host's copy differs from the repo's).
if [ "$caddy" -eq 1 ] && ! grep -q 'handle /api/live-pipeline/operator/\*' "$caddyfile"; then
  backup="$caddyfile.bak-operator-door-$(date -u +%Y%m%dT%H%M%SZ)"
  cp -p "$caddyfile" "$backup"
  candidate="$(mktemp)"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$install_root" python3 -m operator_door caddy-patch "$caddyfile" "$candidate"
  caddy validate --adapter caddyfile --config "$candidate" >/dev/null
  install -o root -g root -m 0644 "$candidate" "$caddyfile"
  rm -f "$candidate"
  if ! systemctl reload caddy || ! curl -fsS --max-time 5 http://127.0.0.1:2019/config/ | grep -q '127.0.0.1:8767'; then
    echo "caddy did not take the operator route; restoring $backup" >&2
    cp -p "$backup" "$caddyfile"
    systemctl reload caddy || true
    exit 1
  fi
fi

echo "{\"installed\": \"$(cat "$install_root/INSTALLED_COMMIT")\", \"upgrade\": $upgrade, \"caddy\": $caddy}"
