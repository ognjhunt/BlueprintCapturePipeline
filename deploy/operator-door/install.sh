#!/bin/bash
# Install or upgrade the operator door on the control-plane host (run as root).
#
#   bash deploy/operator-door/install.sh [--upgrade] [--no-caddy]
#
# Installs the door code to /opt/blueprint/operator-door (independent of pipeline
# releases), its three units, its state directories and an empty token store
# (an existing store is never replaced), then adds one Caddy route for
# /api/live-pipeline/operator/* to the live Caddyfile.
#
# Any failure after the code swap restores the previous code and unit files and
# restarts the previous door (or disables a first install), so a bad install or
# upgrade cannot leave a half-working door behind.
set -eEuo pipefail
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
units_backup="$install_root.previous-units"

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
cp "$source_dir"/door-common.sh "$source_dir"/door-deploy.sh "$source_dir"/door-upgrade.sh \
  "$source_dir"/install.sh "$stage/"
git -C "$repo_root" rev-parse HEAD >"$stage/INSTALLED_COMMIT" 2>/dev/null || echo unknown >"$stage/INSTALLED_COMMIT"
find "$stage" -name '__pycache__' -prune -exec rm -rf {} +
chown -R root:root "$stage"
chmod -R u=rwX,go=rX "$stage"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$stage" python3 -c 'import operator_door.server, operator_door.spool_runner'

had_previous=0
[ -d "$install_root" ] && had_previous=1
swapped=0

rollback() {
  trap - ERR
  echo "operator door install failed; rolling back" >&2
  if [ "$swapped" -eq 1 ]; then
    rm -rf "$install_root.failed"
    mv "$install_root" "$install_root.failed" 2>/dev/null || true
    if [ -d "$install_root.previous" ]; then mv "$install_root.previous" "$install_root"; fi
  fi
  for unit in "${units[@]}"; do
    if [ -f "$units_backup/$unit" ]; then
      install -o root -g root -m 0644 "$units_backup/$unit" "$systemd_dir/$unit"
    elif [ "$had_previous" -eq 0 ]; then
      rm -f "$systemd_dir/$unit"
    fi
  done
  systemctl daemon-reload || true
  if [ "$had_previous" -eq 1 ]; then
    systemctl restart blueprint-operator-door.service || true
  else
    systemctl disable --now blueprint-operator-door.service blueprint-operator-door-runner.path 2>/dev/null || true
  fi
}
trap 'rollback; exit 1' ERR

# 2. Back up the current units, then swap code, keeping the previous version.
rm -rf "$units_backup"
mkdir -p "$units_backup"
for unit in "${units[@]}"; do
  if [ -f "$systemd_dir/$unit" ]; then cp -p "$systemd_dir/$unit" "$units_backup/$unit"; fi
done
if [ "$had_previous" -eq 1 ]; then
  rm -rf "$install_root.previous"
  mv "$install_root" "$install_root.previous"
fi
mv "$stage" "$install_root"
swapped=1

# 3. State and config. Only pending/ is writable by the door (through a group only
#    the door unit has); everything the root runner and scripts write is root's,
#    so no service-account process can plant a symlink where root writes.
getent group blueprint-door >/dev/null || groupadd --system blueprint-door
install -d -o root -g root -m 0755 "$state_root" "$state_root/requests"
install -d -o root -g blueprint-door -m 2770 "$state_root/requests/pending"
for sub in processing completed results; do
  install -d -o root -g root -m 0755 "$state_root/requests/$sub"
done
install -d -o blueprint -g blueprint -m 0750 "$state_root/audit"
install -d -o root -g blueprint -m 2750 "$config_dir"
if [ ! -e "$config_dir/tokens.json" ]; then
  printf '{"schema": "blueprint_operator_door_tokens.v1", "tokens": []}\n' >"$config_dir/tokens.json"
fi
chown root:blueprint "$config_dir/tokens.json"
chmod 0640 "$config_dir/tokens.json"

# 4. Units.
for unit in "${units[@]}"; do
  install -o root -g root -m 0644 "$units_dir/$unit" "$systemd_dir/$unit"
done
systemctl daemon-reload
systemctl enable --now blueprint-operator-door-runner.path
systemctl enable blueprint-operator-door.service
systemctl restart blueprint-operator-door.service

# 5. Health: the listener answers, and as the service account the door can read
#    its tokens and spool (a root-only token file would lock every caller out).
healthy=0
for _ in $(seq 1 40); do
  if curl -fsS --max-time 2 "$health_url" >/dev/null 2>&1; then healthy=1; break; fi
  sleep 0.5
done
if [ "$healthy" -ne 1 ]; then echo "operator door does not answer $health_url" >&2; false; fi
runuser -u blueprint -- env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$install_root" \
  python3 -m operator_door self-test --allow-no-tokens
trap - ERR

# 6. Caddy route, patched into the live file (the host's copy differs from the
#    repository's). A failure here restores the Caddyfile; the door itself stays.
if [ "$caddy" -eq 1 ] && ! grep -q 'handle /api/live-pipeline/operator/\*' "$caddyfile"; then
  backup="$caddyfile.bak-operator-door-$(date -u +%Y%m%dT%H%M%SZ)"
  cp -p "$caddyfile" "$backup"
  candidate="$(mktemp)"
  if ! PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$install_root" python3 -m operator_door caddy-patch "$caddyfile" "$candidate" \
    || ! caddy validate --adapter caddyfile --config "$candidate" >/dev/null; then
    rm -f "$candidate"
    echo "the patched Caddyfile did not validate; the live file is untouched" >&2
    exit 1
  fi
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
