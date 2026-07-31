#!/usr/bin/env bash
# PRIMARY build path for the sealed blueprint-groot-oscar-eval image: crane-snapshot
# an already-provisioned, healthy GR00T x OSCAR closed-loop pod. This freezes
# tonight's proven environment (no re-derivation of the dependency archaeology)
# by streaming an allow-listed filesystem layer onto the pod's base image via
# `crane append` (google/go-containerregistry) — the house method for 40GB+
# images (see the global CLAUDE.md "Docker Snapshot Strategy").
#
# Secret hygiene: the Docker PAT is piped to `crane auth login` on the POD via
# its own stdin channel (never argv, never a layer, never the manifest). The HF
# token is not needed here (weights are already resident on the pod).
#
# This script does NOT launch or terminate pods and does not change spend logic;
# it operates on a pod you already own. Run scripts/gpu_spend_guard.py yourself
# before AND after, and keep the standard pending_teardown discipline.
#
# Hermetic dry run (no pod, no network):
#   ./scripts/snapshot_groot_oscar_eval_pod.sh --print-plan
#
# Real snapshot (needs explicit go):
#   BLUEPRINT_ALLOW_GROOT_OSCAR_SNAPSHOT=true \
#   BLUEPRINT_GROOT_OSCAR_SNAPSHOT_SSH="root@<host> -p <port> -i $HOME/.ssh/id_ed25519" \
#   BLUEPRINT_GROOT_OSCAR_SNAPSHOT_BASE_IMAGE="docker.io/runpod/pytorch:<tag>" \
#   BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF="docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64" \
#   BLUEPRINT_DOCKER_PAT_FILE="$HOME/.blueprint-secrets/docker_pat" \
#   ./scripts/snapshot_groot_oscar_eval_pod.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/artifact_storage.sh"
artifact_cache_root="$(blueprint_artifact_cache_root)"

# Hermetic dry run: emit the snapshot layer plan (from the tested module).
if [[ "${1:-}" == "--print-plan" ]]; then
  PYTHONPATH="$repo_root/src:$repo_root:${PYTHONPATH:-}" \
    python3 -m blueprint_pipeline.groot_oscar_closed_loop_image --print-snapshot-plan
  exit $?
fi

ssh_target="${BLUEPRINT_GROOT_OSCAR_SNAPSHOT_SSH:-}"
base_image="${BLUEPRINT_GROOT_OSCAR_SNAPSHOT_BASE_IMAGE:-}"
image_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF:-}"
docker_pat_file="${BLUEPRINT_DOCKER_PAT_FILE:-$HOME/.blueprint-secrets/docker_pat}"
docker_user="${BLUEPRINT_DOCKER_USERNAME:-$(cat "$HOME/.blueprint-secrets/docker_username" 2>/dev/null || echo nijelhunt)}"
manifest_output="${BLUEPRINT_GROOT_OSCAR_SNAPSHOT_MANIFEST_OUTPUT:-$artifact_cache_root/groot_oscar_snapshot_manifest.json}"
trim_torch="${BLUEPRINT_GROOT_OSCAR_SNAPSHOT_TRIM_TORCH:-false}"

die() { echo "snapshot blocked: $*" >&2; exit 2; }

[[ "${BLUEPRINT_ALLOW_GROOT_OSCAR_SNAPSHOT:-false}" == "true" ]] \
  || die "set BLUEPRINT_ALLOW_GROOT_OSCAR_SNAPSHOT=true after gpu_spend_guard + explicit go"
[[ -n "$ssh_target" ]] || die "missing BLUEPRINT_GROOT_OSCAR_SNAPSHOT_SSH"
[[ -n "$base_image" ]] || die "missing BLUEPRINT_GROOT_OSCAR_SNAPSHOT_BASE_IMAGE (the pod's base image ref)"
[[ -n "$image_ref" ]] || die "missing BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF (target)"
[[ -f "$docker_pat_file" ]] || die "missing docker PAT file: $docker_pat_file"
if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]; then
  die "target image ref must be versioned: $image_ref"
fi
case "$image_ref" in *:latest|*:local|*:dev|*:test) die "refuses unstable tag: $image_ref";; esac

# shellcheck disable=SC2206
ssh_cmd=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=8 $ssh_target)

# --- Step 1: prep the pod (idempotent) + discover site-packages + install crane ---
echo "[snapshot] step 1/3 prep: relocate checkpoints, stamp sealed env, install crane"
prep_out="$("${ssh_cmd[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p /opt/blueprint/ckpts
for pair in "/workspace/sonic_ckpt:/opt/blueprint/ckpts/sonic" "/workspace/oscar_ckpt:/opt/blueprint/ckpts/oscar"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if [[ -d "$src" && ! -d "$dst" ]]; then cp -a "$src" "$dst"; fi
done
# groot-bs16 ships 8 training-intermediate checkpoint-*/ dirs (~52GB) the server
# never loads; drop them so the sealed image stays lean (~6.5GB inference model).
rm -rf /opt/blueprint/ckpts/sonic/checkpoint-* 2>/dev/null || true
cat > /opt/blueprint/groot_oscar_sealed_env.sh <<'ENV'
export MUJOCO_GL=osmesa
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/opt/OSCAR:${PYTHONPATH:-}
export BLUEPRINT_OSCAR_WAM_HF_REVISION=c9781ffa7dd8556d862d7d9f338a2ea008a58ca6
export BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR
export BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT=/opt/blueprint/ckpts/oscar
export BLUEPRINT_GROOT_OSCAR_GROOT_ROOT=/opt/gr00t
export BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT=/opt/blueprint/ckpts/sonic
export GEAR_SONIC_CHECKPOINT_REPO=nvidia/GEAR-SONIC
export GEAR_SONIC_CHECKPOINT_REVISION=5e22ddc69abcea2a9aafc40536b14c232d3f9d7f
export BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true
ENV
# If blueprint_pipeline is an EDITABLE install (pip install -e <src>), its source
# lives outside the captured /opt/* trees (often /workspace/src) and would be lost
# in the snapshot. Reinstall it non-editable into site-packages (captured) from
# whichever source dir exists, so the sealed image self-contains our package.
for src in /workspace /opt/blueprint/blueprint-capture-pipeline; do
  if [[ -f "$src/pyproject.toml" ]]; then
    python3 -m pip install --no-deps --force-reinstall "$src" >/dev/null 2>&1 \
      && echo "BLUEPRINT_PKG_SEALED_FROM=$src" && break
  fi
done
SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
[[ -n "$SITE" ]] || SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "SITE_PACKAGES=$SITE"
if ! command -v crane >/dev/null 2>&1; then
  curl -sL "https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xzf - -C /usr/local/bin crane
fi
crane version >/dev/null && echo "CRANE_OK"
for p in /opt/OSCAR /opt/gr00t /opt/gr00t-venv /opt/blueprint/ckpts/sonic /opt/blueprint/ckpts/oscar; do
  [[ -e "$p" ]] || { echo "MISSING_BAKED_PATH=$p" >&2; exit 7; }
done
echo "PREP_OK"
REMOTE
)"
echo "$prep_out"
grep -q PREP_OK <<<"$prep_out" || die "pod prep failed (see above)"
site_packages="$(sed -n 's/^SITE_PACKAGES=//p' <<<"$prep_out" | head -1)"
[[ -n "$site_packages" ]] || die "could not resolve pod site-packages"
site_rel="${site_packages#/}"

# --- Step 2: authenticate crane on the pod (PAT via its own stdin channel) ---
echo "[snapshot] step 2/3 crane auth login (PAT via stdin; never argv)"
if ! "${ssh_cmd[@]}" "crane auth login docker.io -u '$docker_user' --password-stdin" < "$docker_pat_file"; then
  die "crane auth login failed"
fi

# --- Step 3: stream the allow-listed layer, append onto base, push ---
echo "[snapshot] step 3/3 tar(/opt + $site_packages + /usr/local/bin) | crane append --base $base_image -t $image_ref"
# Pass only space-free values over SSH (image refs, site path, trim flag) and
# build the tar --exclude array ON THE REMOTE — passing a space-containing
# EXCLUDE_FLAGS string over ssh word-splits it and a `--exclude=` token then
# runs as a command.
remote_push="$("${ssh_cmd[@]}" \
  BASE_IMAGE="$base_image" TARGET_IMAGE="$image_ref" SITE_REL="$site_rel" TRIM_TORCH="$trim_torch" \
  'bash -s' <<'REMOTE'
set -eo pipefail
FIFO=/tmp/groot_oscar_layer.tar
rm -f "$FIFO"; mkfifo "$FIFO"
EXC=()
if [ "${TRIM_TORCH:-false}" = "true" ]; then
  for d in torch torchvision torchgen functorch nvidia triton; do
    EXC+=( "--exclude=${SITE_REL}/${d}" "--exclude=${SITE_REL}/${d}-*" "--exclude=${SITE_REL}/${d}_*" )
  done
fi
( tar cf "$FIFO" "${EXC[@]}" -C / \
    opt/OSCAR opt/gr00t opt/gr00t-venv opt/wbc opt/blueprint \
    "$SITE_REL" usr/local/bin 2>/dev/null & )
crane append --base "$BASE_IMAGE" -f "$FIFO" -t "$TARGET_IMAGE"
rm -f "$FIFO"
crane logout docker.io >/dev/null 2>&1 || true
echo "SNAPSHOT_PUSHED=$TARGET_IMAGE"
REMOTE
)"
echo "$remote_push"
grep -q "SNAPSHOT_PUSHED=$image_ref" <<<"$remote_push" || die "crane append/push did not confirm"

python3 - "$manifest_output" "$image_ref" "$base_image" "$site_packages" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out = Path(sys.argv[1]).expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schema_version": "groot_oscar_closed_loop_snapshot_manifest.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "pushed",
    "image_ref": sys.argv[2],
    "base_image": sys.argv[3],
    "snapshot_site_packages": sys.argv[4],
    "build_method": "crane_append_pod_snapshot",
    "raw_secret_values_recorded": False,
    "claim_boundary": {
        "snapshot_is_not_provider_startup": True,
        "snapshot_is_not_policy_inference": True,
        "snapshot_is_not_task_success": True,
        "verify_with_fresh_pull_healthcheck_before_trust": True,
    },
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY

cat <<EOF

[snapshot] pushed: $image_ref
NEXT (do NOT trust until verified):
  1) On a fresh pod:  docker pull $image_ref
  2) python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --require-cuda
  3) 1-step closed-loop smoke: source /opt/blueprint/groot_oscar_sealed_env.sh; then the
     plan from  python3 -m blueprint_pipeline.groot_oscar_closed_loop_image --print-launch-plan --steps 1
  4) only then write the ref to ~/.blueprint-secrets/groot_oscar_closed_loop_image_ref (keep .bak)
EOF
