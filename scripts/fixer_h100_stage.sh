#!/usr/bin/env bash
# =============================================================================
# Fixer H100 Stage Runner
# =============================================================================
# Offloads only the Fixer stage to a Vast.ai H100 instance:
#   1) provision/reuse H100
#   2) sync renders + Fixer code/weights
#   3) run Fixer remotely
#   4) sync refined images back
#
# Intended to be called from scripts/nurec_shim.py.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (can be overridden by args/env)
VASTAI_IMAGE="${FIXER_H100_VASTAI_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
MAX_HOURLY="${FIXER_H100_MAX_HOURLY:-2.50}"
DISK_GB="${FIXER_H100_DISK_GB:-80}"
CUDA_MIN="${FIXER_H100_CUDA_MIN:-12.8}"
INET_DOWN_MIN="${FIXER_H100_INET_DOWN_MIN:-200}"
FIXER_DIR_LOCAL="${FIXER_DIR:-/opt/Fixer}"
FIXER_WEIGHTS_LOCAL="${FIXER_WEIGHTS_DIR:-/opt/Fixer/weights}"
REMOTE_ROOT="${FIXER_H100_REMOTE_ROOT:-/opt/fixer_stage}"
REMOTE_SETUP_CMD="${FIXER_H100_REMOTE_SETUP_CMD:-}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30"

INPUT_RENDERS=""
OUTPUT_DIR=""
INSTANCE_ID="${FIXER_H100_INSTANCE_ID:-}"
KEEP_INSTANCE=false

CREATED_INSTANCE=false

log() {
  echo "[fixer-h100] $(date '+%H:%M:%S') $*"
}

die() {
  echo "[fixer-h100] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  fixer_h100_stage.sh --input-renders DIR --output-dir DIR [options]

Options:
  --input-renders DIR        Local renders dir from 3DGRUT (required)
  --output-dir DIR           Local output dir for refined images (required)
  --instance-id ID           Reuse existing Vast.ai instance ID
  --keep-instance            Keep instance alive if this script creates it
  --max-hourly RATE          Max $/hr when creating H100 instance (default: 2.50)
  --disk-gb GB               Disk size for created instance (default: 80)
  --fixer-dir DIR            Local Fixer source path (default: /opt/Fixer)
  --fixer-weights-dir DIR    Local Fixer weights dir (default: /opt/Fixer/weights)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --input-renders)
      INPUT_RENDERS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --keep-instance)
      KEEP_INSTANCE=true
      shift
      ;;
    --max-hourly)
      MAX_HOURLY="$2"
      shift 2
      ;;
    --disk-gb)
      DISK_GB="$2"
      shift 2
      ;;
    --fixer-dir)
      FIXER_DIR_LOCAL="$2"
      shift 2
      ;;
    --fixer-weights-dir)
      FIXER_WEIGHTS_LOCAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[ -n "$INPUT_RENDERS" ] || die "--input-renders is required"
[ -n "$OUTPUT_DIR" ] || die "--output-dir is required"
[ -d "$INPUT_RENDERS" ] || die "Input renders dir not found: $INPUT_RENDERS"

command -v vastai >/dev/null 2>&1 || die "vastai CLI not found. Install: pip install vastai"
command -v ssh >/dev/null 2>&1 || die "ssh not found"
command -v rsync >/dev/null 2>&1 || die "rsync not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

wait_for_instance() {
  local id="$1"
  local elapsed=0
  local max_wait=420
  log "Waiting for instance $id to reach running..."
  while [ "$elapsed" -lt "$max_wait" ]; do
    local status
    status=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
target = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print('unknown')
    raise SystemExit(0)
for inst in data:
    if str(inst.get('id')) == target:
        print(inst.get('actual_status') or inst.get('status') or 'unknown')
        raise SystemExit(0)
print('unknown')
" "$id")
    if [ "$status" = "running" ]; then
      log "Instance $id is running."
      return 0
    fi
    sleep 15
    elapsed=$((elapsed + 15))
    log "  status=${status} elapsed=${elapsed}s"
  done
  die "Instance $id did not become running within ${max_wait}s"
}

get_ssh_info() {
  local id="$1"
  vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
target = sys.argv[1]
data = json.load(sys.stdin)
for inst in data:
    if str(inst.get('id')) == target:
        host = inst.get('ssh_host') or ''
        port = inst.get('ssh_port') or ''
        print(f'{host} {port}')
        raise SystemExit(0)
print(' ')
" "$id"
}

wait_for_ssh() {
  local host="$1"
  local port="$2"
  local elapsed=0
  local max_wait=300
  log "Waiting for SSH ${host}:${port}..."
  while [ "$elapsed" -lt "$max_wait" ]; do
    if ssh $SSH_OPTS -p "$port" "root@$host" 'echo OK' 2>/dev/null | grep -q OK; then
      log "SSH ready."
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  die "SSH did not become ready within ${max_wait}s"
}

cleanup() {
  if [ "$CREATED_INSTANCE" = true ] && [ "$KEEP_INSTANCE" = false ] && [ -n "$INSTANCE_ID" ]; then
    log "Destroying temporary instance $INSTANCE_ID..."
    vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [ -z "$INSTANCE_ID" ]; then
  log "Searching Vast.ai offers for H100 (<= \$${MAX_HOURLY}/hr, cuda>=${CUDA_MIN})..."
  OFFER_ID=$(
    vastai search offers \
      "num_gpus=1 rentable=true cuda_vers>=${CUDA_MIN} dph<=${MAX_HOURLY} inet_down>=${INET_DOWN_MIN} disk_space>=${DISK_GB}" \
      --order dph --limit 200 --raw 2>/dev/null | python3 -c '
import json, sys
offers = json.load(sys.stdin)
for offer in offers:
    gpu_name = str(offer.get("gpu_name") or offer.get("gpu") or offer.get("gpu_model") or "").lower()
    if "h100" in gpu_name:
        print(offer["id"])
        break
'
  )
  [ -n "$OFFER_ID" ] || die "No H100 offers found in budget. Increase --max-hourly."

  ONSTART_CMD='apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip python3-venv git libgl1 libglib2.0-0 tmux && sleep infinity'

  log "Creating H100 instance from offer $OFFER_ID ($VASTAI_IMAGE)..."
  CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" \
    --image "$VASTAI_IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --onstart-cmd "$ONSTART_CMD" 2>&1)

  INSTANCE_ID=$(
    echo "$CREATE_OUTPUT" | python3 -c '
import json, sys
for line in sys.stdin:
    line = line.strip()
    if "{" not in line:
        continue
    try:
        data = json.loads(line[line.index("{"):])
    except Exception:
        continue
    new_contract = data.get("new_contract")
    if new_contract:
        print(new_contract)
        break
'
  )
  [ -n "$INSTANCE_ID" ] || die "Failed to create H100 instance: $CREATE_OUTPUT"
  CREATED_INSTANCE=true
fi

log "Using instance: $INSTANCE_ID"
wait_for_instance "$INSTANCE_ID"
SSH_INFO="$(get_ssh_info "$INSTANCE_ID")"
SSH_HOST="$(echo "$SSH_INFO" | awk '{print $1}')"
SSH_PORT="$(echo "$SSH_INFO" | awk '{print $2}')"
[ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] || die "Unable to resolve SSH host/port for instance $INSTANCE_ID"
wait_for_ssh "$SSH_HOST" "$SSH_PORT"

REMOTE_RENDERS="${REMOTE_ROOT}/inputs/renders"
REMOTE_OUTPUT="${REMOTE_ROOT}/outputs/fixer_output"
REMOTE_FIXER="${REMOTE_ROOT}/Fixer"

log "Creating remote directories..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" \
  "mkdir -p '${REMOTE_RENDERS}' '${REMOTE_OUTPUT}' '${REMOTE_FIXER}'"

log "Syncing renders..."
rsync -az --delete -e "ssh $SSH_OPTS -p $SSH_PORT" \
  "${INPUT_RENDERS}/" "root@${SSH_HOST}:${REMOTE_RENDERS}/"

REMOTE_HAS_FIXER=$(ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" \
  "if [ -f '${REMOTE_FIXER}/src/inference_pretrained_model.py' ] && [ -f '${REMOTE_FIXER}/weights/pretrained/pretrained_fixer.pkl' ]; then echo yes; else echo no; fi")

if [ "$REMOTE_HAS_FIXER" != "yes" ]; then
  [ -d "$FIXER_DIR_LOCAL" ] || die "Local Fixer source dir not found: $FIXER_DIR_LOCAL"
  log "Syncing Fixer source: $FIXER_DIR_LOCAL -> ${REMOTE_FIXER}"
  rsync -az --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "${FIXER_DIR_LOCAL}/" "root@${SSH_HOST}:${REMOTE_FIXER}/"

  if [ -d "$FIXER_WEIGHTS_LOCAL" ]; then
    log "Syncing Fixer weights: $FIXER_WEIGHTS_LOCAL -> ${REMOTE_FIXER}/weights"
    rsync -az --delete -e "ssh $SSH_OPTS -p $SSH_PORT" \
      "${FIXER_WEIGHTS_LOCAL}/" "root@${SSH_HOST}:${REMOTE_FIXER}/weights/"
  fi
fi

if [ -n "$REMOTE_SETUP_CMD" ]; then
  log "Running custom remote setup command..."
  ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" "bash -lc '$REMOTE_SETUP_CMD'"
else
  log "Running default remote setup (python/pip + Fixer requirements if present)..."
  ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" "bash -lc '
set -euo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip python3-venv git libgl1 libglib2.0-0
python3 -m pip install --upgrade pip setuptools wheel
if [ -f \"${REMOTE_FIXER}/requirements.txt\" ]; then
  python3 -m pip install -r \"${REMOTE_FIXER}/requirements.txt\"
fi
'"
fi

log "Running Fixer inference on H100..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" "bash -lc '
set -euo pipefail
python3 \"${REMOTE_FIXER}/src/inference_pretrained_model.py\" \
  --input_folder \"${REMOTE_RENDERS}\" \
  --output_folder \"${REMOTE_OUTPUT}\" \
  --pretrained_path \"${REMOTE_FIXER}/weights/pretrained/pretrained_fixer.pkl\"
'"

log "Syncing refined images back..."
mkdir -p "$OUTPUT_DIR"
rsync -az --delete -e "ssh $SSH_OPTS -p $SSH_PORT" \
  "root@${SSH_HOST}:${REMOTE_OUTPUT}/" "${OUTPUT_DIR}/"

FOUND_FILES=$(find "$OUTPUT_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" -o -name "*.exr" \) | wc -l | tr -d ' ')
if [ "${FOUND_FILES}" = "0" ]; then
  die "No refined images were produced in $OUTPUT_DIR"
fi

log "Fixer stage complete. Refined image count: ${FOUND_FILES}"
if [ "$CREATED_INSTANCE" = true ] && [ "$KEEP_INSTANCE" = true ]; then
  log "Instance kept alive: $INSTANCE_ID"
fi
