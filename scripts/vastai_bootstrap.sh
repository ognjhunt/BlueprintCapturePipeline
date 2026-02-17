#!/usr/bin/env bash
# =============================================================================
# Vast.ai Instance Bootstrap Script
# =============================================================================
# This script provisions a Vast.ai GPU instance for BlueprintCapturePipeline.
# It handles instance creation, code sync, dependency installation, and
# test verification — all in one command.
#
# Usage:
#   ./scripts/vastai_bootstrap.sh [--test-only] [--instance-id ID] [--install-ml] [--with-fixer]
#
# Options:
#   --test-only      Skip instance creation; sync + test on existing instance
#   --instance-id ID Use a specific existing Vast.ai instance
#   --install-ml     Install full ML stack (CUDA COLMAP + 3DGRUT + SAM3 + DA3)
#   --with-fixer     Include local Fixer install/weights during --install-ml
#
# Prerequisites:
#   - vastai CLI installed and authenticated (pip install vastai)
#   - SSH key registered with Vast.ai (vastai set api-key ...)
#   - Docker snapshot image pushed: nijelhunt/blueprint-capture-pipeline:cuda-snapshot
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Configuration ----------------------------------------------------------
VASTAI_IMAGE="${VASTAI_IMAGE:-nijelhunt/blueprint-capture-pipeline:cuda-snapshot}"
DISK_GB=80
MAX_HOURLY_RATE=0.30
MIN_GPU_RAM=16
CUDA_MIN="12.0"
INET_DOWN_MIN=200

# SSH settings
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"

# ---- Vast.ai onstart command ------------------------------------------------
# This runs INSIDE the container on first boot.
# Fixes applied:
#   1. Installs tmux (Vast.ai bashrc references it, causes exit 127 without it)
#   2. Installs openssh-server (python:3.11-slim doesn't include it)
#   3. Installs build-essential, git, libgl1, libglib2.0-0 for pipeline deps
#   4. Installs Python 3.11 (if not present in CUDA image)
#   5. Creates /run/sshd for SSH daemon
ONSTART_CMD='apt-get update && apt-get install -y --no-install-recommends \
  python3.11 python3.11-venv python3-pip \
  build-essential git libgl1 libglib2.0-0 tmux \
  && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
  && ln -sf /usr/bin/python3.11 /usr/bin/python \
  && sleep infinity'
if [[ "$VASTAI_IMAGE" == "nijelhunt/blueprint-capture-pipeline:"* ]]; then
    # Snapshot image already contains dependencies.
    ONSTART_CMD='sleep infinity'
fi

# ---- Helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

wait_for_instance() {
    local id="$1"
    local max_wait=300  # 5 minutes
    local elapsed=0
    log "Waiting for instance $id to reach 'running' status..."
    while [ $elapsed -lt $max_wait ]; do
        local status
        status=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for inst in data:
    if inst.get('id') == $id:
        print(inst.get('actual_status', inst.get('status', 'unknown')))
        break
" 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            log "Instance $id is running."
            return 0
        fi
        log "  Status: $status (${elapsed}s elapsed)"
        sleep 15
        elapsed=$((elapsed + 15))
    done
    die "Instance $id did not reach 'running' in ${max_wait}s"
}

wait_for_ssh() {
    local host="$1"
    local port="$2"
    local max_wait=180
    local elapsed=0
    log "Waiting for SSH on $host:$port..."
    while [ $elapsed -lt $max_wait ]; do
        if ssh $SSH_OPTS -p "$port" "root@$host" 'echo OK' 2>/dev/null | grep -q OK; then
            log "SSH is ready."
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    die "SSH not ready after ${max_wait}s"
}

get_ssh_info() {
    local id="$1"
    vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for inst in data:
    if inst.get('id') == $id:
        ssh_host = inst.get('ssh_host', '')
        ssh_port = inst.get('ssh_port', '')
        print(f'{ssh_host} {ssh_port}')
        break
"
}

# ---- Parse args -------------------------------------------------------------
INSTANCE_ID=""
TEST_ONLY=false
INSTALL_ML=false
INSTALL_FIXER=false

while [ $# -gt 0 ]; do
    case "$1" in
        --test-only)    TEST_ONLY=true; shift ;;
        --instance-id)  INSTANCE_ID="$2"; shift 2 ;;
        --install-ml)   INSTALL_ML=true; shift ;;
        --with-fixer)   INSTALL_FIXER=true; shift ;;
        *)              die "Unknown argument: $1" ;;
    esac
done

# ---- Step 1: Create or reuse instance ---------------------------------------
if [ -z "$INSTANCE_ID" ] && [ "$TEST_ONLY" = false ]; then
    log "Searching for GPU instances (>=${MIN_GPU_RAM}GB VRAM, <\$${MAX_HOURLY_RATE}/hr)..."
    OFFER_ID=$(vastai search offers \
        "gpu_ram>=${MIN_GPU_RAM} num_gpus=1 rentable=true dph<=${MAX_HOURLY_RATE} cuda_vers>=${CUDA_MIN} inet_down>=${INET_DOWN_MIN} disk_space>=${DISK_GB}" \
        --order 'dph' --limit 1 --raw 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data:
    print(data[0]['id'])
" 2>/dev/null || true)

    [ -z "$OFFER_ID" ] && die "No suitable GPU offers found. Try increasing MAX_HOURLY_RATE."

    log "Creating instance from offer $OFFER_ID ($VASTAI_IMAGE)..."
    CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" \
        --image "$VASTAI_IMAGE" \
        --disk "$DISK_GB" \
        --ssh --direct \
        --onstart-cmd "$ONSTART_CMD" 2>&1)

    INSTANCE_ID=$(echo "$CREATE_OUTPUT" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if '{' in line:
        data = json.loads(line[line.index('{'):])
        print(data.get('new_contract', ''))
        break
" 2>/dev/null || true)

    [ -z "$INSTANCE_ID" ] && die "Failed to create instance: $CREATE_OUTPUT"
    log "Instance created: ID=$INSTANCE_ID"
fi

if [ -z "$INSTANCE_ID" ]; then
    # Find existing running instance
    INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data:
    print(data[0]['id'])
" 2>/dev/null || true)
    [ -z "$INSTANCE_ID" ] && die "No running instances found. Run without --test-only to create one."
fi

log "Using instance: $INSTANCE_ID"

# ---- Step 2: Wait for instance + SSH ----------------------------------------
wait_for_instance "$INSTANCE_ID"

SSH_INFO=$(get_ssh_info "$INSTANCE_ID")
SSH_HOST=$(echo "$SSH_INFO" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_INFO" | awk '{print $2}')

[ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] && die "Could not determine SSH host/port for instance $INSTANCE_ID"

wait_for_ssh "$SSH_HOST" "$SSH_PORT"

# ---- Step 3: Sync code ------------------------------------------------------
log "Syncing pipeline code to instance..."
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='*.egg-info' \
    -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "$PROJECT_ROOT/" "root@${SSH_HOST}:/app/" 2>&1 | tail -5

log "Code sync complete."

# ---- Step 4: Install dependencies -------------------------------------------
log "Installing Python dependencies..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -c "'
cd /app
pip install --quiet --upgrade pip setuptools wheel 2>&1 | tail -1
pip install --quiet -e . 2>&1 | tail -1
pip install --quiet -r requirements.txt 2>&1 | tail -1
pip install --quiet pytest 2>&1 | tail -1
echo DEPS_INSTALLED
'"

# ---- Step 5: Install ML dependencies (optional) ----------------------------
if [ "$INSTALL_ML" = true ]; then
    log "Installing ML dependencies (CUDA COLMAP + 3DGRUT + SAM3 + DA3)..."
    FIXER_FLAG=""
    if [ "$INSTALL_FIXER" = true ]; then
        FIXER_FLAG="--with-fixer"
    fi
    ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -lc "'
set -euo pipefail
cd /app
chmod +x scripts/install_colmap_cuda.sh scripts/install_ml_stack.sh
./scripts/install_ml_stack.sh $FIXER_FLAG
'"
fi

# ---- Step 6: Verify GPU -----------------------------------------------------
log "Verifying GPU access..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -c "'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
'"

# ---- Step 7: Run tests ------------------------------------------------------
log "Running test suite..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -c "'
cd /app && python3 -m pytest tests/ -v --tb=short 2>&1
'"

TEST_EXIT=$?

# ---- Summary ----------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Vast.ai Instance Ready"
echo "=============================================="
echo "  Instance ID:  $INSTANCE_ID"
echo "  SSH Command:  ssh -p $SSH_PORT root@$SSH_HOST"
echo "  GPU:          $(ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>/dev/null)"
echo "  Tests:        $([ $TEST_EXIT -eq 0 ] && echo 'ALL PASSED' || echo 'SOME FAILED')"
echo ""
echo "  Run pipeline:"
echo "    ssh -p $SSH_PORT root@$SSH_HOST"
echo "    python3 /app/scripts/nurec_shim.py --help"
echo ""
echo "  Install ML deps (if not done):"
echo "    $0 --instance-id $INSTANCE_ID --install-ml"
echo "  Install ML deps + Fixer:"
echo "    $0 --instance-id $INSTANCE_ID --install-ml --with-fixer"
echo ""
echo "  Destroy when done:"
echo "    vastai destroy instance $INSTANCE_ID"
echo "=============================================="
