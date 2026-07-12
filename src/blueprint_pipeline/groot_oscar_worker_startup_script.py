"""Shell fragments for sealed GR00T+OSCAR worker startup."""

STARTUP_GATES_SCRIPT = r'''
# Prove the exact kitchen tree, fast RTX path, and review renderer on this
# allocation before policy or task execution starts.
if [ -z "${BLUEPRINT_PROVIDER_ALLOCATION_ID:-}" ]; then
  BLUEPRINT_PROVIDER_ALLOCATION_ID="$(curl -fsS --max-time 5 http://169.254.169.254/metadata/v1/id)"
  export BLUEPRINT_PROVIDER_ALLOCATION_ID
fi
case "$BLUEPRINT_PROVIDER_ALLOCATION_ID" in
  ''|*[!0-9]*) echo "invalid DigitalOcean allocation identity" >&2; exit 42 ;;
esac
STARTUP_DIR=/workspace/closed_loop_out/startup_gates
mkdir -p "$STARTUP_DIR/kitchen" "$STARTUP_DIR/review"
set +e
/isaac-sim/python.sh -m blueprint_pipeline.kitchen_asset_startup_gate \
  --expected-inventory /workspace/kitchen_asset_inventory_checksums.json \
  --out-dir "$STARTUP_DIR/kitchen" --tree-root /workspace/kitchen
KITCHEN_GATE_RC=$?
/isaac-sim/python.sh -m blueprint_pipeline.isaac_worker_runtime_preflight \
  --output "$STARTUP_DIR/isaac_worker_runtime_preflight.json" \
  --require-nvidia-smi --require-rtx-render --smoke-steps 3
FAST_GATE_RC=$?
/isaac-sim/python.sh -m blueprint_pipeline.isaac_review_renderer_canary \
  --output-dir "$STARTUP_DIR/review" \
  --launch-session-id "$BLUEPRINT_LAUNCH_SESSION_ID" \
  --image-digest "$BLUEPRINT_WORKER_IMAGE_DIGEST" --orientation landscape
REVIEW_GATE_RC=$?
KITCHEN_GATE_RC="$KITCHEN_GATE_RC" FAST_GATE_RC="$FAST_GATE_RC" \
REVIEW_GATE_RC="$REVIEW_GATE_RC" python - <<'PY'
import json, os
from pathlib import Path

root = Path('/workspace/closed_loop_out/startup_gates')
leaves = {
    'kitchen_asset_startup_gate': (root/'kitchen/kitchen_asset_startup_gate.json', 'completed', int(os.environ['KITCHEN_GATE_RC'])),
    'fast_startup_canary': (root/'isaac_worker_runtime_preflight.json', 'passed', int(os.environ['FAST_GATE_RC'])),
    'review_renderer_canary': (root/'review/isaac_review_renderer_canary.json', 'passed', int(os.environ['REVIEW_GATE_RC'])),
}
nonce = os.environ.get('BLUEPRINT_LAUNCH_SESSION_ID', '')
image = os.environ.get('BLUEPRINT_WORKER_IMAGE_DIGEST', '')
rows, blockers = {}, []
for gate_id, (path, passing_status, returncode) in leaves.items():
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        payload = {'status': 'missing', 'blockers': [gate_id + '_result_missing'], 'error_type': type(exc).__name__}
    payload['launch_session_id'] = nonce
    payload['image_digest'] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    passed = returncode == 0 and payload.get('status') == passing_status
    row_blockers = [] if passed else list(payload.get('blockers') or [gate_id + '_failed'])
    rows[gate_id] = {'status': 'passed' if passed else 'blocked', 'returncode': returncode,
                     'artifact_path': str(path), 'blockers': row_blockers}
    blockers.extend(row_blockers)
summary = {'schema_version': 'groot_oscar_same_allocation_startup_gates.v1',
           'status': 'passed' if not blockers else 'blocked',
           'launch_session_id': nonce, 'image_digest': image,
           'provider_allocation_binding_required': True, 'gates': rows,
           'blockers': sorted(set(blockers)),
           'claim_boundary': {'startup_proof_only': True, 'proves_task_success': False}}
(root/'supervised_startup_gates.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
raise SystemExit(0 if not blockers else 1)
PY
STARTUP_GATES_RC=$?
set -e
if [ "$STARTUP_GATES_RC" -eq 0 ]; then
  python -m blueprint_pipeline.g1_kitchen_startup_proof \
    --startup-dir "$STARTUP_DIR" \
    --attempt-input-manifest /workspace/attempt_input_manifest.json
fi
if [ "$STARTUP_GATES_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$STARTUP_GATES_RC" \
    BLUEPRINT_WORKER_FAILURE="same_allocation_startup_gates_failed" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$STARTUP_GATES_RC"
fi
upload_phase startup_gates_passed
'''


GEAR_SONIC_READY_SCRIPT = r'''
set +e
/opt/oscar-venv/bin/python - <<'PY'
import msgpack, time, zmq
ctx = zmq.Context(); sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, 'robot_config')
sub.connect('tcp://127.0.0.1:5557')
deadline = time.time() + 900
try:
    while time.time() < deadline:
        if sub.poll(1000, zmq.POLLIN):
            raw = sub.recv()
            payload = msgpack.unpackb(raw[len(b'robot_config'):], raw=False)
            if isinstance(payload, dict) and payload:
                break
    else:
        raise SystemExit('official_gear_sonic_controller_not_ready')
finally:
    sub.close(); ctx.term()
PY
GEAR_SONIC_READY_RC=$?
set -e
if [ "$GEAR_SONIC_READY_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$GEAR_SONIC_READY_RC" \
    BLUEPRINT_WORKER_FAILURE="official_gear_sonic_controller_not_ready" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$GEAR_SONIC_READY_RC"
fi
upload_phase gear_sonic_controller_ready
'''
