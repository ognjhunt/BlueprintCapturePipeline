"""Shell fragments for sealed GR00T+OSCAR worker startup."""

GROOT_CHECKPOINT_PREFLIGHT_SCRIPT = r'''
# The pinned GR00T N1.7 loader selects its backbone from the literal model_name
# before it loads the nested model.  A plain local path is offline-safe but is
# rejected by get_backbone_cls.  Give the same local bytes a selector-bearing
# alias, rewrite only this ephemeral container's SONIC config, and exercise the
# exact selector plus nested processor construction before starting the server.
GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT=/workspace/closed_loop_out/groot_sonic_checkpoint_preflight.json
set +e
PYTHONPATH="${BLUEPRINT_GROOT_RUNTIME_PYTHONPATH:-${PYTHONPATH:-}}" \
  /opt/gr00t/.venv/bin/python - <<'PY'
import json
import os
import sys
from pathlib import Path

checkpoint = Path(
    os.environ.get(
        "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT",
        "/opt/blueprint/ckpts/sonic",
    )
)
artifact = Path(
    os.environ.get(
        "BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT",
        "/workspace/closed_loop_out/groot_sonic_checkpoint_preflight.json",
    )
)
expected_repo = os.environ.get(
    "COSMOS_BACKBONE_REPO", "nvidia/Cosmos-Reason2-2B"
)
payload = {
    "schema_version": "groot_sonic_checkpoint_preflight.v1",
    "status": "blocked",
    "checkpoint_path": str(checkpoint),
    "expected_backbone_repo": expected_repo,
    "checks": {},
    "blockers": [],
    "raw_secret_values_recorded": False,
    "claim_boundary": {
        "preflight_is_not_policy_inference": True,
        "preflight_is_not_episode_success": True,
    },
}
try:
    expected_venv_root = Path(
        os.environ.get("BLUEPRINT_GROOT_VENV_ROOT", "/opt/gr00t/.venv")
    ).resolve()
    oscar_dependency_target = Path(
        os.environ.get(
            "BLUEPRINT_OSCAR_RUNTIME_DEPENDENCY_TARGET",
            "/workspace/oscar_runtime_deps",
        )
    ).resolve()
    pythonpath_entries = [
        Path(value).resolve()
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    sys_path_entries = [Path(value).resolve() for value in sys.path if value]
    payload["checks"]["groot_interpreter_prefix_exact"] = (
        Path(sys.prefix).resolve() == expected_venv_root
    )
    payload["checks"]["oscar_dependency_target_absent_from_pythonpath"] = (
        oscar_dependency_target not in pythonpath_entries
    )
    payload["checks"]["oscar_dependency_target_absent_from_sys_path"] = (
        oscar_dependency_target not in sys_path_entries
    )
    for check_name in (
        "groot_interpreter_prefix_exact",
        "oscar_dependency_target_absent_from_pythonpath",
        "oscar_dependency_target_absent_from_sys_path",
    ):
        if not payload["checks"][check_name]:
            raise RuntimeError("groot_runtime_isolation_failed:" + check_name)

    import accelerate

    accelerate_path = Path(str(accelerate.__file__ or "")).resolve()
    payload["accelerate_module_path"] = str(accelerate_path)
    payload["accelerate_version"] = str(getattr(accelerate, "__version__", ""))
    payload["checks"]["accelerate_resolved_from_groot_venv"] = (
        accelerate_path.is_relative_to(expected_venv_root)
    )
    if not payload["checks"]["accelerate_resolved_from_groot_venv"]:
        raise RuntimeError(
            "groot_runtime_isolation_failed:accelerate_resolved_from_groot_venv"
        )

    config_path = checkpoint / "config.json"
    checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    payload["checks"]["checkpoint_config_loaded"] = True
    original_name = str(
        checkpoint_config.get("blueprint_original_model_name")
        or checkpoint_config.get("model_name")
        or ""
    )
    if original_name != expected_repo:
        raise RuntimeError(
            "groot_sonic_checkpoint_backbone_identity_mismatch:"
            f"expected={expected_repo!r}:observed={original_name!r}"
        )

    processor_config_path = checkpoint / "processor/processor_config.json"
    processor_config = json.loads(
        processor_config_path.read_text(encoding="utf-8")
    )
    processor_kwargs = processor_config.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise RuntimeError("groot_sonic_processor_kwargs_missing")
    payload["checks"]["processor_config_loaded"] = True

    # Keep the exact upstream token and capitalization in the path string: the
    # pinned get_backbone_cls checks for ``nvidia/Cosmos-Reason2`` literally.
    # This mirrors the already-verified external-volume layout: the selector
    # anchor contributes that token while ``../..`` resolves to a flat local
    # model root. Top-level links avoid copying any model bytes.
    selector_root = Path(
        os.environ.get(
            "BLUEPRINT_GROOT_MODEL_ALIAS_ROOT",
            "/workspace/.blueprint-model-aliases/cosmos",
        )
    )
    if selector_root.is_symlink():
        raise RuntimeError("groot_sonic_selector_root_is_symlink")
    selector_root.mkdir(parents=True, exist_ok=True)
    selector_root = selector_root.resolve(strict=True)
    selector_anchor = selector_root / "nvidia/Cosmos-Reason2-2B"
    alias = selector_anchor / "../.."
    alias_name = str(alias)

    configured_model_name = str(checkpoint_config.get("model_name") or "")
    processor_model_name = str(processor_kwargs.get("model_name") or "")
    runtime_model_name = str(
        checkpoint_config.get("blueprint_runtime_model_path") or ""
    )
    runtime_model_kind = str(
        checkpoint_config.get("blueprint_runtime_model_kind") or ""
    )
    if runtime_model_kind not in {
        "",
        "model_root",
        "baked_selector_root_without_namespace",
    }:
        raise RuntimeError(
            "groot_sonic_runtime_model_kind_invalid:" + runtime_model_kind
        )
    alias_already_active = bool(runtime_model_name)
    if alias_already_active:
        if configured_model_name != alias_name:
            raise RuntimeError(
                "groot_sonic_checkpoint_runtime_alias_mismatch:"
                f"expected={alias_name!r}:observed={configured_model_name!r}"
            )
        if processor_model_name != alias_name:
            raise RuntimeError(
                "groot_sonic_processor_runtime_alias_mismatch:"
                f"expected={alias_name!r}:observed={processor_model_name!r}"
            )
        configured_model = Path(runtime_model_name)
    else:
        if processor_model_name not in {expected_repo, configured_model_name}:
            raise RuntimeError(
                "groot_sonic_processor_backbone_identity_mismatch:"
                f"expected={expected_repo!r}:observed={processor_model_name!r}"
            )
        if configured_model_name == alias_name:
            raise RuntimeError("groot_sonic_runtime_model_binding_missing")
        configured_model = Path(configured_model_name)

    if not configured_model.is_absolute() or not configured_model.is_dir():
        raise RuntimeError("groot_sonic_nested_model_is_not_local_directory")
    resolved_model = configured_model.resolve(strict=True)
    if resolved_model == selector_root or selector_root in resolved_model.parents:
        raise RuntimeError("groot_sonic_runtime_model_resolves_to_selector_root")

    source_entries = list(resolved_model.iterdir())
    source_names = {source.name for source in source_entries}
    baked_alias_name = str(
        resolved_model / "nvidia/Cosmos-Reason2-2B/../.."
    )
    recovering_baked_selector = bool(
        not alias_already_active
        and configured_model_name == baked_alias_name
        and processor_model_name == baked_alias_name
    )
    using_recorded_baked_selector = bool(
        alias_already_active
        and runtime_model_kind == "baked_selector_root_without_namespace"
    )
    if "nvidia" in source_names:
        if not (recovering_baked_selector or using_recorded_baked_selector):
            raise RuntimeError("groot_sonic_selector_reserved_name_collision:nvidia")
        baked_namespace = resolved_model / "nvidia"
        baked_anchor = baked_namespace / "Cosmos-Reason2-2B"
        if (
            baked_namespace.is_symlink()
            or baked_anchor.is_symlink()
            or not baked_anchor.is_dir()
            or any(baked_anchor.iterdir())
        ):
            raise RuntimeError("groot_sonic_baked_selector_namespace_invalid")
        source_entries = [source for source in source_entries if source.name != "nvidia"]
        source_names = {source.name for source in source_entries}
        if (
            not source_entries
            or "config.json" not in source_names
            or any(not source.is_symlink() for source in source_entries)
        ):
            raise RuntimeError("groot_sonic_baked_selector_model_links_invalid")
        runtime_model_kind = "baked_selector_root_without_namespace"
    else:
        runtime_model_kind = "model_root"
    for source in source_entries:
        destination = selector_root / source.name
        if destination.is_symlink():
            if destination.resolve(strict=True) != source.resolve(strict=True):
                raise RuntimeError(
                    "groot_sonic_selector_file_binding_mismatch:" + source.name
                )
        elif destination.exists():
            raise RuntimeError(
                "groot_sonic_selector_file_not_symlink:" + source.name
            )
        else:
            destination.symlink_to(source, target_is_directory=source.is_dir())

    unexpected_names = {
        child.name for child in selector_root.iterdir()
    } - source_names - {"nvidia"}
    if unexpected_names:
        raise RuntimeError(
            "groot_sonic_selector_unexpected_entries:"
            + ",".join(sorted(unexpected_names))
        )
    selector_namespace = selector_root / "nvidia"
    if selector_namespace.is_symlink():
        raise RuntimeError("groot_sonic_selector_namespace_is_symlink:nvidia")
    selector_anchor.mkdir(parents=True, exist_ok=True)
    alias = selector_anchor / "../.."
    if not alias.is_dir() or alias.resolve(strict=True) != selector_root:
        raise RuntimeError("groot_sonic_local_backbone_selector_resolution_failed")

    checkpoint_config["blueprint_original_model_name"] = original_name
    checkpoint_config["blueprint_runtime_model_path"] = str(resolved_model)
    checkpoint_config["blueprint_runtime_model_kind"] = runtime_model_kind
    checkpoint_config["model_name"] = str(alias)
    temporary_config = config_path.with_name(".config.json.blueprint-runtime")
    temporary_config.write_text(
        json.dumps(checkpoint_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.rename(temporary_config, config_path)
    processor_kwargs["model_name"] = str(alias)
    processor_config["processor_kwargs"] = processor_kwargs
    temporary_processor_config = processor_config_path.with_name(
        ".processor_config.json.blueprint-runtime"
    )
    temporary_processor_config.write_text(
        json.dumps(processor_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.rename(temporary_processor_config, processor_config_path)
    payload["model_name"] = str(alias)
    payload["processor_model_name"] = str(alias)
    payload["resolved_local_model_path"] = str(resolved_model)
    payload["selector_root"] = str(selector_root)
    payload["alias_activation_state"] = (
        "verified_existing"
        if alias_already_active
        else (
            "recovered_baked_selector"
            if recovering_baked_selector
            else "activated"
        )
    )
    payload["runtime_model_kind"] = runtime_model_kind
    payload["checks"]["original_backbone_identity_preserved"] = True
    payload["checks"]["runtime_model_binding_preserved"] = True
    payload["checks"]["existing_alias_binding_verified"] = alias_already_active
    payload["checks"]["supported_local_model_alias_activated"] = True
    payload["checks"]["root_and_processor_model_names_bound"] = True

    # This is the exact call that rejected attempt 008.  It constructs the
    # registered SONIC config and selects the pinned Qwen3 backbone without
    # allocating the 2B model weights a second time.
    import gr00t.model  # noqa: F401
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls
    from transformers import AutoConfig, AutoProcessor, Qwen3VLProcessor

    model_config = AutoConfig.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=True
    )
    backbone_cls = get_backbone_cls(model_config)
    if backbone_cls.__name__ != "Qwen3Backbone":
        raise RuntimeError(
            "groot_sonic_unexpected_backbone_class:" + backbone_cls.__name__
        )
    payload["backbone_class"] = backbone_cls.__name__
    payload["checks"]["pinned_get_backbone_cls_resolved"] = True

    nested_config = AutoConfig.from_pretrained(
        str(alias), local_files_only=True, trust_remote_code=True
    )
    payload["nested_config_class"] = type(nested_config).__name__
    payload["checks"]["nested_backbone_config_offline_constructible"] = True
    processor = Qwen3VLProcessor.from_pretrained(
        str(alias), local_files_only=True
    )
    payload["nested_processor_class"] = type(processor).__name__
    payload["checks"]["nested_processor_offline_constructible"] = True
    policy_processor = AutoProcessor.from_pretrained(
        str(checkpoint / "processor"), local_files_only=True
    )
    payload["policy_processor_class"] = type(policy_processor).__name__
    payload["checks"]["groot_policy_processor_offline_constructible"] = True
    payload["status"] = "passed"
except Exception as exc:
    payload["error_type"] = type(exc).__name__
    payload["error"] = str(exc)
    payload["blockers"] = ["groot_sonic_checkpoint_preflight_failed"]
finally:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
raise SystemExit(0 if payload["status"] == "passed" else 1)
PY
GROOT_CHECKPOINT_PREFLIGHT_RC=$?
set -e
if [ "$GROOT_CHECKPOINT_PREFLIGHT_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$GROOT_CHECKPOINT_PREFLIGHT_RC" \
    BLUEPRINT_WORKER_FAILURE="groot_sonic_checkpoint_preflight_failed" \
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit "$GROOT_CHECKPOINT_PREFLIGHT_RC"
fi
upload_phase groot_checkpoint_preflight_passed
'''

STARTUP_GATES_SCRIPT = r'''
# Prove the exact kitchen tree, Isaac/GPU runtime imports, and the real review
# renderer on this allocation before policy or task execution starts.  The
# review canary is the rendered-frame authority; asking the fast preflight to
# initialize a second Replicator renderer first can wedge the carrier before
# the authoritative canary runs.
if [ -z "${BLUEPRINT_PROVIDER_ALLOCATION_ID:-}" ]; then
  BLUEPRINT_PROVIDER_ALLOCATION_ID="$(curl -fsS --max-time 5 http://169.254.169.254/metadata/v1/id)"
  export BLUEPRINT_PROVIDER_ALLOCATION_ID
fi
case "$BLUEPRINT_PROVIDER_ALLOCATION_ID" in
  '') echo "provider allocation identity unavailable" >&2; exit 42 ;;
esac
STARTUP_DIR=/workspace/closed_loop_out/startup_gates
mkdir -p "$STARTUP_DIR/kitchen" "$STARTUP_DIR/review"
set +e
/isaac-sim/python.sh -m blueprint_pipeline.kitchen_asset_startup_gate \
  --expected-inventory /workspace/kitchen_asset_inventory_checksums.json \
  --out-dir "$STARTUP_DIR/kitchen" --tree-root /workspace/kitchen
KITCHEN_GATE_RC=$?
timeout --signal=TERM --kill-after=15s 180s \
  /isaac-sim/python.sh -m blueprint_pipeline.isaac_worker_runtime_preflight \
  --output "$STARTUP_DIR/isaac_worker_runtime_preflight.json" \
  --require-nvidia-smi
FAST_GATE_RC=$?
BLUEPRINT_REVIEW_CANARY_TIMEOUT_SECONDS="${BLUEPRINT_REVIEW_CANARY_TIMEOUT_SECONDS:-240}" \
timeout --signal=TERM --kill-after=15s 270s \
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
import hashlib, json, math, os, time
from pathlib import Path

from blueprint_pipeline.gear_sonic_official_zmq_executor import (
    _zmq_roundtrip,
    execute as execute_controller_fk,
)

deadline = time.time() + 900
bridge_required = os.environ.get(
    'BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_REQUIRED', ''
).lower() == 'true'

if bridge_required:
    heartbeat_path = Path(
        '/workspace/closed_loop_out/gear_sonic_isaac_dds_bridge_heartbeat.json'
    )
    snapshot_path = Path(os.environ['BLUEPRINT_GEAR_SONIC_ISAAC_STATE_SNAPSHOT'])
    controller_log = Path('/workspace/gear_sonic_controller.log')
    expected_source_sha = os.environ.get(
        'BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_SOURCE_SHA256', ''
    )
    first_publish_count = None
    while time.time() < deadline:
        for name in ('GEAR_SONIC_PID', 'GEAR_SONIC_ISAAC_BRIDGE_PID', 'ISAAC_TASK_PID'):
            try:
                os.kill(int(os.environ[name]), 0)
            except (OSError, ValueError, KeyError):
                raise SystemExit(name.lower() + '_exited_before_ready')
        try:
            heartbeat = json.loads(heartbeat_path.read_text(encoding='utf-8'))
            snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
        except Exception:
            time.sleep(0.2)
            continue
        publish_count = int(heartbeat.get('publish_count') or 0)
        if first_publish_count is None and publish_count > 0:
            first_publish_count = publish_count
        heartbeat_at_ns = int(heartbeat.get('heartbeat_at_ns') or 0)
        captured_at_ns = int(snapshot.get('captured_at_ns') or 0)
        now_ns = time.time_ns()
        heartbeat_age_ns = now_ns - heartbeat_at_ns
        source_age_ns = now_ns - captured_at_ns
        identity_matches = (
            heartbeat.get('simulator_session_id') == snapshot.get('simulator_session_id')
            and heartbeat.get('stage_id') == snapshot.get('stage_id')
            and bool(heartbeat.get('simulator_session_id'))
            and bool(heartbeat.get('stage_id'))
        )
        controller_initialized = (
            controller_log.is_file()
            and 'Init Done' in controller_log.read_text(encoding='utf-8', errors='replace')
        )
        if (
            heartbeat.get('status') == 'ready'
            and heartbeat.get('source_fresh') is True
            and heartbeat.get('holding_last_validated_snapshot') is False
            and heartbeat.get('surrogate') is False
            and snapshot.get('surrogate') is False
            and snapshot.get('source') == 'live_isaac_articulation'
            and heartbeat.get('bridge_source_sha256') == expected_source_sha
            and len(expected_source_sha) == 64
            and identity_matches
            and 0 <= heartbeat_age_ns <= 500_000_000
            and 0 <= source_age_ns <= 500_000_000
            and first_publish_count is not None
            and publish_count > first_publish_count
            and controller_initialized
        ):
            break
        time.sleep(0.2)
    else:
        raise SystemExit('official_gear_sonic_controller_or_isaac_dds_bridge_not_ready')
else:
    import msgpack, zmq
    ctx = zmq.Context(); sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, 'robot_config')
    sub.connect('tcp://127.0.0.1:5557')
    try:
        while time.time() < deadline:
            try:
                os.kill(int(os.environ['GEAR_SONIC_PID']), 0)
            except (OSError, ValueError):
                raise SystemExit('official_gear_sonic_controller_exited_before_ready')
            if sub.poll(1000, zmq.POLLIN):
                raw = sub.recv()
                payload = msgpack.unpackb(raw[len(b'robot_config'):], raw=False)
                if isinstance(payload, dict) and payload:
                    break
        else:
            raise SystemExit('official_gear_sonic_controller_not_ready')
    finally:
        sub.close(); ctx.term()

# ``Init Done`` and an advancing Isaac DDS bridge prove the controller's
# dependencies are alive, but they do not prove the action endpoint can enter
# CONTROL and return the matching ``g1_debug`` row. Attempt 017 passed those
# weaker checks and then failed on its first real action. Exercise the exact
# protocol-v4 action/state boundary before reporting controller readiness.
readiness_path = Path(
    '/workspace/closed_loop_out/gear_sonic_sim/controller_readiness.json'
)
readiness_path.parent.mkdir(parents=True, exist_ok=True)
controller_fk_readiness_path = Path(
    '/workspace/closed_loop_out/gear_sonic_sim/controller_fk_readiness.json'
)
projection_context_path_text = os.environ.get(
    'BLUEPRINT_CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT', ''
).strip()
controller_fk_readiness_required = bool(projection_context_path_text)

qualification_attempt_nonce = os.environ.get(
    'BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE', ''
).strip()
launch_session_id = os.environ.get('BLUEPRINT_LAUNCH_SESSION_ID', '').strip()
probe_identity_kind = (
    'qualification_attempt_nonce' if qualification_attempt_nonce else 'launch_session_id'
)
probe_identity = qualification_attempt_nonce or launch_session_id
readiness = {
    'schema_version': 'single_g1_kitchen_gear_sonic_controller_readiness.v1',
    'status': 'blocked',
    'topic': 'g1_debug',
    'protocol_version': 4,
    'probe_identity_kind': probe_identity_kind,
    'probe_identity_sha256': (
        hashlib.sha256(probe_identity.encode('utf-8')).hexdigest()
        if probe_identity else None
    ),
    'readiness_probe_sha256': None,
    'readiness_attempts': 0,
    'validated_fields': {},
    'checks': {
        'attempt_local_protocol_v4_pose_sent': False,
        'attempt_identity_bound': bool(probe_identity),
        'exact_motion_token_echo': False,
        'exact_left_hand_echo': False,
        'exact_right_hand_echo': False,
        'finite_body_target': False,
        'finite_body_measured': False,
        'finite_base_quaternion': False,
        'required_processes_alive': False,
        'dds_bridge_still_ready': False,
        'controller_fk_readiness_subprobe': (
            not controller_fk_readiness_required
        ),
    },
    'blockers': [],
    'raw_probe_values_recorded': False,
    'raw_secret_values_recorded': False,
    'claim_boundary': {
        'readiness_probe_only': True,
        'readiness_probe_is_not_episode_policy_action': True,
        'controller_state_ready_is_not_task_success': True,
    },
}

controller_fk_readiness = {
    'schema_version': 'single_g1_kitchen_controller_fk_readiness.v1',
    'status': 'blocked' if controller_fk_readiness_required else 'not_required',
    'required': controller_fk_readiness_required,
    'probe_identity_kind': probe_identity_kind,
    'probe_identity_sha256': readiness['probe_identity_sha256'],
    'source_controller_readiness_probe_sha256': None,
    'projection_context_sha256': None,
    'source_frame_sha256': None,
    'retained_controller_state_sha256': None,
    'controller_fk_result_sha256': None,
    'controller_revision_sha256': None,
    'controller_mapping_digest': None,
    'robot_model_sha256': None,
    'joint_names_sha256': None,
    'joint_positions_sha256': None,
    'landmarks_sha256': None,
    'standing_registration_sha256': None,
    'failure_code': None,
    'failure_exception_type': None,
    'failure_detail_sha256': None,
    'checks': {
        'attempt_identity_bound': bool(probe_identity),
        'live_projection_context_identity_bound': False,
        'live_projection_context_session_bound': False,
        'retained_canary_state_reused': False,
        'injected_transport_used_once': False,
        'no_additional_wire_action_sent': False,
        'pinned_controller_revision_verified': False,
        'official_mujoco_fk_completed': False,
        'protocol_v4_joint_mapping_verified': False,
        'standing_cross_simulator_registration_passed': False,
        'live_camera_projection_completed': False,
        'isaac_action_not_applied_by_subprobe': False,
    },
    'blockers': [],
    'raw_controller_state_recorded': False,
    'raw_joint_positions_recorded': False,
    'raw_landmarks_recorded': False,
    'raw_projection_context_recorded': False,
    'raw_secret_values_recorded': False,
    'claim_boundary': {
        'readiness_subprobe_only': True,
        'retained_controller_state_reused_without_second_wire_action': True,
        'mujoco_fk_does_not_apply_an_isaac_action': True,
        'initial_fk_landmark_visibility_is_not_controller_readiness': True,
        'readiness_subprobe_is_not_episode_policy_action': True,
        'readiness_subprobe_is_not_task_success': True,
    },
}

def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()

def canary_vector(label, attempt, size, *, centered):
    # Integer multiples of 2^-15 survive the float32 message boundary exactly,
    # allowing strict echo comparison without a broad tolerance. The launch
    # nonce and retry index make each pose unique, so a stale prior ``g1_debug``
    # row cannot satisfy a later attempt. Only the canary digest is retained.
    values = []
    block = 0
    while len(values) < size:
        digest = hashlib.sha256(
            f'{probe_identity}:{label}:{attempt}:{block}'.encode('utf-8')
        ).digest()
        for byte in digest:
            integer = (int(byte) - 127) if centered else (int(byte) + 1)
            values.append(integer / 32768.0)
            if len(values) == size:
                break
        block += 1
    return values

def finite_vector(value, size):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError('controller_readiness_vector_missing')
    result = [float(item) for item in value]
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError('controller_readiness_vector_invalid')
    return result

def load_attempt_bound_projection_context():
    path = Path(projection_context_path_text)
    while time.time() < probe_deadline:
        for process_name in required_process_names:
            try:
                os.kill(int(os.environ[process_name]), 0)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    process_name.lower() + '_exited_before_controller_fk_subprobe'
                ) from exc
        try:
            context = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            time.sleep(0.2)
            continue
        if not isinstance(context, dict):
            raise ValueError('controller_fk_projection_context_not_object')

        expected_attempt_id = os.environ.get(
            'BLUEPRINT_SINGLE_EPISODE_ATTEMPT_ID', ''
        ).strip()
        expected_qualification_sha256 = (
            hashlib.sha256(qualification_attempt_nonce.encode('utf-8')).hexdigest()
            if qualification_attempt_nonce else None
        )
        qualification_sequence_text = os.environ.get(
            'BLUEPRINT_QUALIFICATION_ATTEMPT_SEQUENCE', ''
        ).strip()
        qualification_sequence = (
            int(qualification_sequence_text)
            if qualification_sequence_text.isdigit()
            else None
        )
        identity_bound = (
            bool(expected_attempt_id)
            and context.get('attempt_id') == expected_attempt_id
            and context.get('launch_nonce') == probe_identity
            and context.get('allocation_launch_session_id') == launch_session_id
            and context.get('qualification_attempt_bound')
                is bool(qualification_attempt_nonce)
            and context.get('qualification_attempt_nonce_sha256')
                == expected_qualification_sha256
            and (
                (
                    qualification_attempt_nonce
                    and qualification_sequence is not None
                    and qualification_sequence > 0
                    and context.get('qualification_attempt_sequence')
                        == qualification_sequence
                )
                or (
                    not qualification_attempt_nonce
                    and not qualification_sequence_text
                    and context.get('qualification_attempt_sequence') is None
                )
            )
        )
        if not identity_bound:
            raise ValueError('controller_fk_projection_context_attempt_identity_mismatch')

        if bridge_required:
            heartbeat = json.loads(heartbeat_path.read_text(encoding='utf-8'))
            snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
            session_bound = (
                context.get('simulator_session_id')
                    == heartbeat.get('simulator_session_id')
                    == snapshot.get('simulator_session_id')
                and context.get('stage_id')
                    == heartbeat.get('stage_id')
                    == snapshot.get('stage_id')
                and bool(context.get('simulator_session_id'))
                and bool(context.get('stage_id'))
            )
        else:
            session_bound = bool(
                context.get('simulator_session_id') and context.get('stage_id')
            )
        if not session_bound:
            raise ValueError('controller_fk_projection_context_session_identity_mismatch')
        controller_fk_readiness['checks'][
            'live_projection_context_identity_bound'
        ] = True
        controller_fk_readiness['checks'][
            'live_projection_context_session_bound'
        ] = True
        return context
    raise TimeoutError('controller_fk_projection_context_not_ready')

accepted = None
motion = []
left = []
right = []
last_error_type = None
probe_deadline = time.time() + 900
required_process_names = ['GEAR_SONIC_PID', 'ISAAC_TASK_PID']
if bridge_required:
    required_process_names.append('GEAR_SONIC_ISAAC_BRIDGE_PID')
while time.time() < probe_deadline:
    dead_process = None
    for name in required_process_names:
        try:
            os.kill(int(os.environ[name]), 0)
        except (OSError, TypeError, ValueError, KeyError):
            dead_process = name
            break
    if dead_process is not None:
        readiness['blockers'].append(dead_process.lower() + '_exited_before_action_probe')
        break
    readiness['checks']['required_processes_alive'] = True
    readiness['readiness_attempts'] += 1
    probe_attempt = readiness['readiness_attempts']
    motion = canary_vector('motion', probe_attempt, 64, centered=True)
    left = canary_vector('left', probe_attempt, 7, centered=False)
    right = [-value for value in canary_vector('right', probe_attempt, 7, centered=False)]
    canary = {'motion_token': motion, 'left_hand': left, 'right_hand': right}
    readiness['readiness_probe_sha256'] = hashlib.sha256(
        json.dumps(canary, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    try:
        accepted = dict(
            _zmq_roundtrip(
                motion_token=motion,
                left_hand=left,
                right_hand=right,
                frame_index=-1,
                timeout_seconds=min(20.0, max(1.0, probe_deadline - time.time())),
            )
        )
        break
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        last_error_type = type(exc).__name__
        time.sleep(0.5)

if accepted is None:
    if not readiness['blockers']:
        readiness['blockers'].append('official_gear_sonic_matching_g1_debug_not_ready')
    if last_error_type:
        readiness['last_error_type'] = last_error_type
else:
    try:
        token_state = finite_vector(accepted.get('token_state'), 64)
        body_target = finite_vector(accepted.get('body_q_target'), 29)
        body_measured = finite_vector(accepted.get('body_q_measured'), 29)
        base_quaternion = finite_vector(accepted.get('base_quat_measured'), 4)
        echoed_left = finite_vector(accepted.get('last_left_hand_action'), 7)
        echoed_right = finite_vector(accepted.get('last_right_hand_action'), 7)
        readiness['checks'].update({
            'attempt_local_protocol_v4_pose_sent': True,
            'exact_motion_token_echo': token_state == motion,
            'exact_left_hand_echo': echoed_left == left,
            'exact_right_hand_echo': echoed_right == right,
            'finite_body_target': True,
            'finite_body_measured': True,
            'finite_base_quaternion': True,
        })
        readiness['validated_fields'] = {
            'token_state_dimension': len(token_state),
            'body_q_target_dimension': len(body_target),
            'body_q_measured_dimension': len(body_measured),
            'base_quat_measured_dimension': len(base_quaternion),
            'left_hand_echo_dimension': len(echoed_left),
            'right_hand_echo_dimension': len(echoed_right),
        }
        if bridge_required:
            # The controller reply and Isaac snapshot writers are independent
            # processes. A matching controller reply can therefore land after
            # the bridge marks one >500 ms source sample stale but before the
            # next live Isaac sample is observed. Keep the strict freshness
            # contract while tolerating that scheduling-dependent read.
            bridge_revalidation_deadline = min(probe_deadline, time.time() + 20.0)
            bridge_revalidation_floor_publish_count = int(
                heartbeat.get('publish_count') or 0
            )
            bridge_revalidation_attempts = 0
            bridge_revalidated = False
            while time.time() < bridge_revalidation_deadline:
                bridge_revalidation_attempts += 1
                bridge_processes_alive = True
                for process_name in required_process_names:
                    try:
                        os.kill(int(os.environ[process_name]), 0)
                    except (OSError, TypeError, ValueError, KeyError):
                        bridge_processes_alive = False
                        break
                if not bridge_processes_alive:
                    break
                try:
                    live_heartbeat = json.loads(
                        heartbeat_path.read_text(encoding='utf-8')
                    )
                    live_snapshot = json.loads(
                        snapshot_path.read_text(encoding='utf-8')
                    )
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    time.sleep(0.05)
                    continue
                bridge_now_ns = time.time_ns()
                live_heartbeat_age_ns = bridge_now_ns - int(
                    live_heartbeat.get('heartbeat_at_ns') or 0
                )
                live_snapshot_age_ns = bridge_now_ns - int(
                    live_snapshot.get('captured_at_ns') or 0
                )
                bridge_revalidated = (
                    live_heartbeat.get('status') == 'ready'
                    and live_heartbeat.get('source_fresh') is True
                    and live_heartbeat.get('holding_last_validated_snapshot') is False
                    and live_heartbeat.get('surrogate') is False
                    and live_snapshot.get('surrogate') is False
                    and live_snapshot.get('source') == 'live_isaac_articulation'
                    and live_heartbeat.get('bridge_source_sha256') == expected_source_sha
                    and live_heartbeat.get('simulator_session_id')
                        == live_snapshot.get('simulator_session_id')
                    and live_heartbeat.get('stage_id') == live_snapshot.get('stage_id')
                    and bool(live_heartbeat.get('simulator_session_id'))
                    and bool(live_heartbeat.get('stage_id'))
                    and int(live_heartbeat.get('publish_count') or 0)
                        > bridge_revalidation_floor_publish_count
                    and 0 <= live_heartbeat_age_ns <= 500_000_000
                    and 0 <= live_snapshot_age_ns <= 500_000_000
                )
                if bridge_revalidated:
                    break
                time.sleep(0.05)
            readiness['bridge_revalidation_attempts'] = bridge_revalidation_attempts
            readiness['checks']['dds_bridge_still_ready'] = bridge_revalidated
            if not bridge_revalidated:
                readiness['blockers'].append(
                    'official_gear_sonic_dds_bridge_not_fresh_after_matching_g1_debug'
                )
        else:
            readiness['checks']['dds_bridge_still_ready'] = True
        base_failed_checks = sorted(
            name
            for name, passed in readiness['checks'].items()
            if name != 'controller_fk_readiness_subprobe' and passed is not True
        )
        if controller_fk_readiness_required and not base_failed_checks:
            projection_context = load_attempt_bound_projection_context()
            canary_action = {
                'sonic_action_chunk': motion + left + right,
            }
            source_action_sha256 = canonical_sha256(canary_action)
            transport_call_count = [0]

            def retained_canary_transport(**kwargs):
                transport_call_count[0] += 1
                if transport_call_count[0] != 1:
                    raise RuntimeError(
                        'controller_fk_readiness_injected_transport_reused'
                    )
                if (
                    kwargs.get('motion_token') != motion
                    or kwargs.get('left_hand') != left
                    or kwargs.get('right_hand') != right
                    or int(kwargs.get('frame_index')) != -1
                    or kwargs.get('action_frames') is not None
                ):
                    raise RuntimeError(
                        'controller_fk_readiness_injected_transport_action_mismatch'
                    )
                controller_fk_readiness['checks'].update({
                    'retained_canary_state_reused': True,
                    'no_additional_wire_action_sent': True,
                })
                return dict(accepted)

            controller_fk_result = execute_controller_fk(
                {
                    'action': canary_action,
                    'source_action_sha256': source_action_sha256,
                    'step_index': -1,
                    'camera_projection_context': projection_context,
                },
                transport=retained_canary_transport,
            )
            source_frame = dict(
                projection_context.get('source_frame_artifact') or {}
            )
            registration = dict(
                controller_fk_result.get('cross_simulator_registration') or {}
            )
            joint_names = list(controller_fk_result.get('joint_names') or [])
            joint_positions = list(
                controller_fk_result.get('joint_positions') or []
            )
            applied_mapping = list(
                controller_fk_result.get('applied_dof_mapping') or []
            )
            landmarks = list(controller_fk_result.get('landmarks') or [])
            projection_context_sha256 = canonical_sha256(projection_context)
            source_frame_sha256 = str(source_frame.get('sha256') or '').lower()
            projections_bound = bool(landmarks) and all(
                isinstance(row, dict)
                and dict(row.get('image_projection') or {}).get(
                    'projection_context_sha256'
                ) == projection_context_sha256
                and dict(row.get('image_projection') or {}).get(
                    'source_frame_sha256'
                ) == source_frame_sha256
                for row in landmarks
            )
            projection_available = any(
                dict(row.get('image_projection') or {}).get('available') is True
                for row in landmarks if isinstance(row, dict)
            )
            projection_status_explicit = bool(landmarks) and all(
                (
                    dict(row.get('image_projection') or {}).get('available') is True
                    or bool(
                        dict(row.get('image_projection') or {}).get(
                            'unavailable_reason'
                        )
                    )
                )
                for row in landmarks if isinstance(row, dict)
            )
            mapping_bound = (
                bool(joint_names)
                and len(joint_names) == len(joint_positions) == len(applied_mapping)
                and all(
                    isinstance(row, dict)
                    and row.get('joint_name') == joint_names[index]
                    for index, row in enumerate(applied_mapping)
                )
            )
            controller_fk_readiness['checks'].update({
                'injected_transport_used_once': transport_call_count[0] == 1,
                'pinned_controller_revision_verified': bool(
                    controller_fk_result.get('controller_revision')
                ),
                'official_mujoco_fk_completed': bool(landmarks),
                'protocol_v4_joint_mapping_verified': mapping_bound,
                'standing_cross_simulator_registration_passed': (
                    registration.get('status') == 'passed'
                    and registration.get('surrogate') is False
                ),
                'live_camera_projection_completed': (
                    projections_bound and projection_status_explicit
                ),
                # ``execute_controller_fk`` receives only the retained state
                # and constructs local MuJoCo data. It has no Isaac backend or
                # completion-client handle with which to apply an action.
                'isaac_action_not_applied_by_subprobe': True,
            })
            controller_fk_readiness.update({
                'source_controller_readiness_probe_sha256': readiness[
                    'readiness_probe_sha256'
                ],
                'projection_context_sha256': projection_context_sha256,
                'source_frame_sha256': source_frame_sha256,
                'retained_controller_state_sha256': canonical_sha256(accepted),
                'controller_fk_result_sha256': canonical_sha256(
                    controller_fk_result
                ),
                'controller_revision_sha256': hashlib.sha256(
                    str(controller_fk_result.get('controller_revision') or '').encode(
                        'utf-8'
                    )
                ).hexdigest(),
                'controller_mapping_digest': controller_fk_result.get(
                    'mapping_digest'
                ),
                'robot_model_sha256': controller_fk_result.get(
                    'robot_model_sha256'
                ),
                'joint_names_sha256': canonical_sha256(joint_names),
                'joint_positions_sha256': canonical_sha256(joint_positions),
                'landmarks_sha256': canonical_sha256(landmarks),
                'standing_registration_sha256': canonical_sha256(registration),
                'live_camera_projection_summary': {
                    'landmark_count': len(landmarks),
                    'available_landmark_count': sum(
                        1
                        for row in landmarks
                        if isinstance(row, dict)
                        and dict(row.get('image_projection') or {}).get(
                            'available'
                        ) is True
                    ),
                    'any_landmark_in_initial_view': projection_available,
                    'visibility_required_for_controller_readiness': False,
                },
            })
            subprobe_failed_checks = sorted(
                name
                for name, passed in controller_fk_readiness['checks'].items()
                if passed is not True
            )
            if subprobe_failed_checks:
                controller_fk_readiness['blockers'].extend(
                    'controller_fk_readiness_check_failed:' + name
                    for name in subprobe_failed_checks
                )
            else:
                controller_fk_readiness['status'] = 'passed'
                readiness['checks']['controller_fk_readiness_subprobe'] = True
        failed_checks = sorted(
            name for name, passed in readiness['checks'].items() if passed is not True
        )
        if failed_checks:
            readiness['blockers'].extend(
                'controller_readiness_check_failed:' + name for name in failed_checks
            )
        else:
            readiness['status'] = 'ready'
    except (
        ImportError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        failure_detail = str(exc).strip()
        failure_code = failure_detail.split(':', 1)[0].strip()
        if (
            not failure_code
            or len(failure_code) > 160
            or any(
                character not in 'abcdefghijklmnopqrstuvwxyz0123456789_'
                for character in failure_code
            )
        ):
            failure_code = type(exc).__name__
        controller_fk_readiness['failure_code'] = failure_code
        controller_fk_readiness['failure_exception_type'] = type(exc).__name__
        controller_fk_readiness['failure_detail_sha256'] = (
            hashlib.sha256(failure_detail.encode('utf-8')).hexdigest()
            if failure_detail else None
        )
        if controller_fk_readiness_required:
            controller_fk_readiness['blockers'].append(
                'controller_fk_readiness_subprobe_failed:' + failure_code
            )
        readiness['blockers'].append(
            'official_gear_sonic_g1_debug_validation_failed:' + failure_code
        )

readiness['blockers'] = sorted(set(readiness['blockers']))
if (
    controller_fk_readiness_required
    and controller_fk_readiness['status'] != 'passed'
    and not controller_fk_readiness['blockers']
):
    controller_fk_readiness['blockers'].append(
        'controller_action_state_readiness_failed_before_fk_subprobe'
    )
controller_fk_readiness['blockers'] = sorted(
    set(controller_fk_readiness['blockers'])
)
controller_fk_readiness_path.write_text(
    json.dumps(controller_fk_readiness, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
readiness_path.write_text(
    json.dumps(readiness, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
if readiness['status'] != 'ready':
    raise SystemExit('official_gear_sonic_matching_g1_debug_not_ready')
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
if [ "${BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_REQUIRED:-}" = "true" ]; then
  upload_phase gear_sonic_isaac_dds_bridge_ready
fi
upload_phase gear_sonic_controller_ready
'''
