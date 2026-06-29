"""Build the operator packet for the first owner-GPU E2E run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .first_gpu_e2e_readiness import (
    CLAIM_BOUNDARY,
    FORWARD_PREFLIGHT_REPORT_ENV,
    PROVISIONERS,
    SIMULATOR_COMMAND_LOCATIONS,
    build_first_gpu_e2e_readiness,
)
from .local_capture import resolve_local_capture_context
from .simulation_automation import SIMULATOR_FRAMEWORKS


FIRST_GPU_RUN_PACKET_SCHEMA_VERSION = "first_gpu_run_packet.v1"
GPU_PROVIDER_BOOTSTRAP_SCHEMA_VERSION = "first_gpu_gpu_provider_bootstrap.v1"
FIRST_GPU_BLOCKER_RESOLUTION_SCHEMA_VERSION = "first_gpu_blocker_resolution.v1"
FIRST_GPU_VM_SYNC_SCHEMA_VERSION = "first_gpu_vm_sync_manifest.v1"
FIRST_GPU_SCENE_ASSET_ACQUISITION_SCHEMA_VERSION = "first_gpu_scene_asset_acquisition.v1"
FIRST_GPU_WEBAPP_HANDOFF_SCHEMA_VERSION = "first_gpu_webapp_handoff.v1"
FIRST_GPU_VM_RUNTIME_PREFLIGHT_PLAN_SCHEMA_VERSION = "first_gpu_vm_runtime_preflight_plan.v1"
FIRST_GPU_SIMULATOR_PATH_MATRIX_SCHEMA_VERSION = "first_gpu_simulator_path_matrix.v1"
FIRST_GPU_LAUNCH_ORDER_SCHEMA_VERSION = "first_gpu_launch_order.v1"
WEBAPP_HANDOFF_UPSTREAM_FIELDS = (
    "site_submission_id",
    "request_id",
    "buyer_request_id",
    "capture_job_id",
)
WORLDLABS_API_KEY_ENV = "WORLDLABS_API_KEY"
WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV = "BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION"
WEBAPP_FORWARD_URL_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"
WEBAPP_FORWARD_TOKEN_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN"
WEBAPP_FORWARD_REQUIRED_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED"
WEBAPP_FORWARD_CAPTURE_ROOT_ENV = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"
WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV = (
    "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"
)
WEBAPP_STAGED_INPUTS_ENV = "BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"
WEBAPP_UPSTREAM_EVIDENCE_SOURCES = (
    "raw/manifest.json",
    "capture_descriptor.json",
    "pipeline_handoff.json",
    "pipeline/opportunity_handoff.json",
    "robot_eval_job_request.v1 owner_system",
    "robot_eval_job_request.v1 site_package",
)
OWNER_COMMAND_TRACE_ENV_VARS = (
    "BLUEPRINT_SCENE_LOAD_TRACE",
    "BLUEPRINT_SPAWN_TRACE",
    "BLUEPRINT_ACTION_OR_POLICY_TRACE",
    "BLUEPRINT_DEFAULT_SMOKE_POLICY",
    "BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET",
    "BLUEPRINT_POLICY_EXECUTION_TRACE",
    "BLUEPRINT_SIM_ROBOT_POV_EVIDENCE",
    "BLUEPRINT_ARTIFACT_MANIFEST",
    "BLUEPRINT_OWNER_STDOUT",
    "BLUEPRINT_OWNER_STDERR",
)
OWNER_COMMAND_REQUIRED_OUTPUTS = (
    "pipeline/simulation_automation/gpu_owner_system_proof.json",
    "pipeline/simulation_automation/owner_gpu_simulator_execution_proof_manifest.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_scene_load_trace.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_spawn_pose_trace.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_default_smoke_policy.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_action_or_policy_trace.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_sim_robot_pov_evidence_manifest.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_artifact_manifest.json",
    "pipeline/simulation_automation/owner_gpu_proof/owner_simulator_stdout.log",
    "pipeline/simulation_automation/owner_gpu_proof/owner_simulator_stderr.log",
)
OWNER_DEFAULT_SMOKE_HELPER_COMMAND = "blueprint-write-owner-gpu-default-smoke-artifacts"
DEFAULT_ISAAC_G1_ASSET = {
    "name": "Unitree G1",
    "uri_or_path": "Robots/Unitree/G1/g1.usd",
    "source": "isaac_sim_robot_assets",
    "asset_class": "humanoid",
}
DEFAULT_MUJOCO_G1_MODEL_ROOT = (
    "output/external_assets/mujoco_menagerie/unitree_g1"
)
DEFAULT_MUJOCO_G1_ASSET = {
    "name": "Unitree G1",
    "uri_or_path": f"{DEFAULT_MUJOCO_G1_MODEL_ROOT}/g1.xml",
    "source": "google_deepmind_mujoco_menagerie",
    "asset_class": "humanoid_mjcf",
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _unique_strings(values: Sequence[Any]) -> list[str]:
    unique: list[str] = []
    for value in values:
        text = _string(value)
        if text and text not in unique:
            unique.append(text)
    return unique


def _gpu_vm_runtime_preflight_result_summary(result_path: Path) -> Dict[str, Any]:
    blockers: list[str] = []
    if not result_path.is_file():
        return {
            "path": str(result_path),
            "exists": False,
            "status": None,
            "ready_for_owner_command_attempt": False,
            "blockers": ["gpu_vm_runtime_preflight_result_missing"],
        }
    try:
        payload = read_json_any(result_path)
    except Exception as exc:  # pragma: no cover - emitted as packet evidence
        return {
            "path": str(result_path),
            "exists": True,
            "status": None,
            "ready_for_owner_command_attempt": False,
            "blockers": [
                f"gpu_vm_runtime_preflight_result_invalid_json:{exc.__class__.__name__}"
            ],
        }
    if not isinstance(payload, Mapping):
        return {
            "path": str(result_path),
            "exists": True,
            "status": None,
            "ready_for_owner_command_attempt": False,
            "blockers": [
                f"gpu_vm_runtime_preflight_result_invalid_payload:{type(payload).__name__}"
            ],
        }
    status = _string(payload.get("status"))
    for blocker in payload.get("blockers") or []:
        blockers.append(f"gpu_vm_runtime_preflight_result_blocker:{blocker}")
    if status != "ready_for_owner_command_attempt":
        blockers.append(
            f"gpu_vm_runtime_preflight_result_status:{status or 'unknown'}"
        )
    return {
        "path": str(result_path),
        "exists": True,
        "status": status or None,
        "ready_for_owner_command_attempt": not blockers,
        "blockers": _unique_strings(blockers),
    }


def _default_output_dir(capture_root: Path) -> Path:
    return capture_root / "pipeline" / "first_gpu_e2e_run_packet"


def _default_webapp_forwarding_preflight_path(capture_root: Path) -> Path:
    return capture_root / "pipeline" / "webapp_forwarding_preflight.json"


def _selected_webapp_forwarding_preflight_path(
    capture_root: Path,
    explicit: str | Path | None,
) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = _string(os.getenv(FORWARD_PREFLIGHT_REPORT_ENV))
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = _default_webapp_forwarding_preflight_path(capture_root)
    if candidate.is_file():
        return candidate.resolve()
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _robot_asset_for_simulator(simulator: str) -> Dict[str, str]:
    if simulator == "mujoco":
        return dict(DEFAULT_MUJOCO_G1_ASSET)
    return dict(DEFAULT_ISAAC_G1_ASSET)


def _mujoco_g1_model_root_path() -> Path:
    configured = _string(os.getenv("BLUEPRINT_MUJOCO_G1_MODEL_ROOT"))
    candidate = Path(configured or DEFAULT_MUJOCO_G1_MODEL_ROOT)
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    return candidate


def _mujoco_g1_asset_files() -> list[Path]:
    root = _mujoco_g1_model_root_path()
    files = [
        root / "g1.xml",
        root / "scene.xml",
        root / "LICENSE",
        root / "README.md",
    ]
    assets_dir = root / "assets"
    if assets_dir.is_dir():
        files.extend(sorted(assets_dir.glob("*")))
    return files


def _generated_mujoco_owner_command(packet_dir: Path) -> str:
    return f"bash {shlex.quote(str(packet_dir / 'run_mujoco_unitree_g1_smoke.sh'))}"


def _default_owner_command(simulator: str) -> str:
    if simulator == "isaac_sim":
        return "/opt/blueprint/run_isaac_gpu_proof.sh"
    if simulator == "isaac_lab_arena":
        return "/opt/blueprint/run_isaac_lab_arena_gpu_proof.sh"
    return "/opt/blueprint/run_owner_gpu_proof.sh"


def _shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def _shell_default_assignment(name: str, value: str) -> str:
    return "\n".join(
        [
            name + '="${' + name + ':-}"',
            'if [[ -z "${' + name + '}" ]]; then',
            f"  {name}={shlex.quote(value)}",
            "fi",
        ]
    )


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path, *, role: str, required: bool = True) -> Dict[str, Any]:
    exists = path.is_file()
    return {
        "role": role,
        "path": str(path),
        "required": required,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha_file(path) if exists else None,
        "blockers": [] if exists or not required else [f"missing_required_sync_file:{role}"],
    }


def _append_file_entry(entries: list[Dict[str, Any]], path: Path, *, role: str, required: bool = True) -> None:
    path_text = str(path)
    if any(item["path"] == path_text for item in entries):
        return
    entries.append(_file_entry(path, role=role, required=required))


def _raw_video_path(capture_root: Path) -> Path | None:
    raw_dir = capture_root / "raw"
    manifest = _read_mapping(raw_dir / "manifest.json")
    video_uri = _string(manifest.get("video_uri"))
    if video_uri and "://" not in video_uri:
        candidate = raw_dir / video_uri
        if candidate.is_file():
            return candidate
        candidate = capture_root / video_uri
        if candidate.is_file():
            return candidate
    for suffix in (".mov", ".mp4", ".m4v"):
        candidate = raw_dir / f"walkthrough{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _capture_root_by_site_json(*, site_slug: str, capture_root: Path) -> str:
    if not site_slug:
        return "{}"
    return json.dumps({site_slug: str(capture_root)}, sort_keys=True)


def _env_example(
    *,
    capture_root: Path,
    packet_dir: Path,
    webapp_site_slug: str,
    webapp_staged_inputs_path: Path,
    webapp_forwarding_preflight_path: Path | None,
    simulator: str,
    provisioner: str,
    owner_command: str,
) -> str:
    proof_dir = capture_root / "pipeline" / "simulation_automation" / "owner_gpu_proof"
    binding_path = packet_dir / "owner_default_smoke_command_binding.sh"
    robot_asset = _robot_asset_for_simulator(simulator)
    use_generated_smoke = simulator in {"isaac_sim", "isaac_lab_arena", "mujoco"}
    use_binding_default = "false" if use_generated_smoke else "true"
    if simulator in {"isaac_sim", "isaac_lab_arena"}:
        owner_command_line = 'export OWNER_SIMULATOR_COMMAND="$ISAAC_UNITREE_G1_SMOKE_COMMAND"'
    elif simulator == "mujoco":
        owner_command_line = 'export OWNER_SIMULATOR_COMMAND="$MUJOCO_UNITREE_G1_SMOKE_COMMAND"'
    else:
        owner_command_line = 'export OWNER_SIMULATOR_COMMAND="bash $OWNER_DEFAULT_SMOKE_COMMAND_BINDING"'
    lines = [
        "# Source this file after replacing placeholders. It intentionally contains no secrets.",
        _shell_export("CAPTURE_ROOT", str(capture_root)),
        _shell_export("PACKET_DIR", str(packet_dir)),
        _shell_export("WEBAPP_SITE_SLUG", webapp_site_slug or "<webapp-site-slug>"),
        _shell_export("GPU_PROOF_DIR", str(proof_dir)),
        _shell_export(
            FORWARD_PREFLIGHT_REPORT_ENV,
            str(webapp_forwarding_preflight_path or _default_webapp_forwarding_preflight_path(capture_root)),
        ),
        _shell_export(
            "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL",
            "https://<pipeline-host>/api/live-pipeline/job-requests",
        ),
        _shell_export("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "<set-in-shell-not-in-file>"),
        _shell_export("ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED", "true"),
        _shell_export(
            "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
            _capture_root_by_site_json(
                site_slug=webapp_site_slug or "<webapp-site-slug>",
                capture_root=capture_root,
            ),
        ),
        _shell_export("BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH", str(webapp_staged_inputs_path)),
        _shell_export("WORLDLABS_API_KEY", "<set-in-shell-not-in-file>"),
        _shell_export("BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION", "false"),
        _shell_export("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true"),
        _shell_export("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true"),
        _shell_export("BLUEPRINT_OWNER_SIMULATOR", simulator),
        _shell_export("BLUEPRINT_ROBOT_ASSET_NAME", robot_asset["name"]),
        _shell_export("BLUEPRINT_ROBOT_ASSET_URI_OR_PATH", robot_asset["uri_or_path"]),
        _shell_export("BLUEPRINT_ROBOT_ASSET_SOURCE", robot_asset["source"]),
        _shell_export("BLUEPRINT_ROBOT_ASSET_CLASS", robot_asset["asset_class"]),
        _shell_export("BLUEPRINT_MUJOCO_G1_MODEL_ROOT", str(_mujoco_g1_model_root_path())),
        _shell_export("BLUEPRINT_GPU_PROVISIONER", provisioner),
        _shell_export("OWNER_RAW_SIMULATOR_COMMAND", owner_command),
        'export ISAAC_SMOKE_SCRIPT="$PACKET_DIR/isaac_unitree_g1_smoke.py"',
        'export ISAAC_UNITREE_G1_SMOKE_COMMAND="bash $PACKET_DIR/run_isaac_unitree_g1_smoke.sh"',
        'export MUJOCO_UNITREE_G1_SMOKE_SCRIPT="$PACKET_DIR/mujoco_unitree_g1_smoke.py"',
        'export MUJOCO_UNITREE_G1_SMOKE_COMMAND="bash $PACKET_DIR/run_mujoco_unitree_g1_smoke.sh"',
        'export OWNER_DEFAULT_SMOKE_COMMAND_BINDING="$PACKET_DIR/owner_default_smoke_command_binding.sh"',
        _shell_export("BLUEPRINT_USE_DEFAULT_SMOKE_BINDING", use_binding_default),
        owner_command_line,
        'export ISAAC_OWNER_COMMAND="$OWNER_SIMULATOR_COMMAND"',
        _shell_export("OWNER_SCENE_LOAD_COMMAND", "<command-that-loads-scene-and-writes-BLUEPRINT_SCENE_LOAD_TRACE>"),
        _shell_export("OWNER_ROBOT_SPAWN_COMMAND", "<command-that-spawns-robot-and-writes-BLUEPRINT_SPAWN_TRACE>"),
        _shell_export("OWNER_WALK_TO_TARGET_COMMAND", "<command-that-runs-default-walk-to-target-policy>"),
        _shell_export("SIM_ROBOT_POV_FRAME_PATH", "<simulator-pov-frame-path>"),
        _shell_export("SIM_ROBOT_POV_VIDEO_PATH", ""),
        _shell_export("DEFAULT_POLICY_TARGET", "walk_to_target_pose"),
        _shell_export("OWNER_SYSTEM_ID", f"{provisioner}-<instance-id>"),
        _shell_export("SIMULATOR_VERSION", "<simulator-version>"),
        _shell_export("GPU_MODEL", "<gpu-model-from-nvidia-smi>"),
        _shell_export("OPERATOR_ID", "<operator-id>"),
        _shell_export(
            "OPERATOR_ATTESTATION",
            "I ran the first owner GPU simulator proof for this capture and preserved raw trace files.",
        ),
        _shell_export("SIMULATOR_TIMEOUT_SECONDS", "1800"),
        "",
        "# For the generated Isaac Unitree G1 smoke, set:",
        "# export BLUEPRINT_USE_DEFAULT_SMOKE_BINDING=false",
        "# export OWNER_SIMULATOR_COMMAND=\"$ISAAC_UNITREE_G1_SMOKE_COMMAND\"",
        "# For the generated MuJoCo Menagerie Unitree G1 smoke, set:",
        "# export BLUEPRINT_USE_DEFAULT_SMOKE_BINDING=false",
        "# export OWNER_SIMULATOR_COMMAND=\"$MUJOCO_UNITREE_G1_SMOKE_COMMAND\"",
        "# To use the split command binding instead, set:",
        "# export BLUEPRINT_USE_DEFAULT_SMOKE_BINDING=true",
        "# export OWNER_SIMULATOR_COMMAND=\"bash $OWNER_DEFAULT_SMOKE_COMMAND_BINDING\"",
        f"# Generated binding path: {binding_path}",
        "",
    ]
    return "\n".join(lines)


def _local_preflight_commands(
    *,
    capture_root: Path,
    webapp_site_slug: str,
    webapp_staged_inputs_path: Path,
    webapp_forwarding_preflight_path: Path | None,
    simulator: str,
    provisioner: str,
    owner_command: str,
    owner_command_location: str,
    allow_local_webapp_rehearsal: bool,
) -> str:
    slug_default = webapp_site_slug or "<webapp-site-slug>"
    local_rehearsal_flag = (
        "  --allow-local-webapp-rehearsal \\\n" if allow_local_webapp_rehearsal else ""
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("WEBAPP_SITE_SLUG", slug_default)}
{_shell_default_assignment("BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH", str(webapp_staged_inputs_path))}
{_shell_default_assignment(FORWARD_PREFLIGHT_REPORT_ENV, str(webapp_forwarding_preflight_path or _default_webapp_forwarding_preflight_path(capture_root)))}
{_shell_default_assignment("OWNER_SIMULATOR_COMMAND", owner_command)}

FORWARDING_PREFLIGHT_ARGS=()
if [[ -f "${{{FORWARD_PREFLIGHT_REPORT_ENV}}}" ]]; then
  FORWARDING_PREFLIGHT_ARGS=(--webapp-forwarding-preflight "${{{FORWARD_PREFLIGHT_REPORT_ENV}}}")
fi

blueprint-preflight-capture --capture-root "$CAPTURE_ROOT"

blueprint-run-e2e \\
  --capture-root "$CAPTURE_ROOT" \\
  --provider openai \\
  --pipeline-lane current \\
  --run-evaluation-prep \\
  --evaluation-prep-provider manual

blueprint-run-simulation-automation --capture-root "$CAPTURE_ROOT"

blueprint-audit-first-gpu-e2e-readiness \\
  --capture-root "$CAPTURE_ROOT" \\
  --webapp-site-slug "$WEBAPP_SITE_SLUG" \\
  --webapp-staged-inputs "$BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH" \\
  "${{FORWARDING_PREFLIGHT_ARGS[@]}}" \\
  --simulator {shlex.quote(simulator)} \\
  --provisioner {shlex.quote(provisioner)} \\
{local_rehearsal_flag}  --simulator-command "$OWNER_SIMULATOR_COMMAND" \\
  --simulator-command-location {shlex.quote(owner_command_location)}
"""


def _gpu_vm_commands(
    *,
    capture_root: Path,
    packet_dir: Path,
    simulator: str,
    owner_command: str,
) -> str:
    robot_asset = _robot_asset_for_simulator(simulator)
    use_binding_default = "false" if simulator in {"isaac_sim", "isaac_lab_arena", "mujoco"} else "true"
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("PACKET_DIR", str(packet_dir))}
GPU_PROOF_DIR="${{GPU_PROOF_DIR:-$CAPTURE_ROOT/pipeline/simulation_automation/owner_gpu_proof}}"
OWNER_DEFAULT_SMOKE_COMMAND_BINDING="${{OWNER_DEFAULT_SMOKE_COMMAND_BINDING:-$PACKET_DIR/owner_default_smoke_command_binding.sh}}"
ISAAC_SMOKE_SCRIPT="${{ISAAC_SMOKE_SCRIPT:-$PACKET_DIR/isaac_unitree_g1_smoke.py}}"
ISAAC_UNITREE_G1_SMOKE_COMMAND="${{ISAAC_UNITREE_G1_SMOKE_COMMAND:-bash $PACKET_DIR/run_isaac_unitree_g1_smoke.sh}}"
MUJOCO_UNITREE_G1_SMOKE_SCRIPT="${{MUJOCO_UNITREE_G1_SMOKE_SCRIPT:-$PACKET_DIR/mujoco_unitree_g1_smoke.py}}"
MUJOCO_UNITREE_G1_SMOKE_COMMAND="${{MUJOCO_UNITREE_G1_SMOKE_COMMAND:-bash $PACKET_DIR/run_mujoco_unitree_g1_smoke.sh}}"
BLUEPRINT_MUJOCO_G1_MODEL_ROOT="${{BLUEPRINT_MUJOCO_G1_MODEL_ROOT:-{shlex.quote(str(_mujoco_g1_model_root_path()))}}}"
BLUEPRINT_USE_DEFAULT_SMOKE_BINDING="${{BLUEPRINT_USE_DEFAULT_SMOKE_BINDING:-{use_binding_default}}}"
{_shell_default_assignment("OWNER_RAW_SIMULATOR_COMMAND", owner_command)}
OWNER_SIMULATOR_COMMAND="${{OWNER_SIMULATOR_COMMAND:-${{ISAAC_OWNER_COMMAND:-}}}}"
if [[ "$BLUEPRINT_USE_DEFAULT_SMOKE_BINDING" == "true" ]]; then
  OWNER_SIMULATOR_COMMAND="bash $OWNER_DEFAULT_SMOKE_COMMAND_BINDING"
elif [[ -z "${{OWNER_SIMULATOR_COMMAND}}" ]]; then
  if [[ {shlex.quote(simulator)} == "isaac_sim" || {shlex.quote(simulator)} == "isaac_lab_arena" ]]; then
    OWNER_SIMULATOR_COMMAND="$ISAAC_UNITREE_G1_SMOKE_COMMAND"
  elif [[ {shlex.quote(simulator)} == "mujoco" ]]; then
    OWNER_SIMULATOR_COMMAND="$MUJOCO_UNITREE_G1_SMOKE_COMMAND"
  else
    OWNER_SIMULATOR_COMMAND="$OWNER_RAW_SIMULATOR_COMMAND"
  fi
fi
export OWNER_SIMULATOR_COMMAND
SIMULATOR_TIMEOUT_SECONDS="${{SIMULATOR_TIMEOUT_SECONDS:-1800}}"
DEFAULT_POLICY_TARGET="${{DEFAULT_POLICY_TARGET:-walk_to_target_pose}}"
BLUEPRINT_ROBOT_ASSET_NAME="${{BLUEPRINT_ROBOT_ASSET_NAME:-{robot_asset["name"]}}}"
BLUEPRINT_ROBOT_ASSET_URI_OR_PATH="${{BLUEPRINT_ROBOT_ASSET_URI_OR_PATH:-{robot_asset["uri_or_path"]}}}"
BLUEPRINT_ROBOT_ASSET_SOURCE="${{BLUEPRINT_ROBOT_ASSET_SOURCE:-{robot_asset["source"]}}}"
BLUEPRINT_ROBOT_ASSET_CLASS="${{BLUEPRINT_ROBOT_ASSET_CLASS:-{robot_asset["asset_class"]}}}"
: "${{OWNER_SYSTEM_ID:?Set OWNER_SYSTEM_ID to the GPU instance or pod id}}"
: "${{SIMULATOR_VERSION:?Set SIMULATOR_VERSION from the installed simulator}}"
: "${{GPU_MODEL:?Set GPU_MODEL from nvidia-smi}}"
: "${{OPERATOR_ID:?Set OPERATOR_ID for the human or automation owner}}"
: "${{OPERATOR_ATTESTATION:?Set OPERATOR_ATTESTATION for this run}}"

blueprint-run-owner-gpu-proof \\
  --capture-root "$CAPTURE_ROOT" \\
  --command "$OWNER_SIMULATOR_COMMAND" \\
  --proof-dir "$GPU_PROOF_DIR" \\
  --owner-system-id "$OWNER_SYSTEM_ID" \\
  --simulator-backend {shlex.quote(simulator)} \\
  --simulator-version "$SIMULATOR_VERSION" \\
  --gpu-model "$GPU_MODEL" \\
  --operator-id "$OPERATOR_ID" \\
  --operator-attestation "$OPERATOR_ATTESTATION" \\
  --timeout-seconds "$SIMULATOR_TIMEOUT_SECONDS" \\
  --default-policy-target "$DEFAULT_POLICY_TARGET" \\
  --robot-asset-name "$BLUEPRINT_ROBOT_ASSET_NAME" \\
  --robot-asset-uri-or-path "$BLUEPRINT_ROBOT_ASSET_URI_OR_PATH" \\
  --robot-asset-source "$BLUEPRINT_ROBOT_ASSET_SOURCE" \\
  --robot-asset-class "$BLUEPRINT_ROBOT_ASSET_CLASS"

BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true \\
blueprint-run-simulation-automation \\
  --capture-root "$CAPTURE_ROOT" \\
  --allow-simulator-execution \\
  --allow-simulator {shlex.quote(simulator)} \\
  --simulator-command "{simulator}=$OWNER_SIMULATOR_COMMAND"
"""


def _owner_command_binding_template(*, simulator: str) -> str:
    quoted_simulator = shlex.quote(simulator)
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Fail-closed template for OWNER_SIMULATOR_COMMAND.
# Run this through blueprint-run-owner-gpu-proof after replacing the three
# OWNER_*_COMMAND values with commands that operate the selected simulator.

: "${{BLUEPRINT_CAPTURE_ROOT:?Set by blueprint-run-owner-gpu-proof}}"
: "${{BLUEPRINT_SCENE_LOAD_TRACE:?Set by blueprint-run-owner-gpu-proof}}"
: "${{BLUEPRINT_SPAWN_TRACE:?Set by blueprint-run-owner-gpu-proof}}"
: "${{BLUEPRINT_POLICY_EXECUTION_TRACE:?Set by blueprint-run-owner-gpu-proof}}"
: "${{BLUEPRINT_SIM_ROBOT_POV_EVIDENCE:?Set by blueprint-run-owner-gpu-proof}}"
: "${{BLUEPRINT_ARTIFACT_MANIFEST:?Set by blueprint-run-owner-gpu-proof}}"
: "${{BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET:?Set by blueprint-run-owner-gpu-proof}}"
if [[ {quoted_simulator} == "isaac_sim" || {quoted_simulator} == "isaac_lab_arena" ]]; then
  : "${{BLUEPRINT_ROBOT_ASSET_NAME:?Set by blueprint-run-owner-gpu-proof; expected Unitree G1}}"
  : "${{BLUEPRINT_ROBOT_ASSET_URI_OR_PATH:?Set by blueprint-run-owner-gpu-proof; expected Robots/Unitree/G1/g1.usd}}"
  : "${{BLUEPRINT_ROBOT_ASSET_SOURCE:?Set by blueprint-run-owner-gpu-proof; expected isaac_sim_robot_assets}}"
fi
: "${{OWNER_SCENE_LOAD_COMMAND:?Set a command that loads the scene and writes BLUEPRINT_SCENE_LOAD_TRACE}}"
: "${{OWNER_ROBOT_SPAWN_COMMAND:?Set a command that spawns the robot and writes BLUEPRINT_SPAWN_TRACE}}"
: "${{OWNER_WALK_TO_TARGET_COMMAND:?Set a command that runs the default walk_to_target policy}}"

if [[ -z "${{SIM_ROBOT_POV_FRAME_PATH:-}}" && -z "${{SIM_ROBOT_POV_VIDEO_PATH:-}}" ]]; then
  echo "Set SIM_ROBOT_POV_FRAME_PATH or SIM_ROBOT_POV_VIDEO_PATH to evidence emitted by the simulator." >&2
  exit 11
fi

echo "[owner-binding] loading scene"
bash -lc "$OWNER_SCENE_LOAD_COMMAND"
test -s "$BLUEPRINT_SCENE_LOAD_TRACE"

echo "[owner-binding] spawning robot"
bash -lc "$OWNER_ROBOT_SPAWN_COMMAND"
test -s "$BLUEPRINT_SPAWN_TRACE"

echo "[owner-binding] running default walk_to_target policy"
bash -lc "$OWNER_WALK_TO_TARGET_COMMAND"

helper_args=(
  --simulator {quoted_simulator}
  --target "$BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET"
)
if [[ -n "${{SIM_ROBOT_POV_FRAME_PATH:-}}" ]]; then
  test -s "$SIM_ROBOT_POV_FRAME_PATH"
  helper_args+=(--sim-pov-frame "$SIM_ROBOT_POV_FRAME_PATH")
fi
if [[ -n "${{SIM_ROBOT_POV_VIDEO_PATH:-}}" ]]; then
  test -s "$SIM_ROBOT_POV_VIDEO_PATH"
  helper_args+=(--sim-pov-video "$SIM_ROBOT_POV_VIDEO_PATH")
fi

"${{PYTHON:-python3}}" -m blueprint_pipeline.owner_gpu_default_smoke_artifacts "${{helper_args[@]}}"
test -s "$BLUEPRINT_POLICY_EXECUTION_TRACE"
test -s "$BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"
test -s "$BLUEPRINT_ARTIFACT_MANIFEST"
echo "[owner-binding] default smoke policy and simulator POV artifacts written"
"""


def _isaac_unitree_g1_smoke_script() -> str:
    return r'''#!/usr/bin/env python3
"""Isaac Sim Unitree G1 owner proof smoke for Blueprint first-GPU packets."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def find_scene_glb(capture_root: Path) -> Path:
    candidates = [
        capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb",
        capture_root / "pipeline" / "worldlabs_assets" / "scene.glb",
        capture_root / "pipeline" / "marble_sim_assets" / "portable_collider.glb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted((capture_root / "pipeline").glob("**/*.glb"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no GLB scene asset found under {capture_root / 'pipeline'}")


def asset_candidates(asset_path: str) -> list[str]:
    candidates: list[str] = []
    raw = asset_path.strip()
    if raw:
        candidates.append(raw)
    assets_root = ""
    try:
        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path() or ""
    except Exception:
        try:
            from omni.isaac.core.utils.nucleus import get_assets_root_path

            assets_root = get_assets_root_path() or ""
        except Exception:
            assets_root = ""
    if assets_root and raw and not raw.startswith(("omniverse://", "http://", "https://", "/")):
        candidates.append(assets_root.rstrip("/") + "/Isaac/" + raw.lstrip("/"))
        candidates.append(assets_root.rstrip("/") + "/" + raw.lstrip("/"))
    return list(dict.fromkeys(candidates))


async def convert_glb_to_usd(glb_path: Path, output_usd: Path) -> dict[str, Any]:
    try:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
    except Exception as exc:
        print(f"[asset-converter] extension enable warning: {exc}", flush=True)
    import omni.kit.asset_converter

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_camera = False
    context.ignore_light = False
    context.export_preview_surface = True
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = False
    context.merge_all_meshes = False
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(glb_path),
        str(output_usd),
        lambda current, total: print(f"[asset-converter] {current}/{total}", flush=True),
        context,
    )
    success = await task.wait_until_finished()
    status = getattr(task, "get_status", lambda: None)()
    error = getattr(task, "get_error_message", lambda: "")()
    if not success:
        raise RuntimeError(f"World Labs GLB to USD conversion failed: status={status} error={error}")
    return {
        "status": "converted",
        "source_glb": str(glb_path),
        "converted_usd": str(output_usd),
        "converter": "omni.kit.asset_converter",
        "converter_status": str(status),
    }


def import_simulation_app():
    try:
        from isaacsim import SimulationApp

        return SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp

        return SimulationApp


def import_add_reference_to_stage():
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage

        return add_reference_to_stage
    except Exception:
        from omni.isaac.core.utils.stage import add_reference_to_stage

        return add_reference_to_stage


def import_world():
    try:
        from isaacsim.core.api import World

        return World
    except Exception:
        from omni.isaac.core import World

        return World


def render_camera_frame(camera: Any, frame_path: Path) -> str:
    from PIL import Image

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    image = None
    for method_name in ("get_rgba", "get_rgb"):
        method = getattr(camera, method_name, None)
        if callable(method):
            image = method()
            if image is not None:
                break
    if image is None:
        raise RuntimeError("Isaac camera did not return RGB/RGBA data")
    Image.fromarray(image).save(frame_path)
    return str(frame_path)


def main() -> int:
    capture_root = Path(required_env("BLUEPRINT_CAPTURE_ROOT"))
    proof_dir = Path(os.environ.get("BLUEPRINT_GPU_PROOF_DIR", capture_root / "pipeline" / "simulation_automation" / "owner_gpu_proof"))
    proof_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = proof_dir / "isaac_sim_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene_trace_path = required_env("BLUEPRINT_SCENE_LOAD_TRACE")
    spawn_trace_path = required_env("BLUEPRINT_SPAWN_TRACE")
    policy_trace_path = required_env("BLUEPRINT_POLICY_EXECUTION_TRACE")
    pov_manifest_path = required_env("BLUEPRINT_SIM_ROBOT_POV_EVIDENCE")
    artifact_manifest_path = required_env("BLUEPRINT_ARTIFACT_MANIFEST")
    target_label = os.environ.get("BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET", "walk_to_target_pose")
    robot_asset = {
        "name": os.environ.get("BLUEPRINT_ROBOT_ASSET_NAME", "Unitree G1"),
        "uri_or_path": os.environ.get("BLUEPRINT_ROBOT_ASSET_URI_OR_PATH", "Robots/Unitree/G1/g1.usd"),
        "source": os.environ.get("BLUEPRINT_ROBOT_ASSET_SOURCE", "isaac_sim_robot_assets"),
        "asset_class": os.environ.get("BLUEPRINT_ROBOT_ASSET_CLASS", "humanoid"),
    }

    SimulationApp = import_simulation_app()
    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": os.environ.get("ISAAC_RENDERER", "RayTracedLighting"),
            "width": int(os.environ.get("ISAAC_RENDER_WIDTH", "1280")),
            "height": int(os.environ.get("ISAAC_RENDER_HEIGHT", "720")),
        }
    )
    try:
        import numpy as np
        from pxr import Gf, UsdGeom

        add_reference_to_stage = import_add_reference_to_stage()
        World = import_world()

        scene_glb = find_scene_glb(capture_root)
        converted_scene_usd = proof_dir / "worldlabs_scene_converted.usd"
        conversion = asyncio.get_event_loop().run_until_complete(
            convert_glb_to_usd(scene_glb, converted_scene_usd)
        )
        world = World(stage_units_in_meters=1.0)
        add_reference_to_stage(usd_path=str(converted_scene_usd), prim_path="/World/BlueprintWorldLabsScene")

        selected_asset_path = None
        asset_errors: list[str] = []
        for candidate in asset_candidates(robot_asset["uri_or_path"]):
            try:
                add_reference_to_stage(usd_path=candidate, prim_path="/World/UnitreeG1")
                selected_asset_path = candidate
                break
            except Exception as exc:
                asset_errors.append(f"{candidate}: {exc}")
        if not selected_asset_path:
            raise RuntimeError("Unable to reference Unitree G1 USD asset: " + "; ".join(asset_errors))
        robot_asset["resolved_uri_or_path"] = selected_asset_path

        stage = world.stage
        camera_prim_path = "/World/BlueprintSimRobotPOV"
        camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path)
        camera_prim.GetFocalLengthAttr().Set(18.0)
        camera_xform = UsdGeom.Xformable(camera_prim.GetPrim())
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(-1.1, -1.4, 1.35))
        camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(68.0, 0.0, -38.0))

        world.reset()
        try:
            from isaacsim.sensors.camera import Camera
        except Exception:
            from omni.isaac.sensor import Camera

        camera = Camera(prim_path=camera_prim_path, resolution=(1280, 720))
        camera.initialize()

        start = np.array([-0.8, 0.0, 0.793])
        target = np.array([0.8, 0.0, 0.793])
        steps = int(os.environ.get("BLUEPRINT_ISAAC_SMOKE_STEPS", "48"))
        actions: list[dict[str, Any]] = []
        frame_paths: list[str] = []
        for step in range(steps):
            alpha = 0.0 if steps <= 1 else step / float(steps - 1)
            pose = (start + (target - start) * alpha).tolist()
            robot_prim = stage.GetPrimAtPath("/World/UnitreeG1")
            if robot_prim.IsValid():
                xform = UsdGeom.Xformable(robot_prim)
                if not xform.GetOrderedXformOps():
                    xform.AddTranslateOp()
                xform.GetOrderedXformOps()[0].Set(Gf.Vec3d(*pose))
            world.step(render=True)
            actions.append({"step": step, "root_position": pose, "target": target.tolist()})
            if step in {0, max(0, steps // 2), max(0, steps - 1)}:
                frame_paths.append(render_camera_frame(camera, frames_dir / f"isaac_robot_pov_{step:04d}.png"))

        scene_trace = {
            "schema_version": "owner_gpu_scene_load_trace.v1",
            "status": "loaded",
            "scene_loaded": True,
            "simulator_backend": "isaac_sim",
            "source_scene_glb": str(scene_glb),
            "converted_scene_usd": str(converted_scene_usd),
            "conversion": conversion,
            "robot_asset": robot_asset,
            "recorded_at": now(),
        }
        spawn_trace = {
            "schema_version": "owner_gpu_spawn_pose_trace.v1",
            "status": "validated",
            "spawn_pose_loaded": True,
            "simulator_backend": "isaac_sim",
            "robot_asset": robot_asset,
            "spawn_pose": start.tolist(),
            "target_pose": target.tolist(),
            "recorded_at": now(),
        }
        policy_trace = {
            "schema_version": "owner_gpu_policy_execution_trace.v1",
            "status": "completed",
            "policy_id": "blueprint_default_walk_to_target_smoke_policy",
            "policy_kind": "walk_to_target",
            "policy_source": "repo_generated_default_smoke_policy",
            "policy_downloaded_from_online": False,
            "target_label": target_label,
            "default_policy_executed": True,
            "policy_execution_completed": True,
            "policy_semantics": "kinematic_root_pose_smoke_not_balanced_humanoid_locomotion_controller",
            "actions": actions,
            "recorded_at": now(),
        }
        pov_manifest = {
            "schema_version": "owner_gpu_sim_robot_pov_evidence.v1",
            "status": "complete",
            "simulator_backend": "isaac_sim",
            "sim_robot_pov_captured": True,
            "frames": [{"camera": "BlueprintSimRobotPOV", "path": path} for path in frame_paths],
            "frame_count": len(frame_paths),
            "camera_boundary": "Isaac Sim virtual camera evidence, not physical robot POV.",
            "recorded_at": now(),
        }
        artifact_manifest = {
            "schema_version": "owner_gpu_artifact_manifest.v1",
            "status": "complete",
            "simulator_backend": "isaac_sim",
            "robot_asset": robot_asset,
            "artifacts": {
                "scene_trace": scene_trace_path,
                "spawn_trace": spawn_trace_path,
                "policy_trace": policy_trace_path,
                "sim_robot_pov_evidence": pov_manifest_path,
                "source_scene_glb": str(scene_glb),
                "converted_scene_usd": str(converted_scene_usd),
                "frames": frame_paths,
            },
            "files": [str(converted_scene_usd), *frame_paths],
            "recorded_at": now(),
        }
        write_json(scene_trace_path, scene_trace)
        write_json(spawn_trace_path, spawn_trace)
        write_json(policy_trace_path, policy_trace)
        write_json(pov_manifest_path, pov_manifest)
        write_json(artifact_manifest_path, artifact_manifest)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _isaac_unitree_g1_smoke_launcher() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

: "${PACKET_DIR:?Set PACKET_DIR to the first_gpu_e2e_run_packet directory}"
: "${BLUEPRINT_CAPTURE_ROOT:?Set by blueprint-run-owner-gpu-proof}"

ISAAC_SMOKE_SCRIPT="${ISAAC_SMOKE_SCRIPT:-$PACKET_DIR/isaac_unitree_g1_smoke.py}"
if [[ ! -s "$ISAAC_SMOKE_SCRIPT" ]]; then
  echo "Missing Isaac smoke script: $ISAAC_SMOKE_SCRIPT" >&2
  exit 12
fi

if [[ -n "${ISAAC_PYTHON:-}" ]]; then
  exec "$ISAAC_PYTHON" "$ISAAC_SMOKE_SCRIPT"
fi
if command -v python.sh >/dev/null 2>&1; then
  exec python.sh "$ISAAC_SMOKE_SCRIPT"
fi
if [[ -x "/isaac-sim/python.sh" ]]; then
  exec /isaac-sim/python.sh "$ISAAC_SMOKE_SCRIPT"
fi
if [[ -x "/workspace/isaac-sim/python.sh" ]]; then
  exec /workspace/isaac-sim/python.sh "$ISAAC_SMOKE_SCRIPT"
fi

echo "Set ISAAC_PYTHON to Isaac Sim python.sh, or run inside an Isaac Sim container exposing python.sh." >&2
exit 13
"""


def _mujoco_unitree_g1_smoke_script() -> str:
    script_path = _repo_root() / "scripts" / "owner_gpu_mujoco_walk_to_target_smoke.py"
    if script_path.is_file():
        return script_path.read_text(encoding="utf-8")
    return """#!/usr/bin/env python3
import sys

raise SystemExit(
    "owner_gpu_mujoco_walk_to_target_smoke.py was not available when this packet was built; "
    "regenerate the packet from the BlueprintCapturePipeline repo."
)
"""


def _mujoco_unitree_g1_smoke_launcher() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

: "${PACKET_DIR:?Set PACKET_DIR to the first_gpu_e2e_run_packet directory}"
: "${BLUEPRINT_CAPTURE_ROOT:?Set by blueprint-run-owner-gpu-proof}"

MUJOCO_UNITREE_G1_SMOKE_SCRIPT="${MUJOCO_UNITREE_G1_SMOKE_SCRIPT:-$PACKET_DIR/mujoco_unitree_g1_smoke.py}"
BLUEPRINT_MUJOCO_G1_MODEL_ROOT="${BLUEPRINT_MUJOCO_G1_MODEL_ROOT:-}"
if [[ -z "$BLUEPRINT_MUJOCO_G1_MODEL_ROOT" ]]; then
  BLUEPRINT_MUJOCO_G1_MODEL_ROOT="output/external_assets/mujoco_menagerie/unitree_g1"
fi
if [[ ! -s "$MUJOCO_UNITREE_G1_SMOKE_SCRIPT" ]]; then
  echo "Missing MuJoCo smoke script: $MUJOCO_UNITREE_G1_SMOKE_SCRIPT" >&2
  exit 12
fi
if [[ ! -s "$BLUEPRINT_MUJOCO_G1_MODEL_ROOT/g1.xml" ]]; then
  echo "Missing MuJoCo Menagerie Unitree G1 model: $BLUEPRINT_MUJOCO_G1_MODEL_ROOT/g1.xml" >&2
  exit 14
fi

exec "${PYTHON:-python3}" "$MUJOCO_UNITREE_G1_SMOKE_SCRIPT" \
  --capture-root "$BLUEPRINT_CAPTURE_ROOT" \
  --g1-model-root "$BLUEPRINT_MUJOCO_G1_MODEL_ROOT"
"""


def _gpu_vm_runtime_preflight_commands(
    *,
    capture_root: Path,
    packet_dir: Path,
    simulator: str,
    owner_command: str,
) -> str:
    use_binding_default = "false" if simulator in {"isaac_sim", "isaac_lab_arena", "mujoco"} else "true"
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("PACKET_DIR", str(packet_dir))}
GPU_VM_SYNC_MANIFEST="${{GPU_VM_SYNC_MANIFEST:-$PACKET_DIR/gpu_vm_sync_manifest.json}}"
GPU_VM_PREFLIGHT_OUTPUT="${{GPU_VM_PREFLIGHT_OUTPUT:-$PACKET_DIR/gpu_vm_runtime_preflight_result.json}}"
OWNER_DEFAULT_SMOKE_COMMAND_BINDING="${{OWNER_DEFAULT_SMOKE_COMMAND_BINDING:-$PACKET_DIR/owner_default_smoke_command_binding.sh}}"
ISAAC_UNITREE_G1_SMOKE_COMMAND="${{ISAAC_UNITREE_G1_SMOKE_COMMAND:-bash $PACKET_DIR/run_isaac_unitree_g1_smoke.sh}}"
MUJOCO_UNITREE_G1_SMOKE_COMMAND="${{MUJOCO_UNITREE_G1_SMOKE_COMMAND:-bash $PACKET_DIR/run_mujoco_unitree_g1_smoke.sh}}"
BLUEPRINT_MUJOCO_G1_MODEL_ROOT="${{BLUEPRINT_MUJOCO_G1_MODEL_ROOT:-{shlex.quote(str(_mujoco_g1_model_root_path()))}}}"
BLUEPRINT_USE_DEFAULT_SMOKE_BINDING="${{BLUEPRINT_USE_DEFAULT_SMOKE_BINDING:-{use_binding_default}}}"
{_shell_default_assignment("OWNER_RAW_SIMULATOR_COMMAND", owner_command)}
OWNER_SIMULATOR_COMMAND="${{OWNER_SIMULATOR_COMMAND:-${{ISAAC_OWNER_COMMAND:-}}}}"
if [[ "$BLUEPRINT_USE_DEFAULT_SMOKE_BINDING" == "true" ]]; then
  OWNER_SIMULATOR_COMMAND="bash $OWNER_DEFAULT_SMOKE_COMMAND_BINDING"
elif [[ -z "${{OWNER_SIMULATOR_COMMAND}}" ]]; then
  if [[ {shlex.quote(simulator)} == "isaac_sim" || {shlex.quote(simulator)} == "isaac_lab_arena" ]]; then
    OWNER_SIMULATOR_COMMAND="$ISAAC_UNITREE_G1_SMOKE_COMMAND"
  elif [[ {shlex.quote(simulator)} == "mujoco" ]]; then
    OWNER_SIMULATOR_COMMAND="$MUJOCO_UNITREE_G1_SMOKE_COMMAND"
  else
    OWNER_SIMULATOR_COMMAND="$OWNER_RAW_SIMULATOR_COMMAND"
  fi
fi
export OWNER_SIMULATOR_COMMAND

"${{PYTHON:-python3}}" - "$CAPTURE_ROOT" "$PACKET_DIR" "$GPU_VM_SYNC_MANIFEST" "$GPU_VM_PREFLIGHT_OUTPUT" "$OWNER_SIMULATOR_COMMAND" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_probe(command: str) -> dict:
    text = (command or "").strip()
    if not text:
        return dict(
            configured=False,
            executable=None,
            executable_found=False,
            blockers=["owner_simulator_command_missing"],
        )
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return dict(
            configured=True,
            executable=None,
            executable_found=False,
            blockers=["owner_simulator_command_parse_failed"],
            error=str(exc),
        )
    executable = parts[0] if parts else ""
    if "=" in executable and len(parts) > 1:
        executable = parts[1]
    found = bool(shutil.which(executable)) if "/" not in executable else Path(executable).exists()
    return dict(
        configured=True,
        executable=executable or None,
        executable_found=found,
        blockers=[] if found else ["owner_simulator_command_executable_missing_on_vm"],
    )


def run_probe(command: list[str], timeout_seconds: int = 20) -> dict:
    executable = shutil.which(command[0])
    if not executable:
        return dict(found=False, returncode=None, stdout="", stderr="", blockers=[command[0] + "_missing"])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return dict(
        found=True,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        blockers=[] if completed.returncode == 0 else [command[0] + "_failed"],
    )


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for token in str(value or "").strip().split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _nvidia_smi_rows(probe: dict) -> list[dict]:
    rows: list[dict] = []
    for line in str(probe.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append(dict(name=parts[0], memory_total=parts[1], driver_version=parts[2]))
    return rows


def isaac_runtime_probe(simulator: str, nvidia_smi_probe: dict) -> dict:
    if simulator not in ("isaac_sim", "isaac_lab_arena"):
        return dict(status="not_applicable", blockers=[], warnings=[])
    minimum_driver = os.getenv("BLUEPRINT_ISAAC_MIN_DRIVER_VERSION", "580.65.06")
    blockers: list[str] = []
    warnings: list[str] = []
    rows = _nvidia_smi_rows(nvidia_smi_probe)
    if not rows:
        blockers.append("isaac_gpu_query_missing")
    for row in rows:
        gpu_name = str(row.get("name") or "")
        upper_name = gpu_name.upper()
        if "A100" in upper_name or "H100" in upper_name:
            blockers.append("isaac_unsupported_non_rt_core_gpu:" + gpu_name)
        driver_version = str(row.get("driver_version") or "")
        if _version_tuple(driver_version) < _version_tuple(minimum_driver):
            blockers.append(
                "isaac_driver_below_minimum:" + driver_version + "<" + minimum_driver
            )
    vulkaninfo = run_probe(["vulkaninfo", "--summary"], timeout_seconds=30)
    if not vulkaninfo.get("found"):
        blockers.append("vulkaninfo_missing_for_isaac_runtime_preflight")
    elif vulkaninfo.get("returncode") != 0:
        blockers.append("vulkaninfo_failed_for_isaac_runtime_preflight")
    if "llvmpipe" in str(vulkaninfo.get("stdout") or "").lower():
        blockers.append("vulkaninfo_reports_cpu_renderer_for_isaac_runtime")
    if not shutil.which("docker"):
        warnings.append("docker_missing_for_official_isaac_container_path")
    return dict(
        status="ready" if not blockers else "blocked",
        minimum_driver_version=minimum_driver,
        nvidia_gpu_rows=rows,
        vulkaninfo=vulkaninfo,
        blockers=_unique(blockers),
        warnings=_unique(warnings),
    )


def mujoco_runtime_probe(simulator: str) -> dict:
    if simulator != "mujoco":
        return dict(status="not_applicable", blockers=[], warnings=[])
    python_exe = os.getenv("PYTHON") or sys.executable
    mujoco_import = run_probe(
        [python_exe, "-c", "import mujoco; print(getattr(mujoco, '__version__', 'unknown'))"],
        timeout_seconds=30,
    )
    g1_root = Path(os.getenv("BLUEPRINT_MUJOCO_G1_MODEL_ROOT", "")).expanduser()
    g1_xml = g1_root / "g1.xml" if str(g1_root) else Path()
    blockers = []
    if mujoco_import.get("blockers"):
        blockers.append("mujoco_python_import_failed")
    if not g1_xml.is_file():
        blockers.append("mujoco_menagerie_unitree_g1_xml_missing")
    return dict(
        status="ready" if not blockers else "blocked",
        python=python_exe,
        mujoco_import=mujoco_import,
        g1_model_root=str(g1_root) if str(g1_root) else None,
        g1_xml_exists=g1_xml.is_file() if str(g1_root) else False,
        blockers=_unique(blockers),
        warnings=[],
    )


def verify_sync_manifest(path: Path) -> dict:
    if not path.is_file():
        return dict(
            path=str(path),
            exists=False,
            checked_count=0,
            missing_files=[],
            mismatches=[],
            blockers=["gpu_vm_sync_manifest_missing"],
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    checked = []
    missing = []
    mismatches = []
    manifest_blockers = [str(item) for item in payload.get("blockers") or []]
    blockers = ["sync_manifest_blocker:" + item for item in manifest_blockers]
    for item in payload.get("files") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        expected_path = Path(str(item.get("path") or ""))
        expected_sha = item.get("sha256")
        expected_exists = bool(item.get("exists"))
        required = bool(item.get("required", True))
        if expected_exists and not expected_path.is_file():
            missing.append(dict(role=role, path=str(expected_path)))
            blockers.append("synced_file_missing:" + role)
            continue
        if required and not expected_exists:
            blockers.append("sync_manifest_required_file_absent:" + role)
            continue
        if expected_exists and expected_sha:
            actual_sha = sha_file(expected_path)
            checked.append(dict(role=role, path=str(expected_path), sha256=actual_sha))
            if actual_sha != expected_sha:
                mismatches.append(
                    dict(
                        role=role,
                        path=str(expected_path),
                        expected_sha256=expected_sha,
                        actual_sha256=actual_sha,
                    )
                )
                blockers.append("sha256_mismatch:" + role)
    return dict(
        path=str(path),
        exists=True,
        checked_count=len(checked),
        missing_files=missing,
        mismatches=mismatches,
        manifest_blockers=manifest_blockers,
        blockers=blockers,
    )


capture_root = Path(sys.argv[1])
packet_dir = Path(sys.argv[2])
sync_manifest_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
owner_command = sys.argv[5]
simulator_name = {json.dumps(simulator)}
requires_gpu_probe = simulator_name in ("isaac_sim", "isaac_lab_arena", "newton")

nvidia_smi = run_probe(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
)
docker = run_probe(["docker", "--version"])
owner_command_status = command_probe(owner_command)
sync_status = verify_sync_manifest(sync_manifest_path)
isaac_runtime = isaac_runtime_probe(simulator_name, nvidia_smi)
mujoco_runtime = mujoco_runtime_probe(simulator_name)

blockers = []
warnings = []
if not capture_root.is_dir():
    blockers.append("capture_root_missing_on_gpu_vm")
if not packet_dir.is_dir():
    blockers.append("packet_dir_missing_on_gpu_vm")
for source in (owner_command_status, sync_status):
    blockers.extend(source.get("blockers") or [])
if requires_gpu_probe:
    blockers.extend(nvidia_smi.get("blockers") or [])
elif nvidia_smi.get("blockers"):
    warnings.extend("optional_gpu_probe:" + item for item in nvidia_smi.get("blockers") or [])
blockers.extend(isaac_runtime.get("blockers") or [])
blockers.extend(mujoco_runtime.get("blockers") or [])
warnings.extend(isaac_runtime.get("warnings") or [])
warnings.extend(mujoco_runtime.get("warnings") or [])
if docker.get("blockers"):
    warnings.extend(docker.get("blockers") or [])

payload = dict(
    schema_version="first_gpu_vm_runtime_preflight_result.v1",
    generated_at=datetime.now(timezone.utc).isoformat(),
    capture_root=str(capture_root),
    packet_dir=str(packet_dir),
    simulator={json.dumps(simulator)},
    status="ready_for_owner_command_attempt" if not blockers else "blocked",
    blockers=blockers,
    warnings=warnings,
    host=dict(platform=platform.platform(), python=sys.version),
    nvidia_smi=nvidia_smi,
    docker=docker,
    owner_command=owner_command_status,
    sync_manifest=sync_status,
    isaac_runtime=isaac_runtime,
    mujoco_runtime=mujoco_runtime,
    requires_gpu_probe=requires_gpu_probe,
    claim_boundary=dict(
        artifact_purpose="gpu_vm_runtime_preflight_result",
        simulator_execution_performed=False,
        gpu_provisioning_performed=False,
        robot_policy_execution_proven=False,
        rank_fidelity_result_proven=False,
        public_claim_upgrade_allowed=False,
    ),
)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
"""


def _worldlabs_provider_submission_commands(*, capture_root: Path) -> str:
    source_preflight = capture_root / "pipeline" / "source_video_preflight_manifest.json"
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("SOURCE_VIDEO_PREFLIGHT", str(source_preflight))}

if [[ -z "${{WORLDLABS_API_KEY:-}}" ]]; then
  echo "WORLDLABS_API_KEY must be set in shell state; do not write it into packet artifacts." >&2
  exit 2
fi

if [[ "${{BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION:-}}" != "true" ]]; then
  echo "Set BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION=true to submit a live World Labs request." >&2
  exit 2
fi

"${{PYTHON:-python3}}" - "$SOURCE_VIDEO_PREFLIGHT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("source_video_preflight_manifest_missing")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ready":
    raise SystemExit("source_video_preflight_not_ready")
if int(payload.get("ready_for_worldlabs_first_clip_count") or 0) < 1:
    raise SystemExit("no_worldlabs_ready_clip")
PY

BLUEPRINT_PREVIEW_PROVIDER=world_labs \\
blueprint-run-e2e \\
  --capture-root "$CAPTURE_ROOT" \\
  --provider openai \\
  --pipeline-lane current \\
  --run-evaluation-prep \\
  --evaluation-prep-provider manual
"""


def _webapp_upstream_truth_verification_commands(
    *,
    capture_root: Path,
    result_path: Path,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("WEBAPP_UPSTREAM_TRUTH_VERIFICATION_OUTPUT", str(result_path))}

"${{PYTHON:-python3}}" - "$CAPTURE_ROOT" "$WEBAPP_UPSTREAM_TRUTH_VERIFICATION_OUTPUT" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FIELDS = ("site_submission_id", "request_id", "buyer_request_id", "capture_job_id")
EVIDENCE_SOURCES = (
    "raw/manifest.json",
    "capture_descriptor.json",
    "pipeline_handoff.json",
    "pipeline/opportunity_handoff.json",
)


def string(value: Any) -> str:
    return str(value or "").strip()


def mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {{}}


def read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {{}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {{}}


def placeholder_like(value: str, *, scene_id: str, capture_id: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    if lowered in {{scene_id.lower(), capture_id.lower(), "placeholder", "unknown"}}:
        return True
    return any(token in lowered for token in ("replace_with", "todo", "fake-", "sample-"))


capture_root = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
source_payloads = []
for relative in EVIDENCE_SOURCES:
    path = capture_root / relative
    payload = read_mapping(path)
    source_payloads.append((relative, path, payload))

scene_id = ""
capture_id = ""
for _relative, _path, payload in source_payloads:
    scene_id = scene_id or string(payload.get("scene_id"))
    capture_id = capture_id or string(payload.get("capture_id"))
scene_id = scene_id or capture_root.parent.parent.name
capture_id = capture_id or capture_root.name

values: dict[str, str] = {{}}
sources: dict[str, str | None] = {{field: None for field in FIELDS}}
for field in FIELDS:
    for relative, _path, payload in source_payloads:
        candidate = string(payload.get(field))
        if candidate:
            values[field] = candidate
            sources[field] = relative
            break

handoff = read_mapping(capture_root / "pipeline_handoff.json")
owner_system = mapping(handoff.get("owner_system"))
if not values.get("request_id"):
    candidate = string(owner_system.get("request_id"))
    if candidate:
        values["request_id"] = candidate
        sources["request_id"] = "pipeline_handoff.json owner_system"

blockers = []
for field in FIELDS:
    value = values.get(field, "")
    if placeholder_like(value, scene_id=scene_id, capture_id=capture_id):
        blockers.append("missing_or_placeholder_webapp_" + field)

payload = {{
    "schema_version": "first_gpu_webapp_upstream_truth_verification_result.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "capture_root": str(capture_root),
    "status": "ready" if not blockers else "blocked",
    "blockers": blockers,
    "fields": {{field: bool(values.get(field)) for field in FIELDS}},
    "values_redacted": {{field: bool(values.get(field)) for field in FIELDS}},
    "source_artifacts": sources,
    "accepted_evidence_sources": list(EVIDENCE_SOURCES)
    + [
        "pipeline_handoff.json owner_system",
        "robot_eval_job_request.v1 owner_system",
        "robot_eval_job_request.v1 site_package",
    ],
    "claim_boundary": {{
        "artifact_purpose": "first_gpu_webapp_upstream_truth_verification",
        "artifacts_mutated": False,
        "webapp_requests_submitted": False,
        "live_forwarding_performed": False,
        "simulator_execution_performed": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
    }},
}}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if not blockers else 3)
PY
"""


def _webapp_handoff_verification_commands(
    *,
    capture_root: Path,
    webapp_site_slug: str,
    webapp_staged_inputs_path: Path,
    webapp_forwarding_preflight_path: Path | None,
    result_path: Path,
    allow_local_webapp_rehearsal: bool,
) -> str:
    allow_rehearsal = "true" if allow_local_webapp_rehearsal else "false"
    python_block = r'''
"${PYTHON:-python3}" - "$CAPTURE_ROOT" "$WEBAPP_SITE_SLUG" "$BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH" "$CAPTURE_ROOT_OVERRIDE_JSON" "$CAPTURE_ROOT_OVERRIDE_GLOBAL" "$WEBAPP_HANDOFF_VERIFICATION_OUTPUT" "$ALLOW_LOCAL_WEBAPP_REHEARSAL" "$ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION = "blueprint_live_pipeline_staged_inputs.v1"
WEBAPP_JOB_REQUEST_QUEUE_CONTRACT = "robot_eval_job_request_inbox.v1"
WEBAPP_JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND = "local_first_gpu_rehearsal_request"
FORWARD_PREFLIGHT_SCHEMA_VERSION = "blueprint.webapp.robot_eval_forwarding_readiness.v1"
WEBAPP_UPSTREAM_REQUIRED_FIELDS = (
    "site_submission_id",
    "request_id",
    "buyer_request_id",
    "capture_job_id",
)


def string(value: Any) -> str:
    return str(value or "").strip()


def mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def path_matches(value: str, expected: Path) -> bool:
    if not value:
        return False
    try:
        return Path(value).expanduser().resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def request_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        return mapping(payload.get("job_request"))
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return dict(payload)
    return {}


def nested_source(request: Mapping[str, Any], field: str) -> str:
    source = mapping(request.get("source"))
    selection = mapping(source.get("selection_state"))
    owner_system = mapping(request.get("owner_system"))
    site_package = mapping(request.get("site_package"))
    for candidate in (request, source, selection, owner_system, site_package):
        value = string(candidate.get(field))
        if value:
            return value
    return ""


capture_root = Path(sys.argv[1]).expanduser()
site_slug = string(sys.argv[2])
staged_inputs_path = Path(sys.argv[3]).expanduser()
override_json = string(sys.argv[4])
override_global = string(sys.argv[5])
output_path = Path(sys.argv[6]).expanduser()
allow_local_rehearsal = string(sys.argv[7]).lower() == "true"
forwarding_preflight_path_text = string(sys.argv[8])

blockers: list[str] = []
warnings: list[str] = []
forwarding_preflight = {
    "configured": bool(forwarding_preflight_path_text),
    "path": forwarding_preflight_path_text or None,
    "ready": False,
    "status": "not_configured",
    "site_slug_covered": False,
    "single_capture_root_override_configured": False,
    "probe_status": None,
    "blockers": [],
}
if forwarding_preflight_path_text:
    preflight_path = Path(forwarding_preflight_path_text).expanduser()
    preflight_blockers: list[str] = []
    if not preflight_path.is_file():
        preflight_blockers.append("webapp_forwarding_preflight_report_missing")
        forwarding_preflight.update(
            {
                "status": "blocked",
                "blockers": sorted(set(preflight_blockers)),
            }
        )
    else:
        try:
            preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            preflight_payload = {}
            preflight_blockers.append(
                f"webapp_forwarding_preflight_report_read_failed:{type(exc).__name__}"
            )
        if not isinstance(preflight_payload, Mapping):
            preflight_payload = {}
            preflight_blockers.append("webapp_forwarding_preflight_report_not_json_object")
        if preflight_payload.get("schema_version") != FORWARD_PREFLIGHT_SCHEMA_VERSION:
            preflight_blockers.append("webapp_forwarding_preflight_schema_mismatch")
        preflight_status = string(preflight_payload.get("status"))
        if preflight_status not in {
            "ready_for_required_forwarding",
            "ready_for_required_forwarding_with_probe",
            "ready_for_optional_forwarding",
            "ready_for_optional_forwarding_with_probe",
        }:
            preflight_blockers.append(
                "webapp_forwarding_preflight_status:" + (preflight_status or "unknown")
            )
        configured_env = mapping(preflight_payload.get("configured_env"))
        forward_url = mapping(configured_env.get("forward_url"))
        if forward_url.get("valid") is not True:
            preflight_blockers.append("webapp_forwarding_preflight_forward_url_invalid")
        forward_token = mapping(configured_env.get("forward_token"))
        if forward_token.get("configured") is not True:
            preflight_blockers.append("webapp_forwarding_preflight_token_not_configured")
        if forward_token.get("redacted") is not True:
            preflight_blockers.append("webapp_forwarding_preflight_token_not_redacted")
        capture_root_by_site = mapping(configured_env.get("capture_root_by_site_json"))
        site_slugs = {
            string(item)
            for item in capture_root_by_site.get("site_slugs") or []
            if string(item)
        }
        single_override = mapping(configured_env.get("single_capture_root_override"))
        single_override_configured = single_override.get("configured") is True
        site_slug_covered = bool(site_slug and site_slug in site_slugs)
        if site_slug and not site_slug_covered and not single_override_configured:
            preflight_blockers.append("webapp_forwarding_preflight_missing_site_slug")
        report_blockers = [
            string(item) for item in preflight_payload.get("blockers") or [] if string(item)
        ]
        if report_blockers:
            preflight_blockers.append("webapp_forwarding_preflight_report_has_blockers")
        proof_boundary = mapping(preflight_payload.get("proof_boundary"))
        for field in (
            "command_is_read_only",
            "no_job_queued",
            "no_pipeline_mutation_requested",
            "no_gpu_allocated",
            "no_simulator_execution_proven",
            "no_rank_fidelity_result_proven",
            "no_public_claim_upgrade_allowed",
        ):
            if proof_boundary.get(field) is not True:
                preflight_blockers.append(
                    "webapp_forwarding_preflight_boundary_missing:" + field
                )
        probe = mapping(preflight_payload.get("probe"))
        if probe.get("requested") is True and probe.get("status") != "reachable":
            preflight_blockers.append("webapp_forwarding_preflight_probe_not_reachable")
        forwarding_preflight.update(
            {
                "ready": not preflight_blockers,
                "status": "ready" if not preflight_blockers else "blocked",
                "preflight_status": preflight_status or None,
                "site_slug_covered": site_slug_covered,
                "site_slugs": sorted(site_slugs),
                "single_capture_root_override_configured": single_override_configured,
                "probe_status": probe.get("status"),
                "blockers": sorted(set(preflight_blockers)),
            }
        )
    blockers.extend(forwarding_preflight["blockers"])

preflight_ready = bool(forwarding_preflight["ready"])
override_source = None
override_value = ""
if override_json:
    try:
        overrides = json.loads(override_json)
    except json.JSONDecodeError:
        overrides = {}
        blockers.append("invalid_capture_root_by_site_json")
    if not isinstance(overrides, Mapping):
        overrides = {}
        blockers.append("invalid_capture_root_by_site_json")
    if site_slug:
        override_value = string(overrides.get(site_slug))
        if override_value:
            override_source = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"
        else:
            blockers.append("missing_site_slug_capture_root_override")
    else:
        blockers.append("missing_webapp_site_slug")
elif override_global:
    override_value = override_global
    override_source = "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"
elif preflight_ready and (
    forwarding_preflight["site_slug_covered"]
    or forwarding_preflight["single_capture_root_override_configured"]
):
    override_value = "<redacted-preflight-report>"
    override_source = "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT"
else:
    blockers.append("missing_capture_root_override_env")

if (
    override_value
    and override_source != "ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT"
    and not path_matches(override_value, capture_root)
):
    blockers.append("capture_root_override_does_not_match_capture_root")

request_path_text = ""
job_id = ""
fields_present = {field: False for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS}
local_rehearsal_only = False
request_capture_root_configured = False
if not staged_inputs_path.is_file():
    blockers.append("webapp_staged_inputs_missing")
else:
    try:
        staged_payload = json.loads(staged_inputs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        staged_payload = {}
        blockers.append(f"webapp_staged_inputs_read_failed:{type(exc).__name__}")
    if not isinstance(staged_payload, Mapping):
        staged_payload = {}
        blockers.append("webapp_staged_inputs_not_json_object")
    if staged_payload.get("schema_version") != LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION:
        blockers.append("webapp_staged_inputs_schema_mismatch")
    configured_capture_root = string(staged_payload.get("configured_capture_root"))
    if not path_matches(configured_capture_root, capture_root):
        blockers.append("webapp_staged_inputs_capture_root_mismatch")
    webapp = mapping(staged_payload.get("webapp_request"))
    source_kind = string(webapp.get("source_kind") or staged_payload.get("source_kind"))
    local_rehearsal_only = (
        bool(staged_payload.get("local_rehearsal_only"))
        or source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND
    )
    if not bool(webapp.get("staged")):
        blockers.append("webapp_request_not_staged")
    if not bool(webapp.get("ready")):
        blockers.append("webapp_request_not_ready")
    job_id = string(webapp.get("job_id"))
    request_path_text = string(webapp.get("target_path") or webapp.get("path"))
    if not request_path_text:
        blockers.append("webapp_request_path_missing")
    else:
        request_path = Path(request_path_text).expanduser()
        if not request_path.is_file():
            blockers.append("webapp_request_file_missing")
        else:
            try:
                request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                request_payload = {}
                blockers.append(f"webapp_request_read_failed:{type(exc).__name__}")
            request = request_from_payload(request_payload) if isinstance(request_payload, Mapping) else {}
            if not request:
                blockers.append("webapp_request_not_robot_eval_job_request_v1")
            else:
                source_kind = source_kind or string(request.get("source_kind"))
                if source_kind == LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND:
                    local_rehearsal_only = True
                job_id = job_id or string(request.get("job_id"))
                site_package = mapping(request.get("site_package"))
                request_capture_root = string(site_package.get("capture_root"))
                request_capture_root_configured = bool(request_capture_root)
                if not path_matches(request_capture_root, capture_root):
                    blockers.append("webapp_request_capture_root_mismatch")
                fields_present = {
                    field: bool(nested_source(request, field))
                    for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS
                }
                missing_fields = [
                    field for field, present in fields_present.items() if not present
                ]
                if missing_fields:
                    blockers.append("webapp_request_missing_required_upstream_ids")
    if local_rehearsal_only:
        warnings.append("local_webapp_rehearsal_not_live_forwarding_proof")
        if not allow_local_rehearsal:
            blockers.append("webapp_staged_inputs_local_rehearsal_only")
if not job_id:
    blockers.append("webapp_request_job_id_missing")

payload = {
    "schema_version": "first_gpu_webapp_handoff_verification_result.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "capture_root": str(capture_root),
    "webapp_site_slug": site_slug or None,
    "status": "ready" if not blockers else "blocked",
    "blockers": sorted(set(blockers)),
    "warnings": sorted(set(warnings)),
    "forwarding": {
        "forward_url_configured": bool(os.getenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL")),
        "forward_token_configured": bool(os.getenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN")),
        "forward_url_evidence_present": (
            bool(os.getenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL")) or preflight_ready
        ),
        "forward_token_evidence_present": (
            bool(os.getenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN")) or preflight_ready
        ),
        "forward_token_value_redacted": True,
        "capture_root_override_source": override_source,
        "capture_root_override_configured": bool(override_value),
        "forwarding_preflight": forwarding_preflight,
    },
    "staged_request": {
        "path": str(staged_inputs_path),
        "exists": staged_inputs_path.is_file(),
        "job_id_present": bool(job_id),
        "request_path": request_path_text or None,
        "fields_present": fields_present,
        "local_rehearsal_only": local_rehearsal_only,
        "local_rehearsal_allowed": allow_local_rehearsal,
        "request_capture_root_configured": request_capture_root_configured,
    },
    "claim_boundary": {
        "artifact_purpose": "first_gpu_webapp_handoff_verification",
        "webapp_request_submitted": False,
        "live_forwarding_performed": False,
        "simulator_execution_performed": False,
        "gpu_provisioning_performed": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
    },
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if not blockers else 3)
PY
'''
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("WEBAPP_SITE_SLUG", webapp_site_slug or "<webapp-site-slug>")}
{_shell_default_assignment(WEBAPP_STAGED_INPUTS_ENV, str(webapp_staged_inputs_path))}
{_shell_default_assignment(FORWARD_PREFLIGHT_REPORT_ENV, str(webapp_forwarding_preflight_path or ""))}
{_shell_default_assignment("WEBAPP_HANDOFF_VERIFICATION_OUTPUT", str(result_path))}
{_shell_default_assignment("ALLOW_LOCAL_WEBAPP_REHEARSAL", allow_rehearsal)}

if [[ -z "${{{WEBAPP_FORWARD_URL_ENV}:-}}" && ! -f "${{{FORWARD_PREFLIGHT_REPORT_ENV}:-}}" ]]; then
  echo "{WEBAPP_FORWARD_URL_ENV} must be set to the Pipeline intake endpoint." >&2
  exit 2
fi

if [[ -z "${{{WEBAPP_FORWARD_TOKEN_ENV}:-}}" && ! -f "${{{FORWARD_PREFLIGHT_REPORT_ENV}:-}}" ]]; then
  echo "{WEBAPP_FORWARD_TOKEN_ENV} must be set in shell state; do not write it into packet artifacts." >&2
  exit 2
fi

CAPTURE_ROOT_OVERRIDE_JSON="${{{WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV}:-}}"
CAPTURE_ROOT_OVERRIDE_GLOBAL="${{{WEBAPP_FORWARD_CAPTURE_ROOT_ENV}:-}}"
""" + python_block


def _gpu_provider_bootstrap_manifest(
    *,
    capture_root: Path,
    simulator: str,
    provisioner: str,
    owner_command: str,
    owner_command_location: str,
) -> Dict[str, Any]:
    if simulator in {"isaac_sim", "isaac_lab_arena"}:
        gpu_guidance = {
            "recommended_gpu_class": "RTX-class GPU with RT cores",
            "minimum_vram_gb": 16,
            "preferred_vram_gb": 24,
            "good_first_smoke_examples": [
                "RTX 4090",
                "RTX 6000 Ada",
                "RTX A6000",
                "L40S",
            ],
            "avoid_for_isaac_sim": ["A100", "H100"],
            "avoid_reason": "Isaac Sim rendering requires RT cores; data-center tensor GPUs are not the first smoke target.",
        }
        recommended_provider_path = (
            "Use a RunPod Pod or equivalent interactive GPU VM for the first smoke. "
            "Do not use serverless inference for simulator bring-up."
        )
        first_smoke_path = {
            "primary_simulator_lane": simulator,
            "cheapest_serious_path": False,
            "requires_paid_gpu_for_owner_runtime": True,
            "local_cpu_preflight_available": False,
            "reason": (
                "Isaac is the richer USD/renderer path, but it requires an owner "
                "runtime with RT-core GPU and Vulkan proof."
            ),
        }
    elif simulator == "mujoco":
        gpu_guidance = {
            "recommended_gpu_class": "CPU-first MuJoCo runtime; GPU optional for rendering acceleration",
            "minimum_vram_gb": 0,
            "preferred_vram_gb": 0,
            "good_first_smoke_examples": [
                "local workstation CPU",
                "low-cost CPU VM",
                "low-cost interactive GPU VM only if owner isolation is required",
            ],
            "avoid_for_isaac_sim": [],
            "avoid_reason": None,
        }
        recommended_provider_path = (
            "Run local CPU MuJoCo first. If owner-runtime isolation is required, use "
            "the cheapest interactive VM that can install the mujoco Python package "
            "and sync the Menagerie Unitree G1 assets; no serverless inference path is needed."
        )
        first_smoke_path = {
            "primary_simulator_lane": "mujoco",
            "cheapest_serious_path": True,
            "requires_paid_gpu_for_owner_runtime": False,
            "local_cpu_preflight_available": True,
            "robot_asset": dict(DEFAULT_MUJOCO_G1_ASSET),
            "required_asset_root": str(_mujoco_g1_model_root_path()),
            "reason": (
                "MuJoCo is free/open source and the selected smoke uses the real "
                "MuJoCo Menagerie Unitree G1 MJCF asset before any Isaac spend."
            ),
        }
    else:
        gpu_guidance = {
            "recommended_gpu_class": "CUDA-capable NVIDIA GPU sized for selected simulator",
            "minimum_vram_gb": 16,
            "preferred_vram_gb": 24,
            "good_first_smoke_examples": ["RTX 4090", "RTX A6000", "L40S"],
            "avoid_for_isaac_sim": [],
            "avoid_reason": None,
        }
        recommended_provider_path = (
            "Use an interactive owner runtime for the selected simulator. Do not use "
            "serverless inference for simulator bring-up."
        )
        first_smoke_path = {
            "primary_simulator_lane": simulator,
            "cheapest_serious_path": False,
            "requires_paid_gpu_for_owner_runtime": simulator in {"newton"},
            "local_cpu_preflight_available": simulator in {"pybullet"},
            "reason": "Selected simulator requires owner-command-specific validation.",
        }
    return {
        "schema_version": GPU_PROVIDER_BOOTSTRAP_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "simulator": simulator,
        "provisioner": provisioner,
        "recommended_provider_path": recommended_provider_path,
        "first_smoke_path": first_smoke_path,
        "gpu_guidance": gpu_guidance,
        "nvidia_nim_boundary": {
            "primary_for_first_smoke": False,
            "role": (
                "Optional model inference service for later VLM/LLM/perception/policy "
                f"endpoints; not the {simulator}/physics simulator runtime."
            ),
        },
        "required_mounts_or_sync": [
            str(capture_root),
            str(capture_root / "pipeline" / "simulation_automation"),
            str(capture_root / "pipeline" / "simulation_automation" / "owner_gpu_proof"),
        ],
        "required_vm_checks": [
            "nvidia-smi returns the selected GPU and driver",
            "Docker and NVIDIA Container Toolkit can run a GPU container",
            "MuJoCo-selected packets can run CPU-first, but the mujoco Python package and Menagerie Unitree G1 MJCF assets must be present",
            "capture_root is mounted or synchronized at the same path used by the wrapper",
            "scene assets referenced by gpu_handoff_packet.json exist on the GPU VM",
            "owner simulator command exits nonzero when scene load, spawn, default policy trace, simulator POV evidence, or proof files are missing",
            f"{OWNER_DEFAULT_SMOKE_HELPER_COMMAND} is installed if the owner command relies on the repo helper for default policy and simulator POV manifests",
        ],
        "owner_command": owner_command,
        "owner_command_location": owner_command_location,
        "wrapper_command": "blueprint-run-owner-gpu-proof",
        "first_smoke_success_criteria": [
            "owner_scene_load_trace.json exists and reports scene_loaded=true",
            "owner_spawn_pose_trace.json exists and reports spawn_pose_loaded=true",
            "owner_default_smoke_policy.json exists and defines the walk_to_target policy",
            "owner_action_or_policy_trace.json exists and contains at least one walk_to_target action attempt",
            "owner_sim_robot_pov_evidence_manifest.json exists and references simulator camera/video/frame evidence",
            "owner_artifact_manifest.json exists and lists logs, policy traces, and rendered POV artifacts",
            "gpu_owner_system_proof.json validates after wrapper execution",
        ],
        "hard_stop_conditions": [
            "gpu_handoff_packet.json is not ready_for_owner_gpu_preflight_handoff",
            "missing_local_scene_asset",
            "missing_scene_frame_estimate",
            "scene_bounds_missing_or_invalid",
            "webapp staged request is local rehearsal but the run is being claimed as live WebApp proof",
            "owner command is only a template or dry-run script",
        ],
        "claim_boundary": {
            "artifact_purpose": "first_gpu_provider_bootstrap_packet",
            "live_provider_calls_performed": False,
            "gpu_provisioning_performed": False,
            "simulator_execution_performed": False,
            "robot_policy_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _gpu_provider_bootstrap_markdown(payload: Mapping[str, Any]) -> str:
    guidance = payload.get("gpu_guidance") if isinstance(payload.get("gpu_guidance"), Mapping) else {}
    mounts = "\n".join(f"- `{item}`" for item in payload.get("required_mounts_or_sync", []))
    checks = "\n".join(f"- {item}" for item in payload.get("required_vm_checks", []))
    success = "\n".join(f"- {item}" for item in payload.get("first_smoke_success_criteria", []))
    stops = "\n".join(f"- `{item}`" for item in payload.get("hard_stop_conditions", []))
    examples = ", ".join(str(item) for item in guidance.get("good_first_smoke_examples", []))
    avoid = ", ".join(str(item) for item in guidance.get("avoid_for_isaac_sim", []))
    avoid_line = f"\nAvoid for Isaac Sim first smoke: {avoid}." if avoid else ""
    first_smoke_path = (
        payload.get("first_smoke_path")
        if isinstance(payload.get("first_smoke_path"), Mapping)
        else {}
    )
    return f"""# GPU Provider Bootstrap

{payload.get("recommended_provider_path")}

Serverless inference and NVIDIA NIM are not the primary simulator runtime for
this milestone. NIM can be useful later for model inference endpoints, but this
run needs an owner-controlled runtime that can execute `{payload.get("wrapper_command")}`.

## Cheapest Serious Path

- Primary simulator lane: `{first_smoke_path.get("primary_simulator_lane")}`
- Cheapest serious path: `{first_smoke_path.get("cheapest_serious_path")}`
- Requires paid GPU for owner runtime: `{first_smoke_path.get("requires_paid_gpu_for_owner_runtime")}`
- Local CPU preflight available: `{first_smoke_path.get("local_cpu_preflight_available")}`
- Reason: {first_smoke_path.get("reason")}

## GPU Selection

- Simulator: `{payload.get("simulator")}`
- Provisioner label: `{payload.get("provisioner")}`
- Recommended GPU class: {guidance.get("recommended_gpu_class")}
- Minimum VRAM: {guidance.get("minimum_vram_gb")} GB
- Preferred VRAM: {guidance.get("preferred_vram_gb")} GB
- Good first-smoke examples: {examples}{avoid_line}

## Required Mounts Or Sync

{mounts}

## VM Checks

{checks}

## Owner Command

The owner simulator command is:

```bash
{payload.get("owner_command")}
```

Command location: `{payload.get("owner_command_location")}`.

It must be a real simulator command, not a dry-run or template. It must fail
closed when scene load, spawn, action trace, or proof file generation fails.

## First Smoke Success Criteria

{success}

## Hard Stops

{stops}

The bootstrap packet does not provision a GPU, run a simulator, or prove robot
readiness. It only makes the owner-GPU setup requirements explicit.
"""


def _simulator_path_details(framework: str) -> Dict[str, Any]:
    details: Dict[str, Dict[str, Any]] = {
        "isaac_sim": {
            "lane": "first_owner_gpu_scene_load_spawn_and_action_trace",
            "execution_environment": "owner_gpu_vm",
            "can_run_without_gpu_preflight": False,
            "recommended_first_gpu_smoke": True,
            "proof_role": "Primary first-GPU proof target for rich scene load, spawn, action traces, logs, and rendered artifacts.",
            "requires": [
                "RTX-class GPU with RT cores",
                "materialized local scene asset",
                "finite scene frame estimate and spawn validation",
                "owner simulator command that writes the proof contract traces",
            ],
            "not_a_substitute_for": [
                "real robot deployment outcome",
                "policy quality proof",
                "site-wide compatibility across future captures",
            ],
        },
        "isaac_lab_arena": {
            "lane": "arena_or_policy_training_followup",
            "execution_environment": "owner_gpu_vm",
            "can_run_without_gpu_preflight": False,
            "recommended_first_gpu_smoke": False,
            "proof_role": "Follow-up path after scene, robot profile, task binding, and Arena package inputs are ready.",
            "requires": [
                "Isaac Lab or Arena environment installed on the owner GPU VM",
                "robot/team policy package or Arena package",
                "same scene/spawn preflight gates as Isaac Sim",
            ],
            "not_a_substitute_for": [
                "first scene-load smoke when the basic Isaac Sim path has not run",
                "real robot POV evidence",
            ],
        },
        "mujoco": {
            "lane": "cpu_first_unitree_g1_mjcf_scene_smoke",
            "execution_environment": "local_or_owner_runtime",
            "can_run_without_gpu_preflight": True,
            "recommended_first_gpu_smoke": False,
            "proof_role": (
                "Cheapest serious primary lane when selected: load the staged scene "
                "with the real MuJoCo Menagerie Unitree G1 MJCF and run the default "
                "walk_to_target smoke through the owner proof contract."
            ),
            "requires": [
                "mujoco Python package or owner MuJoCo install",
                "MuJoCo Menagerie unitree_g1/g1.xml and mesh assets",
                "staged World Labs GLB or compatible scene asset converted to MJCF/OBJ support",
            ],
            "not_a_substitute_for": [
                "Isaac Sim rich-scene rendering proof",
                "real robot POV evidence",
                "balanced humanoid locomotion policy quality",
                "physics/contact/off-scope validation",
            ],
        },
        "pybullet": {
            "lane": "cpu_or_lightweight_spawn_collision_preflight",
            "execution_environment": "local_or_owner_runtime",
            "can_run_without_gpu_preflight": True,
            "recommended_first_gpu_smoke": False,
            "proof_role": "Useful for URDF/proxy load and simple spawn/collision sanity; not rich scene proof by default.",
            "requires": [
                "URDF-compatible scene or proxy export",
                "PyBullet package or owner PyBullet install",
            ],
            "not_a_substitute_for": [
                "Isaac Sim rich-scene rendering proof",
                "real robot contact, safety, or policy proof",
            ],
        },
        "newton": {
            "lane": "future_or_specialized_physics_path",
            "execution_environment": "owner_runtime",
            "can_run_without_gpu_preflight": False,
            "recommended_first_gpu_smoke": False,
            "proof_role": "Specialized physics path only when an owner Newton/Warp-compatible command is supplied.",
            "requires": [
                "owner Newton/Warp runtime",
                "compatible scene representation",
                "explicit simulator command and proof contract traces",
            ],
            "not_a_substitute_for": [
                "Isaac Sim first smoke unless intentionally selected for this capture",
                "WebApp live handoff proof",
            ],
        },
    }
    return dict(details.get(framework, {}))


def _simulator_path_matrix_manifest(
    *,
    capture_root: Path,
    simulator: str,
    provisioner: str,
    owner_command_location: str,
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    readiness_blockers = [str(item) for item in readiness.get("blockers") or []]
    selected_path_blockers = [
        item
        for item in readiness_blockers
        if item.startswith("pipeline_gpu_handoff:")
        or item.startswith("simulator_runtime:")
    ]
    paths: list[Dict[str, Any]] = []
    for framework in SIMULATOR_FRAMEWORKS:
        details = _simulator_path_details(framework)
        recommended_first_smoke = bool(details.get("recommended_first_gpu_smoke"))
        if simulator == "mujoco":
            recommended_first_smoke = framework == "mujoco"
        paths.append(
            {
                "framework": framework,
                "selected_for_this_packet": framework == simulator,
                "lane": details.get("lane"),
                "execution_environment": details.get("execution_environment"),
                "can_run_without_gpu_preflight": bool(
                    details.get("can_run_without_gpu_preflight")
                ),
                "recommended_first_gpu_smoke": recommended_first_smoke,
                "proof_role": details.get("proof_role"),
                "requires": details.get("requires") or [],
                "not_a_substitute_for": details.get("not_a_substitute_for") or [],
                "selected_path_blockers": selected_path_blockers
                if framework == simulator
                else [],
            }
        )
    return {
        "schema_version": FIRST_GPU_SIMULATOR_PATH_MATRIX_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "selected_simulator": simulator,
        "selected_provisioner": provisioner,
        "owner_command_location": owner_command_location,
        "status": (
            "blocked_for_selected_simulator_attempt"
            if selected_path_blockers
            else "selected_simulator_path_clear"
        ),
        "selected_path_blockers": selected_path_blockers,
        "paths": paths,
        "nvidia_nim_boundary": {
            "primary_simulator_runtime": False,
            "useful_later_for": [
                "vision-language or perception inference endpoint",
                "policy or planner inference endpoint",
                "post-run analysis or summarization service",
            ],
            "not_a_substitute_for": [
                "owner simulator command execution",
                "scene load trace",
                "spawn pose trace",
                "physics contact or off-scope validation",
            ],
        },
        "first_gpu_recommendation": {
            "recommended_first_path": "mujoco" if simulator == "mujoco" else "isaac_sim",
            "selected_matches_recommendation": (
                simulator == ("mujoco" if simulator == "mujoco" else "isaac_sim")
            ),
            "reason": (
                "MuJoCo is selected as the cheapest serious primary lane for now: "
                "prove real Menagerie Unitree G1 MJCF load, scene conversion, "
                "default walk_to_target execution, and simulator POV before GPU spend."
                if simulator == "mujoco"
                else (
                    "The first smoke should prove rich scene load, spawn, and trace capture "
                    "before Arena/policy or lightweight physics paths are treated as evidence."
                )
            ),
        },
        "safe_commands": [
            (
                "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
                "blueprint-run-simulation-automation "
                f"--capture-root {shlex.quote(str(capture_root))} "
                f"--allow-simulator {shlex.quote(simulator)} "
                f"--simulator-command {shlex.quote(simulator + '=$OWNER_SIMULATOR_COMMAND')}"
            ),
        ],
        "claim_boundary": {
            "artifact_purpose": "first_gpu_simulator_path_matrix",
            "simulator_selected": simulator,
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "nvidia_nim_used_as_simulator": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _simulator_path_matrix_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Simulator Path Matrix",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Selected simulator: `{payload.get('selected_simulator')}`",
        f"- Provisioner: `{payload.get('selected_provisioner')}`",
        "",
        "## Paths",
        "",
        "| Framework | Selected | Lane | Environment | CPU preflight | First GPU smoke |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload.get("paths") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| "
            f"`{item.get('framework')}` | "
            f"`{item.get('selected_for_this_packet')}` | "
            f"{item.get('lane')} | "
            f"{item.get('execution_environment')} | "
            f"`{item.get('can_run_without_gpu_preflight')}` | "
            f"`{item.get('recommended_first_gpu_smoke')}` |"
        )
    blockers = [str(item) for item in payload.get("selected_path_blockers") or []]
    if blockers:
        lines.extend(["", "## Selected Path Blockers", ""])
        lines.extend(f"- `{item}`" for item in blockers)
    nim = payload.get("nvidia_nim_boundary") if isinstance(payload.get("nvidia_nim_boundary"), Mapping) else {}
    lines.extend(["", "## NVIDIA NIM Boundary", ""])
    lines.append(f"- Primary simulator runtime: `{nim.get('primary_simulator_runtime')}`")
    lines.append("- Useful later for:")
    lines.extend(f"  - {item}" for item in nim.get("useful_later_for") or [])
    lines.append("- Not a substitute for:")
    lines.extend(f"  - {item}" for item in nim.get("not_a_substitute_for") or [])
    commands = [str(item) for item in payload.get("safe_commands") or []]
    if commands:
        lines.extend(["", "## Safe Commands", "", "```bash"])
        lines.extend(commands)
        lines.extend(["```", ""])
    lines.extend(
        [
            "This matrix does not run simulators, provision GPUs, invoke NIM, or prove generated-world rank fidelity.",
            "",
        ]
    )
    return "\n".join(lines)


def _gpu_vm_runtime_preflight_plan_manifest(
    *,
    capture_root: Path,
    packet_dir: Path,
    script_path: Path,
    result_path: Path,
    sync_manifest_path: Path,
    readiness: Mapping[str, Any],
    simulator: str,
    provisioner: str,
    owner_command: str,
    owner_command_location: str,
    owner_command_supplied: bool,
    vm_sync_manifest: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    readiness_blockers = [str(item) for item in readiness.get("blockers") or []]
    result_summary = _gpu_vm_runtime_preflight_result_summary(result_path)
    sync_status = _string(vm_sync_manifest.get("status")) if vm_sync_manifest else "not_generated"
    sync_blockers = [
        f"gpu_vm_sync_manifest:{item}" for item in (vm_sync_manifest or {}).get("blockers") or []
    ]
    if vm_sync_manifest and sync_status != "ready" and not sync_blockers:
        sync_blockers.append(f"gpu_vm_sync_manifest_status:{sync_status or 'unknown'}")
    hard_stop_blockers = [
        item
        for item in readiness_blockers
        if item.startswith("pipeline_gpu_handoff:")
        or item.startswith("simulator_runtime:missing_simulator_command")
        or item.startswith("simulator_runtime:simulator_command")
    ]
    hard_stop_blockers.extend(sync_blockers)
    if not owner_command_supplied and "owner_command_not_supplied_to_run_packet" not in hard_stop_blockers:
        hard_stop_blockers.append("owner_command_not_supplied_to_run_packet")
    if simulator == "mujoco":
        runtime_checks = [
            "CAPTURE_ROOT exists on the owner runtime at the same path used by the packet",
            "PACKET_DIR exists on the owner runtime",
            "mujoco imports from the selected Python interpreter",
            "BLUEPRINT_MUJOCO_G1_MODEL_ROOT contains unitree_g1/g1.xml and mesh assets",
            "docker --version is optional unless the owner runtime depends on containers",
            "OWNER_SIMULATOR_COMMAND resolves to the generated MuJoCo Unitree G1 smoke or another executable owner MuJoCo command",
            "gpu_vm_sync_manifest.json files exist and SHA-256 values match after mount or copy",
        ]
    else:
        runtime_checks = [
            "CAPTURE_ROOT exists on the GPU VM at the same path used by the packet",
            "PACKET_DIR exists on the GPU VM",
            "nvidia-smi exists and can query GPU name, memory, and driver",
            "Isaac Sim/Lab paths require an RT-core GPU, driver >= 580.65.06 by default, and a working Vulkan probe",
            "docker --version is available when the owner runtime depends on containers",
            "OWNER_SIMULATOR_COMMAND or ISAAC_OWNER_COMMAND resolves to an executable on the GPU VM",
            "gpu_vm_sync_manifest.json files exist and SHA-256 values match after mount or copy",
        ]
    status = (
        "blocked_for_owner_gpu_attempt"
        if hard_stop_blockers
        else "ready_to_run_on_gpu_vm_before_owner_command"
    )
    return {
        "schema_version": FIRST_GPU_VM_RUNTIME_PREFLIGHT_PLAN_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "packet_dir": str(packet_dir),
        "simulator": simulator,
        "provisioner": provisioner,
        "owner_command": owner_command,
        "owner_command_location": owner_command_location,
        "owner_command_supplied": owner_command_supplied,
        "gpu_vm_sync_status": sync_status,
        "gpu_vm_sync_blockers": sync_blockers,
        "status": status,
        "hard_stop_blockers": hard_stop_blockers,
        "full_e2e_blockers": readiness_blockers,
        "result": result_summary,
        "script": {
            "path": str(script_path),
            "default_result_path": str(result_path),
            "safe_to_run_on_gpu_vm": not hard_stop_blockers,
            "runs_owner_simulator_command": False,
        },
        "inputs_checked_when_script_runs": runtime_checks,
        "ordered_next_steps": [
            "Copy or mount the capture root and run packet to the GPU VM.",
            "Set OWNER_SIMULATOR_COMMAND to the real simulator command on the GPU VM.",
            "Run gpu_vm_runtime_preflight.sh before blueprint-run-owner-gpu-proof.",
            "Only run gpu_vm_commands.sh after this preflight writes a ready result.",
        ],
        "safe_commands": [
            f"bash {shlex.quote(str(script_path))}",
            (
                "GPU_VM_PREFLIGHT_OUTPUT=<result-json> "
                f"bash {shlex.quote(str(script_path))}"
            ),
        ],
        "claim_boundary": {
            "artifact_purpose": "first_gpu_vm_runtime_preflight_plan",
            "live_provider_calls_performed": False,
            "remote_asset_downloads_performed": False,
            "files_copied": False,
            "gpu_provisioning_performed": False,
            "simulator_execution_performed": False,
            "owner_simulator_command_executed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
        "related_artifacts": {
            "gpu_vm_sync_manifest": str(sync_manifest_path),
            "gpu_vm_commands": str(packet_dir / "gpu_vm_commands.sh"),
            "owner_command_contract": str(packet_dir / "owner_command_contract.md"),
        },
    }


def _gpu_vm_runtime_preflight_plan_markdown(payload: Mapping[str, Any]) -> str:
    script = payload.get("script") if isinstance(payload.get("script"), Mapping) else {}
    lines = [
        "# GPU VM Runtime Preflight",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Capture root: `{payload.get('capture_root')}`",
        f"- Script: `{script.get('path')}`",
        f"- Result: `{script.get('default_result_path')}`",
        f"- GPU VM sync status: `{payload.get('gpu_vm_sync_status')}`",
        "",
    ]
    blockers = [str(item) for item in payload.get("hard_stop_blockers") or []]
    if blockers:
        lines.extend(["## Hard Stops", ""])
        lines.extend(f"- `{item}`" for item in blockers)
        lines.append("")
    lines.extend(["## Checks When Run On The GPU VM", ""])
    lines.extend(f"- {item}" for item in payload.get("inputs_checked_when_script_runs") or [])
    lines.extend(["", "## Ordered Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("ordered_next_steps") or [])
    commands = [str(item) for item in payload.get("safe_commands") or []]
    if commands:
        lines.extend(["", "## Safe Commands", "", "```bash"])
        lines.extend(commands)
        lines.extend(["```", ""])
    lines.extend(
        [
            "This preflight checks the GPU VM environment and file sync only. It does not run the owner simulator command, provision GPUs, or prove generated-world rank fidelity.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _blockers_with_prefix(blockers: Sequence[str], prefix: str) -> list[str]:
    return [item for item in blockers if item.startswith(prefix)]


def _blocker_details_matching(
    details: Sequence[Mapping[str, Any]],
    blocker_ids: Sequence[str],
    *,
    severities: Sequence[str] = (),
) -> list[Dict[str, Any]]:
    wanted_ids = set(blocker_ids)
    wanted_severities = set(severities)
    matches: list[Dict[str, Any]] = []
    for item in details:
        if not isinstance(item, Mapping):
            continue
        blocker_id = _string(item.get("blocker_id"))
        severity = _string(item.get("severity"))
        if blocker_id in wanted_ids or (wanted_severities and severity in wanted_severities):
            matches.append(dict(item))
    return matches


def _strip_stage_prefix(blocker: str, stage: str) -> str:
    prefix = f"{stage}:"
    return blocker[len(prefix):] if blocker.startswith(prefix) else blocker


def _webapp_upstream_blocker_details(blockers: Sequence[str]) -> list[Dict[str, Any]]:
    normalized = {
        _strip_stage_prefix(_string(item), "webapp_upstream_truth")
        for item in blockers
    }
    details: list[Dict[str, Any]] = []
    for field in WEBAPP_HANDOFF_UPSTREAM_FIELDS:
        blocker_id = f"missing_or_placeholder_webapp_{field}"
        if blocker_id not in normalized:
            continue
        details.append(
            {
                "blocker_id": blocker_id,
                "field": field,
                "source_artifact": "webapp_upstream_truth_stage",
                "severity": "hard_pre_gpu_blocker",
                "required_input": (
                    f"Populate a real non-placeholder {field} from the live "
                    "WebApp/Capture request path."
                ),
                "accepted_evidence_sources": list(WEBAPP_UPSTREAM_EVIDENCE_SOURCES),
                "proof_boundary": (
                    "required upstream identity only; does not prove live forwarding, "
                    "simulator execution, or generated-world rank fidelity"
                ),
            }
        )
    return details


def _owner_gpu_command_blocker_details(
    blockers: Sequence[str],
    *,
    capture_root: Path,
) -> list[Dict[str, Any]]:
    if not blockers:
        return []
    return [
        {
            "blocker_id": _strip_stage_prefix(str(item), "simulator_runtime"),
            "source_artifact": "owner_command_contract.md",
            "severity": "hard_pre_gpu_blocker",
            "required_input": (
                "Set --owner-command or OWNER_SIMULATOR_COMMAND to the real simulator "
                "command available on the GPU VM."
            ),
            "wrapper_command": "blueprint-run-owner-gpu-proof",
            "trace_environment_variables": list(OWNER_COMMAND_TRACE_ENV_VARS),
            "expected_outputs": [
                str(capture_root / output)
                for output in OWNER_COMMAND_REQUIRED_OUTPUTS
            ],
            "proof_boundary": (
                "owner command wiring only; does not prove simulator execution until "
                "blueprint-run-owner-gpu-proof validates gpu_owner_system_proof.json"
            ),
        }
        for item in blockers
    ]


def _category(
    *,
    category_id: str,
    title: str,
    blockers: Sequence[str],
    next_actions: Sequence[str],
    evidence_required: Sequence[str],
    safe_commands: Sequence[str] = (),
    warnings: Sequence[str] = (),
    blocker_details: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    status = "blocked" if blockers else "warning" if warnings else "ready"
    return {
        "category_id": category_id,
        "title": title,
        "status": status,
        "blockers": _unique_strings(blockers),
        "warnings": _unique_strings(warnings),
        "blocker_details": [
            dict(item)
            for item in blocker_details
            if isinstance(item, Mapping)
        ],
        "next_actions": list(next_actions),
        "evidence_required": list(evidence_required),
        "safe_commands": list(safe_commands),
    }


def _blocker_resolution_actions(categories: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    actions: list[Dict[str, Any]] = []
    for category in categories:
        category_id = _string(category.get("category_id"))
        title = _string(category.get("title")) or category_id
        status = _string(category.get("status"))
        blockers = _unique_strings([str(item) for item in category.get("blockers") or []])
        warnings = _unique_strings([str(item) for item in category.get("warnings") or []])
        if status not in {"blocked", "warning"} or (not blockers and not warnings):
            continue
        action = {
            "action_id": f"{len(actions) + 1:02d}_{category_id}",
            "priority": len(actions) + 1,
            "category_id": category_id,
            "title": title,
            "status": status,
            "must_clear_before_gpu_spend": bool(blockers),
            "blockers": blockers,
            "warnings": warnings,
            "blocker_details": [
                dict(item)
                for item in category.get("blocker_details") or []
                if isinstance(item, Mapping)
            ],
            "next_actions": [str(item) for item in category.get("next_actions") or []],
            "evidence_required": [str(item) for item in category.get("evidence_required") or []],
            "safe_commands": [str(item) for item in category.get("safe_commands") or []],
            "proof_boundary": (
                "Clearing this action only satisfies the named first-GPU packet gate; it "
                "does not submit WebApp requests, call providers, provision GPUs, run "
                "simulators, or prove generated-world rank fidelity."
            ),
        }
        actions.append(action)
    return actions


def _source_video_category(capture_root: Path) -> Dict[str, Any]:
    manifest_path = capture_root / "pipeline" / "source_video_preflight_manifest.json"
    manifest = _read_mapping(manifest_path)
    blockers: list[str] = []
    warnings: list[str] = []
    if not manifest:
        warnings.append("source_video_preflight_manifest_missing")
    elif manifest.get("status") != "ready":
        blockers.extend(str(item) for item in manifest.get("blockers") or [])
        for candidate in manifest.get("candidates") or []:
            if isinstance(candidate, Mapping):
                blockers.extend(str(item) for item in candidate.get("staging_blockers") or [])
                blockers.extend(str(item) for item in candidate.get("worldlabs_blockers") or [])
                warnings.extend(str(item) for item in candidate.get("warnings") or [])
    return {
        **_category(
            category_id="source_video_preflight",
            title="Source Video Preflight",
            blockers=blockers,
            warnings=warnings,
            next_actions=[
                "Run strict sample-video preflight before staging or before spending GPU time.",
            ],
            evidence_required=[
                "pipeline/source_video_preflight_manifest.json is ready and lists at least one World Labs-ready first clip.",
            ],
            safe_commands=[
                (
                    "blueprint-audit-first-gpu-sample-video --source-video <source-video> "
                    "--require-probe --output <capture-root>/pipeline/source_video_preflight_manifest.json"
                ),
            ],
        ),
        "manifest_path": str(manifest_path),
        "manifest_status": manifest.get("status") if manifest else None,
        "ready_for_worldlabs_first_clip_count": (
            manifest.get("ready_for_worldlabs_first_clip_count") if manifest else 0
        ),
    }


def _blocker_resolution_manifest(
    *,
    capture_root: Path,
    readiness: Mapping[str, Any],
    webapp_site_slug: str,
    webapp_staged_inputs_path: Path,
    simulator: str,
    provisioner: str,
    owner_command: str,
    owner_command_location: str,
    allow_local_webapp_rehearsal: bool,
) -> Dict[str, Any]:
    blockers = [str(item) for item in readiness.get("blockers") or []]
    warnings = [str(item) for item in readiness.get("warnings") or []]
    pipeline_handoff_stage = {}
    stages = readiness.get("stages")
    if isinstance(stages, Mapping) and isinstance(stages.get("pipeline_gpu_handoff"), Mapping):
        pipeline_handoff_stage = dict(stages["pipeline_gpu_handoff"])
    pre_gpu_blocker_details = [
        dict(item)
        for item in pipeline_handoff_stage.get("pre_gpu_blocker_details") or []
        if isinstance(item, Mapping)
    ]
    scene_blocker_details = _blocker_details_matching(
        pre_gpu_blocker_details,
        [
            "missing_local_scene_asset",
            "missing_scene_frame_estimate",
            "scene_bounds_missing_or_invalid",
        ],
        severities=["hard_pre_gpu_blocker"],
    )
    handoff_blocker_details = _blocker_details_matching(
        pre_gpu_blocker_details,
        [
            "missing_local_scene_asset",
            "missing_scene_frame_estimate",
            "scene_bounds_missing_or_invalid",
            "portable_collider_glb_missing",
        ],
        severities=["hard_pre_gpu_blocker", "review_or_backend_selection_blocker"],
    )
    webapp_upstream_blockers = _blockers_with_prefix(blockers, "webapp_upstream_truth:")
    owner_gpu_command_blockers = _blockers_with_prefix(
        blockers,
        "simulator_runtime:missing_simulator_command",
    ) + _blockers_with_prefix(blockers, "simulator_runtime:simulator_command")
    categories: list[Dict[str, Any]] = [_source_video_category(capture_root)]
    categories.extend(
        [
            _category(
                category_id="webapp_upstream_truth",
                title="Real WebApp Upstream IDs",
                blockers=webapp_upstream_blockers,
                blocker_details=_webapp_upstream_blocker_details(webapp_upstream_blockers),
                next_actions=[
                    "Attach real WebApp/Capture IDs from the actual site submission and robot-eval request path.",
                    "Do not use capture IDs, placeholders, or local rehearsal IDs as live upstream truth.",
                ],
                evidence_required=[
                    "raw/manifest.json, capture_descriptor.json, or pipeline_handoff.json contains non-placeholder site_submission_id, request_id, buyer_request_id, and capture_job_id.",
                ],
            ),
            _category(
                category_id="webapp_forwarding_env",
                title="WebApp Forwarding Environment",
                blockers=_blockers_with_prefix(blockers, "webapp_forwarding:"),
                next_actions=[
                    "Configure WebApp-to-Pipeline forwarding and map the WebApp site slug to this capture root.",
                ],
                evidence_required=[
                    "Forwarding URL and token are present in the runtime environment without being written to artifacts.",
                    "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON maps the site slug to CAPTURE_ROOT.",
                ],
                safe_commands=[
                    _shell_export(
                        "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL",
                        "https://<pipeline-host>/api/live-pipeline/job-requests",
                    ),
                    _shell_export("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "<redacted>"),
                    _shell_export("ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED", "true"),
                    _shell_export(
                        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
                        _capture_root_by_site_json(
                            site_slug=webapp_site_slug or "<webapp-site-slug>",
                            capture_root=capture_root,
                        ),
                    ),
                ],
            ),
            _category(
                category_id="webapp_staged_request",
                title="Staged WebApp Robot-Eval Request",
                blockers=_blockers_with_prefix(blockers, "webapp_staged_request:"),
                next_actions=[
                    "Stage a validated robot_eval_job_request.v1 through Pipeline intake.",
                ],
                evidence_required=[
                    "pipeline/live_pipeline_staged_inputs.json points at a staged WebApp request matching this capture root.",
                ],
                safe_commands=[
                    (
                        "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
                        "--webapp-job-request <robot_eval_job_request.json> --stage-webapp-request "
                        f"--staged-inputs-path {shlex.quote(str(webapp_staged_inputs_path))}"
                    ),
                ],
            ),
            _category(
                category_id="scene_spawn_preflight",
                title="Scene Asset And Spawn Preflight",
                blockers=_blockers_with_prefix(blockers, "pipeline_gpu_handoff:spawn_validation"),
                blocker_details=scene_blocker_details,
                next_actions=[
                    "Provide materialized scene geometry with finite bounds, then rerun simulation automation.",
                ],
                evidence_required=[
                    "scene_asset_preflight.json names a local scene asset.",
                    "scene_frame_estimate.json has finite bounds and floor estimate.",
                    "spawn_pose_validation_manifest.json has at least one valid or reviewable spawn candidate.",
                ],
                safe_commands=[
                    (
                        "blueprint-run-simulation-automation "
                        f"--capture-root {shlex.quote(str(capture_root))} "
                        "--scene-asset <materialized-scene>"
                    ),
                ],
            ),
            _category(
                category_id="pipeline_gpu_handoff",
                title="Pipeline GPU Handoff Packet",
                blockers=[
                    item
                    for item in _blockers_with_prefix(blockers, "pipeline_gpu_handoff:")
                    if "spawn_validation" not in item
                ],
                blocker_details=handoff_blocker_details,
                next_actions=[
                    "Rerun simulation automation after scene asset and spawn blockers are cleared.",
                ],
                evidence_required=[
                    "pipeline/simulation_automation/gpu_handoff_packet.json status is ready_for_owner_gpu_preflight_handoff.",
                ],
                safe_commands=[
                    f"blueprint-run-simulation-automation --capture-root {shlex.quote(str(capture_root))}",
                ],
            ),
            _category(
                category_id="owner_gpu_command",
                title="Owner GPU Simulator Command",
                blockers=owner_gpu_command_blockers,
                blocker_details=_owner_gpu_command_blocker_details(
                    owner_gpu_command_blockers,
                    capture_root=capture_root,
                ),
                next_actions=[
                    "Replace the owner command placeholder with the real simulator command that will run inside the GPU VM.",
                ],
                evidence_required=[
                    "OWNER_SIMULATOR_COMMAND points to an executable command on the GPU VM and writes the traces named in owner_command_contract.md.",
                ],
                safe_commands=[
                    (
                        "blueprint-build-first-gpu-run-packet "
                        f"--capture-root {shlex.quote(str(capture_root))} "
                        f"--webapp-site-slug {shlex.quote(webapp_site_slug or '<webapp-site-slug>')} "
                        f"--simulator {shlex.quote(simulator)} --provisioner {shlex.quote(provisioner)} "
                        "--owner-command <real-owner-simulator-command> "
                        f"--owner-command-location {shlex.quote(owner_command_location)}"
                    ),
                ],
            ),
            _category(
                category_id="owner_gpu_gate",
                title="Explicit Owner GPU Execution Gate",
                blockers=_blockers_with_prefix(
                    blockers,
                    "simulator_runtime:missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
                ),
                warnings=_blockers_with_prefix(
                    warnings,
                    "simulator_runtime:missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING",
                ),
                next_actions=[
                    "Enable simulator execution only for the actual owner GPU attempt.",
                    "Enable GPU provisioning only if this run packet is allowed to request or control GPU allocation.",
                ],
                evidence_required=[
                    "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true is present for the actual owner GPU proof attempt.",
                    "BLUEPRINT_ALLOW_GPU_PROVISIONING=true is present only when provider provisioning is intentionally allowed.",
                ],
                safe_commands=[
                    _shell_export("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true"),
                    _shell_export("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true"),
                ],
            ),
        ]
    )
    actions = _blocker_resolution_actions(categories)
    return {
        "schema_version": FIRST_GPU_BLOCKER_RESOLUTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "webapp_site_slug": webapp_site_slug or None,
        "webapp_staged_inputs_path": str(webapp_staged_inputs_path),
        "simulator": simulator,
        "provisioner": provisioner,
        "owner_command": owner_command,
        "owner_command_location": owner_command_location,
        "allow_local_webapp_rehearsal": allow_local_webapp_rehearsal,
        "readiness_status": readiness.get("status"),
        "ready_for_first_gpu_attempt": bool(readiness.get("ready_for_first_gpu_attempt")),
        "blocker_count": sum(len(item["blockers"]) for item in categories),
        "warning_count": sum(len(item["warnings"]) for item in categories),
        "action_count": len(actions),
        "blocked_action_count": sum(1 for item in actions if item["must_clear_before_gpu_spend"]),
        "actions": actions,
        "categories": categories,
        "final_verification_commands": [
            (
                "blueprint-audit-first-gpu-e2e-readiness "
                f"--capture-root {shlex.quote(str(capture_root))} "
                f"--webapp-site-slug {shlex.quote(webapp_site_slug or '<webapp-site-slug>')} "
                f"--webapp-staged-inputs {shlex.quote(str(webapp_staged_inputs_path))} "
                f"--simulator {shlex.quote(simulator)} --provisioner {shlex.quote(provisioner)} "
                "--simulator-command \"$OWNER_SIMULATOR_COMMAND\""
                f" --simulator-command-location {shlex.quote(owner_command_location)}"
                + (" --allow-local-webapp-rehearsal" if allow_local_webapp_rehearsal else "")
            )
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _append_blocker_details_markdown(
    lines: list[str],
    details: Sequence[Mapping[str, Any]],
) -> None:
    if not details:
        return
    lines.append("Blocker details:")
    for detail in details:
        detail_bits = [
            f"id=`{detail.get('blocker_id')}`",
            f"source=`{detail.get('source_artifact')}`",
            f"severity=`{detail.get('severity')}`",
        ]
        lines.append(f"- {'; '.join(detail_bits)}")
        if detail.get("field"):
            lines.append(f"  Field: `{detail.get('field')}`")
        if detail.get("required_input"):
            lines.append(f"  Required input: {detail.get('required_input')}")
        if detail.get("accepted_evidence_sources"):
            sources = ", ".join(str(value) for value in detail["accepted_evidence_sources"])
            lines.append(f"  Accepted evidence sources: {sources}")
        if detail.get("wrapper_command"):
            lines.append(f"  Wrapper command: `{detail.get('wrapper_command')}`")
        if detail.get("trace_environment_variables"):
            trace_vars = ", ".join(str(value) for value in detail["trace_environment_variables"])
            lines.append(f"  Trace env vars: {trace_vars}")
        if detail.get("expected_outputs"):
            lines.append("  Expected outputs:")
            lines.extend(f"  - `{value}`" for value in detail["expected_outputs"])
        if detail.get("safe_next_command"):
            lines.append(f"  Safe next command: `{detail.get('safe_next_command')}`")
    lines.append("")


def _blocker_resolution_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# First GPU Blocker Resolution",
        "",
        f"- Capture root: `{payload.get('capture_root')}`",
        f"- Readiness status: `{payload.get('readiness_status')}`",
        f"- Ready for first GPU attempt: `{payload.get('ready_for_first_gpu_attempt')}`",
        f"- Action count: `{payload.get('action_count')}`",
        "",
        "## Immediate Actions",
        "",
    ]
    actions = [item for item in payload.get("actions") or [] if isinstance(item, Mapping)]
    if not actions:
        lines.extend(["No blocker-resolution actions are currently required.", ""])
    for item in actions:
        lines.extend(
            [
                f"### {item.get('priority')}. {item.get('title')}",
                "",
                f"Status: `{item.get('status')}`",
                f"Must clear before GPU spend: `{item.get('must_clear_before_gpu_spend')}`",
                "",
            ]
        )
        blockers = [str(value) for value in item.get("blockers") or []]
        warnings = [str(value) for value in item.get("warnings") or []]
        if blockers:
            lines.append("Blockers:")
            lines.extend(f"- `{value}`" for value in blockers)
            lines.append("")
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- `{value}`" for value in warnings)
            lines.append("")
        details = [value for value in item.get("blocker_details") or [] if isinstance(value, Mapping)]
        _append_blocker_details_markdown(lines, details)
        next_actions = [str(value) for value in item.get("next_actions") or []]
        if next_actions:
            lines.append("Next actions:")
            lines.extend(f"- {value}" for value in next_actions)
            lines.append("")
        evidence = [str(value) for value in item.get("evidence_required") or []]
        if evidence:
            lines.append("Evidence required:")
            lines.extend(f"- {value}" for value in evidence)
            lines.append("")
        commands = [str(value) for value in item.get("safe_commands") or []]
        if commands:
            lines.append("Safe commands:")
            lines.append("")
            lines.append("```bash")
            lines.extend(commands)
            lines.append("```")
            lines.append("")
    lines.extend(["## Category Detail", ""])
    for item in payload.get("categories") or []:
        if not isinstance(item, Mapping):
            continue
        lines.extend(
            [
                f"### {item.get('title')}",
                "",
                f"Status: `{item.get('status')}`",
                "",
            ]
        )
        blockers = [str(value) for value in item.get("blockers") or []]
        warnings = [str(value) for value in item.get("warnings") or []]
        if blockers:
            lines.append("Blockers:")
            lines.extend(f"- `{value}`" for value in blockers)
            lines.append("")
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- `{value}`" for value in warnings)
            lines.append("")
        details = [value for value in item.get("blocker_details") or [] if isinstance(value, Mapping)]
        _append_blocker_details_markdown(lines, details)
        next_actions = [str(value) for value in item.get("next_actions") or []]
        if next_actions:
            lines.append("Next actions:")
            lines.extend(f"- {value}" for value in next_actions)
            lines.append("")
        evidence = [str(value) for value in item.get("evidence_required") or []]
        if evidence:
            lines.append("Evidence required:")
            lines.extend(f"- {value}" for value in evidence)
            lines.append("")
        commands = [str(value) for value in item.get("safe_commands") or []]
        if commands:
            lines.append("Safe commands:")
            lines.append("")
            lines.append("```bash")
            lines.extend(commands)
            lines.append("```")
            lines.append("")
    lines.extend(
        [
            "## Final Verification",
            "",
            "```bash",
            *[str(value) for value in payload.get("final_verification_commands") or []],
            "```",
            "",
            "This artifact does not run providers, submit WebApp requests, run simulators, provision GPUs, or upgrade proof claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _launch_step(
    *,
    step_id: str,
    title: str,
    phase_group: str,
    blockers: Sequence[str],
    warnings: Sequence[str] = (),
    required_artifacts: Sequence[str] = (),
    safe_commands: Sequence[str] = (),
    proof_boundary: str,
    status_override: str | None = None,
    may_run_now_override: bool | None = None,
) -> Dict[str, Any]:
    status = status_override or ("blocked" if blockers else "warning" if warnings else "ready")
    may_run_now = may_run_now_override if may_run_now_override is not None else not blockers
    return {
        "step_id": step_id,
        "title": title,
        "phase_group": phase_group,
        "status": status,
        "may_run_now": may_run_now,
        "blockers": _unique_strings(blockers),
        "warnings": _unique_strings(warnings),
        "required_artifacts": list(required_artifacts),
        "safe_commands": list(safe_commands),
        "proof_boundary": proof_boundary,
    }


def _first_gpu_launch_order_manifest(
    *,
    capture_root: Path,
    packet_dir: Path,
    readiness: Mapping[str, Any],
    webapp_handoff: Mapping[str, Any],
    scene_asset_acquisition: Mapping[str, Any],
    simulator_path_matrix: Mapping[str, Any],
    gpu_vm_runtime_preflight_plan: Mapping[str, Any],
    webapp_site_slug: str,
    webapp_staged_inputs_path: Path,
    simulator: str,
    provisioner: str,
    owner_command_supplied: bool,
    vm_sync_manifest: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    source_video = _source_video_category(capture_root)
    readiness_blockers = [str(item) for item in readiness.get("blockers") or []]
    source_blockers = [str(item) for item in source_video.get("blockers") or []]
    source_warnings = [str(item) for item in source_video.get("warnings") or []]
    webapp_blockers = [str(item) for item in webapp_handoff.get("blockers") or []]
    webapp_warnings = [str(item) for item in webapp_handoff.get("warnings") or []]
    scene_blockers = [str(item) for item in scene_asset_acquisition.get("blockers") or []]
    selected_simulator_blockers = [
        str(item) for item in simulator_path_matrix.get("selected_path_blockers") or []
    ]
    gpu_handoff_blockers = [
        item for item in readiness_blockers if item.startswith("pipeline_gpu_handoff:")
    ]
    owner_command_blockers = [
        item
        for item in readiness_blockers
        if item.startswith("simulator_runtime:missing_simulator_command")
        or item.startswith("simulator_runtime:simulator_command")
    ]
    owner_gate_blockers = [
        item
        for item in readiness_blockers
        if item.startswith("simulator_runtime:missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION")
    ]
    vm_sync_status = _string(vm_sync_manifest.get("status")) if vm_sync_manifest else ""
    sync_manifest_blockers = [
        f"gpu_vm_sync_manifest:{item}" for item in (vm_sync_manifest or {}).get("blockers") or []
    ]
    if vm_sync_manifest and vm_sync_status != "ready" and not sync_manifest_blockers:
        sync_manifest_blockers.append(f"gpu_vm_sync_manifest_status:{vm_sync_status or 'unknown'}")
    vm_sync_blockers: list[str] = list(sync_manifest_blockers)
    if not vm_sync_manifest and (scene_blockers or gpu_handoff_blockers):
        vm_sync_blockers.append("prior_scene_or_gpu_handoff_blocked")
    vm_runtime_blockers = [str(item) for item in gpu_vm_runtime_preflight_plan.get("hard_stop_blockers") or []]
    if vm_sync_blockers:
        vm_runtime_blockers.append("gpu_vm_sync_not_ready")
    runtime_result = (
        gpu_vm_runtime_preflight_plan.get("result")
        if isinstance(gpu_vm_runtime_preflight_plan.get("result"), Mapping)
        else {}
    )
    if not bool(runtime_result.get("ready_for_owner_command_attempt")):
        vm_runtime_blockers.append("gpu_vm_runtime_preflight_result_not_ready")
        vm_runtime_blockers.extend(
            f"gpu_vm_runtime_preflight_result:{item}"
            for item in runtime_result.get("blockers") or []
        )
    owner_setup_blockers = ["gpu_vm_runtime_preflight_not_ready"] if vm_runtime_blockers else []

    steps = [
        _launch_step(
            step_id="sample_video_preflight",
            title="Sample Video Preflight",
            phase_group="pre_gpu_inputs",
            blockers=source_blockers,
            warnings=source_warnings,
            required_artifacts=[str(capture_root / "pipeline" / "source_video_preflight_manifest.json")],
            safe_commands=[
                (
                    "blueprint-audit-first-gpu-sample-video --source-video <source-video> "
                    f"--output {shlex.quote(str(capture_root / 'pipeline' / 'source_video_preflight_manifest.json'))}"
                )
            ],
            proof_boundary="Checks local source-video suitability only; does not call providers or prove scene geometry.",
        ),
        _launch_step(
            step_id="webapp_live_handoff",
            title="WebApp Live Handoff",
            phase_group="pre_gpu_inputs",
            blockers=webapp_blockers,
            warnings=webapp_warnings,
            required_artifacts=[
                str(packet_dir / "first_gpu_webapp_handoff.json"),
                str(webapp_staged_inputs_path),
            ],
            safe_commands=[
                (
                    "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
                    "--webapp-job-request <robot_eval_job_request.json> --stage-webapp-request "
                    f"--staged-inputs-path {shlex.quote(str(webapp_staged_inputs_path))}"
                )
            ],
            proof_boundary="Staged request proves handoff shape only; it does not run the job or simulator.",
        ),
        _launch_step(
            step_id="scene_asset_acquisition",
            title="Scene Asset Acquisition",
            phase_group="pre_gpu_inputs",
            blockers=scene_blockers,
            required_artifacts=[str(packet_dir / "first_gpu_scene_asset_acquisition.json")],
            safe_commands=[
                (
                    f"blueprint-run-simulation-automation --capture-root {shlex.quote(str(capture_root))} "
                    "--scene-asset <materialized-scene-asset>"
                )
            ],
            proof_boundary="Scene asset acquisition may prepare simulator inputs; it does not run the owner GPU simulator.",
        ),
        _launch_step(
            step_id="pipeline_gpu_handoff",
            title="Pipeline GPU Handoff",
            phase_group="gpu_handoff",
            blockers=gpu_handoff_blockers,
            required_artifacts=[
                str(capture_root / "pipeline" / "simulation_automation" / "gpu_handoff_packet.json")
            ],
            safe_commands=[
                f"blueprint-run-simulation-automation --capture-root {shlex.quote(str(capture_root))}"
            ],
            proof_boundary="Handoff readiness is a pre-GPU contract; it is not owner simulator proof.",
        ),
        _launch_step(
            step_id="gpu_vm_sync",
            title="GPU VM Sync Verification",
            phase_group="gpu_vm_setup",
            blockers=vm_sync_blockers,
            required_artifacts=[str(packet_dir / "gpu_vm_sync_manifest.json")],
            safe_commands=[
                "Verify every file listed in gpu_vm_sync_manifest.json on the GPU VM before execution."
            ],
            proof_boundary="Sync verification checks files only; it does not copy files or run simulators.",
        ),
        _launch_step(
            step_id="gpu_vm_runtime_preflight",
            title="GPU VM Runtime Preflight",
            phase_group="gpu_vm_setup",
            blockers=vm_runtime_blockers,
            required_artifacts=[
                str(packet_dir / "gpu_vm_runtime_preflight.sh"),
                str(packet_dir / "gpu_vm_runtime_preflight_result.json"),
            ],
            safe_commands=[f"bash {shlex.quote(str(packet_dir / 'gpu_vm_runtime_preflight.sh'))}"],
            proof_boundary="Runtime preflight checks VM environment and hashes only; it does not run the owner simulator command.",
        ),
        _launch_step(
            step_id="owner_gpu_simulator_proof",
            title="Owner GPU Simulator Proof",
            phase_group="owner_gpu_execution",
            blockers=selected_simulator_blockers
            + owner_command_blockers
            + owner_gate_blockers
            + owner_setup_blockers
            + ([] if owner_command_supplied else ["owner_command_not_supplied_to_run_packet"]),
            required_artifacts=[
                str(packet_dir / "owner_command_contract.md"),
                str(capture_root / "pipeline" / "simulation_automation" / "gpu_owner_system_proof.json"),
            ],
            safe_commands=[f"bash {shlex.quote(str(packet_dir / 'gpu_vm_commands.sh'))}"],
            proof_boundary="Only this step may produce owner GPU simulator proof, and only if the wrapper validates required traces.",
        ),
        _launch_step(
            step_id="post_gpu_readiness_audit",
            title="Post-GPU Readiness Audit",
            phase_group="closure",
            blockers=[] if readiness.get("owner_gpu_proof_ready") else ["owner_gpu_proof_not_ready"],
            status_override=(
                "ready" if readiness.get("owner_gpu_proof_ready") else "pending_after_owner_gpu_run"
            ),
            may_run_now_override=bool(readiness.get("owner_gpu_proof_ready")),
            required_artifacts=[str(packet_dir / "first_gpu_e2e_readiness_manifest.json")],
            safe_commands=[
                (
                    "blueprint-audit-first-gpu-e2e-readiness "
                    f"--capture-root {shlex.quote(str(capture_root))} "
                    f"--webapp-site-slug {shlex.quote(webapp_site_slug or '<webapp-site-slug>')} "
                    f"--webapp-staged-inputs {shlex.quote(str(webapp_staged_inputs_path))} "
                    f"--simulator {shlex.quote(simulator)} --provisioner {shlex.quote(provisioner)} "
                    "--simulator-command \"$OWNER_SIMULATOR_COMMAND\" --simulator-command-location remote"
                )
            ],
            proof_boundary="Audit can accept proof only after owner GPU proof files exist and validate.",
        ),
    ]
    blocked_steps = [step["step_id"] for step in steps if step["status"] == "blocked"]
    pre_gpu_blocked = [
        step["step_id"]
        for step in steps
        if step["phase_group"] in {"pre_gpu_inputs", "gpu_handoff"} and step["status"] == "blocked"
    ]
    vm_setup_blocked = [
        step["step_id"]
        for step in steps
        if step["phase_group"] == "gpu_vm_setup" and step["status"] == "blocked"
    ]
    owner_gpu_step = next(
        (step for step in steps if step.get("step_id") == "owner_gpu_simulator_proof"),
        {},
    )
    gpu_execution_allowed = (
        bool(readiness.get("ready_for_first_gpu_attempt"))
        and bool(owner_gpu_step.get("may_run_now"))
        and not pre_gpu_blocked
        and not vm_setup_blocked
    )
    next_action_step_ids = (
        pre_gpu_blocked
        or vm_setup_blocked
        or ([str(owner_gpu_step["step_id"])] if gpu_execution_allowed and owner_gpu_step else [])
        or [step["step_id"] for step in steps if step["status"] == "blocked"][:1]
    )
    forbidden_actions = (
        ["do_not_claim_owner_gpu_or_rank_fidelity"]
        if gpu_execution_allowed
        else [
            "do_not_run_gpu_vm_commands",
            "do_not_claim_webapp_live_forwarding",
            "do_not_claim_scene_asset_ready",
            "do_not_claim_owner_gpu_or_rank_fidelity",
        ]
    )
    return {
        "schema_version": FIRST_GPU_LAUNCH_ORDER_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "packet_dir": str(packet_dir),
        "webapp_site_slug": webapp_site_slug or None,
        "selected_simulator": simulator,
        "selected_provisioner": provisioner,
        "status": "ready_for_owner_gpu_launch" if gpu_execution_allowed else "blocked",
        "gpu_execution_allowed": gpu_execution_allowed,
        "blocked_step_ids": blocked_steps,
        "next_action_step_ids": next_action_step_ids,
        "forbidden_actions_until_ready": forbidden_actions,
        "steps": steps,
        "claim_boundary": {
            "artifact_purpose": "first_gpu_launch_order",
            "webapp_requests_submitted": False,
            "live_provider_calls_performed": False,
            "remote_asset_downloads_performed": False,
            "files_copied": False,
            "gpu_provisioning_performed": False,
            "simulator_execution_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _launch_order_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# First GPU Launch Order",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- GPU execution allowed: `{payload.get('gpu_execution_allowed')}`",
        f"- Selected simulator: `{payload.get('selected_simulator')}`",
        "",
        "## Next Action Steps",
        "",
    ]
    next_steps = [str(item) for item in payload.get("next_action_step_ids") or []]
    lines.extend(f"- `{item}`" for item in next_steps)
    lines.extend(["", "## Ordered Steps", ""])
    for step in payload.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        lines.extend(
            [
                f"### {step.get('title')}",
                "",
                f"- Step id: `{step.get('step_id')}`",
                f"- Group: `{step.get('phase_group')}`",
                f"- Status: `{step.get('status')}`",
                f"- May run now: `{step.get('may_run_now')}`",
            ]
        )
        blockers = [str(item) for item in step.get("blockers") or []]
        if blockers:
            lines.append("- Blockers:")
            lines.extend(f"  - `{item}`" for item in blockers)
        lines.append(f"- Proof boundary: {step.get('proof_boundary')}")
        lines.append("")
    forbidden = [str(item) for item in payload.get("forbidden_actions_until_ready") or []]
    if forbidden:
        lines.extend(["## Forbidden Until Ready", ""])
        lines.extend(f"- `{item}`" for item in forbidden)
        lines.append("")
    lines.append(
        "This launch order does not submit WebApp requests, call providers, copy files, provision GPUs, run simulators, or prove generated-world rank fidelity."
    )
    lines.append("")
    return "\n".join(lines)


def _gpu_vm_sync_manifest(
    *,
    capture_root: Path,
    packet_dir: Path,
    generated_files: Mapping[str, str],
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    entries: list[Dict[str, Any]] = []
    raw_dir = capture_root / "raw"
    _append_file_entry(entries, raw_dir / "manifest.json", role="raw_manifest")
    _append_file_entry(entries, raw_dir / "capture_context.json", role="raw_capture_context")
    _append_file_entry(
        entries,
        raw_dir / "capture_upload_complete.json",
        role="raw_upload_completion_marker",
    )
    raw_video = _raw_video_path(capture_root)
    if raw_video is not None:
        _append_file_entry(entries, raw_video, role="raw_walkthrough_video")
    else:
        _append_file_entry(entries, raw_dir / "walkthrough.mov", role="raw_walkthrough_video")

    _append_file_entry(
        entries,
        capture_root / "pipeline" / "source_video_preflight_manifest.json",
        role="source_video_preflight_manifest",
        required=False,
    )
    simulation_dir = capture_root / "pipeline" / "simulation_automation"
    for role, filename in (
        ("gpu_handoff_packet", "gpu_handoff_packet.json"),
        ("gpu_owner_system_proof_schema", "gpu_owner_system_proof_schema.json"),
        ("gpu_run_checklist", "gpu_run_checklist.md"),
        ("owner_gpu_blocked_manifest", "owner_gpu_simulator_execution_blocked_manifest.json"),
        ("simulator_engine_plugin_registry", "simulator_engine_plugin_registry.json"),
        ("scene_asset_preflight", "scene_asset_preflight.json"),
        ("scene_frame_estimate", "scene_frame_estimate.json"),
        ("spawn_pose_validation_manifest", "spawn_pose_validation_manifest.json"),
    ):
        _append_file_entry(entries, simulation_dir / filename, role=role)
    for role, path in sorted(generated_files.items()):
        if role in {"gpu_vm_sync_manifest", "gpu_vm_sync_markdown"}:
            continue
        _append_file_entry(entries, Path(path), role=f"run_packet_{role}")

    selected_simulator = _string(readiness.get("simulator"))
    if selected_simulator == "mujoco":
        for index, asset_path in enumerate(_mujoco_g1_asset_files()):
            role = "mujoco_menagerie_unitree_g1_asset"
            if asset_path.name == "g1.xml":
                role = "mujoco_menagerie_unitree_g1_g1_xml"
            elif asset_path.name == "scene.xml":
                role = "mujoco_menagerie_unitree_g1_scene_xml"
            elif asset_path.name == "LICENSE":
                role = "mujoco_menagerie_unitree_g1_license"
            _append_file_entry(entries, asset_path, role=f"{role}_{index:03d}")

    blockers = [
        blocker
        for entry in entries
        for blocker in entry.get("blockers", [])
    ]
    hard_preflight_blockers: list[str] = []
    pipeline_stage = readiness.get("stages", {}).get("pipeline_gpu_handoff", {})
    if isinstance(pipeline_stage, Mapping):
        hard_preflight_blockers = [
            str(item) for item in pipeline_stage.get("hard_preflight_blockers") or []
        ]
    if "missing_local_scene_asset" in hard_preflight_blockers:
        blockers.append("scene_asset_missing_for_gpu_vm_sync")
    return {
        "schema_version": FIRST_GPU_VM_SYNC_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "packet_dir": str(packet_dir),
        "status": "ready" if not blockers else "blocked",
        "required_file_count": sum(1 for item in entries if item["required"]),
        "missing_required_file_count": sum(
            1 for item in entries if item["required"] and not item["exists"]
        ),
        "total_existing_bytes": sum(int(item["size_bytes"]) for item in entries if item["exists"]),
        "files": entries,
        "blockers": blockers,
        "hard_preflight_blockers": hard_preflight_blockers,
        "recommended_sync_roots": [
            str(capture_root),
            str(packet_dir),
            str(capture_root / "pipeline" / "simulation_automation"),
        ],
        "post_sync_verification": [
            "Verify every file with exists=true and sha256 in this manifest still matches on the GPU VM.",
            "Do not run the owner simulator until scene asset and spawn blockers are cleared.",
            "Do not treat this sync manifest as simulator execution or generated-world rank fidelity proof.",
        ],
        "claim_boundary": {
            "artifact_purpose": "first_gpu_vm_sync_manifest",
            "files_copied": False,
            "live_provider_calls_performed": False,
            "gpu_provisioning_performed": False,
            "simulator_execution_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _gpu_vm_sync_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GPU VM Sync Manifest",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Capture root: `{payload.get('capture_root')}`",
        f"- Existing bytes: `{payload.get('total_existing_bytes')}`",
        "",
        "## Recommended Sync Roots",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload.get("recommended_sync_roots") or [])
    blockers = [str(item) for item in payload.get("blockers") or []]
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in blockers)
    lines.extend(["", "## Required Files", ""])
    for entry in payload.get("files") or []:
        if not isinstance(entry, Mapping) or not entry.get("required"):
            continue
        lines.append(
            "- "
            f"`{entry.get('role')}` "
            f"exists=`{entry.get('exists')}` "
            f"size=`{entry.get('size_bytes')}` "
            f"sha256=`{entry.get('sha256')}` "
            f"path=`{entry.get('path')}`"
        )
    lines.extend(["", "## Post-Sync Verification", ""])
    lines.extend(f"- {item}" for item in payload.get("post_sync_verification") or [])
    lines.extend(
        [
            "",
            "This manifest does not copy files, provision GPUs, run simulators, or prove generated-world rank fidelity.",
            "",
        ]
    )
    return "\n".join(lines)


def _scene_asset_candidates(capture_root: Path) -> list[Dict[str, Any]]:
    pipeline_dir = capture_root / "pipeline"
    candidates: list[Dict[str, Any]] = []
    export_manifest = _read_mapping(pipeline_dir / "worldlabs_export_manifest.json")
    materialized = _read_mapping(
        pipeline_dir / "worldlabs_assets" / "materialized_assets_manifest.json"
    )
    for role, value in (
        ("worldlabs_output_collider_mesh", export_manifest.get("output_collider_mesh_path")),
        ("worldlabs_collider_mesh_glb", export_manifest.get("collider_mesh_glb_url")),
    ):
        text = _string(value)
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (pipeline_dir / path).resolve()
        candidates.append(
            {
                "role": role,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _sha_file(path) if path.is_file() else None,
            }
        )
    for item in materialized.get("downloads") or []:
        if not isinstance(item, Mapping):
            continue
        path_text = _string(item.get("local_path"))
        if not path_text:
            continue
        path = Path(path_text).expanduser()
        candidates.append(
            {
                "role": _string(item.get("kind")) or "worldlabs_materialized_asset",
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _sha_file(path) if path.is_file() else None,
            }
        )
    seen: set[str] = set()
    unique: list[Dict[str, Any]] = []
    for candidate in candidates:
        key = candidate["path"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _scene_asset_acquisition_manifest(
    *,
    capture_root: Path,
    webapp_site_slug: str,
    provider_submission_script_path: Path,
) -> Dict[str, Any]:
    pipeline_dir = capture_root / "pipeline"
    source_video_manifest = _read_mapping(pipeline_dir / "source_video_preflight_manifest.json")
    provider_run_manifest = _read_mapping(pipeline_dir / "provider_run_manifest.json")
    request_manifest_path = pipeline_dir / "worldlabs_request_manifest.json"
    world_manifest_path = pipeline_dir / "worldlabs_world_manifest.json"
    materialization_manifest_path = pipeline_dir / "worldlabs_assets" / "materialized_assets_manifest.json"
    export_manifest_path = pipeline_dir / "worldlabs_export_manifest.json"
    candidate_assets = _scene_asset_candidates(capture_root)
    existing_candidates = [item for item in candidate_assets if item.get("exists")]
    source_video_ready = (
        source_video_manifest.get("status") == "ready"
        and int(source_video_manifest.get("ready_for_worldlabs_first_clip_count") or 0) > 0
    )
    worldlabs_api_key_configured = bool(_string(os.getenv(WORLDLABS_API_KEY_ENV)))
    worldlabs_provider_submission_gate_enabled = (
        _string(os.getenv(WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV)).lower() == "true"
    )
    request_manifest_exists = request_manifest_path.is_file()
    world_manifest_exists = world_manifest_path.is_file()
    materialization_manifest_exists = materialization_manifest_path.is_file()
    provider_request_inputs_ready = source_video_ready and not request_manifest_exists
    if not source_video_ready:
        provider_submission_status = "blocked_source_video_preflight"
    elif not request_manifest_exists and not worldlabs_api_key_configured:
        provider_submission_status = "blocked_missing_worldlabs_api_key"
    elif not request_manifest_exists and not worldlabs_provider_submission_gate_enabled:
        provider_submission_status = "blocked_missing_worldlabs_submission_gate"
    elif not request_manifest_exists:
        provider_submission_status = "ready_to_submit_worldlabs_request"
    elif not world_manifest_exists:
        provider_submission_status = "waiting_for_worldlabs_world_manifest"
    elif not materialization_manifest_exists or not existing_candidates:
        provider_submission_status = "ready_to_materialize_worldlabs_assets"
    else:
        provider_submission_status = "ready_for_scene_preflight_rerun"
    provider_submission_required_env = (
        [WORLDLABS_API_KEY_ENV, WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV]
        if provider_request_inputs_ready
        else []
    )
    provider_submission_missing_env = []
    if provider_request_inputs_ready and not worldlabs_api_key_configured:
        provider_submission_missing_env.append(WORLDLABS_API_KEY_ENV)
    if provider_request_inputs_ready and not worldlabs_provider_submission_gate_enabled:
        provider_submission_missing_env.append(WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV)
    blockers: list[str] = []
    if not source_video_ready:
        blockers.append("source_video_preflight_not_ready_for_scene_generation")
    if not request_manifest_exists:
        blockers.append("worldlabs_request_manifest_missing")
    if not world_manifest_exists:
        blockers.append("worldlabs_world_manifest_missing")
    if not materialization_manifest_exists:
        blockers.append("worldlabs_asset_materialization_manifest_missing")
    if not existing_candidates:
        blockers.append("materialized_scene_asset_missing")
    status = "ready_for_scene_preflight_rerun" if not blockers else "blocked"
    return {
        "schema_version": FIRST_GPU_SCENE_ASSET_ACQUISITION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "webapp_site_slug": webapp_site_slug or None,
        "status": status,
        "blockers": blockers,
        "source_video_preflight": {
            "path": str(pipeline_dir / "source_video_preflight_manifest.json"),
            "status": source_video_manifest.get("status"),
            "ready_for_worldlabs_first_clip_count": (
                source_video_manifest.get("ready_for_worldlabs_first_clip_count") or 0
            ),
        },
        "provider_submission": {
            "status": provider_submission_status,
            "input_status": (
                "ready_for_worldlabs_request_inputs"
                if provider_request_inputs_ready
                else provider_submission_status
            ),
            "ready_for_worldlabs_request_inputs": provider_request_inputs_ready,
            "ready_to_submit_worldlabs_request": (
                provider_submission_status == "ready_to_submit_worldlabs_request"
            ),
            "safe_to_submit_before_gpu_spend": provider_request_inputs_ready,
            "requires_env": provider_submission_required_env,
            "missing_env": provider_submission_missing_env,
            "required_env_status": {
                WORLDLABS_API_KEY_ENV: {
                    "configured": worldlabs_api_key_configured,
                    "value_redacted": True,
                },
                WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV: {
                    "configured": worldlabs_provider_submission_gate_enabled,
                    "required_value": "true",
                },
            }
            if provider_request_inputs_ready
            else {},
            "script": {
                "path": str(provider_submission_script_path),
                "safe_to_run_now": (
                    provider_submission_status == "ready_to_submit_worldlabs_request"
                ),
                "requires_explicit_allow_env": WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV,
                "requires_explicit_allow_value": "true",
                "runs_live_provider_call": (
                    provider_submission_status == "ready_to_submit_worldlabs_request"
                ),
            },
            "requires_gpu": False,
            "requires_live_provider_call": provider_request_inputs_ready,
            "input_video_preflight_ready": source_video_ready,
            "worldlabs_request_manifest_exists": request_manifest_exists,
            "worldlabs_world_manifest_exists": world_manifest_exists,
            "materialization_manifest_exists": materialization_manifest_exists,
            "candidate_scene_asset_count": len(existing_candidates),
            "proof_boundary": (
                "Provider submission readiness only means the source video is ready for "
                "World Labs scene generation. It does not call providers, download assets, "
                "run simulators, provision GPUs, or prove generated-world rank fidelity."
            ),
        },
        "provider_preview": {
            "provider_run_manifest_path": str(pipeline_dir / "provider_run_manifest.json"),
            "provider_run_manifest_exists": bool(provider_run_manifest),
            "provider_status": provider_run_manifest.get("status") if provider_run_manifest else None,
            "worldlabs_request_manifest_path": str(request_manifest_path),
            "worldlabs_request_manifest_exists": request_manifest_exists,
            "worldlabs_world_manifest_path": str(world_manifest_path),
            "worldlabs_world_manifest_exists": world_manifest_exists,
        },
        "materialization": {
            "materialization_manifest_path": str(materialization_manifest_path),
            "materialization_manifest_exists": materialization_manifest_exists,
            "worldlabs_export_manifest_path": str(export_manifest_path),
            "worldlabs_export_manifest_exists": export_manifest_path.is_file(),
            "candidate_assets": candidate_assets,
        },
        "safe_commands": {
            "run_provider_preview": (
                "BLUEPRINT_PREVIEW_PROVIDER=world_labs "
                "WORLDLABS_API_KEY=<set-in-shell-not-artifact> "
                f"blueprint-run-e2e --capture-root {shlex.quote(str(capture_root))} "
                "--provider openai --pipeline-lane current --run-evaluation-prep "
                "--evaluation-prep-provider manual"
            ),
            "materialize_worldlabs_assets": (
                f"blueprint-materialize-worldlabs-assets --capture-root {shlex.quote(str(capture_root))} "
                "--include-visual-assets"
            ),
            "rerun_simulation_automation_with_scene_asset": (
                f"blueprint-run-simulation-automation --capture-root {shlex.quote(str(capture_root))} "
                "--scene-asset <materialized-scene-asset>"
            ),
            "regenerate_run_packet": (
                f"blueprint-build-first-gpu-run-packet --capture-root {shlex.quote(str(capture_root))} "
                f"--webapp-site-slug {shlex.quote(webapp_site_slug or '<webapp-site-slug>')} "
                "--simulator isaac_sim --provisioner runpod"
            ),
        },
        "ordered_next_steps": [
            "Submit or complete the provider-preview path only with an explicit World Labs key in shell state.",
            "Wait for pipeline/worldlabs_world_manifest.json before materialization.",
            "Run blueprint-materialize-worldlabs-assets to download already-generated asset URLs.",
            "Rerun simulation automation with the materialized scene asset path.",
            "Regenerate the first-GPU run packet and re-check gpu_vm_sync_manifest.json.",
        ],
        "claim_boundary": {
            "artifact_purpose": "first_gpu_scene_asset_acquisition_plan",
            "live_provider_calls_performed": False,
            "remote_asset_downloads_performed": False,
            "scene_asset_attached": bool(existing_candidates),
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _scene_asset_acquisition_markdown(payload: Mapping[str, Any]) -> str:
    commands = payload.get("safe_commands") if isinstance(payload.get("safe_commands"), Mapping) else {}
    materialization = (
        payload.get("materialization") if isinstance(payload.get("materialization"), Mapping) else {}
    )
    provider_submission = (
        payload.get("provider_submission")
        if isinstance(payload.get("provider_submission"), Mapping)
        else {}
    )
    lines = [
        "# Scene Asset Acquisition",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Capture root: `{payload.get('capture_root')}`",
        "",
    ]
    blockers = [str(item) for item in payload.get("blockers") or []]
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- `{item}`" for item in blockers)
        lines.append("")
    if provider_submission:
        lines.extend(
            [
                "## Provider Submission",
                "",
                f"- Status: `{provider_submission.get('status')}`",
                f"- Input status: `{provider_submission.get('input_status')}`",
                "- Request inputs ready: "
                f"`{provider_submission.get('ready_for_worldlabs_request_inputs')}`",
                "- Ready to submit World Labs request: "
                f"`{provider_submission.get('ready_to_submit_worldlabs_request')}`",
                "- Safe before GPU spend: "
                f"`{provider_submission.get('safe_to_submit_before_gpu_spend')}`",
                "- Requires GPU: "
                f"`{provider_submission.get('requires_gpu')}`",
            ]
        )
        requires_env = [str(item) for item in provider_submission.get("requires_env") or []]
        if requires_env:
            lines.append("- Requires env:")
            lines.extend(f"  - `{item}`" for item in requires_env)
        missing_env = [str(item) for item in provider_submission.get("missing_env") or []]
        if missing_env:
            lines.append("- Missing env:")
            lines.extend(f"  - `{item}`" for item in missing_env)
        if provider_submission.get("proof_boundary"):
            lines.append(f"- Proof boundary: {provider_submission.get('proof_boundary')}")
        lines.append("")
    lines.extend(["## Candidate Assets", ""])
    candidates = materialization.get("candidate_assets") or []
    if candidates:
        for item in candidates:
            if isinstance(item, Mapping):
                lines.append(
                    f"- `{item.get('role')}` exists=`{item.get('exists')}` "
                    f"sha256=`{item.get('sha256')}` path=`{item.get('path')}`"
                )
    else:
        lines.append("- None found for this capture.")
    lines.extend(["", "## Ordered Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("ordered_next_steps") or [])
    lines.extend(["", "## Safe Commands", "", "```bash"])
    lines.extend(str(value) for value in commands.values())
    lines.extend(
        [
            "```",
            "",
            "This plan does not call providers, download assets, run simulators, provision GPUs, or prove generated-world rank fidelity.",
            "",
        ]
    )
    return "\n".join(lines)


def _stage_payload(readiness: Mapping[str, Any], name: str) -> Dict[str, Any]:
    stages = readiness.get("stages") if isinstance(readiness.get("stages"), Mapping) else {}
    stage = stages.get(name) if isinstance(stages, Mapping) else {}
    return dict(stage) if isinstance(stage, Mapping) else {}


def _prefixed_stage_items(stage_name: str, stage: Mapping[str, Any], field: str) -> list[str]:
    return [f"{stage_name}:{item}" for item in stage.get(field) or []]


def _webapp_handoff_manifest(
    *,
    capture_root: Path,
    webapp_site_slug: str,
    webapp_staged_inputs_path: Path,
    verification_script_path: Path,
    verification_result_path: Path,
    readiness: Mapping[str, Any],
    allow_local_webapp_rehearsal: bool,
    simulator: str,
    provisioner: str,
    owner_command_location: str,
) -> Dict[str, Any]:
    upstream = _stage_payload(readiness, "webapp_upstream_truth")
    forwarding = _stage_payload(readiness, "webapp_forwarding")
    staged = _stage_payload(readiness, "webapp_staged_request")
    stage_map = {
        "webapp_upstream_truth": upstream,
        "webapp_forwarding": forwarding,
        "webapp_staged_request": staged,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    for stage_name, stage in stage_map.items():
        if not stage:
            blockers.append(f"{stage_name}:stage_missing")
            continue
        blockers.extend(_prefixed_stage_items(stage_name, stage, "blockers"))
        warnings.extend(_prefixed_stage_items(stage_name, stage, "warnings"))

    upstream_fields = (
        upstream.get("fields") if isinstance(upstream.get("fields"), Mapping) else {}
    )
    staged_fields = (
        staged.get("fields_present") if isinstance(staged.get("fields_present"), Mapping) else {}
    )
    upstream_id_requirements = [
        {
            "field": field,
            "capture_manifest_present": bool(upstream_fields.get(field)),
            "staged_request_present": bool(staged_fields.get(field)),
        }
        for field in WEBAPP_HANDOFF_UPSTREAM_FIELDS
    ]
    status = "ready_for_webapp_handoff_verification" if not blockers else "blocked"
    staged_path = Path(_string(staged.get("path")) or str(webapp_staged_inputs_path)).expanduser()
    request_path = _string(staged.get("request_path"))
    forward_url_configured = bool(forwarding.get("forward_url_configured"))
    forward_token_configured = bool(forwarding.get("forward_token_configured"))
    forward_url_evidence_present = bool(
        forwarding.get("forward_url_evidence_present", forward_url_configured)
    )
    forward_token_evidence_present = bool(
        forwarding.get("forward_token_evidence_present", forward_token_configured)
    )
    capture_root_override_configured = bool(forwarding.get("capture_root_override_configured"))
    forwarding_preflight = (
        forwarding.get("forwarding_preflight")
        if isinstance(forwarding.get("forwarding_preflight"), Mapping)
        else {}
    )
    forwarding_preflight_path = _string(forwarding_preflight.get("path"))
    forwarding_preflight_ready = bool(forwarding_preflight.get("ready"))
    missing_env: list[str] = []
    if not forward_url_evidence_present:
        missing_env.append(WEBAPP_FORWARD_URL_ENV)
    if not forward_token_evidence_present:
        missing_env.append(WEBAPP_FORWARD_TOKEN_ENV)
    if not capture_root_override_configured:
        missing_env.append(
            f"{WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV} or {WEBAPP_FORWARD_CAPTURE_ROOT_ENV}"
        )
    return {
        "schema_version": FIRST_GPU_WEBAPP_HANDOFF_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(capture_root),
        "webapp_site_slug": webapp_site_slug or None,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "upstream_id_requirements": upstream_id_requirements,
        "forwarding": {
            "forward_url_configured": forward_url_configured,
            "forward_token_configured": forward_token_configured,
            "forward_url_evidence_present": forward_url_evidence_present,
            "forward_token_evidence_present": forward_token_evidence_present,
            "forward_token_value_redacted": True,
            "capture_root_override_configured": capture_root_override_configured,
            "capture_root_override_source": forwarding.get("capture_root_override_source"),
            "forwarding_preflight": dict(forwarding_preflight),
            "expected_capture_root": str(capture_root),
        },
        "verification": {
            "requires_env": [
                (
                    f"{WEBAPP_FORWARD_URL_ENV} and {WEBAPP_FORWARD_TOKEN_ENV}, "
                    f"or {FORWARD_PREFLIGHT_REPORT_ENV}"
                ),
                (
                    f"{WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV} or "
                    f"{WEBAPP_FORWARD_CAPTURE_ROOT_ENV}, or {FORWARD_PREFLIGHT_REPORT_ENV}"
                ),
            ],
            "missing_env": missing_env,
            "required_env_status": {
                WEBAPP_FORWARD_URL_ENV: {
                    "configured": forward_url_configured,
                    "value_redacted": True,
                },
                WEBAPP_FORWARD_TOKEN_ENV: {
                    "configured": forward_token_configured,
                    "value_redacted": True,
                },
                WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV: {
                    "configured": bool(_string(os.getenv(WEBAPP_FORWARD_CAPTURE_ROOT_BY_SITE_ENV))),
                    "expected_site_slug": webapp_site_slug or None,
                    "value_redacted": True,
                },
                WEBAPP_FORWARD_CAPTURE_ROOT_ENV: {
                    "configured": bool(_string(os.getenv(WEBAPP_FORWARD_CAPTURE_ROOT_ENV))),
                    "value_redacted": True,
                },
                FORWARD_PREFLIGHT_REPORT_ENV: {
                    "configured": bool(forwarding_preflight_path),
                    "path": forwarding_preflight_path or None,
                    "ready": forwarding_preflight_ready,
                    "value_redacted": False,
                },
            },
            "script": {
                "path": str(verification_script_path),
                "default_result_path": str(verification_result_path),
                "safe_to_run_now": True,
                "runs_live_webapp_call": False,
                "stages_request": False,
                "requires_forwarding_token_in_shell": not forwarding_preflight_ready,
            },
        },
        "staged_request": {
            "path": str(staged_path),
            "exists": staged_path.is_file(),
            "required": bool(staged.get("required", True)),
            "status": staged.get("status"),
            "job_id_present": bool(staged.get("job_id")),
            "request_path": request_path or None,
            "request_path_exists": Path(request_path).expanduser().is_file()
            if request_path
            else False,
            "source_kind": staged.get("source_kind"),
            "local_rehearsal_only": bool(staged.get("local_rehearsal_only")),
            "local_rehearsal_allowed": allow_local_webapp_rehearsal,
            "request_capture_root_configured": bool(
                staged.get("request_capture_root_configured")
            ),
        },
        "ordered_next_steps": [
            "Populate real WebApp/Capture upstream IDs before claiming live WebApp handoff.",
            "Set forwarding URL, token, required flag, and capture-root-by-site override in shell or host env.",
            "Run the authenticated Pipeline intake service on the forwarding host.",
            "Submit the real WebApp robot-eval request for this site slug.",
            "Validate or stage the request through blueprint-intake-live-pipeline-inputs.",
            "Regenerate this run packet and rerun the first-GPU readiness audit.",
        ],
        "safe_commands": {
            "configure_webapp_forwarding_env": "\n".join(
                [
                    _shell_export(
                        "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL",
                        "https://<pipeline-host>/api/live-pipeline/job-requests",
                    ),
                    _shell_export("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "<redacted>"),
                    _shell_export("ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED", "true"),
                    _shell_export(
                        "ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON",
                        _capture_root_by_site_json(
                            site_slug=webapp_site_slug or "<webapp-site-slug>",
                            capture_root=capture_root,
                        ),
                    ),
                    _shell_export(
                        FORWARD_PREFLIGHT_REPORT_ENV,
                        forwarding_preflight_path
                        or str(_default_webapp_forwarding_preflight_path(capture_root)),
                    ),
                    _shell_export(
                        "BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH",
                        str(webapp_staged_inputs_path),
                    ),
                ]
            ),
            "write_webapp_forwarding_preflight_report": (
                "cd /Users/nijelhunt_1/workspace/Blueprint-WebApp && "
                "npm run pipeline:forwarding:preflight -- --require-forwarding "
                "--probe-intake-audit "
                f"--output {shlex.quote(forwarding_preflight_path or str(_default_webapp_forwarding_preflight_path(capture_root)))}"
            ),
            "start_pipeline_intake_service": "\n".join(
                [
                    _shell_export("BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN", "<redacted>"),
                    "blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765",
                ]
            ),
            "stage_or_verify_webapp_request": (
                "blueprint-intake-live-pipeline-inputs --manifest-path <control-plane-manifest> "
                "--webapp-job-request <robot_eval_job_request.json> --stage-webapp-request "
                f"--staged-inputs-path {shlex.quote(str(webapp_staged_inputs_path))}"
            ),
            "verify_webapp_handoff": f"bash {shlex.quote(str(verification_script_path))}",
            "rerun_first_gpu_readiness": (
                "blueprint-audit-first-gpu-e2e-readiness "
                f"--capture-root {shlex.quote(str(capture_root))} "
                f"--webapp-site-slug {shlex.quote(webapp_site_slug or '<webapp-site-slug>')} "
                f"--webapp-staged-inputs {shlex.quote(str(webapp_staged_inputs_path))} "
                f"--webapp-forwarding-preflight {shlex.quote(forwarding_preflight_path or str(_default_webapp_forwarding_preflight_path(capture_root)))} "
                f"--simulator {shlex.quote(simulator)} --provisioner {shlex.quote(provisioner)} "
                "--simulator-command \"$OWNER_SIMULATOR_COMMAND\" "
                f"--simulator-command-location {shlex.quote(owner_command_location)}"
                + (" --allow-local-webapp-rehearsal" if allow_local_webapp_rehearsal else "")
            ),
        },
        "claim_boundary": {
            "artifact_purpose": "first_gpu_webapp_handoff_plan",
            "webapp_request_submitted_by_this_packet": False,
            "live_forwarding_performed_by_this_packet": False,
            "live_forwarding_proven": False,
            "local_rehearsal_only_observed": bool(staged.get("local_rehearsal_only")),
            "webapp_request_staged_observed": bool(staged.get("ready")),
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _webapp_handoff_markdown(payload: Mapping[str, Any]) -> str:
    commands = payload.get("safe_commands") if isinstance(payload.get("safe_commands"), Mapping) else {}
    verification = (
        payload.get("verification")
        if isinstance(payload.get("verification"), Mapping)
        else {}
    )
    verification_script = (
        verification.get("script")
        if isinstance(verification.get("script"), Mapping)
        else {}
    )
    lines = [
        "# WebApp Handoff Packet",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Capture root: `{payload.get('capture_root')}`",
        f"- WebApp site slug: `{payload.get('webapp_site_slug')}`",
        "",
    ]
    blockers = [str(item) for item in payload.get("blockers") or []]
    warnings = [str(item) for item in payload.get("warnings") or []]
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- `{item}`" for item in blockers)
        lines.append("")
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- `{item}`" for item in warnings)
        lines.append("")
    lines.extend(["## Upstream IDs", ""])
    for item in payload.get("upstream_id_requirements") or []:
        if isinstance(item, Mapping):
            lines.append(
                "- "
                f"`{item.get('field')}` "
                f"capture_manifest_present=`{item.get('capture_manifest_present')}` "
                f"staged_request_present=`{item.get('staged_request_present')}`"
            )
    lines.extend(["", "## Ordered Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("ordered_next_steps") or [])
    if verification:
        lines.extend(["", "## Verification Script", ""])
        lines.append(f"- Path: `{verification_script.get('path')}`")
        lines.append(f"- Safe to run now: `{verification_script.get('safe_to_run_now')}`")
        lines.append(
            f"- Runs live WebApp call: `{verification_script.get('runs_live_webapp_call')}`"
        )
        requires_env = [str(item) for item in verification.get("requires_env") or []]
        missing_env = [str(item) for item in verification.get("missing_env") or []]
        if requires_env:
            lines.append("- Requires env:")
            lines.extend(f"  - `{item}`" for item in requires_env)
        if missing_env:
            lines.append("- Missing env:")
            lines.extend(f"  - `{item}`" for item in missing_env)
    lines.extend(["", "## Safe Commands", ""])
    for command in commands.values():
        lines.extend(["```bash", str(command), "```", ""])
    lines.extend(
        [
            "This packet does not submit WebApp requests, perform live forwarding, run simulators, provision GPUs, or prove generated-world rank fidelity.",
            "",
        ]
    )
    return "\n".join(lines)


def _owner_command_contract(*, simulator: str) -> str:
    robot_asset = _robot_asset_for_simulator(simulator)
    if simulator == "mujoco":
        selected_smoke = """For MuJoCo packets, the run packet includes a concrete smoke command:

```bash
export PACKET_DIR="<packet-dir>"
export BLUEPRINT_MUJOCO_G1_MODEL_ROOT="<repo>/output/external_assets/mujoco_menagerie/unitree_g1"
export OWNER_SIMULATOR_COMMAND="bash $PACKET_DIR/run_mujoco_unitree_g1_smoke.sh"
```

That launcher runs `mujoco_unitree_g1_smoke.py` with the canonical MuJoCo Python
bindings. The script loads the staged World Labs GLB as converted OBJ support,
loads the real MuJoCo Menagerie Unitree G1 MJCF and mesh assets, runs the default
kinematic walk-to-target smoke, captures MuJoCo renderer frames, and writes the
trace files named above. It still does not create physical robot POV,
robot-team policy proof, contact/safety proof, or generated-world rank fidelity.
"""
    else:
        selected_smoke = """For Isaac Sim packets, the run packet also includes a concrete smoke command:

```bash
export PACKET_DIR="<packet-dir>"
export OWNER_SIMULATOR_COMMAND="bash $PACKET_DIR/run_isaac_unitree_g1_smoke.sh"
```

That launcher runs `isaac_unitree_g1_smoke.py` inside Isaac Sim Python. The script
converts the staged World Labs GLB to USD with `omni.kit.asset_converter`, references
the Unitree G1 USD asset, runs the default kinematic walk-to-target smoke, captures
Isaac Sim virtual camera frames, and writes the trace files named above. It still
does not create real robot POV or robot-team policy proof.
"""
    return f"""# Owner GPU Command Contract

The command passed to `blueprint-run-owner-gpu-proof --command` must run inside the
selected owner runtime and write the trace files named by these environment variables:

- `BLUEPRINT_CAPTURE_ROOT`: staged capture root.
- `BLUEPRINT_GPU_PROOF_DIR`: folder for owner proof support files.
- `BLUEPRINT_SCENE_LOAD_TRACE`: JSON trace proving the simulator loaded the scene.
- `BLUEPRINT_SPAWN_TRACE`: JSON trace proving the robot spawn pose was attempted and validated.
- `BLUEPRINT_DEFAULT_SMOKE_POLICY`: wrapper-written JSON spec for the default walk-to-target smoke policy.
- `BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET`: target label or pose id for the default smoke policy.
- `BLUEPRINT_ROBOT_ASSET_NAME`: expected robot asset name; defaults to `{robot_asset["name"]}` for `{simulator}`.
- `BLUEPRINT_ROBOT_ASSET_URI_OR_PATH`: expected robot asset path; defaults to `{robot_asset["uri_or_path"]}`.
- `BLUEPRINT_ROBOT_ASSET_SOURCE`: expected source; defaults to `{robot_asset["source"]}`.
- `BLUEPRINT_ACTION_OR_POLICY_TRACE`: JSON trace for the action, policy, or scripted task attempt.
- `BLUEPRINT_POLICY_EXECUTION_TRACE`: same required trace path, named explicitly for policy execution.
- `BLUEPRINT_SIM_ROBOT_POV_EVIDENCE`: JSON manifest for simulator robot POV video or frame evidence.
- `BLUEPRINT_ARTIFACT_MANIFEST`: JSON manifest for rendered frames, logs, USD files, and other outputs.
- `BLUEPRINT_OWNER_STDOUT` and `BLUEPRINT_OWNER_STDERR`: wrapper-owned log paths.

After the command loads the scene, spawns the robot, and captures at least one simulator
robot camera frame or video, it may call the repo helper to write the default policy and
simulator POV artifacts:

```bash
{OWNER_DEFAULT_SMOKE_HELPER_COMMAND} \\
  --simulator {simulator} \\
  --sim-pov-frame "$SIM_ROBOT_POV_FRAME_PATH"
```

The helper writes `BLUEPRINT_POLICY_EXECUTION_TRACE`, `BLUEPRINT_SIM_ROBOT_POV_EVIDENCE`,
and merges those outputs into `BLUEPRINT_ARTIFACT_MANIFEST`. It requires a real simulator
frame or video path from the owner command and does not write scene-load or spawn proof.
The generated run packet also includes `owner_default_smoke_command_binding.sh`, a
fail-closed shell template that wires owner-provided scene-load, spawn, and
walk-to-target commands to this helper.

{selected_smoke}

Minimum trace shape:

```json
{{"status":"loaded","simulator":"{simulator}","scene_loaded":true,"scene_artifacts":["<path>"]}}
```

```json
{{"status":"validated","spawn_pose_loaded":true,"robot_asset":{{"name":"{robot_asset["name"]}","uri_or_path":"{robot_asset["uri_or_path"]}","source":"{robot_asset["source"]}","asset_class":"{robot_asset["asset_class"]}"}}}}
```

```json
{{"status":"completed","default_policy_executed":true,"actions":[{{"name":"walk_to_target","target":"<BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET>","status":"attempted"}}]}}
```

```json
{{"status":"complete","sim_robot_pov_captured":true,"frames":[{{"camera":"front_rgbd","path":"<path>"}}]}}
```

```json
{{"status":"complete","artifacts":[{{"kind":"log","path":"<path>"}}]}}
```

The wrapper can accept simulator execution, default smoke-policy execution, and simulator POV
evidence from this command. That still does not prove real robot POV, generated-world rank fidelity,
safety, physics contact validity, robot-team policy quality, or public claim upgrades.
For `{simulator}`, launch-level selected-asset proof requires the wrapper robot asset
and spawn trace robot asset to match. A procedural humanoid proxy can be recorded as
fallback simulator evidence, but it does not clear the selected Unitree G1 asset proof.
"""


def _live_policy_execution_contract() -> str:
    return """# Live Policy Execution Contract

The first owner-GPU smoke can prove default simulator policy execution only:
`owner_gpu_default_policy_execution_proven=true` / `default_sim_policy_execution_proven=true`.
That is not robot-team policy execution proof.

Live robot-team policy execution is proven only by the robot-eval job artifacts:

- `pipeline/robot_eval_jobs/<job_id>/policy_execution_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/policy_execution_trace.json`

Required manifest evidence:

```json
{
  "status": "completed",
  "selected_modalities": ["policy_api_endpoint"],
  "robot_policy_execution_proven": true,
  "modality_results": {
    "policy_api_endpoint": {
      "status": "completed",
      "execution_performed": true,
      "reference_replayed": false,
      "robot_policy_execution_proven": true
    }
  }
}
```

Required trace evidence:

```json
{
  "status": "completed",
  "robot_policy_execution_proven": true,
  "scenario_eval_run_coverage_complete": true,
  "attempts": [
    {
      "scenario_eval_run_id": "<scenario-eval-run-id>",
      "scenario_variation_instance_id": "<variation-id>",
      "status": "completed",
      "actions": [{"action": "walk_to_target"}]
    }
  ]
}
```

The closure audit rejects reference replay, empty traces, missing action or skill
traces, missing selected modality results, and incomplete scenario-eval-run coverage.
To run a live policy command through the job orchestrator, both are required:

- `BLUEPRINT_ALLOW_POLICY_EXECUTION=true`
- the job call passes `allow_policy_execution=True` or `--allow-policy-execution`

For a default test run without a robot-team policy package, stage the request with
an explicit default policy:

```json
{
  "default_test_policy": {
    "policy_kind": "walk_to_target",
    "target": "<scene-target-or-pose-id>"
  }
}
```

With the same `BLUEPRINT_ALLOW_POLICY_EXECUTION=true` gate, the job orchestrator
runs the built-in default `walk_to_target` adapter and writes:

```json
{
  "status": "completed",
  "selected_modalities": ["high_level_skill_trace"],
  "robot_policy_execution_proven": true,
  "default_test_policy_execution_proven": true,
  "robot_team_policy_execution_proven": false,
  "scenario_eval_run_coverage_complete": true
}
```

This proves only the default Blueprint test policy for that job. It is useful for
the first controlled run, but it is still not proof of robot-team policy quality.

Policy package intake alone is not proof. It only stages robot-team references for
the later gated execution bundle. Generated/default smoke policy artifacts are also
not proof of robot-team policy quality or physical generated-world rank fidelity.
"""


def _default_test_robot_eval_job_request_template(
    *,
    capture_root: Path,
    webapp_site_slug: str,
    simulator: str,
) -> Dict[str, Any]:
    robot_asset = _robot_asset_for_simulator(simulator)
    return {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": "<real-webapp-job-id>",
        "customer": {
            "id": "<real-robot-team-id>",
            "name": "<real-robot-team-name>",
        },
        "site_package": {
            "capture_root": str(capture_root),
            "site_slug": webapp_site_slug or "<webapp-site-slug>",
            "site_submission_id": "<real-site-submission-id>",
            "request_id": "<real-webapp-request-id>",
            "buyer_request_id": "<real-buyer-request-id>",
            "capture_job_id": "<real-capture-job-id>",
            "package_uri": "<pipeline-or-storage-package-uri>",
        },
        "requested_tasks": [
            {
                "task_id": "<task-id-from-real-request>",
                "scenario_ids": ["<scenario-id-from-evaluation-prep>"],
            }
        ],
        "requested_scenario_eval_runs": [
            {
                "scenario_eval_run_id": "<scenario-eval-run-id>",
                "scenario_variation_instance_id": "<scenario-variation-instance-id>",
                "task_id": "<task-id-from-real-request>",
                "scenario_id": "<scenario-id-from-evaluation-prep>",
                "variation_name": "<variation-name>",
            }
        ],
        "robot_profile": {
            "robot_profile_id": "<robot-profile-id>",
            "embodiment": "<robot-embodiment>",
            "sensors": ["rgb", "depth"],
            "simulator_robot_asset": {
                "name": robot_asset["name"],
                "uri_or_path": robot_asset["uri_or_path"],
                "source": robot_asset["source"],
                "asset_class": robot_asset["asset_class"],
                "fail_closed_if_missing": True,
            },
        },
        "default_test_policy": {
            "policy_kind": "walk_to_target",
            "target": "walk_to_target_pose",
        },
        "operation": "evaluate_only",
        "simulator_preference": simulator,
        "rights_privacy_scope": {
            "status": "cleared_for_robot_eval",
            "external_use_allowed": True,
            "privacy_scope": "<approved-privacy-scope>",
        },
        "owner_system": {
            "name": "<real-owner-system-name>",
            "request_id": "<real-webapp-request-id>",
        },
        "provenance": {
            "submitted_at": "<webapp-submitted-at-iso8601>",
            "timestamp_alignment": "trace_timestamps_aligned_to_capture",
        },
        "claim_boundary": {
            "template_only": True,
            "default_test_policy_execution_requested": True,
            "robot_team_policy_execution_requested": False,
            "real_robot_pov_required_separately": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _real_robot_pov_manifest_template() -> Dict[str, Any]:
    return {
        "schema_version": "real_robot_pov_manifest.v1",
        "job_id": "<real-webapp-job-id>",
        "owner_system": "<physical-robot-owner-system>",
        "timestamp_alignment": "aligned_to_scenario_eval_run",
        "records": [
            {
                "evidence_id": "<real-pov-evidence-id>",
                "task_id": "<task-id-from-real-request>",
                "scenario_id": "<scenario-id-from-evaluation-prep>",
                "scenario_eval_run_id": "<scenario-eval-run-id>",
                "scenario_variation_instance_id": "<scenario-variation-instance-id>",
                "variation_name": "<variation-name>",
                "robot_camera_video_uri": "<owner-system-robot-camera-video-uri>",
                "action_log_uri": "<owner-system-action-log-uri>",
                "robot_state_log_uri": "<owner-system-robot-state-log-uri>",
                "owner_evidence_refs": {
                    "camera": "<owner-system-robot-camera-video-uri>",
                    "action_log": "<owner-system-action-log-uri>",
                },
                "operator_attestation": {
                    "attested_by": "<operator-id>",
                    "attestation": (
                        "Robot POV video and action log are from the physical robot "
                        "run for this exact scenario eval run."
                    ),
                },
            }
        ],
        "claim_boundary": {
            "template_only": True,
            "real_robot_pov_evidence_required": True,
            "generated_or_simulator_pov_not_accepted": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _live_input_staging_commands(
    *,
    capture_root: Path,
    packet_dir: Path,
    webapp_job_request_template_path: Path,
    real_robot_pov_template_path: Path,
    webapp_staged_inputs_path: Path,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

{_shell_default_assignment("CAPTURE_ROOT", str(capture_root))}
{_shell_default_assignment("PACKET_DIR", str(packet_dir))}
{_shell_default_assignment("WEBAPP_JOB_REQUEST_PATH", str(webapp_job_request_template_path))}
{_shell_default_assignment("REAL_ROBOT_POV_MANIFEST_PATH", str(real_robot_pov_template_path))}
{_shell_default_assignment("BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH", str(webapp_staged_inputs_path))}
: "${{BLUEPRINT_LIVE_PIPELINE_CONTROL_PLANE_MANIFEST:?Set the live control-plane manifest path}}"
: "${{BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS:?Set true only after replacing template placeholders with real WebApp and robot POV evidence}}"

if [[ "$BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS" != "true" ]]; then
  echo "Refusing to stage first-GPU live inputs without BLUEPRINT_ALLOW_STAGING_FIRST_GPU_LIVE_INPUTS=true" >&2
  exit 2
fi

"${{PYTHON:-python3}}" - "$WEBAPP_JOB_REQUEST_PATH" "$REAL_ROBOT_POV_MANIFEST_PATH" <<'PY'
from pathlib import Path
import json
import sys

for raw in sys.argv[1:]:
    path = Path(raw).expanduser()
    payload = path.read_text(encoding="utf-8")
    json.loads(payload)
    if "<" in payload or ">" in payload:
        raise SystemExit(f"placeholder values remain in {{path}}")
PY

blueprint-intake-live-pipeline-inputs \\
  --manifest-path "$BLUEPRINT_LIVE_PIPELINE_CONTROL_PLANE_MANIFEST" \\
  --webapp-job-request "$WEBAPP_JOB_REQUEST_PATH" \\
  --real-robot-pov "$REAL_ROBOT_POV_MANIFEST_PATH" \\
  --stage-webapp-request \\
  --stage-real-robot-pov \\
  --staged-inputs-path "$BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"
"""


def build_first_gpu_run_packet(
    *,
    capture_root: str | Path,
    webapp_site_slug: str = "",
    webapp_staged_inputs_path: str | Path | None = None,
    webapp_forwarding_preflight_path: str | Path | None = None,
    simulator: str = "isaac_sim",
    provisioner: str = "runpod",
    owner_command: str | None = None,
    owner_command_location: str = "remote",
    output_dir: str | Path | None = None,
    require_webapp_forwarding: bool = True,
    require_webapp_staged_request: bool = True,
    allow_local_webapp_rehearsal: bool = False,
    require_gpu_gates: bool = True,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    selected_simulator = _string(simulator) or "isaac_sim"
    selected_provisioner = _string(provisioner) or "runpod"
    selected_owner_command_location = _string(owner_command_location) or "remote"
    if selected_owner_command_location not in SIMULATOR_COMMAND_LOCATIONS:
        selected_owner_command_location = "remote"
    selected_staged_inputs_path = (
        Path(webapp_staged_inputs_path).expanduser().resolve()
        if webapp_staged_inputs_path
        else context.capture_root / "pipeline" / "live_pipeline_staged_inputs.json"
    )
    selected_forwarding_preflight_path = _selected_webapp_forwarding_preflight_path(
        context.capture_root,
        webapp_forwarding_preflight_path,
    )
    packet_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _default_output_dir(context.capture_root)
    )
    ensure_dir(packet_dir)
    owner_command_was_supplied = bool(_string(owner_command))
    owner_command_generated_by_packet = selected_simulator == "mujoco" and not owner_command_was_supplied
    selected_owner_command = (
        _string(owner_command)
        or (_generated_mujoco_owner_command(packet_dir) if selected_simulator == "mujoco" else "")
        or _default_owner_command(selected_simulator)
    )
    owner_command_available = owner_command_was_supplied or owner_command_generated_by_packet
    readiness_owner_command = selected_owner_command if owner_command_available else ""

    readiness_path = packet_dir / "first_gpu_e2e_readiness_manifest.json"
    readiness = build_first_gpu_e2e_readiness(
        capture_root=context.capture_root,
        webapp_site_slug=webapp_site_slug,
        webapp_staged_inputs_path=selected_staged_inputs_path,
        webapp_forwarding_preflight_path=selected_forwarding_preflight_path,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        simulator_command=readiness_owner_command,
        simulator_command_location=selected_owner_command_location,
        require_webapp_forwarding=require_webapp_forwarding,
        require_webapp_staged_request=require_webapp_staged_request,
        allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
        require_gpu_gates=require_gpu_gates,
    )
    write_json(readiness_path, readiness)

    env_path = packet_dir / "first_gpu_env.example"
    local_commands_path = packet_dir / "local_preflight_commands.sh"
    worldlabs_provider_submission_commands_path = (
        packet_dir / "worldlabs_provider_submission_commands.sh"
    )
    webapp_upstream_truth_verification_commands_path = (
        packet_dir / "webapp_upstream_truth_verification_commands.sh"
    )
    webapp_upstream_truth_verification_result_path = (
        packet_dir / "webapp_upstream_truth_verification_result.json"
    )
    webapp_handoff_verification_commands_path = (
        packet_dir / "webapp_handoff_verification_commands.sh"
    )
    webapp_handoff_verification_result_path = (
        packet_dir / "webapp_handoff_verification_result.json"
    )
    gpu_commands_path = packet_dir / "gpu_vm_commands.sh"
    owner_contract_path = packet_dir / "owner_command_contract.md"
    owner_command_binding_template_path = packet_dir / "owner_default_smoke_command_binding.sh"
    isaac_unitree_g1_smoke_script_path = packet_dir / "isaac_unitree_g1_smoke.py"
    isaac_unitree_g1_smoke_launcher_path = packet_dir / "run_isaac_unitree_g1_smoke.sh"
    mujoco_unitree_g1_smoke_script_path = packet_dir / "mujoco_unitree_g1_smoke.py"
    mujoco_unitree_g1_smoke_launcher_path = packet_dir / "run_mujoco_unitree_g1_smoke.sh"
    live_policy_execution_contract_path = packet_dir / "live_policy_execution_contract.md"
    default_test_job_request_template_path = (
        packet_dir / "default_test_robot_eval_job_request.template.json"
    )
    real_robot_pov_template_path = packet_dir / "real_robot_pov_manifest.template.json"
    live_input_staging_commands_path = packet_dir / "stage_first_gpu_live_inputs.sh"
    provider_bootstrap_path = packet_dir / "gpu_provider_bootstrap.md"
    provider_bootstrap_manifest_path = packet_dir / "gpu_provider_bootstrap.json"
    blocker_resolution_path = packet_dir / "first_gpu_blocker_resolution.json"
    blocker_resolution_markdown_path = packet_dir / "first_gpu_blocker_resolution.md"
    scene_asset_acquisition_path = packet_dir / "first_gpu_scene_asset_acquisition.json"
    scene_asset_acquisition_markdown_path = packet_dir / "first_gpu_scene_asset_acquisition.md"
    webapp_handoff_path = packet_dir / "first_gpu_webapp_handoff.json"
    webapp_handoff_markdown_path = packet_dir / "first_gpu_webapp_handoff.md"
    gpu_vm_runtime_preflight_script_path = packet_dir / "gpu_vm_runtime_preflight.sh"
    gpu_vm_runtime_preflight_plan_path = packet_dir / "gpu_vm_runtime_preflight_plan.json"
    gpu_vm_runtime_preflight_markdown_path = packet_dir / "gpu_vm_runtime_preflight_plan.md"
    gpu_vm_runtime_preflight_result_path = packet_dir / "gpu_vm_runtime_preflight_result.json"
    simulator_path_matrix_path = packet_dir / "first_gpu_simulator_path_matrix.json"
    simulator_path_matrix_markdown_path = packet_dir / "first_gpu_simulator_path_matrix.md"
    launch_order_path = packet_dir / "first_gpu_launch_order.json"
    launch_order_markdown_path = packet_dir / "first_gpu_launch_order.md"
    vm_sync_manifest_path = packet_dir / "gpu_vm_sync_manifest.json"
    vm_sync_markdown_path = packet_dir / "gpu_vm_sync_manifest.md"
    packet_path = packet_dir / "first_gpu_run_packet.json"
    provider_bootstrap = _gpu_provider_bootstrap_manifest(
        capture_root=context.capture_root,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command=selected_owner_command,
        owner_command_location=selected_owner_command_location,
    )
    simulator_path_matrix = _simulator_path_matrix_manifest(
        capture_root=context.capture_root,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command_location=selected_owner_command_location,
        readiness=readiness,
    )
    gpu_vm_runtime_preflight_plan = _gpu_vm_runtime_preflight_plan_manifest(
        capture_root=context.capture_root,
        packet_dir=packet_dir,
        script_path=gpu_vm_runtime_preflight_script_path,
        result_path=gpu_vm_runtime_preflight_result_path,
        sync_manifest_path=vm_sync_manifest_path,
        readiness=readiness,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command=selected_owner_command,
        owner_command_location=selected_owner_command_location,
        owner_command_supplied=owner_command_available,
    )
    blocker_resolution = _blocker_resolution_manifest(
        capture_root=context.capture_root,
        readiness=readiness,
        webapp_site_slug=webapp_site_slug,
        webapp_staged_inputs_path=selected_staged_inputs_path,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command=selected_owner_command,
        owner_command_location=selected_owner_command_location,
        allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
    )
    scene_asset_acquisition = _scene_asset_acquisition_manifest(
        capture_root=context.capture_root,
        webapp_site_slug=webapp_site_slug,
        provider_submission_script_path=worldlabs_provider_submission_commands_path,
    )
    webapp_handoff = _webapp_handoff_manifest(
        capture_root=context.capture_root,
        webapp_site_slug=webapp_site_slug,
        webapp_staged_inputs_path=selected_staged_inputs_path,
        verification_script_path=webapp_handoff_verification_commands_path,
        verification_result_path=webapp_handoff_verification_result_path,
        readiness=readiness,
        allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command_location=selected_owner_command_location,
    )
    launch_order = _first_gpu_launch_order_manifest(
        capture_root=context.capture_root,
        packet_dir=packet_dir,
        readiness=readiness,
        webapp_handoff=webapp_handoff,
        scene_asset_acquisition=scene_asset_acquisition,
        simulator_path_matrix=simulator_path_matrix,
        gpu_vm_runtime_preflight_plan=gpu_vm_runtime_preflight_plan,
        webapp_site_slug=webapp_site_slug,
        webapp_staged_inputs_path=selected_staged_inputs_path,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command_supplied=owner_command_available,
    )

    write_text(
        env_path,
        _env_example(
            capture_root=context.capture_root,
            packet_dir=packet_dir,
            webapp_site_slug=webapp_site_slug,
            webapp_staged_inputs_path=selected_staged_inputs_path,
            webapp_forwarding_preflight_path=selected_forwarding_preflight_path,
            simulator=selected_simulator,
            provisioner=selected_provisioner,
            owner_command=selected_owner_command,
        ),
    )
    write_text(
        local_commands_path,
        _local_preflight_commands(
            capture_root=context.capture_root,
            webapp_site_slug=webapp_site_slug,
            webapp_staged_inputs_path=selected_staged_inputs_path,
            webapp_forwarding_preflight_path=selected_forwarding_preflight_path,
            simulator=selected_simulator,
            provisioner=selected_provisioner,
            owner_command=selected_owner_command,
            owner_command_location=selected_owner_command_location,
            allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
        ),
    )
    write_text(
        worldlabs_provider_submission_commands_path,
        _worldlabs_provider_submission_commands(capture_root=context.capture_root),
    )
    write_text(
        webapp_upstream_truth_verification_commands_path,
        _webapp_upstream_truth_verification_commands(
            capture_root=context.capture_root,
            result_path=webapp_upstream_truth_verification_result_path,
        ),
    )
    write_text(
        webapp_handoff_verification_commands_path,
        _webapp_handoff_verification_commands(
            capture_root=context.capture_root,
            webapp_site_slug=webapp_site_slug,
            webapp_staged_inputs_path=selected_staged_inputs_path,
            webapp_forwarding_preflight_path=selected_forwarding_preflight_path,
            result_path=webapp_handoff_verification_result_path,
            allow_local_webapp_rehearsal=allow_local_webapp_rehearsal,
        ),
    )
    write_text(
        gpu_commands_path,
        _gpu_vm_commands(
            capture_root=context.capture_root,
            packet_dir=packet_dir,
            simulator=selected_simulator,
            owner_command=selected_owner_command,
        ),
    )
    write_text(
        gpu_vm_runtime_preflight_script_path,
        _gpu_vm_runtime_preflight_commands(
            capture_root=context.capture_root,
            packet_dir=packet_dir,
            simulator=selected_simulator,
            owner_command=selected_owner_command,
        ),
    )
    write_text(
        owner_command_binding_template_path,
        _owner_command_binding_template(simulator=selected_simulator),
    )
    write_text(isaac_unitree_g1_smoke_script_path, _isaac_unitree_g1_smoke_script())
    write_text(isaac_unitree_g1_smoke_launcher_path, _isaac_unitree_g1_smoke_launcher())
    write_text(mujoco_unitree_g1_smoke_script_path, _mujoco_unitree_g1_smoke_script())
    write_text(mujoco_unitree_g1_smoke_launcher_path, _mujoco_unitree_g1_smoke_launcher())
    write_text(owner_contract_path, _owner_command_contract(simulator=selected_simulator))
    write_text(live_policy_execution_contract_path, _live_policy_execution_contract())
    write_json(
        default_test_job_request_template_path,
        _default_test_robot_eval_job_request_template(
            capture_root=context.capture_root,
            webapp_site_slug=webapp_site_slug,
            simulator=selected_simulator,
        ),
    )
    write_json(real_robot_pov_template_path, _real_robot_pov_manifest_template())
    write_text(
        live_input_staging_commands_path,
        _live_input_staging_commands(
            capture_root=context.capture_root,
            packet_dir=packet_dir,
            webapp_job_request_template_path=default_test_job_request_template_path,
            real_robot_pov_template_path=real_robot_pov_template_path,
            webapp_staged_inputs_path=selected_staged_inputs_path,
        ),
    )
    write_text(provider_bootstrap_path, _gpu_provider_bootstrap_markdown(provider_bootstrap))
    write_json(provider_bootstrap_manifest_path, provider_bootstrap)
    write_json(simulator_path_matrix_path, simulator_path_matrix)
    write_text(
        simulator_path_matrix_markdown_path,
        _simulator_path_matrix_markdown(simulator_path_matrix),
    )
    write_json(gpu_vm_runtime_preflight_plan_path, gpu_vm_runtime_preflight_plan)
    write_text(
        gpu_vm_runtime_preflight_markdown_path,
        _gpu_vm_runtime_preflight_plan_markdown(gpu_vm_runtime_preflight_plan),
    )
    write_json(blocker_resolution_path, blocker_resolution)
    write_text(blocker_resolution_markdown_path, _blocker_resolution_markdown(blocker_resolution))
    write_json(scene_asset_acquisition_path, scene_asset_acquisition)
    write_text(
        scene_asset_acquisition_markdown_path,
        _scene_asset_acquisition_markdown(scene_asset_acquisition),
    )
    write_json(webapp_handoff_path, webapp_handoff)
    write_text(webapp_handoff_markdown_path, _webapp_handoff_markdown(webapp_handoff))
    write_json(launch_order_path, launch_order)
    write_text(launch_order_markdown_path, _launch_order_markdown(launch_order))

    generated_files = {
        "readiness_manifest": str(readiness_path),
        "env_example": str(env_path),
        "local_preflight_commands": str(local_commands_path),
        "worldlabs_provider_submission_commands": str(
            worldlabs_provider_submission_commands_path
        ),
        "webapp_upstream_truth_verification_commands": str(
            webapp_upstream_truth_verification_commands_path
        ),
        "webapp_handoff_verification_commands": str(
            webapp_handoff_verification_commands_path
        ),
        "gpu_vm_commands": str(gpu_commands_path),
        "gpu_vm_runtime_preflight_script": str(gpu_vm_runtime_preflight_script_path),
        "gpu_vm_runtime_preflight_plan": str(gpu_vm_runtime_preflight_plan_path),
        "gpu_vm_runtime_preflight_markdown": str(gpu_vm_runtime_preflight_markdown_path),
        "owner_command_binding_template": str(owner_command_binding_template_path),
        "isaac_unitree_g1_smoke_script": str(isaac_unitree_g1_smoke_script_path),
        "isaac_unitree_g1_smoke_launcher": str(isaac_unitree_g1_smoke_launcher_path),
        "mujoco_unitree_g1_smoke_script": str(mujoco_unitree_g1_smoke_script_path),
        "mujoco_unitree_g1_smoke_launcher": str(mujoco_unitree_g1_smoke_launcher_path),
        "live_policy_execution_contract": str(live_policy_execution_contract_path),
        "default_test_robot_eval_job_request_template": str(
            default_test_job_request_template_path
        ),
        "real_robot_pov_manifest_template": str(real_robot_pov_template_path),
        "live_input_staging_commands": str(live_input_staging_commands_path),
        "simulator_path_matrix": str(simulator_path_matrix_path),
        "simulator_path_matrix_markdown": str(simulator_path_matrix_markdown_path),
        "launch_order": str(launch_order_path),
        "launch_order_markdown": str(launch_order_markdown_path),
        "owner_command_contract": str(owner_contract_path),
        "gpu_provider_bootstrap": str(provider_bootstrap_path),
        "gpu_provider_bootstrap_manifest": str(provider_bootstrap_manifest_path),
        "blocker_resolution": str(blocker_resolution_path),
        "blocker_resolution_markdown": str(blocker_resolution_markdown_path),
        "scene_asset_acquisition": str(scene_asset_acquisition_path),
        "scene_asset_acquisition_markdown": str(scene_asset_acquisition_markdown_path),
        "webapp_handoff": str(webapp_handoff_path),
        "webapp_handoff_markdown": str(webapp_handoff_markdown_path),
        "run_packet": str(packet_path),
        "gpu_vm_sync_manifest": str(vm_sync_manifest_path),
        "gpu_vm_sync_markdown": str(vm_sync_markdown_path),
    }
    packet = {
        "schema_version": FIRST_GPU_RUN_PACKET_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "capture_root": str(context.capture_root),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "webapp_site_slug": webapp_site_slug or None,
        "webapp_staged_inputs_path": str(selected_staged_inputs_path),
        "webapp_forwarding_preflight_path": (
            str(selected_forwarding_preflight_path)
            if selected_forwarding_preflight_path
            else None
        ),
        "simulator": selected_simulator,
        "provisioner": selected_provisioner,
        "owner_command_placeholder": selected_owner_command,
        "owner_command_supplied": owner_command_was_supplied,
        "owner_command_generated_by_packet": owner_command_generated_by_packet,
        "owner_command_available_for_selected_path": owner_command_available,
        "owner_command_location": selected_owner_command_location,
        "allow_local_webapp_rehearsal": allow_local_webapp_rehearsal,
        "readiness_status": readiness.get("status"),
        "ready_for_first_gpu_attempt": bool(readiness.get("ready_for_first_gpu_attempt")),
        "owner_gpu_proof_ready": bool(readiness.get("owner_gpu_proof_ready")),
        "blockers": readiness.get("blockers") or [],
        "warnings": readiness.get("warnings") or [],
        "generated_files": generated_files,
        "provider_guidance": {
            "recommended_provider_path": provider_bootstrap["recommended_provider_path"],
            "first_smoke_path": provider_bootstrap["first_smoke_path"],
            "gpu_guidance": provider_bootstrap["gpu_guidance"],
            "nvidia_nim_boundary": provider_bootstrap["nvidia_nim_boundary"],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(packet_path, packet)
    vm_sync_manifest = _gpu_vm_sync_manifest(
        capture_root=context.capture_root,
        packet_dir=packet_dir,
        generated_files=generated_files,
        readiness=readiness,
    )
    write_json(vm_sync_manifest_path, vm_sync_manifest)
    write_text(vm_sync_markdown_path, _gpu_vm_sync_markdown(vm_sync_manifest))
    gpu_vm_runtime_preflight_plan = _gpu_vm_runtime_preflight_plan_manifest(
        capture_root=context.capture_root,
        packet_dir=packet_dir,
        script_path=gpu_vm_runtime_preflight_script_path,
        result_path=gpu_vm_runtime_preflight_result_path,
        sync_manifest_path=vm_sync_manifest_path,
        readiness=readiness,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command=selected_owner_command,
        owner_command_location=selected_owner_command_location,
        owner_command_supplied=owner_command_available,
        vm_sync_manifest=vm_sync_manifest,
    )
    write_json(gpu_vm_runtime_preflight_plan_path, gpu_vm_runtime_preflight_plan)
    write_text(
        gpu_vm_runtime_preflight_markdown_path,
        _gpu_vm_runtime_preflight_plan_markdown(gpu_vm_runtime_preflight_plan),
    )
    launch_order = _first_gpu_launch_order_manifest(
        capture_root=context.capture_root,
        packet_dir=packet_dir,
        readiness=readiness,
        webapp_handoff=webapp_handoff,
        scene_asset_acquisition=scene_asset_acquisition,
        simulator_path_matrix=simulator_path_matrix,
        gpu_vm_runtime_preflight_plan=gpu_vm_runtime_preflight_plan,
        webapp_site_slug=webapp_site_slug,
        webapp_staged_inputs_path=selected_staged_inputs_path,
        simulator=selected_simulator,
        provisioner=selected_provisioner,
        owner_command_supplied=owner_command_available,
        vm_sync_manifest=vm_sync_manifest,
    )
    write_json(launch_order_path, launch_order)
    write_text(launch_order_markdown_path, _launch_order_markdown(launch_order))
    vm_sync_manifest = _gpu_vm_sync_manifest(
        capture_root=context.capture_root,
        packet_dir=packet_dir,
        generated_files=generated_files,
        readiness=readiness,
    )
    write_json(vm_sync_manifest_path, vm_sync_manifest)
    write_text(vm_sync_markdown_path, _gpu_vm_sync_markdown(vm_sync_manifest))
    return {
        "schema_version": "first_gpu_run_packet_result.v1",
        "packet_path": str(packet_path),
        "packet_dir": str(packet_dir),
        "readiness_manifest_path": str(readiness_path),
        "readiness_status": packet["readiness_status"],
        "ready_for_first_gpu_attempt": packet["ready_for_first_gpu_attempt"],
        "blockers": packet["blockers"],
        "generated_files": packet["generated_files"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local/GPU command packet for the first owner-GPU E2E run"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--webapp-site-slug", default="")
    parser.add_argument("--webapp-staged-inputs", default=None)
    parser.add_argument("--webapp-forwarding-preflight", default=None)
    parser.add_argument("--simulator", choices=SIMULATOR_FRAMEWORKS, default="isaac_sim")
    parser.add_argument("--provisioner", choices=PROVISIONERS, default="runpod")
    parser.add_argument("--owner-command", default=None)
    parser.add_argument(
        "--owner-command-location",
        choices=SIMULATOR_COMMAND_LOCATIONS,
        default="remote",
        help=(
            "Use remote for commands that exist inside the GPU VM; use local to require "
            "the executable on this machine."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-require-webapp-forwarding", action="store_true")
    parser.add_argument("--no-require-webapp-staged-request", action="store_true")
    parser.add_argument("--allow-local-webapp-rehearsal", action="store_true")
    parser.add_argument("--no-require-gpu-gates", action="store_true")
    args = parser.parse_args(argv)

    result = build_first_gpu_run_packet(
        capture_root=args.capture_root,
        webapp_site_slug=args.webapp_site_slug,
        webapp_staged_inputs_path=args.webapp_staged_inputs,
        webapp_forwarding_preflight_path=args.webapp_forwarding_preflight,
        simulator=args.simulator,
        provisioner=args.provisioner,
        owner_command=args.owner_command,
        owner_command_location=args.owner_command_location,
        output_dir=args.output_dir,
        require_webapp_forwarding=not args.no_require_webapp_forwarding,
        require_webapp_staged_request=not args.no_require_webapp_staged_request,
        allow_local_webapp_rehearsal=args.allow_local_webapp_rehearsal,
        require_gpu_gates=not args.no_require_gpu_gates,
    )
    print(f"[first-gpu-run-packet] readiness_status={result['readiness_status']}")
    print(f"[first-gpu-run-packet] packet={result['packet_path']}")
    if result["blockers"]:
        print("[first-gpu-run-packet] blockers=" + ",".join(result["blockers"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
