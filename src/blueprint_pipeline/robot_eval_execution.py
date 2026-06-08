"""Robot-eval execution, robot-POV, and deployment calibration helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, resolve_gs_uri_to_path, write_json


ROBOT_POV_OBSERVATION_SCHEMA_VERSION = "robot_pov_observation_manifest.v1"
POLICY_EXECUTION_MANIFEST_SCHEMA_VERSION = "robot_policy_execution_manifest.v1"
POLICY_EXECUTION_TRACE_SCHEMA_VERSION = "robot_policy_execution_trace.v1"
DEPLOYMENT_OUTCOME_LEDGER_SCHEMA_VERSION = "deployment_outcome_ledger.v1"
SIM_VS_REAL_CALIBRATION_SCHEMA_VERSION = "sim_vs_real_calibration_report.v1"
PREDICTION_VS_ACTUAL_DEPLOYMENT_SCHEMA_VERSION = "prediction_vs_actual_deployment_summary.v1"
SIMULATOR_COMMAND_ARTIFACTS_SCHEMA_VERSION = "simulator_command_artifacts.v1"

POLICY_MODALITIES = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "robot_eval_execution_and_calibration_support",
    "repo_local_default": True,
    "external_policy_calls_allowed_only_with_runtime_gates": True,
    "robot_pov_generated_from_capture_task_scenario_context": True,
    "generated_robot_pov_is_support_artifact_not_raw_capture_evidence": True,
    "robot_policy_execution_proven": False,
    "simulator_execution_proven": False,
    "real_world_outcome_proven": False,
    "robot_readiness_proven": False,
    "safety_validated": False,
    "public_claim_upgrade_allowed": False,
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "passed", "success", "succeeded"}


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _safe_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return cleaned.strip("-_") or "item"


def _relative_to(base: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    try:
        payload = read_json_any(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return _mapping(payload)


def _read_optional_any(path: Path) -> Any:
    try:
        return read_json_any(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _local_reference_path(
    value: Any,
    *,
    capture_root: Path,
    job_dir: Path,
) -> Path | None:
    text = _string(value)
    if not text:
        return None
    if text.startswith("file://"):
        return Path(text[7:]).expanduser()
    if text.startswith("gs://"):
        default_gcs_root = capture_root.parents[3] if len(capture_root.parents) > 3 else capture_root
        return resolve_gs_uri_to_path(text, Path(os.getenv("GCS_ROOT", str(default_gcs_root))))
    if "://" in text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    job_candidate = job_dir / path
    if job_candidate.exists():
        return job_candidate
    return capture_root / path


def _load_reference_json(
    value: Any,
    *,
    capture_root: Path,
    job_dir: Path,
) -> Any:
    path = _local_reference_path(value, capture_root=capture_root, job_dir=job_dir)
    if path is None or not path.is_file():
        return None
    return _read_optional_any(path)


def _cards_by_id(cards_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in cards_payload.get("cards", []) or []:
        if isinstance(item, Mapping):
            scenario_id = _string(item.get("scenario_id") or item.get("task_id"))
            if scenario_id:
                out[scenario_id] = dict(item)
    return out


def _requested_scenarios(
    request: Mapping[str, Any],
    scenario_cards: Mapping[str, Any],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for task in request.get("requested_tasks") or request.get("requestedTasks") or []:
        if not isinstance(task, Mapping):
            continue
        task_id = _string(task.get("task_id") or task.get("taskId"))
        scenario_ids = _string_list(task.get("scenario_ids") or task.get("scenarioIds"))
        if not scenario_ids:
            scenario_ids = [
                _string(card.get("scenario_id"))
                for card in scenario_cards.get("cards", []) or []
                if isinstance(card, Mapping) and _string(card.get("task_id")) == task_id
            ]
        for scenario_id in scenario_ids:
            rows.append({"task_id": task_id, "scenario_id": scenario_id})
    if rows:
        return rows
    for item in scenario_cards.get("cards", []) or []:
        if isinstance(item, Mapping):
            rows.append(
                {
                    "task_id": _string(item.get("task_id")),
                    "scenario_id": _string(item.get("scenario_id")),
                }
            )
    return [row for row in rows if row["scenario_id"]]


def _attempt_video_index(attempt_trace: Mapping[str, Any]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        scenario_id = _string(attempt.get("scenario_id"))
        video_path = _string(
            attempt.get("video_path")
            or _mapping(attempt.get("artifact_paths")).get("robot_pov_video")
            or _mapping(attempt.get("artifact_paths")).get("video")
        )
        if scenario_id and video_path and scenario_id not in index:
            index[scenario_id] = video_path
    return index


def _write_observation_png(path: Path, lines: Sequence[str]) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    ensure_dir(path.parent)
    image = Image.new("RGB", (960, 540), (24, 29, 35))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 936, 516), outline=(90, 160, 220), width=3)
    draw.rectangle((64, 330, 896, 460), outline=(175, 190, 205), width=2)
    draw.line((480, 120, 480, 465), fill=(210, 210, 130), width=2)
    draw.text((64, 56), "Blueprint robot POV observation", fill=(245, 248, 250))
    y = 104
    for line in lines[:10]:
        draw.text((64, y), line[:110], fill=(220, 230, 238))
        y += 34
    image.save(path)
    return True


def build_robot_pov_observation_bundle(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    generated_at: str,
    attempt_trace: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build deterministic robot-POV observation packets for every requested scenario."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    robot_eval_dir = capture_path / "pipeline" / "robot_eval_dataset"
    scenario_cards = _read_optional_mapping(robot_eval_dir / "scenario_cards.json")
    task_cards = _read_optional_mapping(robot_eval_dir / "task_cards.json")
    episode_spec = _read_optional_mapping(
        capture_path / "pipeline" / "simulation_automation" / "episode_spec.v1.json"
    )
    scenarios_by_id = _cards_by_id(scenario_cards)
    tasks_by_id = _cards_by_id(task_cards)
    requested = _requested_scenarios(job_request, scenario_cards)
    robot_profile = _mapping(job_request.get("robot_profile") or job_request.get("robotProfile"))
    robot_profile_id = _string(robot_profile.get("robot_profile_id") or robot_profile.get("id"))
    video_index = _attempt_video_index(attempt_trace or {})

    observations: List[Dict[str, Any]] = []
    frame_dir = resolved_job_dir / "robot_pov"
    for index, row in enumerate(requested, start=1):
        scenario_id = row["scenario_id"]
        task_id = row["task_id"]
        scenario = scenarios_by_id.get(scenario_id, {})
        task = tasks_by_id.get(task_id, {})
        observation_id = f"robot_pov_{_safe_id(task_id)}_{_safe_id(scenario_id)}"
        frame_path = frame_dir / f"{observation_id}.png"
        lines = [
            f"task_id: {task_id or 'unknown'}",
            f"scenario_id: {scenario_id}",
            f"robot_profile_id: {robot_profile_id or 'unknown'}",
            f"task: {_string(task.get('task_statement')) or 'from task card'}",
            f"normal: {_string(_mapping(scenario.get('normal_scenario')).get('statement'))}",
            f"variation: {_string(_mapping(scenario.get('variation')).get('statement'))}",
            f"edge_case: {_string(_mapping(scenario.get('edge_case')).get('statement'))}",
        ]
        frame_written = _write_observation_png(frame_path, lines)
        observations.append(
            {
                "observation_id": observation_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "robot_profile_id": robot_profile_id or None,
                "sequence_index": index,
                "camera": {
                    "name": "front_rgbd",
                    "frame": "site_coordinate_frame",
                    "resolution": {"width": 960, "height": 540},
                    "horizontal_fov_degrees": 75.0,
                    "mount": "robot_front",
                    "extrinsics_source": "episode_spec_or_deterministic_default",
                },
                "observation_schema": {
                    "rgb": "image/png",
                    "depth": "optional_depth_map_or_missing",
                    "robot_state": ["base_pose", "joint_state_optional", "gripper_state_optional"],
                    "task_instruction": "task_card.task_statement",
                },
                "generated_frame_path": _relative_to(resolved_job_dir, frame_path)
                if frame_written
                else None,
                "sim_or_real_video_path": video_index.get(scenario_id),
                "source_artifacts": {
                    "scenario_card": "pipeline/robot_eval_dataset/scenario_cards.json",
                    "task_card": "pipeline/robot_eval_dataset/task_cards.json",
                    "episode_spec": "pipeline/simulation_automation/episode_spec.v1.json"
                    if episode_spec
                    else None,
                },
                "claim_boundary": (
                    "generated_robot_pov_observation_packet_not_raw_robot_camera_evidence"
                ),
            }
        )

    manifest = {
        "schema_version": ROBOT_POV_OBSERVATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if observations else "blocked_missing_scenarios",
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "observation_count": len(observations),
        "robot_profile": robot_profile,
        "observations": observations,
        "robot_pov_generated": bool(observations),
        "sim_or_real_robot_pov_video_available": any(
            _string(item.get("sim_or_real_video_path")) for item in observations
        ),
        "robot_pov_evidence_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_job_dir / "robot_pov_observation_manifest.json", manifest)
    _write_jsonl(resolved_job_dir / "robot_pov_observations.jsonl", observations)
    return manifest


def _modality_payload(policy_package: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    camel = {
        "policy_api_endpoint": "policyApiEndpoint",
        "docker_container": "dockerContainer",
        "recorded_action_trace": "recordedActionTrace",
        "high_level_skill_trace": "highLevelSkillTrace",
        "teleop_demo": "teleopDemo",
        "sim_controller_plugin": "simControllerPlugin",
    }[modality]
    return _mapping(policy_package.get(modality) or policy_package.get(camel))


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("token", "secret", "password", "auth")):
                out[key_text] = "<redacted>"
            else:
                out[key_text] = _redact(child)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _command_from_payload(
    *,
    modality: str,
    payload: Mapping[str, Any],
    commands: Mapping[str, str],
) -> str:
    return _string(
        commands.get(modality)
        or payload.get("execution_command")
        or payload.get("executionCommand")
        or payload.get("adapter_command")
        or payload.get("adapterCommand")
    )


def _normalize_policy_attempts(
    *,
    payload: Any,
    modality: str,
    observations: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> List[Dict[str, Any]]:
    raw_attempts: List[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        for key in ("attempts", "actions", "skills", "episodes", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_attempts.extend(item for item in value if isinstance(item, Mapping))
        if not raw_attempts and payload:
            raw_attempts = [payload]
    elif isinstance(payload, list):
        raw_attempts = [item for item in payload if isinstance(item, Mapping)]

    if not raw_attempts and modality == "high_level_skill_trace":
        raw_attempts = [{"status": "completed", "actions": []}]

    attempts: List[Dict[str, Any]] = []
    if not observations:
        observations = [{"observation_id": "observation_1", "scenario_id": "", "task_id": ""}]
    for index, raw in enumerate(raw_attempts or [{}], start=1):
        observation = observations[(index - 1) % len(observations)]
        status = _string(raw.get("status") or raw.get("result") or "completed").lower()
        success_raw = raw.get("success")
        success = _boolish(success_raw) if success_raw is not None else status in {
            "completed",
            "success",
            "succeeded",
            "passed",
        }
        attempts.append(
            {
                "attempt_id": _string(raw.get("attempt_id") or raw.get("attemptId"))
                or f"{modality}_attempt_{index:04d}",
                "modality": modality,
                "observation_id": _string(raw.get("observation_id"))
                or _string(observation.get("observation_id")),
                "scenario_id": _string(raw.get("scenario_id") or raw.get("scenarioId"))
                or _string(observation.get("scenario_id")),
                "task_id": _string(raw.get("task_id") or raw.get("taskId"))
                or _string(observation.get("task_id")),
                "policy_id": _string(raw.get("policy_id") or raw.get("policyId") or modality),
                "status": status,
                "success": bool(success),
                "actions": raw.get("actions") if isinstance(raw.get("actions"), list) else [],
                "skills": raw.get("skills") if isinstance(raw.get("skills"), list) else [],
                "metrics": _mapping(raw.get("metrics")),
                "artifact_paths": _mapping(raw.get("artifact_paths") or raw.get("artifactPaths")),
                "generated_at": generated_at,
                "claim_boundary": "policy_submission_trace_not_robot_readiness_proof",
            }
        )
    return attempts


def _run_command(
    *,
    command_text: str,
    output_path: Path,
    observation_manifest_path: Path,
    modality: str,
    timeout_seconds: int,
) -> tuple[str, Any, Dict[str, Any]]:
    command = shlex.split(command_text)
    ensure_dir(output_path.parent)
    env = {
        **os.environ,
        "BLUEPRINT_POLICY_OBSERVATION_MANIFEST": str(observation_manifest_path),
        "BLUEPRINT_POLICY_EXECUTION_OUTPUT": str(output_path),
        "BLUEPRINT_POLICY_MODALITY": modality,
    }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        return "blocked", None, {"blockers": ["missing_policy_command_dependency"]}
    except subprocess.TimeoutExpired as exc:
        return "failed", None, {"blockers": ["policy_command_timeout"], "stdout": exc.stdout or ""}
    payload = _read_optional_any(output_path) if output_path.is_file() else None
    if payload is None and completed.stdout.strip().startswith(("{", "[")):
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    status = "completed" if completed.returncode == 0 and payload is not None else "failed"
    detail = {
        "command": command,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "exit_code": completed.returncode,
        "blockers": [] if status == "completed" else [f"policy_command_exit:{completed.returncode}"],
    }
    return status, payload, detail


def _call_policy_api(
    *,
    endpoint: str,
    observation_manifest: Mapping[str, Any],
    timeout_seconds: int,
) -> tuple[str, Any, Dict[str, Any]]:
    data = json.dumps({"observation_manifest": observation_manifest}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return "completed", payload, {"http_status": response.status, "blockers": []}
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return "failed", None, {"blockers": ["policy_api_call_failed"], "error": str(exc)}


def _docker_command(payload: Mapping[str, Any]) -> str:
    image = _string(payload.get("image_ref") or payload.get("imageRef"))
    entrypoint = _string(payload.get("entrypoint"))
    if not image:
        return ""
    base = f"docker run --rm -i {shlex.quote(image)}"
    return f"{base} {entrypoint}" if entrypoint else base


def _replay_reference_payload(
    *,
    modality: str,
    payload: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> Any:
    keys = {
        "recorded_action_trace": ("trace_manifest_uri", "traceManifestUri"),
        "teleop_demo": ("demo_artifact_uri", "demoArtifactUri"),
        "sim_controller_plugin": ("plugin_uri", "pluginUri"),
        "policy_api_endpoint": ("response_manifest_uri", "responseManifestUri", "local_response_path"),
        "docker_container": ("output_manifest_uri", "outputManifestUri", "local_output_path"),
    }.get(modality, ())
    for key in keys:
        loaded = _load_reference_json(payload.get(key), capture_root=capture_root, job_dir=job_dir)
        if loaded is not None:
            return loaded
    if modality == "high_level_skill_trace":
        sequence = payload.get("ordered_skill_sequence") or payload.get("orderedSkillSequence") or []
        return {"attempts": [{"status": "completed", "skills": list(sequence), "success": True}]}
    return None


def build_policy_execution_bundle(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    observation_manifest: Mapping[str, Any],
    allow_policy_execution: bool = False,
    allow_reference_replay: bool = True,
    policy_execution_commands: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    generated_at: str,
) -> Dict[str, Any]:
    """Execute or replay robot-team policy submissions into normalized traces."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    policy_package = _mapping(job_request.get("policy_package") or job_request.get("policyPackage"))
    commands = dict(policy_execution_commands or {})
    env_allows = _boolish(os.getenv("BLUEPRINT_ALLOW_POLICY_EXECUTION"))
    observation_manifest_path = resolved_job_dir / "robot_pov_observation_manifest.json"
    observations = [
        dict(item)
        for item in observation_manifest.get("observations", []) or []
        if isinstance(item, Mapping)
    ]
    modality_results: Dict[str, Dict[str, Any]] = {}
    all_attempts: List[Dict[str, Any]] = []

    for modality in POLICY_MODALITIES:
        payload = _modality_payload(policy_package, modality)
        if not payload:
            modality_results[modality] = {
                "status": "not_selected",
                "execution_performed": False,
                "attempt_count": 0,
                "missing_inputs": [],
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
            continue
        command_text = _command_from_payload(
            modality=modality,
            payload=payload,
            commands=commands,
        )
        if not command_text and modality == "docker_container" and allow_policy_execution and env_allows:
            command_text = _docker_command(payload)

        payload_result: Any = None
        detail: Dict[str, Any] = {}
        execution_performed = False
        if command_text:
            if not allow_policy_execution or not env_allows:
                modality_results[modality] = {
                    "status": "blocked_policy_execution_gate",
                    "execution_performed": False,
                    "attempt_count": 0,
                    "reference": _redact(payload),
                    "blockers": [
                        "Set BLUEPRINT_ALLOW_POLICY_EXECUTION=true and pass allow_policy_execution.",
                    ],
                    "claim_boundary": dict(CLAIM_BOUNDARY),
                }
                continue
            output_path = resolved_job_dir / "policy_execution" / f"{modality}_output.json"
            status, payload_result, detail = _run_command(
                command_text=command_text,
                output_path=output_path,
                observation_manifest_path=observation_manifest_path,
                modality=modality,
                timeout_seconds=timeout_seconds,
            )
            execution_performed = True
        elif modality == "policy_api_endpoint" and allow_policy_execution and env_allows:
            endpoint = _string(payload.get("endpoint_url") or payload.get("endpointUrl") or payload.get("url"))
            status, payload_result, detail = _call_policy_api(
                endpoint=endpoint,
                observation_manifest=observation_manifest,
                timeout_seconds=timeout_seconds,
            )
            execution_performed = True
        elif not allow_reference_replay:
            status = "blocked_reference_replay_gate"
            payload_result = None
            detail = {"blockers": ["reference_replay_disabled_by_validation_or_rights_gate"]}
        else:
            payload_result = _replay_reference_payload(
                modality=modality,
                payload=payload,
                capture_root=capture_path,
                job_dir=resolved_job_dir,
            )
            status = "completed_reference_replay" if payload_result is not None else "reference_ready"
            detail = {"blockers": [] if payload_result is not None else ["local_reference_not_available"]}

        attempts = (
            _normalize_policy_attempts(
                payload=payload_result,
                modality=modality,
                observations=observations,
                generated_at=generated_at,
            )
            if payload_result is not None
            else []
        )
        all_attempts.extend(attempts)
        modality_results[modality] = {
            "status": status,
            "execution_performed": execution_performed,
            "reference_replayed": payload_result is not None and not execution_performed,
            "attempt_count": len(attempts),
            "reference": _redact(payload),
            "detail": detail,
            "robot_policy_execution_proven": execution_performed and status == "completed",
            "policy_submission_trace_available": bool(attempts),
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "robot_policy_execution_proven": execution_performed and status == "completed",
            },
        }

    execution_proven = any(
        bool(item.get("robot_policy_execution_proven")) for item in modality_results.values()
    )
    trace = {
        "schema_version": POLICY_EXECUTION_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if all_attempts else "blocked_missing_policy_execution_trace",
        "attempt_count": len(all_attempts),
        "attempts": all_attempts,
        "robot_policy_execution_proven": execution_proven,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "robot_policy_execution_proven": execution_proven,
        },
    }
    manifest = {
        "schema_version": POLICY_EXECUTION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if all_attempts else "blocked",
        "selected_modalities": [
            modality
            for modality, result in modality_results.items()
            if result.get("status") != "not_selected"
        ],
        "env_BLUEPRINT_ALLOW_POLICY_EXECUTION": env_allows,
        "allow_policy_execution_flag": bool(allow_policy_execution),
        "modality_results": modality_results,
        "attempt_count": len(all_attempts),
        "policy_execution_trace_path": "policy_execution_trace.json",
        "robot_policy_execution_proven": execution_proven,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "robot_policy_execution_proven": execution_proven,
        },
    }
    write_json(resolved_job_dir / "policy_execution_manifest.json", manifest)
    write_json(resolved_job_dir / "policy_execution_trace.json", trace)
    _write_jsonl(resolved_job_dir / "policy_execution_trace.jsonl", all_attempts)
    return {"manifest": manifest, "trace": trace}


def _records_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("records", "actual_outcomes", "actualOutcomes", "outcomes", "pilot_runs", "runs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if payload:
            return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _simulator_attempts_from_payload(
    *,
    payload: Any,
    simulator: str,
    generated_at: str,
) -> List[Dict[str, Any]]:
    records = _records_from_payload(payload)
    attempts: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        status = _string(record.get("status") or record.get("result") or "completed").lower()
        success = (
            _boolish(record.get("success"))
            if record.get("success") is not None
            else status in {"completed", "success", "succeeded", "passed"}
        )
        failure_ids = _failure_ids(record, "failure_mode_ids", "failure_modes", "failures")
        if not success and not failure_ids:
            failure_ids = [_string(record.get("failure_reason")) or "simulator_failure"]
        attempts.append(
            {
                "attempt_id": _string(record.get("attempt_id") or record.get("attemptId"))
                or f"{simulator}_attempt_{index:04d}",
                "episode_id": _string(record.get("episode_id") or record.get("episodeId"))
                or f"{simulator}_episode_{index:04d}",
                "scenario_id": _string(record.get("scenario_id") or record.get("scenarioId")),
                "scenario_run_id": _string(
                    record.get("scenario_run_id") or record.get("scenarioRunId")
                )
                or f"{simulator}_scenario_run_{index:04d}",
                "task_id": _string(record.get("task_id") or record.get("taskId")),
                "policy_id": _string(record.get("policy_id") or record.get("policyId")),
                "engine": simulator,
                "runner": "command_adapter",
                "status": status,
                "success": bool(success),
                "failure_reason": _string(record.get("failure_reason") or record.get("reason"))
                or None,
                "failure_mode_ids": failure_ids,
                "metrics": _mapping(record.get("metrics")),
                "action_trace": record.get("actions") if isinstance(record.get("actions"), list) else [],
                "contact_trace": record.get("contact_trace")
                if isinstance(record.get("contact_trace"), list)
                else [],
                "safety_events": record.get("safety_events")
                if isinstance(record.get("safety_events"), list)
                else [],
                "video_path": _string(record.get("video_path") or record.get("videoPath")) or None,
                "artifact_paths": _mapping(record.get("artifact_paths") or record.get("artifactPaths")),
                "generated_at": generated_at,
                "claim_boundary": "simulator_command_output_not_real_robot_deployment_proof",
            }
        )
    return attempts


def build_simulator_command_artifacts(
    *,
    job_dir: str | Path,
    simulator: str,
    simulator_output: Any,
    generated_at: str,
) -> Dict[str, Any]:
    """Normalize simulator command output into evaluator/package artifacts."""

    resolved_job_dir = Path(job_dir).resolve()
    attempts = _simulator_attempts_from_payload(
        payload=simulator_output,
        simulator=simulator,
        generated_at=generated_at,
    )
    failures = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    trace = {
        "schema_version": "robot_eval_simulator_command_normalized_attempt_trace.v1",
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked_missing_simulator_attempts",
        "backend": simulator,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "result_ingested": bool(attempts),
        "simulator_execution_proven": bool(attempts),
        "robot_policy_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    labels = {
        "schema_version": "robot_eval_simulator_command_failure_labels.v1",
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_failures_labeled",
        "label_count": len(failures),
        "labels": [
            {
                "label_id": f"label_{_safe_id(_string(attempt.get('attempt_id')))}",
                "attempt_id": attempt.get("attempt_id"),
                "scenario_id": attempt.get("scenario_id"),
                "failure_mode_ids": attempt.get("failure_mode_ids") or [],
                "failure_reason": attempt.get("failure_reason"),
                "status": "review_required",
                "proof_effect": "none_until_review_accepted_or_owner_proof_supplied",
            }
            for attempt in failures
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    prediction_records = [
        {
            "scenario_id": attempt.get("scenario_id"),
            "task_id": attempt.get("task_id"),
            "policy_id": attempt.get("policy_id"),
            "predicted_status": "passed" if attempt.get("success") else "failed",
            "predicted_success": bool(attempt.get("success")),
            "predicted_cycle_time_seconds": _number(
                _mapping(attempt.get("metrics")).get("cycle_time_seconds")
            ),
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "source": f"{simulator}_command_output",
            "actual_status": "needs_actual_outcome",
        }
        for attempt in attempts
    ]
    prediction_ledger = {
        "schema_version": "robot_eval_simulator_prediction_outcome_ledger.v1",
        "generated_at": generated_at,
        "status": "completed" if attempts else "not_available",
        "record_count": len(prediction_records),
        "records": prediction_records,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    calibration_report = {
        "schema_version": "robot_eval_simulator_calibration_report.v1",
        "generated_at": generated_at,
        "status": "needs_real_world_outcomes" if attempts else "not_available",
        "record_count": len(prediction_records),
        "records": prediction_records,
        "sim_vs_real_calibration_score": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    breakage_library = {
        "schema_version": "robot_eval_simulator_breakage_library.v1",
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "record_count": len(failures),
        "records": [
            {
                "scenario_id": attempt.get("scenario_id"),
                "task_id": attempt.get("task_id"),
                "failure_mode_ids": attempt.get("failure_mode_ids") or [],
                "failure_reason": attempt.get("failure_reason"),
                "review_required": True,
            }
            for attempt in failures
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest = {
        "schema_version": SIMULATOR_COMMAND_ARTIFACTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked_missing_simulator_attempts",
        "simulator": simulator,
        "attempt_count": len(attempts),
        "artifact_paths": {
            "normalized_attempt_trace": "normalized_attempt_trace.json",
            "failure_labels": "failure_labels.json",
            "prediction_outcome_ledger": "prediction_outcome_ledger.json",
            "calibration_report": "calibration_report.json",
            "breakage_library": "breakage_library.json",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_job_dir / "normalized_attempt_trace.json", trace)
    write_json(resolved_job_dir / "failure_labels.json", labels)
    write_json(resolved_job_dir / "prediction_outcome_ledger.json", prediction_ledger)
    write_json(resolved_job_dir / "calibration_report.json", calibration_report)
    write_json(resolved_job_dir / "breakage_library.json", breakage_library)
    write_json(resolved_job_dir / "simulator_command_artifacts_manifest.json", manifest)
    return {
        "manifest": manifest,
        "normalized_attempt_trace": trace,
        "failure_labels": labels,
        "prediction_outcome_ledger": prediction_ledger,
        "calibration_report": calibration_report,
        "breakage_library": breakage_library,
    }


def _load_actual_outcome_payload(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
) -> Any:
    explicit_refs = [
        job_request.get("actual_outcome_manifest_uri"),
        job_request.get("actualOutcomeManifestUri"),
        job_request.get("deployment_outcome_manifest_uri"),
        job_request.get("deploymentOutcomeManifestUri"),
    ]
    for ref in explicit_refs:
        loaded = _load_reference_json(ref, capture_root=capture_root, job_dir=job_dir)
        if loaded is not None:
            return loaded
    for path in (
        capture_root / "pipeline" / "robot_eval_inputs" / "deployment_outcome_manifest.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "actual_outcome_manifest.json",
    ):
        loaded = _read_optional_any(path)
        if loaded is not None:
            return loaded
    return None


def _prediction_index(
    prediction_ledger: Mapping[str, Any],
    attempt_trace: Mapping[str, Any],
) -> Dict[tuple[str, str], Dict[str, Any]]:
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in prediction_ledger.get("records", []) or []:
        if not isinstance(record, Mapping):
            continue
        key = (_string(record.get("task_id")), _string(record.get("scenario_id")))
        if key != ("", "") and key not in index:
            index[key] = dict(record)
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        key = (_string(attempt.get("task_id")), _string(attempt.get("scenario_id")))
        if key == ("", "") or key in index:
            continue
        index[key] = {
            "task_id": key[0],
            "scenario_id": key[1],
            "predicted_success": attempt.get("predicted_success"),
            "predicted_cycle_time_seconds": attempt.get("predicted_cycle_time_seconds"),
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "source": "normalized_attempt_trace",
        }
    return index


def _predicted_success(record: Mapping[str, Any]) -> bool | None:
    if "predicted_success" in record:
        value = record.get("predicted_success")
        return _boolish(value) if value is not None else None
    status = _string(record.get("predicted_status") or record.get("prediction_status")).lower()
    if status in {"pass", "passed", "success", "succeeded", "completed"}:
        return True
    if status in {"fail", "failed", "failure", "predicted_failure"}:
        return False
    failures = _string_list(record.get("failure_mode_ids"))
    if failures:
        return False
    return None


def _actual_success(record: Mapping[str, Any]) -> bool | None:
    for key in ("actual_success", "actualSuccess", "success", "passed"):
        if key in record and record.get(key) is not None:
            return _boolish(record.get(key))
    status = _string(record.get("actual_status") or record.get("status")).lower()
    if status in {"pass", "passed", "success", "succeeded", "completed"}:
        return True
    if status in {"fail", "failed", "failure", "timeout", "collision"}:
        return False
    return None


def _failure_ids(record: Mapping[str, Any], *keys: str) -> List[str]:
    for key in keys:
        values = _string_list(record.get(key))
        if values:
            return values
    return []


def _calibration_score(rows: Sequence[Mapping[str, Any]]) -> float | None:
    scored = [
        row
        for row in rows
        if row.get("predicted_success") is not None and row.get("actual_success") is not None
    ]
    if not scored:
        return None
    matches = sum(
        1
        for row in scored
        if bool(row.get("predicted_success")) == bool(row.get("actual_success"))
    )
    return round(matches / len(scored), 6)


def build_deployment_validation_bundle(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    attempt_trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    """Ingest real deployment outcomes and compute sim-vs-real calibration."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    payload = _load_actual_outcome_payload(
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
    )
    actual_records = _records_from_payload(payload)
    predictions = _prediction_index(prediction_ledger, attempt_trace)
    rows: List[Dict[str, Any]] = []
    for index, actual in enumerate(actual_records, start=1):
        task_id = _string(actual.get("task_id") or actual.get("taskId"))
        scenario_id = _string(actual.get("scenario_id") or actual.get("scenarioId"))
        prediction = predictions.get((task_id, scenario_id), {})
        predicted_failures = _failure_ids(prediction, "failure_mode_ids", "predicted_failures")
        actual_failures = _failure_ids(actual, "failure_mode_ids", "actual_failures", "failures")
        predicted_success = _predicted_success(prediction)
        actual_success = _actual_success(actual)
        site_modifications = actual.get("site_modifications") or actual.get("siteModifications") or []
        row = {
            "record_id": _string(actual.get("outcome_id") or actual.get("record_id"))
            or f"deployment_outcome_{index:04d}",
            "task_id": task_id,
            "scenario_id": scenario_id,
            "policy_id": _string(actual.get("policy_id") or actual.get("policyId")),
            "prediction_source": prediction.get("source"),
            "predicted_success": predicted_success,
            "actual_success": actual_success,
            "predicted_failures": predicted_failures,
            "actual_failures": actual_failures,
            "missed_failures": sorted(set(actual_failures) - set(predicted_failures)),
            "false_alarm_failures": sorted(set(predicted_failures) - set(actual_failures)),
            "predicted_cycle_time_seconds": _number(
                prediction.get("predicted_cycle_time_seconds")
                or prediction.get("cycle_time_seconds")
            ),
            "actual_cycle_time_seconds": _number(
                actual.get("cycle_time_seconds") or actual.get("actualCycleTimeSeconds")
            ),
            "intervention_count": _number(
                actual.get("intervention_count") or actual.get("interventions"),
                0.0,
            ),
            "real_world_tuning_needed": bool(
                actual.get("real_world_tuning_needed")
                or actual.get("realWorldTuningNeeded")
                or actual.get("tuning_notes")
            ),
            "tuning_iterations": int(_number(actual.get("tuning_iterations"), 0.0) or 0),
            "tuning_hours": _number(actual.get("tuning_hours") or actual.get("tuningHours"), 0.0),
            "tuning_notes": _string_list(actual.get("tuning_notes") or actual.get("tuningNotes")),
            "site_modifications": site_modifications if isinstance(site_modifications, list) else [],
            "site_modifications_helped": actual.get("site_modifications_helped")
            if actual.get("site_modifications_helped") is not None
            else actual.get("siteModificationsHelped"),
            "evidence_refs": _mapping(actual.get("evidence_refs") or actual.get("evidenceRefs")),
            "matched_prediction": bool(prediction),
            "claim_boundary": "real_world_outcome_requires_owner_system_evidence_review",
        }
        rows.append(row)

    score = _calibration_score(rows)
    status = "completed" if rows else "blocked_missing_real_world_outcomes"
    ledger = {
        "schema_version": DEPLOYMENT_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "record_count": len(rows),
        "records": rows,
        "real_world_outcome_proven": bool(rows),
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "real_world_outcome_proven": bool(rows),
        },
    }
    report = {
        "schema_version": SIM_VS_REAL_CALIBRATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if score is not None else status,
        "paired_record_count": len(
            [
                row
                for row in rows
                if row.get("predicted_success") is not None
                and row.get("actual_success") is not None
            ]
        ),
        "sim_vs_real_calibration_score": score,
        "missed_failure_count": sum(len(_string_list(row.get("missed_failures"))) for row in rows),
        "false_alarm_failure_count": sum(
            len(_string_list(row.get("false_alarm_failures"))) for row in rows
        ),
        "site_modification_count": sum(len(row.get("site_modifications") or []) for row in rows),
        "tuning_hours_total": round(sum(_number(row.get("tuning_hours"), 0.0) or 0.0 for row in rows), 4),
        "records": rows,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "real_world_outcome_proven": bool(rows),
        },
    }
    summary = {
        "schema_version": PREDICTION_VS_ACTUAL_DEPLOYMENT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if rows else "blocked_missing_real_world_outcomes",
        "what_eval_predicted": [
            {
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "predicted_success": row.get("predicted_success"),
                "predicted_failures": row.get("predicted_failures"),
            }
            for row in rows
        ],
        "what_actually_happened": [
            {
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "actual_success": row.get("actual_success"),
                "actual_failures": row.get("actual_failures"),
            }
            for row in rows
        ],
        "which_scenarios_predicted_failure": [
            row.get("scenario_id") for row in rows if row.get("predicted_success") is False
        ],
        "which_failures_were_missed": [
            {
                "scenario_id": row.get("scenario_id"),
                "missed_failures": row.get("missed_failures"),
            }
            for row in rows
            if row.get("missed_failures")
        ],
        "how_much_real_world_tuning_was_needed": {
            "tuning_hours_total": report["tuning_hours_total"],
            "tuning_iterations_total": sum(int(row.get("tuning_iterations") or 0) for row in rows),
            "records_with_tuning": sum(1 for row in rows if row.get("real_world_tuning_needed")),
        },
        "whether_site_modifications_helped": [
            {
                "scenario_id": row.get("scenario_id"),
                "site_modifications": row.get("site_modifications"),
                "site_modifications_helped": row.get("site_modifications_helped"),
            }
            for row in rows
            if row.get("site_modifications")
        ],
        "sim_vs_real_calibration_score": score,
        "claim_boundary": report["claim_boundary"],
    }
    write_json(resolved_job_dir / "deployment_outcome_ledger.json", ledger)
    write_json(resolved_job_dir / "sim_vs_real_calibration_report.json", report)
    write_json(resolved_job_dir / "prediction_vs_actual_deployment_summary.json", summary)
    return {"ledger": ledger, "calibration_report": report, "summary": summary}


def fingerprint_execution_artifacts(*payloads: Mapping[str, Any]) -> str:
    encoded = json.dumps([dict(payload) for payload in payloads], sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
