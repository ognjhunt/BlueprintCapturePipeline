"""Run a persistent Vast Unitree GR00T/SONIC + WAM session.

This runner exists to avoid the fragile pattern of allocating a fresh GPU
provider instance for each policy or WAM step. It stages one provider bundle
whose remote entrypoint starts a local policy worker and a local WAM worker,
calls their ``/infer`` endpoints repeatedly, and lets the Vast adapter tear the
single instance down after the session output is uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_policy_runtime import POLICY_ID
from .unitree_groot_n17_sonic_vast_policy_command import (
    ALLOWED_MACHINE_ID_ENVS,
    ALLOW_UNPINNED_FALLBACK_ENV,
    EXCLUDED_MACHINE_ID_ENVS,
    INNER_POLICY_COMMAND_ENV,
    OBJECT_STORE_KEY_PREFIX_ENV,
    PUBLIC_IMAGE_ENV as UNITREE_PUBLIC_IMAGE_ENV,
    VAST_LAUNCH_MODE_ENV,
)
from .vast_provider_adapter import (
    DEFAULT_PUBLIC_CUDA_IMAGE,
    VAST_IMAGE_LOGIN_MODE_ENV,
    run_vast_provider_adapter,
)
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE
from .wam_provider_object_store import stage_wam_provider_bundle_object_store


SCHEMA_VERSION = "unitree_groot_n17_sonic_vast_persistent_session.v1"
BUNDLE_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_bundle.v1"
OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_output.v1"
DEFAULT_BUNDLE_FILENAME = "unitree_groot_n17_sonic_wam_persistent_session_bundle.zip"
DEFAULT_OBJECT_STORE_KEY_PREFIX = "blueprint/unitree-groot-sonic-persistent-session"
PERSISTENT_SESSION_JOB_ROOT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_SESSION_JOB_ROOT"
PERSISTENT_SESSION_PUBLIC_IMAGE_ENV = "BLUEPRINT_VAST_UNITREE_WAM_PERSISTENT_SESSION_PUBLIC_IMAGE"
PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV = (
    "BLUEPRINT_ALLOW_PERSISTENT_SESSION_STRUCTURAL_WAM_FALLBACK"
)
PERSISTENT_SESSION_USE_LIVE_WAM_ENV = "BLUEPRINT_PERSISTENT_SESSION_USE_LIVE_WAM"
PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV = (
    "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND"
)
DEFAULT_INNER_POLICY_COMMAND = (
    "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(_string(os.getenv(name)) or default)
    except ValueError:
        return int(default)


def _machine_ids_from_env(env_names: Sequence[str]) -> list[int]:
    values: list[int] = []
    for env_name in env_names:
        for chunk in _string(os.getenv(env_name)).replace(",", " ").split():
            try:
                machine_id = int(chunk)
            except ValueError:
                continue
            if machine_id > 0 and machine_id not in values:
                values.append(machine_id)
    return values


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_policy_observation(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path).expanduser())
    observation = payload.get("observation") if isinstance(payload.get("observation"), Mapping) else payload
    if not isinstance(observation, Mapping):
        raise ValueError("policy_observation_json_must_contain_object")
    return dict(observation)


def _camera_frame_path(observation: Mapping[str, Any]) -> Path | None:
    visual = _mapping(observation.get("visual_observation"))
    for candidate in (
        visual.get("camera_frame_path"),
        _mapping(observation.get("sensor_surrogates")).get("camera_frame_path"),
        observation.get("camera_frame_path"),
    ):
        text = _string(candidate)
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


PERSISTENT_SESSION_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib import request as urllib_request
from urllib import error as urllib_error

OUTPUT_SCHEMA_VERSION = "unitree_groot_n17_sonic_wam_persistent_session_output.v1"
POLICY_ID = "unitree_groot_n17_sonic_policy"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _command_available(command: str | None) -> bool:
    text = _string(command)
    if not text:
        return False
    try:
        parts = shlex.split(text)
    except ValueError:
        return False
    if not parts:
        return False
    executable = parts[0]
    return bool(shutil.which(executable) or Path(executable).expanduser().is_file())


def _command_uses_policy_server_client(command: str | None) -> bool:
    return "unitree_groot_n17_sonic_policy_server_command" in _string(command)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _phase(name: str, **fields: Any) -> None:
    print(
        "BLUEPRINT_PERSISTENT_SESSION_PHASE:"
        + json.dumps(
            {
                "phase": name,
                "observed_at_epoch": round(time.time(), 3),
                "raw_secret_values_recorded": False,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    value = json.loads(raw.decode("utf-8") or "{}")
    return dict(value) if isinstance(value, Mapping) else {}


def _send(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(dict(payload)).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8") or "{}")
            status_code = int(getattr(response, "status", 200) or 200)
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw or "{}")
        except Exception:
            parsed = {
                "status": "blocked",
                "blockers": [f"persistent_worker_http_error:{exc.code}"],
                "error_message_redacted": raw[-1000:],
            }
        status_code = int(exc.code)
        if isinstance(parsed, Mapping):
            parsed = dict(parsed)
            parsed.setdefault("status", "blocked")
            parsed.setdefault("blockers", [f"persistent_worker_http_error:{exc.code}"])
            parsed["http_status_code"] = status_code
            parsed["http_error"] = True
            return parsed
        parsed = {"status": "blocked", "blockers": [f"persistent_worker_http_error:{exc.code}"], "http_status_code": status_code}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _http_post_json_with_retries(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
    attempts: int = 3,
    sleep_seconds: float = 5.0,
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    for attempt in range(1, max(1, attempts) + 1):
        response = _http_post_json(url, payload, timeout_seconds=timeout_seconds)
        response["persistent_http_attempt_index"] = attempt
        if not response.get("http_error"):
            return response
        if attempt < attempts:
            time.sleep(sleep_seconds)
    return response


def _extract_action(response: Mapping[str, Any]) -> dict[str, Any]:
    action = response.get("action") or response.get("policy_action") or response.get("normalized_action")
    return dict(action) if isinstance(action, Mapping) else {}


def _copy_or_extract_wam_frame(payload: Mapping[str, Any], target_frame: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for key in ("generated_next_observation_frame_path", "camera_frame_path", "frame_path", "image_path"):
        value = _string(payload.get(key))
        if value:
            candidates.append(Path(value).expanduser())
    visual = _mapping(payload.get("visual_observation"))
    if _string(visual.get("camera_frame_path")):
        candidates.append(Path(_string(visual.get("camera_frame_path"))).expanduser())
    for rollout in payload.get("rollouts") or []:
        if isinstance(rollout, Mapping):
            for key in ("generated_video_path", "video_path", "output_video_path"):
                value = _string(rollout.get(key))
                if value:
                    candidates.append(Path(value).expanduser())
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target_frame.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target_frame)
            return {
                "status": "completed",
                "source_kind": "image",
                "source_path": str(candidate),
                "materialized_frame_path": str(target_frame),
            }
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            try:
                import cv2
            except Exception as exc:
                return {
                    "status": "blocked",
                    "blockers": [f"opencv_import_failed:{type(exc).__name__}"],
                    "source_path": str(candidate),
                }
            cap = cv2.VideoCapture(str(candidate))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if ok and frame is not None:
                target_frame.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(target_frame), frame):
                    return {
                        "status": "completed",
                        "source_kind": "video_first_frame",
                        "source_path": str(candidate),
                        "materialized_frame_path": str(target_frame),
                    }
    return {
        "status": "blocked",
        "blockers": ["wam_output_missing_materializable_frame_or_video"],
    }


def _structural_wam_frame(source_frame: Path, target_frame: Path, step_index: int) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"pillow_import_failed:{type(exc).__name__}"]}
    try:
        image = Image.open(source_frame).convert("RGB")
    except Exception:
        image = Image.new("RGB", (640, 480), (32, 35, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 42), fill=(14, 22, 34))
    try:
        font = ImageFont.load_default()
        draw.text((12, 14), f"structural WAM fallback step {step_index}", fill=(240, 246, 250), font=font)
    except Exception:
        pass
    target_frame.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_frame, quality=92)
    return {
        "status": "completed",
        "source_kind": "structural_fallback_image",
        "source_path": str(source_frame),
        "materialized_frame_path": str(target_frame),
    }


class PolicyWorker(BaseHTTPRequestHandler):
    policy_command = ""
    command_source = ""
    command_available = False
    command_uses_policy_server_client = False
    policy_server_url = ""
    timeout_seconds = 240.0
    output_dir = Path(".")

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in {"/readyz", "/healthz"}:
            _send(self, 404, {"status": "not_found"})
            return
        _send(
            self,
            200,
            {
                "schema_version": "persistent_policy_worker_ready.v1",
                "status": "ready" if self.policy_command else "blocked",
                "ready_for_inference": bool(self.policy_command),
                "policy_id": POLICY_ID,
                "persistent_policy_worker_command_source": self.command_source,
                "persistent_policy_worker_command_available": bool(self.command_available),
                "persistent_policy_worker_command_uses_policy_server_client": bool(
                    self.command_uses_policy_server_client
                ),
                "raw_secret_values_recorded": False,
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            _send(self, 404, {"status": "not_found"})
            return
        started = time.monotonic()
        try:
            payload = _read_body(self)
            observation = _mapping(payload.get("observation")) or _mapping(payload)
            from blueprint_pipeline.unitree_groot_n17_sonic_policy_command_adapter import run_unitree_groot_n17_sonic_policy

            uses_policy_server_client = bool(self.command_uses_policy_server_client)
            response, exit_code = run_unitree_groot_n17_sonic_policy(
                payload={"observation": observation},
                command=self.policy_command,
                n17_checkpoint=os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT"),
                sonic_checkpoint=(
                    None
                    if uses_policy_server_client
                    else os.environ.get("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT")
                ),
                groot_root=(
                    None
                    if uses_policy_server_client
                    else os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT")
                ),
                wbc_root=(
                    None
                    if uses_policy_server_client
                    else os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT")
                ),
                policy_server_url=self.policy_server_url,
                sim2sim_command=os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND"),
                timeout_seconds=self.timeout_seconds,
            )
            _send(
                self,
                200 if exit_code == 0 else 502,
                {
                    **dict(response),
                    "persistent_worker_infer": True,
                    "persistent_worker_duration_seconds": round(time.monotonic() - started, 6),
                    "provider_instance_reused_for_policy_infer": True,
                    "persistent_policy_worker_command_configured": bool(self.policy_command),
                    "persistent_policy_worker_command_source": self.command_source,
                    "persistent_policy_worker_command_available": bool(self.command_available),
                    "persistent_policy_worker_command_uses_policy_server_client": uses_policy_server_client,
                    "raw_secret_values_recorded": False,
                },
            )
        except Exception as exc:
            _send(
                self,
                500,
                {
                    "status": "blocked",
                    "policy_id": POLICY_ID,
                    "blockers": [f"persistent_policy_worker_infer_failed:{type(exc).__name__}"],
                    "error": str(exc)[:800],
                    "persistent_policy_worker_command_configured": bool(self.policy_command),
                    "persistent_policy_worker_command_source": self.command_source,
                    "persistent_policy_worker_command_available": bool(self.command_available),
                    "persistent_policy_worker_command_uses_policy_server_client": bool(
                        self.command_uses_policy_server_client
                    ),
                    "raw_secret_values_recorded": False,
                },
            )


class WamWorker(BaseHTTPRequestHandler):
    output_dir = Path(".")
    use_live_wam = True
    allow_structural_fallback = False
    timeout_seconds = 3600.0

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in {"/readyz", "/healthz"}:
            _send(self, 404, {"status": "not_found"})
            return
        _send(
            self,
            200,
            {
                "schema_version": "persistent_wam_worker_ready.v1",
                "status": "ready",
                "ready_for_inference": True,
                "use_live_wam": self.use_live_wam,
                "allow_structural_fallback": self.allow_structural_fallback,
                "raw_secret_values_recorded": False,
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            _send(self, 404, {"status": "not_found"})
            return
        started = time.monotonic()
        payload = _read_body(self)
        step_index = int(payload.get("step_index") or 0)
        source_frame = Path(_string(payload.get("source_frame"))).expanduser()
        step_dir = self.output_dir / "wam_worker_steps" / f"step_{step_index:04d}"
        target_frame = self.output_dir / "generated_next_observations" / f"wam_generated_next_observation_step_{step_index:04d}.jpg"
        step_dir.mkdir(parents=True, exist_ok=True)
        step_input = {
            "schema_version": "wam_generation_step_input.v1",
            "generated_at": payload.get("generated_at"),
            "step_index": step_index,
            "wam_evaluator_backend": "persistent_oscar_wam_worker",
            "source_policy_observation_frame_path": str(source_frame),
            "source_policy_action": _mapping(payload.get("source_policy_action")),
            "current_policy_observation": _mapping(payload.get("current_policy_observation")),
            "requested_output": {
                "next_observation_frame_path": str(target_frame),
                "action_conditioned_generation_required": True,
            },
            "claim_boundary": {
                "wam_generation_is_not_robot_policy": True,
                "physical_robot_sensor_proof": False,
            },
        }
        step_input_path = step_dir / "wam_generation_step_input.json"
        _write_json(step_input_path, step_input)
        live_payload: dict[str, Any] = {}
        live_blockers: list[str] = []
        materialization: dict[str, Any] = {}
        live_ran = False
        if self.use_live_wam:
            try:
                from blueprint_pipeline.oscar_wam_provider_bundle import build_oscar_wam_provider_bundle

                bundle = build_oscar_wam_provider_bundle(
                    job_dir=step_dir / "oscar_wam_worker_bundle",
                    wam_rollout_input_manifest=step_input_path,
                    timeout_seconds=int(self.timeout_seconds),
                    num_steps=int(os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "12")),
                    num_frames=int(os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "24")),
                    height=int(os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT", "480")),
                    width=int(os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH", "640")),
                    fps=float(os.environ.get("BLUEPRINT_OSCAR_WAM_FPS", "15")),
                )
                if bundle.get("status") != "completed":
                    live_blockers.extend(bundle.get("blockers") or ["oscar_wam_provider_bundle_blocked"])
                else:
                    bundle_root = Path(str(bundle["job_dir"])) / "oscar_wam_provider_bundle"
                    output_dir = step_dir / "oscar_runtime_output"
                    env = os.environ.copy()
                    env["BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR"] = str(bundle_root)
                    env["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"] = str(output_dir)
                    env["BLUEPRINT_WAM_PROVIDER_WORK_DIR"] = str(self.output_dir / "persistent_wam_shared_work")
                    env["BLUEPRINT_WAM_ROLLOUT_INPUT"] = str(bundle_root / "provider_runtime" / "wam_rollout_input_manifest.json")
                    completed = subprocess.run(
                        ["bash", str(bundle_root / "provider_runtime" / "run_wam_provider_runtime.sh")],
                        cwd=str(bundle_root),
                        env=env,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=self.timeout_seconds,
                    )
                    live_ran = True
                    provider_output_path = output_dir / "wam_provider_output.json"
                    if provider_output_path.is_file():
                        live_payload = json.loads(provider_output_path.read_text(encoding="utf-8"))
                        live_payload = dict(live_payload) if isinstance(live_payload, Mapping) else {}
                    if completed.returncode != 0:
                        live_blockers.append("persistent_wam_worker_oscar_runtime_nonzero_exit")
                    if not live_payload:
                        live_blockers.append("persistent_wam_worker_missing_oscar_provider_output")
                    materialization = _copy_or_extract_wam_frame(live_payload, target_frame)
                    if materialization.get("status") != "completed":
                        live_blockers.extend(materialization.get("blockers") or ["persistent_wam_frame_materialization_failed"])
                    _write_json(
                        step_dir / "persistent_wam_worker_command_execution.json",
                        {
                            "schema_version": "persistent_wam_worker_command_execution.v1",
                            "status": "completed" if completed.returncode == 0 else "blocked",
                            "returncode": completed.returncode,
                            "stdout_size_bytes": len(completed.stdout or ""),
                            "stderr_size_bytes": len(completed.stderr or ""),
                            "stdout_omitted_to_avoid_secret_leakage": bool(completed.stdout),
                            "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
                            "bundle_manifest": bundle,
                            "raw_secret_values_recorded": False,
                        },
                    )
            except Exception as exc:
                live_blockers.append(f"persistent_wam_worker_live_infer_failed:{type(exc).__name__}")
                _write_json(
                    step_dir / "persistent_wam_worker_exception.json",
                    {
                        "status": "blocked",
                        "error_type": type(exc).__name__,
                        "traceback_tail": traceback.format_exc()[-4000:],
                        "raw_secret_values_recorded": False,
                    },
                )
        fallback_used = False
        structural_fallback_requested = (
            self.allow_structural_fallback and (bool(live_blockers) or not self.use_live_wam)
        )
        if structural_fallback_requested:
            fallback_used = True
            materialization = _structural_wam_frame(source_frame, target_frame, step_index)
            if materialization.get("status") != "completed":
                live_blockers.extend(
                    materialization.get("blockers")
                    or ["persistent_structural_wam_fallback_materialization_failed"]
                )
        completed = target_frame.is_file() and (
            (not live_blockers and materialization.get("status") == "completed")
            or (fallback_used and materialization.get("status") == "completed")
        )
        if not completed and not live_blockers and not self.use_live_wam:
            live_blockers.append("persistent_wam_live_disabled_without_structural_fallback")
        response = {
            "schema_version": "persistent_wam_worker_infer_response.v1",
            "status": "completed" if completed else "blocked",
            "step_index": step_index,
            "wam_evaluator_backend": "persistent_oscar_wam_worker" if not fallback_used else "persistent_structural_wam_fallback",
            "provider_instance_reused_for_wam_infer": True,
            "persistent_wam_worker_infer": True,
            "persistent_worker_duration_seconds": round(time.monotonic() - started, 6),
            "live_wam_generation_command_ran": bool(live_ran and not fallback_used),
            "learned_oscar_or_cosmos_model_ran": bool(
                not fallback_used and live_payload.get("status") == "completed"
            ),
            "wam_model_checkpoint_used": bool(
                not fallback_used and _mapping(live_payload.get("model_provenance")).get("checkpoint_path")
            ),
            "action_conditioned_generation_ran": bool(completed),
            "generated_next_observation_frame_path": str(target_frame) if target_frame.is_file() else None,
            "materialization": materialization,
            "live_wam_payload_redacted": live_payload,
            "structural_fallback_used": fallback_used,
            "blockers": [] if completed else sorted(set(live_blockers)),
            "claim_boundary": {
                "wam_is_next_observation_generator_not_robot_policy": True,
                "generated_observation_is_not_raw_capture": True,
                "structural_fallback_is_not_live_wam_model": fallback_used,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
            },
            "raw_secret_values_recorded": False,
        }
        _write_json(step_dir / "persistent_wam_worker_infer_response.json", response)
        _send(self, 200 if completed else 502, response)


def _start_server(port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _side_by_side_html(path: Path, rows: list[Mapping[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<section><h2>Step {}</h2><pre>{}</pre></section>".format(
                row.get("transition_index"),
                json.dumps(dict(row), indent=2, sort_keys=True),
            )
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>Persistent Policy WAM Trace</title>",
                "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7}section{background:white;border:1px solid #ddd;border-radius:8px;padding:16px;margin:0 0 16px}pre{white-space:pre-wrap;font-size:12px}</style>",
                "</head><body><h1>Persistent Policy WAM Trace</h1>",
                *cards,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    runtime_dir = Path(__file__).resolve().parent
    output_dir = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR", runtime_dir / "runtime_output")).resolve()
    output_path = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", output_dir / "unitree_groot_n17_sonic_policy_provider_output.json")).resolve()
    session_input_path = Path(os.environ.get("BLUEPRINT_PERSISTENT_SESSION_INPUT", runtime_dir / "persistent_session_input.json")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    policy_server_process = None
    policy_server = None
    wam_server = None
    try:
        session_input = json.loads(session_input_path.read_text(encoding="utf-8"))
        observation = _mapping(session_input.get("initial_observation"))
        loop_step_count = max(2, int(session_input.get("loop_step_count") or 12))
        policy_port = int(session_input.get("policy_worker_port") or 8765)
        wam_port = int(session_input.get("wam_worker_port") or 8766)
        use_live_wam = bool(session_input.get("use_live_wam") is not False)
        allow_structural_fallback = bool(session_input.get("allow_structural_wam_fallback"))
        timeout_seconds = float(session_input.get("timeout_seconds") or 3600.0)
        initial_frame = runtime_dir / "initial_policy_frame.png"
        visual = _mapping(observation.get("visual_observation"))
        visual["camera_frame_path"] = str(initial_frame)
        observation["visual_observation"] = visual
        observation["camera_frame_path"] = str(initial_frame)

        _phase("bootstrap_policy_server_started")
        from blueprint_pipeline import unitree_groot_n17_sonic_provider_smoke as provider_smoke

        bootstrap_namespace: dict[str, Any] = {
            "__name__": "blueprint_persistent_session_bootstrap",
            "__file__": str(runtime_dir / "unitree_groot_n17_sonic_provider_runner.py"),
        }
        exec(provider_smoke.PROVIDER_RUNNER, bootstrap_namespace)
        _bootstrap_gr00t_policy_server = bootstrap_namespace["_bootstrap_gr00t_policy_server"]

        policy_server_url = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL", "tcp://127.0.0.1:5550")
        policy_server_bootstrap, policy_server_process = _bootstrap_gr00t_policy_server(
            output_dir=output_dir,
            policy_server_url=policy_server_url,
            model_path=os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT") or "LucaFrat/groot-bs16",
        )
        _phase("bootstrap_policy_server_completed", status=policy_server_bootstrap.get("status"))
        if policy_server_bootstrap.get("status") != "completed":
            raise RuntimeError("persistent_session_policy_server_bootstrap_blocked")

        configured_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PERSISTENT_INNER_POLICY_COMMAND", "")
        configured_command_source = "persistent_inner_policy_command_env" if configured_command else ""
        if not configured_command:
            configured_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_INNER_POLICY_COMMAND", "")
            configured_command_source = "vast_inner_policy_command_env" if configured_command else configured_command_source
        if not configured_command:
            configured_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", "")
            configured_command_source = "policy_command_env" if configured_command else configured_command_source
        command = configured_command
        command_source = configured_command_source or "unset"
        repo_root = _mapping(policy_server_bootstrap.get("checkout")).get("repo_root")
        venv_python = policy_server_bootstrap.get("venv_python")
        if not repo_root and venv_python:
            derived_repo_root = Path(str(venv_python)).expanduser().resolve().parent.parent.parent / "Isaac-GR00T"
            if derived_repo_root.is_dir():
                repo_root = str(derived_repo_root)
        venv_python_path = Path(str(venv_python)).expanduser() if venv_python else None
        venv_python_available = bool(venv_python_path and venv_python_path.is_file())
        if policy_server_bootstrap.get("status") == "completed" and venv_python_available:
            command = f"{shlex.quote(str(venv_python))} -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
            command_source = "bootstrap_venv_policy_server_client_for_persistent_session"
            if repo_root:
                os.environ["PYTHONPATH"] = str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
        if not command:
            command = "python3 -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
            command_source = "python3_policy_server_client_fallback"
        command_available = _command_available(command)
        command_uses_policy_server_client = _command_uses_policy_server_client(command)
        _phase(
            "policy_worker_command_selected",
            command_source=command_source,
            command_available=command_available,
            command_uses_policy_server_client=command_uses_policy_server_client,
            repo_root_configured=bool(repo_root),
            venv_python_available=venv_python_available,
            configured_command_source=configured_command_source or None,
        )

        PolicyWorker.policy_command = command
        PolicyWorker.command_source = command_source
        PolicyWorker.command_available = command_available
        PolicyWorker.command_uses_policy_server_client = command_uses_policy_server_client
        PolicyWorker.policy_server_url = policy_server_url
        PolicyWorker.timeout_seconds = timeout_seconds
        PolicyWorker.output_dir = output_dir
        WamWorker.output_dir = output_dir
        WamWorker.use_live_wam = use_live_wam
        WamWorker.allow_structural_fallback = allow_structural_fallback
        WamWorker.timeout_seconds = timeout_seconds
        policy_server = _start_server(policy_port, PolicyWorker)
        wam_server = _start_server(wam_port, WamWorker)
        _phase("workers_started", policy_port=policy_port, wam_port=wam_port)

        policy_calls: list[dict[str, Any]] = []
        wam_calls: list[dict[str, Any]] = []
        side_rows: list[dict[str, Any]] = []
        current_observation = observation
        current_frame = initial_frame
        current_action: dict[str, Any] = {}
        blockers: list[str] = []
        for step_index in range(loop_step_count):
            _phase("policy_infer_started", step_index=step_index)
            policy_response = _http_post_json_with_retries(
                f"http://127.0.0.1:{policy_port}/infer",
                {"observation": current_observation},
                timeout_seconds=timeout_seconds,
                attempts=3,
                sleep_seconds=5.0,
            )
            action = _extract_action(policy_response)
            policy_row = {
                "step_index": step_index,
                "status": "completed" if policy_response.get("status") == "completed" and action else "blocked",
                "policy_id": policy_response.get("policy_id"),
                "policy_observation_frame_path": str(current_frame),
                "action": action,
                "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                    policy_response.get("unitree_groot_n17_sonic_policy_action_command_ran")
                ),
                "unitree_policy_action_command_ran": bool(policy_response.get("unitree_policy_action_command_ran")),
                "provider_output_replay_used": bool(policy_response.get("provider_output_replay_used")),
                "worker_response_redacted": policy_response,
            }
            policy_calls.append(policy_row)
            _write_json(output_dir / "policy_calls" / f"policy_call_{step_index:04d}.json", policy_row)
            if policy_row["status"] != "completed":
                blockers.extend(policy_response.get("blockers") or ["persistent_policy_infer_blocked"])
                break
            current_action = action
            if step_index >= loop_step_count - 1:
                break
            _phase("wam_infer_started", step_index=step_index + 1)
            wam_response = _http_post_json_with_retries(
                f"http://127.0.0.1:{wam_port}/infer",
                {
                    "generated_at": session_input.get("generated_at"),
                    "step_index": step_index + 1,
                    "source_frame": str(current_frame),
                    "current_policy_observation": current_observation,
                    "source_policy_action": current_action,
                },
                timeout_seconds=timeout_seconds,
                attempts=2,
                sleep_seconds=5.0,
            )
            wam_calls.append(wam_response)
            _write_json(output_dir / "wam_calls" / f"wam_call_{step_index + 1:04d}.json", wam_response)
            if wam_response.get("status") != "completed":
                blockers.extend(wam_response.get("blockers") or ["persistent_wam_infer_blocked"])
                break
            next_frame = Path(_string(wam_response.get("generated_next_observation_frame_path"))).expanduser()
            generated_observation = {
                "schema_version": "wam_generated_next_observation.v1",
                "generated_observation_index": step_index + 1,
                "observation_source": "persistent_wam_worker_next_observation",
                "wam_evaluator_backend": wam_response.get("wam_evaluator_backend"),
                "wam_model_checkpoint_used": bool(wam_response.get("wam_model_checkpoint_used")),
                "action_conditioned_generation_ran": bool(wam_response.get("action_conditioned_generation_ran")),
                "live_wam_generation_command_ran": bool(wam_response.get("live_wam_generation_command_ran")),
                "learned_oscar_or_cosmos_model_ran": bool(wam_response.get("learned_oscar_or_cosmos_model_ran")),
                "generated_next_observation_frame_path": str(next_frame),
                "visual_observation": {
                    **_mapping(current_observation.get("visual_observation")),
                    "camera_frame_path": str(next_frame),
                    "wam_generated_observation": True,
                    "physical_robot_sensor_proof": False,
                },
            }
            side_rows.append(
                {
                    "schema_version": "persistent_policy_wam_side_by_side_trace_row.v1",
                    "transition_index": step_index + 1,
                    "policy_pov_frame_path": str(current_frame),
                    "policy_action_summary": {
                        "action_type": current_action.get("action_type"),
                        "action_chunk_length": len(current_action.get("action_chunk") or []),
                    },
                    "wam_generated_next_observation_frame_path": str(next_frame),
                    "wam_evaluator_backend": wam_response.get("wam_evaluator_backend"),
                    "live_wam_generation_command_ran": bool(wam_response.get("live_wam_generation_command_ran")),
                    "learned_oscar_or_cosmos_model_ran": bool(wam_response.get("learned_oscar_or_cosmos_model_ran")),
                    "next_policy_call_expected": True,
                }
            )
            current_observation = {
                **current_observation,
                **generated_observation,
                "camera_frame_path": str(next_frame),
                "visual_observation": generated_observation["visual_observation"],
            }
            current_frame = next_frame
        repeated_policy_calls = sum(
            1
            for row in policy_calls
            if row.get("status") == "completed"
            and row.get("unitree_policy_action_command_ran")
            and not row.get("provider_output_replay_used")
        )
        generated_count = sum(1 for row in wam_calls if row.get("status") == "completed")
        live_wam_count = sum(1 for row in wam_calls if row.get("live_wam_generation_command_ran"))
        learned_wam_count = sum(1 for row in wam_calls if row.get("learned_oscar_or_cosmos_model_ran"))
        _write_jsonl(output_dir / "robot_policy_wam_loop_trace.jsonl", policy_calls)
        _write_jsonl(output_dir / "wam_generated_next_observations.jsonl", wam_calls)
        _write_jsonl(output_dir / "robot_policy_wam_side_by_side_trace.jsonl", side_rows)
        _side_by_side_html(output_dir / "robot_policy_wam_side_by_side_trace.html", side_rows)
        completed = bool(
            repeated_policy_calls >= 2
            and generated_count >= 1
            and not blockers
            and (live_wam_count >= 1 or allow_structural_fallback)
        )
        result = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "completed" if completed else "blocked",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "policy_worker_url_redacted": f"http://127.0.0.1:{policy_port}/infer",
            "wam_worker_url_redacted": f"http://127.0.0.1:{wam_port}/infer",
            "policy_server_bootstrap": policy_server_bootstrap,
            "requested_loop_step_count": loop_step_count,
            "repeated_policy_calls_count": repeated_policy_calls,
            "generated_next_observation_count": generated_count,
            "live_wam_generation_success_count": live_wam_count,
            "learned_wam_model_success_count": learned_wam_count,
            "policy_observes_wam_generated_next_observation": repeated_policy_calls >= 2 and generated_count >= 1,
            "wam_evaluator_in_control_loop": generated_count >= 1,
            "unitree_groot_n17_sonic_model_executed": repeated_policy_calls >= 1,
            "unitree_groot_n17_sonic_policy_action_command_ran": repeated_policy_calls >= 1,
            "unitree_policy_action_command_ran": repeated_policy_calls >= 1,
            "policy_action_model_command_ran": repeated_policy_calls >= 1,
            "provider_output_replay_used": False,
            "trace_path": str(output_dir / "robot_policy_wam_loop_trace.jsonl"),
            "wam_generated_next_observations_jsonl": str(output_dir / "wam_generated_next_observations.jsonl"),
            "side_by_side_trace_path": str(output_dir / "robot_policy_wam_side_by_side_trace.jsonl"),
            "side_by_side_trace_html_path": str(output_dir / "robot_policy_wam_side_by_side_trace.html"),
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "duration_seconds": round(time.monotonic() - started, 6),
            "claim_boundary": {
                "simulator_generated_world_proof_only": True,
                "persistent_provider_session_is_runtime_proof_not_task_success": True,
                "wam_is_next_observation_generator_not_robot_policy": True,
                "generated_observations_are_not_raw_capture": True,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "action": policy_calls[-1].get("action") if policy_calls else None,
        }
        _write_json(output_path, result)
        _write_json(output_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json", result)
        return 0 if completed else 2
    except Exception as exc:
        result = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "failed",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "persistent_provider_session_used": True,
            "unitree_groot_n17_sonic_model_executed": False,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "unitree_policy_action_command_ran": False,
            "policy_action_model_command_ran": False,
            "provider_output_replay_used": False,
            "traceback_tail": traceback.format_exc()[-4000:],
            "blockers": [f"persistent_session_runner_failed:{type(exc).__name__}"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": {
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
        }
        _write_json(output_path, result)
        _write_json(output_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json", result)
        return 1
    finally:
        if policy_server is not None:
            policy_server.shutdown()
        if wam_server is not None:
            wam_server.shutdown()
        if policy_server_process is not None and policy_server_process.poll() is None:
            policy_server_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUN_SCRIPT = """#!/usr/bin/env bash
set +e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
RUNTIME_PY="${RUNTIME_PY:-python3}"
if ! "$RUNTIME_PY" - <<'PY' >/tmp/blueprint_persistent_session_deps_probe.log 2>&1
import importlib.util
missing=[m for m in ['numpy','PIL','zmq','msgpack','msgpack_numpy','cv2'] if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  "$RUNTIME_PY" -m pip install --quiet --only-binary=:all: --timeout 60 --retries 1 --break-system-packages numpy pillow pyzmq msgpack msgpack-numpy opencv-python-headless >/tmp/blueprint_persistent_session_pip_install.log 2>&1
fi
"$RUNTIME_PY" unitree_groot_n17_sonic_wam_persistent_session_runner.py
runner_rc=$?
if [ $runner_rc -ne 0 ] && [ ! -f "${BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT:-}" ]; then
"$RUNTIME_PY" - <<'PY'
import json
import os
from pathlib import Path
out = Path(os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", "unitree_groot_n17_sonic_policy_provider_output.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_output.v1",
    "status": "failed",
    "policy_id": "unitree_groot_n17_sonic_policy",
    "persistent_provider_session_used": True,
    "unitree_groot_n17_sonic_model_executed": False,
    "unitree_groot_n17_sonic_policy_action_command_ran": False,
    "policy_action_model_command_ran": False,
    "provider_output_replay_used": False,
    "blockers": ["persistent_session_runner_failed_without_runtime_result"],
    "legacy_blockers": [
        "unitree_groot_n17_sonic_provider_runner_failed_without_runtime_result",
        "blocked_unitree_groot_n17_sonic_process_exited_without_result"
    ],
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
fi
exit $runner_rc
"""


def _copy_blueprint_runtime(runtime_dir: Path) -> list[str]:
    package_dir = runtime_dir / "blueprint_pipeline"
    ensure_dir(package_dir)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    copied = ["provider_runtime/blueprint_pipeline/__init__.py"]
    source_dir = Path(__file__).resolve().parent
    for filename in (
        "common.py",
        "unitree_groot_n17_sonic_policy_command_adapter.py",
        "unitree_groot_n17_sonic_policy_runtime.py",
        "unitree_groot_n17_sonic_policy_server_command.py",
        "unitree_groot_n17_sonic_provider_smoke.py",
        "oscar_wam_provider_bundle.py",
        "oscar_wam_command_adapter.py",
        "wam_generated_video_review.py",
    ):
        shutil.copy2(source_dir / filename, package_dir / filename)
        copied.append(f"provider_runtime/blueprint_pipeline/{filename}")
    return copied


def build_persistent_session_provider_bundle(
    *,
    job_dir: str | Path,
    policy_observation_path: str | Path,
    loop_step_count: int = 12,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool = False,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    runtime_dir = job / "provider_runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    ensure_dir(runtime_dir)
    observation = _load_policy_observation(policy_observation_path)
    if task_prompt and not any(observation.get(key) for key in ("task_prompt", "prompt", "task_description")):
        observation["task_prompt"] = task_prompt
    frame_path = _camera_frame_path(observation)
    blockers: list[str] = []
    if frame_path is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    else:
        shutil.copy2(frame_path, runtime_dir / "initial_policy_frame.png")
        shutil.copy2(frame_path, runtime_dir / "input_frame.png")
    copied = _copy_blueprint_runtime(runtime_dir)
    _write_executable(
        runtime_dir / "unitree_groot_n17_sonic_wam_persistent_session_runner.py",
        PERSISTENT_SESSION_RUNNER,
    )
    _write_executable(
        runtime_dir / "unitree_groot_n17_sonic_provider_runner.py",
        PERSISTENT_SESSION_RUNNER,
    )
    _write_executable(runtime_dir / "run_unitree_groot_n17_sonic_provider_runtime.sh", RUN_SCRIPT)
    session_input = {
        "schema_version": "unitree_groot_n17_sonic_wam_persistent_session_input.v1",
        "generated_at": generated,
        "initial_observation": observation,
        "loop_step_count": int(loop_step_count),
        "timeout_seconds": float(timeout_seconds),
        "use_live_wam": bool(use_live_wam),
        "allow_structural_wam_fallback": bool(allow_structural_wam_fallback),
        "policy_worker_port": 8765,
        "wam_worker_port": 8766,
        "claim_boundary": {
            "simulator_generated_world_proof_only": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
    }
    write_json(runtime_dir / "persistent_session_input.json", session_input)
    write_json(runtime_dir / "policy_input.json", {"observation": observation})
    write_json(
        runtime_dir / "unitree_groot_n17_sonic_policy_provider_manifest.json",
        {
            "schema_version": "unitree_groot_n17_sonic_policy_provider_bundle.v1",
            "generated_at": generated,
            "status": "bundle_ready" if not blockers else "blocked",
            "local_bundle_ready_for_remote_staging": not blockers,
            "persistent_session_bundle": True,
            "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
            "runner_path": "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py",
            "legacy_runner_path_for_vast_preflight": "provider_runtime/unitree_groot_n17_sonic_provider_runner.py",
            "policy_id": POLICY_ID,
            "blockers": blockers,
            "claim_boundary": {
                "bundle_build_is_not_model_execution": True,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    )
    bundle_path = job / bundle_filename
    if bundle_path.exists():
        bundle_path.unlink()
    zip_entries: list[str] = []
    if not blockers:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(runtime_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(job).as_posix())
        with zipfile.ZipFile(bundle_path) as archive:
            zip_entries = sorted(archive.namelist())
            if archive.testzip() is not None:
                blockers.append("persistent_session_bundle_zip_integrity_failed")
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "bundle_ready" if not blockers else "blocked",
        "job_dir": str(job),
        "bundle_path": str(bundle_path),
        "bundle_present": bundle_path.is_file(),
        "bundle_size_bytes": bundle_path.stat().st_size if bundle_path.is_file() else 0,
        "runtime_entrypoint": "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh",
        "runtime_runner": "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py",
        "policy_observation_path": str(Path(policy_observation_path).expanduser()),
        "initial_frame_path": str(frame_path) if frame_path else None,
        "loop_step_count": int(loop_step_count),
        "use_live_wam": bool(use_live_wam),
        "allow_structural_wam_fallback": bool(allow_structural_wam_fallback),
        "zip_entry_count": len(zip_entries),
        "zip_entries": zip_entries,
        "copied_blueprint_runtime_files": copied,
        "provider_bundle_kind": "unitree_groot_n17_sonic",
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "bundle_build_is_not_model_execution": True,
            "persistent_session_reuses_provider_instance_after_launch": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
    }
    write_json(job / "persistent_session_provider_bundle_manifest.json", manifest)
    return manifest


def _job_dir(root: str | Path | None = None) -> Path:
    root_path = Path(root).expanduser() if root else Path(
        _string(os.getenv(PERSISTENT_SESSION_JOB_ROOT_ENV))
        or Path.cwd() / "unitree_groot_n17_sonic_vast_persistent_session"
    )
    job = root_path / utc_now_iso().replace(":", "").replace("+", "_").replace("-", "")
    ensure_dir(job)
    return job.resolve()


def _blocked_payload(
    *,
    generated_at: str,
    job_dir: Path,
    blockers: Sequence[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "job_dir": str(job_dir),
        "blockers": sorted({str(item) for item in blockers if str(item)}),
        "details": dict(details or {}),
        "persistent_provider_session_used": False,
        "unitree_groot_n17_sonic_model_executed": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_policy_action_command_ran": False,
        "policy_action_model_command_ran": False,
        "provider_output_replay_used": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _load_json_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            value = _read_json(path)
        except Exception:
            continue
        rows.append(value)
    return rows


def _action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    chunk = action.get("action_chunk")
    return {
        "action_type": action.get("action_type"),
        "action_chunk_present": isinstance(chunk, Sequence) and not isinstance(chunk, (str, bytes, bytearray)),
        "action_chunk_length": len(chunk)
        if isinstance(chunk, Sequence) and not isinstance(chunk, (str, bytes, bytearray))
        else None,
        "source_action_key": action.get("source_action_key"),
        "control_fields": action.get("unitree_g1_sonic_control_fields"),
    }


def _concat_file_line(path: Path) -> str:
    return "file '{}'\n".format(str(path).replace("'", "'\\''"))


def _write_review_video(
    *,
    job: Path,
    extraction_dir: Path,
    generated_at: str,
    fps: float = 2.0,
) -> dict[str, Any]:
    review_dir = job / "review_video"
    ensure_dir(review_dir)
    output_path = review_dir / "persistent_policy_wam_review.mp4"
    initial_frame = job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    frames = [initial_frame] if initial_frame.is_file() else []
    frames.extend(sorted((extraction_dir / "generated_next_observations").glob("*.jpg")))
    frames = [path.resolve() for path in frames if path.is_file()]
    concat_path = review_dir / "persistent_policy_wam_review_frames.ffconcat"
    status = {
        "schema_version": "persistent_policy_wam_video_review_status.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "review_video_path": str(output_path),
        "frame_count": len(frames),
        "fps_requested": fps,
        "ffmpeg_command_ran": False,
        "ffprobe_command_ran": False,
        "ffprobe_metadata": {},
        "blockers": [],
        "claim_boundary": {
            "video_is_review_artifact_not_task_success_proof": True,
            "structural_fallback_video_is_not_live_wam_model_proof": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    if len(frames) < 2:
        status["blockers"] = ["not_enough_frames_for_review_video"]
        write_json(job / "video_review_status.json", status)
        return status
    if shutil.which("ffmpeg") is None:
        status["blockers"] = ["ffmpeg_not_available_for_review_video"]
        write_json(job / "video_review_status.json", status)
        return status
    duration = 1.0 / float(fps)
    with concat_path.open("w", encoding="utf-8") as handle:
        handle.write("ffconcat version 1.0\n")
        for frame in frames:
            handle.write(_concat_file_line(frame))
            handle.write(f"duration {duration:.6f}\n")
        handle.write(_concat_file_line(frames[-1]))
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags",
        "+faststart",
        "-r",
        str(fps),
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    status["ffmpeg_command_ran"] = True
    status["ffmpeg_returncode"] = completed.returncode
    status["ffmpeg_stdout_size_bytes"] = len(completed.stdout or "")
    status["ffmpeg_stderr_size_bytes"] = len(completed.stderr or "")
    if completed.returncode != 0 or not output_path.is_file():
        status["blockers"] = ["ffmpeg_review_video_failed"]
        write_json(job / "video_review_status.json", status)
        return status
    if shutil.which("ffprobe") is not None:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        status["ffprobe_command_ran"] = True
        status["ffprobe_returncode"] = probe.returncode
        if probe.returncode == 0:
            try:
                parsed = json.loads(probe.stdout or "{}")
            except json.JSONDecodeError:
                parsed = {}
            status["ffprobe_metadata"] = parsed if isinstance(parsed, Mapping) else {}
    status["status"] = "completed"
    status["blockers"] = []
    write_json(job / "video_review_status.json", status)
    return status


def _postprocess_imported_persistent_session_artifacts(
    *,
    job: Path,
    extraction_dir: Path,
    imported: Mapping[str, Any],
    generated_at: str,
    policy_observation_path: str | Path,
    vast_result: Mapping[str, Any],
    vast_run_dir: Path,
) -> dict[str, Any]:
    policy_calls = _load_json_rows(sorted((extraction_dir / "policy_calls").glob("policy_call_*.json")))
    wam_calls = _load_json_rows(sorted((extraction_dir / "wam_calls").glob("wam_call_*.json")))
    trace_rows = _jsonl_rows(extraction_dir / "robot_policy_wam_loop_trace.jsonl")
    side_rows = _jsonl_rows(extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl")
    wam_rows = _jsonl_rows(extraction_dir / "wam_generated_next_observations.jsonl")
    for filename in (
        "robot_policy_wam_side_by_side_trace.jsonl",
        "robot_policy_wam_side_by_side_trace.html",
        "wam_generated_next_observations.jsonl",
        "robot_policy_wam_loop_trace.jsonl",
    ):
        source = extraction_dir / filename
        if source.is_file():
            shutil.copy2(source, job / filename)
    first_policy = policy_calls[0] if policy_calls else {}
    first_action = _mapping(first_policy.get("action"))
    policy_completed_count = sum(1 for row in policy_calls if row.get("status") == "completed")
    wam_completed_count = sum(1 for row in wam_rows if row.get("status") == "completed")
    structural_wam_count = sum(1 for row in wam_rows if row.get("structural_fallback_used") is True)
    live_wam_count = int(imported.get("live_wam_generation_success_count") or 0)
    learned_wam_count = int(imported.get("learned_wam_model_success_count") or 0)
    write_json(
        job / "policy_action_model_command_discovery.json",
        {
            "schema_version": "policy_action_model_command_discovery.v1",
            "generated_at": generated_at,
            "status": "completed" if policy_completed_count else "blocked",
            "selected_candidate_id": POLICY_ID,
            "candidate_checkpoint": "LucaFrat/groot-bs16",
            "candidate_priority": "default_experimental_unitree_g1_sonic",
            "trusted_for_production": False,
            "policy_server_client_used": True,
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "policy_action_model_command_execution.json",
        {
            "schema_version": "policy_action_model_command_execution.v1",
            "generated_at": generated_at,
            "status": "completed" if policy_completed_count else "blocked",
            "policy_call_count": len(policy_calls),
            "completed_policy_call_count": policy_completed_count,
            "persistent_policy_worker_command_source": _mapping(
                first_policy.get("worker_response_redacted")
            ).get("persistent_policy_worker_command_source"),
            "policy_server_bootstrap_status": _mapping(
                imported.get("policy_server_bootstrap")
            ).get("status"),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "policy_action_model_command_output.json",
        {
            "schema_version": "policy_action_model_command_output.v1",
            "generated_at": generated_at,
            "status": "completed" if first_action else "blocked",
            "selected_candidate_id": POLICY_ID,
            "policy_calls_dir": str(extraction_dir / "policy_calls"),
            "first_action_summary": _action_summary(first_action),
            "full_action_payloads_are_in_policy_call_artifacts": True,
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "wam_generation_command_discovery.json",
        {
            "schema_version": "wam_generation_command_discovery.v1",
            "generated_at": generated_at,
            "status": "completed" if wam_completed_count else "blocked",
            "wam_evaluator_backend": "persistent_structural_wam_fallback"
            if structural_wam_count
            else "persistent_oscar_wam_worker",
            "live_wam_generation_success_count": live_wam_count,
            "learned_wam_model_success_count": learned_wam_count,
            "structural_fallback_count": structural_wam_count,
            "live_wam_model_configured": live_wam_count > 0,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "wam_generation_command_execution.json",
        {
            "schema_version": "wam_generation_command_execution.v1",
            "generated_at": generated_at,
            "status": "completed" if wam_completed_count else "blocked",
            "wam_call_count": len(wam_calls) or len(wam_rows),
            "completed_wam_call_count": wam_completed_count,
            "action_conditioned_generation_ran": bool(wam_completed_count),
            "live_wam_generation_command_ran": live_wam_count > 0,
            "structural_fallback_used": structural_wam_count > 0,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "wam_generation_command_output.json",
        {
            "schema_version": "wam_generation_command_output.v1",
            "generated_at": generated_at,
            "status": "completed" if wam_completed_count else "blocked",
            "wam_generated_next_observations_jsonl": str(
                job / "wam_generated_next_observations.jsonl"
            ),
            "generated_next_observations_dir": str(extraction_dir / "generated_next_observations"),
            "generated_next_observation_count": wam_completed_count,
            "live_wam_generation_success_count": live_wam_count,
            "learned_wam_model_success_count": learned_wam_count,
            "structural_fallback_used": structural_wam_count > 0,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    write_json(
        job / "robot_policy_wam_loop_manifest.json",
        {
            "schema_version": "robot_policy_wam_loop_manifest.v1",
            "generated_at": generated_at,
            "status": imported.get("status"),
            "policy_observation_path": str(Path(policy_observation_path).expanduser()),
            "persistent_provider_session_used": bool(imported.get("persistent_provider_session_used")),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "repeated_policy_calls_count": int(imported.get("repeated_policy_calls_count") or 0),
            "generated_next_observation_count": int(imported.get("generated_next_observation_count") or 0),
            "policy_observes_wam_generated_next_observation": bool(
                imported.get("policy_observes_wam_generated_next_observation")
            ),
            "trace_row_count": len(trace_rows),
            "side_by_side_trace_row_count": len(side_rows),
            "policy_calls_dir": str(extraction_dir / "policy_calls"),
            "wam_calls_dir": str(extraction_dir / "wam_calls"),
            "provider_output_replay_used": False,
            "raw_credentials_written_to_artifacts": False,
        },
    )
    success_proven = bool(
        live_wam_count > 0
        and learned_wam_count > 0
        and imported.get("manipulation_success_evaluator_result") == "success"
    )
    judge = {
        "schema_version": "manipulation_success_evaluator_results.v1",
        "generated_at": generated_at,
        "status": "completed",
        "question": "Did the sink handle end up turned on?",
        "answer": "not_proven" if not success_proven else "yes",
        "did_sink_handle_end_up_turned_on": bool(success_proven),
        "sink_handle_turned_on_proven": bool(success_proven),
        "success_proof_separate_from_structural_loop_proof": True,
        "structural_loop_completed": imported.get("status") == "completed",
        "live_wam_generation_success_count": live_wam_count,
        "learned_wam_model_success_count": learned_wam_count,
        "structural_fallback_used": structural_wam_count > 0,
        "reason": (
            "The loop completed with structural WAM fallback only; no live learned WAM or physics "
            "state proved a sink-handle state transition."
            if not success_proven
            else "A live evaluator reported sink-handle success."
        ),
        "raw_credentials_written_to_artifacts": False,
    }
    write_json(job / "manipulation_success_evaluator_results.json", judge)
    claim_boundary = {
        "schema_version": "persistent_policy_wam_claim_boundary.v1",
        "generated_at": generated_at,
        "simulator_generated_world_proof_only": True,
        "structural_loop_proof_completed": imported.get("status") == "completed",
        "success_proof_completed": success_proven,
        "local_structural_wam_generator_is_not_live_oscar_or_cosmos_model": structural_wam_count > 0,
        "frame_copy_placeholder_until_live_wam_model_configured": structural_wam_count > 0,
        "wam_evaluator_is_not_robot_policy": True,
        "provider_output_replay_used": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "claim_boundary.json", claim_boundary)
    if not success_proven:
        write_json(
            job / "failure_labels.json",
            {
                "schema_version": "persistent_policy_wam_failure_labels.v1",
                "generated_at": generated_at,
                "status": "completed",
                "labels": [
                    "task_success_not_proven",
                    "live_wam_not_run" if live_wam_count == 0 else "live_wam_success_not_judged",
                    "structural_wam_fallback_only" if structural_wam_count else "wam_generation_missing",
                    "physics_contact_not_validated",
                ],
                "raw_credentials_written_to_artifacts": False,
            },
        )
    video_status = _write_review_video(job=job, extraction_dir=extraction_dir, generated_at=generated_at)
    return {
        "schema_version": "persistent_session_postprocess_artifacts.v1",
        "generated_at": generated_at,
        "status": "completed",
        "policy_action_model_command_discovery": str(job / "policy_action_model_command_discovery.json"),
        "policy_action_model_command_execution": str(job / "policy_action_model_command_execution.json"),
        "policy_action_model_command_output": str(job / "policy_action_model_command_output.json"),
        "wam_generation_command_discovery": str(job / "wam_generation_command_discovery.json"),
        "wam_generation_command_execution": str(job / "wam_generation_command_execution.json"),
        "wam_generation_command_output": str(job / "wam_generation_command_output.json"),
        "robot_policy_wam_loop_manifest": str(job / "robot_policy_wam_loop_manifest.json"),
        "manipulation_success_evaluator_results": str(job / "manipulation_success_evaluator_results.json"),
        "video_review_status": str(job / "video_review_status.json"),
        "review_video_path": _mapping(video_status).get("review_video_path"),
        "claim_boundary": str(job / "claim_boundary.json"),
        "failure_labels": str(job / "failure_labels.json") if (job / "failure_labels.json").is_file() else None,
        "vast_provider_adapter_result_path": str(vast_run_dir / "vast_provider_adapter_result.json"),
        "estimated_cost_usd": vast_result.get("estimated_cost_usd"),
        "raw_credentials_written_to_artifacts": False,
    }


def run_persistent_session(
    *,
    policy_observation_path: str | Path,
    job_dir: str | Path | None = None,
    loop_step_count: int = 12,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool | None = None,
) -> tuple[dict[str, Any], int]:
    generated_at = utc_now_iso()
    job = _job_dir(job_dir)
    allow_fallback = (
        _truthy(os.getenv(PERSISTENT_SESSION_ALLOW_STRUCTURAL_WAM_FALLBACK_ENV))
        if allow_structural_wam_fallback is None
        else bool(allow_structural_wam_fallback)
    )
    inner_policy_command = _string(os.getenv(INNER_POLICY_COMMAND_ENV)) or DEFAULT_INNER_POLICY_COMMAND
    previous_policy_command = os.environ.get("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND")
    previous_persistent_inner_policy_command = os.environ.get(
        PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV
    )
    previous_vast_inner_policy_command = os.environ.get(INNER_POLICY_COMMAND_ENV)
    os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = inner_policy_command
    os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = inner_policy_command
    os.environ[INNER_POLICY_COMMAND_ENV] = inner_policy_command
    try:
        bundle = build_persistent_session_provider_bundle(
            job_dir=job / "provider_bundle",
            policy_observation_path=policy_observation_path,
            loop_step_count=loop_step_count,
            task_prompt=task_prompt,
            timeout_seconds=timeout_seconds,
            use_live_wam=use_live_wam,
            allow_structural_wam_fallback=allow_fallback,
            generated_at=generated_at,
        )
        if bundle.get("status") != "bundle_ready":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=bundle.get("blockers") or ["persistent_session_provider_bundle_blocked"],
                details={"bundle_manifest_path": str(job / "provider_bundle" / "persistent_session_provider_bundle_manifest.json")},
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        bundle_path = Path(str(bundle["bundle_path"])).expanduser().resolve()
        staging = stage_wam_provider_bundle_object_store(
            job_dir=job / "object_store_staging",
            bundle_path=bundle_path,
            key_prefix=_string(os.getenv(OBJECT_STORE_KEY_PREFIX_ENV)) or DEFAULT_OBJECT_STORE_KEY_PREFIX,
            expiration_seconds=_int_env("BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIGNED_URL_SECONDS", 21600),
            generated_at=generated_at,
        )
        if staging.get("status") != "completed":
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=staging.get("blockers") or ["persistent_session_object_store_staging_blocked"],
                details={"object_store_staging_manifest_path": str(job / "object_store_staging" / "wam_provider_object_store_staging_manifest.json")},
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        staging_dir = job / "object_store_staging"
        bundle_url = (staging_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
        output_put_url = (staging_dir / "provider_output_put_url.txt").read_text(encoding="utf-8").strip()
        output_get_url = (staging_dir / "provider_output_get_url.txt").read_text(encoding="utf-8").strip()
        excluded_machine_ids = _machine_ids_from_env(EXCLUDED_MACHINE_ID_ENVS)
        allowed_machine_ids = _machine_ids_from_env(ALLOWED_MACHINE_ID_ENVS)
        machine_avoidlist_path = job / "vast_machine_avoidlist.json"
        if excluded_machine_ids:
            write_json(
                machine_avoidlist_path,
                {
                    "schema_version": "vast_machine_avoidlist.v1",
                    "generated_at": generated_at,
                    "status": "loaded_from_env",
                    "machine_ids": sorted(excluded_machine_ids),
                    "raw_secret_values_recorded": False,
                },
            )

        def run_remote_attempt(run_dir: Path, attempt_allowed_machine_ids: Sequence[int]) -> tuple[dict[str, Any], Path]:
            output_zip = run_dir / "vast_provider_runtime_output.zip"
            result = run_vast_provider_adapter(
                job_dir=run_dir,
                mode="live-startup-probe",
                allow_vast_api_call=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_API_CALLS")),
                allow_instance_launch=_truthy(os.getenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")),
                max_hourly_rate=_float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_HOURLY_RATE", 0.60),
                target_spend_usd=_float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_TARGET_SPEND_USD", 3.0),
                hard_cap_usd=_float_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HARD_CAP_USD", 3.0),
                max_live_minutes=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MAX_LIVE_MINUTES", 55),
                session_max_live_minutes=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_SESSION_MAX_LIVE_MINUTES", 420),
                public_image=(
                    _string(os.getenv(PERSISTENT_SESSION_PUBLIC_IMAGE_ENV))
                    or _string(os.getenv("BLUEPRINT_VAST_WAM_PUBLIC_IMAGE"))
                    or _string(os.getenv(UNITREE_PUBLIC_IMAGE_ENV))
                    or DEFAULT_WAM_PUBLIC_IMAGE
                    or DEFAULT_PUBLIC_CUDA_IMAGE
                ),
                provider_bundle=bundle_path,
                provider_bundle_url=bundle_url,
                provider_output_put_url=output_put_url,
                provider_output_get_url=output_get_url,
                provider_runtime_output_zip=output_zip,
                enable_blueprint_bundle=True,
                provider_bundle_kind="unitree_groot_n17_sonic",
                vast_launch_mode=_string(os.getenv(VAST_LAUNCH_MODE_ENV)) or "ssh_direct",
                ngc_image_login_mode=os.getenv(VAST_IMAGE_LOGIN_MODE_ENV),
                disk_gb=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_DISK_GB", 120),
                min_gpu_ram_mb=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_MIN_GPU_RAM_MB", 48000),
                poll_interval_seconds=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_POLL_SECONDS", 15),
                startup_timeout_seconds=_int_env("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_STARTUP_TIMEOUT_SECONDS", 1800),
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_machine_ids=attempt_allowed_machine_ids,
                verify_staging_urls=True,
            )
            return result, output_zip

        run_dir = job / "vast_persistent_session_run"
        effective_run_dir = run_dir
        vast_result, output_zip = run_remote_attempt(run_dir, allowed_machine_ids)
        fallback_result: dict[str, Any] | None = None
        if (
            allowed_machine_ids
            and _truthy(os.getenv(ALLOW_UNPINNED_FALLBACK_ENV))
            and vast_result.get("status") != "completed"
            and "no_vast_offer_matching_allowed_machine_ids"
            in {str(item) for item in (vast_result.get("blockers") or [])}
        ):
            effective_run_dir = job / "vast_persistent_session_run_unpinned_fallback"
            fallback_result, output_zip = run_remote_attempt(effective_run_dir, [])
            vast_result = fallback_result
        if vast_result.get("status") != "completed" or not output_zip.is_file():
            output = _blocked_payload(
                generated_at=generated_at,
                job_dir=job,
                blockers=vast_result.get("blockers") or ["persistent_session_vast_provider_blocked"],
                details={
                    "vast_provider_adapter_result_path": str(effective_run_dir / "vast_provider_adapter_result.json"),
                    "vast_teardown_manifest_path": str(effective_run_dir / "vast_teardown_manifest.json"),
                    "fallback_vast_provider_adapter_result_path": str(
                        job / "vast_persistent_session_run_unpinned_fallback" / "vast_provider_adapter_result.json"
                    )
                    if fallback_result is not None
                    else None,
                },
            )
            write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
            return output, 2
        extraction_dir = job / "imported_persistent_session_output"
        ensure_dir(extraction_dir)
        with zipfile.ZipFile(output_zip) as archive:
            archive.extractall(extraction_dir)
        imported_path = extraction_dir / "unitree_groot_n17_sonic_wam_persistent_session_output.json"
        if not imported_path.is_file():
            imported_path = extraction_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
        imported = _read_json(imported_path) if imported_path.is_file() else {}
        postprocess = _postprocess_imported_persistent_session_artifacts(
            job=job,
            extraction_dir=extraction_dir,
            imported=imported,
            generated_at=generated_at,
            policy_observation_path=policy_observation_path,
            vast_result=vast_result,
            vast_run_dir=effective_run_dir,
        )
        completed = imported.get("status") == "completed"
        output = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if completed else "blocked",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "job_dir": str(job),
            "persistent_provider_session_used": bool(imported.get("persistent_provider_session_used")),
            "provider_instance_reused_for_policy_and_wam_loop": bool(
                imported.get("provider_instance_reused_for_policy_and_wam_loop")
            ),
            "repeated_policy_calls_count": int(imported.get("repeated_policy_calls_count") or 0),
            "generated_next_observation_count": int(imported.get("generated_next_observation_count") or 0),
            "live_wam_generation_success_count": int(imported.get("live_wam_generation_success_count") or 0),
            "learned_wam_model_success_count": int(imported.get("learned_wam_model_success_count") or 0),
            "unitree_groot_n17_sonic_model_executed": bool(imported.get("unitree_groot_n17_sonic_model_executed")),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                imported.get("unitree_groot_n17_sonic_policy_action_command_ran")
            ),
            "unitree_policy_action_command_ran": bool(imported.get("unitree_policy_action_command_ran")),
            "policy_action_model_command_ran": bool(imported.get("policy_action_model_command_ran")),
            "provider_output_replay_used": bool(imported.get("provider_output_replay_used")),
            "blockers": [] if completed else imported.get("blockers") or ["persistent_session_provider_output_blocked"],
            "imported_provider_output_dir": str(extraction_dir),
            "imported_provider_output_path": str(imported_path) if imported_path.is_file() else None,
            "vast_provider_adapter_result_path": str(effective_run_dir / "vast_provider_adapter_result.json"),
            "vast_teardown_manifest_path": str(effective_run_dir / "vast_teardown_manifest.json"),
            "estimated_cost_usd": vast_result.get("estimated_cost_usd"),
            "postprocess_artifacts": postprocess,
            "review_video_path": postprocess.get("review_video_path"),
            "video_review_status_path": postprocess.get("video_review_status"),
            "manipulation_success_evaluator_results_path": postprocess.get(
                "manipulation_success_evaluator_results"
            ),
            "claim_boundary_path": postprocess.get("claim_boundary"),
            "claim_boundary": {
                "simulator_generated_world_proof_only": True,
                "persistent_provider_session_is_runtime_proof_not_task_success": True,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        write_json(job / "unitree_groot_n17_sonic_vast_persistent_session_result.json", output)
        return output, 0 if completed else 2
    finally:
        if previous_policy_command is None:
            os.environ.pop("BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND", None)
        else:
            os.environ["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] = previous_policy_command
        if previous_persistent_inner_policy_command is None:
            os.environ.pop(PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV, None)
        else:
            os.environ[PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV] = (
                previous_persistent_inner_policy_command
            )
        if previous_vast_inner_policy_command is None:
            os.environ.pop(INNER_POLICY_COMMAND_ENV, None)
        else:
            os.environ[INNER_POLICY_COMMAND_ENV] = previous_vast_inner_policy_command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-observation", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--loop-step-count", type=int, default=12)
    parser.add_argument("--task-prompt")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--disable-live-wam", action="store_true")
    parser.add_argument("--allow-structural-wam-fallback", action="store_true")
    args = parser.parse_args(argv)
    result, exit_code = run_persistent_session(
        policy_observation_path=args.policy_observation,
        job_dir=args.job_dir,
        loop_step_count=args.loop_step_count,
        task_prompt=args.task_prompt,
        timeout_seconds=args.timeout_seconds,
        use_live_wam=not args.disable_live_wam,
        allow_structural_wam_fallback=args.allow_structural_wam_fallback,
    )
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
