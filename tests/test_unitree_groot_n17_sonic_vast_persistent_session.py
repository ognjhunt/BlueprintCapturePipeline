from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline import unitree_groot_n17_sonic_vast_persistent_session as session
from blueprint_pipeline import persistent_wam_short_visual_sanity as short_sanity


def _policy_observation(path: Path, frame: Path) -> Path:
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "visual_observation": {"camera_frame_path": str(frame)},
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
        "unitree_g1_sonic_state_source": "test_contract_probe",
    }
    path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    return path


def _synthetic_policy_observation(path: Path, frame: Path) -> Path:
    observation = {
        "schema_version": "selected_initial_policy_observation.v1",
        "status": "ready",
        "source_kind": "synthetic_fallback",
        "selection_source_kind": "synthetic_fallback",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(frame),
            "source_kind": "synthetic_fallback",
            "synthetic_fallback": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
        },
        "provenance": {
            "source_kind": "synthetic_fallback",
            "synthetic_fallback": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
        },
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
        "unitree_g1_sonic_state_source": "synthetic_fallback_contract_probe",
        "claim_boundary": {
            "selected_synthetic_fallback": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
        },
    }
    path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    return path


def _write_reviewable_frame(path: Path, *, size: tuple[int, int] = (640, 480)) -> Path:
    width, height = size
    gradient = np.tile(np.linspace(55, 215, width, dtype=np.uint8), (height, 1))
    frame = np.dstack((gradient, np.roll(gradient, 40, axis=1), np.flipud(gradient)))
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (width // 2 - 70, height // 2 - 50, width // 2 + 70, height // 2 + 50),
        outline=(255, 255, 255),
        width=5,
    )
    draw.ellipse(
        (width // 2 - 22, height // 2 - 22, width // 2 + 22, height // 2 + 22), fill=(235, 80, 50)
    )
    for x in range(0, width, 32):
        draw.line((x, 0, x, height), fill=(20, 20, 20), width=1)
    for y in range(0, height, 32):
        draw.line((0, y, width, y), fill=(235, 235, 235), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_dark_frame(path: Path, *, size: tuple[int, int] = (640, 480)) -> Path:
    image = Image.new("RGB", size, (8, 8, 8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(24, 24, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_fake_image_model_remediator(path: Path) -> Path:
    path.write_text(
        """
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

request_path = Path(os.environ["BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_REQUEST_PATH"])
output_dir = Path(os.environ["BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_OUTPUT_DIR"])
response_path = Path(os.environ["BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_RESPONSE_PATH"])
request = json.loads(request_path.read_text(encoding="utf-8"))
width, height = 640, 480
gradient = np.tile(np.linspace(48, 226, width, dtype=np.uint8), (height, 1))
frame = np.dstack((gradient, np.roll(gradient, 72, axis=1), np.flipud(gradient)))
image = Image.fromarray(frame, mode="RGB")
draw = ImageDraw.Draw(image)
draw.rectangle((width // 2 - 82, height // 2 - 58, width // 2 + 82, height // 2 + 58), outline=(255, 255, 255), width=6)
draw.ellipse((width // 2 - 26, height // 2 - 26, width // 2 + 26, height // 2 + 26), fill=(232, 72, 46))
for x in range(0, width, 24):
    draw.line((x, 0, x, height), fill=(24, 24, 24), width=1)
for y in range(0, height, 24):
    draw.line((0, y, width, y), fill=(238, 238, 238), width=1)
enhanced = output_dir / "fake_enhanced.png"
image.save(enhanced)
response_path.write_text(
    json.dumps(
        {
            "status": "completed",
            "provider": "unit_test_fake_image_model",
            "model": request.get("model", "gpt-image-2"),
            "enhanced_image_path": str(enhanced),
        },
        indent=2,
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_passed_short_sanity_manifest(root: Path, observation_path: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_qa = root / "source_policy_observation_visual_qa.json"
    report = root / "wam_rollout_visual_quality_report.json"
    contact_sheet = _write_reviewable_frame(root / "wam_rollout_contact_sheet.jpg")
    video_status = root / "video_review_status.json"
    review_video = root / "review.mp4"
    source_qa.write_text(
        json.dumps({"status": "passed_visual_quality_gate"}),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "status": "passed_visual_quality_gate",
                "visual_profile": "review_quality",
                "visual_success": True,
                "profile_contract": {
                    "review_quality_profile": True,
                    "review_quality_minimum_satisfied": True,
                    "smoke_only": False,
                },
            }
        ),
        encoding="utf-8",
    )
    video_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": {
                    "streams": [
                        {
                            "width": 640,
                            "height": 480,
                            "avg_frame_rate": "15/1",
                            "nb_frames": "24",
                        }
                    ],
                    "format": {"duration": "1.6", "size": "1000"},
                },
            }
        ),
        encoding="utf-8",
    )
    review_video.write_bytes(b"mp4")
    manifest = root / "persistent_wam_short_visual_sanity_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION,
                "generated_at": "now",
                "status": "passed_short_visual_sanity",
                "short_visual_sanity_passed": True,
                "policy_observation_path": str(observation_path.resolve()),
                "provider": "runpod",
                "requested_transition_count": 2,
                "requested_loop_step_count": 3,
                "generated_transition_count": 2,
                "visual_profile": "review_quality",
                "source_policy_observation_visual_qa_status": "passed_visual_quality_gate",
                "source_policy_observation_visual_qa_path": str(source_qa),
                "wam_rollout_visual_success": True,
                "wam_rollout_visual_quality_report_path": str(report),
                "wam_rollout_contact_sheet_path": str(contact_sheet),
                "video_review_status_path": str(video_status),
                "review_video_path": str(review_video),
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": {
                    "streams": [
                        {
                            "width": 640,
                            "height": 480,
                            "avg_frame_rate": "15/1",
                            "nb_frames": "24",
                        }
                    ],
                    "format": {"duration": "1.6", "size": "1000"},
                },
                "live_wam_generation_success_count": 2,
                "learned_wam_model_success_count": 2,
                "structural_fallback_used": False,
                "paid_provider": {
                    "provider": "runpod",
                    "used": False,
                    "teardown_status": "not_required_no_paid_provider",
                    "teardown_performed": False,
                    "continuing_spend_from_this_run": False,
                },
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _python_heredoc_chunks(script: str) -> list[str]:
    chunks: list[str] = []
    lines = script.splitlines()
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        start = index + 1
        end = start
        while end < len(lines) and lines[end] != "PY":
            end += 1
        chunks.append("\n".join(lines[start:end]) + "\n")
        index = end + 1
    return chunks


def test_persistent_session_bundle_uses_proven_policy_server_rewrite(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["loop_step_count"] == 12
    assert manifest["allow_structural_wam_fallback"] is True
    bundle_path = Path(str(manifest["bundle_path"]))
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        runner = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py"
        ).decode()
        runpod_wrapper = archive.read(
            "provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh"
        ).decode()
        run_script = archive.read(
            "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh"
        ).decode()
        wam_carrier = archive.read("provider_runtime/run_wam_provider_runtime.sh").decode()
        provider_smoke = archive.read(
            "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_provider_smoke.py"
        ).decode()
        session_input = json.loads(archive.read("provider_runtime/persistent_session_input.json"))
        runtime_auxiliary = json.loads(
            archive.read(
                "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_manifest.json"
            )
        )

    assert "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" in names
    assert "provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh" in names
    assert "provider_runtime/run_wam_provider_runtime.sh" in names
    assert "provider_runtime/unitree_groot_n17_sonic_provider_runner.py" in names
    assert "provider_runtime/policy_input.json" in names
    assert "provider_runtime/input_frame.png" in names
    assert (
        "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_manifest.json"
        in names
    )
    assert (
        "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_claim_boundary.json"
        in names
    )
    assert "provider_instance_reused_for_policy_and_wam_loop" in runner
    assert "bootstrap_venv_policy_server_client_for_persistent_session" in runner
    assert "persistent_policy_worker_command_uses_policy_server_client" in runner
    assert "not self.use_live_wam" in runner
    assert "shlex.quote(str(venv_python))" in runner
    assert "_http_post_json_with_retries" in runner
    assert "loop_step_count = max(1" in runner
    assert "required_wam_transition_count" in runner
    assert "persistent_wam_worker_runtime_stdout_stderr.log" in runner
    assert "persistent_wam_worker_oscar_runtime_timeout" in runner
    assert "persistent_wam_worker_oscar_runtime_started" in runner
    assert "persistent_wam_worker_oscar_runtime_waiting" in runner
    assert "subprocess.Popen" in runner
    assert "start_new_session=True" in runner
    assert "proc.poll()" in runner
    assert "timeout_deadline" in runner
    assert "os.killpg(process_group_id or os.getpgid(proc.pid), signal.SIGTERM)" in runner
    assert "os.killpg(process_group_id or os.getpgid(proc.pid), signal.SIGKILL)" in runner
    assert "process_group_id" in runner
    assert "process_group_terminated" in runner
    assert "process_group_killed" in runner
    assert "stdout_stderr_streamed_to_log" in runner
    assert "_upload_phase_heartbeat(payload)" in runner
    assert "BLUEPRINT_PERSISTENT_SESSION_PHASE_HEARTBEAT_UPLOAD_OK" in runner
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS" in runner
    assert '"_blueprint_outer_phase_callback": _phase' in session.PERSISTENT_SESSION_RUNNER
    assert "_blueprint_outer_phase_callback" in provider_smoke
    assert "gr00t_model_snapshot_completed" in provider_smoke
    assert "gr00t_policy_server_process_started" in provider_smoke
    module_source = Path(str(session.__file__)).read_text(encoding="utf-8")
    assert 'or "wam"' in module_source
    assert (
        "runpod_unitree_groot_sonic_bundle_wrapper_exited_before_runtime_result" in runpod_wrapper
    )
    assert "runpod_unitree_groot_sonic_remote_heartbeat" in runpod_wrapper
    assert (
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_BOOTSTRAP_HEARTBEAT:-true"
        in runpod_wrapper
    )
    assert "run_unitree_groot_n17_sonic_runpod_wrapper.sh" in wam_carrier
    assert "BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR" in wam_carrier
    assert "os.walk(output_dir)" in runpod_wrapper
    assert "dirs[:] = sorted(item for item in dirs if item not in excluded_dirs)" in runpod_wrapper
    assert '"checkpoints"' in runpod_wrapper
    assert "zipfile.is_zipfile(zip_path)" in runpod_wrapper
    assert "invalid_or_empty_runtime_output_zip" in runpod_wrapper
    assert "runpod_runtime_output_zip_creation_failed" in runpod_wrapper
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS" in runpod_wrapper
    assert "write_unitree_groot_sonic_phase_heartbeat" in runpod_wrapper
    assert "runpod_system_dependency_check_started" in runpod_wrapper
    assert "runpod_entrypoint_subprocess_starting" in runpod_wrapper
    assert "runpod_entrypoint_subprocess_running" in runpod_wrapper
    assert "entrypoint_log_tail" in runpod_wrapper
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_HEARTBEAT_SECONDS" in runpod_wrapper
    assert "blueprint_phase_heartbeat" in run_script
    assert "runpod_entrypoint_dependency_probe_started" in run_script
    assert "runpod_entrypoint_runner_starting" in run_script
    assert 'json.dumps(payload, indent=2, sort_keys=True) + "\\n"' in run_script
    for script_name, script_text in {
        "runpod_wrapper": runpod_wrapper,
        "run_script": run_script,
        "wam_carrier": wam_carrier,
    }.items():
        for index, chunk in enumerate(_python_heredoc_chunks(script_text)):
            compile(chunk, f"<{script_name}:heredoc:{index}>", "exec")
    assert session_input["loop_step_count"] == 12
    assert session_input["use_live_wam"] is False
    assert session_input["allow_structural_wam_fallback"] is True
    assert session_input["wam_auxiliary_observation"]["status"] == "completed"
    assert (
        session_input["initial_observation"]["wam_auxiliary_observation"]["modalities_available"][
            "proprioception"
        ]
        is True
    )
    assert runtime_auxiliary["schema_version"] == "wam_auxiliary_observation_manifest.v1"
    assert runtime_auxiliary["source_image_path"] == "provider_runtime/initial_policy_frame.png"
    assert str(tmp_path) not in json.dumps(runtime_auxiliary)
    assert runtime_auxiliary["claim_boundary"]["collision_truth"] is False
    assert (
        runtime_auxiliary["oscar_conditioning_support"][
            "raw_aux_modalities_consumed_by_public_oscar_entrypoint"
        ]
        is False
    )


def test_synthetic_fallback_cannot_build_live_wam_bundle_without_experimental_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "synthetic.jpg")
    observation_path = _synthetic_policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.delenv(session.SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV, raising=False)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert (
        "synthetic_fallback_live_or_review_wam_launch_requires_experimental_env"
        in manifest["blockers"]
    )
    gate = manifest["synthetic_fallback_wam_launch_gate"]
    assert gate["synthetic_fallback_initial_observation_used"] is True
    assert gate["experimental_env_enabled"] is False
    assert gate["capture_truth"] is False
    assert gate["geometry_truth"] is False
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert manifest["claim_boundary"]["visually_useful_rollout"] is False
    assert (
        manifest["claim_boundary"]["provider_success_separate_from_visually_useful_rollout"]
        is True
    )
    assert Path(manifest["bundle_path"]).is_file() is False


def test_synthetic_fallback_review_wam_bundle_requires_and_records_experimental_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "synthetic.jpg")
    observation_path = _synthetic_policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(session.SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV, "true")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=False,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["blockers"] == []
    gate = manifest["synthetic_fallback_wam_launch_gate"]
    assert gate["launch_path_requires_gate"] is True
    assert gate["experimental_env"] == session.SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV
    assert gate["experimental_env_enabled"] is True
    assert gate["capture_truth"] is False
    assert gate["geometry_truth"] is False
    assert manifest["claim_boundary"]["synthetic_fallback_initial_observation_used"] is True
    assert manifest["claim_boundary"]["synthetic_fallback_wam_launch_experiment_enabled"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert manifest["claim_boundary"]["visually_useful_rollout"] is False
    with zipfile.ZipFile(Path(manifest["bundle_path"])) as archive:
        session_input = json.loads(archive.read("provider_runtime/persistent_session_input.json"))
        policy_input = json.loads(archive.read("provider_runtime/policy_input.json"))
    initial_observation = session_input["initial_observation"]
    assert initial_observation["claim_boundary"]["capture_truth"] is False
    assert initial_observation["claim_boundary"]["geometry_truth"] is False
    assert (
        initial_observation["claim_boundary"][
            "synthetic_fallback_wam_launch_experiment_enabled"
        ]
        is True
    )
    assert policy_input["observation"]["visual_observation"]["capture_truth"] is False
    assert policy_input["observation"]["visual_observation"]["geometry_truth"] is False
    assert (
        session_input["claim_boundary"]["provider_success_separate_from_visually_useful_rollout"]
        is True
    )
    assert session_input["claim_boundary"]["visually_useful_rollout"] is False


def test_persistent_session_runner_phase_heartbeat_helper_is_self_contained(
    tmp_path: Path,
    monkeypatch,
) -> None:
    namespace: dict[str, object] = {
        "__name__": "blueprint_test_persistent_session_runner",
        "__file__": str(tmp_path / "unitree_groot_n17_sonic_wam_persistent_session_runner.py"),
    }
    exec(session.PERSISTENT_SESSION_RUNNER, namespace)
    output_dir = tmp_path / "runtime_output"
    output_path = output_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
    uploads: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        uploads["url"] = request.full_url
        uploads["data"] = request.data
        uploads["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(namespace["urllib_request"], "urlopen", fake_urlopen)  # type: ignore[index]
    monkeypatch.setenv("OUTPUT_PUT_URL", "https://upload.example/provider-output.zip")
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS", "true")

    namespace["_upload_phase_heartbeat"]({"phase": "bootstrap_policy_server_started"})  # type: ignore[index,operator]

    assert uploads["url"] == "https://upload.example/provider-output.zip"
    assert uploads["timeout"] == 20
    assert output_path.is_file()
    heartbeat = json.loads(output_path.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "running"
    assert heartbeat["runtime_phase"] == "bootstrap_policy_server_started"
    assert zipfile.is_zipfile(tmp_path / "unitree_groot_n17_sonic_provider_phase_heartbeat.zip")


def test_run_persistent_session_imports_reused_worker_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    for env_name in session.ALLOWED_MACHINE_ID_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    for env_name in session.EXCLUDED_MACHINE_ID_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_vast(**kwargs):
        captured.update(kwargs)
        captured["policy_command_env"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
        )
        captured["persistent_inner_policy_command_env"] = session.os.environ.get(
            session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV
        )
        captured["vast_inner_policy_command_env"] = session.os.environ.get(
            session.INNER_POLICY_COMMAND_ENV
        )
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                "unitree_groot_n17_sonic_wam_persistent_session_output.json",
                json.dumps(
                    {
                        "schema_version": session.OUTPUT_SCHEMA_VERSION,
                        "status": "completed",
                        "persistent_provider_session_used": True,
                        "provider_instance_reused_for_policy_and_wam_loop": True,
                        "repeated_policy_calls_count": 12,
                        "generated_next_observation_count": 11,
                        "live_wam_generation_success_count": 0,
                        "learned_wam_model_success_count": 0,
                        "unitree_groot_n17_sonic_model_executed": True,
                        "unitree_groot_n17_sonic_policy_action_command_ran": True,
                        "unitree_policy_action_command_ran": True,
                        "policy_action_model_command_ran": True,
                        "provider_output_replay_used": False,
                        "blockers": [],
                    }
                ),
            )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.01}

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "run_vast_provider_adapter", fake_vast)

    output, exit_code = session.run_persistent_session(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=12,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
    )

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["persistent_provider_session_used"] is True
    assert output["provider_instance_reused_for_policy_and_wam_loop"] is True
    assert output["repeated_policy_calls_count"] == 12
    assert output["generated_next_observation_count"] == 11
    assert output["provider_output_replay_used"] is False
    assert captured["provider_bundle_kind"] == "unitree_groot_n17_sonic"
    assert captured["enable_blueprint_bundle"] is True
    assert captured["allowed_machine_ids"] == []
    assert captured["policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND
    assert captured["persistent_inner_policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND
    assert captured["vast_inner_policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND


def test_postprocess_live_wam_not_task_success_labels_are_consistent(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    extraction_dir = tmp_path / "extracted"
    policy_calls_dir = extraction_dir / "policy_calls"
    policy_calls_dir.mkdir(parents=True)
    (extraction_dir / "wam_calls").mkdir()
    (policy_calls_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 0,
                "action": {"action": "turn_sink_handle"},
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "wam_generated_next_observations.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "structural_fallback_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (extraction_dir / "robot_policy_wam_loop_trace.jsonl").write_text("", encoding="utf-8")
    (extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()

    session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=tmp_path / "observation.json",
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    labels = json.loads((job / "failure_labels.json").read_text(encoding="utf-8"))
    assert "live_wam_success_not_task_success_proof" in labels["labels"]
    assert "wam_generation_missing" not in labels["labels"]
    assert "structural_wam_fallback_only" not in labels["labels"]

    judge = json.loads(
        (job / "manipulation_success_evaluator_results.json").read_text(encoding="utf-8")
    )
    assert judge["answer"] == "not_proven"
    assert judge["live_wam_generation_success_count"] == 1
    assert judge["structural_fallback_used"] is False
    assert "live learned WAM generations" in judge["reason"]
    assert "structural WAM fallback only" not in judge["reason"]


def test_vast_probe_env_forwards_persistent_inner_policy_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV,
        session.DEFAULT_INNER_POLICY_COMMAND,
    )
    monkeypatch.setenv(session.INNER_POLICY_COMMAND_ENV, session.DEFAULT_INNER_POLICY_COMMAND)

    env = vast_provider_adapter._probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
    )

    assert (
        env[session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV]
        == session.DEFAULT_INNER_POLICY_COMMAND
    )
    assert env[session.INNER_POLICY_COMMAND_ENV] == session.DEFAULT_INNER_POLICY_COMMAND


def test_runpod_persistent_session_defaults_to_wam_carrier_and_wait_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.delenv(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_PROVIDER_BUNDLE_KIND", raising=False
    )
    monkeypatch.setenv(
        session.PERSISTENT_SESSION_PUBLIC_IMAGE_ENV,
        "docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1",
    )
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_WAM_PUBLIC_IMAGE",
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
    )
    monkeypatch.setenv(
        "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_PUBLIC_IMAGE",
        "docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1",
    )
    monkeypatch.setenv(
        "BLUEPRINT_VAST_WAM_PUBLIC_IMAGE",
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
    )
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS", "180")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WAIT_BUFFER_SECONDS", "30")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["provider_bundle_kind"] = kwargs["provider_bundle_kind"]
        captured["image_name"] = kwargs["image_name"]
        captured["container_disk_gb"] = kwargs["container_disk_gb"]
        captured["volume_gb"] = kwargs["volume_gb"]
        captured["wam_carrier_enabled"] = session.os.environ.get(
            "BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"
        )
        captured["wam_visual_profile"] = session.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
        )
        captured["wam_num_steps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS")
        captured["wam_num_frames"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        captured["wam_height"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT")
        captured["wam_width"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH")
        captured["wam_fps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_FPS")
        captured["wam_checkpoint_timeout"] = session.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS"
        )
        captured["groot_bootstrap_mode"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE"
        )
        captured["groot_sparse_checkout"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT"
        )
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        captured["max_wait_seconds"] = kwargs["max_wait_seconds"]
        runpod_dir = Path(kwargs["job_dir"])
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "output_zip_present": False,
                    "provider_command_status": "blocked",
                    "provider_command_blockers": [
                        "runpod_provider_runtime_output_zip_not_received_locally"
                    ],
                    "teardown_performed": True,
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=2,
        timeout_seconds=60,
        max_wait_seconds=20,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert captured["provider_bundle_kind"] == "wam"
    assert captured["image_name"] == "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
    assert captured["wam_carrier_enabled"] == "true"
    assert captured["wam_visual_profile"] == "smoke"
    assert captured["wam_num_steps"] == "2"
    assert captured["wam_num_frames"] == "9"
    assert captured["wam_height"] == "128"
    assert captured["wam_width"] == "128"
    assert captured["wam_fps"] == "4"
    assert captured["wam_checkpoint_timeout"] == "1200"
    assert captured["groot_bootstrap_mode"] == "system_python_minimal"
    assert captured["groot_sparse_checkout"] == "true"
    assert captured["container_disk_gb"] == 240
    assert captured["volume_gb"] == 120
    assert captured["max_wait_seconds"] == 210


def test_runpod_persistent_session_review_quality_profile_uses_higher_fidelity_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["wam_visual_profile"] = session.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
        )
        captured["wam_num_frames"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        captured["wam_height"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT")
        captured["wam_width"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH")
        captured["wam_fps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_FPS")
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        runpod_dir = Path(kwargs["job_dir"])
        poll_manifest = {
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(poll_manifest),
            encoding="utf-8",
        )
        return poll_manifest

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=2,
        timeout_seconds=60,
        max_wait_seconds=20,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert captured["wam_visual_profile"] == "review_quality"
    assert captured["wam_num_frames"] == "24"
    assert captured["wam_height"] == "480"
    assert captured["wam_width"] == "640"
    assert captured["wam_fps"] == "15"


def test_review_quality_profile_rejects_128px_bundle_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "9")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_HEIGHT", "128")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_WIDTH", "128")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_FPS", "4")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "review_quality_profile_width_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_height_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_fps_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_num_frames_below_minimum" in manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_qa["status"] == "passed_visual_quality_gate"


def test_review_quality_profile_rejects_bad_source_frame_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "source_policy_observation_visual_qa_failed_for_review_quality" in manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert "source_policy_observation_too_dark_for_review" in source_qa["blockers"]


def test_persistent_session_bundle_blocks_offscreen_semantic_initial_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    target = {
        "object_id": "Sink054_handle",
        "label": "right sink handle",
        "bbox": {"x": 700, "y": 180, "width": 120, "height": 90},
        "keypoints": {"center": [760, 225]},
        "confidence": 0.95,
        "occlusion": "visible",
    }
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {"camera_frame_path": str(frame)},
        "object_index": {"objects": [target]},
        "eval_ready_task_grounding": {
            "schema_version": "eval_ready_task_grounding.v1",
            "status": "ready_for_learned_wam_rollout_request",
            "selected_task_target": target,
            "readiness": {
                "learned_rollout_request_ready": True,
                "target_crop_available": False,
                "target_mask_or_keypoint_available": True,
                "blockers": [],
            },
        },
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "source_policy_observation_visual_qa_failed_for_review_quality" in manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_qa["target_visibility_status"] == "failed_semantic_gate"
    assert "target_object_offscreen_in_source_observation" in source_qa["blockers"]


def test_review_quality_profile_remediates_bad_source_frame_with_image_model_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    command = _write_fake_image_model_remediator(tmp_path / "fake_remediate.py")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_ALLOW_IMAGE_MODEL_RENDER_REMEDIATION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_COMMAND",
        f"{sys.executable} {command}",
    )

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["image_model_render_remediation_applied"] is True
    assert manifest["image_model_render_remediation_status"] == "completed"
    assert manifest["original_initial_frame_path"] == str(frame.resolve())
    assert manifest["initial_frame_path"].endswith("enhanced_initial_policy_frame.png")
    assert not manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    original_qa = json.loads(
        (tmp_path / "bundle" / "original_source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    remediation = json.loads(
        Path(manifest["image_model_render_remediation_manifest_path"]).read_text(encoding="utf-8")
    )
    policy_input = json.loads(
        (tmp_path / "bundle" / "provider_runtime" / "policy_input.json").read_text(encoding="utf-8")
    )

    assert source_qa["status"] == "passed_visual_quality_gate"
    assert original_qa["status"] == "failed_visual_quality_gate"
    assert remediation["claim_boundary"]["capture_truth"] is False
    assert remediation["claim_boundary"]["geometry_truth"] is False
    assert remediation["claim_boundary"]["collision_truth"] is False
    assert remediation["claim_boundary"]["near_preserving_image_enhancement_only"] is True
    assert (
        tmp_path / "bundle" / "image_model_render_remediation" / "original_initial_policy_frame.jpg"
    ).is_file()
    assert (
        tmp_path / "bundle" / "image_model_render_remediation" / "enhanced_initial_policy_frame.png"
    ).is_file()
    assert (
        tmp_path
        / "bundle"
        / "provider_runtime"
        / "image_model_render_remediation"
        / "enhanced_initial_policy_frame.png"
    ).is_file()
    with Image.open(tmp_path / "bundle" / "provider_runtime" / "initial_policy_frame.png") as image:
        assert image.size == (640, 480)
        assert np.asarray(image).mean() > 80.0
    observation = policy_input["observation"]
    assert observation["source_kind"] == "image_model_enhanced_3d_render_seed"
    assert observation["claim_boundary"]["capture_truth"] is False
    assert observation["visual_observation"]["original_3d_render_frame_path"] == str(
        frame.resolve()
    )
    with zipfile.ZipFile(Path(manifest["bundle_path"])) as archive:
        names = set(archive.namelist())
    assert (
        "provider_runtime/image_model_render_remediation/enhanced_initial_policy_frame.png" in names
    )


def test_review_quality_profile_records_blocked_image_model_remediation_without_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_ALLOW_IMAGE_MODEL_RENDER_REMEDIATION", "true")
    monkeypatch.delenv("BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_COMMAND", raising=False)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["image_model_render_remediation_applied"] is False
    assert manifest["image_model_render_remediation_status"] == "blocked"
    assert "image_model_render_remediation_command_not_configured" in manifest["blockers"]
    assert "source_policy_observation_visual_qa_failed_for_review_quality" in manifest["blockers"]
    remediation = json.loads(
        Path(manifest["image_model_render_remediation_manifest_path"]).read_text(encoding="utf-8")
    )
    assert remediation["status"] == "blocked"
    assert "image_model_render_remediation_command_not_configured" in remediation["blockers"]
    assert remediation["claim_boundary"]["capture_truth"] is False
    assert (
        tmp_path / "bundle" / "image_model_render_remediation" / "original_initial_policy_frame.jpg"
    ).is_file()


def test_review_quality_long_rollout_requires_passed_short_visual_sanity_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(session.PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_ENV, "true")
    monkeypatch.delenv(session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV, raising=False)

    blocked = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked["status"] == "blocked"
    assert "review_quality_long_rollout_requires_passed_short_visual_sanity" in blocked["blockers"]
    assert "short_visual_sanity_manifest_env_missing" in blocked["blockers"]
    assert (
        "review_quality_long_rollout_env_override_requires_short_visual_sanity_manifest"
        in blocked["blockers"]
    )

    sanity_manifest = _write_passed_short_sanity_manifest(
        tmp_path / "short-sanity",
        observation_path,
    )
    structural_fallback_manifest = tmp_path / "short-sanity" / "structural-fallback.json"
    structural_payload = json.loads(sanity_manifest.read_text(encoding="utf-8"))
    structural_payload["structural_fallback_used"] = True
    structural_fallback_manifest.write_text(json.dumps(structural_payload), encoding="utf-8")
    structural_validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        structural_fallback_manifest,
        policy_observation_path=observation_path,
    )
    assert structural_validation["status"] == "blocked"
    assert (
        "short_visual_sanity_structural_fallback_cannot_unlock_long_rollout"
        in structural_validation["blockers"]
    )

    monkeypatch.setenv(
        session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV, str(sanity_manifest)
    )
    blocked_without_strategy = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-without-strategy-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked_without_strategy["status"] == "blocked"
    assert (
        "review_quality_12_step_paid_rollout_requires_clean_frame_reanchoring_or_drift_blocker"
        in blocked_without_strategy["blockers"]
    )
    quality_gate = json.loads(
        (
            tmp_path
            / "blocked-without-strategy-bundle"
            / "long_review_rollout_quality_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert quality_gate["status"] == "blocked_missing_long_rollout_quality_proof"
    assert quality_gate["paid_rollout_launch_allowed"] is False

    monkeypatch.setenv(session.PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV, "2")
    allowed = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "allowed-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert allowed["status"] == "bundle_ready"
    assert (
        "review_quality_long_rollout_requires_passed_short_visual_sanity" not in allowed["blockers"]
    )
    allowed_gate = json.loads(
        (tmp_path / "allowed-bundle" / "long_review_rollout_quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert allowed_gate["status"] == "passed_periodic_clean_frame_reanchoring"
    assert allowed_gate["paid_rollout_launch_allowed"] is True
    assert allowed_gate["clean_frame_reanchoring"]["interval_steps"] == 2
    runtime_input = json.loads(
        (
            tmp_path / "allowed-bundle" / "provider_runtime" / "persistent_session_input.json"
        ).read_text(encoding="utf-8")
    )
    assert runtime_input["clean_frame_reanchoring"]["periodic_clean_frame_reanchoring_proven"] is True
    assert runtime_input["clean_frame_reanchoring"]["expected_reanchor_transition_indices"] == [
        2,
        4,
        6,
        8,
        10,
    ]


def test_review_quality_12_step_rollout_blocks_on_concrete_drift_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    sanity_manifest = _write_passed_short_sanity_manifest(
        tmp_path / "short-sanity",
        observation_path,
    )
    drift_report = tmp_path / "prior_wam_rollout_visual_quality_report.json"
    drift_report.write_text(
        json.dumps(
            {
                "schema_version": "persistent_policy_wam_visual_quality_report.v1",
                "status": "failed_visual_quality_gate",
                "visual_success": False,
                "visual_profile": "review_quality",
                "generated_frame_count": 11,
                "autoregressive_chain_guard": {
                    "autoregressive_chain_used": True,
                    "generated_frame_count": 11,
                    "long_horizon_visual_drift_blocker": True,
                    "long_rollout_should_not_be_overclaimed": True,
                },
                "blockers": [
                    "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout"
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(
        session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV,
        str(sanity_manifest),
    )
    monkeypatch.setenv(
        session.PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST_ENV,
        str(drift_report),
    )
    monkeypatch.delenv(session.PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV, raising=False)

    blocked = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-drift-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked["status"] == "blocked"
    assert (
        "autoregressive_chain_drift_blocker_present_before_12_step_paid_rollout"
        in blocked["blockers"]
    )
    gate = json.loads(
        (tmp_path / "blocked-drift-bundle" / "long_review_rollout_quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["status"] == "blocked_autoregressive_drift_confirmed"
    assert gate["concrete_autoregressive_drift_blocker_proven"] is True
    assert gate["paid_rollout_launch_allowed"] is False


def test_short_visual_sanity_blocks_before_provider_when_source_qa_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "dark.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    called = False

    def fake_runpod(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider runner should not start after source QA failure")

    monkeypatch.setattr(short_sanity, "run_persistent_session_runpod", fake_runpod)

    manifest, exit_code = short_sanity.run_short_visual_sanity(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "short",
        provider="runpod",
        transition_count=2,
    )

    assert exit_code == 2
    assert called is False
    assert manifest["status"] == "blocked"
    assert manifest["paid_provider"]["used"] is False
    assert manifest["paid_provider"]["teardown_status"] == "not_required_prelaunch_blocked"
    assert manifest["provider_success"] is False
    assert manifest["visually_useful_rollout"] is False
    assert manifest["claim_boundary"]["provider_success_separate_from_visually_useful_rollout"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert "source_policy_observation_too_dark_for_review" in manifest["blockers"]
    assert Path(manifest["short_visual_sanity_manifest_path"]).is_file()


def test_short_visual_sanity_pass_manifest_records_review_artifacts_and_teardown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    captured: dict[str, object] = {}

    def fake_runpod(**kwargs):
        captured["loop_step_count"] = kwargs["loop_step_count"]
        captured["visual_profile"] = short_sanity.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
        )
        captured["num_frames"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        captured["height"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT")
        captured["width"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH")
        captured["fps"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_FPS")
        job = Path(kwargs["job_dir"]) / "short-run"
        job.mkdir(parents=True)
        runpod_dir = job / "runpod_persistent_session_run"
        runpod_dir.mkdir()
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created", "pod_id": "pod-123"}),
            encoding="utf-8",
        )
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "pod_id": "pod-123",
                    "teardown_performed": True,
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        (runpod_dir / "runpod_wam_async_delete_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "pod_id": "pod-123",
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        source_qa = job / "source_policy_observation_visual_qa.json"
        report = job / "wam_rollout_visual_quality_report.json"
        contact_sheet = _write_reviewable_frame(job / "wam_rollout_contact_sheet.jpg")
        frame_stats = job / "wam_rollout_frame_stats.jsonl"
        video_status = job / "video_review_status.json"
        review_video = job / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
        review_video.parent.mkdir()
        source_qa.write_text(
            json.dumps({"status": "passed_visual_quality_gate"}),
            encoding="utf-8",
        )
        report.write_text(
            json.dumps(
                {
                    "status": "passed_visual_quality_gate",
                    "visual_profile": "review_quality",
                    "visual_success": True,
                    "profile_contract": {
                        "review_quality_profile": True,
                        "review_quality_minimum_satisfied": True,
                        "smoke_only": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        frame_stats.write_text("{}\n", encoding="utf-8")
        video_status.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "ffprobe_command_ran": True,
                    "ffprobe_returncode": 0,
                    "ffprobe_metadata": {
                        "streams": [
                            {
                                "width": 640,
                                "height": 480,
                                "avg_frame_rate": "15/1",
                                "nb_frames": "24",
                            }
                        ],
                        "format": {"duration": "1.6", "size": "1000"},
                    },
                }
            ),
            encoding="utf-8",
        )
        review_video.write_bytes(b"mp4")
        result = {
            "status": "completed",
            "job_dir": str(job),
            "generated_next_observation_count": 2,
            "live_wam_generation_success_count": 2,
            "learned_wam_model_success_count": 2,
            "runpod_create_manifest_path": str(
                runpod_dir / "runpod_wam_async_create_manifest.json"
            ),
            "runpod_poll_manifest_path": str(runpod_dir / "runpod_wam_async_poll_manifest.json"),
            "runpod_teardown_manifest_path": str(
                runpod_dir / "runpod_wam_async_delete_manifest.json"
            ),
            "postprocess_artifacts": {
                "source_policy_observation_visual_qa": str(source_qa),
                "wam_rollout_visual_quality_report": str(report),
                "wam_rollout_contact_sheet": str(contact_sheet),
                "wam_rollout_frame_stats": str(frame_stats),
                "video_review_status": str(video_status),
                "review_video_path": str(review_video),
            },
            "review_video_path": str(review_video),
            "video_review_status_path": str(video_status),
            "source_policy_observation_visual_qa_path": str(source_qa),
            "wam_rollout_visual_quality_report_path": str(report),
            "wam_rollout_contact_sheet_path": str(contact_sheet),
            "wam_rollout_visual_success": True,
        }
        (job / "unitree_groot_n17_sonic_vast_persistent_session_result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result, 0

    monkeypatch.setattr(short_sanity, "run_persistent_session_runpod", fake_runpod)

    manifest, exit_code = short_sanity.run_short_visual_sanity(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "short",
        provider="runpod",
        transition_count=2,
    )

    assert exit_code == 0
    assert manifest["status"] == "passed_short_visual_sanity"
    assert manifest["short_visual_sanity_passed"] is True
    assert captured["loop_step_count"] == 3
    assert captured["visual_profile"] == "review_quality"
    assert captured["num_frames"] == "24"
    assert captured["height"] == "480"
    assert captured["width"] == "640"
    assert captured["fps"] == "15"
    assert manifest["requested_transition_count"] == 2
    assert manifest["generated_transition_count"] == 2
    assert manifest["ffprobe_metadata"]["streams"][0]["width"] == 640
    assert manifest["provider_success"] is True
    assert manifest["visually_useful_rollout"] is True
    assert manifest["claim_boundary"]["provider_success"] is True
    assert manifest["claim_boundary"]["visually_useful_rollout"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert Path(manifest["wam_rollout_contact_sheet_path"]).is_file()
    assert manifest["paid_provider"]["used"] is True
    assert manifest["paid_provider"]["teardown_status"] == "completed"
    assert manifest["paid_provider"]["continuing_spend_from_this_run"] is False
    validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        manifest["short_visual_sanity_manifest_path"],
        policy_observation_path=observation_path,
    )
    assert validation["status"] == "passed_short_visual_sanity"


def test_short_visual_sanity_can_wrap_existing_persistent_session_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    existing = tmp_path / "existing-run"
    existing.mkdir()
    runpod_dir = existing / "runpod_persistent_session_run"
    runpod_dir.mkdir()
    (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
        json.dumps({"status": "pod_created", "pod_id": "pod-123"}),
        encoding="utf-8",
    )
    (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pod_id": "pod-123",
                "teardown_performed": True,
                "continuing_spend_from_this_run": False,
            }
        ),
        encoding="utf-8",
    )
    (runpod_dir / "runpod_wam_async_delete_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pod_id": "pod-123",
                "continuing_spend_from_this_run": False,
            }
        ),
        encoding="utf-8",
    )
    old_source_qa = existing / "old_source_policy_observation_visual_qa.json"
    old_source_qa.write_text(
        json.dumps(
            {
                "status": "passed_visual_quality_gate",
                "visual_profile": "smoke",
                "review_quality_required": False,
            }
        ),
        encoding="utf-8",
    )
    report = existing / "wam_rollout_visual_quality_report.json"
    report.write_text(
        json.dumps({"status": "passed_visual_quality_gate", "visual_success": True}),
        encoding="utf-8",
    )
    contact_sheet = _write_reviewable_frame(existing / "wam_rollout_contact_sheet.jpg")
    frame_stats = existing / "wam_rollout_frame_stats.jsonl"
    frame_stats.write_text("{}\n", encoding="utf-8")
    video_status = existing / "video_review_status.json"
    review_video = existing / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
    review_video.parent.mkdir()
    review_video.write_bytes(b"mp4")
    video_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": {
                    "streams": [
                        {
                            "width": 320,
                            "height": 256,
                            "avg_frame_rate": "6/1",
                            "nb_frames": "9",
                        }
                    ],
                    "format": {"duration": "1.5", "size": "1000"},
                },
            }
        ),
        encoding="utf-8",
    )
    result = {
        "status": "completed",
        "job_dir": str(existing),
        "generated_next_observation_count": 1,
        "live_wam_generation_success_count": 1,
        "learned_wam_model_success_count": 1,
        "runpod_create_manifest_path": str(runpod_dir / "runpod_wam_async_create_manifest.json"),
        "runpod_poll_manifest_path": str(runpod_dir / "runpod_wam_async_poll_manifest.json"),
        "runpod_teardown_manifest_path": str(runpod_dir / "runpod_wam_async_delete_manifest.json"),
        "postprocess_artifacts": {
            "source_policy_observation_visual_qa": str(old_source_qa),
            "wam_rollout_visual_quality_report": str(report),
            "wam_rollout_contact_sheet": str(contact_sheet),
            "wam_rollout_frame_stats": str(frame_stats),
            "video_review_status": str(video_status),
            "review_video_path": str(review_video),
        },
        "review_video_path": str(review_video),
        "video_review_status_path": str(video_status),
        "source_policy_observation_visual_qa_path": str(old_source_qa),
        "wam_rollout_visual_quality_report_path": str(report),
        "wam_rollout_contact_sheet_path": str(contact_sheet),
        "wam_rollout_visual_success": True,
    }
    result_path = existing / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    called = False

    def fake_runpod(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("imported result should not relaunch provider")

    monkeypatch.setattr(short_sanity, "run_persistent_session_runpod", fake_runpod)

    manifest, exit_code = short_sanity.run_short_visual_sanity(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "short-manifest",
        provider="runpod",
        transition_count=1,
        persistent_session_result_path=result_path,
    )

    assert exit_code == 2
    assert called is False
    assert manifest["status"] == "blocked"
    assert manifest["short_visual_sanity_passed"] is False
    assert manifest["manifest_source"] == "imported_persistent_session_result"
    assert Path(manifest["short_visual_sanity_manifest_path"]).parent == (
        tmp_path / "short-manifest"
    ).resolve()
    assert manifest["persistent_session_result_path"] == str(result_path.resolve())
    assert manifest["requested_transition_count"] == 1
    assert manifest["generated_transition_count"] == 1
    assert manifest["review_media_resolution"] == {
        "width": 320,
        "height": 256,
        "fps": 6.0,
        "frame_count": 9,
        "minimum_width": 320,
        "minimum_height": 256,
        "minimum_fps": 8.0,
        "minimum_num_frames": 12,
        "resolution_passed": True,
        "fps_passed": False,
        "frame_count_passed": False,
        "passed": False,
    }
    assert (
        "short_visual_sanity_review_video_fps_below_review_quality_minimum"
        in manifest["blockers"]
    )
    assert (
        "short_visual_sanity_review_video_frame_count_below_review_quality_minimum"
        in manifest["blockers"]
    )
    assert "short_visual_sanity_quality_report_not_review_quality_profile" in manifest[
        "blockers"
    ]
    source_qa = json.loads(
        Path(manifest["source_policy_observation_visual_qa_path"]).read_text(encoding="utf-8")
    )
    assert source_qa["visual_profile"] == "review_quality"
    assert source_qa["review_quality_required"] is True
    assert manifest["paid_provider"]["used"] is True
    assert manifest["paid_provider"]["teardown_status"] == "completed"
    assert manifest["paid_provider"]["continuing_spend_from_this_run"] is False
    validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        manifest["short_visual_sanity_manifest_path"],
        policy_observation_path=observation_path,
    )
    assert validation["status"] == "blocked"
    assert (
        "short_visual_sanity_video_status_review_video_fps_below_review_quality_minimum"
        in validation["blockers"]
    )
    assert (
        "short_visual_sanity_video_status_review_video_frame_count_below_review_quality_minimum"
        in validation["blockers"]
    )


def test_short_visual_sanity_validation_rechecks_referenced_artifacts(
    tmp_path: Path,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    manifest_path = _write_passed_short_sanity_manifest(
        tmp_path / "short-sanity",
        observation_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_qa_path = Path(payload["source_policy_observation_visual_qa_path"])
    visual_report_path = Path(payload["wam_rollout_visual_quality_report_path"])
    video_status_path = Path(payload["video_review_status_path"])
    review_video_path = Path(payload["review_video_path"])
    teardown_path = tmp_path / "short-sanity" / "runpod_wam_async_delete_manifest.json"

    source_qa_path.write_text(
        json.dumps(
            {
                "status": "failed_visual_quality_gate",
                "blockers": ["source_policy_observation_too_dark_for_review"],
            }
        ),
        encoding="utf-8",
    )
    visual_report_path.write_text(
        json.dumps(
            {
                "status": "failed_visual_quality_gate",
                "visual_success": False,
                "structural_fallback_used": True,
                "blockers": [
                    "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout",
                    "wam_generated_frame_too_dark_for_review",
                ],
            }
        ),
        encoding="utf-8",
    )
    video_status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 2,
                "ffprobe_metadata": {},
            }
        ),
        encoding="utf-8",
    )
    review_video_path.write_bytes(b"")
    teardown_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "continuing_spend_from_this_run": True,
            }
        ),
        encoding="utf-8",
    )
    payload["paid_provider"] = {
        "provider": "runpod",
        "used": True,
        "teardown_status": "completed",
        "teardown_performed": True,
        "continuing_spend_from_this_run": False,
        "teardown_manifest_path": str(teardown_path),
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        manifest_path,
        policy_observation_path=observation_path,
    )

    assert validation["status"] == "blocked"
    assert "short_visual_sanity_source_qa_artifact_not_passed" in validation["blockers"]
    assert "source_policy_observation_too_dark_for_review" in validation["blockers"]
    assert "short_visual_sanity_quality_report_visual_success_not_passed" in validation[
        "blockers"
    ]
    assert "short_visual_sanity_quality_report_structural_fallback_used" in validation[
        "blockers"
    ]
    assert "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout" in validation[
        "blockers"
    ]
    assert "wam_generated_frame_too_dark_for_review" in validation["blockers"]
    assert "short_visual_sanity_video_status_ffprobe_returncode_not_zero" in validation[
        "blockers"
    ]
    assert "short_visual_sanity_video_status_ffprobe_metadata_missing" in validation[
        "blockers"
    ]
    assert "short_visual_sanity_review_video_empty" in validation["blockers"]
    assert "short_visual_sanity_paid_provider_teardown_artifact_not_zero_spend" in validation[
        "blockers"
    ]


def test_runpod_persistent_session_clamps_tiny_oscar_frame_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "3")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["wam_num_frames"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        runpod_dir = Path(kwargs["job_dir"])
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "output_zip_present": False,
                    "provider_command_status": "blocked",
                    "provider_command_blockers": [
                        "runpod_provider_runtime_output_zip_not_received_locally"
                    ],
                    "teardown_performed": True,
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=2,
        timeout_seconds=60,
        max_wait_seconds=20,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert captured["wam_num_frames"] == "5"
    assert session.os.environ["BLUEPRINT_OSCAR_WAM_NUM_FRAMES"] == "3"


def test_runpod_persistent_session_launches_full_loop_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.delenv(session.RUNPOD_FULL_LOOP_OVERRIDE_ENV, raising=False)
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["provider_bundle_kind"] = kwargs["provider_bundle_kind"]
        captured["loop_step_count"] = json.loads(
            (
                Path(kwargs["job_dir"]).parent
                / "provider_bundle"
                / "provider_runtime"
                / "persistent_session_input.json"
            ).read_text(encoding="utf-8")
        )["loop_step_count"]
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        runpod_dir = Path(kwargs["job_dir"])
        poll_manifest = {
            "status": "running",
            "output_zip_present": False,
            "provider_command_status": "running",
            "provider_command_blockers": [],
            "teardown_performed": False,
            "continuing_spend_from_this_run": True,
        }
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(poll_manifest),
            encoding="utf-8",
        )
        return poll_manifest

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=12,
        timeout_seconds=60,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["blockers"] == ["runpod_persistent_session_still_running"]
    assert output["details"]["poll_manifest"]["status"] == "running"
    assert captured["provider_bundle_kind"] == "wam"
    assert captured["loop_step_count"] == 12


def test_runpod_live_wam_blocker_classifies_missing_provider_artifact(tmp_path: Path) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_wrapper_or_upload_watchdog_no_valid_provider_artifact"
    )
    assert classification["evidence"]["output_zip_present"] is False
    assert (tmp_path / "runpod_live_wam_blocker_classification.json").is_file()


def test_runpod_live_wam_blocker_classifies_terminal_upload_after_heartbeat(
    tmp_path: Path,
) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(
                    tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
                ),
            },
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_terminal_output_upload_failed_after_remote_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_result_status"] == "running"


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_bootstrap(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "bootstrap_policy_server_started",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_policy_server_bootstrap_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "bootstrap_policy_server_started"
    )


def test_runpod_live_wam_blocker_classifies_pod_disappeared_after_model_snapshot(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_model_snapshot_completed",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_after_gr00t_model_snapshot_before_policy_server_ready"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "gr00t_model_snapshot_completed"
    )


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_policy_server_process(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_policy_server_waiting_for_listen",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_gr00t_policy_server_process_start_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "gr00t_policy_server_waiting_for_listen"
    )


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_uv_sync(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_uv_sync_started",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_gr00t_uv_sync_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == "gr00t_uv_sync_started"


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_system_python_deps(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_system_python_minimal_deps_install_started",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_gr00t_system_python_minimal_deps_install_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "gr00t_system_python_minimal_deps_install_started"
    )


def test_runpod_live_wam_blocker_classifies_running_after_heartbeat_until_timeout(
    tmp_path: Path,
) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "RUNNING",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(
                    tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
                ),
            },
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_remote_runtime_still_running_after_heartbeat_until_local_timeout"
    )


def test_runpod_live_wam_blocker_classifies_policy_runtime_bootstrap_timeout(
    tmp_path: Path,
) -> None:
    extraction_dir = tmp_path / "imported_persistent_session_output"
    extraction_dir.mkdir()
    (extraction_dir / "runpod_unitree_groot_sonic_entrypoint_execution.json").write_text(
        json.dumps(
            {
                "status": "timed_out",
                "timed_out": True,
                "timeout_seconds": 240,
                "returncode": -15,
            }
        ),
        encoding="utf-8",
    )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "completed",
            "output_zip_present": True,
            "provider_command_status": "completed",
            "runtime_result_status": "blocked",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
        extraction_dir=extraction_dir,
        imported={
            "status": "blocked",
            "blockers": ["persistent_session_entrypoint_exited_without_runtime_result"],
        },
    )

    assert classification["status"] == "blocked"
    assert classification["classified_blocker"] == "policy_runtime_bootstrap_timeout"
    assert classification["evidence"]["entrypoint_timed_out"] is True
    assert classification["evidence"]["entrypoint_timeout_seconds"] == 240


def test_runpod_live_wam_blocker_classifies_oscar_temporal_window(
    tmp_path: Path,
) -> None:
    extraction_dir = tmp_path / "imported_persistent_session_output"
    policy_dir = extraction_dir / "policy_calls"
    wam_dir = extraction_dir / "wam_calls"
    runtime_dir = extraction_dir / "wam_worker_steps" / "step_0001" / "oscar_runtime_output"
    policy_dir.mkdir(parents=True)
    wam_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    (policy_dir / "policy_call_0000.json").write_text(
        json.dumps({"status": "completed", "unitree_policy_action_command_ran": True}),
        encoding="utf-8",
    )
    (wam_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "blockers": [
                    "persistent_wam_worker_oscar_runtime_nonzero_exit",
                    "wam_output_missing_materializable_frame_or_video",
                ],
                "materialization": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "wam_runtime_result.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "inference_detail": {
                    "stderr_tail_redacted": (
                        "worldsim/_src/tokenizers/wan2pt1.py RuntimeError: "
                        "Kernel size can't be greater than actual input size"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "completed",
            "output_zip_present": True,
            "provider_command_status": "completed",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
        extraction_dir=extraction_dir,
        imported={
            "status": "blocked",
            "required_policy_call_count": 2,
            "required_wam_transition_count": 1,
            "repeated_policy_calls_count": 1,
            "generated_next_observation_count": 0,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 0,
            "blockers": ["wam_output_missing_materializable_frame_or_video"],
        },
    )

    assert classification["status"] == "blocked"
    assert classification["classified_blocker"] == "oscar_wam_temporal_window_too_short"
    assert classification["evidence"]["oscar_temporal_tokenizer_blocked"] is True


def test_runpod_live_wam_blocker_classifies_frame_materialization(
    tmp_path: Path,
) -> None:
    extraction_dir = tmp_path / "imported_persistent_session_output"
    policy_dir = extraction_dir / "policy_calls"
    wam_dir = extraction_dir / "wam_calls"
    policy_dir.mkdir(parents=True)
    wam_dir.mkdir(parents=True)
    (policy_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "unitree_policy_action_command_ran": True,
                "provider_output_replay_used": False,
            }
        ),
        encoding="utf-8",
    )
    (wam_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "blockers": ["wam_output_missing_materializable_frame_or_video"],
                "materialization": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "completed",
            "output_zip_present": True,
            "provider_command_status": "completed",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
        extraction_dir=extraction_dir,
        imported={
            "status": "blocked",
            "required_policy_call_count": 2,
            "required_wam_transition_count": 1,
            "repeated_policy_calls_count": 1,
            "generated_next_observation_count": 0,
            "live_wam_generation_success_count": 0,
            "learned_wam_model_success_count": 0,
            "blockers": ["wam_output_missing_materializable_frame_or_video"],
        },
    )

    assert classification["status"] == "blocked"
    assert classification["classified_blocker"] == "wam_frame_materialization_blocked"
    assert classification["evidence"]["policy_call_artifact_count"] == 1
    assert classification["evidence"]["wam_call_artifact_count"] == 1
