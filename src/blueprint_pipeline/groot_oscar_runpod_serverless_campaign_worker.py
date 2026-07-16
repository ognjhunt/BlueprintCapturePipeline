"""Bounded GR00T + OSCAR kitchen campaign inside one warm RunPod worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .common import write_json
from .g1_kitchen_leaf_evidence import load_attempt_identity
from .g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from .groot_oscar_closed_loop_image import (
    IMAGE_REF_ENV,
    SEALED_CONFIRMED_ENV,
    build_sealed_launch_plan,
)
from .runtime_ephemeral_trust import SIGNERS, create_attempt_trust
from .kitchen_attempt_lineage import ATTEMPT_INPUT_SCHEMA_VERSION


SCHEMA_VERSION = "groot_oscar_runpod_serverless_kitchen_campaign.v1"
INPUT_SCHEMA_VERSION = "groot_oscar_runpod_serverless_campaign_input.v1"
ARTIFACT_SCHEMA_VERSION = "groot_oscar_runpod_serverless_campaign_artifacts.v1"
ATTEMPT_SCHEMA_VERSION = "groot_oscar_runpod_serverless_campaign_attempt.v1"
EXPECTED_ATTEMPTS = (
    ("smoke", "smoke", 1000, 300),
    ("episode_001", "episode", 1001, 900),
    ("episode_002", "episode", 1002, 900),
    ("episode_003", "episode", 1003, 900),
)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_SOURCE_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
WORKSPACE_ROOT = Path("/workspace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"json_object_required:{path.name}")
    return dict(value)


def _safe_volume_path(root: Path, relative: Any, digest: Any) -> Path:
    text = str(relative or "").strip().replace("\\", "/")
    candidate = Path(text)
    expected = str(digest or "").strip().lower()
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not _SHA256.fullmatch(expected)
    ):
        raise ValueError("campaign_volume_reference_invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise ValueError("campaign_volume_reference_escape")
    if _sha256(resolved) != expected:
        raise ValueError("campaign_volume_reference_sha256_mismatch")
    return resolved


def _extract_zip(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise ValueError("campaign_bundle_unsafe_member")
            target = (root / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError("campaign_bundle_member_escape")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def _provider_allocation_id() -> str:
    for name in ("BLUEPRINT_PROVIDER_ALLOCATION_ID", "RUNPOD_POD_ID", "RUNPOD_WORKER_ID"):
        value = str(os.getenv(name) or "").strip()
        if value and _SAFE_ID.fullmatch(value):
            return value
    return "runpod-serverless-worker"


def _validate_campaign_input(
    payload: Mapping[str, Any],
    *,
    source_commit: str,
    image_ref: str,
    model_manifest_digest: str,
) -> list[dict[str, Any]]:
    blockers: list[str] = []
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        blockers.append("campaign_input_schema_invalid")
    if not _SOURCE_SHA.fullmatch(source_commit) or payload.get("source_commit") != source_commit:
        blockers.append("campaign_source_commit_mismatch")
    if payload.get("worker_image_ref") != image_ref or "@sha256:" not in image_ref:
        blockers.append("campaign_worker_image_mismatch")
    if payload.get("model_manifest_digest") != model_manifest_digest:
        blockers.append("campaign_model_manifest_mismatch")
    runtime = payload.get("runtime")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    if not (
        runtime.get("dynamic_episode_termination") is True
        and runtime.get("stop_immediately_on_declared_completion") is True
        and runtime.get("fixed_frame_count") is None
        and int(runtime.get("review_width") or 0) >= 640
        and int(runtime.get("review_height") or 0) >= 480
    ):
        blockers.append("campaign_runtime_contract_invalid")
    rows = payload.get("attempts")
    rows = list(rows) if isinstance(rows, list) else []
    normalized: list[dict[str, Any]] = []
    for expected, row in zip(EXPECTED_ATTEMPTS, rows, strict=False):
        attempt = dict(row) if isinstance(row, Mapping) else {}
        attempt_id, kind, seed, timeout = expected
        if (
            attempt.get("attempt_id") != attempt_id
            or attempt.get("kind") != kind
            or attempt.get("seed") != seed
            or attempt.get("timeout_seconds") != timeout
        ):
            blockers.append(f"campaign_attempt_contract_invalid:{attempt_id}")
        normalized.append(attempt)
    if len(rows) != len(EXPECTED_ATTEMPTS):
        blockers.append("campaign_requires_smoke_and_three_episodes")
    if blockers:
        raise ValueError(",".join(sorted(set(blockers))))
    return normalized


def _validate_attempt_manifest(
    identity: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    source_commit: str,
    image_digest: str,
    bundle_sha256: str,
    work: Path,
) -> None:
    blockers: list[str] = []
    if identity.get("schema_version") != ATTEMPT_INPUT_SCHEMA_VERSION:
        blockers.append("campaign_attempt_schema_mismatch")
    if (
        identity.get("attempt_id") != attempt.get("attempt_id")
        or identity.get("source_commit") != source_commit
        or identity.get("image_digest") != image_digest
    ):
        blockers.append("campaign_attempt_identity_mismatch")
    if (
        identity.get("source_dirty_patch_sha256")
        != CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
    ):
        blockers.append("campaign_attempt_source_not_canonical_clean")
    runtime = identity.get("serverless_runtime_qualification_contract")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    if not (
        runtime.get("schema_version")
        == "g1_kitchen_serverless_runtime_qualification.v1"
        and runtime.get("startup_reverified_in_campaign_job") is True
        and runtime.get("strict_three_action_probe_required_before_campaign") is True
        and runtime.get("same_runtime_worker_identity_required") is True
    ):
        blockers.append("campaign_attempt_runtime_qualification_contract_invalid")
    artifacts = identity.get("artifacts")
    artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    expected_files = {
        "route": work / "route.json",
        "task_success_contract": work / "task_success_contract.json",
        "kitchen_inventory": work / "kitchen_asset_inventory_checksums.json",
    }
    bundle_ref = artifacts.get("bundle")
    bundle_ref = dict(bundle_ref) if isinstance(bundle_ref, Mapping) else {}
    if bundle_ref.get("sha256") != bundle_sha256:
        blockers.append("campaign_attempt_bundle_digest_mismatch")
    for name, path in expected_files.items():
        ref = artifacts.get(name)
        ref = dict(ref) if isinstance(ref, Mapping) else {}
        if ref.get("sha256") != _sha256(path):
            blockers.append(f"campaign_attempt_artifact_digest_mismatch:{name}")
    selection = artifacts.get("selection")
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    if not _SHA256.fullmatch(str(selection.get("sha256") or "")):
        blockers.append("campaign_attempt_selection_digest_missing")
    if blockers:
        raise ValueError(",".join(sorted(set(blockers))))


def _terminate(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            # The process group exited after the initial poll.
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # SIGKILL has been sent; the kernel may still be reaping the child.
            return


def _start(
    command: Sequence[str], *, log_path: Path, env: Mapping[str, str]
) -> tuple[subprocess.Popen[bytes], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    process = subprocess.Popen(
        list(command),
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=dict(env),
        start_new_session=True,
    )
    return process, handle


def _wait_tcp(process: subprocess.Popen[Any], port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        with socket.socket() as probe:
            probe.settimeout(2)
            try:
                probe.connect(("127.0.0.1", port))
            except OSError:
                time.sleep(2)
            else:
                return True
    return False


def _wait_gear(process: subprocess.Popen[Any], deadline: float) -> bool:
    import msgpack
    import zmq

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "robot_config")
    subscriber.connect("tcp://127.0.0.1:5557")
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            if subscriber.poll(1000, zmq.POLLIN):
                raw = subscriber.recv()
                value = msgpack.unpackb(raw[len(b"robot_config") :], raw=False)
                if isinstance(value, Mapping) and value:
                    return True
        return False
    finally:
        subscriber.close()
        context.term()


def _wait_isaac_state(
    process: subprocess.Popen[Any], state_path: Path, deadline: float
) -> bool:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if state_path.is_file():
            try:
                state = _read(state_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                state = {}
            measurement = state.get("measurement")
            if isinstance(measurement, Mapping) and measurement.get("surrogate") is False:
                return True
        time.sleep(2)
    return False


def _replace_workspace_paths(command: Sequence[Any], work: Path) -> list[str]:
    result: list[str] = []
    for raw in command:
        value = str(raw)
        if value.startswith("/workspace/"):
            value = str(work / value.removeprefix("/workspace/"))
        result.append(value)
    return result


def _replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def _attempt_environment(
    *, work: Path, attempt_manifest: Path, public_manifest: Path, base: Mapping[str, str]
) -> dict[str, str]:
    attempt_id = _read(attempt_manifest).get("attempt_id")
    secret_root = work / ".runtime-secrets" / str(attempt_id)
    environment_file = secret_root / "trust_env.sh"
    identity = load_attempt_identity(
        attempt_manifest, provider_allocation_id=_provider_allocation_id()
    )
    trust = create_attempt_trust(
        secret_root=secret_root,
        environment_file=environment_file,
        public_manifest=public_manifest,
        identity_binding=identity,
    )
    public_hashes = trust.get("public_key_sha256")
    public_hashes = dict(public_hashes) if isinstance(public_hashes, Mapping) else {}
    env = dict(base)
    for name, _role, private_env, trust_env in SIGNERS:
        env[private_env] = str(secret_root / f"{name}.pem")
        env[trust_env] = str(public_hashes[trust_env])
    return env


def _review_video(episode_dir: Path) -> dict[str, Any]:
    trace_path = episode_dir / "oscar_isaac_closed_loop_trace.jsonl"
    clips: list[Path] = []
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate = Path(str(value.get("wam_generated_video") or ""))
            if candidate.is_file() and candidate not in clips:
                clips.append(candidate)
    blockers: list[str] = []
    output = episode_dir / "final_review.mp4"
    if not clips:
        blockers.append("no_ordered_oscar_step_clips")
    else:
        concat = episode_dir / "ordered_review_clips.ffconcat"
        concat.write_text(
            "ffconcat version 1.0\n"
            + "".join(f"file '{path}'\n" for path in clips),
            encoding="utf-8",
        )
        encoded = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat),
                "-c", "copy", "-an", str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if encoded.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            blockers.append(f"ffmpeg_concat_failed:{encoded.returncode}")
    width = height = frame_count = 0
    duration = 0.0
    codec: str | None = None
    if not blockers:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,nb_read_frames,duration",
                "-show_entries", "format=duration", "-of", "json", str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            metadata = json.loads(probe.stdout)
            stream = metadata["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
            frame_count = int(stream.get("nb_read_frames") or 0)
            duration = float(
                stream.get("duration")
                or metadata.get("format", {}).get("duration")
                or 0
            )
            codec = str(stream.get("codec_name") or "") or None
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            blockers.append("ffprobe_metadata_invalid")
        if probe.returncode != 0:
            blockers.append(f"ffprobe_failed:{probe.returncode}")
        if width < 640 or height < 480:
            blockers.append(f"review_resolution_below_640x480:{width}x{height}")
        if frame_count < 1 or duration <= 0:
            blockers.append("review_video_empty_or_zero_duration")
    result = {
        "schema_version": "groot_oscar_episode_review_validation.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "path": str(output) if output.is_file() else None,
        "sha256": _sha256(output) if output.is_file() else None,
        "codec": codec,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "ordered_clip_count": len(clips),
        "video_frame_count_mode": "dynamic_from_actual_episode_duration",
        "minimum_resolution": {"width": 640, "height": 480},
    }
    write_json(episode_dir / "final_review_validation.json", result)
    return result


def _smoke_admission(episode_dir: Path, command_rc: int) -> dict[str, Any]:
    manifest_path = episode_dir / "oscar_isaac_closed_loop_manifest.json"
    trace_path = episode_dir / "oscar_isaac_closed_loop_trace.jsonl"
    manifest = _read(manifest_path) if manifest_path.is_file() else {}
    proof = manifest.get("proof")
    proof = dict(proof) if isinstance(proof, Mapping) else {}
    steps = int(manifest.get("steps_executed") or manifest.get("steps_completed") or 0)
    actions = int(proof.get("isaac_policy_actions_recorded") or 0)
    requeries = int(proof.get("learned_policy_requery_count") or 0)
    trace_lines = (
        [line for line in trace_path.read_text(errors="replace").splitlines() if line.strip()]
        if trace_path.is_file()
        else []
    )
    checks = {
        "command_rc_zero": command_rc == 0,
        "manifest_completed": manifest.get("status") == "completed",
        "real_provider_probe": manifest.get("harness_backend_kind") == "real_provider_probe",
        "simulator_steps_at_least_three": steps >= 3,
        "groot_policy_actions_at_least_three": actions >= 3,
        "learned_policy_requery_nonempty": requeries >= 1,
        "learned_action_trace_nonempty": len(trace_lines) >= 1,
        "real_perception_backend_used": (
            manifest.get("real_perception_backend_used") is True
            or int(proof.get("real_perception_backend_steps") or 0) >= 1
        ),
        "sam3_completed": int(proof.get("sam3_completed_steps") or 0) >= 1,
        "da3_completed": int(proof.get("da3_completed_steps") or 0) >= 1,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": "g1_kitchen_smoke_admission.v1",
        "status": "passed" if not blockers else "blocked",
        "seed": 1000,
        "checks": checks,
        "blockers": blockers,
        "command_rc": command_rc,
        "steps_executed": steps,
        "isaac_policy_actions_recorded": actions,
        "learned_policy_requery_count": requeries,
        "learned_action_trace_lines": len(trace_lines),
        "full_episodes_authorized": not blockers,
    }
    write_json(episode_dir / "smoke_admission.json", result)
    return result


def _semantic_success(manifest: Mapping[str, Any]) -> bool | None:
    proof = manifest.get("success_proof")
    proof = dict(proof) if isinstance(proof, Mapping) else {}
    for key in (
        "manipulation_success_proven",
        "did_target_manipulation_succeed",
        "simulated_manipulation_success_shown",
    ):
        if isinstance(proof.get(key), bool):
            return bool(proof[key])
    return None


def _run_attempt(
    *,
    attempt: Mapping[str, Any],
    attempt_manifest: Path,
    work: Path,
    artifact_root: Path,
    plan: Mapping[str, Any],
    base_env: Mapping[str, str],
) -> dict[str, Any]:
    attempt_id = str(attempt["attempt_id"])
    seed = int(attempt["seed"])
    timeout_seconds = int(attempt["timeout_seconds"])
    started = time.monotonic()
    deadline = started + timeout_seconds
    episode_dir = artifact_root / attempt_id
    episode_dir.mkdir(parents=True, exist_ok=False)
    live_attempt = work / "attempt_input_manifest.json"
    shutil.copyfile(attempt_manifest, live_attempt)
    shutil.copyfile(attempt_manifest, episode_dir / "attempt_input_manifest.json")
    env = _attempt_environment(
        work=work,
        attempt_manifest=live_attempt,
        public_manifest=episode_dir / "runtime_ephemeral_trust.json",
        base=base_env,
    )
    state_path = work / "initial_g1_sonic_state.json"
    state_path.unlink(missing_ok=True)
    isaac_command = _replace_workspace_paths(
        list(plan.get("isaac_task_executor_command") or []), work
    )
    _replace_option(isaac_command, "--attempt-input-manifest", str(live_attempt))
    _replace_option(isaac_command, "--initial-state-output", str(state_path))
    isaac, isaac_log = _start(
        isaac_command,
        log_path=episode_dir / "isaac_task_executor.log",
        env=env,
    )
    command_rc: int | None = None
    timed_out = False
    try:
        if not _wait_isaac_state(isaac, state_path, deadline):
            timed_out = time.monotonic() >= deadline
            blockers = [
                "attempt_wall_timeout_during_isaac_startup"
                if timed_out
                else "persistent_isaac_task_executor_not_ready"
            ]
        else:
            command = _replace_workspace_paths(
                list(plan.get("closed_loop_command") or []), work
            )
            _replace_option(command, "--steps", "3" if attempt_id == "smoke" else "48")
            _replace_option(command, "--output-dir", str(episode_dir))
            _replace_option(command, "--attempt-input-manifest", str(live_attempt))
            command.extend(["--oscar-seed", str(seed)])
            stdout = (episode_dir / "console_stdout.log").open("wb")
            stderr = (episode_dir / "console_stderr.log").open("wb")
            runner = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
            try:
                remaining = max(0.1, deadline - time.monotonic())
                runner.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate(runner)
            finally:
                stdout.close()
                stderr.close()
            command_rc = runner.returncode
            (episode_dir / "command_rc.txt").write_text(
                f"{command_rc if command_rc is not None else 'timeout'}\n",
                encoding="utf-8",
            )
            blockers = []
            if timed_out:
                blockers.append("attempt_wall_timeout")
            elif command_rc != 0:
                blockers.append(f"closed_loop_command_exit:{command_rc}")
    finally:
        _terminate(isaac)
        isaac_log.close()
    elapsed = round(time.monotonic() - started, 3)
    manifest_path = episode_dir / "oscar_isaac_closed_loop_manifest.json"
    manifest = _read(manifest_path) if manifest_path.is_file() else {}
    if not manifest:
        blockers.append("closed_loop_manifest_missing")
    elif manifest.get("status") != "completed":
        blockers.append("closed_loop_manifest_not_completed")
    review = _review_video(episode_dir)
    if review.get("status") != "passed":
        blockers.append("final_review_not_passed")
    smoke = (
        _smoke_admission(episode_dir, command_rc if command_rc is not None else -1)
        if attempt_id == "smoke"
        else None
    )
    if smoke is not None and smoke.get("status") != "passed":
        blockers.append("strict_kitchen_smoke_gate_failed")
    proof = manifest.get("proof")
    proof = dict(proof) if isinstance(proof, Mapping) else {}
    policy_actions = int(proof.get("isaac_policy_actions_recorded") or 0)
    learned_requeries = int(proof.get("learned_policy_requery_count") or 0)
    fresh_wam_steps = int(proof.get("fresh_oscar_provider_model_run_steps") or 0)
    if manifest and policy_actions < 3:
        blockers.append("fewer_than_three_groot_policy_actions")
    if manifest and learned_requeries < 1:
        blockers.append("learned_policy_requery_missing")
    if manifest and fresh_wam_steps < 1:
        blockers.append("fresh_learned_wam_generation_missing")
    if manifest and proof.get("wam_evaluator_in_control_loop") is not True:
        blockers.append("wam_evaluator_not_in_control_loop")
    termination = manifest.get("episode_termination")
    termination = dict(termination) if isinstance(termination, Mapping) else {}
    result = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "kind": attempt.get("kind"),
        "seed": seed,
        "status": "completed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "timeout_seconds": timeout_seconds,
        "execution_elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "dynamic_episode_termination": True,
        "stop_immediately_on_declared_completion": True,
        "fixed_frame_count": None,
        "command_rc": command_rc,
        "manifest_status": manifest.get("status"),
        "termination_reason": termination.get("reason"),
        "simulator_steps": manifest.get("steps_executed") or manifest.get("steps_completed"),
        "policy_actions": policy_actions,
        "learned_policy_requery_count": learned_requeries,
        "fresh_learned_wam_generation_steps": fresh_wam_steps,
        "learned_wam_model_ran": fresh_wam_steps >= 1,
        "wam_evaluator_in_control_loop": proof.get("wam_evaluator_in_control_loop")
        is True,
        "semantic_task_success": _semantic_success(manifest),
        "review": review,
        "smoke_admission": smoke,
    }
    write_json(episode_dir / "attempt_result.json", result)
    return result


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "campaign_artifact_manifest.json":
            continue
        files.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "completed" if files else "blocked",
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
        "raw_secret_values_recorded": False,
    }
    write_json(root / "campaign_artifact_manifest.json", payload)
    return payload


def run_kitchen_campaign(
    *,
    network_volume_root: Path,
    campaign_manifest_relative_path: str,
    campaign_manifest_sha256: str,
    output_relative_path: str,
    source_commit: str,
    image_ref: str,
    model_manifest_digest: str,
) -> dict[str, Any]:
    started = time.monotonic()
    campaign_manifest = _safe_volume_path(
        network_volume_root,
        campaign_manifest_relative_path,
        campaign_manifest_sha256,
    )
    payload = _read(campaign_manifest)
    attempts = _validate_campaign_input(
        payload,
        source_commit=source_commit,
        image_ref=image_ref,
        model_manifest_digest=model_manifest_digest,
    )
    output_candidate = Path(str(output_relative_path or "").replace("\\", "/"))
    if (
        not str(output_candidate)
        or output_candidate.is_absolute()
        or ".." in output_candidate.parts
    ):
        raise ValueError("campaign_output_relative_path_invalid")
    volume_root = network_volume_root.resolve()
    artifact_root = (volume_root / output_candidate).resolve()
    if volume_root not in artifact_root.parents or artifact_root.exists():
        raise ValueError("campaign_output_path_invalid_or_exists")
    artifact_root.mkdir(parents=True)
    work = WORKSPACE_ROOT / f"blueprint-campaign-{campaign_manifest_sha256[:16]}"
    if work.exists():
        shutil.rmtree(work)
    bundle_ref = payload.get("payload_bundle")
    bundle_ref = dict(bundle_ref) if isinstance(bundle_ref, Mapping) else {}
    bundle = _safe_volume_path(
        volume_root, bundle_ref.get("relative_path"), bundle_ref.get("sha256")
    )
    _extract_zip(bundle, work)
    required = (
        "initial_policy_frame.png",
        "route.json",
        "task_prompt.txt",
        "task_success_contract.json",
        "kitchen_asset_inventory_checksums.json",
        "kitchen/KitchenRoom.usd",
    )
    missing = [name for name in required if not (work / name).is_file()]
    if missing:
        raise ValueError("campaign_payload_bundle_required_files_missing")
    allocation = _provider_allocation_id()
    os.environ["BLUEPRINT_PROVIDER_ALLOCATION_ID"] = allocation
    task_prompt = (work / "task_prompt.txt").read_text(encoding="utf-8").strip()
    plan_env = dict(os.environ)
    plan_env[SEALED_CONFIRMED_ENV] = "true"
    plan_env[IMAGE_REF_ENV] = image_ref
    plan = build_sealed_launch_plan(
        start_frame="/workspace/initial_policy_frame.png",
        route_file="/workspace/route.json",
        steps=48,
        task_prompt=task_prompt,
        output_dir="/workspace/closed_loop_out",
        attempt_input_manifest_path="/workspace/attempt_input_manifest.json",
        # Learned WAM execution is required below from the runtime proof.  It is
        # not, by itself, a forward/inverse consistency scorer and must not be
        # promoted into that separate evidence family.
        require_forward_inverse_consistency=False,
        allow_wam_consistency_scoring=False,
        env=plan_env,
    )
    if plan.get("sealed_active") is not True or plan.get("blockers"):
        raise ValueError("campaign_sealed_launch_plan_blocked")
    base_env = {**os.environ, **{str(k): str(v) for k, v in dict(plan["env"]).items()}}
    base_env["BLUEPRINT_LAUNCH_SESSION_ID"] = str(payload.get("campaign_id") or "")
    base_env["BLUEPRINT_WORKER_IMAGE_DIGEST"] = image_ref
    base_env["BLUEPRINT_PROVIDER_ALLOCATION_ID"] = allocation

    attempt_paths: dict[str, Path] = {}
    image_digest = image_ref.rsplit("@", 1)[-1]
    bundle_sha256 = str(bundle_ref.get("sha256") or "")
    for attempt in attempts:
        ref = attempt.get("attempt_manifest")
        ref = dict(ref) if isinstance(ref, Mapping) else {}
        path = _safe_volume_path(volume_root, ref.get("relative_path"), ref.get("sha256"))
        identity = _read(path)
        _validate_attempt_manifest(
            identity,
            attempt=attempt,
            source_commit=source_commit,
            image_digest=image_digest,
            bundle_sha256=bundle_sha256,
            work=work,
        )
        attempt_paths[str(attempt["attempt_id"])] = path

    kitchen_gate_dir = artifact_root / "startup_gates" / "kitchen"
    kitchen_gate_dir.mkdir(parents=True)
    kitchen_gate = subprocess.run(
        [
            "/isaac-sim/python.sh",
            "-m",
            "blueprint_pipeline.kitchen_asset_startup_gate",
            "--expected-inventory",
            str(work / "kitchen_asset_inventory_checksums.json"),
            "--out-dir",
            str(kitchen_gate_dir),
            "--tree-root",
            str(work / "kitchen"),
        ],
        check=False,
        env=base_env,
    )
    if kitchen_gate.returncode != 0:
        raise RuntimeError("campaign_kitchen_asset_startup_gate_failed")

    groot_command = _replace_workspace_paths(
        list(plan.get("groot_server_command") or []), work
    )
    gear_command = [str(item) for item in plan.get("gear_sonic_controller_command") or []]
    groot, groot_log = _start(
        groot_command, log_path=artifact_root / "groot_server.log", env=base_env
    )
    gear, gear_log = _start(
        gear_command, log_path=artifact_root / "gear_sonic_controller.log", env=base_env
    )
    runs: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        readiness_deadline = time.monotonic() + 300
        if not _wait_tcp(groot, int(plan["policy_server_port"]), readiness_deadline):
            blockers.append("campaign_groot_policy_server_not_ready")
        elif not _wait_gear(gear, readiness_deadline):
            blockers.append("campaign_gear_sonic_controller_not_ready")
        else:
            for attempt in attempts:
                try:
                    row = _run_attempt(
                        attempt=attempt,
                        attempt_manifest=attempt_paths[str(attempt["attempt_id"])],
                        work=work,
                        artifact_root=artifact_root,
                        plan=plan,
                        base_env=base_env,
                    )
                except Exception as exc:
                    row = {
                        "schema_version": ATTEMPT_SCHEMA_VERSION,
                        "attempt_id": str(attempt["attempt_id"]),
                        "kind": attempt.get("kind"),
                        "seed": int(attempt["seed"]),
                        "status": "blocked",
                        "blockers": [f"campaign_attempt_exception:{type(exc).__name__}"],
                        "timeout_seconds": int(attempt["timeout_seconds"]),
                        "semantic_task_success": None,
                    }
                    attempt_output = artifact_root / str(attempt["attempt_id"])
                    attempt_output.mkdir(parents=True, exist_ok=True)
                    write_json(attempt_output / "attempt_result.json", row)
                runs.append(row)
                if row.get("status") != "completed":
                    blockers.append(f"campaign_attempt_failed:{row['attempt_id']}")
                    break
    finally:
        _terminate(groot)
        _terminate(gear)
        groot_log.close()
        gear_log.close()
        if work.exists():
            shutil.rmtree(work)
    if len(runs) != len(EXPECTED_ATTEMPTS):
        blockers.append("campaign_all_required_attempts_not_completed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "campaign_id": payload.get("campaign_id"),
        "source_commit": source_commit,
        "worker_image_ref": image_ref,
        "model_manifest_digest": model_manifest_digest,
        "provider_allocation_id": allocation,
        "runs": runs,
        "smoke_passed": bool(runs and runs[0].get("status") == "completed"),
        "all_dynamic_episodes_completed": len(runs) == 4 and not blockers,
        "semantic_task_success_by_attempt": {
            str(row["attempt_id"]): row.get("semantic_task_success") for row in runs
        },
        "semantic_task_success_not_inferred_from_execution": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_relative_path": output_candidate.as_posix(),
        "raw_secret_values_recorded": False,
        "physical_robot_control_performed": False,
    }
    write_json(artifact_root / "campaign_result.json", result)
    artifact_manifest = _artifact_manifest(artifact_root)
    return {**result, "artifact_manifest": artifact_manifest}
