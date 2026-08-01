"""Trusted, bounded executor for compiled native-360 COLMAP plans."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .native_360_colmap_plan import NATIVE_360_COLMAP_PLAN_SCHEMA_VERSION
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_pose_estimation_request,
    build_pose_estimation_result,
)


NATIVE_360_COLMAP_RUNNER_VERSION = "native_360_colmap_runner.v1"
_EXPECTED_STEPS = {
    "extract_features": "feature_extractor",
    "configure_fixed_rig": "rig_configurator",
    "match_sequential_frames": "sequential_matcher",
    "map_fixed_calibrated_rig": "mapper",
    "export_registered_model_text": "model_converter",
}
_MODEL_PATHS = {
    "/opt/models/colmap/aliked-n16rot.onnx",
    "/opt/models/colmap/aliked-lightglue.onnx",
    "/opt/models/colmap/sift-lightglue.onnx",
    "/opt/models/colmap/bruteforce-matcher.onnx",
}


class Native360ColmapRunnerError(ValueError):
    """Stable refusal before a plan is admitted for execution."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


Runner = Callable[[Sequence[str], Path, float], subprocess.CompletedProcess[bytes]]


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_relative(value: Any) -> str | None:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _bound_input(root: Path, relative_path: Any, expected_digest: Any, maximum_bytes: int) -> Path:
    relative = _safe_relative(relative_path)
    if relative is None or not _digest(expected_digest):
        raise Native360ColmapRunnerError(["native_colmap_runner_input_reference_invalid"])
    lexical = root / Path(*PurePosixPath(relative).parts)
    if lexical.is_symlink():
        raise Native360ColmapRunnerError(["native_colmap_runner_input_symlink_forbidden"])
    resolved = lexical.resolve()
    if (
        (resolved != root and root not in resolved.parents)
        or not resolved.is_file()
        or resolved.stat().st_size > maximum_bytes
        or _sha256_file(resolved) != expected_digest
    ):
        raise Native360ColmapRunnerError(["native_colmap_runner_input_binding_invalid"])
    return resolved


def _destination(root: Path, relative_path: Any) -> Path:
    relative = _safe_relative(relative_path)
    if relative is None:
        raise Native360ColmapRunnerError(["native_colmap_runner_destination_reference_invalid"])
    destination = root / Path(*PurePosixPath(relative).parts)
    resolved_parent = destination.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise Native360ColmapRunnerError(["native_colmap_runner_destination_escape"])
    return destination


def _default_runner(
    argv: Sequence[str], cwd: Path, timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )


def _validated_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        plan = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise Native360ColmapRunnerError(["native_colmap_runner_plan_not_json"]) from exc
    errors: list[str] = []
    plan_digest = plan.get("native_360_colmap_execution_plan_digest")
    if (
        plan.get("schema_version") != NATIVE_360_COLMAP_PLAN_SCHEMA_VERSION
        or not _digest(plan_digest)
        or plan_digest
        != canonical_digest(plan, digest_field="native_360_colmap_execution_plan_digest")
        or plan.get("producing_method") != "blueprint.native_360_colmap_plan_compiler"
        or plan.get("plan_executed") is not False
        or plan.get("shell_invocation_allowed") is not False
        or plan.get("network_access_allowed") is not False
        or plan.get("hidden_heldout_access_allowed") is not False
        or plan.get("candidate_can_change_split") is not False
        or plan.get("agent_can_change_calibration") is not False
        or plan.get("blockers") != []
    ):
        errors.append("native_colmap_runner_plan_contract_invalid")
    commands = plan.get("commands")
    if not isinstance(commands, list) or len(commands) != len(_EXPECTED_STEPS):
        errors.append("native_colmap_runner_commands_invalid")
    else:
        seen: set[str] = set()
        for ordinal, raw in enumerate(commands):
            command = dict(raw) if isinstance(raw, Mapping) else {}
            step = str(command.get("step_id") or "")
            argv = command.get("argv")
            argv_items = argv if isinstance(argv, list) else []
            if (
                step not in _EXPECTED_STEPS
                or step in seen
                or not isinstance(argv, list)
                or len(argv) < 2
                or argv[:2] != ["colmap", _EXPECTED_STEPS.get(step)]
                or any(
                    not isinstance(item, str)
                    or not item
                    or "\x00" in item
                    or "\n" in item
                    or "\r" in item
                    for item in argv
                )
            ):
                errors.append(f"native_colmap_runner_command_invalid:{ordinal}")
            for argument in argv_items:
                if not isinstance(argument, str):
                    continue
                normalized = argument.replace("\\", "/")
                if ".." in PurePosixPath(normalized).parts or (
                    "/" in normalized
                    and not normalized.startswith("workspace/")
                    and normalized not in _MODEL_PATHS
                ):
                    errors.append(f"native_colmap_runner_command_path_invalid:{ordinal}")
            seen.add(step)
        if seen != set(_EXPECTED_STEPS):
            errors.append("native_colmap_runner_command_set_invalid")
    images = plan.get("image_materialization")
    masks = plan.get("mask_materialization")
    if not isinstance(images, list) or not images or not isinstance(masks, list) or not masks:
        errors.append("native_colmap_runner_materialization_missing")
    else:
        image_destinations = {
            str(row.get("destination_relative_path")) for row in images if isinstance(row, Mapping)
        }
        if (
            len(image_destinations) != len(images)
            or any(
                "held_out" in value or "evaluator_hidden" in value for value in image_destinations
            )
            or any(
                not isinstance(row, Mapping)
                or row.get("captured_observation") is not True
                or not str(row.get("frame_id") or "")
                or _safe_relative(row.get("source_relative_path")) is None
                or _safe_relative(row.get("destination_relative_path")) is None
                or not str(row.get("destination_relative_path")).startswith("workspace/images/")
                or not _digest(row.get("source_digest"))
                for row in images
            )
        ):
            errors.append("native_colmap_runner_candidate_materialization_invalid")
        mask_destinations = {
            str(row.get("destination_relative_path")) for row in masks if isinstance(row, Mapping)
        }
        if (
            len(masks) != len(images)
            or len(mask_destinations) != len(masks)
            or any(
                not isinstance(row, Mapping)
                or row.get("generated_or_inferred") is not False
                or _safe_relative(row.get("source_relative_path")) is None
                or _safe_relative(row.get("destination_relative_path")) is None
                or not str(row.get("destination_relative_path")).startswith("workspace/masks/")
                or not _digest(row.get("source_digest"))
                for row in masks
            )
        ):
            errors.append("native_colmap_runner_mask_materialization_invalid")
    if errors:
        raise Native360ColmapRunnerError(errors)
    return plan


def _bounded_log_payload(
    completed: subprocess.CompletedProcess[bytes], maximum_bytes: int
) -> bytes:
    stdout = completed.stdout
    stderr = completed.stderr
    stdout = stdout.encode() if isinstance(stdout, str) else stdout
    stderr = stderr.encode() if isinstance(stderr, str) else stderr
    stdout = stdout if isinstance(stdout, bytes) else b""
    stderr = stderr if isinstance(stderr, bytes) else b""
    payload = b"[stdout]\n" + stdout + b"\n[stderr]\n" + stderr
    if len(payload) > maximum_bytes:
        raise Native360ColmapRunnerError(["native_colmap_runner_command_output_oversized"])
    return payload


def _registered_image_names(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise Native360ColmapRunnerError(["native_colmap_runner_registered_images_missing"])
    names: list[str] = []
    expect_image_row = True
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if expect_image_row:
            fields = line.split()
            if len(fields) < 10:
                raise Native360ColmapRunnerError(
                    ["native_colmap_runner_registered_images_malformed"]
                )
            names.append(fields[9].replace("\\", "/"))
        expect_image_row = not expect_image_row
    if not names:
        raise Native360ColmapRunnerError(["native_colmap_runner_no_registered_images"])
    return names


def _output_digest_rows(root: Path) -> list[dict[str, str]]:
    paths = [root / "workspace/database.db"]
    paths.extend(sorted((root / "workspace/sparse_text").glob("*.txt")))
    paths.extend(sorted((root / "workspace/logs").glob("*.log")))
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            continue
        rows.append(
            {
                "artifact_id": path.relative_to(root).as_posix(),
                "digest": _sha256_file(path),
            }
        )
    return rows


def _result_path(root: Path) -> Path:
    return root / "pose_estimation_result.json"


def _validated_replay(root: Path, plan: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _result_path(root)
    if not root.is_dir() or path.is_symlink() or not path.is_file():
        return None
    try:
        result = build_pose_estimation_result(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ReconstructionWorkerContractError) as exc:
        raise Native360ColmapRunnerError(["native_colmap_runner_existing_result_invalid"]) from exc
    if result.get("pose_estimation_request_digest") != plan.get(
        "pose_estimation_request_digest"
    ) or result.get("native_360_colmap_execution_plan_digest") != plan.get(
        "native_360_colmap_execution_plan_digest"
    ):
        raise Native360ColmapRunnerError(["native_colmap_runner_existing_result_binding_mismatch"])
    for row in result.get("output_digests", []):
        relative = _safe_relative(row.get("artifact_id"))
        if relative is None:
            raise Native360ColmapRunnerError(["native_colmap_runner_existing_output_invalid"])
        output = root / Path(*PurePosixPath(relative).parts)
        if output.is_symlink() or not output.is_file() or _sha256_file(output) != row.get("digest"):
            raise Native360ColmapRunnerError(["native_colmap_runner_existing_output_invalid"])
    return result


def execute_native_360_colmap_plan(
    *,
    plan: Mapping[str, Any],
    input_root: str | Path,
    artifact_root: str | Path,
    timestamp: str,
    runner: Runner | None = None,
    maximum_input_bytes: int = 256 * 1024 * 1024,
    maximum_command_output_bytes: int = 8 * 1024 * 1024,
    maximum_command_timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Execute an admitted plan once and return a typed pose candidate result."""

    admitted = _validated_plan(plan)
    source_lexical = Path(input_root)
    output_lexical = Path(artifact_root)
    if source_lexical.is_symlink() or output_lexical.is_symlink():
        raise Native360ColmapRunnerError(["native_colmap_runner_root_symlink_forbidden"])
    source_root = source_lexical.resolve()
    output_parent = output_lexical.resolve()
    if (
        not source_root.is_dir()
        or maximum_input_bytes <= 0
        or maximum_command_output_bytes <= 0
        or maximum_command_timeout_seconds <= 0
    ):
        raise Native360ColmapRunnerError(["native_colmap_runner_execution_bounds_invalid"])
    output_parent.mkdir(parents=True, exist_ok=True)
    final_root = output_parent / (
        "native_colmap_execution_" + admitted["native_360_colmap_execution_plan_digest"][7:23]
    )
    replay = _validated_replay(final_root, admitted)
    if replay is not None:
        return replay
    if final_root.exists():
        raise Native360ColmapRunnerError(["native_colmap_runner_existing_artifact_conflict"])

    staging = Path(tempfile.mkdtemp(prefix=".native_colmap_attempt_", dir=output_parent)).resolve()
    workspace = staging / "workspace"
    for relative in ("images", "masks", "sparse", "sparse_text", "logs"):
        (workspace / relative).mkdir(parents=True, exist_ok=True)
    for collection_name in ("image_materialization", "mask_materialization"):
        for row in admitted[collection_name]:
            source = _bound_input(
                source_root,
                row["source_relative_path"],
                row["source_digest"],
                maximum_input_bytes,
            )
            destination = _destination(staging, row["destination_relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise Native360ColmapRunnerError(["native_colmap_runner_materialization_conflict"])
            shutil.copyfile(source, destination)
            if _sha256_file(destination) != row["source_digest"]:
                raise Native360ColmapRunnerError(
                    ["native_colmap_runner_materialization_digest_mismatch"]
                )
    (workspace / "rig_config.json").write_text(
        canonical_json(admitted["rig_config"]) + "\n", encoding="utf-8"
    )

    run = runner or _default_runner
    started = time.monotonic()
    command_receipts: list[dict[str, Any]] = []
    failure_code: str | None = None
    blockers: list[str] = []
    for command in admitted["commands"]:
        step_id = command["step_id"]
        try:
            completed = run(
                tuple(command["argv"]),
                staging,
                min(
                    float(admitted["execution_bounds"]["timeout_seconds"]),
                    float(maximum_command_timeout_seconds),
                ),
            )
            payload = _bounded_log_payload(completed, maximum_command_output_bytes)
        except subprocess.TimeoutExpired:
            failure_code = "pose_estimation_failure"
            blockers.append(f"native_colmap_command_timeout:{step_id}")
            payload = b"[typed_failure]\nnative_colmap_command_timeout\n"
            completed = subprocess.CompletedProcess(command["argv"], 124, b"", b"")
        except OSError:
            failure_code = "worker_startup_failure"
            blockers.append(f"native_colmap_command_startup_failed:{step_id}")
            payload = b"[typed_failure]\nnative_colmap_command_startup_failed\n"
            completed = subprocess.CompletedProcess(command["argv"], 127, b"", b"")
        log_path = workspace / "logs" / f"{len(command_receipts):02d}_{step_id}.log"
        log_path.write_bytes(payload)
        command_receipts.append(
            {
                "step_id": step_id,
                "argv_digest": canonical_digest({"argv": command["argv"]}),
                "returncode": int(completed.returncode),
                "log_relative_path": log_path.relative_to(staging).as_posix(),
                "log_digest": _sha256_file(log_path),
            }
        )
        if failure_code is not None or completed.returncode != 0:
            if failure_code is None:
                failure_code = "pose_estimation_failure"
                blockers.append(f"native_colmap_command_failed:{step_id}")
            break

    image_to_frame = {
        str(row["destination_relative_path"]).removeprefix("workspace/images/"): str(
            row["frame_id"]
        )
        for row in admitted["image_materialization"]
    }
    registered: list[str] = []
    if failure_code is None:
        try:
            registered_names = _registered_image_names(workspace / "sparse_text/images.txt")
            if any(name not in image_to_frame for name in registered_names):
                raise Native360ColmapRunnerError(
                    ["native_colmap_runner_registered_image_not_in_candidate"]
                )
            registered = sorted({image_to_frame[name] for name in registered_names})
        except Native360ColmapRunnerError as exc:
            failure_code = (
                "weak_registration"
                if "native_colmap_runner_no_registered_images" in exc.codes
                else "malformed_output"
            )
            blockers.extend(exc.codes)
    all_frame_ids = sorted(set(image_to_frame.values()))
    rejected = sorted(set(all_frame_ids) - set(registered))
    output_rows = _output_digest_rows(staging)
    if failure_code is None and not output_rows:
        failure_code = "malformed_output"
        blockers.append("native_colmap_runner_outputs_missing")

    duration = max(0.0, time.monotonic() - started)
    original_files = [
        {"artifact_id": row["relative_path"], "digest": row["digest"]}
        for row in admitted["original_file_references"]
    ]
    result_value = {
        "stable_run_identity": admitted["stable_run_identity"],
        "source_capture_identity": admitted["source_capture_identity"],
        "source_capture_digest": admitted["source_capture_digest"],
        "original_file_references": original_files,
        "producing_method": NATIVE_360_COLMAP_RUNNER_VERSION,
        "implementation_version": NATIVE_360_COLMAP_RUNNER_VERSION,
        "container_image_digest": admitted["container_image_digest"],
        "source_commit_sha": admitted["source_commit_sha"],
        "deterministic_configuration_digest": admitted["deterministic_configuration_digest"],
        "input_digests": admitted["input_digests"],
        "output_digests": output_rows,
        "train_heldout_split_digest": admitted["train_heldout_split_digest"],
        "camera_calibration_binding": admitted["camera_calibration_binding"],
        "coordinate_frame_declaration": admitted["coordinate_frame_declaration"],
        "units": "unknown",
        "metric_scale_status": "anchor_required",
        "provider_runtime_identity": admitted["provider_runtime_identity"],
        "cost_usd": 0.0,
        "duration_seconds": duration,
        "authority_used": admitted["authority_used"],
        "warnings": sorted(
            set(admitted["warnings"]) | {"metric_scale_anchor_required_after_pose_estimation"}
        ),
        "blockers": sorted(set(blockers)),
        "proof_effect": "calibrated_trajectory_candidate_only",
        "claim_ceiling": "calibrated_camera_trajectory",
        "parent_artifact_or_event": {
            "native_360_colmap_execution_plan_digest": admitted[
                "native_360_colmap_execution_plan_digest"
            ]
        },
        "timestamp": timestamp,
        "pose_estimation_request_digest": admitted["pose_estimation_request_digest"],
        "status": "succeeded" if failure_code is None else "failed",
        "failure_code": failure_code,
        "registered_observation_ids": registered,
        "rejected_observation_ids": rejected,
        "heldout_labels_included": False,
        "candidate_self_graded": False,
        "native_360_colmap_execution_plan_digest": admitted[
            "native_360_colmap_execution_plan_digest"
        ],
        "command_receipts": command_receipts,
        "legal_next_actions": (
            ["request_metric_anchor", "preserve_evidence_and_stop"]
            if failure_code is None
            else ["diagnose_reconstruction_failure", "preserve_evidence_and_stop"]
        ),
    }
    try:
        result = build_pose_estimation_result(result_value)
    except ReconstructionWorkerContractError as exc:
        raise Native360ColmapRunnerError(
            ["native_colmap_runner_result_contract_invalid", *exc.codes]
        ) from exc
    _result_path(staging).write_text(canonical_json(result) + "\n", encoding="utf-8")
    try:
        os.rename(staging, final_root)
    except FileExistsError as exc:
        replay = _validated_replay(final_root, admitted)
        if replay is not None:
            return replay
        raise Native360ColmapRunnerError(["native_colmap_runner_publish_conflict"]) from exc
    return result


def build_native_360_colmap_pose_estimator_service(
    *,
    plan: Mapping[str, Any],
    input_root: str | Path,
    timestamp: str,
    runner: Runner | None = None,
    runner_identity_digest: str | None = None,
    maximum_input_bytes: int = 256 * 1024 * 1024,
    maximum_command_output_bytes: int = 8 * 1024 * 1024,
    maximum_command_timeout_seconds: float = 3600.0,
) -> Callable[..., dict[str, Any]]:
    """Build the trusted callable injected behind ``run_pose_estimation``."""

    admitted = _validated_plan(plan)
    if (
        not str(timestamp or "").strip()
        or isinstance(maximum_input_bytes, bool)
        or not isinstance(maximum_input_bytes, int)
        or maximum_input_bytes <= 0
        or isinstance(maximum_command_output_bytes, bool)
        or not isinstance(maximum_command_output_bytes, int)
        or maximum_command_output_bytes <= 0
        or isinstance(maximum_command_timeout_seconds, bool)
        or not isinstance(maximum_command_timeout_seconds, (int, float))
        or maximum_command_timeout_seconds <= 0
        or (runner is None and runner_identity_digest is not None)
        or (runner is not None and not _digest(runner_identity_digest))
    ):
        raise Native360ColmapRunnerError(["native_colmap_service_binding_invalid"])
    lexical_input_root = Path(input_root)
    if lexical_input_root.is_symlink():
        raise Native360ColmapRunnerError(["native_colmap_runner_root_symlink_forbidden"])
    resolved_input_root = lexical_input_root.resolve()
    if not resolved_input_root.is_dir():
        raise Native360ColmapRunnerError(["native_colmap_runner_execution_bounds_invalid"])
    runtime_binding_digest = canonical_digest(
        {
            "runner_version": NATIVE_360_COLMAP_RUNNER_VERSION,
            "execution_plan_digest": admitted["native_360_colmap_execution_plan_digest"],
            "input_root_identity_digest": canonical_digest(
                {"resolved_input_root": str(resolved_input_root)}
            ),
            "maximum_input_bytes": maximum_input_bytes,
            "maximum_command_output_bytes": maximum_command_output_bytes,
            "maximum_command_timeout_seconds": maximum_command_timeout_seconds,
            "result_timestamp": timestamp,
            "runner_kind": "default_subprocess" if runner is None else "injected_test_runtime",
            "runner_identity_digest": runner_identity_digest,
        }
    )

    def estimator(*, request: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
        try:
            accepted_request = build_pose_estimation_request(request)
        except ReconstructionWorkerContractError as exc:
            raise Native360ColmapRunnerError(
                ["native_colmap_service_pose_request_invalid"]
            ) from exc
        if (
            accepted_request["pose_estimation_request_digest"]
            != admitted["pose_estimation_request_digest"]
        ):
            raise Native360ColmapRunnerError(
                ["native_colmap_service_pose_request_binding_mismatch"]
            )
        return execute_native_360_colmap_plan(
            plan=admitted,
            input_root=resolved_input_root,
            artifact_root=output_root,
            timestamp=timestamp,
            runner=runner,
            maximum_input_bytes=maximum_input_bytes,
            maximum_command_output_bytes=maximum_command_output_bytes,
            maximum_command_timeout_seconds=maximum_command_timeout_seconds,
        )

    setattr(estimator, "blueprint_runtime_digest", runtime_binding_digest)
    return estimator


__all__ = [
    "NATIVE_360_COLMAP_RUNNER_VERSION",
    "Native360ColmapRunnerError",
    "build_native_360_colmap_pose_estimator_service",
    "execute_native_360_colmap_plan",
]
