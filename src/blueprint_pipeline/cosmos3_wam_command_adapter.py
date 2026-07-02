"""Command adapter for Cosmos3-Nano action-conditioned WAM rollout generation.

The adapter is the ``cosmos3_wam`` process boundary behind the swappable
provider-command interface. It mirrors the OSCAR adapter honesty mechanics:

- The operator supplies the Cosmos3 source tree, checkpoint path, and run
  gates through environment variables or CLI flags; nothing auto-runs.
- The adapter emits the trusted output schema
  ``cosmos3_wam_command_adapter.v1`` and self-reports its backbone as
  ``base_model`` so the provider runtime can hard-fail on family mismatches.
- ``learned_wam_model_ran`` is set only when the checkpoint and source
  identity verify as Cosmos3-Nano and a reviewable generated MP4 exists.
- The SC3-Eval recipe metadata (80/10/10 forward/cross-view/inverse mixture,
  predict-24/execute-16 horizon decoupling) is recorded as declared operator
  config, never as proof that the recipe was trained or executed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .oscar_cosmos_wam_command_adapter import (
    _failure_signals,
    _materialize_cosmos_input_package,
    _read_json,
    _redacted_argv,
    _runtime_env,
    _string,
    _write_json,
)
from .wam_generated_video_review import validate_generated_mp4_for_review


ADAPTER_ID = "blueprint_cosmos3_nano_wam_command_adapter"
SCHEMA_VERSION = "cosmos3_wam_command_adapter.v1"
SUBSTRATE = "cosmos3_wam"
EXPECTED_BASE_MODEL = "Cosmos3-Nano"
EXPECTED_MODEL_FAMILY = "Cosmos 3"
LOCAL_MODEL_GATE_ENV = "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL"
DEFAULT_MODEL = "cosmos3/nano/action-cond"
DEFAULT_ENTRYPOINT_RELPATH = "examples/action_conditioned.py"

CHECKPOINT_IDENTITY_FILENAMES = (
    "blueprint_checkpoint_identity.json",
    "cosmos3_checkpoint_manifest.json",
    "checkpoint_manifest.json",
    "model_index.json",
    "config.json",
    "metadata.json",
)
CHECKPOINT_IDENTITY_KEYS = (
    "base_model",
    "model_family",
    "model_name",
    "model_id",
    "_name_or_path",
    "architecture",
    "backbone",
)
# Any declared identity that matches one of these markers without also
# matching Cosmos3-Nano is a wrong model family and must fail closed.
WRONG_FAMILY_MARKERS = (
    "predict2",
    "predict-2",
    "oscar",
    "cosmos1",
    "cosmos2",
    "cosmos3super",
    "cosmos3edge",
)
COSMOS3_NANO_IDENTITY_TOKEN = "cosmos3nano"

# Declared upstream recipe per SC3-Eval (arXiv 2606.18610). This block is
# configuration metadata only: emitting it never claims the mixture was
# trained, reproduced, or validated by this adapter invocation.
SC3_RECIPE_DECLARED_CONFIG = {
    "schema_version": "sc3_recipe_declared_config.v1",
    "recipe_id": "sc3_eval_self_consistency_recipe",
    "recipe_source": "arXiv:2606.18610 (SC3-Eval), Cosmos3-Nano backbone",
    "training_mixture": {
        "forward_dynamics": 0.8,
        "cross_view": 0.1,
        "inverse_dynamics": 0.1,
    },
    "horizon_decoupling": {
        "predict_horizon_frames": 24,
        "execute_horizon_frames": 16,
    },
    "claim_boundary": {
        "recipe_metadata_is_operator_declared_config": True,
        "recipe_metadata_is_execution_or_training_proof": False,
        "recipe_declared_config_does_not_prove_rank_fidelity": True,
    },
}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_existing_path(paths: Sequence[str]) -> Path | None:
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _source_root_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_COSMOS3_WAM_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_COSMOS3_NANO_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_COSMOS3_SOURCE_ROOT", ""),
        ]
    )


def _checkpoint_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_COSMOS3_WAM_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_COSMOS3_NANO_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", ""),
        ]
    )


def _normalized_identity(value: Any) -> str:
    text = _string(value).lower()
    return "".join(char for char in text if char.isalnum() or char == ".")


def _identity_value_is_cosmos3_nano(value: Any) -> bool:
    return COSMOS3_NANO_IDENTITY_TOKEN in _normalized_identity(value).replace(".", "")


def _identity_value_is_wrong_family(value: Any) -> bool:
    normalized = _normalized_identity(value).replace(".", "").replace("-", "")
    if COSMOS3_NANO_IDENTITY_TOKEN in normalized:
        return False
    return any(
        marker.replace("-", "").replace(".", "") in normalized
        for marker in WRONG_FAMILY_MARKERS
    )


def _identity_candidate_files(checkpoint: Path) -> list[Path]:
    roots = [checkpoint] if checkpoint.is_dir() else [checkpoint.parent]
    files: list[Path] = []
    for root in roots:
        for name in CHECKPOINT_IDENTITY_FILENAMES:
            candidate = root / name
            if candidate.is_file():
                files.append(candidate)
    return files


def checkpoint_identity_probe(checkpoint: Path) -> dict[str, Any]:
    """Machine-check the operator-supplied checkpoint's declared identity."""

    declared: list[str] = []
    scanned: list[str] = []
    for identity_file in _identity_candidate_files(checkpoint):
        scanned.append(str(identity_file))
        try:
            payload = _read_json(identity_file)
        except (json.JSONDecodeError, OSError):
            continue
        for key in CHECKPOINT_IDENTITY_KEYS:
            value = _string(payload.get(key))
            if value and value not in declared:
                declared.append(value)
    verified = any(_identity_value_is_cosmos3_nano(value) for value in declared)
    wrong_family_values = [
        value for value in declared if _identity_value_is_wrong_family(value)
    ]
    return {
        "schema_version": "cosmos3_checkpoint_identity_probe.v1",
        "checkpoint_path": str(checkpoint),
        "identity_files_scanned": scanned,
        "declared_identity_values": declared,
        "expected_base_model": EXPECTED_BASE_MODEL,
        "checkpoint_identity_verified": verified,
        "wrong_model_family_detected": bool(wrong_family_values) and not verified,
        "wrong_family_identity_values": wrong_family_values if not verified else [],
    }


def source_identity_probe(source_root: Path) -> dict[str, Any]:
    """Machine-check that the source tree declares a Cosmos 3 lineage."""

    declared: list[str] = []
    scanned: list[str] = []
    for name in ("pyproject.toml", "setup.py", "setup.cfg", "README.md"):
        candidate = source_root / name
        if not candidate.is_file():
            continue
        scanned.append(str(candidate))
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        declared.append(text[:20000])
    combined = "\n".join(declared).lower()
    has_cosmos3_marker = "cosmos3" in combined.replace("-", "").replace("_", "")
    has_wrong_family_marker = any(
        marker in combined.replace("-", "").replace("_", "")
        for marker in ("cosmospredict2", "predict2.5", "oscarpublic")
    )
    verified = has_cosmos3_marker
    return {
        "schema_version": "cosmos3_source_identity_probe.v1",
        "source_root": str(source_root),
        "identity_files_scanned": scanned,
        "source_identity_verified": verified,
        "wrong_model_family_detected": has_wrong_family_marker and not verified,
    }


def _probe_modules() -> list[str]:
    configured = os.getenv("BLUEPRINT_COSMOS3_WAM_PROBE_MODULES", "")
    values = configured.split(",") if configured else ["torch"]
    return [_string(value) for value in values if _string(value)]


def _run_import_probe(
    *, python: str, source_root: Path, timeout_seconds: float
) -> dict[str, Any]:
    modules = _probe_modules()
    started = time.monotonic()
    result = subprocess.run(
        [
            python,
            "-c",
            (
                "import json, importlib.util, sys; "
                "mods = json.loads(sys.argv[1]); "
                "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))"
            ),
            json.dumps(modules),
        ],
        cwd=str(source_root),
        env=_runtime_env(source_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    available: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            available = json.loads(result.stdout)
        except json.JSONDecodeError:
            available = {}
    missing = [name for name, present in available.items() if not present]
    return {
        "schema_version": "cosmos3_runtime_import_probe.v1",
        "status": "completed" if result.returncode == 0 and not missing else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "module_available": available,
        "blockers": []
        if result.returncode == 0 and not missing
        else ["blocked_missing_cosmos3_runtime_import"],
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
    }


def _entrypoint_relpath() -> str:
    return (
        _string(os.getenv("BLUEPRINT_COSMOS3_WAM_ENTRYPOINT"))
        or DEFAULT_ENTRYPOINT_RELPATH
    )


def _run_cosmos3(
    *,
    python: str,
    source_root: Path,
    checkpoint: Path,
    package_manifest: Mapping[str, Any],
    output_dir: Path,
    model: str,
    timeout_seconds: float,
    extra_args: Sequence[str],
) -> dict[str, Any]:
    entrypoint = source_root / _entrypoint_relpath()
    inference_params = Path(_string(package_manifest.get("inference_params_path")))
    argv = [
        python,
        str(entrypoint),
        "-i",
        str(inference_params),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        str(checkpoint),
        "--model",
        model,
    ]
    argv.extend(extra_args)
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=str(source_root),
            env=_runtime_env(source_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": "cosmos3_subprocess_result.v1",
            "status": "blocked",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "argv_redacted": _redacted_argv(argv, checkpoint),
            "stdout_size_bytes": len(exc.stdout or ""),
            "stderr_size_bytes": len(exc.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(exc.stderr),
            "blockers": ["cosmos3_wam_command_timeout"],
        }
    failure_signals = _failure_signals(result.stdout or "", result.stderr or "")
    blockers = [] if result.returncode == 0 else ["cosmos3_wam_command_nonzero"]
    blockers.extend(signal for signal in failure_signals if signal not in blockers)
    return {
        "schema_version": "cosmos3_subprocess_result.v1",
        "status": "completed" if result.returncode == 0 else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "argv_redacted": _redacted_argv(argv, checkpoint),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": blockers,
    }


def _run_gate_status() -> dict[str, Any]:
    enabled = _env_truthy(LOCAL_MODEL_GATE_ENV)
    return {
        "local_model_gate_env": LOCAL_MODEL_GATE_ENV,
        "local_model_gate_enabled": enabled,
        "auto_run_allowed_without_gate": False,
    }


def _base_payload(
    *,
    status: str,
    blockers: Sequence[str],
    source_root: Path | None,
    checkpoint: Path | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "adapter_id": ADAPTER_ID,
        "evaluation_substrate": SUBSTRATE,
        "expected_base_model": EXPECTED_BASE_MODEL,
        "run_gates": _run_gate_status(),
        "sc3_recipe_declared_config": dict(SC3_RECIPE_DECLARED_CONFIG),
        "blockers": list(blockers),
        "source_root": str(source_root) if source_root else None,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "learned_wam_model_ran": False,
        "fresh_model_command_executed_this_invocation": False,
        "fresh_model_run_claimed": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    payload.update(dict(extra or {}))
    return payload


def _rollout_payload(
    *,
    package_manifest: Mapping[str, Any],
    checkpoint: Path,
    source_root: Path,
    subprocess_detail: Mapping[str, Any],
    model: str,
    checkpoint_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    save_root = Path(_string(package_manifest.get("save_root")))
    generated_videos = sorted(path.resolve() for path in save_root.rglob("*.mp4"))
    video_validations = [
        validate_generated_mp4_for_review(path) for path in generated_videos
    ]
    subprocess_completed = subprocess_detail.get("status") == "completed"
    identity_verified = bool(
        checkpoint_identity.get("checkpoint_identity_verified")
        and source_identity.get("source_identity_verified")
    )
    rollouts = []
    for index, (path, validation) in enumerate(
        zip(generated_videos, video_validations), start=1
    ):
        if validation.get("status") != "completed":
            continue
        rollouts.append(
            {
                "rollout_id": f"cosmos3_wam_rollout_{index:04d}",
                "policy_id": ADAPTER_ID,
                "model_candidate": os.getenv("BLUEPRINT_WAM_MODEL_CANDIDATE")
                or SUBSTRATE,
                "base_model": EXPECTED_BASE_MODEL,
                "model": model,
                "generated_video_path": str(path),
                "source_review_video_path": package_manifest.get(
                    "source_review_video_path"
                ),
                "source_camera": package_manifest.get("source_camera"),
                "scenario_eval_run_id": package_manifest.get("scenario_eval_run_id"),
                "task_id": package_manifest.get("task_id"),
                "spawn_id": package_manifest.get("spawn_id"),
                "model_rollout_confidence": None,
                "generated_rollout_termination_reason": "cosmos3_command_completed",
                "success_label_source": "generated_video_requires_review",
                "generated_video_review_validation": validation,
            }
        )
    status = "completed" if rollouts else "blocked"
    validation_blockers = sorted(
        {
            str(blocker)
            for validation in video_validations
            for blocker in validation.get("blockers", [])
            if str(blocker)
        }
    )
    blockers = (
        []
        if rollouts
        else [
            "blocked_generated_cosmos3_mp4_not_reviewable"
            if generated_videos
            else "blocked_no_generated_cosmos3_mp4",
            *validation_blockers,
        ]
    )
    model_ran = bool(rollouts and subprocess_completed and identity_verified)
    return _base_payload(
        status=status,
        blockers=blockers,
        source_root=source_root,
        checkpoint=checkpoint,
        extra={
            "base_model": EXPECTED_BASE_MODEL,
            "rollouts": rollouts,
            "generated_video_count": len(generated_videos),
            "generated_video_review_validations": video_validations,
            "model_provenance": {
                "candidate": os.getenv("BLUEPRINT_WAM_MODEL_CANDIDATE") or SUBSTRATE,
                "base_model": EXPECTED_BASE_MODEL,
                "model_family": EXPECTED_MODEL_FAMILY,
                "source_root": str(source_root),
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
                "model": model,
                "checkpoint_identity_probe": dict(checkpoint_identity),
                "source_identity_probe": dict(source_identity),
            },
            "input_package": dict(package_manifest),
            "cosmos3_subprocess": dict(subprocess_detail),
            "fresh_model_command_executed_this_invocation": bool(
                rollouts and subprocess_completed
            ),
            "fresh_model_run_claimed": model_ran,
            "learned_wam_model_ran": model_ran,
            "truth_boundary": {
                "generated_video_is_model_output": bool(
                    rollouts and subprocess_completed
                ),
                "checkpoint_identity_verified_as_cosmos3_nano": bool(
                    checkpoint_identity.get("checkpoint_identity_verified")
                ),
                "source_identity_verified_as_cosmos3": bool(
                    source_identity.get("source_identity_verified")
                ),
                "cosmos3_identity_match_required_for_learned_wam_claim": True,
                "sc3_recipe_metadata_is_declared_config_not_proof": True,
                "generated_rollout_not_physical_robot_proof": True,
                "generated_success_label_requires_external_vlm_or_human_judge": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
            },
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--python",
        default=os.getenv("BLUEPRINT_COSMOS3_WAM_PYTHON") or sys.executable,
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--model",
        default=os.getenv("BLUEPRINT_COSMOS3_WAM_MODEL") or DEFAULT_MODEL,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("BLUEPRINT_COSMOS3_WAM_CHUNK_SIZE", "16")),
    )
    parser.add_argument(
        "--resolution",
        default=os.getenv("BLUEPRINT_COSMOS3_WAM_RESOLUTION") or "256,320",
    )
    parser.add_argument(
        "--guidance",
        type=int,
        default=int(os.getenv("BLUEPRINT_COSMOS3_WAM_GUIDANCE", "0")),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.getenv("BLUEPRINT_COSMOS3_WAM_NUM_STEPS", "35")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("BLUEPRINT_COSMOS3_WAM_TIMEOUT_SECONDS", "3600")),
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--probe-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root
        else _source_root_from_env()
    )
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else _checkpoint_from_env()
    )
    output_path = Path(
        os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")
    ).resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_path.parent / "cosmos3_wam_command_workspace"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    if source_root is None:
        blockers.append("blocked_missing_cosmos3_source_root")
    elif not (source_root / _entrypoint_relpath()).is_file():
        blockers.append("blocked_missing_cosmos3_entrypoint")
    if checkpoint is None:
        blockers.append("blocked_missing_cosmos3_checkpoint")
    elif not checkpoint.exists():
        blockers.append("blocked_configured_cosmos3_checkpoint_path_missing")
    if not shutil.which(args.python) and not Path(args.python).expanduser().is_file():
        blockers.append("blocked_configured_python_missing")

    if blockers:
        payload = _base_payload(
            status="blocked",
            blockers=blockers,
            source_root=source_root,
            checkpoint=checkpoint,
        )
        _write_json(output_path, payload)
        return payload

    assert source_root is not None
    assert checkpoint is not None

    checkpoint_identity = checkpoint_identity_probe(checkpoint)
    source_identity = source_identity_probe(source_root)
    _write_json(work_dir / "cosmos3_checkpoint_identity_probe.json", checkpoint_identity)
    _write_json(work_dir / "cosmos3_source_identity_probe.json", source_identity)

    identity_blockers: list[str] = []
    if checkpoint_identity.get("wrong_model_family_detected"):
        identity_blockers.append("blocked_wrong_model_family_checkpoint_for_cosmos3_wam")
    elif not checkpoint_identity.get("checkpoint_identity_verified"):
        identity_blockers.append("blocked_cosmos3_checkpoint_identity_unverified")
    if source_identity.get("wrong_model_family_detected"):
        identity_blockers.append("blocked_wrong_model_family_source_for_cosmos3_wam")
    if identity_blockers:
        payload = _base_payload(
            status="blocked",
            blockers=identity_blockers,
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload

    probe = _run_import_probe(
        python=args.python,
        source_root=source_root,
        timeout_seconds=min(args.timeout_seconds, 120.0),
    )
    _write_json(work_dir / "cosmos3_import_probe.json", probe)
    if args.probe_only:
        payload = _base_payload(
            status=probe["status"],
            blockers=list(probe.get("blockers", [])),
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "probe_only": True,
                "import_probe": probe,
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload
    if probe["status"] != "completed":
        payload = _base_payload(
            status="blocked",
            blockers=list(probe.get("blockers", [])),
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "import_probe": probe,
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload

    if not _env_truthy(LOCAL_MODEL_GATE_ENV):
        payload = _base_payload(
            status="blocked",
            blockers=[f"blocked_{LOCAL_MODEL_GATE_ENV}_not_enabled"],
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "import_probe": probe,
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload

    rollout_input = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"]).expanduser().resolve()
    rollout_manifest = _read_json(rollout_input)
    package_manifest = _materialize_cosmos_input_package(
        rollout_manifest=rollout_manifest,
        work_dir=work_dir,
        chunk_size=args.chunk_size,
        resolution=args.resolution,
        guidance=args.guidance,
        num_steps=args.num_steps,
    )
    cosmos3_output_dir = work_dir / "cosmos3_output"
    subprocess_detail = _run_cosmos3(
        python=args.python,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_dir=cosmos3_output_dir,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        extra_args=[item for value in args.extra_arg for item in shlex.split(value)],
    )
    payload = _rollout_payload(
        package_manifest=package_manifest,
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail=subprocess_detail,
        model=args.model,
        checkpoint_identity=checkpoint_identity,
        source_identity=source_identity,
    )
    if subprocess_detail["status"] != "completed" and not payload["rollouts"]:
        payload["status"] = "blocked"
        payload["blockers"] = list(subprocess_detail.get("blockers") or [])
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(argv)
    except Exception as exc:
        output_path = Path(
            os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")
        ).resolve()
        payload = _base_payload(
            status="blocked",
            blockers=[f"cosmos3_wam_adapter_exception:{type(exc).__name__}"],
            source_root=None,
            checkpoint=None,
        )
        _write_json(output_path, payload)
    print(json.dumps({"adapter_id": ADAPTER_ID, "status": payload.get("status")}, sort_keys=True))
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
