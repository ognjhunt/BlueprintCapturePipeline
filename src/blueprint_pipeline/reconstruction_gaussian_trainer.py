"""Execute pinned candidate-only 3DGRUT MCMC training inside the GPU worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_training_request,
    build_training_result,
)


THREEDGRUT_ROOT = Path("/opt/3dgrut")
THREEDGRUT_CONFIG = "apps/colmap_3dgut_mcmc.yaml"


class GaussianTrainerRuntimeError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_dataset(root: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path))
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in {"", ".", "..", "evaluator_hidden", "held_out"} for part in relative.parts)
    ):
        raise GaussianTrainerRuntimeError("training_dataset_path_unsafe_or_hidden")
    dataset = (root.resolve() / relative).resolve()
    if root.resolve() not in dataset.parents or dataset.is_symlink():
        raise GaussianTrainerRuntimeError("training_dataset_path_escape_or_symlink")
    required = [
        dataset / "images",
        dataset / "sparse/0/cameras.txt",
        dataset / "sparse/0/images.txt",
    ]
    if any(not path.exists() or path.is_symlink() for path in required):
        raise GaussianTrainerRuntimeError("training_dataset_colmap_layout_incomplete")
    if any(
        "held_out" in path.parts or "evaluator_hidden" in path.parts for path in dataset.rglob("*")
    ):
        raise GaussianTrainerRuntimeError("training_dataset_hidden_heldout_present")
    return dataset


def _default_runner(
    command: Sequence[str], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=False, capture_output=True, text=True, timeout=timeout_seconds
    )


def _failure(text: str, *, timed_out: bool = False) -> str:
    lowered = text.lower()
    if timed_out:
        return "training_timeout"
    if "out of memory" in lowered:
        return "gpu_out_of_memory"
    if "nan" in lowered or "non-finite" in lowered or "nonfinite" in lowered:
        return "nan_output"
    return "training_divergence"


def _latest_checkpoint(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.name == "ckpt_last.pt":
            return path

    def iteration(path: Path) -> int:
        matches = re.findall(r"\d+", path.stem)
        return int(matches[-1]) if matches else -1

    return max(paths, key=iteration, default=None)


def _stable_checkpoint(run_dir: Path) -> Path | None:
    stable = run_dir / "checkpoint_last.pt"
    if stable.is_file() and not stable.is_symlink():
        return stable
    latest = _latest_checkpoint(sorted((run_dir / "training").rglob("ckpt*.pt")))
    if latest is None:
        return None
    shutil.copyfile(latest, stable)
    return stable


def _result_base(request: Mapping[str, Any], *, duration: float) -> dict[str, Any]:
    lineage = {
        key: json.loads(json.dumps(request[key]))
        for key in (
            "stable_run_identity",
            "source_capture_identity",
            "source_capture_digest",
            "original_file_references",
            "container_image_digest",
            "source_commit_sha",
            "deterministic_configuration_digest",
            "input_digests",
            "train_heldout_split_digest",
            "camera_calibration_binding",
            "coordinate_frame_declaration",
            "units",
            "metric_scale_status",
            "provider_runtime_identity",
            "authority_used",
            "parent_artifact_or_event",
        )
    }
    lineage.update(
        producing_method="blueprint.pinned_3dgrut_3dgut_mcmc_trainer",
        implementation_version="1.0.0",
        cost_usd=0.0,
        duration_seconds=max(0.0, float(duration)),
        timestamp=utc_now_iso(),
        proof_effect="appearance_asset_candidate_only",
        claim_ceiling="appearance_reconstruction",
        reconstruction_training_request_digest=request["reconstruction_training_request_digest"],
        heldout_labels_included=False,
        candidate_self_graded=False,
        registered_observation_ids=[],
        rejected_observation_ids=[],
        training_metrics={"heldout_metrics_computed": False},
        peak_resource_use={"measurement_status": "worker_runtime_measurement_unavailable"},
    )
    return lineage


def _verified_replay(path: Path, *, request_digest: str) -> dict[str, Any]:
    try:
        result = build_training_result(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ReconstructionWorkerContractError) as exc:
        raise GaussianTrainerRuntimeError("training_replay_result_invalid") from exc
    if result["reconstruction_training_request_digest"] != request_digest:
        raise GaussianTrainerRuntimeError("training_replay_request_binding_mismatch")
    run_dir = path.parent.resolve()
    for reference in [*result["output_digests"], *result["checkpoint_references"]]:
        relative = Path(str(reference.get("artifact_id") or ""))
        if (
            not relative.parts
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise GaussianTrainerRuntimeError("training_replay_artifact_path_invalid")
        artifact = (run_dir / relative).resolve()
        if run_dir not in artifact.parents or not artifact.is_file() or artifact.is_symlink():
            raise GaussianTrainerRuntimeError("training_replay_artifact_missing_or_unsafe")
        if _sha256(artifact) != reference.get("digest"):
            raise GaussianTrainerRuntimeError("training_replay_artifact_digest_mismatch")
    return result


def _record_interrupted_run(
    *,
    run_dir: Path,
    request: Mapping[str, Any],
    observation_ids: Sequence[str],
    rejected_ids: Sequence[str],
) -> dict[str, Any]:
    log_path = run_dir / "training.log"
    if not log_path.is_file() or log_path.is_symlink():
        log_path.write_text(
            "recovered interrupted run without a terminal training receipt\n",
            encoding="utf-8",
        )
    checkpoint = _stable_checkpoint(run_dir)
    result = _result_base(request, duration=0.0)
    result.update(
        status="interrupted",
        failure_code="provider_interruption",
        output_digests=[{"artifact_id": "training.log", "digest": _sha256(log_path)}],
        checkpoint_references=(
            [{"artifact_id": "checkpoint_last.pt", "digest": _sha256(checkpoint)}]
            if checkpoint is not None
            else []
        ),
        registered_observation_ids=list(observation_ids),
        rejected_observation_ids=list(rejected_ids),
        warnings=["interrupted_training_evidence_preserved"],
        blockers=["provider_interruption"],
        legal_next_actions=(
            ["resume_bound_checkpoint"]
            if checkpoint is not None
            else ["preserve_evidence_and_stop"]
        ),
    )
    normalized = build_training_result(result)
    write_json(run_dir / "reconstruction_training_result.json", normalized)
    return normalized


def run_gaussian_reconstruction_training(
    *,
    training_request: Mapping[str, Any],
    dataset_export: Mapping[str, Any],
    artifact_root: str | Path,
    output_root: str | Path,
    command_runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
    | None = None,
    python_executable: str = "/opt/venv/bin/python",
    threedgrut_root: str | Path = THREEDGRUT_ROOT,
) -> dict[str, Any]:
    """Run 3DGUT without exposing or evaluating the independent held-out set."""

    try:
        request = build_training_request(training_request)
    except ReconstructionWorkerContractError as exc:
        raise GaussianTrainerRuntimeError("training_request_contract_invalid") from exc
    if dataset_export.get("schema_version") != "colmap_training_dataset_export_result.v1":
        raise GaussianTrainerRuntimeError("training_dataset_export_contract_invalid")
    if (
        dataset_export.get("hidden_heldout_pixels_included") is not False
        or dataset_export.get("trainer_self_grading_permitted") is not False
        or dataset_export.get("colmap_training_dataset_digest")
        != request["reconstruction_dataset_digest"]
    ):
        raise GaussianTrainerRuntimeError("training_dataset_binding_or_isolation_invalid")
    observation_ids = dataset_export.get("observation_ids")
    rejected_ids = dataset_export.get("rejected_observation_ids")
    if (
        not isinstance(observation_ids, list)
        or not observation_ids
        or any(not isinstance(value, str) or not value for value in observation_ids)
        or not isinstance(rejected_ids, list)
        or any(not isinstance(value, str) or not value for value in rejected_ids)
    ):
        raise GaussianTrainerRuntimeError("training_dataset_observation_ledger_invalid")
    if (
        len(set(observation_ids)) != len(observation_ids)
        or len(set(rejected_ids)) != len(rejected_ids)
        or set(observation_ids) & set(rejected_ids)
    ):
        raise GaussianTrainerRuntimeError("training_dataset_observation_ledger_invalid")
    if request["method_profile_id"] not in {
        "gsplat_3dgut_mcmc_v1",
        "nvidia_3dgrut_3dgut_mcmc_v1",
    }:
        raise GaussianTrainerRuntimeError("trainer_method_not_executable_by_3dgrut_adapter")
    image = str(os.environ.get("BLUEPRINT_CONTAINER_IMAGE_DIGEST") or "")
    if image and image != request["container_image_digest"]:
        raise GaussianTrainerRuntimeError("training_worker_image_digest_mismatch")
    dataset = _safe_dataset(Path(artifact_root), str(dataset_export.get("relative_path") or ""))
    run_dir = Path(output_root).resolve() / request["reconstruction_training_request_digest"][7:23]
    prior_run = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "reconstruction_training_result.json"
    if result_path.exists():
        return _verified_replay(
            result_path,
            request_digest=request["reconstruction_training_request_digest"],
        )
    if prior_run:
        return _record_interrupted_run(
            run_dir=run_dir,
            request=request,
            observation_ids=observation_ids,
            rejected_ids=rejected_ids,
        )
    log_path = run_dir / "training.log"
    log_path.write_text("training process started; terminal result pending\n", encoding="utf-8")
    ply_path = run_dir / "appearance_candidate.ply"
    iterations = int(request["iteration_budget"])
    command = [
        python_executable,
        str(Path(threedgrut_root) / "train.py"),
        "--config-name",
        THREEDGRUT_CONFIG,
        f"path={dataset}",
        f"out_dir={run_dir}",
        "experiment_name=training",
        f"n_iterations={iterations}",
        f"checkpoint.iterations=[{iterations}]",
        f"seed_initialization={int(request['random_seed'])}",
        "dataset.test_split_interval=-1",
        "dataset.load_exif=false",
        "test_last=false",
        "compute_extra_metrics=false",
        "val_frequency=999999999",
        "with_gui=false",
        "with_viser_gui=false",
        "use_wandb=false",
        "export_ply.enabled=true",
        f"export_ply.path={ply_path}",
    ]
    started = time.monotonic()
    timed_out = False
    try:
        completed = (command_runner or _default_runner)(command, float(request["timeout_seconds"]))
        output_text = (completed.stdout or "") + (completed.stderr or "")
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output_text = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + (
            (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        )
        returncode = 124
    except (OSError, subprocess.SubprocessError) as exc:
        output_text = repr(exc)
        returncode = 127
    duration = time.monotonic() - started
    log_path.write_text(output_text[-2_000_000:], encoding="utf-8")
    stable_checkpoint = _stable_checkpoint(run_dir)
    checkpoints = [stable_checkpoint] if stable_checkpoint is not None else []
    result = _result_base(request, duration=duration)
    result["registered_observation_ids"] = list(observation_ids)
    result["rejected_observation_ids"] = list(rejected_ids)
    result["checkpoint_references"] = [
        {"artifact_id": path.relative_to(run_dir).as_posix(), "digest": _sha256(path)}
        for path in checkpoints
    ]
    log_ref = {"artifact_id": "training.log", "digest": _sha256(log_path)}
    if returncode == 0 and checkpoints and ply_path.is_file() and ply_path.stat().st_size > 0:
        result.update(
            status="succeeded",
            failure_code=None,
            output_digests=[
                log_ref,
                {"artifact_id": "appearance_candidate.ply", "digest": _sha256(ply_path)},
            ],
            warnings=["provider_cost_requires_allocator_receipt"],
            blockers=[],
            legal_next_actions=["preserve_evidence_and_stop"],
        )
    else:
        failure = (
            "worker_startup_failure"
            if returncode == 127
            else "checkpoint_acquisition_failure"
            if returncode == 0 and not checkpoints
            else "malformed_output"
            if returncode == 0
            else _failure(output_text, timed_out=timed_out)
        )
        result.update(
            status="timed_out" if timed_out else "failed",
            failure_code=failure,
            output_digests=[log_ref],
            warnings=["failed_training_evidence_preserved"],
            blockers=[failure],
            legal_next_actions=(
                ["resume_bound_checkpoint"] if checkpoints else ["preserve_evidence_and_stop"]
            ),
        )
    normalized = build_training_result(result)
    write_json(result_path, normalized)
    return normalized


def bind_gaussian_reconstruction_trainer(
    *,
    dataset_export: Mapping[str, Any],
    artifact_root: str | Path,
    command_runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
    | None = None,
    python_executable: str = "/opt/venv/bin/python",
    threedgrut_root: str | Path = THREEDGRUT_ROOT,
) -> Callable[..., dict[str, Any]]:
    """Bind trusted worker state to the digest-only registered supervisor tool."""

    return partial(
        _run_registered_gaussian_reconstruction_training,
        dataset_export=dataset_export,
        artifact_root=artifact_root,
        command_runner=command_runner,
        python_executable=python_executable,
        threedgrut_root=threedgrut_root,
    )


def _run_registered_gaussian_reconstruction_training(
    *,
    request: Mapping[str, Any],
    output_root: str | Path,
    dataset_export: Mapping[str, Any],
    artifact_root: str | Path,
    command_runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] | None,
    python_executable: str,
    threedgrut_root: str | Path,
) -> dict[str, Any]:
    return run_gaussian_reconstruction_training(
        training_request=request,
        dataset_export=dataset_export,
        artifact_root=artifact_root,
        output_root=output_root,
        command_runner=command_runner,
        python_executable=python_executable,
        threedgrut_root=threedgrut_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--dataset-export", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_gaussian_reconstruction_training(
        training_request=json.loads(args.request.read_text(encoding="utf-8")),
        dataset_export=json.loads(args.dataset_export.read_text(encoding="utf-8")),
        artifact_root=args.artifact_root,
        output_root=args.output_root,
    )
    print(json.dumps({"status": result["status"], "failure_code": result["failure_code"]}))
    return 0 if result["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GaussianTrainerRuntimeError",
    "bind_gaussian_reconstruction_trainer",
    "run_gaussian_reconstruction_training",
]
