"""Prepare, execute, and score the ADP-009 controlled known-background firebreak."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .decision_evidence_contracts import canonical_digest


CASE_SCHEMA_VERSION = "adp009_controlled_background_case.v1"
EXECUTION_SCHEMA_VERSION = "adp009_controlled_background_execution.v1"
SCORE_SCHEMA_VERSION = "adp009_controlled_background_score.v1"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 512
VIEW_COUNT = 4
DEPTH_MIN_M = 0.8
DEPTH_MAX_M = 1.6
LAMA_SOURCE_SHA256 = "sha256:9d7b65057ee3adecd70567d13f85d4a6972eece08671e38360cfd9f3a5a4263b"
LAMA_CHECKPOINT_SHA256 = "sha256:d7161bba4d68b438f9fa7f09dcb750a223804c300c68d214a5e0be16251fba8d"
LAMA_REPOSITORY = "https://github.com/advimman/lama"
LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
DOCKER_IMAGE = "blueprint/adp009-controlled-lama:cpu-v1"
DOCKER_IMAGE_ID = "sha256:8f794c9812e2dafba0a608132e26cd7a3a2f628dcf77c1879e6191cf64552934"
THRESHOLDS = {
    "rgb_mask_psnr_db_min": 22.0,
    "rgb_mask_ssim_min": 0.8,
    "rgb_inner_boundary_mae_255_max": 20.0,
    "depth_mask_rmse_m_max": 0.035,
    "depth_mask_p95_abs_m_max": 0.06,
    "depth_plane_rmse_m_max": 0.025,
}


class ControlledBackgroundError(ValueError):
    """The controlled-background firebreak is incomplete or invalid."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    root = root.resolve()
    if root not in resolved.parents or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ControlledBackgroundError(f"artifact_invalid:{role}")
    return {
        "role": role,
        "relative_path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ControlledBackgroundError(f"json_object_required:{path.name}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_receipt(value: Mapping[str, Any], field: str, error: str) -> None:
    if value.get(field) != canonical_digest(value, digest_field=field):
        raise ControlledBackgroundError(error)


def _verify_records(records: Any, root: Path, *, required_roles: set[str]) -> None:
    if not isinstance(records, list) or {row.get("role") for row in records} != required_roles:
        raise ControlledBackgroundError("artifact_inventory_invalid")
    for row in records:
        if not isinstance(row, Mapping):
            raise ControlledBackgroundError("artifact_record_invalid")
        relative = str(row.get("relative_path") or "")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ControlledBackgroundError("artifact_outside_case_root")
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ControlledBackgroundError(f"artifact_bytes_changed:{relative}")


def _clean_rgb(view_index: int) -> np.ndarray:
    y, x = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(np.float32)
    xn = x / (IMAGE_WIDTH - 1)
    yn = y / (IMAGE_HEIGHT - 1)
    phase = view_index * 0.37
    grain = 7.0 * np.sin(2 * np.pi * (xn * 3.2 + phase))
    grain += 4.0 * np.sin(2 * np.pi * (yn * 7.0 - xn * 0.6 + phase * 0.5))
    base = 194.0 + 24.0 * xn - 18.0 * yn + grain
    rgb = np.stack((base + 10.0, base + 2.0, base - 13.0), axis=-1)
    seam = np.mod((x + 35 * view_index) + 0.19 * y, 190.0)
    rgb[seam < 3.0] *= 0.72
    highlight = np.exp(-((xn - 0.35 - 0.04 * view_index) ** 2 + (yn - 0.35) ** 2) / 0.025)
    rgb += highlight[..., None] * np.array([18.0, 16.0, 12.0], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _clean_depth_m(view_index: int) -> np.ndarray:
    y, x = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(np.float32)
    xn = x / (IMAGE_WIDTH - 1)
    yn = y / (IMAGE_HEIGHT - 1)
    return (1.12 + 0.12 * yn + 0.025 * xn + 0.003 * view_index).astype(np.float32)


def _mask(view_index: int) -> np.ndarray:
    image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    center_x = 342 + 28 * view_index
    center_y = 292 - 10 * view_index
    width = 76 - 3 * view_index
    height = 176 - 5 * view_index
    left, right = center_x - width // 2, center_x + width // 2
    top, bottom = center_y - height // 2, center_y + height // 2
    radius = width // 2
    draw.rectangle((left, top + radius, right, bottom - radius), fill=255)
    draw.ellipse((left, top, right, top + 2 * radius), fill=255)
    draw.ellipse((left, bottom - 2 * radius, right, bottom), fill=255)
    return np.asarray(image, dtype=np.uint8)


def _encode_depth(depth_m: np.ndarray) -> np.ndarray:
    scaled = (depth_m - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def prepare_case(output_root: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ControlledBackgroundError("controlled_case_output_not_empty")
    runtime_records: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    for view_index in range(VIEW_COUNT):
        camera_id = f"view_{view_index:02d}"
        clean_rgb = _clean_rgb(view_index)
        clean_depth = _clean_depth_m(view_index)
        mask = _mask(view_index)
        selected = mask > 0
        occluded_rgb = clean_rgb.copy()
        occluded_rgb[selected] = np.array([28, 188, 104], dtype=np.uint8)
        encoded_depth = _encode_depth(clean_depth)
        occluded_depth = np.repeat(encoded_depth[..., None], 3, axis=2)
        occluded_depth[selected] = _encode_depth(np.array(0.92, dtype=np.float32))

        files = {
            "runtime_rgb": (output_root / "runtime_inputs/rgb" / f"{camera_id}.png", occluded_rgb),
            "runtime_rgb_mask": (
                output_root / "runtime_inputs/rgb" / f"{camera_id}_mask.png",
                mask,
            ),
            "runtime_depth": (
                output_root / "runtime_inputs/depth" / f"{camera_id}.png",
                occluded_depth,
            ),
            "runtime_depth_mask": (
                output_root / "runtime_inputs/depth" / f"{camera_id}_mask.png",
                mask,
            ),
            "truth_rgb": (output_root / "withheld_truth/rgb" / f"{camera_id}.png", clean_rgb),
            "truth_depth_mm": (
                output_root / "withheld_truth/depth" / f"{camera_id}.png",
                np.rint(clean_depth * 1000.0).astype(np.uint16),
            ),
        }
        for role, (path, array) in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path, compress_level=6)
            record = _record(path, output_root, role)
            (truth_records if role.startswith("truth_") else runtime_records).append(record)

    receipt: dict[str, Any] = {
        "schema_version": CASE_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009C",
        "case_id": "blueprint-authored-plane-synthetic-occluder-v1",
        "status": "prepared_truth_withheld",
        "created_at": _utc_now(),
        "authoring": {
            "source": "Blueprint deterministic analytic RGB and metric-depth plane",
            "rights": "Blueprint-controlled",
            "seed": 0,
            "camera_count": VIEW_COUNT,
            "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "synthetic_occluder": "rounded_can_silhouette",
        },
        "method": {
            "repository": LAMA_REPOSITORY,
            "commit": LAMA_COMMIT,
            "source_archive_sha256": LAMA_SOURCE_SHA256,
            "checkpoint_archive_sha256": LAMA_CHECKPOINT_SHA256,
            "license": "Apache-2.0",
            "same_color_and_depth_completion_path": True,
        },
        "runtime_inputs": runtime_records,
        "withheld_truth": truth_records,
        "firebreak": {
            "truth_must_not_be_mounted_during_completion": True,
            "truth_release_requires_sealed_completion_digest": True,
            "thresholds_frozen_before_execution": True,
        },
        "depth_encoding": {
            "runtime_min_m": DEPTH_MIN_M,
            "runtime_max_m": DEPTH_MAX_M,
            "truth_unit": "millimeter_uint16",
        },
        "thresholds": THRESHOLDS,
        "method_outcomes_observed": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(output_root / "controlled_background_case_receipt.json", receipt)
    return receipt


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ControlledBackgroundError("archive_path_traversal")
        archive.extractall(destination)


def _docker_image_id() -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != DOCKER_IMAGE_ID:
        raise ControlledBackgroundError("controlled_lama_docker_image_identity_mismatch")
    return result.stdout.strip()


def execute_case(
    *, case_root: Path, source_archive: Path, checkpoint_archive: Path
) -> dict[str, Any]:
    case_root = case_root.expanduser().resolve()
    if platform.system() == "Darwin" and Path("/private") in case_root.parents:
        raise ControlledBackgroundError("controlled_case_root_not_docker_shared")
    receipt_path = case_root / "controlled_background_case_receipt.json"
    case = _read(receipt_path)
    _verify_receipt(case, "receipt_digest", "controlled_case_receipt_digest_mismatch")
    if case.get("status") != "prepared_truth_withheld" or case.get("method_outcomes_observed") is not False:
        raise ControlledBackgroundError("controlled_case_not_preregistered")
    _verify_records(
        case.get("runtime_inputs"),
        case_root,
        required_roles={
            role
            for index in range(VIEW_COUNT)
            for role in (
                "runtime_rgb",
                "runtime_rgb_mask",
                "runtime_depth",
                "runtime_depth_mask",
            )
        },
    )
    # Roles repeat by camera, so explicitly verify the expected record count too.
    if len(case["runtime_inputs"]) != VIEW_COUNT * 4:
        raise ControlledBackgroundError("controlled_case_runtime_input_count_invalid")
    if _sha256(source_archive) != LAMA_SOURCE_SHA256 or _sha256(checkpoint_archive) != LAMA_CHECKPOINT_SHA256:
        raise ControlledBackgroundError("lama_source_or_checkpoint_changed")
    image_id = _docker_image_id()
    attempts_root = case_root / "attempts"
    attempt_number = 1
    while (attempts_root / f"attempt_{attempt_number:03d}").exists():
        attempt_number += 1
    runtime_root = attempts_root / f"attempt_{attempt_number:03d}"
    source_root = runtime_root / "source"
    model_root = runtime_root / "model"
    output_root = runtime_root / "outputs"
    _safe_extract(source_archive, source_root)
    _safe_extract(checkpoint_archive, model_root)
    output_root.mkdir(parents=True)
    logs: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for modality in ("rgb", "depth"):
        modality_output = output_root / modality
        modality_output.mkdir()
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--mount",
            f"type=bind,src={source_root},dst=/source,readonly",
            "--mount",
            f"type=bind,src={model_root},dst=/model,readonly",
            "--mount",
            f"type=bind,src={case_root / 'runtime_inputs' / modality},dst=/input,readonly",
            "--mount",
            f"type=bind,src={modality_output},dst=/output",
            "--env",
            "PYTHONPATH=/source",
            DOCKER_IMAGE,
            "python",
            "/source/bin/predict.py",
            "model.path=/model/big-lama",
            "indir=/input",
            "outdir=/output",
            f"hydra.run.dir=/tmp/hydra_{modality}",
        ]
        started_at = _utc_now()
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        finished_at = _utc_now()
        log_path = runtime_root / f"lama_{modality}.log"
        log_path.write_text(
            "STDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise ControlledBackgroundError(f"lama_{modality}_execution_failed:{result.returncode}")
        commands.append(command)
        logs.append(
            {
                "modality": modality,
                "started_at": started_at,
                "finished_at": finished_at,
                "exit_status": result.returncode,
                **_record(log_path, case_root, f"lama_{modality}_log"),
            }
        )
    output_records: list[dict[str, Any]] = []
    for modality in ("rgb", "depth"):
        for index in range(VIEW_COUNT):
            path = output_root / modality / f"view_{index:02d}_mask.png"
            output_records.append(_record(path, case_root, f"completed_{modality}"))
    freeze = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", DOCKER_IMAGE, "python", "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    freeze_path = runtime_root / "pip-freeze.txt"
    freeze_path.write_text(freeze.stdout, encoding="utf-8")
    execution: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009C",
        "status": "completion_sealed_truth_unreleased",
        "attempt_id": f"attempt_{attempt_number:03d}",
        "case_receipt_digest": case["receipt_digest"],
        "method": case["method"],
        "container": {
            "image": DOCKER_IMAGE,
            "image_id": image_id,
            "network": "none",
            "root_filesystem_read_only": True,
            "withheld_truth_mounted": False,
            "mounted_paths": ["source", "model", "runtime_inputs/rgb", "runtime_inputs/depth", "runtime/outputs"],
        },
        "commands": commands,
        "logs": logs,
        "outputs": output_records,
        "environment": _record(freeze_path, case_root, "pip_freeze"),
        "truth_opened_for_scoring": False,
        "completion_digest": "",
    }
    execution["completion_digest"] = canonical_digest(
        execution, digest_field="completion_digest"
    )
    _write(runtime_root / "controlled_background_execution_receipt.json", execution)
    return execution


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    selected = mask > 0
    interior = selected.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= np.roll(selected, shift=(dy, dx), axis=(0, 1))
    boundary = selected & ~interior
    # Widen inward to four pixels without scipy.
    widened = boundary.copy()
    layer = boundary.copy()
    for _ in range(3):
        next_layer = np.zeros_like(layer)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_layer |= np.roll(layer, shift=(dy, dx), axis=(0, 1))
        layer = next_layer & selected
        widened |= layer
    return widened


def _masked_psnr(predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    error = predicted.astype(np.float64)[mask] - truth.astype(np.float64)[mask]
    mse = float(np.mean(error * error))
    return 100.0 if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)


def _global_ssim(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted_f = predicted.astype(np.float64)
    truth_f = truth.astype(np.float64)
    mu_predicted = float(predicted_f.mean())
    mu_truth = float(truth_f.mean())
    var_predicted = float(predicted_f.var())
    var_truth = float(truth_f.var())
    covariance = float(
        np.mean((predicted_f - mu_predicted) * (truth_f - mu_truth))
    )
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    return float(
        ((2.0 * mu_predicted * mu_truth + c1) * (2.0 * covariance + c2))
        / ((mu_predicted**2 + mu_truth**2 + c1) * (var_predicted + var_truth + c2))
    )


def score_case(
    *, case_root: Path, execution_receipt_path: Path, output_path: Path | None
) -> dict[str, Any]:
    case_root = case_root.expanduser().resolve()
    case = _read(case_root / "controlled_background_case_receipt.json")
    execution_receipt_path = execution_receipt_path.expanduser().resolve()
    if case_root not in execution_receipt_path.parents:
        raise ControlledBackgroundError("execution_receipt_outside_case_root")
    execution_root = execution_receipt_path.parent
    execution = _read(execution_receipt_path)
    _verify_receipt(case, "receipt_digest", "controlled_case_receipt_digest_mismatch")
    _verify_receipt(execution, "completion_digest", "controlled_completion_digest_mismatch")
    if (
        execution.get("status") != "completion_sealed_truth_unreleased"
        or execution.get("case_receipt_digest") != case.get("receipt_digest")
        or execution.get("truth_opened_for_scoring") is not False
        or (execution.get("container") or {}).get("withheld_truth_mounted") is not False
    ):
        raise ControlledBackgroundError("seal_before_truth_release_not_proven")
    if len(execution.get("outputs") or []) != VIEW_COUNT * 2:
        raise ControlledBackgroundError("controlled_completion_output_count_invalid")
    for row in execution["outputs"]:
        path = (case_root / row["relative_path"]).resolve()
        if (
            case_root not in path.parents
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ControlledBackgroundError("controlled_completion_output_changed")
    if len(case.get("withheld_truth") or []) != VIEW_COUNT * 2:
        raise ControlledBackgroundError("controlled_truth_inventory_invalid")
    _verify_records(
        case["withheld_truth"],
        case_root,
        required_roles={"truth_rgb", "truth_depth_mm"},
    )

    views: list[dict[str, Any]] = []
    for index in range(VIEW_COUNT):
        camera_id = f"view_{index:02d}"
        mask = np.asarray(Image.open(case_root / f"runtime_inputs/rgb/{camera_id}_mask.png")) > 0
        truth_rgb = np.asarray(Image.open(case_root / f"withheld_truth/rgb/{camera_id}.png").convert("RGB"))
        predicted_rgb = np.asarray(
            Image.open(execution_root / f"outputs/rgb/{camera_id}_mask.png").convert("RGB")
        )
        truth_depth = (
            np.asarray(Image.open(case_root / f"withheld_truth/depth/{camera_id}.png"), dtype=np.float32)
            / 1000.0
        )
        predicted_depth_image = np.asarray(
            Image.open(execution_root / f"outputs/depth/{camera_id}_mask.png").convert("RGB"),
            dtype=np.float32,
        ).mean(axis=2)
        predicted_depth = DEPTH_MIN_M + predicted_depth_image / 255.0 * (DEPTH_MAX_M - DEPTH_MIN_M)
        ys, xs = np.where(mask)
        pad = 16
        y0, y1 = max(0, int(ys.min()) - pad), min(IMAGE_HEIGHT, int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(IMAGE_WIDTH, int(xs.max()) + pad + 1)
        rgb_ssim = _global_ssim(
            predicted_rgb[y0:y1, x0:x1], truth_rgb[y0:y1, x0:x1]
        )
        depth_error = predicted_depth[mask] - truth_depth[mask]
        boundary = _inner_boundary(mask)
        boundary_mae = float(
            np.mean(
                np.abs(
                    predicted_rgb.astype(np.float32)[boundary]
                    - truth_rgb.astype(np.float32)[boundary]
                )
            )
        )
        design = np.stack(
            (xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, dtype=np.float64)),
            axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(design, predicted_depth[mask], rcond=None)
        fitted = design @ coefficients
        plane_rmse = float(np.sqrt(np.mean((fitted - truth_depth[mask]) ** 2)))
        views.append(
            {
                "camera_id": camera_id,
                "mask_pixel_count": int(mask.sum()),
                "rgb_mask_psnr_db": _masked_psnr(predicted_rgb, truth_rgb, mask),
                "rgb_crop_ssim": float(rgb_ssim),
                "rgb_inner_boundary_mae_255": boundary_mae,
                "depth_mask_rmse_m": float(np.sqrt(np.mean(depth_error * depth_error))),
                "depth_mask_p95_abs_m": float(np.percentile(np.abs(depth_error), 95)),
                "depth_plane_rmse_m": plane_rmse,
            }
        )
    aggregate = {
        "view_count": len(views),
        "mean_rgb_mask_psnr_db": float(np.mean([row["rgb_mask_psnr_db"] for row in views])),
        "mean_rgb_mask_ssim": float(np.mean([row["rgb_crop_ssim"] for row in views])),
        "mean_rgb_inner_boundary_mae_255": float(
            np.mean([row["rgb_inner_boundary_mae_255"] for row in views])
        ),
        "mean_depth_mask_rmse_m": float(np.mean([row["depth_mask_rmse_m"] for row in views])),
        "max_depth_mask_p95_abs_m": float(max(row["depth_mask_p95_abs_m"] for row in views)),
        "mean_depth_plane_rmse_m": float(np.mean([row["depth_plane_rmse_m"] for row in views])),
    }
    checks = {
        "rgb_mask_psnr": aggregate["mean_rgb_mask_psnr_db"] >= THRESHOLDS["rgb_mask_psnr_db_min"],
        "rgb_mask_ssim": aggregate["mean_rgb_mask_ssim"] >= THRESHOLDS["rgb_mask_ssim_min"],
        "rgb_boundary": aggregate["mean_rgb_inner_boundary_mae_255"]
        <= THRESHOLDS["rgb_inner_boundary_mae_255_max"],
        "depth_rmse": aggregate["mean_depth_mask_rmse_m"] <= THRESHOLDS["depth_mask_rmse_m_max"],
        "depth_p95": aggregate["max_depth_mask_p95_abs_m"] <= THRESHOLDS["depth_mask_p95_abs_m_max"],
        "depth_plane": aggregate["mean_depth_plane_rmse_m"] <= THRESHOLDS["depth_plane_rmse_m_max"],
    }
    score: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009C",
        "status": "scored_factual_recovery",
        "case_receipt_digest": case["receipt_digest"],
        "completion_digest": execution["completion_digest"],
        "truth_released_after_completion_seal": True,
        "thresholds_frozen_before_execution": True,
        "thresholds": THRESHOLDS,
        "views": views,
        "aggregate": aggregate,
        "checks": checks,
        "quality_passed": all(checks.values()),
        "claim_ceiling": "blueprint_authored_known_background_factual_recovery_only",
        "claim_boundaries": {
            "not_interiorgs_hidden_background_truth": True,
            "not_real_scene_measurement": True,
            "not_physical_evidence": True,
            "not_method_generalization": True,
        },
        "receipt_digest": "",
    }
    score["receipt_digest"] = canonical_digest(score, digest_field="receipt_digest")
    if output_path is not None:
        _write(output_path.expanduser().resolve(), score)
    return score


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--case-root", type=Path, required=True)
    execute.add_argument("--source-archive", type=Path, required=True)
    execute.add_argument("--checkpoint-archive", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--case-root", type=Path, required=True)
    score.add_argument("--execution-receipt", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_case(args.output_root)
        digest = result["receipt_digest"]
    elif args.command == "execute":
        result = execute_case(
            case_root=args.case_root,
            source_archive=args.source_archive,
            checkpoint_archive=args.checkpoint_archive,
        )
        digest = result["completion_digest"]
    else:
        result = score_case(
            case_root=args.case_root,
            execution_receipt_path=args.execution_receipt,
            output_path=args.output,
        )
        digest = result["receipt_digest"]
    print(json.dumps({"status": result["status"], "digest": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
