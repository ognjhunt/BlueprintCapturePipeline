"""Immutable blank-stage Isaac diagnostic for articulated task assets.

This is the low-cost native gate before an articulated asset enters the Franka
scenario harness.  It deliberately does not install Arena or provision a
policy.  The same capped Vast transport used by ADP-009D executes the bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint


REQUEST_SCHEMA = "articulated_native_diagnostic_request.v1"
BUNDLE_SCHEMA = "articulated_native_diagnostic_bundle.v1"
PROBE_KIND = "adp009d-franka-native-microcheck"
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.0-dev2@"
    "sha256:c3e7bef5b2bfdb9972807c34195206078372bf8c6cff79716be130a3fe3e9ce9"
)


class ArticulatedNativeDiagnosticError(ValueError):
    """Stable, sorted construction errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite(value: Any, *, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and minimum <= number <= maximum else None


def _path(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith("/") and ".." not in text else ""


def build_articulated_native_diagnostic_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and digest a scene-neutral articulated native request."""

    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_request_not_json"]
        ) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("articulated_native_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1":
        errors.append("articulated_native_request_program_invalid")
    if request.get("learned_policy_outcomes_observed") is not False:
        errors.append("articulated_native_request_policy_outcome_leakage")
    asset = request.get("asset")
    if not isinstance(asset, Mapping) or not _is_sha256(asset.get("sha256")):
        errors.append("articulated_native_request_asset_invalid")
    task = request.get("articulation")
    if not isinstance(task, Mapping):
        errors.append("articulated_native_request_articulation_missing")
    else:
        for field in (
            "root_prim_path",
            "fixed_base_body_prim_path",
            "driven_joint_prim_path",
        ):
            if not _path(task.get(field)):
                errors.append(f"articulated_native_request_{field}_invalid")
        locked = task.get("locked_joint_prim_paths")
        if (
            not isinstance(locked, list)
            or len(set(map(str, locked))) != len(locked)
            or any(not _path(item) for item in locked)
            or task.get("driven_joint_prim_path") in locked
        ):
            errors.append("articulated_native_request_locked_joints_invalid")
        expected_count = task.get("expected_joint_count")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 1
            or isinstance(locked, list)
            and expected_count < len(locked) + 1
        ):
            errors.append("articulated_native_request_joint_count_invalid")
        angles = task.get("commanded_angles_degrees")
        if (
            not isinstance(angles, list)
            or len(angles) < 2
            or any(_finite(item, minimum=-360.0, maximum=360.0) is None for item in angles)
            or any(float(b) <= float(a) for a, b in zip(angles, angles[1:], strict=False))
        ):
            errors.append("articulated_native_request_angles_invalid")
    runtime = request.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("articulated_native_request_runtime_missing")
    else:
        steps = runtime.get("settle_steps_per_command")
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 2400:
            errors.append("articulated_native_request_settle_steps_invalid")
        for field, minimum, maximum in (
            ("joint_readback_tolerance_degrees", 0.01, 10.0),
            ("locked_joint_tolerance_degrees", 0.01, 10.0),
            ("fixed_base_translation_tolerance_m", 1e-7, 0.05),
            ("fixed_base_rotation_tolerance_degrees", 1e-4, 5.0),
            ("maximum_abs_joint_velocity_rad_s_after_settle", 1e-4, 10.0),
            ("drive_stiffness", 1.0, 1e8),
            ("drive_damping", 0.0, 1e8),
            ("drive_max_force", 0.1, 1e9),
        ):
            if _finite(runtime.get(field), minimum=minimum, maximum=maximum) is None:
                errors.append(f"articulated_native_request_{field}_invalid")
    appearance = request.get("render_appearance")
    if not isinstance(appearance, Mapping):
        errors.append("articulated_native_request_render_appearance_missing")
    else:
        if not _is_sha256(appearance.get("static_appearance_receipt_digest")):
            errors.append("articulated_native_request_appearance_receipt_invalid")
        material_paths = appearance.get("required_material_paths")
        if (
            not isinstance(material_paths, list)
            or not material_paths
            or len(set(map(str, material_paths))) != len(material_paths)
            or any(not _path(item) for item in material_paths)
        ):
            errors.append("articulated_native_request_render_materials_invalid")
        resolution = appearance.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in resolution)
            or any(not 64 <= int(item) <= 4096 for item in resolution)
        ):
            errors.append("articulated_native_request_render_resolution_invalid")
        if _finite(
            appearance.get("vertical_fov_degrees"), minimum=10.0, maximum=150.0
        ) is None:
            errors.append("articulated_native_request_render_fov_invalid")
        if _finite(
            appearance.get("minimum_pixel_stddev"), minimum=0.1, maximum=100.0
        ) is None:
            errors.append("articulated_native_request_render_variance_invalid")
        cameras = appearance.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            errors.append("articulated_native_request_render_cameras_invalid")
        else:
            camera_ids: list[str] = []
            for row in cameras:
                if not isinstance(row, Mapping):
                    errors.append("articulated_native_request_render_camera_invalid")
                    continue
                camera_id = str(row.get("camera_id") or "")
                camera_ids.append(camera_id)
                for field in ("position_asset_m", "look_at_asset_m"):
                    vector = row.get(field)
                    if (
                        not isinstance(vector, list)
                        or len(vector) != 3
                        or any(
                            _finite(item, minimum=-1000.0, maximum=1000.0) is None
                            for item in vector
                        )
                    ):
                        errors.append(
                            f"articulated_native_request_render_camera_{field}_invalid"
                        )
                if row.get("role") not in {"material_readback", "review_only"}:
                    errors.append("articulated_native_request_render_camera_role_invalid")
            if (
                any(not item for item in camera_ids)
                or len(camera_ids) != len(set(camera_ids))
            ):
                errors.append("articulated_native_request_render_camera_ids_invalid")
    if errors:
        raise ArticulatedNativeDiagnosticError(errors)
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_request_digest_mismatch"]
        )
    request["request_digest"] = expected
    return request


def _inspect_bound_asset(asset_path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_openusd_runtime_missing"]
        ) from exc
    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_asset_unreadable"]
        )
    articulation = request["articulation"]
    root = stage.GetPrimAtPath(articulation["root_prim_path"])
    fixed = stage.GetPrimAtPath(articulation["fixed_base_body_prim_path"])
    driven = stage.GetPrimAtPath(articulation["driven_joint_prim_path"])
    locked_paths = list(articulation["locked_joint_prim_paths"])
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    errors = []
    if float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0:
        errors.append("articulated_native_asset_not_metric")
    if str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z":
        errors.append("articulated_native_asset_not_z_up")
    if not root.IsValid() or not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        errors.append("articulated_native_articulation_root_missing")
    if not fixed.IsValid() or not fixed.HasAPI(UsdPhysics.RigidBodyAPI):
        errors.append("articulated_native_fixed_base_body_missing")
    if not driven.IsValid() or not driven.IsA(UsdPhysics.RevoluteJoint):
        errors.append("articulated_native_driven_revolute_joint_missing")
    if len(joints) != articulation["expected_joint_count"]:
        errors.append("articulated_native_joint_count_mismatch")
    for path in locked_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdPhysics.RevoluteJoint):
            errors.append(f"articulated_native_locked_revolute_joint_missing:{path}")
    render_material_paths: list[str] = []
    for path in request["render_appearance"]["required_material_paths"]:
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        try:
            connected, _invalid = material.GetSurfaceOutput().GetConnectedSources()
        except Exception:
            connected = []
        if not material or not material.GetPrim().IsValid() or not connected:
            errors.append(f"articulated_native_render_material_missing:{path}")
        else:
            render_material_paths.append(str(material.GetPath()))
    if errors:
        raise ArticulatedNativeDiagnosticError(errors)
    return {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)).upper(),
        "articulation_root_prim_path": str(root.GetPath()),
        "fixed_base_body_prim_path": str(fixed.GetPath()),
        "joint_prim_paths": sorted(str(prim.GetPath()) for prim in joints),
        "rigid_body_prim_paths": sorted(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        "collision_prim_count": sum(
            1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
        ),
        "render_material_paths": sorted(render_material_paths),
        "static_appearance_receipt_digest": request["render_appearance"][
            "static_appearance_receipt_digest"
        ],
    }


ENTRYPOINT = """#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
mkdir -p "$OUT_DIR"
if [ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-0}" = "1" ]; then
  for required in \
    "$RUNTIME_DIR/assets/articulated_task_asset.usda" \
    "$RUNTIME_DIR/articulated_native_diagnostic_request.v1.json" \
    "$RUNTIME_DIR/articulated_native_diagnostic_runtime.py" \
    "$RUNTIME_DIR/adp009d_franka_eval_harness_manifest.v1.json"; do
    if [ ! -s "$required" ]; then
      echo "missing exact-bundle rehearsal member: $required" >&2
      exit 2
    fi
  done
  python3 - "$OUT_DIR/provider_bundle_rehearsal.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "provider_bundle_entrypoint_rehearsal.v1",
    "status": "passed",
    "entrypoint": "run_adp_arena_provider_runtime.sh",
    "archive_extraction_executed": True,
    "gpu_runtime_started": False,
    "paid_inference_performed": False,
    "provider_mutations_performed": 0,
    "stopped_before": "isaac_sim_startup",
}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
  exit 0
fi
write_articulated_native_missing_result() {
  reason="$1"
  if [ ! -s "$OUT_DIR/adp009d_native_microcheck.json" ]; then
    /isaac-sim/python.sh - "$OUT_DIR/adp009d_native_microcheck.json" "$reason" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "adp009d_articulated_native_diagnostic.v1",
    "status": "blocked",
    "candidate_policy_queried": False,
    "candidate_outcomes_accessed": False,
    "provider_zero_required_after_return": True,
    "blockers": [sys.argv[2]],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
}
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:articulated_native_diagnostic:started"
/isaac-sim/python.sh "$RUNTIME_DIR/articulated_native_diagnostic_runtime.py" \
  --asset "$RUNTIME_DIR/assets/articulated_task_asset.usda" \
  --request "$RUNTIME_DIR/articulated_native_diagnostic_request.v1.json" \
  --output "$OUT_DIR/adp009d_native_microcheck.json" \
  >"$OUT_DIR/articulated_native_diagnostic.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  write_articulated_native_missing_result "articulated_native_runner_failed_without_runtime_result"
fi
cat "$OUT_DIR/articulated_native_diagnostic.log" || true
echo "BLUEPRINT_WAM_RUNTIME_PHASE:adp009d:articulated_native_diagnostic:completed:rc=$rc"
exit "$rc"
"""


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_articulated_native_diagnostic_bundle(
    *,
    job_dir: str | Path,
    asset_path: str | Path,
    request_path: str | Path,
    harness_manifest_path: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Bind exact bytes into the canonical ADP-009D paid transport shape."""

    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_implementation_commit_invalid"]
        )
    asset = Path(asset_path).expanduser().resolve()
    request_source = Path(request_path).expanduser().resolve()
    harness = Path(harness_manifest_path).expanduser().resolve()
    if any(not path.is_file() or path.is_symlink() for path in (asset, request_source, harness)):
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_bundle_input_missing"]
        )
    try:
        raw_request = json.loads(request_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_request_unreadable"]
        ) from exc
    request = build_articulated_native_diagnostic_request(raw_request)
    if request["asset"]["sha256"] != _sha256(asset):
        raise ArticulatedNativeDiagnosticError(
            ["articulated_native_asset_digest_mismatch"]
        )
    inventory = _inspect_bound_asset(asset, request)

    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    assets = runtime / "assets"
    ensure_dir(assets)
    shutil.copy2(asset, assets / "articulated_task_asset.usda")
    (runtime / "articulated_native_diagnostic_request.v1.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_root = Path(__file__).resolve().parent
    shutil.copy2(
        source_root / "articulated_native_diagnostic_runtime.py",
        runtime / "articulated_native_diagnostic_runtime.py",
    )
    shutil.copy2(harness, runtime / "adp009d_franka_eval_harness_manifest.v1.json")
    _write_executable(runtime / "run_adp_arena_provider_runtime.sh", ENTRYPOINT)
    generated = generated_at or utc_now_iso()
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "generated_at": generated,
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "diagnostic_kind": "blank_stage_articulated_asset",
        "implementation_commit": implementation_commit,
        "container_image": DEFAULT_IMAGE,
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": "adp009d_native_microcheck.json",
        "asset_binding": {
            "filename": "articulated_task_asset.usda",
            "sha256": _sha256(asset),
            "size_bytes": asset.stat().st_size,
        },
        "request_digest": request["request_digest"],
        "static_inventory": inventory,
        "harness_manifest_sha256": _sha256(harness),
        "policy_candidate_id": None,
        "controls_requested": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "private_data_uploaded": False,
        "retry_cap": 0,
        "provider_zero_required_after_return": True,
        "blockers": [],
    }
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
    write_json(runtime / "adp_arena_provider_manifest.json", manifest)
    bundle_path = job / "adp009d_articulated_native_diagnostic_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", allowZip64=True) as archive:
        for path in sorted(runtime.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_STORED
            )
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path="provider_runtime/run_adp_arena_provider_runtime.sh",
        evidence_path=job / "articulated_native_exact_bundle_rehearsal.json",
    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    write_json(job / "articulated_native_diagnostic_bundle_receipt.json", receipt)
    return receipt


__all__ = [
    "ArticulatedNativeDiagnosticError",
    "BUNDLE_SCHEMA",
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "REQUEST_SCHEMA",
    "build_articulated_native_diagnostic_bundle",
    "build_articulated_native_diagnostic_request",
]
