"""Optional NVIDIA usd-convert-gsplat conformance oracle.

Blueprint's direct ParticleField author remains the production path. This lane
runs a pinned external converter and compares composed USD arrays without
granting the oracle authority over capture truth or changing the default path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import ensure_dir, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256, executable_identity, run_json_worker
from .local_capture import resolve_local_capture_context
from .nvidia_siggraph_policy import evaluate_stop_rules
from .nvidia_experiment_resource import (
    load_resource_closeout,
    load_resource_context,
    resource_stop_evidence,
)
from .particlefield_usd import PARTICLEFIELD_SCHEMA, write_particlefield_usd


SOURCE_URL = "https://github.com/NVIDIA-Omniverse/usd-convert-gsplat"
ATTRIBUTES = (
    "positions",
    "scales",
    "orientations",
    "opacities",
    "radiance:sphericalHarmonicsCoefficients",
    "radiance:sphericalHarmonicsDegree",
    "extent",
)


def _path_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _particlefield_prim(stage: Any) -> Any:
    matches = [prim for prim in stage.Traverse() if prim.GetTypeName() == PARTICLEFIELD_SCHEMA]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {PARTICLEFIELD_SCHEMA} prim; observed {len(matches)}"
        )
    return matches[0]


def _numpy_value(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imag = value.GetImaginary()
        return np.asarray([value.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)
    if isinstance(value, (int, float)):
        return np.asarray(value)
    rows = []
    for item in value:
        if hasattr(item, "GetReal") and hasattr(item, "GetImaginary"):
            imag = item.GetImaginary()
            rows.append([item.GetReal(), imag[0], imag[1], imag[2]])
        elif hasattr(item, "__len__") and not isinstance(item, (str, bytes)):
            rows.append(list(item))
        else:
            rows.append(item)
    return np.asarray(rows)


def _quaternion_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        return float("inf")
    direct = np.linalg.norm(left - right, axis=-1)
    sign_flipped = np.linalg.norm(left + right, axis=-1)
    return float(np.max(np.minimum(direct, sign_flipped)))


def compare_particlefield_usd(
    blueprint_usd: str | Path,
    oracle_usd: str | Path,
    *,
    absolute_tolerance: float = 1.0e-5,
    relative_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    left_path = Path(blueprint_usd).resolve()
    right_path = Path(oracle_usd).resolve()
    left_stage = Usd.Stage.Open(str(left_path))
    right_stage = Usd.Stage.Open(str(right_path))
    if not left_stage or not right_stage:
        return {"status": "blocked", "blockers": ["particlefield_stage_open_failed"]}
    try:
        left_prim = _particlefield_prim(left_stage)
        right_prim = _particlefield_prim(right_stage)
    except ValueError as exc:
        return {"status": "blocked", "blockers": [str(exc)]}
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name in ATTRIBUTES:
        left_value = left_prim.GetAttribute(name).Get()
        right_value = right_prim.GetAttribute(name).Get()
        left = _numpy_value(left_value)
        right = _numpy_value(right_value)
        shape_equal = left.shape == right.shape
        if name == "orientations":
            max_error = _quaternion_error(left.astype(float), right.astype(float))
            equal = shape_equal and max_error <= absolute_tolerance
        elif shape_equal and left.dtype.kind in "iuf" and right.dtype.kind in "iuf":
            difference = np.abs(left.astype(float) - right.astype(float))
            max_error = float(np.max(difference)) if difference.size else 0.0
            equal = bool(
                np.allclose(
                    left.astype(float),
                    right.astype(float),
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    equal_nan=False,
                )
            )
        else:
            max_error = 0.0 if shape_equal and np.array_equal(left, right) else float("inf")
            equal = bool(shape_equal and np.array_equal(left, right))
        if not equal:
            blockers.append(f"particlefield_attribute_mismatch:{name}")
        rows.append(
            {
                "attribute": name,
                "blueprint_shape": list(left.shape),
                "oracle_shape": list(right.shape),
                "shape_equal": shape_equal,
                "within_tolerance": equal,
                "maximum_absolute_or_quaternion_sign_invariant_error": max_error,
            }
        )
    metadata = {
        "blueprint": {
            "up_axis": str(UsdGeom.GetStageUpAxis(left_stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(left_stage)),
            "prim_path": str(left_prim.GetPath()),
        },
        "oracle": {
            "up_axis": str(UsdGeom.GetStageUpAxis(right_stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(right_stage)),
            "prim_path": str(right_prim.GetPath()),
        },
    }
    if metadata["blueprint"]["up_axis"] != metadata["oracle"]["up_axis"]:
        blockers.append("particlefield_up_axis_mismatch")
    if not np.isclose(
        metadata["blueprint"]["meters_per_unit"],
        metadata["oracle"]["meters_per_unit"],
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        blockers.append("particlefield_meters_per_unit_mismatch")
    return {
        "schema_version": "particlefield_usd_conformance_comparison.v1",
        "status": "conformant" if not blockers else "nonconformant",
        "blueprint_usd_sha256": sha256_file(left_path),
        "oracle_usd_sha256": sha256_file(right_path),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "metadata": metadata,
        "attribute_comparisons": rows,
        "blockers": blockers,
    }


def run_gsplat_conformance_oracle(
    *,
    capture_root: str | Path,
    source_ply: str | Path,
    converter_command: str | Sequence[str] | None,
    converter_version: str,
    source_revision: str,
    license_id: str = "Apache-2.0 AND CC-BY-4.0",
    license_compatible: bool = False,
    timeout_seconds: int = 300,
    resource_context_path: str | Path | None = None,
    resource_closeout_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    output_dir = context.pipeline_root / "gsplat_conformance"
    ensure_dir(output_dir)
    source = Path(source_ply).resolve()
    blueprint_usd = output_dir / "blueprint_particlefield.usdc"
    oracle_usd = output_dir / "nvidia_oracle_particlefield.usdc"
    raw_report = output_dir / "oracle_worker_report.json"
    blockers: list[str] = []
    resource_context, resource_blockers = load_resource_context(resource_context_path)
    blockers.extend(resource_blockers)
    resource_closeout, closeout_blockers = load_resource_closeout(
        resource_context, resource_closeout_path
    )
    blockers.extend(closeout_blockers)
    if not source.is_file() or not _path_within(source, context.pipeline_root):
        blockers.append("source_ply_missing_or_not_privacy_safe_pipeline_derived")
    if not converter_version or not source_revision:
        blockers.append("converter_version_or_revision_not_pinned")
    if not license_id or not license_compatible:
        blockers.append("license_not_verified_compatible")
    if converter_command is None:
        blockers.append("converter_command_not_configured")
    authoring = (
        write_particlefield_usd(source, blueprint_usd) if not blockers else {"status": "blocked"}
    )
    if authoring.get("status") != "completed" and not blockers:
        blockers.extend(authoring.get("blockers") or ["blueprint_particlefield_authoring_failed"])
    request = {
        "schema_version": "usd_convert_gsplat_conformance_request.v1",
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source) if source.is_file() else None,
        },
        "blueprint_output_path": str(blueprint_usd),
        "oracle_output_path": str(oracle_usd),
        "converter": {
            "version": converter_version,
            "source_revision": source_revision,
            "source_url": SOURCE_URL,
            "license_id": license_id,
            "license_compatible": license_compatible,
            "executable_identity": executable_identity(converter_command, env=env)
            if converter_command
            else {},
        },
        "execution_policy": {
            "network_policy": "disabled",
            "transformations_beyond_format_conversion_allowed": False,
            "replaces_blueprint_authoring": False,
            "paid_resource_allocation_performed": False,
        },
        "resource_context": resource_context,
        "resource_closeout": resource_closeout or None,
        "blockers": blockers,
    }
    request["request_fingerprint"] = canonical_sha256(request)
    request_path = output_dir / "request.json"
    write_json(request_path, request)
    execution: dict[str, Any] | None = None
    if not blockers and converter_command is not None:
        execution = run_json_worker(
            command=converter_command,
            replacements={
                "input": str(source),
                "output": str(raw_report),
                "oracle_output": str(oracle_usd),
                "blueprint_output": str(blueprint_usd),
                "converter_version": converter_version,
                "source_revision": source_revision,
            },
            working_directory=output_dir,
            output_directory=output_dir,
            raw_report_path=raw_report,
            timeout_seconds=timeout_seconds,
            network_policy="disabled",
            env=env,
            log_prefix="usd_convert_gsplat",
        )
        if execution.get("exit_code") != 0 or not oracle_usd.is_file():
            blockers.append("nvidia_converter_worker_failed")
        if raw_report.is_file():
            try:
                worker_report = json.loads(raw_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                worker_report = {}
            if worker_report.get("converter_version") != converter_version:
                blockers.append("converter_worker_version_mismatch")
            if worker_report.get("source_revision") != source_revision:
                blockers.append("converter_worker_revision_mismatch")
    comparison = (
        compare_particlefield_usd(blueprint_usd, oracle_usd)
        if not blockers and oracle_usd.is_file()
        else {"status": "blocked", "blockers": list(blockers)}
    )
    if comparison.get("status") != "conformant":
        blockers.extend(comparison.get("blockers") or ["particlefield_conformance_not_proven"])
    stop_rules = evaluate_stop_rules(
        component="usd_convert_gsplat",
        require_measured_value=False,
        evidence={
            "component_version_pinned": bool(converter_version and source_revision),
            "license_compatible": license_compatible,
            "stable_normalized_receipts": comparison.get("status") == "conformant",
            "privacy_safe_inputs_only": bool(
                source.is_file() and _path_within(source, context.pipeline_root)
            ),
            "dependency_isolated": True,
            "input_output_digests_preserved": bool(source.is_file() and oracle_usd.is_file()),
            "proof_boundaries_separated": True,
            "useful_failure_class_or_cost_gain": False,
            **resource_stop_evidence(resource_context, resource_closeout),
        },
    )
    result = {
        "schema_version": "usd_convert_gsplat_conformance_result.v1",
        "generated_at": utc_now_iso(),
        "status": "conformant_advisory" if not blockers else "blocked_or_nonconformant",
        "request_path": str(request_path),
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "blueprint_output_sha256": sha256_file(blueprint_usd) if blueprint_usd.is_file() else None,
        "oracle_output_sha256": sha256_file(oracle_usd) if oracle_usd.is_file() else None,
        "execution": execution,
        "resource_context": resource_context,
        "resource_closeout": resource_closeout or None,
        "comparison": comparison,
        "stop_rule_evaluation": stop_rules,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "optional_conformance_oracle_only": True,
            "blueprint_particlefield_authoring_replaced": False,
            "isaac_or_ovrtx_render_proven": False,
            "capture_truth_modified": False,
            "rank_fidelity_result_proven": False,
        },
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Blueprint and NVIDIA ParticleField USD")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--source-ply", required=True)
    parser.add_argument("--converter-command", required=True)
    parser.add_argument("--converter-version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license-id", default="Apache-2.0 AND CC-BY-4.0")
    parser.add_argument("--license-compatible", action="store_true")
    parser.add_argument("--resource-context")
    parser.add_argument("--resource-closeout")
    args = parser.parse_args(argv)
    result = run_gsplat_conformance_oracle(
        capture_root=args.capture_root,
        source_ply=args.source_ply,
        converter_command=args.converter_command,
        converter_version=args.converter_version,
        source_revision=args.source_revision,
        license_id=args.license_id,
        license_compatible=args.license_compatible,
        resource_context_path=args.resource_context,
        resource_closeout_path=args.resource_closeout,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "conformant_advisory" else 2


if __name__ == "__main__":
    raise SystemExit(main())
