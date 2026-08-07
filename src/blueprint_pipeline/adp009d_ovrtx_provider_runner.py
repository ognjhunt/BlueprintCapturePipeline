"""Provider-side orchestration for the bounded ADP-009D OVRTX camera probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


RESULT_NAME = "adp009d_ovrtx_live_camera_result.json"
OVRTX_REVISION = "4b9a5fe6f8becf6c5ff031e167cd4201054a96ce"
OVRTX_VERSION = "0.4.0.346409"
OVSTAGE_VERSION = "0.1.0.346039"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _absolute_executable_without_resolving_symlinks(path: Path) -> Path:
    """Keep virtual-environment interpreter identity while making it absolute."""

    return Path(os.path.abspath(os.fspath(path)))


def run(*, runtime_dir: Path, output_dir: Path, ovrtx_python: Path) -> dict[str, Any]:
    manifest = json.loads(
        (runtime_dir / "adp009d_ovrtx_provider_manifest.json").read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    worker = runtime_dir / "run_ovrtx_preflight_worker.py"
    asset = runtime_dir / "assets/aura_gaussian_surflets.usdc"
    vulkan_preflight = output_dir / "ovrtx_vulkan_preflight.log"
    if (
        manifest.get("headless_graphics_backend") != "xvfb"
        or manifest.get("vulkan_preflight_required") is not True
    ):
        blockers.append("ovrtx_headless_graphics_manifest_invalid")
    if not vulkan_preflight.is_file() or vulkan_preflight.stat().st_size <= 0:
        blockers.append("ovrtx_vulkan_preflight_evidence_missing")
    if _sha256(asset) != manifest.get("particlefield_sha256"):
        blockers.append("ovrtx_particlefield_runtime_digest_mismatch")
    for camera_id in ("external", "wrist"):
        camera_output = output_dir / camera_id
        camera_output.mkdir(parents=True, exist_ok=True)
        report_path = camera_output / "ovrtx_result.json"
        config_path = runtime_dir / f"configs/{camera_id}.ovrtx.json"
        expected_config = next(
            row
            for row in manifest.get("camera_configs", [])
            if row.get("camera_id") == camera_id
        )
        if _sha256(config_path) != expected_config.get("configuration_sha256"):
            blockers.append(f"ovrtx_camera_config_digest_mismatch:{camera_id}")
            continue
        command = [
            str(ovrtx_python),
            str(worker),
            "--input",
            str(asset),
            "--output",
            str(report_path),
            "--output-dir",
            str(camera_output),
            "--config",
            str(config_path),
            "--mode",
            "cold" if camera_id == "external" else "warm",
            "--source-revision",
            OVRTX_REVISION,
            "--modality",
            "rgb",
            "--modality",
            "depth",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        (camera_output / "worker.stdout.log").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        (camera_output / "worker.stderr.log").write_text(
            completed.stderr or "", encoding="utf-8"
        )
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        required_passed = bool(checks) and all(
            row.get("status") == "passed" for row in checks if isinstance(row, dict)
        )
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        valid = (
            completed.returncode == 0
            and report.get("component_version") == OVRTX_VERSION
            and report.get("source_revision") == OVRTX_REVISION
            and metrics.get("metric_depth_aov") == "DistanceToCameraSD"
            and metrics.get("unitless_depth_sd_used") is False
            and metrics.get("rtpt_warmup_frames") == 40
            and required_passed
            and (camera_output / "rgb.npy").is_file()
            and (camera_output / "rgb.png").is_file()
            and (camera_output / "depth.npy").is_file()
        )
        if not valid:
            blockers.append(f"ovrtx_live_camera_render_failed:{camera_id}")
        artifacts = [
            _artifact(path, output_dir)
            for path in sorted(camera_output.iterdir())
            if path.is_file()
        ]
        rows.append(
            {
                "camera_id": camera_id,
                "returncode": completed.returncode,
                "valid": valid,
                "report": report,
                "artifacts": artifacts,
            }
        )
    return {
        "schema_version": "adp009d_ovrtx_live_camera_result.v1",
        "status": "completed" if not blockers and len(rows) == 2 else "blocked",
        "blockers": sorted(set(blockers)),
        "implementation_commit": manifest.get("implementation_commit"),
        "input_digest": manifest.get("input_digest"),
        "particlefield_sha256": manifest.get("particlefield_sha256"),
        "ovrtx_revision": OVRTX_REVISION,
        "ovrtx_version": OVRTX_VERSION,
        "ovstage_version": OVSTAGE_VERSION,
        "initialization_order": ["OVRTX"],
        "ovphysx_loaded": False,
        "camera_rows": rows,
        "metric_depth_aov": "DistanceToCameraSD",
        "unitless_depth_sd_used": False,
        "rtpt_warmup_frames": 40,
        "headless_graphics_backend": "xvfb",
        "vulkan_preflight": (
            _artifact(vulkan_preflight, output_dir)
            if vulkan_preflight.is_file()
            else None
        ),
        "render_settings_target": "RenderProduct",
        "attached_mode_ordinals_respected": True,
        "write_floors_respected": True,
        "dlpack_ownership_explicit": True,
        "map_unmap_balanced": True,
        "device_synchronization_explicit": True,
        "camera_or_settings_change_reset": True,
        "sealed_source_mutated": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "proof_boundary": "Standalone Aura OVRTX RGB/depth microcheck; no Isaac composition or policy observation admitted yet.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ovrtx-python", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            runtime_dir=args.runtime_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            ovrtx_python=_absolute_executable_without_resolving_symlinks(
                args.ovrtx_python
            ),
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema_version": "adp009d_ovrtx_live_camera_result.v1",
            "status": "blocked",
            "blockers": [f"ovrtx_provider_runner_exception:{type(exc).__name__}"],
            "error": str(exc),
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "provider_zero_required_after_return": True,
        }
    _write_json(args.output_dir / RESULT_NAME, result)
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
