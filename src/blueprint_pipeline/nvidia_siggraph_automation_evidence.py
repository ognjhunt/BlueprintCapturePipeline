"""Small integration helpers for advisory NVIDIA simulation evidence.

Keeping these path and summary adapters outside ``simulation_automation``
preserves that orchestrator's governed module budget while leaving Blueprint's
existing automation contract in control.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .common import read_json_any


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read(path: Path) -> dict[str, Any]:
    return _mapping(read_json_any(path)) if path.is_file() else {}


def relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def relative_if_file(base_dir: Path, target: Path) -> str | None:
    return relative_to(base_dir, target) if target.is_file() else None


def nvidia_source_artifact_paths(pipeline_dir: Path) -> dict[str, Path]:
    """Return optional NVIDIA evidence paths surfaced by automation."""

    return {
        "external_simready_validation_result": (
            pipeline_dir / "simready" / "external_validation_result.json"
        ),
        "simready_rule_calibration": pipeline_dir / "simready" / "rule_calibration.json",
        "ovrtx_preflight_result": pipeline_dir / "sensor_preflight" / "ovrtx_result.json",
        "ovrtx_preflight_runtime_receipt": (
            pipeline_dir / "sensor_preflight" / "ovrtx_runtime_receipt.json"
        ),
        "ovphysx_preflight_result": pipeline_dir / "physics_preflight" / "ovphysx_result.json",
        "ovphysx_preflight_runtime_receipt": (
            pipeline_dir / "physics_preflight" / "ovphysx_runtime_receipt.json"
        ),
        "omniverse_preflight_benchmark": pipeline_dir / "omniverse_preflight_benchmark.json",
        "omniverse_preflight_benchmark_suite": (
            pipeline_dir / "omniverse_preflight_benchmark_suite.json"
        ),
        "cosmos3_edge_experiment_result": (
            pipeline_dir / "cosmos3_edge_experiment" / "result.json"
        ),
        "cosmos3_edge_attempt_manifest": (
            pipeline_dir / "cosmos3_edge_experiment" / "attempt_manifest.json"
        ),
        "cosmos3_edge_qualification": (
            pipeline_dir / "cosmos3_edge_experiment" / "qualification.json"
        ),
        "gsplat_conformance_result": pipeline_dir / "gsplat_conformance" / "result.json",
        "nvidia_asset_conditioning_review": (
            pipeline_dir / "nvidia_asset_conditioning" / "review.json"
        ),
        "nvidia_experiment_resource_closeout": (
            pipeline_dir / "nvidia_experiment_resource_closeout.json"
        ),
        "nvidia_siggraph_completion_matrix": (
            pipeline_dir / "nvidia_siggraph_2026_completion_matrix.json"
        ),
        "nvidia_siggraph_capability_registry": (
            pipeline_dir / "nvidia_siggraph_2026_capability_registry.json"
        ),
    }


def nvidia_experiment_plan_summary(pipeline_dir: Path) -> dict[str, Any]:
    """Summarize optional evidence without promoting it to simulator proof."""

    paths = nvidia_source_artifact_paths(pipeline_dir)
    status_keys = {
        "external_simready_validation_status": "external_simready_validation_result",
        "simready_rule_calibration_status": "simready_rule_calibration",
        "ovrtx_preflight_status": "ovrtx_preflight_result",
        "ovphysx_preflight_status": "ovphysx_preflight_result",
        "cosmos3_edge_experiment_status": "cosmos3_edge_experiment_result",
        "cosmos3_edge_qualification_status": "cosmos3_edge_qualification",
        "gsplat_conformance_status": "gsplat_conformance_result",
    }
    summary = {
        output_key: str(_read(paths[path_key]).get("status") or "optional_not_run")
        for output_key, path_key in status_keys.items()
    }
    return {
        **summary,
        "all_artifacts_advisory_only": True,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
    }


def nvidia_experiment_result_artifacts(
    *, automation_dir: Path, pipeline_dir: Path
) -> dict[str, Any]:
    """Return repository-relative links for present advisory artifacts."""

    paths = nvidia_source_artifact_paths(pipeline_dir)
    selected = {
        "external_simready_validation_result": "external_simready_validation_result",
        "simready_rule_calibration": "simready_rule_calibration",
        "ovrtx_preflight_result": "ovrtx_preflight_result",
        "ovphysx_preflight_result": "ovphysx_preflight_result",
        "cosmos3_edge_experiment_result": "cosmos3_edge_experiment_result",
        "cosmos3_edge_qualification": "cosmos3_edge_qualification",
        "omniverse_preflight_benchmark_suite": "omniverse_preflight_benchmark_suite",
        "asset_conditioning_review": "nvidia_asset_conditioning_review",
        "resource_closeout": "nvidia_experiment_resource_closeout",
        "completion_matrix": "nvidia_siggraph_completion_matrix",
        "gsplat_conformance_result": "gsplat_conformance_result",
    }
    links = {
        output_key: (
            relative_to(automation_dir, paths[path_key])
            if paths[path_key].is_file()
            else None
        )
        for output_key, path_key in selected.items()
    }
    return {**links, "advisory_only": True}
