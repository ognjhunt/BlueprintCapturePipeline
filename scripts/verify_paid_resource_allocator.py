#!/usr/bin/env python3
"""Reject paid CPU-build or GPU-canary paths that bypass canonical allocation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src/blueprint_pipeline/paid_resource_allocator.py"
CPU_ADAPTER = ROOT / "src/blueprint_pipeline/groot_oscar_digitalocean_builder.py"
GPU_ADAPTER = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_canary.py"
RUNBOOK = ROOT / "docs/runbooks/groot-oscar-thin-release.md"
LEGACY_BUILD_SCRIPTS = (
    ROOT / "scripts/build_push_groot_oscar_foundation_image.sh",
    ROOT / "scripts/build_push_groot_oscar_release_image.sh",
    ROOT / "scripts/build_push_groot_oscar_closed_loop_image.sh",
)
RELEASE_WORKFLOW = ROOT / ".github/workflows/groot-oscar-thin-release.yml"


def _function_calls(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = child.func
                if isinstance(target, ast.Name):
                    calls.add(target.id)
                elif isinstance(target, ast.Attribute):
                    calls.add(target.attr)
        result[node.name] = calls
    return result


def verify() -> list[str]:
    blockers: list[str] = []
    canonical = CANONICAL.read_text(encoding="utf-8")
    cpu = CPU_ADAPTER.read_text(encoding="utf-8")
    gpu = GPU_ADAPTER.read_text(encoding="utf-8")
    runbook = RUNBOOK.read_text(encoding="utf-8")

    if "run_builder(" not in canonical or "run_canary(" not in canonical:
        blockers.append("canonical_allocator_missing_adapter_route")
    if "legacy_cpu_builder_launcher_disabled" not in cpu:
        blockers.append("legacy_cpu_builder_not_hard_disabled")
    if "legacy_gpu_canary_launcher_disabled" not in gpu:
        blockers.append("legacy_gpu_canary_not_hard_disabled")
    if "cpu-build" not in canonical or "gpu-canary" not in canonical:
        blockers.append("canonical_allocator_subcommands_missing")
    if "python -m blueprint_pipeline.paid_resource_allocator" not in runbook:
        blockers.append("canonical_allocator_command_missing_from_runbook")
    legacy_docs = (
        "python -m blueprint_pipeline.groot_oscar_digitalocean_builder launch",
        "python -m blueprint_pipeline.groot_oscar_runpod_canary",
    )
    if any(item in runbook for item in legacy_docs):
        blockers.append("runbook_recommends_legacy_paid_launcher")
    if any(
        "BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT" not in path.read_text(encoding="utf-8")
        for path in LEGACY_BUILD_SCRIPTS
    ):
        blockers.append("legacy_cpu_build_script_not_hard_disabled")
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    if "paid_resource_allocator cpu-build-local" not in workflow:
        blockers.append("release_workflow_bypasses_canonical_cpu_allocator")

    cpu_calls = _function_calls(CPU_ADAPTER)
    if "_request" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_provider_mutation_moved_outside_guarded_adapter")
    if "build_cpu_build_execution_admission" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_build_missing_live_execution_admission")
    if "require_paid_resource_admission" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_allocator_bypasses_shared_admission")
    gpu_calls = _function_calls(GPU_ADAPTER)
    if "run_runpod_provider_adapter" not in gpu_calls.get("run_canary", set()):
        blockers.append("gpu_provider_mutation_moved_outside_guarded_adapter")
    if "require_paid_resource_admission" not in gpu_calls.get("run_canary", set()):
        blockers.append("gpu_allocator_bypasses_shared_admission")
    return sorted(set(blockers))


def main() -> int:
    blockers = verify()
    if blockers:
        for blocker in blockers:
            print(blocker, file=sys.stderr)
        return 2
    print("paid_resource_allocator_verification=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
