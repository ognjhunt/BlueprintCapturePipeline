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
THIN_RELEASE_CONTRACT = ROOT / "src/blueprint_pipeline/thin_release_image_contract.py"
RUNBOOK = ROOT / "docs/runbooks/groot-oscar-thin-release.md"
THIN_ENTRYPOINT = (
    ROOT
    / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/thin_release_entrypoint.sh"
)
RUNPOD_WATCHDOG = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_watchdog.py"
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
    thin_release = THIN_RELEASE_CONTRACT.read_text(encoding="utf-8")
    thin_entrypoint = THIN_ENTRYPOINT.read_text(encoding="utf-8")
    runpod_watchdog = RUNPOD_WATCHDOG.read_text(encoding="utf-8")
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
    for path in LEGACY_BUILD_SCRIPTS:
        script = path.read_text(encoding="utf-8")
        prefix = script.split("repo_root=", 1)[0]
        if (
            'echo "legacy build path disabled; use paid_resource_allocator cpu-build"' not in prefix
            or "exit 2" not in prefix
            or "BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT" in prefix
        ):
            blockers.append("legacy_cpu_build_script_not_hard_disabled")
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    if "paid_resource_allocator cpu-build-local" not in workflow:
        blockers.append("release_workflow_bypasses_canonical_cpu_allocator")
    prerequisite_index = workflow.find("verify_groot_oscar_live_prerequisites.py")
    allocator_index = workflow.find("paid_resource_allocator cpu-build-local")
    if prerequisite_index < 0 or not prerequisite_index < allocator_index:
        blockers.append("release_workflow_live_prerequisite_refresh_missing")
    elif "--live" not in workflow[prerequisite_index:allocator_index]:
        blockers.append("release_workflow_live_prerequisite_refresh_not_live")
    for unsafe_expression in (
        "--foundation-ref '${{ inputs.foundation_image_ref }}'",
        "--release-ref '${{ inputs.release_image_ref }}'",
        'expected_source_commit":"${{ inputs.source_ref }}',
    ):
        if unsafe_expression in workflow:
            blockers.append("release_workflow_dispatch_input_reaches_shell_parser")

    cpu_calls = _function_calls(CPU_ADAPTER)
    if "_request" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_provider_mutation_moved_outside_guarded_adapter")
    if "build_cpu_build_execution_admission" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_build_missing_live_execution_admission")
    if "require_paid_resource_admission" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_allocator_bypasses_shared_admission")
    if "_reconcile_ambiguous_create" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_ambiguous_create_reconciliation_missing")
    if "ambiguous_create_reconciliation.json" not in cpu:
        blockers.append("cpu_ambiguous_create_evidence_missing")
    if "_list_droplets_by_tag" not in cpu_calls.get("_live_profile", set()):
        blockers.append("cpu_builder_inventory_pagination_missing")
    if "_delete_with_fail_closed_evidence" not in cpu_calls.get("run_builder", set()):
        blockers.append("cpu_teardown_error_evidence_missing")
    if "digitalocean_builder_teardown_unverified" not in cpu:
        blockers.append("cpu_teardown_unverified_blocker_missing")
    if "validate_remote_build_results" not in cpu or "check=True" not in cpu:
        blockers.append("cpu_remote_build_result_copy_not_fail_closed")
    for remote_secret_path in (
        "/root/blueprint-build/docker_username",
        "/root/blueprint-build/docker_pat",
    ):
        if remote_secret_path not in cpu:
            blockers.append("cpu_registry_secret_fixed_remote_name_missing")
    if "_delete_with_fail_closed_evidence" not in cpu_calls.get("watchdog", set()):
        blockers.append("cpu_watchdog_teardown_error_evidence_missing")
    if 'set -- blueprint-run-robot-eval-worker "$@"' not in thin_entrypoint:
        blockers.append("thin_release_worker_executable_restore_missing")
    if "teardown_unverified" not in runpod_watchdog:
        blockers.append("gpu_watchdog_teardown_error_evidence_missing")
    if 'item.get("size")' not in thin_release:
        blockers.append("thin_release_native_registry_layer_size_missing")
    if 'Path("/root/blueprint-builder-ready").is_file()' not in cpu:
        blockers.append("local_cpu_builder_ready_marker_not_observed")
    if "/root/blueprint-builder-ready" not in runbook:
        blockers.append("local_cpu_builder_ready_marker_not_documented")
    packet_builder = (
        ROOT / "src/blueprint_pipeline/groot_oscar_thin_remote_build_packet.py"
    ).read_text(encoding="utf-8")
    if "_SAFE_VERSIONED_IMAGE_REF" not in packet_builder or "shlex.quote" not in packet_builder:
        blockers.append("remote_build_packet_image_ref_shell_safety_missing")
    gpu_calls = _function_calls(GPU_ADAPTER)
    if "run_runpod_provider_adapter" not in gpu_calls.get("run_canary", set()):
        blockers.append("gpu_provider_mutation_moved_outside_guarded_adapter")
    if "require_paid_resource_admission" not in gpu_calls.get("run_canary", set()):
        blockers.append("gpu_allocator_bypasses_shared_admission")
    if 'mode=RUNPOD_IMAGE_STARTUP_CANARY_MODE' not in gpu:
        blockers.append("gpu_canary_dry_run_mode_differs_from_execution")
    runpod_adapter = (
        ROOT / "src/blueprint_pipeline/runpod_provider_adapter.py"
    ).read_text(encoding="utf-8")
    if 'shape["docker_entrypoint"] = ["/opt/blueprint/thin_release_entrypoint.sh"]' not in gpu:
        blockers.append("gpu_canary_bypasses_thin_release_entrypoint")
    if "use_thin_entrypoint" not in runpod_adapter:
        blockers.append("generic_gpu_canary_forces_thin_release_entrypoint")
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
