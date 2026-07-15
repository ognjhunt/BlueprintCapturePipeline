#!/usr/bin/env python3
"""Reject paid CPU-build or GPU-canary paths that bypass canonical allocation."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src/blueprint_pipeline/paid_resource_allocator.py"
CPU_ADAPTER = ROOT / "src/blueprint_pipeline/groot_oscar_digitalocean_builder.py"
GPU_ADAPTER = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_canary.py"
MODEL_VOLUME_ADAPTER = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_model_volume.py"
STORAGE_VOLUME_ADAPTER = (
    ROOT / "src/blueprint_pipeline/groot_oscar_runpod_storage_volume.py"
)
RUNPOD_PREFLIGHT = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_preflight.py"
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
MUTATION_SURFACE_MANIFEST = (
    ROOT / "docs/architecture/paid-resource-mutation-surfaces.json"
)
OPERATOR_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/FIRST_GPU_E2E_RUNBOOK.md",
    ROOT / "docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md",
    ROOT / "docs/architecture/command-safety-matrix.md",
)
PRODUCTION_SCAN_EXCLUSIONS = {
    "scripts/verify_paid_resource_allocator.py",
}
APPROVED_ADMISSION_ISSUERS = {
    "src/blueprint_pipeline/groot_oscar_digitalocean_builder.py",
    "src/blueprint_pipeline/groot_oscar_runpod_canary.py",
    "src/blueprint_pipeline/groot_oscar_runpod_storage_volume.py",
    "src/blueprint_pipeline/paid_resource_allocator.py",
}
APPROVED_LANE_ADMISSION_BUILDERS = {
    "src/blueprint_pipeline/groot_oscar_runpod_canary.py",
}
APPROVED_S3_TRANSPORT_CAPABILITY_CALLERS = {
    (
        "src/blueprint_pipeline/groot_oscar_runpod_s3_model_cache.py",
        "upload_and_verify_model_cache",
    ),
    (
        "src/blueprint_pipeline/groot_oscar_model_cache_s3_remote_executor.py",
        "execute_remote_packet",
    ),
}
SURFACE_CLASSIFICATIONS = {
    "canonical_allocator",
    "canonical_adapter",
    "grant_gated_legacy_adapter",
    "hard_disabled_legacy_launcher",
    "legacy_orchestrator_no_grant_issuer",
    "read_only_provider_inventory",
    "metered_object_storage_data_plane",
}


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


def _all_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            calls.add(target.id)
        elif isinstance(target, ast.Attribute):
            calls.add(target.attr)
    return calls


def _direct_paid_mutation_signals(source: str) -> set[str]:
    signals: set[str] = set()
    runpod_api_named = "RUNPOD_REST_API_BASE" in source or bool(
        re.search(r"https://api\.runpod\.io(?:[/:\"']|$)", source)
    )
    runpod_post_named = 'method="POST"' in source or '"method": "POST"' in source
    if (
        re.search(r"[\"']POST[\"']\s*,\s*(?:f)?[\"']/pods", source)
        or ('path="/pods"' in source and 'method="POST"' in source)
        or (runpod_api_named and "/pods" in source and runpod_post_named)
    ):
        signals.add("runpod_pod_create")
    if re.search(r"[\"']POST[\"']\s*,\s*[\"']/networkvolumes", source):
        signals.add("runpod_volume_create")
    if re.search(r"https://api\.digitalocean\.com(?:[/:\"']|$)", source) and (
        'method="POST", path="/droplets"' in source
        or '"POST", "/v2/droplets"' in source
        or '"POST", "/droplets"' in source
    ):
        signals.add("digitalocean_droplet_create")
    if re.search(r"method=[\"']PUT[\"']\s*,\s*\n?\s*path=f?[\"']/asks/", source):
        signals.add("vast_instance_create")
    if "instance-operations/launch" in source and '"method": "POST"' in source:
        signals.add("lambda_instance_create")
    if ".run_instances(" in source:
        signals.add("aws_instance_create")
    if re.search(r"https://compute\.googleapis\.com(?:[/:\"']|$)", source) and re.search(
        r"_call\([\"']POST[\"'].{0,160}/instances", source
    ):
        signals.add("gcp_instance_create")
    if ".upload_file(" in source or ".delete_object(" in source:
        signals.add("s3_object_write_or_delete")
    return signals


def _unclassified_direct_mutators(
    source_by_path: dict[str, str], known_paths: set[str]
) -> set[str]:
    return {
        path
        for path, source in source_by_path.items()
        if _direct_paid_mutation_signals(source) and path not in known_paths
    }


def _s3_transport_capability_callers(
    source_by_path: dict[str, str],
) -> set[tuple[str, str]]:
    observed: set[tuple[str, str]] = set()
    protected = {
        "_issue_transport_execution_capability",
        "_upload_and_verify_model_cache_impl",
    }
    for relative, source in source_by_path.items():
        tree = ast.parse(source, filename=relative)
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in protected:
                        aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            references: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    references.add(aliases.get(child.id, child.id))
                elif isinstance(child, ast.Attribute):
                    references.add(child.attr)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if child.value in protected:
                        references.add(child.value)
            if references & protected:
                observed.add((relative, node.name))
    return observed


def _production_python_paths(root: Path = ROOT) -> list[Path]:
    paths = [
        *root.glob("src/blueprint_pipeline/**/*.py"),
        *root.glob("scripts/**/*.py"),
    ]
    return sorted(
        path
        for path in paths
        if path.relative_to(root).as_posix() not in PRODUCTION_SCAN_EXCLUSIONS
    )


def _forbidden_operator_doc_commands(source: str) -> set[str]:
    patterns = {
        "legacy_gpu_provider_paid_launch": (
            r"blueprint-run-gpu-provider-launcher.{0,500}--allow-provider-launch"
        ),
        "legacy_runpod_adapter_paid_mode": (
            r"blueprint-run-runpod-provider-adapter.{0,700}"
            r"--mode\s+(?:serverless-run|on-demand-pod|existing-pod-start|"
            r"image-startup-canary-pod)"
        ),
        "legacy_runpod_live_mutation": (
            r"blueprint-collect-runpod-live-execution-proof.{0,900}"
            r"(?:--stop-on-startup-artifact-timeout|--terminate-pod|"
            r"--allow-runpod-api-call)"
        ),
        "legacy_unitree_runpod_launch": (
            r"blueprint-launch-unitree-unifolm-runpod-server\s+launch"
        ),
        "legacy_runpod_wam_create": r"runpod_wam_async_runner\s+create",
        "legacy_lambda_instance_launch": (
            r"blueprint-run-lambda-provider-adapter.{0,700}"
            r"--mode\s+launch-instance"
        ),
    }
    flattened = source.replace("\n", " ")
    return {
        label
        for label, pattern in patterns.items()
        if re.search(pattern, flattened, flags=re.IGNORECASE)
    }


def _verify_operator_docs() -> list[str]:
    blockers: list[str] = []
    for path in OPERATOR_DOCS:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            blockers.append("paid_resource_operator_doc_missing:" + path.name)
            continue
        for label in sorted(_forbidden_operator_doc_commands(source)):
            blockers.append(f"forbidden_paid_resource_doc_command:{path.name}:{label}")
    return blockers


def _verify_mutation_surface_contract() -> list[str]:
    blockers: list[str] = []
    try:
        manifest = json.loads(MUTATION_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["paid_resource_mutation_surface_manifest_unreadable"]
    if manifest.get("schema_version") != "paid_resource_mutation_surfaces.v1":
        blockers.append("paid_resource_mutation_surface_manifest_schema_invalid")
    issuer_allowlist = manifest.get("issuer_allowlist") or {}
    if set(issuer_allowlist.get("require_paid_resource_admission") or []) != (
        APPROVED_ADMISSION_ISSUERS
    ):
        blockers.append("paid_resource_admission_issuer_allowlist_changed")
    if set(issuer_allowlist.get("build_paid_lane_admission") or []) != (
        APPROVED_LANE_ADMISSION_BUILDERS
    ):
        blockers.append("paid_lane_admission_builder_allowlist_changed")

    rows = manifest.get("surfaces")
    rows = rows if isinstance(rows, list) else []
    by_path: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            blockers.append("paid_resource_mutation_surface_row_invalid")
            continue
        relative = row["path"]
        if relative in by_path:
            blockers.append("paid_resource_mutation_surface_duplicate:" + relative)
            continue
        by_path[relative] = row
        classification = row.get("classification")
        if classification not in SURFACE_CLASSIFICATIONS:
            blockers.append("paid_resource_mutation_surface_class_invalid:" + relative)
        path = ROOT / relative
        if not path.is_file():
            blockers.append("paid_resource_mutation_surface_missing:" + relative)
            continue
        source = path.read_text(encoding="utf-8")
        for marker in row.get("required_markers") or []:
            if not isinstance(marker, str) or marker not in source:
                blockers.append("paid_resource_mutation_surface_marker_missing:" + relative)
        calls = _all_calls(path)
        if classification == "grant_gated_legacy_adapter" and (
            "require_paid_resource_admission_grant" not in calls
        ):
            blockers.append("legacy_paid_adapter_grant_validation_missing:" + relative)
        if classification in {
            "grant_gated_legacy_adapter",
            "hard_disabled_legacy_launcher",
            "legacy_orchestrator_no_grant_issuer",
            "read_only_provider_inventory",
            "metered_object_storage_data_plane",
        } and calls & {"require_paid_resource_admission", "build_paid_lane_admission"}:
            blockers.append("legacy_paid_surface_can_issue_its_own_grant:" + relative)

    observed_admission_issuers: set[str] = set()
    observed_lane_builders: set[str] = set()
    source_by_path: dict[str, str] = {}
    observed_direct_mutators: dict[str, set[str]] = {}
    for path in _production_python_paths():
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        source_by_path[relative] = source
        calls = _all_calls(path)
        if "require_paid_resource_admission" in calls:
            observed_admission_issuers.add(relative)
        if "build_paid_lane_admission" in calls:
            observed_lane_builders.add(relative)
        signals = _direct_paid_mutation_signals(source)
        if signals:
            observed_direct_mutators[relative] = signals
    if observed_admission_issuers != APPROVED_ADMISSION_ISSUERS:
        blockers.append("paid_resource_admission_issuer_set_mismatch")
    if observed_lane_builders != APPROVED_LANE_ADMISSION_BUILDERS:
        blockers.append("paid_lane_admission_builder_set_mismatch")
    observed_s3_capability_callers = _s3_transport_capability_callers(source_by_path)
    if observed_s3_capability_callers != APPROVED_S3_TRANSPORT_CAPABILITY_CALLERS:
        blockers.append("runpod_s3_transport_capability_caller_set_mismatch")
    transport_module = "src/blueprint_pipeline/groot_oscar_runpod_s3_model_cache.py"
    for relative, source in source_by_path.items():
        if relative != transport_module and (
            "_TRANSPORT_CAPABILITY_ISSUER" in source
            or "_TransportExecutionCapability" in source
        ):
            blockers.append("runpod_s3_transport_capability_private_state_imported")
    for relative in sorted(_unclassified_direct_mutators(source_by_path, set(by_path))):
        blockers.append("unclassified_paid_resource_mutation_surface:" + relative)
    for relative in sorted(observed_direct_mutators):
        row = by_path.get(relative)
        if row is not None and row.get("classification") not in {
            "canonical_adapter",
            "grant_gated_legacy_adapter",
            "metered_object_storage_data_plane",
        }:
            blockers.append("direct_paid_mutator_classification_invalid:" + relative)
    return blockers


def verify() -> list[str]:
    blockers: list[str] = [
        *_verify_mutation_surface_contract(),
        *_verify_operator_docs(),
    ]
    canonical = CANONICAL.read_text(encoding="utf-8")
    cpu = CPU_ADAPTER.read_text(encoding="utf-8")
    gpu = GPU_ADAPTER.read_text(encoding="utf-8")
    model_volume = MODEL_VOLUME_ADAPTER.read_text(encoding="utf-8")
    storage_volume = STORAGE_VOLUME_ADAPTER.read_text(encoding="utf-8")
    runpod_preflight = RUNPOD_PREFLIGHT.read_text(encoding="utf-8")
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
    if not all(item in canonical for item in ("cpu-build", "model-volume", "gpu-canary")):
        blockers.append("canonical_allocator_subcommands_missing")
    if "run_storage_model_volume(" not in canonical:
        blockers.append("canonical_allocator_missing_model_volume_route")
    model_calls = _function_calls(STORAGE_VOLUME_ADAPTER)
    storage_calls = model_calls.get("run_storage_model_volume", set())
    if "require_paid_resource_admission" not in storage_calls:
        blockers.append("model_volume_bypasses_shared_admission")
    if "run_builder" not in storage_calls:
        blockers.append("model_volume_missing_canonical_cpu_builder_route")
    if '"/pods"' in storage_volume:
        blockers.append("storage_model_volume_may_allocate_runpod_pod")
    if "_delete_volume" not in storage_calls:
        blockers.append("model_volume_failure_volume_teardown_missing")
    if "watchdog_handoff.json" not in storage_volume:
        blockers.append("model_volume_independent_watchdog_missing")
    if "WATCHDOG_HANDOFF_SCHEMA_VERSION" not in storage_volume:
        blockers.append("model_volume_watchdog_handoff_schema_missing")
    if not all(
        marker in storage_volume
        for marker in ("watchdog_pid", "watchdog_state_path", "watchdog_nonce")
    ):
        blockers.append("model_volume_watchdog_process_handoff_missing")
    if not all(
        marker in storage_volume
        for marker in (
            "_watchdog_process_running",
            "storage_model_volume_watchdog_exited_before_builder",
            "volume_ready_watchdog_retained",
        )
    ):
        blockers.append("model_volume_ready_handoff_liveness_guard_missing")
    if (
        "legacy_gpu_model_volume_preparation_disabled_use_storage_only_allocator"
        not in model_volume
    ):
        blockers.append("legacy_gpu_model_volume_preparation_not_hard_disabled")
    if not all(
        marker in runpod_preflight
        for marker in (
            "build_model_volume_watchdog_handoff_evidence",
            "model_volume_watchdog_handoff_volume_mismatch",
            "model_volume_watchdog_ttl_does_not_cover_canary",
            "model_volume_watchdog_process_not_alive",
            "model_volume_watchdog_process_identity_invalid",
            "watchdog_process_identity_verified",
            "MODEL_VOLUME_WATCHDOG_MARGIN_SECONDS",
        )
    ):
        blockers.append("gpu_preflight_model_volume_watchdog_handoff_guard_missing")
    if "model_volume_watchdog_handoff=" not in gpu:
        blockers.append("gpu_launch_refresh_drops_model_volume_watchdog_handoff")
    if "pod_prefix=None" not in storage_volume or "volume_prefix=None" not in storage_volume:
        blockers.append("model_volume_global_inventory_guard_missing")
    if "build_runpod_network_volume_evidence(" not in storage_volume:
        blockers.append("model_volume_provider_reported_size_guard_missing")
    if "python -m blueprint_pipeline.paid_resource_allocator" not in runbook:
        blockers.append("canonical_allocator_command_missing_from_runbook")
    if "--model-volume-watchdog-handoff" not in runbook:
        blockers.append("runbook_model_volume_watchdog_handoff_missing")
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
    canonical_workflow_command = "paid_resource_allocator cpu-build"
    if canonical_workflow_command not in workflow or "--execution-plane local" not in workflow:
        blockers.append("release_workflow_bypasses_canonical_cpu_allocator")
    prerequisite_index = workflow.find("verify_groot_oscar_live_prerequisites.py")
    allocator_index = workflow.find(canonical_workflow_command)
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
    if 'set -- /opt/oscar-venv/bin/blueprint-run-robot-eval-worker "$@"' not in thin_entrypoint:
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
    try:
        release_candidate_at = packet_builder.index(
            '-t "$release_candidate_ref" --push'
        )
        release_validation_at = packet_builder.index("validate-thin-release")
        release_contract_at = packet_builder.index(
            "thin_release_contract_not_passed"
        )
        release_promotion_at = packet_builder.index(
            'imagetools create --tag "$release_ref" "$release_exact"'
        )
        terminal_result_at = packet_builder.index(
            'mv "$validation_result" "$result"'
        )
    except ValueError:
        blockers.append("remote_build_final_tag_promotion_guard_missing")
    else:
        if not (
            release_candidate_at
            < release_validation_at
            < release_contract_at
            < release_promotion_at
            < terminal_result_at
        ):
            blockers.append("remote_build_final_tag_promotion_order_invalid")
    if '-t "$release_ref" --push' in packet_builder:
        blockers.append("remote_build_pushes_unvalidated_final_release_tag")
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
