#!/usr/bin/env python3
"""Reject paid CPU-build or GPU-canary paths that bypass canonical allocation."""

from __future__ import annotations

import ast
import functools
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src/blueprint_pipeline/paid_resource_allocator.py"
MODEL_VOLUME_WATCHDOG = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_volume_watchdog.py"
LAMBDA_ADAPTER = ROOT / "src/blueprint_pipeline/lambda_provider_adapter.py"
RUNPOD_WATCHDOG = ROOT / "src/blueprint_pipeline/groot_oscar_runpod_watchdog.py"
MUTATION_SURFACE_MANIFEST = ROOT / "docs/architecture/paid-resource-mutation-surfaces.json"
SHARED_ADMISSION = ROOT / "src/blueprint_pipeline/paid_resource_admission.py"
#: RunPod is retired (owner decision 2026-11-05): every shared admission entry that
#: can open paid work must apply the RunPod refusal.
RUNPOD_RETIREMENT_ADMISSION_ENTRIES = (
    "require_paid_resource_budget",
    "require_paid_resource_admission",
    "build_paid_lane_admission",
)
AGENTS = ROOT / "AGENTS.md"
REMOTE_CPU_ALLOCATOR = ROOT / "src/blueprint_pipeline/remote_cpu_job_allocator.py"
CLOUD_RUN_JOBS_CLIENT = ROOT / "src/blueprint_pipeline/cloud_run_jobs_client.py"
CANONICAL_SUBCOMMANDS = ("gpu-canary", "remote-cpu-job")
# Plan 14 §8: what run_remote_cpu_job must reach, and the blocker each missing call raises.
REMOTE_CPU_REQUIRED_CALLS = {
    "require_paid_resource_admission": "remote_cpu_lane_bypasses_shared_admission",
    "reconcile_ambiguous_dispatch": "remote_cpu_ambiguous_dispatch_reconciliation_missing",
    "list_all_executions": "remote_cpu_execution_listing_pagination_missing",
    "prove_compute_zero": "remote_cpu_compute_zero_proof_missing",
    "prove_provider_zero": "remote_cpu_provider_zero_proof_missing",
}
REMOTE_CPU_REQUIRED_MARKERS = (
    "remote_cpu_provider_zero_unproven",
    "remote_cpu_ambiguous_dispatch_unresolved",
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
    "src/blueprint_pipeline/g1_microwave_finetune_provider_job.py",
    "src/blueprint_pipeline/openpi_policy_ranking_runpod.py",
    "src/blueprint_pipeline/nvidia_warehouse_native_camera_gpu_admission.py",
    "src/blueprint_pipeline/native_g1_paid_campaign.py",
    "src/blueprint_pipeline/native_g1_team_paid_policy.py",
    "src/blueprint_pipeline/native_g1_beta_paid.py",
    "src/blueprint_pipeline/parallel_findall_admission.py",
    "src/blueprint_pipeline/policy_canary_allocator_lane.py",
    "src/blueprint_pipeline/qualification_control_admission.py",
    "src/blueprint_pipeline/remote_cpu_job_allocator.py",
    "src/blueprint_pipeline/rigid_robot_free_probe_lane.py",
    "src/blueprint_pipeline/deployment_embodiment_preparation_lane.py",
    "src/blueprint_pipeline/single_g1_kitchen_episode_runpod.py",
    "src/blueprint_pipeline/single_g1_kitchen_qualification_session.py",
    "src/blueprint_pipeline/task_evaluation_scene_configuration_allocator.py",
    "src/blueprint_pipeline/website_task_context.py",
}
APPROVED_LANE_ADMISSION_BUILDERS = {
    "src/blueprint_pipeline/openpi_policy_ranking_gpu_admission.py",
    "src/blueprint_pipeline/native_g1_paid_campaign.py",
    "src/blueprint_pipeline/native_g1_team_paid_policy.py",
    "src/blueprint_pipeline/native_g1_beta_paid.py",
    "src/blueprint_pipeline/parallel_findall_admission.py",
    "src/blueprint_pipeline/policy_canary_allocator_lane.py",
    "src/blueprint_pipeline/nvidia_warehouse_native_camera_gpu_admission.py",
    "src/blueprint_pipeline/qualification_control_admission.py",
    "src/blueprint_pipeline/remote_cpu_job_allocator.py",
    "src/blueprint_pipeline/rigid_robot_free_probe_lane.py",
    "src/blueprint_pipeline/deployment_embodiment_preparation_lane.py",
    "src/blueprint_pipeline/task_evaluation_scene_configuration_allocator.py",
}
APPROVED_S3_TRANSPORT_CAPABILITY_CALLERS: set[tuple[str, str]] = set()
S3_TRANSPORT_CAPABILITIES = frozenset(
    {"_issue_transport_execution_capability", "_upload_and_verify_model_cache_impl"}
)
# Plan 14: B2 write authority for remote CPU staging.  The presigned PUT and the sentinel also
# require the attempt's grant; the copy, the version delete and the discard of a promoted object
# whose readback failed run in later processes without one (the collector, a resumed teardown), so
# only these functions may call any of them.
REMOTE_CPU_OBJECT_STORE_WRITERS = frozenset(
    {
        "presign_remote_cpu_put",
        "copy_remote_cpu_staging_to_cas",
        "delete_remote_cpu_staging_versions",
        "discard_remote_cpu_output_object",
        "remote_cpu_object_store_sentinel",
    }
)
APPROVED_REMOTE_CPU_OBJECT_STORE_CALLERS = {
    ("src/blueprint_pipeline/remote_cpu_job_allocator.py", "mint_transport"),
    ("src/blueprint_pipeline/remote_cpu_job_allocator.py", "_probe_checks"),
    ("src/blueprint_pipeline/remote_cpu_job_allocator.py", "prove_provider_zero"),
    (
        "src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py",
        "remote_cpu_object_store_sentinel",
    ),
    # Plan 14 §9 (PR 4): promotion and one readback of an episode compilation's output.
    ("src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py", "_promote"),
}
# The Atlas pose adapter reserves its own spend once the release gate it is
# handed passes. Only the canonical allocator entry may hand it that gate.
ATLAS_POSE_CAPABILITIES = frozenset({"run_admitted_atlas_pose", "submit_atlas_pose_operation"})
APPROVED_ATLAS_POSE_CALLERS = {
    ("src/blueprint_pipeline/paid_resource_allocator.py", "run_sponsored_website_atlas_pose"),
    ("src/blueprint_pipeline/website_atlas_pose.py", "run_admitted_atlas_pose"),
}
ATLAS_TASK_CAPABILITIES = frozenset({"run_admitted_atlas_task", "_run_admitted_atlas_task_attempt"})
APPROVED_ATLAS_TASK_CALLERS = {
    ("src/blueprint_pipeline/paid_resource_allocator.py", "run_sponsored_website_atlas_task"),
    ("src/blueprint_pipeline/website_atlas_tasks.py", "run_admitted_atlas_task"),
    ("src/blueprint_pipeline/website_atlas_tasks.py", "_retry_task"),
}
ATLAS_PLAN_CAPABILITIES = frozenset({"review_survey"})
APPROVED_ATLAS_PLAN_CALLERS = {
    ("src/blueprint_pipeline/paid_resource_allocator.py", "run_sponsored_website_atlas_plan"),
}
#: API-priced call sites (OpenAI, Gemini, World Labs, the hosted SAM service).
#: Every site the signals below find is classified in this table; see
#: ``_api_call_site_blockers``.
API_CALL_SITE_TABLE = ROOT / "scripts/paid_api_call_sites.json"
API_CALL_SITE_CLASSES = frozenset({"admitted", "not_automatic", "not_paid", "unadmitted"})
#: Sites that can spend and pass no admission, reservation or ledger before the
#: request on every path. A site classified ``unadmitted`` must be named here, and
#: tests/test_paid_resource_allocator_verifier.py pins this set: adding an entry
#: needs an explicit edit to that test, while removing one passes.
UNADMITTED_API_CALL_SITES: frozenset[str] = frozenset()
_API_HOSTS = {
    "openai": re.compile(r"https://api\.openai\.com(?![\w.-])"),
    "gemini": re.compile(r"https://generativelanguage\.googleapis\.com(?![\w.-])"),
    "world_labs": re.compile(r"https://api\.(?:atlas-beta\.)?worldlabs\.ai(?![\w.-])"),
    "meta_sam": re.compile(r"https://api\.meta\.ai(?![\w.-])"),
    "anthropic": re.compile(r"https://api\.anthropic\.com(?![\w.-])"),
    # Other priced model and generation APIs the repository calls.
    "nvidia_nim": re.compile(r"https://integrate\.api\.nvidia\.com(?![\w.-])"),
    "deepinfra": re.compile(r"https://api\.deepinfra\.com(?![\w.-])"),
    "dreamlabs": re.compile(r"https://api\.dreamlabs\.ai(?![\w.-])"),
    "palatial": re.compile(r"https://dashboard\.palatial\.cloud/api(?![\w.-])"),
    "runpod_serverless": re.compile(r"https://api\.runpod\.ai(?![\w.-])"),
}
#: Calls that open a priced client or reach the repository's own provider wrappers.
_API_CALLS = {
    "OpenAI": ("openai", "openai_client_construction"),
    "AsyncOpenAI": ("openai", "openai_client_construction"),
    "AzureOpenAI": ("openai", "openai_client_construction"),
    "Codex": ("openai", "openai_codex_client_construction"),
    "run_codex_cli_operator": ("openai", "openai_codex_cli"),
    "OpenAIProvider": ("openai", "openai_agents_provider"),
    "OpenAIResponsesModel": ("openai", "openai_agents_provider"),
    "OpenAIChatCompletionsModel": ("openai", "openai_agents_provider"),
    "Anthropic": ("anthropic", "anthropic_client_construction"),
    "AsyncAnthropic": ("anthropic", "anthropic_client_construction"),
    "_file_based_openai_api_key": ("openai", "openai_credentials"),
    "_execute_frame_request": ("openai", "openai_image_edit_request"),
    "atlas_api_config": ("world_labs", "world_labs_atlas_credentials"),
    "_worldlabs_api_request": ("world_labs", "world_labs_marble_request"),
    "meta_api_key": ("meta_sam", "meta_sam_credentials"),
}
_API_SDK_IMPORTS = {
    "openai": {"OpenAI", "AsyncOpenAI", "AzureOpenAI"},
    "openai_codex": {"Codex"},
    "anthropic": {"Anthropic", "AsyncAnthropic"},
    "google.genai": {"Client"},
}
_API_SDK_IMPORT_TEXT = re.compile(
    r"from\s+(?:openai|openai_codex|anthropic|google\.genai|google)\s+import|import\s+google\.genai"
)
#: Text every signal needs; a module without any of it is not parsed.
_API_PREFILTER = (
    "OpenAI", "openai.com", "Codex", "genai", "generativelanguage", "worldlabs.ai", "atlas_api_config",
    "_worldlabs_api_request", "meta_api_key", "api.meta.ai", "Runner", "_file_based_openai_api_key",
    "_execute_frame_request", "anthropic", "Anthropic", "codex", "nvidia.com", "deepinfra.com", "dreamlabs.ai",
    "palatial.cloud", "runpod.ai", "--skip-git-repo-check",
)
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


def _findall_constant_post_call(source: str) -> bool:
    """Catch executable uses of the exported FindAll URL, including aliases."""
    if "parallel_findall" not in source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    urls: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (node.module or "").endswith("parallel_findall") and alias.name == "RUNS_URL":
                    urls.add(alias.asname or alias.name)
                elif alias.name == "parallel_findall":
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("parallel_findall"):
                    modules.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        url = node.args[0] if node.args else next(
            (item.value for item in node.keywords if item.arg in {"url", "full_url"}), None
        )
        known_url = (
            isinstance(url, ast.Name) and url.id in urls
        ) or (
            isinstance(url, ast.Attribute) and url.attr == "RUNS_URL"
            and ast.unparse(url.value) in modules
        )
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        post = name.lower() == "post" or any(
            item.arg == "method" and isinstance(item.value, ast.Constant)
            and item.value.value == "POST" for item in node.keywords
        )
        if known_url and post:
            return True
    return False


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
    if any(
        call in source
        for call in (
            ".upload_file(",
            ".delete_object(",
            ".delete_objects(",
            ".copy_object(",
            ".upload_part_copy(",
        )
    ):
        signals.add("s3_object_write_or_delete")
    # A presigned PUT, POST, part upload or delete is write authority held by whoever has the URL.
    if "generate_presigned_post(" in source or (
        "generate_presigned_url(" in source
        and re.search(r"[\"'](?:put_object|upload_part|delete_object)[\"']", source)
    ):
        signals.add("s3_presigned_write_authority")
    # Starting, creating-and-starting or cancelling a Cloud Run job execution, by any client.
    if (
        re.search(r"run\.googleapis\.com", source)
        and re.search(r":(?:run|cancel)(?![A-Za-z0-9_])", source)
    ) or re.search(
        r"\bgoogle\.cloud(?:\.|\s+import\s+)run_v2\b"
        r"|\b(?:JobsClient|ExecutionsClient|RunJobRequest)\b"
        r"|discovery\.build\(\s*[\"']run[\"']"
        r"|\b(?:start|run)(?:ExecutionToken|_execution_token)\b"
        # Global flags and release tracks may sit between gcloud and its command group.
        r"|gcloud\b[\s\S]{0,200}?\brun\W+jobs\W+"
        r"(?:execute|create|deploy|replace|update|executions\W+cancel)\b",
        source,
    ) or (
        # The discovery client's build, imported by name.
        re.search(r"\bgoogleapiclient\b", source)
        and re.search(r"(?<![\w.])build\(\s*[\"']run[\"']", source)
    ):
        signals.add("gcp_cloud_run_job_mutation")
    if (
        "teleport.varjo.com" in source
        and "/api/v1/captures" in source
        and any(method in source for method in ('"POST"', '"PUT"', '"DELETE"'))
    ):
        signals.add("teleport_capture_create_upload_or_delete")
    # Keep offline FindAll preparation/readers distinct from executable POSTs.
    if (
        re.search(r"\.findall\.(?:create|ingest|extend|enrich)\s*\(", source)
        or ("parallel_findall" in source and "self._post(RUNS_URL" in source)
        or _findall_constant_post_call(source)
        or (
            "api.parallel.ai/v1beta/findall/runs" in source
            and re.search(r"(?:method\s*=\s*['\"]POST['\"]|\.post\s*\()", source)
        )
    ):
        signals.add("parallel_findall_paid_mutation")
    return signals


def _reachable_calls(source: str, entry: str) -> set[str]:
    """Names called from ``entry`` or from any same-module function it reaches."""

    functions = {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls: set[str] = set()
    reached: set[str] = set()
    pending = [entry]
    while pending:
        name = pending.pop()
        if name in reached or name not in functions:
            continue
        reached.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call):
                target = node.func
                called = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
                if called:
                    calls.add(called)
                    pending.append(called)
    return calls


def _remote_cpu_lane_blockers(allocator: str, client: str) -> list[str]:
    """The remote CPU lane admits, reconciles from a complete listing and proves both zeros."""

    calls = _reachable_calls(allocator, "run_remote_cpu_job")
    blockers = [blocker for call, blocker in REMOTE_CPU_REQUIRED_CALLS.items() if call not in calls]
    blockers.extend(
        f"remote_cpu_lane_marker_missing:{marker}"
        for marker in REMOTE_CPU_REQUIRED_MARKERS
        if marker not in allocator
    )
    if "require_paid_resource_admission_grant" not in _reachable_calls(client, "run_job"):
        blockers.append("remote_cpu_run_job_grant_validation_missing")
    if "nextPageToken" not in client:
        blockers.append("remote_cpu_execution_listing_pagination_missing")
    return sorted(set(blockers))


def _canonical_subcommand_blockers(canonical: str, agents: str) -> list[str]:
    """Rule (h): the allocator registers every canonical subcommand, and AGENTS.md lists each."""

    blockers = []
    if not all(item in canonical for item in CANONICAL_SUBCOMMANDS):
        blockers.append("canonical_allocator_subcommands_missing")
    blockers.extend(
        f"agents_md_paid_allocator_command_missing:{item}"
        for item in CANONICAL_SUBCOMMANDS
        if f"python -m blueprint_pipeline.paid_resource_allocator {item}" not in agents
    )
    return blockers


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
    protected: frozenset[str] = S3_TRANSPORT_CAPABILITIES,
) -> set[tuple[str, str]]:
    """Every (module, function) whose body names a protected capability, directly or by alias."""

    observed: set[tuple[str, str]] = set()
    for relative, source in source_by_path.items():
        if not any(name in source for name in protected):
            continue  # a reference, an alias or a string naming one always contains its text
        tree = ast.parse(source, filename=relative)
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in protected:
                        aliases[alias.asname or alias.name] = alias.name
        def referenced(nodes) -> set[str]:
            references: set[str] = set()
            for child in (inner for node in nodes for inner in ast.walk(node)):
                if isinstance(child, ast.Name):
                    references.add(aliases.get(child.id, child.id))
                elif isinstance(child, ast.Attribute):
                    references.add(child.attr)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if child.value in protected:
                        references.add(child.value)
            return references

        functions = (ast.FunctionDef, ast.AsyncFunctionDef)
        for node in ast.walk(tree):
            if isinstance(node, functions) and referenced([node]) & protected:
                observed.add((relative, node.name))
            elif isinstance(node, ast.ClassDef):
                # Class-level statements run at import, outside any method.
                body = [child for child in node.body if not isinstance(child, (*functions, ast.ClassDef))]
                if referenced(body) & protected:
                    observed.add((relative, node.name))
        def exports(node) -> bool:  # ``__all__`` names exports; it calls nothing
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            return any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets)

        # Module-level statements run at import, outside any function or class.
        top = [node for node in tree.body
               if not isinstance(node, (*functions, ast.ClassDef, ast.Import, ast.ImportFrom)) and not exports(node)]
        if referenced(top) & protected:
            observed.add((relative, "<module>"))
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
        # A call can only name an issuer whose text the source contains; skip
        # parsing the rest so this whole-repository check stays a PR sentinel.
        if "require_paid_resource_admission" in source or "build_paid_lane_admission" in source:
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
    if _s3_transport_capability_callers(
        source_by_path, REMOTE_CPU_OBJECT_STORE_WRITERS
    ) != APPROVED_REMOTE_CPU_OBJECT_STORE_CALLERS:
        blockers.append("remote_cpu_object_store_writer_caller_set_mismatch")
    if _s3_transport_capability_callers(source_by_path, ATLAS_POSE_CAPABILITIES) != APPROVED_ATLAS_POSE_CALLERS:
        blockers.append("atlas_pose_capability_caller_set_mismatch")
    if _s3_transport_capability_callers(source_by_path, ATLAS_TASK_CAPABILITIES) != APPROVED_ATLAS_TASK_CALLERS:
        blockers.append("atlas_task_capability_caller_set_mismatch")
    if _s3_transport_capability_callers(source_by_path, ATLAS_PLAN_CAPABILITIES) != APPROVED_ATLAS_PLAN_CALLERS:
        blockers.append("atlas_plan_capability_caller_set_mismatch")
    # Same set as _unclassified_direct_mutators(source_by_path, set(by_path)),
    # from the signals already computed above.
    for relative in sorted(set(observed_direct_mutators) - set(by_path)):
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


def _codex_cli_argv(elements: list[ast.expr]) -> bool:
    """An argv that runs the Codex CLI: ``codex``/``@openai/codex`` then ``exec``, or a Codex-only flag."""

    for index, element in enumerate(elements):
        text = ast.unparse(element)
        if isinstance(element, ast.Constant) and element.value in {"--skip-git-repo-check"}:
            return True  # a Codex-only flag, whatever the argv prefix
        if isinstance(element, ast.Constant) and isinstance(element.value, str) and "@openai/codex" in element.value:
            return True
        if (index + 1 < len(elements) and "codex" in text.lower()
                and isinstance(elements[index + 1], ast.Constant) and elements[index + 1].value == "exec"):
            return True
    return False


def _api_call_sites(relative: str, source: str) -> dict[str, tuple[str, int]]:
    """Each API-priced call site in one module: ``path::scope::signal`` -> (provider, first line)."""

    if not any(text in source for text in _API_PREFILTER):
        return {}
    tree = ast.parse(source, filename=relative)
    aliases: dict[str, str] = {}  # local name -> SDK class it binds
    genai_modules = {"genai"}
    imports_sdk = _API_SDK_IMPORT_TEXT.search(source) is not None
    for node in ast.walk(tree) if imports_sdk else ():
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _API_SDK_IMPORTS.get(node.module or "", ()):
                    aliases[alias.asname or alias.name] = alias.name
                if node.module == "google" and alias.name == "genai":
                    genai_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "google.genai" and alias.asname:
                    genai_modules.add(alias.asname)
    sites: dict[str, tuple[str, int]] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = (*scope, child.name)
            hit = None
            if isinstance(child, ast.Call):
                target = child.func
                if isinstance(target, ast.Name):
                    name = aliases.get(target.id, target.id)
                    hit = _API_CALLS.get(name)
                    if name == "Client" and target.id in aliases:
                        hit = ("gemini", "gemini_client_construction")
                elif isinstance(target, ast.Attribute):
                    if target.attr in {"Client", "GenerativeModel"} and ast.unparse(target.value) in {
                        *genai_modules, "google.genai"
                    }:
                        hit = ("gemini", "gemini_client_construction")
                    else:
                        hit = _API_CALLS.get(target.attr)
            elif (isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                  and child.value.id == "Runner" and child.attr.startswith("run")):
                hit = ("openai", "openai_agents_sdk_runner")  # the Agents SDK's model loop
            elif isinstance(child, (ast.List, ast.Tuple)) and _codex_cli_argv(child.elts):
                hit = ("openai", "openai_codex_cli")  # the paid ``codex exec`` CLI
            elif isinstance(child, ast.Constant) and isinstance(child.value, str) and "https://" in child.value:
                for provider, pattern in _API_HOSTS.items():
                    if pattern.search(child.value):
                        hit = (provider, provider + "_api_url")
            if hit is not None:
                key = f"{relative}::{'.'.join(scope) or '<module>'}::{hit[1]}"
                sites.setdefault(key, (hit[0], child.lineno))
            visit(child, inner)

    visit(tree, ())
    return sites


def _repository_api_call_sites(root: Path = ROOT) -> dict[str, tuple[str, int]]:
    sites: dict[str, tuple[str, int]] = {}
    for path in _production_python_paths(root):
        sites.update(_api_call_sites(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    return sites


@functools.lru_cache(maxsize=None)
def _reference_lines(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """First line calling, and first line defining, each name in ``path``."""

    calls: dict[str, int] = {}
    defs: dict[str, int] = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return calls, defs
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            called = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
            if called:
                calls[called] = min(node.lineno, calls.get(called, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = min(node.lineno, defs.get(node.name, node.lineno))
    return calls, defs


def _first_reference_line(path: Path, name: str) -> int | None:
    """The line defining ``name`` in ``path``, else the first line calling it."""

    calls, defs = _reference_lines(path)
    return defs.get(name, calls.get(name))


def _api_call_site_blockers(
    sites: dict[str, tuple[str, int]],
    table: dict,
    root: Path = ROOT,
    unadmitted_ceiling: frozenset[str] = UNADMITTED_API_CALL_SITES,
) -> list[str]:
    """Every API-priced call site is admitted, not paid, or a named unadmitted site."""

    if table.get("schema_version") != "paid_api_call_sites.v1" or not isinstance(table.get("sites"), dict):
        return ["paid_api_call_site_table_invalid"]
    rows: dict = table["sites"]
    how = (f"classify it in {API_CALL_SITE_TABLE.relative_to(ROOT).as_posix()} as admitted"
           " (with via and admission), not_paid (with reason) or unadmitted (and name it in"
           " UNADMITTED_API_CALL_SITES and the pinned list in tests/test_paid_resource_allocator_verifier.py)")
    blockers: list[str] = []
    for key, (_provider, line) in sorted(sites.items()):
        row = rows.get(key)
        if not isinstance(row, dict):
            blockers.append(f"unclassified_paid_api_call_site:{key} (line {line}): {how}")
            continue
        kind = row.get("class")
        if kind not in API_CALL_SITE_CLASSES:
            blockers.append(f"paid_api_call_site_class_invalid:{key}: {how}")
        elif kind == "not_paid" and not str(row.get("reason") or "").strip():
            blockers.append(f"paid_api_call_site_not_paid_reason_missing:{key}")
        elif kind == "unadmitted" and key not in unadmitted_ceiling:
            blockers.append(f"new_unadmitted_paid_api_call_site:{key}: route it through an existing"
                            " admission or ledger; adding an unadmitted site needs an explicit edit to the pinned"
                            " list in tests/test_paid_resource_allocator_verifier.py")
        elif kind == "admitted":
            # ``via``: where the site's path reaches the admission; ``admission``: the
            # admission, reservation or ledger function. Both are ``path::name``.
            for field in ("via", "admission"):
                path, _, name = str(row.get(field) or "").partition("::")
                if not name or _first_reference_line(root / path, name) is None:
                    blockers.append(f"paid_api_call_site_{field}_missing:{key}")
            blockers.extend(_structural_claim_blockers(key, row, root))
        elif kind == "not_automatic":
            path, _, name = str(row.get("entry_point") or "").partition("::")
            if not name or not str(row.get("reason") or "").strip() or (
                    _first_reference_line(root / path, name) is None and name not in {"<module>", "main"}):
                blockers.append(f"paid_api_call_site_not_automatic_entry_point_missing:{key}")
            blockers.extend(_structural_claim_blockers(key, row, root))
    return blockers


def _structural_claim_blockers(key: str, row: dict, root: Path) -> list[str]:
    """Check a ``not_automatic`` or ``guards``/``reachable_only_through`` claim on the call graph.

    An ``admitted`` row without either keeps the older label check, unless it says why the graph
    cannot express its admission (``structural_check``: a callable injected at run time).
    """

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paid_call_site_reachability import claim_blockers, unproven_guards

    guarded = bool(row.get("guards") or row.get("reachable_only_through"))
    if row.get("class") == "admitted" and not guarded:
        return []
    blockers = [f"paid_api_call_site_guard_unproven:{key}:{name}"
                for name in unproven_guards(frozenset(row.get("guards") or []), root)]
    if row.get("reachable_only_through") and not (
            str(row.get("enclosing_admission") or "").strip() and str(row.get("bound") or "").strip()):
        blockers.append(f"paid_api_call_site_enclosing_admission_missing:{key}")
    return blockers + claim_blockers(key, row, root)


def _verify_api_call_sites() -> list[str]:
    try:
        table = json.loads(API_CALL_SITE_TABLE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["paid_api_call_site_table_unreadable"]
    return _api_call_site_blockers(_repository_api_call_sites(), table)


def _runpod_retirement_blockers(path: Path = SHARED_ADMISSION) -> list[str]:
    """The shared admission keeps RunPod retired at every entry that can open paid work."""

    source = path.read_text(encoding="utf-8")
    calls = _function_calls(path)
    if "RUNPOD_PROVIDER_RETIRED = True" not in source or any(
        "runpod_retirement_blockers" not in calls.get(entry, set())
        for entry in RUNPOD_RETIREMENT_ADMISSION_ENTRIES
    ):
        return ["runpod_retirement_admission_refusal_missing"]
    return []


def verify() -> list[str]:
    blockers: list[str] = [
        *_verify_mutation_surface_contract(),
        *_verify_operator_docs(),
        *_runpod_retirement_blockers(),
        *_verify_api_call_sites(),
    ]
    canonical = CANONICAL.read_text(encoding="utf-8")
    model_volume_watchdog = MODEL_VOLUME_WATCHDOG.read_text(encoding="utf-8")
    lambda_adapter = LAMBDA_ADAPTER.read_text(encoding="utf-8")
    runpod_watchdog = RUNPOD_WATCHDOG.read_text(encoding="utf-8")

    if "runpod_allocator_request_blockers(args)" not in canonical:
        blockers.append("canonical_allocator_runpod_retirement_refusal_missing")
    blockers.extend(_canonical_subcommand_blockers(canonical, AGENTS.read_text(encoding="utf-8")))
    blockers.extend(
        _remote_cpu_lane_blockers(
            REMOTE_CPU_ALLOCATOR.read_text(encoding="utf-8"),
            CLOUD_RUN_JOBS_CLIENT.read_text(encoding="utf-8"),
        )
    )
    if "_runpod_call(" not in model_volume_watchdog or "def watchdog(" not in model_volume_watchdog:
        blockers.append("model_volume_watchdog_teardown_surface_missing")
    if '"POST"' in model_volume_watchdog:
        blockers.append("model_volume_watchdog_may_create_provider_resources")
    if not all(
        marker in lambda_adapter
        for marker in (
            "MUTATING_API_MODES = LIVE_LAUNCH_MODES | {TERMINATE_MODE}",
            "if mode in MUTATING_API_MODES:",
            "if args.mode in MUTATING_API_MODES:",
        )
    ):
        blockers.append("lambda_termination_shared_admission_guard_missing")

    if "teardown_unverified" not in runpod_watchdog:
        blockers.append("gpu_watchdog_teardown_error_evidence_missing")
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
