"""Financial closeout of a typed native runtime preflight, never a policy score."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest

SCHEMA = "native_task_arena_runtime_preflight.v1"


def runtime_preflight_terminal_evidence(*, instance_id: int, result_path: Path,
        result: Mapping[str, Any], result_bytes: bytes, json_file: Callable,
        record: Callable, error_factory: Callable) -> dict[str, Any] | None:
    if result.get("schema_version") != SCHEMA:
        return None

    def require(condition, suffix):
        if not condition:
            raise error_factory("vast_official_runtime_preflight_" + suffix)

    root = result_path.parent
    attempt = root / "allocator/attempts/attempt_001"
    require(result_path.name == "allocator-result.json" and result.get("attempt_root") == str(attempt)
        and result.get("attempt_number") == 1 and result.get("status") in {"completed", "blocked"}
        and result.get("result_digest") == canonical_digest(result, digest_field="result_digest")
        and result.get("retry_cap") == 0 and result.get("continuing_spend_from_this_run") is False
        and result.get("all_staged_objects_absent") is True and result.get("raw_secret_values_recorded") is False
        and result.get("candidate_policy_query_expected") is False, "terminal_invalid")
    paths = {
        "provider_adapter_result": ("adapter_result_path", attempt / "vast_provider_run/vast_provider_adapter_result.json"),
        "teardown_manifest": ("teardown_manifest_path", attempt / "vast_provider_run/vast_teardown_manifest.json"),
        "artifact_manifest": ("artifact_manifest_path", attempt / "artifact_manifest.json"),
        "independent_watchdog": ("watchdog_receipt_path", attempt / "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json"),
        "object_store_cleanup": ("object_store_cleanup_path", attempt / "object_store_staging/wam_provider_object_store_cleanup.json"),
    }
    loaded = {}
    for role, (field, path) in paths.items():
        require(result.get(field) == str(path), "artifact_path_invalid")
        loaded[role] = json_file(path, code="vast_official_runtime_preflight_" + role + "_invalid")
    for role, path in (
        ("operator_authority", root / "operator-diagnostic-authority.json"),
        ("provider_bundle_receipt", root / "frozen-bundle/composition_provider_bundle_receipt.json"),
        ("startup_manifest", attempt / "vast_provider_run/vast_startup_probe_manifest.json"),
        ("staging_manifest", attempt / "object_store_staging/wam_provider_object_store_staging_manifest.json"),
    ):
        loaded[role] = json_file(path, code="vast_official_runtime_preflight_" + role + "_invalid")
    adapter, teardown, watchdog, cleanup, artifact, authority, bundle, startup, staging = (
        loaded[key][1] for key in ("provider_adapter_result", "teardown_manifest", "independent_watchdog",
            "object_store_cleanup", "artifact_manifest", "operator_authority", "provider_bundle_receipt", "startup_manifest", "staging_manifest"))
    closeout = result.get("provider_closeout") or {}
    for role, key in (("provider_adapter_result", "adapter_result"), ("teardown_manifest", "teardown_manifest")):
        path, _value, payload = loaded[role]
        require(closeout.get(key) == record(path, payload), "closeout_record_changed")
    require(adapter.get("schema_version") == "vast_provider_adapter_result.v1"
        and adapter.get("vast_instance_ids") == [instance_id]
        and adapter.get("continuing_spend_from_this_run") is False
        and adapter.get("status") in {"completed", "failed"}
        and teardown.get("schema_version") == "vast_teardown_manifest.v1"
        and teardown.get("status") == "completed" and teardown.get("vast_instance_ids") == [instance_id]
        and teardown.get("runner_gpu_teardown_completed") is True
        and teardown.get("continuing_spend_from_this_run") is False
        and teardown.get("retention_authorized") is False
        and closeout.get("provider_zero_confirmed") is True
        and closeout.get("warm_session_retained") is False
        and closeout.get("all_staged_objects_absent") is True, "resource_not_closed")
    embedded = result.get("independent_watchdog") or {}
    recorded = watchdog.get("recorded_vast_instance") or {}
    absent = watchdog.get("recorded_vast_instance_teardown") or {}
    require(watchdog.get("schema_version") == "groot_oscar_runpod_canary_watchdog.v1"
        and watchdog.get("provider") == "vast" and watchdog.get("status") == "provider_terminal"
        and watchdog.get("provider_absence_confirmed") is True
        and str(recorded.get("instance_id")) == str(instance_id) and recorded.get("scope_confirmed") is True
        and str(absent.get("instance_id")) == str(instance_id) and absent.get("provider_absence_confirmed") is True
        and embedded.get("status") == "provider_terminal" and embedded.get("instance_ids") == [instance_id]
        and embedded.get("provider_absence_confirmed") is True, "watchdog_identity_invalid")
    require(all((watchdog.get(key) or {}).get("api_confirmed") is True
        and (watchdog.get(key) or {}).get("live_resource_count") == 0
        and (watchdog.get(key) or {}).get("resources") == []
        for key in ("initial_global_inventory", "final_global_inventory")), "global_zero_unverified")
    label = (startup.get("create_request_summary") or {}).get("label")
    require(startup.get("instance_id") == instance_id and isinstance(label, str)
        and label.startswith(str(watchdog.get("pod_name_prefix") or ""))
        and bool(watchdog.get("pod_name_prefix")), "launch_label_invalid")
    require(authority.get("schema_version") == "operator_native_diagnostic_authority.v1"
        and authority.get("authorization_digest") == canonical_digest(authority, digest_field="authorization_digest")
        and authority.get("run_id") == root.name and authority.get("maximum_provider_allocations") == 1
        and authority.get("retry_cap") == 0 and authority.get("new_asset_model_calls_authorized") == 0
        and authority.get("policy_model_calls_authorized") == 0
        and authority.get("hard_cap_usd") == result.get("hard_cap_usd")
        and authority.get("hard_ttl_seconds") == result.get("hard_ttl_seconds")
        and re.fullmatch(r"[0-9a-f]{40}", str(authority.get("source_commit") or "")) is not None
        and bundle.get("schema_version") == "native_task_arena_provider_bundle.v1"
        and bundle.get("status") == "ready" and bundle.get("execution_mode") == "runtime_preflight"
        and bundle.get("implementation_commit") == authority["source_commit"]
        and bundle.get("bundle_sha256") == result.get("bundle_sha256")
        and adapter.get("provider_bundle_sha256") == result.get("bundle_sha256")
        and bundle.get("candidate_policy_queried") is False and bundle.get("candidate_outcomes_accessed") is False
        and bundle.get("native_application_claimed") is False, "source_or_bundle_binding_invalid")
    bundle_path = Path(str(bundle.get("bundle_path") or ""))
    require(bundle_path.is_relative_to(root / "frozen-bundle") and not bundle_path.is_symlink()
        and bundle_path.is_file() and bundle_path.stat().st_size == bundle.get("bundle_size_bytes"), "bundle_file_invalid")
    with bundle_path.open("rb") as stream:
        require("sha256:" + hashlib.file_digest(stream, "sha256").hexdigest() == bundle["bundle_sha256"], "bundle_bytes_changed")
    require(cleanup.get("schema_version") == "wam_provider_object_store_cleanup.v1"
        and cleanup.get("status") == "completed" and cleanup.get("blockers") == []
        and cleanup.get("all_objects_absent") is True and cleanup.get("all_ephemeral_objects_absent") is True
        and cleanup.get("signed_url_files_removed") is True
        and cleanup.get("staging_manifest_sha256") == hashlib.sha256(loaded["staging_manifest"][2]).hexdigest()
        and staging.get("bundle_sha256") == bundle["bundle_sha256"].removeprefix("sha256:")
        and isinstance(cleanup.get("objects"), list) and bool(cleanup["objects"])
        and cleanup.get("exact_object_count") == len(cleanup["objects"])
        and all(row.get("absence", {}).get("absence_confirmed") is True for row in cleanup["objects"]), "object_cleanup_invalid")
    binding = artifact.get("binding") or {}
    require(artifact.get("schema_version") == "task_evaluation_artifact_manifest.v1"
        and artifact.get("manifest_digest") == canonical_digest(artifact, digest_field="manifest_digest")
        and artifact.get("status") in {"completed", "blocked"}
        and binding.get("result_schema_version") == SCHEMA and binding.get("bundle_sha256") == bundle["bundle_sha256"]
        and binding.get("retry_cap") == 0, "artifact_manifest_invalid")
    rows = {row.get("relative_path"): row for row in artifact.get("files", [])}
    for role in ("provider_adapter_result", "teardown_manifest"):
        path, _value, payload = loaded[role]
        row = rows.get(path.relative_to(attempt).as_posix()) or {}
        require(row.get("sha256") == "sha256:" + hashlib.sha256(payload).hexdigest()
            and row.get("size_bytes") == len(payload), "artifact_manifest_binding_changed")
    require(isinstance(result.get("estimated_cost_usd"), (int, float))
        and math.isfinite(result["estimated_cost_usd"]) and result["estimated_cost_usd"] >= 0, "estimate_invalid")
    evidence = {"financial_closeout_kind": SCHEMA, "terminal_status": result["status"],
        "launch_label": label, "source_commit": authority["source_commit"], "bundle_sha256": bundle["bundle_sha256"],
        "provider_absence_confirmed": True, "provider_zero_verified": True, "continuing_spend_from_this_run": False,
        "retry_cap": 0, "policy_evaluation_qualified": False, "scientific_success_inferred": False,
        "terminal_result": record(result_path, result_bytes)}
    for role, (path, _value, payload) in loaded.items():
        evidence[role] = record(path, payload)
    return evidence
