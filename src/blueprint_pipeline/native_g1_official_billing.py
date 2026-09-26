"""Terminal evidence adapter for an exact G1 campaign's posted Vast charge.

Financial closeout is separate from policy evaluation: a blocked first episode
still consumed a provider instance and must have a verifiable charge receipt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


def g1_paid_campaign_terminal_evidence(
    *,
    instance_id: int,
    result_path: Path,
    result: Mapping[str, Any],
    result_bytes: bytes,
    json_file: Callable[..., tuple[Path, dict[str, Any], bytes]],
    record: Callable[[Path, bytes], dict[str, Any]],
    error_factory: Callable[[str], Exception],
) -> dict[str, Any] | None:
    if result.get("schema_version") != "native_g1_paid_campaign_result.v1":
        return None
    if result_path.name != "adp_arena_vast_result.json":
        raise error_factory("vast_official_g1_terminal_result_invalid")
    run_root = result_path.parent
    attempt_root = run_root / "attempts/attempt_001"
    closeout = result.get("provider_closeout")
    watchdog_close = result.get("independent_watchdog_close")
    watchdog = result.get("independent_watchdog")
    watchdog_handoff = result.get("independent_watchdog_handoff")
    if (
        result.get("attempt_number") != 1
        or result.get("attempt_root") != str(attempt_root)
        or result.get("status") not in {"completed", "blocked"}
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("raw_secret_values_recorded") is not False
        or result.get("all_staged_objects_absent") is not True
        or not isinstance(closeout, Mapping)
        or closeout.get("provider_zero_confirmed") is not True
        or closeout.get("warm_session_retained") is not False
        or closeout.get("all_staged_objects_absent") is not True
        or not isinstance(watchdog_handoff, Mapping)
        or watchdog_handoff.get("status") != "armed"
        or watchdog_handoff.get("watchdog_armed_before_allocation") is not True
        or not isinstance(watchdog, Mapping)
        or watchdog != watchdog_close
        or watchdog.get("schema_version") != "vast_independent_watchdog_handoff.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("instance_ids") != [instance_id]
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("provider_mutations_performed") != 0
        or watchdog.get("raw_secret_values_recorded") is not False
    ):
        raise error_factory("vast_official_g1_terminal_result_invalid")
    paths = {
        "attempt_result": attempt_root / "adp_arena_vast_result.json",
        "provider_adapter_result": (
            attempt_root / "vast_provider_run/vast_provider_adapter_result.json"
        ),
        "teardown_manifest": (
            attempt_root / "vast_provider_run/vast_teardown_manifest.json"
        ),
        "artifact_manifest": attempt_root / "artifact_manifest.json",
        "native_result": (
            attempt_root / "immutable_execution/native_g1_provider_campaign_result.v1.json"
        ),
        "watchdog_receipt": (
            attempt_root / "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json"
        ),
        "object_store_cleanup": (
            attempt_root / "object_store_staging/wam_provider_object_store_cleanup.json"
        ),
    }
    for role, field in (
        ("provider_adapter_result", "adapter_result_path"),
        ("teardown_manifest", "teardown_manifest_path"),
        ("artifact_manifest", "artifact_manifest_path"),
        ("native_result", "native_control_result_path"),
        ("watchdog_receipt", "watchdog_receipt_path"),
        ("object_store_cleanup", "object_store_cleanup_path"),
    ):
        if result.get(field) != str(paths[role]):
            raise error_factory("vast_official_g1_terminal_path_invalid")
    loaded = {
        role: json_file(path, code="vast_official_g1_" + role + "_invalid")
        for role, path in paths.items()
    }
    if loaded["attempt_result"][2] != result_bytes:
        raise error_factory("vast_official_g1_attempt_result_mismatch")
    adapter = loaded["provider_adapter_result"][1]
    teardown = loaded["teardown_manifest"][1]
    artifact = loaded["artifact_manifest"][1]
    native = loaded["native_result"][1]
    watcher = loaded["watchdog_receipt"][1]
    cleanup = loaded["object_store_cleanup"][1]
    if (
        closeout.get("adapter_result")
        != record(loaded["provider_adapter_result"][0], loaded["provider_adapter_result"][2])
        or closeout.get("teardown_manifest")
        != record(loaded["teardown_manifest"][0], loaded["teardown_manifest"][2])
        or adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("provider_bundle_kind") != "native_g1_development_campaign"
        or adapter.get("vast_instance_ids") != [instance_id]
        or adapter.get("status") not in {"completed", "blocked"}
        or adapter.get("final_validation_status") != "passed"
        or adapter.get("continuing_spend_from_this_run") is not False
        or adapter.get("retained_owned") is not False
        or adapter.get("raw_api_key_stored") is not False
        or adapter.get("secret_values_in_artifact") is not False
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("vast_instance_ids") != [instance_id]
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("retention_authorized") is not False
        or teardown.get("raw_secret_values_recorded") is not False
        or artifact.get("schema_version") != "task_evaluation_artifact_manifest.v1"
        or artifact.get("status") != "completed"
        or artifact.get("blockers") != []
        or artifact.get("manifest_digest") != canonical_digest(artifact, digest_field="manifest_digest")
        or (artifact.get("binding") or {}).get("bundle_sha256") != result.get("bundle_sha256")
        or (artifact.get("binding") or {}).get("attempt_number") != 1
        or native.get("schema_version") != "native_g1_provider_campaign_result.v1"
        or native.get("status") != result.get("status")
        or native.get("claim_ceiling") != "development_only"
        or native.get("ranking_eligible") is not False
        or native.get("physical_outcome_claimed") is not False
        or native.get("result_digest") != canonical_digest(native, digest_field="result_digest")
        or result.get("native_control_result_digest") != native.get("result_digest")
        or watcher.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watcher.get("provider") != "vast"
        or watcher.get("status") != "provider_terminal"
        or watcher.get("provider_absence_confirmed") is not True
        or (watcher.get("final_global_inventory") or {}).get("api_confirmed") is not True
        or (watcher.get("final_global_inventory") or {}).get("live_resource_count") != 0
        or watcher.get("raw_secret_values_recorded") is not False
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("all_ephemeral_objects_absent") is not True
        or cleanup.get("blockers") != []
        or cleanup.get("raw_secret_values_recorded") is not False
    ):
        raise error_factory("vast_official_g1_terminal_closure_invalid")
    evidence = {
        "financial_closeout_kind": "native_g1_paid_campaign.v1",
        "terminal_status": result["status"],
        "provider_absence_confirmed": True,
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
        "policy_evaluation_qualified": False,
        "scientific_success_inferred": False,
        "terminal_result": record(result_path, result_bytes),
    }
    for role, (path, _value, payload) in loaded.items():
        evidence[role] = record(path, payload)
    return evidence


__all__ = ["g1_paid_campaign_terminal_evidence"]
