"""Adopt one canonical direct Arena execution into its Website launch lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import valid_adp_paid_provider_zero


SCHEMA_VERSION = "task_evaluation_native_direct_execution_adoption.v1"
STATUS = "blocked"
FILENAME = "native_direct_execution_adoption.v1.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    try:
        source = candidate.resolve(strict=True)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if (
        candidate.is_symlink()
        or source != absolute
        or not source.is_file()
        or not isinstance(value, dict)
    ):
        raise ValueError(code)
    return source, value


def _record(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema_version": value.get("schema_version"),
    }
    if value.get("status") is not None:
        record["status"] = value.get("status")
    return record


def _identity(
    path: Path, *, schema_version: str, digest_field: str, code: str
) -> tuple[Path, dict[str, Any]]:
    source, value = _read(path, code=code)
    if (
        value.get("schema_version") != schema_version
        or value.get(digest_field)
        != canonical_digest(value, digest_field=digest_field)
    ):
        raise ValueError(code)
    return source, value


def _inside(path: Path, root: Path, *, code: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(code)


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _build_adoption(
    *,
    run_root: str | Path,
    standing_consumption_path: str | Path,
    direct_allocator_result_path: str | Path,
    direct_attempt_authority_path: str | Path,
    direct_authority_consumption_path: str | Path,
    post_teardown_provider_zero_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("direct_execution_run_root_invalid")
    request_path, request = _identity(
        root / "launch_request.json",
        schema_version="task_evaluation_launch_request.v1",
        digest_field="request_digest",
        code="direct_execution_website_identity_invalid",
    )
    profile_path, profile = _identity(
        root / "launch_profile.json",
        schema_version="task_evaluation_launch_profile.v1",
        digest_field="profile_digest",
        code="direct_execution_website_identity_invalid",
    )
    binding_path, binding = _identity(
        root / "launch_binding.json",
        schema_version="task_evaluation_launch_binding.v1",
        digest_field="binding_digest",
        code="direct_execution_website_identity_invalid",
    )
    started_path, started = _identity(
        root / "launch_started.json",
        schema_version="task_evaluation_launch_started.v1",
        digest_field="started_digest",
        code="direct_execution_website_identity_invalid",
    )
    receipt_path, receipt = _identity(
        root / "launch_receipt.json",
        schema_version="task_evaluation_launch_receipt.v1",
        digest_field="receipt_digest",
        code="direct_execution_dispatcher_refusal_invalid",
    )
    dispatcher_result_path, dispatcher_result = _read(
        root / "allocator" / "result.json",
        code="direct_execution_dispatcher_refusal_invalid",
    )
    launch_id = request.get("launch_id")
    request_digest = request.get("request_digest")
    profile_digest = profile.get("profile_digest")
    binding_digest = binding.get("binding_digest")
    terminal = receipt.get("terminal_evidence")
    terminal_result = terminal.get("result") if isinstance(terminal, Mapping) else None
    if (
        not isinstance(launch_id, str)
        or not launch_id
        or root.name != launch_id
        or request.get("run_id") != launch_id
        or request.get("launch_profile_id") != profile.get("profile_id")
        or request.get("launch_profile_digest") != profile_digest
        or binding.get("launch_id") != launch_id
        or binding.get("run_id") != launch_id
        or binding.get("request_digest") != request_digest
        or binding.get("profile_digest") != profile_digest
        or binding.get("execute_requested") is not True
        or started.get("launch_id") != launch_id
        or started.get("run_id") != launch_id
        or started.get("request_digest") != request_digest
        or started.get("binding_digest") != binding_digest
        or started.get("automatic_retry_authorized") is not False
        or receipt.get("status") != "blocked"
        or receipt.get("launch_id") != launch_id
        or receipt.get("run_id") != launch_id
        or receipt.get("request_digest") != request_digest
        or receipt.get("launch_profile_digest") != profile_digest
        or receipt.get("binding_digest") != binding_digest
        or receipt.get("execute_requested") is not True
        or not isinstance(terminal_result, Mapping)
        or terminal_result.get("path") != str(dispatcher_result_path)
        or terminal_result.get("digest") != _sha256(dispatcher_result_path)
        or terminal_result.get("exists") is not True
        or dispatcher_result.get("status") != "blocked"
        or dispatcher_result.get("result_digest")
        != canonical_digest(dispatcher_result, digest_field="result_digest")
        or dispatcher_result.get("provider_mutations_performed") != 0
        or not dispatcher_result.get("blockers")
    ):
        raise ValueError("direct_execution_dispatcher_refusal_invalid")

    standing_path, standing = _read(
        standing_consumption_path,
        code="direct_execution_standing_consumption_invalid",
    )
    _inside(standing_path, root.parent.parent, code="direct_execution_standing_consumption_invalid")
    allocator = profile.get("allocator")
    if (
        standing.get("schema_version")
        != "task_evaluation_standing_launch_authorization.v1"
        or standing.get("profile_id") != profile.get("profile_id")
        or standing.get("launch_id") != launch_id
        or not isinstance(allocator, Mapping)
        or standing.get("max_spend_usd") != allocator.get("max_spend_usd")
    ):
        raise ValueError("direct_execution_standing_consumption_invalid")

    direct_path, direct = _identity(
        Path(direct_allocator_result_path),
        schema_version="native_task_arena_vast_run.v1",
        digest_field="result_digest",
        code="direct_execution_allocator_result_invalid",
    )
    direct_root = direct_path.parent
    if (
        direct_path.name != "result.json"
        or direct_root.parent != root / "allocator"
        or not (
            direct_root.name == "direct-execute"
            or direct_root.name.startswith("direct-execute-r")
        )
    ):
        raise ValueError("direct_execution_allocator_result_invalid")
    job_result_path, job_result = _read(
        direct_root / "arena-construction-job" / "adp_arena_vast_result.json",
        code="direct_execution_allocator_result_invalid",
    )
    if job_result != direct or job_result_path.read_bytes() != direct_path.read_bytes():
        raise ValueError("direct_execution_allocator_result_invalid")

    authority_path, authority = _identity(
        Path(direct_attempt_authority_path),
        schema_version="native_task_arena_paid_attempt_authority.v1",
        digest_field="authorization_digest",
        code="direct_execution_authority_invalid",
    )
    authority_consumption_path, authority_consumption = _read(
        direct_authority_consumption_path,
        code="direct_execution_authority_consumption_invalid",
    )
    direct_consumption = direct.get("authorization_consumption")
    if (
        authority.get("provider") != "vast"
        or authority.get("paid_compute_authorized") is not True
        or authority.get("maximum_paid_attempts") != 1
        or authority.get("maximum_provider_allocations") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("retain_warm_session") is not False
        or authority_consumption.get("schema_version")
        != "native_task_arena_authority_consumption.v1"
        or authority_consumption.get("authorization_digest")
        != authority.get("authorization_digest")
        or authority_consumption.get("bundle_sha256") != authority.get("bundle_sha256")
        or authority_consumption.get("maximum_provider_allocations") != 1
        or not isinstance(direct_consumption, Mapping)
        or direct_consumption.get("status") != "consumed"
        or direct_consumption.get("authorization_digest")
        != authority.get("authorization_digest")
        or direct_consumption.get("consumption_record_sha256")
        != _sha256(authority_consumption_path)
        or direct.get("bundle_sha256") != authority.get("bundle_sha256")
    ):
        raise ValueError("direct_execution_authority_consumption_invalid")

    attempt_root = Path(str(direct.get("attempt_root") or ""))
    expected_attempt_parent = direct_root / "arena-construction-job" / "attempts"
    if attempt_root.parent != expected_attempt_parent:
        raise ValueError("direct_execution_attempt_layout_invalid")
    paths = {
        "provider_adapter_result": Path(str(direct.get("adapter_result_path") or "")),
        "teardown_manifest": Path(str(direct.get("teardown_manifest_path") or "")),
        "independent_watchdog": Path(str(direct.get("watchdog_receipt_path") or "")),
        "object_store_cleanup": Path(str(direct.get("object_store_cleanup_path") or "")),
        "artifact_manifest": Path(str(direct.get("artifact_manifest_path") or "")),
        "native_construction_result": Path(str(direct.get("native_control_result_path") or "")),
    }
    expected_paths = {
        "provider_adapter_result": attempt_root / "vast_provider_run" / "vast_provider_adapter_result.json",
        "teardown_manifest": attempt_root / "vast_provider_run" / "vast_teardown_manifest.json",
        "independent_watchdog": attempt_root / "independent_vast_watchdog" / "groot_oscar_runpod_canary_watchdog.json",
        "object_store_cleanup": attempt_root / "object_store_staging" / "wam_provider_object_store_cleanup.json",
        "artifact_manifest": attempt_root / "artifact_manifest.json",
        "native_construction_result": attempt_root / "immutable_execution" / "native_task_arena_construction_result.v1.json",
    }
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for role, path in paths.items():
        if path != expected_paths[role]:
            raise ValueError(f"direct_execution_{role}_invalid")
        loaded[role] = _read(path, code=f"direct_execution_{role}_invalid")
    adapter = loaded["provider_adapter_result"][1]
    teardown = loaded["teardown_manifest"][1]
    watchdog = loaded["independent_watchdog"][1]
    cleanup = loaded["object_store_cleanup"][1]
    artifact = loaded["artifact_manifest"][1]
    native = loaded["native_construction_result"][1]
    instance_ids = adapter.get("vast_instance_ids")
    embedded_watchdog = direct.get("independent_watchdog")
    if (
        direct.get("status") != "blocked"
        or direct.get("retry_cap") != 0
        or direct.get("continuing_spend_from_this_run") is not False
        or direct.get("raw_secret_values_recorded") is not False
        or not _finite_nonnegative(direct.get("estimated_cost_usd"))
        or direct.get("all_staged_objects_absent") is not True
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or not isinstance(instance_ids[0], int)
        or isinstance(instance_ids[0], bool)
        or adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("status") != "completed"
        or adapter.get("provider_create_attempted") is not True
        or adapter.get("continuing_spend_from_this_run") is not False
        or adapter.get("final_validation_status") != "passed"
        or adapter.get("retained_owned") is not False
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("vast_instance_ids") != instance_ids
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("retention_authorized") is not False
        or not isinstance(embedded_watchdog, Mapping)
        or embedded_watchdog.get("status") != "provider_terminal"
        or embedded_watchdog.get("instance_ids") != instance_ids
        or embedded_watchdog.get("provider_absence_confirmed") is not True
        or embedded_watchdog.get("provider_mutations_performed") != 0
        or watchdog.get("status") != "provider_terminal"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or artifact.get("status") != "completed"
        or artifact.get("manifest_digest")
        != canonical_digest(artifact, digest_field="manifest_digest")
        or native.get("schema_version") != "native_task_arena_construction_result.v1"
        or native.get("status") != "blocked"
        or native.get("construction_gate_qualified") is not False
        or native.get("result_digest")
        != canonical_digest(native, digest_field="result_digest")
        or native.get("blockers") != direct.get("blockers")
    ):
        raise ValueError("direct_execution_terminal_evidence_invalid")

    zero_path, zero = _read(
        post_teardown_provider_zero_path,
        code="direct_execution_provider_zero_invalid",
    )
    if zero_path.parent != direct_root or not valid_adp_paid_provider_zero(zero):
        raise ValueError("direct_execution_provider_zero_invalid")

    sources = {
        "launch_request": _record(request_path, request),
        "launch_profile": _record(profile_path, profile),
        "launch_binding": _record(binding_path, binding),
        "launch_started": _record(started_path, started),
        "original_launch_receipt": _record(receipt_path, receipt),
        "original_dispatcher_result": _record(dispatcher_result_path, dispatcher_result),
        "standing_authorization_consumption": _record(standing_path, standing),
        "direct_allocator_result": _record(direct_path, direct),
        "direct_job_result": _record(job_result_path, job_result),
        "direct_attempt_authority": _record(authority_path, authority),
        "direct_authority_consumption": _record(
            authority_consumption_path, authority_consumption
        ),
        "post_teardown_provider_zero": _record(zero_path, zero),
        **{role: _record(path, value) for role, (path, value) in loaded.items()},
    }
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request_digest,
        "launch_profile_id": profile.get("profile_id"),
        "launch_profile_digest": profile_digest,
        "binding_digest": binding_digest,
        "original_launch_receipt_digest": receipt.get("receipt_digest"),
        "direct_execution_kind": "canonical_allocator_manual_rescue_adopted",
        "direct_execution_directory": direct_root.name,
        "paid_execution_performed": True,
        "provider_mutations_performed": 1,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "bundle_sha256": direct.get("bundle_sha256"),
        "estimated_cost_usd": direct.get("estimated_cost_usd"),
        "authorization_consumption": dict(direct_consumption),
        "provider_instance_id": instance_ids[0],
        "construction_gate_qualified": False,
        "controls_qualified": False,
        "evaluation_ready": False,
        "blockers": list(native["blockers"]),
        "website_projection": {
            "configured_scene_offering_status": "configured_controls_pending",
            "native_construction_status": "blocked",
            "native_construction_blockers": list(native["blockers"]),
            "controls_qualified": False,
            "evaluation_ready": False,
            "qualification_upgrade_performed": False,
        },
        "source_receipts": sources,
        "history_overwritten": False,
        "automatic_retry_performed": False,
        "provider_mutation_performed_by_adoption": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This receipt adopts terminal native construction and resource-closeout "
            "evidence only. It does not qualify controls, promote evaluation readiness, "
            "or replace the original Website dispatcher receipt."
        ),
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("direct_execution_adoption_output_exists") from exc


def materialize_native_direct_execution_adoption(
    *,
    run_root: str | Path,
    standing_consumption_path: str | Path,
    direct_allocator_result_path: str | Path,
    direct_attempt_authority_path: str | Path,
    direct_authority_consumption_path: str | Path,
    post_teardown_provider_zero_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    value = _build_adoption(
        run_root=run_root,
        standing_consumption_path=standing_consumption_path,
        direct_allocator_result_path=direct_allocator_result_path,
        direct_attempt_authority_path=direct_attempt_authority_path,
        direct_authority_consumption_path=direct_authority_consumption_path,
        post_teardown_provider_zero_path=post_teardown_provider_zero_path,
    )
    output = Path(output_path).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    if output.parent != root:
        raise ValueError("direct_execution_adoption_output_invalid")
    _write_exclusive(output, value)
    validate_native_direct_execution_adoption(output)
    return value


def validate_native_direct_execution_adoption(path: str | Path) -> dict[str, Any]:
    source, value = _read(path, code="direct_execution_adoption_invalid")
    sources = value.get("source_receipts")
    if (
        source.name != FILENAME
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("direct_execution_adoption_invalid")
    expected = _build_adoption(
        run_root=source.parent,
        standing_consumption_path=(sources.get("standing_authorization_consumption") or {}).get("path", ""),
        direct_allocator_result_path=(sources.get("direct_allocator_result") or {}).get("path", ""),
        direct_attempt_authority_path=(sources.get("direct_attempt_authority") or {}).get("path", ""),
        direct_authority_consumption_path=(sources.get("direct_authority_consumption") or {}).get("path", ""),
        post_teardown_provider_zero_path=(sources.get("post_teardown_provider_zero") or {}).get("path", ""),
    )
    if expected != value:
        raise ValueError("direct_execution_adoption_invalid")
    return value


def direct_execution_terminal_evidence(
    path: str | Path, *, instance_id: int
) -> dict[str, Any]:
    """Reopen an adoption as the terminal evidence expected by billing."""

    source, _value = _read(path, code="direct_execution_adoption_invalid")
    adoption = validate_native_direct_execution_adoption(source)
    if adoption.get("provider_instance_id") != instance_id:
        raise ValueError("direct_execution_adoption_instance_invalid")
    sources = adoption["source_receipts"]
    terminal_record = _record(source, adoption)
    terminal_record["receipt_digest"] = adoption["receipt_digest"]
    evidence = {
        "terminal_status": adoption["status"],
        "provider_absence_confirmed": True,
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
        "launch_id": adoption["launch_id"],
        "run_id": adoption["run_id"],
        "request_digest": adoption["request_digest"],
        "profile_id": adoption["launch_profile_id"],
        "profile_digest": adoption["launch_profile_digest"],
        "terminal_result": terminal_record,
        "direct_execution_adoption": terminal_record,
    }
    role_map = {
        "provider_adapter_result": "provider_adapter_result",
        "teardown_manifest": "teardown_manifest",
        "post_teardown_provider_zero": "post_teardown_provider_zero",
        "launch_request": "launch_request",
        "launch_profile": "launch_profile",
        "launch_binding": "launch_binding",
        "launch_started": "launch_started",
        "original_launch_receipt": "launch_receipt",
        "independent_watchdog": "independent_watchdog",
        "object_store_cleanup": "object_store_cleanup",
        "artifact_manifest": "artifact_manifest",
        "native_construction_result": "native_construction_result",
    }
    for source_role, evidence_role in role_map.items():
        evidence[evidence_role] = dict(sources[source_role])
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--standing-consumption", required=True)
    parser.add_argument("--direct-allocator-result", required=True)
    parser.add_argument("--direct-attempt-authority", required=True)
    parser.add_argument("--direct-authority-consumption", required=True)
    parser.add_argument("--post-teardown-provider-zero", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = materialize_native_direct_execution_adoption(
            run_root=args.run_root,
            standing_consumption_path=args.standing_consumption,
            direct_allocator_result_path=args.direct_allocator_result,
            direct_attempt_authority_path=args.direct_attempt_authority,
            direct_authority_consumption_path=args.direct_authority_consumption,
            post_teardown_provider_zero_path=args.post_teardown_provider_zero,
            output_path=args.output,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps({"status": "materialized", "receipt_digest": value["receipt_digest"]}, sort_keys=True))
    return 0


__all__ = [
    "FILENAME",
    "SCHEMA_VERSION",
    "direct_execution_terminal_evidence",
    "main",
    "materialize_native_direct_execution_adoption",
    "validate_native_direct_execution_adoption",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
