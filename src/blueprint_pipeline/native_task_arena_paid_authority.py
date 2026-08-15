"""Single-use paid authority and spend chain for native Arena execution.

The first native Arena attempt follows the zero-closed paired replacement-import
probe.  Every later construction, control, or policy attempt follows the prior
zero-closed native Arena attempt.  Binding only the immediate predecessor keeps
the interface small while its digest-bound authority retains the complete
recursive spend lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_construction_bundle import (
    load_verified_native_task_arena_construction_bundle,
)
from .native_task_arena_controls_bundle import (
    load_verified_native_task_arena_controls_bundle,
)
from .native_task_arena_policy_bundle import load_verified_native_task_arena_policy_bundle
from .paid_attempt_authority import (
    bind_lane_prior_spend,
    normalize_active_instance_allowlist,
    validate_bound_lane_prior_spend,
)
from .spend_authority_consumption_root import consumption_root


AUTHORITY_SCHEMA_VERSION = "native_task_arena_paid_attempt_authority.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "native_task_arena_provider_zero.v1"
CONSUMPTION_SCHEMA_VERSION = "native_task_arena_authority_consumption.v1"
AGGREGATE_GOAL_SPEND_CAP_USD = 12.0
MAX_HARD_CAP_USD = 2.0
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 14_400


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound_record(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def _finite_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("native_task_arena_terminal_cost_invalid")
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("native_task_arena_terminal_cost_invalid")
    return cost


def validate_terminal_spend_chain(
    *, authority_path: str | Path, result_path: str | Path, provider_zero_path: str | Path
) -> dict[str, Any]:
    """Validate one immediate terminal predecessor and return cumulative spend."""

    authority_file = Path(authority_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    zero_file = Path(provider_zero_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_predecessor_authority_invalid")
    result = _read(result_file, "native_task_arena_predecessor_result_invalid")
    zero = _read(zero_file, "native_task_arena_predecessor_provider_zero_invalid")
    schema = authority.get("schema_version")
    if schema == "paired_target_native_import_paid_attempt_authority.v1":
        authority_digest_field = "authorization_digest"
        result_schema = "paired_target_native_import_vast_run.v1"
        zero_schema = "paired_target_native_import_provider_zero.v1"
        zero_digest_field = "receipt_digest"
        consumption = result.get("authorization_consumption") or {}
        result_authority_digest = consumption.get("authorization_digest")
        zero_authority_digest = zero.get("attempt_authority_digest")
        zero_result_record = zero.get("terminal_result")
    elif schema == AUTHORITY_SCHEMA_VERSION:
        authority_digest_field = "authorization_digest"
        result_schema = "native_task_arena_vast_run.v1"
        zero_schema = PROVIDER_ZERO_SCHEMA_VERSION
        zero_digest_field = "receipt_digest"
        consumption = result.get("authorization_consumption") or {}
        result_authority_digest = consumption.get("authorization_digest")
        zero_authority_digest = zero.get("attempt_authority_digest")
        zero_result_record = zero.get("terminal_result")
    else:
        raise ValueError("native_task_arena_predecessor_authority_schema_invalid")
    authorization_digest = authority.get(authority_digest_field)
    cost = _finite_cost(result.get("estimated_cost_usd"))
    before = _finite_cost(authority.get("aggregate_goal_spend_before_attempt_usd"))
    cap = _finite_cost(authority.get("aggregate_goal_spend_cap_usd"))
    zero_result_path, zero_result_bound = _bound_record(
        zero_result_record, "native_task_arena_predecessor_zero_result_unbound"
    )
    if (
        authorization_digest != canonical_digest(authority, digest_field=authority_digest_field)
        or result.get("schema_version") != result_schema
        or result.get("status") not in {"completed", "blocked"}
        or result_authority_digest != authorization_digest
        or result.get("bundle_sha256") != authority.get("bundle_sha256")
        or result.get("hard_cap_usd") != authority.get("hard_attempt_spend_cap_usd")
        or result.get("hard_ttl_seconds")
        != authority.get("maximum_single_resource_ttl_seconds")
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or zero.get("schema_version") != zero_schema
        or zero.get("status") != "completed"
        or zero.get(zero_digest_field)
        != canonical_digest(zero, digest_field=zero_digest_field)
        or zero_authority_digest != authorization_digest
        or zero.get("provider_zero_confirmed") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("all_staged_objects_absent") is not True
        or zero_result_bound.get("sha256") != _sha256(result_file)
        or before + cost > cap
    ):
        raise ValueError("native_task_arena_terminal_spend_chain_invalid")
    return {
        "authority_digest": authorization_digest,
        "attempt_cost_usd": round(cost, 6),
        "aggregate_goal_spend_before_attempt_usd": round(before, 6),
        "aggregate_goal_spend_after_attempt_usd": round(before + cost, 6),
        "aggregate_goal_spend_cap_usd": cap,
        "records": {
            "authority": _record(authority_file),
            # The zero receipt names the canonical retained terminal result.
            # A caller may supply a byte-identical allocator alias, but the
            # successor always binds the path that provider-zero actually
            # attested.
            "terminal_result": _record(zero_result_path),
            "provider_zero": _record(zero_file),
        },
    }


def _bundle_loader(mode: str):
    return {
        "construction_canary": load_verified_native_task_arena_construction_bundle,
        "controls": load_verified_native_task_arena_controls_bundle,
        "policy": load_verified_native_task_arena_policy_bundle,
    }[mode]


def materialize_native_task_arena_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    prior_authority_path: str | Path,
    prior_result_path: str | Path,
    prior_provider_zero_path: str | Path,
    prior_spend_reconciliation_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Seal one zero-retry authority against a bundle and terminal predecessor."""

    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    raw_bundle = _read(receipt_file, "native_task_arena_bundle_receipt_invalid")
    mode = str(raw_bundle.get("execution_mode") or "")
    bundle = _bundle_loader(mode)(
        receipt_file,
        expected_implementation_commit=blueprint_commit,
        expected_packet_receipt_digest=raw_bundle.get("packet_receipt_digest"),
        expected_runtime_source_packet_digest=(raw_bundle.get("runtime_source_packet") or {}).get(
            "receipt_digest"
        ),
    )
    prior = validate_terminal_spend_chain(
        authority_path=prior_authority_path,
        result_path=prior_result_path,
        provider_zero_path=prior_provider_zero_path,
    )
    reconciled = bind_lane_prior_spend(
        prior_result_paths=(prior["records"]["terminal_result"]["path"],),
        reconciliation_path=prior_spend_reconciliation_path,
        lane="native_task_arena",
    )
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    prior_spend = round(
        prior["aggregate_goal_spend_before_attempt_usd"]
        + reconciled["actual_total_usd"],
        6,
    )
    aggregate_cap = min(AGGREGATE_GOAL_SPEND_CAP_USD, prior["aggregate_goal_spend_cap_usd"])
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
        or prior_spend + hard_cap_usd > aggregate_cap
        or any(value <= 0 for value in allowed)
    ):
        raise ValueError("native_task_arena_authority_configuration_invalid")
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_native_task_arena_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt": _record(receipt_file),
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_input_digest": bundle["input_digest"],
        "packet_receipt_digest": bundle["packet_receipt_digest"],
        "runtime_source_packet_receipt_digest": bundle["runtime_source_packet"][
            "receipt_digest"
        ],
        "execution_mode": mode,
        "policy_candidate_id": bundle.get("policy_candidate_id"),
        "blueprint_commit": blueprint_commit,
        "container_image": bundle["container_image"],
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": prior_spend,
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "prior_terminal_attempt": {
            **prior["records"],
            "authority_digest": prior["authority_digest"],
            "attempt_cost_usd": prior["attempt_cost_usd"],
            "actual_provider_charge_usd": reconciled["actual_total_usd"],
        },
        "prior_terminal_attempts": reconciled["prior_terminal_attempts"],
        "prior_spend_reconciliation": reconciled["reconciliation"],
        "prior_actual_provider_spend_usd": reconciled["actual_total_usd"],
        "active_instance_allowlist": {
            "external_provider_owned": list(allowed),
            "same_goal_concurrent": [],
        },
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("native_task_arena_authority_output_exists")
    ensure_dir(output.parent)
    write_json(output, authority)
    validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed,
    )
    return authority


def validate_native_task_arena_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    value = dict(authority)
    expected_allowlist = {
        "external_provider_owned": tuple(sorted(set(allowed_active_instance_ids))),
        "same_goal_concurrent": (),
    }
    observed_allowlist = normalize_active_instance_allowlist(value.get("active_instance_allowlist"))
    errors: list[str] = []
    expected = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "purpose": "one_shot_native_task_arena_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "bundle_input_digest": prepared_bundle.get("input_digest"),
        "packet_receipt_digest": prepared_bundle.get("packet_receipt_digest"),
        "runtime_source_packet_receipt_digest": (
            prepared_bundle.get("runtime_source_packet") or {}
        ).get("receipt_digest"),
        "execution_mode": prepared_bundle.get("execution_mode"),
        "policy_candidate_id": prepared_bundle.get("policy_candidate_id"),
        "blueprint_commit": prepared_bundle.get("implementation_commit"),
        "container_image": prepared_bundle.get("container_image"),
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
    }
    errors.extend(
        f"{key}_mismatch" for key, expected_value in expected.items() if value.get(key) != expected_value
    )
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("digest_invalid")
    if observed_allowlist != expected_allowlist:
        errors.append("active_instance_allowlist_mismatch")
    try:
        receipt_path, _ = _bound_record(value.get("bundle_receipt"), "bundle_receipt_unbound")
        if receipt_path != Path(str(prepared_bundle.get("bundle_path"))).parent / "native_task_arena_provider_bundle_receipt.v1.json":
            errors.append("bundle_receipt_path_mismatch")
        predecessor = value.get("prior_terminal_attempt")
        if not isinstance(predecessor, Mapping):
            raise ValueError("predecessor_invalid")
        paths = {
            key: _bound_record(predecessor.get(key), "predecessor_unbound")[0]
            for key in ("authority", "terminal_result", "provider_zero")
        }
        prior = validate_terminal_spend_chain(
            authority_path=paths["authority"],
            result_path=paths["terminal_result"],
            provider_zero_path=paths["provider_zero"],
        )
        reconciled = validate_bound_lane_prior_spend(
            value, lane="native_task_arena"
        )
        actual_after = round(
            prior["aggregate_goal_spend_before_attempt_usd"]
            + reconciled["actual_total_usd"],
            6,
        )
        if (
            predecessor.get("authority_digest") != prior["authority_digest"]
            or predecessor.get("attempt_cost_usd") != prior["attempt_cost_usd"]
            or predecessor.get("actual_provider_charge_usd")
            != reconciled["actual_total_usd"]
            or value.get("aggregate_goal_spend_before_attempt_usd")
            != actual_after
            or value.get("aggregate_goal_spend_cap_usd")
            != prior["aggregate_goal_spend_cap_usd"]
            or value.get("aggregate_goal_spend_before_attempt_usd", 0) + hard_cap_usd
            > value.get("aggregate_goal_spend_cap_usd", 0)
        ):
            errors.append("prior_terminal_spend_mismatch")
    except ValueError:
        errors.append("prior_terminal_spend_invalid")
    if errors:
        raise ValueError("native_task_arena_authority_invalid:" + ",".join(sorted(set(errors))))
    return value


def consume_native_task_arena_authority_once(authority: Mapping[str, Any]) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["native_task_arena_authority_identity_invalid"]}
    root = consumption_root()
    payload = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "authorization_digest": digest,
        "bundle_sha256": authority.get("bundle_sha256"),
        "blueprint_commit": authority.get("blueprint_commit"),
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 1,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = root.stat()
        if root.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise OSError("insecure_root")
        destination = root / f"native-task-arena-{digest[7:]}.json"
        temporary = root / f".native-task-arena-{digest[7:]}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["native_task_arena_authority_consumed"]}
    except OSError:
        return {"status": "blocked", "blockers": ["native_task_arena_authority_consumption_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def materialize_native_task_arena_provider_zero(
    *, authority_path: str | Path, result_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Bind terminal teardown, object cleanup, and watchdog API inventory."""

    authority_file = Path(authority_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_authority_unreadable")
    result = _read(result_file, "native_task_arena_result_unreadable")
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = Path(str(result.get("object_store_cleanup_path") or "")).resolve()
    adapter_path = Path(str(result.get("adapter_result_path") or "")).resolve()
    teardown_path = Path(str(result.get("teardown_manifest_path") or "")).resolve()
    watchdog = _read(watchdog_path, "native_task_arena_watchdog_unreadable")
    cleanup = _read(cleanup_path, "native_task_arena_cleanup_unreadable")
    adapter = _read(adapter_path, "native_task_arena_adapter_unreadable")
    teardown = _read(teardown_path, "native_task_arena_teardown_unreadable")
    inventory = watchdog.get("final_global_inventory")
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or result.get("schema_version") != "native_task_arena_vast_run.v1"
        or result.get("status") not in {"completed", "blocked"}
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or teardown.get("continuing_spend_from_this_run") is not False
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
        or adapter.get("continuing_spend_from_this_run") is not False
    ):
        raise ValueError("native_task_arena_provider_zero_invalid")
    receipt: dict[str, Any] = {
        "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed",
        "attempt_authority": _record(authority_file),
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(result_file),
        "provider_adapter": _record(adapter_path),
        "teardown": _record(teardown_path),
        "watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_zero_confirmed": True,
        "inventory": inventory,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("native_task_arena_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "PROVIDER_ZERO_SCHEMA_VERSION",
    "consume_native_task_arena_authority_once",
    "materialize_native_task_arena_paid_attempt_authority",
    "materialize_native_task_arena_provider_zero",
    "validate_native_task_arena_paid_attempt_authority",
    "validate_terminal_spend_chain",
]
