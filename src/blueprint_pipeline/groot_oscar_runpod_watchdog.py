"""Independent name-bound hard-TTL watchdog for the GR00T + OSCAR canary.

The historical module name and evidence filename are retained for compatibility,
but the watchdog can guard either a RunPod pod or a Vast instance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider
from .paid_lane_guard import load_pending_teardowns
from .watchdog_owner_teardown_contract import (
    OWNER_TEARDOWN_CANCEL_NAME,
    OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION,
    WATCHDOG_EVIDENCE_NAME,
    write_owner_teardown_cancel_request as _write_owner_teardown_cancel_request,
)

SCHEMA_VERSION = "groot_oscar_runpod_canary_watchdog.v1"
EVIDENCE_NAME = WATCHDOG_EVIDENCE_NAME
write_owner_teardown_cancel_request = _write_owner_teardown_cancel_request
SUPPORTED_PROVIDERS = ("runpod", "vast")
VAST_STARTED_INSTANCE_ID_NAME = "started_vast_instance_id.txt"
CANARY_NAME_PREFIXES = (
    "blueprint-groot-oscar-canary-",
    # SAM 3.1 source-track jobs are bounded Vast canaries with their own
    # exact-prefix instance handoff and teardown scope.
    "blueprint-sam31-source-tracks-",
    # ADP's exact-mask Aura residual lane is a bounded, independently
    # watched Vast canary. Keep its resource name scope explicit rather than
    # borrowing the historical GR00T name family.
    "blueprint-adp-aura-exact-residual-",
    # Reusable 1--5 replacement-object native Isaac import canary. Its exact
    # per-run suffix and instance handoff are still bound by the watchdog.
    "blueprint-adp-paired-native-import-",
    "blueprint-native-warehouse-camera-",
    "blueprint-reconstruction-",
    "blueprint-measurement-isaac-",
    "blueprint-measurement-dlo-",
    "blueprint-measurement-chrono-dem-",
)
CAMPAIGN_PENDING_TEARDOWN_LANES = {
    "persistent_policy_wam_loop": "runpod_wam_async",
    "openpi_policy_ranking": "openpi_policy_ranking_gpu_canary",
    "nvidia_warehouse_native_camera": ("nvidia_warehouse_native_camera_gpu_canary"),
}


def _provider_name(value: Any) -> str:
    name = str(value or "runpod").strip().lower()
    if name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"watchdog_provider_unsupported:{name}")
    return name


def _owner_teardown_cancel_request(
    *, root: Path, pod_name_prefix: str, provider_name: str
) -> dict[str, Any]:
    """Load a private, exact-prefix cancellation request from owner teardown.

    The request is only a prompt to perform the watchdog's normal provider API
    teardown/absence verification early. It never turns file contents into a
    provider-terminal claim by themselves.
    """

    path = root / OWNER_TEARDOWN_CANCEL_NAME
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or metadata.st_size > 16 * 1024
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    if (
        payload.get("schema_version") != OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION
        or payload.get("requested_by") != "qualification_owner_teardown"
        or payload.get("provider") != provider_name
        or payload.get("pod_name_prefix") != pod_name_prefix
        or not str(payload.get("instance_id") or "").strip()
        or payload.get("provider_absence_confirmed") is not True
        or payload.get("provider_absence_evidence")
        != "provider_api_exact_id_prefix_and_global_inventory"
    ):
        return {}
    return dict(payload)


def _vast_billable_inventory(*, provider: Any, name_prefix: str) -> dict[str, Any]:
    """Return active Vast instances whose launch labels match ``name_prefix``.

    ``VastRenderProvider.stop`` already destroys an instance, but its generic
    base-class inventory method cannot prove name-scoped absence.  Keep this
    small read-only adapter in the independent watchdog so a Vast allocation is
    not left without a hard-TTL owner.  An active row without a label is
    ambiguous and therefore blocks an absence claim instead of being ignored.
    """

    key_reader = getattr(provider, "_key", None)
    api_key = key_reader() if callable(key_reader) else None
    if not api_key:
        return {
            "status": "blocked",
            "provider": "vast",
            "name_prefix": str(name_prefix),
            "live_resource_count": None,
            "resources": [],
            "api_confirmed": False,
            "blockers": ["vast_api_key_missing"],
            "raw_provider_response_recorded": False,
        }
    try:
        from .vast_provider_adapter import (
            VAST_TERMINAL_INSTANCE_STATUSES,
            _api_json,
            _instance_list_rows,
            _instance_status,
        )

        http_status, payload = _api_json(
            method="GET",
            path="/instances/",
            api_key=api_key,
            timeout_seconds=30,
        )
    except Exception as exc:  # noqa: BLE001 - evidence stays secret-safe
        return {
            "status": "blocked",
            "provider": "vast",
            "name_prefix": str(name_prefix),
            "live_resource_count": None,
            "resources": [],
            "api_confirmed": False,
            "blockers": ["vast_billable_inventory_failed"],
            "error_type": type(exc).__name__,
            "raw_provider_response_recorded": False,
        }
    if not 200 <= int(http_status) < 300 or not isinstance(payload, Mapping):
        return {
            "status": "blocked",
            "provider": "vast",
            "name_prefix": str(name_prefix),
            "live_resource_count": None,
            "resources": [],
            "api_confirmed": False,
            "blockers": ["vast_billable_inventory_failed"],
            "http": http_status,
            "raw_provider_response_recorded": False,
        }

    prefix = str(name_prefix or "")
    terminal_statuses = {str(item).lower() for item in VAST_TERMINAL_INSTANCE_STATUSES}
    resources: list[dict[str, Any]] = []
    ambiguous_live_rows = 0
    for row in _instance_list_rows(payload):
        status = str(_instance_status(row) or "").strip().lower()
        if status in terminal_statuses:
            continue
        label = str(row.get("label") or "").strip()
        if not label:
            ambiguous_live_rows += 1
            continue
        if prefix and not label.startswith(prefix):
            continue
        resources.append(
            {
                "instance_id": str(
                    row.get("id") or row.get("instance_id") or row.get("contract_id") or ""
                ),
                "name": label,
                "status": status or None,
                "gpu_name": row.get("gpu_name") or row.get("gpu_display_name"),
                "cost_per_hour": row.get("dph_total") or row.get("price_per_hour"),
            }
        )
    if ambiguous_live_rows:
        return {
            "status": "blocked",
            "provider": "vast",
            "name_prefix": prefix,
            "live_resource_count": None,
            "resources": resources,
            "api_confirmed": False,
            "ambiguous_live_resource_count": ambiguous_live_rows,
            "blockers": ["vast_active_instance_label_missing"],
            "http": http_status,
            "raw_provider_response_recorded": False,
        }
    return {
        "status": "observed",
        "provider": "vast",
        "name_prefix": prefix,
        "live_resource_count": len(resources),
        "resources": resources,
        "api_confirmed": True,
        "http": http_status,
        "raw_provider_response_recorded": False,
    }


def _billable_inventory(*, provider: Any, provider_name: str, name_prefix: str) -> dict[str, Any]:
    if provider_name == "vast":
        return _vast_billable_inventory(provider=provider, name_prefix=name_prefix)
    return provider.billable_inventory(name_prefix=name_prefix)


def _inventory_contains_only_allowed_resources(
    inventory: Mapping[str, Any], *, allowed_instance_ids: set[str]
) -> bool:
    """Prove global inventory contains only explicitly authorized siblings."""

    resources = inventory.get("resources")
    count = inventory.get("live_resource_count")
    if (
        inventory.get("api_confirmed") is not True
        or isinstance(count, bool)
        or not isinstance(count, int)
        or not isinstance(resources, list)
        or len(resources) != count
    ):
        return False
    observed: set[str] = set()
    for row in resources:
        if not isinstance(row, Mapping):
            return False
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            return False
        observed.add(instance_id)
    return len(observed) == count and observed.issubset(allowed_instance_ids)


def _recorded_vast_instance(*, armed: Mapping[str, Any], pod_name_prefix: str) -> dict[str, Any]:
    """Read the one Vast id owned by this watchdog's attempt directory.

    Vast's list API intentionally excludes terminal rows.  A contract can
    therefore disappear from billable inventory after its workload exits but
    before ``DELETE /instances/{id}/`` destroys the contract.  The launcher's
    attempt-local started-id file is the durable ownership record for closing
    that gap.  Only the exact file under the armed watchdog directory is
    accepted, and its armed name prefix must match the teardown scope.
    """

    root_value = str(armed.get("watchdog_out_dir") or "").strip()
    if not root_value:
        return {"status": "not_recorded", "required": False}
    root = Path(root_value).expanduser().resolve()
    path = root / VAST_STARTED_INSTANCE_ID_NAME
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return {
            "status": "not_recorded",
            "required": False,
            "path": str(path),
        }
    except OSError as exc:
        return {
            "status": "blocked",
            "required": True,
            "path": str(path),
            "blockers": ["vast_started_instance_id_unreadable"],
            "error_type": type(exc).__name__,
        }

    armed_prefix = str(armed.get("pod_name_prefix") or "").strip()
    if armed_prefix != pod_name_prefix:
        return {
            "status": "blocked",
            "required": True,
            "path": str(path),
            "blockers": ["vast_started_instance_id_scope_mismatch"],
        }
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
        return {
            "status": "blocked",
            "required": True,
            "path": str(path),
            "blockers": ["vast_started_instance_id_unsafe_file"],
        }
    if file_stat.st_size > 64:
        return {
            "status": "blocked",
            "required": True,
            "path": str(path),
            "blockers": ["vast_started_instance_id_invalid"],
        }
    try:
        instance_id = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        return {
            "status": "blocked",
            "required": True,
            "path": str(path),
            "blockers": ["vast_started_instance_id_unreadable"],
            "error_type": type(exc).__name__,
        }
    if (
        not instance_id
        or not instance_id.isascii()
        or not instance_id.isdigit()
        or int(instance_id) <= 0
    ):
        return {
            "status": "blocked",
            "required": True,
            "path": str(path),
            "blockers": ["vast_started_instance_id_invalid"],
        }
    return {
        "status": "recorded",
        "required": True,
        "path": str(path),
        "instance_id": instance_id,
        "scope_confirmed": True,
        "pod_name_prefix": pod_name_prefix,
    }


def _safe_vast_inspect_evidence(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, Mapping) else {}
    allowed = (
        "status",
        "provider",
        "http",
        "instance_id",
        "api_confirmed",
        "provider_absence_confirmed",
        "desiredStatus",
        "actual_status",
        "cur_state",
        "intended_status",
        "name",
        "error_type",
        "blockers",
    )
    return {key: row[key] for key in allowed if key in row}


def _vast_instance_absence_proven(value: Mapping[str, Any], instance_id: str) -> bool:
    observed_id = str(value.get("instance_id") or "").strip()
    return bool(
        value.get("api_confirmed") is True
        and value.get("provider_absence_confirmed") is True
        and value.get("status") == "absent"
        and (not observed_id or observed_id == instance_id)
    )


def _vast_instance_presence_proven(value: Mapping[str, Any], instance_id: str) -> bool:
    observed_id = str(value.get("instance_id") or "").strip()
    return bool(
        value.get("api_confirmed") is True
        and value.get("status") == "observed"
        and (not observed_id or observed_id == instance_id)
    )


def _vast_delete_absence_proven(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("status") in {"stopped", "terminated"}
        and value.get("already_gone") is True
        and value.get("http") in {404, 410}
    )


def arm_watchdog(
    *,
    out_dir: str | Path,
    pod_name_prefix: str,
    deadline_epoch: float,
    pid: int | None = None,
    provider_name: str = "runpod",
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not pod_name_prefix.startswith(CANARY_NAME_PREFIXES):
        raise ValueError("watchdog_pod_name_prefix_not_canary_scoped")
    if float(deadline_epoch) <= time.time() + 60:
        raise ValueError("watchdog_deadline_must_be_more_than_60_seconds_future")
    resolved_provider = _provider_name(provider_name)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "armed",
        "independent_process": True,
        "pid": int(pid if pid is not None else os.getpid()),
        "armed_at": utc_now_iso(),
        "deadline_epoch": float(deadline_epoch),
        "provider": resolved_provider,
        "allowed_active_instance_ids": sorted(
            {int(value) for value in allowed_active_instance_ids}
        ),
        "pod_name_prefix": pod_name_prefix,
        # Reconstruction executors use the provider-neutral ``name_prefix``
        # spelling while the historical watchdog contract uses
        # ``pod_name_prefix``. Bind both to the same exact teardown scope.
        "name_prefix": pod_name_prefix,
        "watchdog_out_dir": str(root),
        "provider_mutation_trigger": "hard_deadline_only",
        "pre_deadline_provider_mutation_allowed": False,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    write_json(root / EVIDENCE_NAME, payload)
    return payload


def terminate_canary_resources(
    *,
    provider: Any,
    pod_name_prefix: str,
    armed: Mapping[str, Any],
    provider_name: str | None = None,
) -> dict[str, Any]:
    resolved_provider = _provider_name(
        provider_name or armed.get("provider") or getattr(provider, "name", None)
    )
    recorded_vast_instance = (
        _recorded_vast_instance(
            armed=armed,
            pod_name_prefix=pod_name_prefix,
        )
        if resolved_provider == "vast"
        else None
    )
    try:
        inventory = _billable_inventory(
            provider=provider,
            provider_name=resolved_provider,
            name_prefix=pod_name_prefix,
        )
    except Exception as exc:  # noqa: BLE001 - persist terminal watchdog uncertainty
        if resolved_provider != "vast":
            return {
                **dict(armed),
                "status": "teardown_unverified",
                "completed_at": utc_now_iso(),
                "initial_inventory": {"api_confirmed": False},
                "terminations": [],
                "final_inventory": {"api_confirmed": False},
                "provider_absence_confirmed": False,
                "provider_mutations_performed": 0,
                "teardown_error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
        # The attempt-local Vast id is sufficient authority to attempt exact-id
        # destruction even when the name-scoped inventory endpoint is down.
        # Absence still fails closed unless the final inventory also recovers.
        inventory = {
            "status": "blocked",
            "provider": "vast",
            "name_prefix": pod_name_prefix,
            "live_resource_count": None,
            "resources": [],
            "api_confirmed": False,
            "blockers": ["vast_billable_inventory_failed"],
            "error_type": type(exc).__name__,
        }
    resources = inventory.get("resources")
    resources = resources if isinstance(resources, list) else []
    terminations: list[dict[str, Any]] = []
    recorded_vast_instance_id = ""
    if (
        isinstance(recorded_vast_instance, Mapping)
        and recorded_vast_instance.get("status") == "recorded"
    ):
        recorded_vast_instance_id = str(recorded_vast_instance.get("instance_id") or "").strip()
    recorded_delete_proven = False
    terminated_ids: list[str] = []
    for row in resources:
        row = row if isinstance(row, Mapping) else {}
        instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
        if not instance_id:
            terminations.append({"status": "blocked", "reason": "resource_id_missing"})
            continue
        try:
            result = provider.terminate(instance_id)
        except Exception as exc:  # noqa: BLE001 - continue bounded name-scope cleanup
            result = {
                "status": "teardown_unverified",
                "error_type": type(exc).__name__,
            }
        termination = {"instance_id": instance_id, **dict(result)}
        if instance_id == recorded_vast_instance_id:
            termination["ownership_source"] = VAST_STARTED_INSTANCE_ID_NAME
            recorded_delete_proven = recorded_delete_proven or _vast_delete_absence_proven(
                termination
            )
        terminations.append(termination)
        terminated_ids.append(instance_id)
    if recorded_vast_instance_id and recorded_vast_instance_id not in terminated_ids:
        try:
            result = provider.terminate(recorded_vast_instance_id)
        except Exception as exc:  # noqa: BLE001 - exact-id cleanup still gets verified
            result = {
                "status": "teardown_unverified",
                "error_type": type(exc).__name__,
            }
        termination = {
            "instance_id": recorded_vast_instance_id,
            **dict(result),
            "ownership_source": VAST_STARTED_INSTANCE_ID_NAME,
        }
        recorded_delete_proven = _vast_delete_absence_proven(termination)
        terminations.append(termination)

    recorded_vast_verification: dict[str, Any] | None = None
    if recorded_vast_instance_id:
        inspect_attempts: list[dict[str, Any]] = []
        recorded_absent = False
        # One exact-id GET after the first DELETE is the primary proof. If the
        # provider still reports a terminal row (or the GET is inconclusive),
        # repeat DELETE once and inspect again. A DELETE 404/410 with
        # ``already_gone`` is equivalent exact-contract absence proof.
        for inspect_number in (1, 2):
            try:
                inspected = provider.inspect(recorded_vast_instance_id)
            except Exception as exc:  # noqa: BLE001 - retain secret-safe evidence
                inspected = {
                    "status": "unavailable",
                    "instance_id": recorded_vast_instance_id,
                    "api_confirmed": False,
                    "error_type": type(exc).__name__,
                }
            safe_inspected = _safe_vast_inspect_evidence(inspected)
            safe_inspected["attempt"] = inspect_number
            inspect_attempts.append(safe_inspected)
            if _vast_instance_absence_proven(safe_inspected, recorded_vast_instance_id):
                recorded_absent = True
                break
            if _vast_instance_presence_proven(safe_inspected, recorded_vast_instance_id):
                # A later exact-id GET overrides an earlier DELETE 404/410.
                recorded_delete_proven = False
            if inspect_number == 1:
                try:
                    repeated = provider.terminate(recorded_vast_instance_id)
                except Exception as exc:  # noqa: BLE001 - still inspect afterward
                    repeated = {
                        "status": "teardown_unverified",
                        "error_type": type(exc).__name__,
                    }
                repeated_termination = {
                    "instance_id": recorded_vast_instance_id,
                    **dict(repeated),
                    "ownership_source": VAST_STARTED_INSTANCE_ID_NAME,
                    "attempt": 2,
                }
                terminations.append(repeated_termination)
                if _vast_delete_absence_proven(repeated_termination):
                    recorded_delete_proven = True
        recorded_absent = recorded_absent or recorded_delete_proven
        recorded_vast_verification = {
            "status": "absent" if recorded_absent else "teardown_unverified",
            "instance_id": recorded_vast_instance_id,
            "provider_absence_confirmed": recorded_absent,
            "inspect_attempts": inspect_attempts,
            "delete_absence_proven": recorded_delete_proven,
        }
    elif (
        isinstance(recorded_vast_instance, Mapping)
        and recorded_vast_instance.get("status") == "blocked"
    ):
        recorded_vast_verification = {
            "status": "teardown_unverified",
            "provider_absence_confirmed": False,
            "blockers": list(recorded_vast_instance.get("blockers") or []),
        }
    try:
        final_inventory = _billable_inventory(
            provider=provider,
            provider_name=resolved_provider,
            name_prefix=pod_name_prefix,
        )
        final_error_type = None
    except Exception as exc:  # noqa: BLE001 - persist terminal watchdog uncertainty
        final_inventory = {"api_confirmed": False}
        final_error_type = type(exc).__name__
    inventory_absent = bool(
        final_inventory.get("api_confirmed") is True
        and final_inventory.get("live_resource_count") == 0
    )
    recorded_vast_absent = bool(
        recorded_vast_verification is None
        or recorded_vast_verification.get("provider_absence_confirmed") is True
    )
    absent = inventory_absent and recorded_vast_absent
    result = {
        **dict(armed),
        "status": "provider_terminal" if absent else "teardown_unverified",
        "completed_at": utc_now_iso(),
        "initial_inventory": inventory,
        "terminations": terminations,
        "final_inventory": final_inventory,
        "provider_absence_confirmed": absent,
        "provider_mutations_performed": len(terminations),
        "teardown_error_type": final_error_type,
        "raw_secret_values_recorded": False,
    }
    if resolved_provider == "vast":
        result["recorded_vast_instance"] = recorded_vast_instance
        result["recorded_vast_instance_teardown"] = recorded_vast_verification
    return result


def run_watchdog(
    *,
    out_dir: str | Path,
    pod_name_prefix: str,
    deadline_epoch: float,
    provider_name: str = "runpod",
    allowed_active_instance_ids: Sequence[int] = (),
    provider_factory: Callable[[str], Any] = get_render_provider,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(out_dir).expanduser().resolve()
    resolved_provider = _provider_name(provider_name)
    allowed_ids = {
        str(int(value))
        for value in allowed_active_instance_ids
        if not isinstance(value, bool) and int(value) > 0
    }
    if len(allowed_ids) != len(tuple(allowed_active_instance_ids)):
        raise ValueError("watchdog_allowed_active_instance_ids_invalid")
    armed = arm_watchdog(
        out_dir=root,
        pod_name_prefix=pod_name_prefix,
        deadline_epoch=deadline_epoch,
        provider_name=resolved_provider,
        allowed_active_instance_ids=[int(value) for value in sorted(allowed_ids)],
    )
    owner_teardown_cancel: dict[str, Any] = {}
    cancel_zero_result: dict[str, Any] = {}
    while clock() < deadline_epoch:
        cancel_candidate = _owner_teardown_cancel_request(
            root=root,
            pod_name_prefix=pod_name_prefix,
            provider_name=resolved_provider,
        )
        if cancel_candidate:
            recorded_vast_instance: dict[str, Any] = {}
            recorded_vast_verification: dict[str, Any] = {}
            try:
                cancel_provider = provider_factory(resolved_provider)
                first_zero = _billable_inventory(
                    provider=cancel_provider,
                    provider_name=resolved_provider,
                    name_prefix=pod_name_prefix,
                )
                first_global_zero = _billable_inventory(
                    provider=cancel_provider,
                    provider_name=resolved_provider,
                    name_prefix="",
                )
                second_zero = _billable_inventory(
                    provider=cancel_provider,
                    provider_name=resolved_provider,
                    name_prefix=pod_name_prefix,
                )
                second_global_zero = _billable_inventory(
                    provider=cancel_provider,
                    provider_name=resolved_provider,
                    name_prefix="",
                )
                exact_contract_zero = True
                if resolved_provider == "vast":
                    recorded_vast_instance = _recorded_vast_instance(
                        armed=armed,
                        pod_name_prefix=pod_name_prefix,
                    )
                    recorded_id = str(recorded_vast_instance.get("instance_id") or "").strip()
                    cancel_id = str(cancel_candidate.get("instance_id") or "").strip()
                    exact_inspects = (
                        [
                            _safe_vast_inspect_evidence(cancel_provider.inspect(recorded_id)),
                            _safe_vast_inspect_evidence(cancel_provider.inspect(recorded_id)),
                        ]
                        if (
                            recorded_vast_instance.get("status") == "recorded"
                            and recorded_id == cancel_id
                        )
                        else []
                    )
                    exact_contract_zero = bool(
                        len(exact_inspects) == 2
                        and all(
                            _vast_instance_absence_proven(row, recorded_id)
                            for row in exact_inspects
                        )
                    )
                    recorded_vast_verification = {
                        "status": "absent" if exact_contract_zero else "unverified",
                        "instance_id": recorded_id or None,
                        "provider_absence_confirmed": exact_contract_zero,
                        "inspect_attempts": exact_inspects,
                        "provider_mutations_performed": 0,
                    }
            except Exception:  # noqa: BLE001 - hard-deadline protection remains armed
                first_zero = {}
                first_global_zero = {}
                second_zero = {}
                second_global_zero = {}
                exact_contract_zero = False
            lane_prefix_zero = all(
                inventory.get("api_confirmed") is True
                and inventory.get("live_resource_count") == 0
                for inventory in (first_zero, second_zero)
            )
            global_inventory_admitted = all(
                _inventory_contains_only_allowed_resources(
                    inventory,
                    allowed_instance_ids=allowed_ids,
                )
                for inventory in (first_global_zero, second_global_zero)
            )
            # Vast exposes a stable instance id and an exact inspect endpoint.
            # Double absence of that id plus double-zero for this lane's unique
            # prefix proves lane-owned provider-zero even when an independent
            # provider lane starts after this watchdog was armed.  Other
            # providers retain the stricter global-zero/allowlist requirement
            # because they do not have this exact-id contract here.
            independently_zero = bool(
                lane_prefix_zero
                and exact_contract_zero
                and (
                    resolved_provider == "vast"
                    or global_inventory_admitted
                )
            )
            if independently_zero:
                owner_teardown_cancel = cancel_candidate
                cancel_zero_result = {
                    **armed,
                    "status": "provider_terminal",
                    "completed_at": utc_now_iso(),
                    "initial_inventory": first_zero,
                    "initial_global_inventory": first_global_zero,
                    "terminations": [],
                    "final_inventory": second_zero,
                    "final_global_inventory": second_global_zero,
                    "provider_absence_confirmed": True,
                    "provider_mutations_performed": 0,
                    "teardown_error_type": None,
                    "raw_secret_values_recorded": False,
                }
                if resolved_provider == "vast":
                    cancel_zero_result["recorded_vast_instance"] = recorded_vast_instance
                    cancel_zero_result["recorded_vast_instance_teardown"] = (
                        recorded_vast_verification
                    )
                    cancel_zero_result["provider_absence_scope"] = (
                        "recorded_instance_and_lane_prefix"
                    )
                    cancel_zero_result["global_inventory_informational_only"] = True
                break
        sleeper(min(10.0, max(0.0, deadline_epoch - clock())))
    if cancel_zero_result:
        result = cancel_zero_result
    else:
        try:
            provider = provider_factory(resolved_provider)
        except Exception as exc:  # noqa: BLE001 - evidence survives init failure
            result = {
                **dict(armed),
                "status": "teardown_unverified",
                "completed_at": utc_now_iso(),
                "provider_absence_confirmed": False,
                "provider_mutations_performed": 0,
                "teardown_error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
        else:
            result = terminate_canary_resources(
                provider=provider,
                pod_name_prefix=pod_name_prefix,
                armed=armed,
                provider_name=resolved_provider,
            )
    result["owner_teardown_cancel_requested"] = bool(owner_teardown_cancel)
    result["owner_teardown_cancel_request_valid"] = bool(owner_teardown_cancel)
    if owner_teardown_cancel:
        result["provider_mutation_trigger"] = "owner_teardown_cancel_request_after_provider_zero"
    receipt_path = root / "provider_lane_handoff_receipt.json"
    receipt: dict[str, Any] = {}
    receipt_control_required = receipt_path.exists() or receipt_path.is_symlink()
    receipt_safe = False
    if receipt_control_required:
        try:
            receipt_stat = receipt_path.lstat()
            receipt_safe = bool(
                receipt_path.is_file()
                and not receipt_path.is_symlink()
                and not receipt_stat.st_mode & 0o077
            )
        except OSError:
            receipt_safe = False
    if receipt_safe and result.get("provider_absence_confirmed") is True:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            receipt = {}
        if isinstance(receipt, Mapping) and receipt.get("pod_name_prefix") == pod_name_prefix:
            from .paid_lane_guard import (
                cancel_pending_teardown,
                close_pending_teardown,
            )
            from .paid_provider_lane_lease import (
                build_paid_provider_lane_reconciliation,
                release_transferred_paid_provider_lane_lease,
                restore_paid_provider_lane_lease_to_retained_watchdog,
            )

            pending_path = str(receipt.get("pod_pending_teardown_record") or "")
            try:
                pending_record = json.loads(Path(pending_path).read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                pending_record = {}
            campaign_kind = str(receipt.get("campaign_kind") or "")
            expected_pending_lane = CAMPAIGN_PENDING_TEARDOWN_LANES.get(
                campaign_kind,
                "groot_oscar_gpu_canary" if not campaign_kind else "",
            )
            declared_paid_lane = str(receipt.get("paid_lane") or "")
            if declared_paid_lane and declared_paid_lane != expected_pending_lane:
                expected_pending_lane = ""
            pending_valid = bool(
                isinstance(pending_record, Mapping)
                and pending_record.get("status") == "open"
                and pending_record.get("provider") == resolved_provider
                and expected_pending_lane
                and pending_record.get("lane") == expected_pending_lane
                and pending_record.get("resource_kind") == "compute_instance"
                and str(pending_record.get("resource_name") or "").startswith(pod_name_prefix)
            )
            receipt_pod_id = str(receipt.get("pod_id") or "")
            pending_pod_id = str(pending_record.get("instance_id") or "")
            if receipt_pod_id and pending_pod_id and receipt_pod_id != pending_pod_id:
                pending_valid = False
            effective_pod_id = receipt_pod_id or pending_pod_id
            pre_provider_absent = receipt.get("pre_provider_mutation_confirmed_absent") is True
            if pre_provider_absent and not pending_path:
                pending_close = {"status": "cancelled_no_allocation"}
            elif pending_valid and effective_pod_id:
                pending_close = close_pending_teardown(
                    pending_path,
                    {
                        "status": "PASS",
                        "provider_absence_confirmed": True,
                        "instance_id": effective_pod_id,
                    },
                )
            elif pending_valid:
                pending_close = cancel_pending_teardown(
                    pending_path,
                    reason="canary_watchdog_provider_inventory_verified_zero",
                    evidence={"provider_absence_confirmed": True},
                )
            else:
                pending_close = {"status": "invalid"}
            result["pod_pending_teardown_close"] = pending_close
            if receipt.get("provider_lane_release_mode") == "watchdog_direct_compute":
                terminal_reconciliation = build_paid_provider_lane_reconciliation(
                    provider=resolved_provider,
                    lane=expected_pending_lane,
                    provider_inventory=(
                        result.get("final_inventory")
                        if isinstance(result.get("final_inventory"), Mapping)
                        else {}
                    ),
                    open_pending_teardowns=load_pending_teardowns(),
                )
                result["provider_lane_terminal_release"] = (
                    release_transferred_paid_provider_lane_lease(
                        lease_path_value=str(receipt.get("lease_path") or ""),
                        teardown_owner_pid=os.getpid(),
                        terminal_reconciliation=terminal_reconciliation,
                        reason="gpu_canary_watchdog_provider_and_pending_terminal",
                    )
                )
            else:
                result["provider_lane_owner_return"] = (
                    restore_paid_provider_lane_lease_to_retained_watchdog(receipt)
                )
    if receipt_control_required:
        control_terminal = bool(
            receipt_safe
            and result.get("pod_pending_teardown_close", {}).get("status")
            in {"closed", "cancelled_no_allocation"}
            and (
                result.get("provider_lane_owner_return", {}).get("status") == "restored"
                or result.get("provider_lane_terminal_release", {}).get("status") == "released"
            )
        )
        if not control_terminal:
            if result.get("provider_absence_confirmed") is True:
                result["status"] = "provider_terminal_control_plane_open"
            result["control_plane_terminal"] = False
        else:
            result["control_plane_terminal"] = True
    budget_context = receipt.get("campaign_budget") if isinstance(receipt, Mapping) else None
    if (
        isinstance(budget_context, Mapping)
        and budget_context.get("status") == "reserved"
        and result.get("provider_absence_confirmed") is True
        and result.get("control_plane_terminal") is True
    ):
        try:
            from .production_gpu_campaign_budget import ProductionGpuCampaignBudget

            identity = budget_context["identity"]
            reservation = budget_context["reservation"]
            elapsed = max(
                0,
                math.ceil(clock() - float(budget_context["reserved_at_epoch"])),
            )
            reserved_seconds = int(reservation["reserved_gpu_seconds"])
            if elapsed > reserved_seconds:
                result["campaign_budget_settlement"] = {
                    "status": "retained_open_budget_breach",
                    "elapsed_gpu_seconds": elapsed,
                    "reserved_gpu_seconds": reserved_seconds,
                }
                result["status"] = "provider_terminal_budget_reservation_exceeded"
                write_json(root / EVIDENCE_NAME, result)
                return result
            charged_seconds = elapsed
            charged_usd = round(
                min(
                    float(reservation["reserved_usd"]),
                    float(reservation["max_hourly_rate_usd"]) * charged_seconds / 3600.0,
                ),
                6,
            )
            budget = ProductionGpuCampaignBudget(
                budget_context["ledger_path"],
                initial_spent_usd=identity["initial_spent_usd"],
                initial_used_gpu_seconds=identity["initial_used_gpu_seconds"],
                total_spend_cap_usd=identity["total_spend_cap_usd"],
                combined_gpu_wall_cap_seconds=identity["combined_gpu_wall_cap_seconds"],
            )
            result["campaign_budget_settlement"] = budget.settle(
                reservation_id=budget_context["reservation_id"],
                charged_gpu_seconds=charged_seconds,
                charged_usd=charged_usd,
                outcome="canary_watchdog_provider_and_control_plane_terminal",
            )
        except (KeyError, TypeError, ValueError) as exc:
            result["campaign_budget_settlement"] = {
                "status": "retained_open",
                "error_type": type(exc).__name__,
            }
            result["status"] = "provider_terminal_budget_settlement_unverified"
    write_json(root / EVIDENCE_NAME, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pod-name-prefix", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="runpod")
    parser.add_argument(
        "--allowed-active-instance-id",
        action="append",
        type=int,
        default=[],
    )
    args = parser.parse_args(argv)
    result = run_watchdog(
        out_dir=args.out_dir,
        pod_name_prefix=args.pod_name_prefix,
        deadline_epoch=args.deadline_epoch,
        provider_name=args.provider,
        allowed_active_instance_ids=args.allowed_active_instance_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "provider_terminal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
