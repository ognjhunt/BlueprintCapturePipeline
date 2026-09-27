"""Read a caller-surviving watchdog's later terminal receipt without changing its snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def late_watchdog_instance(
    *,
    result_path: Path,
    result: Mapping[str, Any],
    read_json: Callable[[Path, str], Mapping[str, Any]],
    error_factory: Callable[[str], Exception],
) -> tuple[int, Path] | None:
    snapshot = result.get("independent_watchdog")
    if not isinstance(snapshot, Mapping) or snapshot.get("status") != "retained_until_hard_ttl":
        return None
    ids = snapshot.get("instance_ids")
    if (not isinstance(ids, list) or len(ids) != 1 or isinstance(ids[0], bool)
            or not isinstance(ids[0], int) or ids[0] <= 0):
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    instance_id = ids[0]
    adapter_path = Path(str(result.get("adapter_result_path") or ""))
    if not adapter_path.is_absolute() or adapter_path.name != "vast_provider_adapter_result.json":
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    attempt_root = adapter_path.parent.parent
    if adapter_path.parent.name != "vast_provider_run" or attempt_root.parent.name != "attempts":
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    if not attempt_root.is_relative_to(result_path.parent):
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    terminal_path = (
        attempt_root / "independent_vast_watchdog" / "groot_oscar_runpod_canary_watchdog.json"
    )
    if (snapshot.get("watchdog_evidence_path") != str(terminal_path)
            or snapshot.get("watchdog_out_dir") != str(terminal_path.parent)
            or terminal_path.is_symlink()):
        raise error_factory("policy_canary_late_watchdog_identity_invalid")
    if not terminal_path.is_file():
        return None
    terminal = read_json(terminal_path, "policy_canary_late_watchdog_terminal_invalid")
    recorded = terminal.get("recorded_vast_instance")
    teardown = terminal.get("recorded_vast_instance_teardown")
    if (
        terminal.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or terminal.get("status") != "provider_terminal"
        or terminal.get("provider_absence_confirmed") is not True
        or terminal.get("provider_absence_scope") != "recorded_instance_and_lane_prefix"
        or terminal.get("raw_secret_values_recorded") is not False
        or terminal.get("pod_name_prefix") != snapshot.get("pod_name_prefix")
        or terminal.get("resource_name_exact") != snapshot.get("resource_name_exact")
        or terminal.get("watchdog_out_dir") != str(terminal_path.parent)
        or not isinstance(recorded, Mapping)
        or recorded.get("instance_id") != str(instance_id)
        or recorded.get("scope_confirmed") is not True
        or recorded.get("pod_name_prefix") != snapshot.get("pod_name_prefix")
        or not isinstance(teardown, Mapping)
        or teardown.get("instance_id") != str(instance_id)
        or teardown.get("status") != "absent"
        or teardown.get("provider_absence_confirmed") is not True
    ):
        raise error_factory("policy_canary_late_watchdog_terminal_invalid")
    return instance_id, terminal_path


__all__ = ["late_watchdog_instance"]
