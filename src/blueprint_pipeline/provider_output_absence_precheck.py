"""Read remote output absence before terminal startup cleanup can delete inputs."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def output_absence_precheck(*, keys: Sequence[str], blockers: Sequence[str],
                           make_client: Callable[[], Any],
                           inspect_absence: Callable[[Any, str], Mapping[str, Any]],
                           transfer_error: Callable[[Exception], Mapping[str, Any]],
                           schema_version: str, generated_at: str) -> dict[str, Any]:
    """No deletion; a present, invalid or unreadable output preserves recovery."""
    checks: list[dict[str, Any]] = []
    if not blockers:
        try:
            client = make_client()
            checks = [{"key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                       "absence": dict(inspect_absence(client, key))} for key in keys]
        except Exception as exc:  # noqa: BLE001 - unknown outcomes retain every object
            checks = [{"absence": {"absence_confirmed": False, **transfer_error(exc)}}]
    if blockers or not checks or any(row["absence"].get("absence_confirmed") is not True for row in checks):
        return {"schema_version": schema_version, "generated_at": generated_at, "status": "blocked",
                "blockers": sorted(set([*blockers, "provider_output_uncollected_objects_retained"])),
                "all_objects_absent": False, "all_ephemeral_objects_absent": False,
                "output_absence_precheck": checks, "cleanup_attempts": 0,
                "remote_objects_retained_for_recovery": True, "signed_url_files_preserved": True,
                "raw_secret_values_recorded": False}
    return {"status": "passed", "output_absence_precheck": checks}
