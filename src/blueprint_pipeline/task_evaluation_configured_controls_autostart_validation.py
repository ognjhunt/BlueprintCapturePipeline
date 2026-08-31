"""Small pure validators extracted from the configured-controls autostart spine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def configuration_adoption_valid(
    *,
    adoption: Mapping[str, Any],
    source_launch_id: str,
    terminal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    revision: Mapping[str, Any],
    publication: Mapping[str, Any],
    sync: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> bool:
    return bool(
        adoption.get("mode") == "explicit_terminal_adoption"
        and adoption.get("source_launch_id") == source_launch_id
        and adoption.get("source_launch_receipt_digest")
        == receipt.get("receipt_digest")
        and adoption.get("terminal_result_digest") == terminal.get("result_digest")
        and adoption.get("configured_scene_revision_digest")
        == revision.get("revision_digest")
        and adoption.get("publication_result_digest")
        == publication.get("result_digest")
        and adoption.get("webapp_sync_result_digest")
        == sync.get("sync_result_digest")
        and adoption.get("provider_zero_receipt_digest")
        == zero.get("provider_zero_receipt_digest")
    )


def configuration_adoption_validator(
    error_type: type[Exception],
) -> Callable[..., None]:
    def require(**kwargs: Any) -> None:
        if not configuration_adoption_valid(**kwargs):
            raise error_type("configured_controls_autostart_adoption_evidence_invalid")

    return require


__all__ = ["configuration_adoption_valid", "configuration_adoption_validator"]
