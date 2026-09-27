"""How a launch activation reserves disk.

Admission reads the shared activation root's filesystem, but the footprint
sample measures only this activation's own tree.  Samples are labelled by lane,
because lanes materialize very different reference sets, and the reservation
never drops below the reference bytes the request itself declares.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .control_plane_disk_footprints import workload_name
from .task_evaluation_activation_runtime_layers import collect_request_references

# The release window, runtime layers and activation receipts written beside the
# declared references.
ACTIVATION_RESERVATION_MARGIN_BYTES = 256 * 1024**2


def activation_reservation_terms(
    request: Mapping[str, Any], activation_base: str | Path
) -> dict[str, Any]:
    """Keyword terms for the activation's ``reserve_control_plane_disk`` call."""

    declared = {
        (row["digest"], row["size_bytes"]) for row in collect_request_references(request)
    }
    return {
        "workspace": Path(activation_base) / str(request["activation_id"]),
        "workload": workload_name("activation", request.get("lane") or "unlabelled"),
        "minimum_bytes": sum(size for _digest, size in declared)
        + ACTIVATION_RESERVATION_MARGIN_BYTES,
    }


__all__ = ["ACTIVATION_RESERVATION_MARGIN_BYTES", "activation_reservation_terms"]
