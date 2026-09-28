"""Reserved experiments cannot enter unsupported persistent consumer families."""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from .control_plane_lane_owner_target_versions import OwnerTargetVersionError

_RESERVED = re.compile(r"(?:^|[/\\])g1[/\\]registered-[0-9a-f]{32}(?:$|[/\\])")


def refuse_registered_references(*values):
    """Inspect decoded string leaves before the existing publisher touches disk.

    Matching names are reserved independently of marker presence. This grants
    no ownership and no inventory completeness; unsupported families refuse.
    """
    pending = [(value, 0) for value in values]
    visited, text_bytes = 0, 0
    while pending:
        value, depth = pending.pop()
        visited += 1
        if visited > 100000 or depth > 64:
            raise OwnerTargetVersionError("experiment_publisher_input_limit")
        if isinstance(value, (str, Path)):
            text = str(value)
            text_bytes += len(text.encode("utf-8"))
            if text_bytes > 2 * 1024 * 1024:
                raise OwnerTargetVersionError("experiment_publisher_input_limit")
            if _RESERVED.search(text):
                raise OwnerTargetVersionError("experiment_external_publisher_unsupported")
        elif isinstance(value, Mapping):
            if len(value) > 100000 - visited:
                raise OwnerTargetVersionError("experiment_publisher_input_limit")
            pending.extend((item, depth + 1) for pair in value.items() for item in pair)
        elif isinstance(value, (list, tuple)):
            if len(value) > 100000 - visited:
                raise OwnerTargetVersionError("experiment_publisher_input_limit")
            pending.extend((item, depth + 1) for item in value)
