"""A bound artifact reference (``path``, ``sha256``, ``size_bytes``), checked with the standard library only.

The G1 runtime on a rented machine validates its execution spec with this
(``beta_policy_screening_g1._record``). That machine's Python has no jsonschema, which
``beta_policy_screening_plan`` needs at load through the candidate catalog; the plan module's own
``_receipt`` delegates here, so both read one rule.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

RECEIPT_KEYS = frozenset({"path", "sha256", "size_bytes"})
#: Equal to ``beta_policy_candidate_catalog.DIGEST_PATTERN`` (a test pins it); not imported, for the reason above.
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
INVALID = "beta_screening_artifact_reference_invalid"


def artifact_reference(value: Any, *, error: type[ValueError] = ValueError) -> dict[str, Any]:
    """The reference, copied, or ``error(INVALID)``: an absolute path with no ``..``, a sha256, a positive size."""
    if not isinstance(value, Mapping) or set(value) != RECEIPT_KEYS:
        raise error(INVALID)
    path = value.get("path")
    if not isinstance(path, str) or not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise error(INVALID)
    if not isinstance(value.get("sha256"), str) or not DIGEST.fullmatch(value["sha256"]):
        raise error(INVALID)
    if type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0:
        raise error(INVALID)
    return dict(value)


__all__ = ["DIGEST", "INVALID", "RECEIPT_KEYS", "artifact_reference"]
