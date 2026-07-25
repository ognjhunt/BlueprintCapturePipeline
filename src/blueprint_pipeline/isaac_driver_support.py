"""Reject NVIDIA drivers the pinned Isaac RTX runtime cannot render on.

The existing driver check was a *blocklist* of one known-bad range (570.0.0 to
570.158.1) plus a sort that ranked ``major >= 580`` as the most preferred
branch. That encodes "newer is better", which is false for a pinned Omniverse
runtime: Isaac ships its own Vulkan/RTX stack, and a driver newer than the one
it was built against fails to create a device at all.

Attempt 069 paid for that assumption. The selector drew an L40S on driver
595.71.05 -- outside the known-bad range and ``major >= 580``, so it sorted
*first* -- and Isaac died before the episode began::

    [Error] [omni.rtx] VkResult: ERROR_INCOMPATIBLE_DRIVER
    [Error] [omni.rtx] vkCreateInstance failed. Vulkan 1.1 is not supported
    [Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU
        devices, including an attempt with compatibility mode.
    {"status": "blocked", "blockers": ["review_canary_timeout_renderer_never_ready"]}

Because 595 hosts are also cheaper than 580 hosts ($0.471/hr vs ~$0.688/hr),
cost broke the tie toward the broken driver on every retry -- the failure was
deterministic, not unlucky.

Every host that has ever rendered successfully ran the 580 branch: 580.119.02
(attempt 067, the first door contact), 580.159.03, and 580.159.04. This module
turns that evidence into an admission ceiling, the same shape as the TensorRT
compute-cap ceiling: a defect class that cannot be fixed at runtime is made
unrentable instead of merely diagnosed.

The ceiling is exclusive and deliberately conservative. Branches between 581
and 594 are untested, not known-good, and a wrong admission costs a full
allocate-plus-boot-plus-episode cycle while a wrong rejection costs only a
different offer from a large pool. Set ``BLUEPRINT_VAST_MAX_ISAAC_DRIVER`` to
raise it (e.g. after qualifying a newer branch) or to ``0`` to disable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

# Highest driver branch proven to render, plus one: the 580 branch is known
# good, 595 is known broken, everything between is unqualified.
ISAAC_MAX_SUPPORTED_DRIVER_EXCLUSIVE: tuple[int, int, int] = (581, 0, 0)
MAX_ISAAC_DRIVER_ENV = "BLUEPRINT_VAST_MAX_ISAAC_DRIVER"

ABOVE_CEILING_STATUS = "above_supported_omniverse_rtx_driver_ceiling"

DRIVER_VERSION_KEYS = (
    "driver_version",
    "driverVersion",
    "cuda_driver_version",
    "nvidia_driver_version",
    "driver",
)


def driver_version_tuple(value: Any) -> tuple[int, ...] | None:
    """Parse a dotted driver version into a comparable tuple."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts: list[int] = []
    for chunk in text.split("."):
        # Vendor builds append suffixes ("550.90.07-ubuntu"); take the leading
        # digits so the numeric component still classifies correctly.
        digits = ""
        for character in chunk.strip():
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return None
    # Pad to a fixed width so abbreviated spellings compare identically: without
    # this, "581" parses to (581,) which sorts BELOW the exclusive ceiling
    # (581, 0, 0) and would admit the very branch the ceiling rejects.
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def offer_driver_version(offer: Mapping[str, Any]) -> str:
    for key in DRIVER_VERSION_KEYS:
        value = offer.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def resolve_max_isaac_driver(
    explicit: Any = None,
    *,
    default: tuple[int, int, int] = ISAAC_MAX_SUPPORTED_DRIVER_EXCLUSIVE,
) -> tuple[int, ...] | None:
    """Resolve the ceiling, honouring an explicit value then the environment.

    Returns ``None`` when the ceiling is disabled, so callers admit everything.
    """

    for candidate in (explicit, os.getenv(MAX_ISAAC_DRIVER_ENV)):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        if text in {"0", "none", "off", "disabled"}:
            return None
        parsed = driver_version_tuple(text)
        if parsed is not None:
            return parsed
    return default


def meets_max_isaac_driver(offer: Mapping[str, Any], max_driver: tuple[int, ...] | None) -> bool:
    """True when the offer's driver is below the ceiling.

    Permissive when the ceiling is disabled or the driver is unreported: an
    absent version is not evidence of an incompatible one, and the live
    renderer gate still fails closed on the box itself.
    """

    if max_driver is None:
        return True
    version = driver_version_tuple(offer_driver_version(offer))
    if version is None:
        return True
    return version < max_driver


def driver_excluded_count(
    summaries: Sequence[Mapping[str, Any]], max_driver: tuple[int, ...] | None
) -> int:
    """How many candidate offers the ceiling removed, for admission evidence."""

    if max_driver is None:
        return 0
    return sum(1 for item in summaries if not meets_max_isaac_driver(item, max_driver))


def any_offer_exceeds_driver_ceiling(
    summaries: Sequence[Mapping[str, Any]], max_driver: tuple[int, ...] | None
) -> bool:
    return driver_excluded_count(summaries, max_driver) > 0


def format_driver(version: tuple[int, ...] | None) -> str:
    return ".".join(str(part) for part in version) if version else "disabled"


# The original known-bad window: drivers too OLD for the Omniverse RTX stack.
# Retained verbatim so the ceiling added above complements it rather than
# replacing it -- Isaac has both a floor and a ceiling, and 069 proved only the
# floor was ever encoded.
ISAAC_KNOWN_UNSUPPORTED_DRIVER_FLOOR: tuple[int, int, int] = (570, 0, 0)
ISAAC_KNOWN_UNSUPPORTED_DRIVER_CEILING_EXCLUSIVE: tuple[int, int, int] = (570, 158, 1)

SUPPORTED_STATUS = "outside_known_unsupported_omniverse_rtx_driver_range"
KNOWN_UNSUPPORTED_STATUS = "known_unsupported_omniverse_rtx_driver_range"
UNKNOWN_STATUS = "unknown_driver_version"


_UNSET = object()


def isaac_driver_support_status(driver_version: Any, *, max_driver: Any = _UNSET) -> str:
    """Classify a driver, resolving the ceiling from the environment by default.

    Resolving here rather than binding an import-time constant is what makes
    ``BLUEPRINT_VAST_MAX_ISAAC_DRIVER`` effective in live selection: the status
    this returns is the value the Isaac admission filter keys on.
    """

    ceiling = resolve_max_isaac_driver() if max_driver is _UNSET else max_driver
    version = driver_version_tuple(driver_version)
    if version is None:
        return UNKNOWN_STATUS
    if (
        ISAAC_KNOWN_UNSUPPORTED_DRIVER_FLOOR
        <= version
        < ISAAC_KNOWN_UNSUPPORTED_DRIVER_CEILING_EXCLUSIVE
    ):
        return KNOWN_UNSUPPORTED_STATUS
    if ceiling is not None and version >= ceiling:
        return ABOVE_CEILING_STATUS
    return SUPPORTED_STATUS


def driver_sort_rank(summary: Mapping[str, Any]) -> int:
    status = str(summary.get("isaac_driver_support_status") or "")
    if status == SUPPORTED_STATUS:
        return 0
    if status == UNKNOWN_STATUS:
        return 1
    if status == KNOWN_UNSUPPORTED_STATUS:
        return 2
    # Above-ceiling sorts last, after the unrecognized-status fallthrough: such
    # offers are filtered out before selection, so this only orders diagnostics.
    if status == ABOVE_CEILING_STATUS:
        return 4
    return 3


def driver_newer_branch_sort_rank(summary: Mapping[str, Any]) -> int:
    """Prefer the newest branch that is still at or below the ceiling.

    Ranking ``major >= 580`` best was correct only while 580 was the newest
    good branch; it silently promoted 595 the moment such hosts appeared.
    """

    version = driver_version_tuple(summary.get("driver_version"))
    if version is None:
        return 4
    ceiling = resolve_max_isaac_driver()
    if ceiling is not None and version >= ceiling:
        return 5
    major = version[0]
    if major >= 580:
        return 0
    if major == 570 and version >= ISAAC_KNOWN_UNSUPPORTED_DRIVER_CEILING_EXCLUSIVE:
        return 1
    if major >= 575:
        return 2
    return 3
