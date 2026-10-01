"""Existing provider artifact size/disk contracts without artifact publication imports."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .task_evaluation_scene_configuration_transfer_budget import (
    scene_configuration_provider_transfer_byte_budget,
)

ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES = 2_209_255_046


PROVISIONING_DOWNLOAD_OVERHEAD_BYTES = 10_000_000_000


PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES = 1_000_000_000


PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER = 2


PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO = 4


PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT = 10_000


PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES = 512 * 1024 * 1024


class TaskEvaluationSceneConfigurationVastError(RuntimeError):
    """The canonical scene-configuration Vast lane refused an unsafe input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _provider_output_disk_requirements(
    maximum_archive_bytes: int,
) -> dict[str, int]:
    """Return the one capacity formula used before transfer and extraction."""

    if (
        isinstance(maximum_archive_bytes, bool)
        or not isinstance(maximum_archive_bytes, int)
        or maximum_archive_bytes <= 0
        or PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO < 1
        or PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES <= 0
    ):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_provider_output_disk_requirement_invalid"
        )
    maximum_expanded_bytes = (
        maximum_archive_bytes * PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO
    )
    return {
        "maximum_archive_bytes": maximum_archive_bytes,
        "maximum_expanded_bytes": maximum_expanded_bytes,
        "operational_reserve_bytes": PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES,
        "required_free_bytes_before_download": (
            maximum_archive_bytes
            + maximum_expanded_bytes
            + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
        "required_free_bytes_before_extraction": (
            maximum_expanded_bytes + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
        ),
    }


def _provider_transfer_byte_budget(
    receipt: Mapping[str, Any],
) -> tuple[int, int]:
    return scene_configuration_provider_transfer_byte_budget(
        receipt,
        provisioning_download_overhead_bytes=PROVISIONING_DOWNLOAD_OVERHEAD_BYTES,
        artifixer_pinned_wheel_download_floor_bytes=(
            ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES
        ),
        provider_output_upload_minimum_bytes=PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES,
        provider_output_upload_bundle_multiplier=(
            PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER
        ),
        error_factory=TaskEvaluationSceneConfigurationVastError,
    )
