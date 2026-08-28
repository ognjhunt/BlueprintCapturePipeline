"""Provider artifact publication, transfer budgeting, and disk-capacity guards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .provider_output_disk_capacity import observe_provider_output_disk_capacity
from .task_evaluation_configured_scene_object_store import (
    publish_configured_scene_artifact,
)
from .task_evaluation_scene_artifact_retention import (
    seal_scene_artifact_remote_index,
)
from .task_evaluation_scene_configuration_transfer_budget import (
    scene_configuration_provider_transfer_byte_budget,
)


#: The bundle is not all this lane pulls. Before the first stage the onstart
#: apt-installs its build and render toolchain, and the ArtiFixer runtime then
#: fetches ``uv`` and builds a venv. Declaring only ``bundle_size_bytes``
#: left all of that outside the hard-cap projection.
#:
#: The production ``dual_target_artifixer3d_only`` path skips the VIBE editor,
#: but it does *not* skip Torch: ``run_public_scene_artifixer3d.sh`` installs
#: the CUDA 12.8 build before importing the 3DGRUT JIT graph. The exact Torch
#: wheel plus only five mandatory Linux wheels already total this many bytes.
ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES = 2_209_255_046

#: Conservative pre-allocation ceiling for all non-bundle downloads above.
#: Keep this fail-closed: a tighter value needs a complete publisher-size
#: inventory, not an assumption that one executed install branch is skipped.
PROVISIONING_DOWNLOAD_OVERHEAD_BYTES = 10_000_000_000

#: The result ZIP is bounded before its signed PUT is used. The floor leaves
#: room for generated evidence when a test bundle is tiny; the receipt-relative
#: term scales for production bundles with scene, renderer, and native pieces.
PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES = 1_000_000_000
PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER = 2
PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO = 4
PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT = 10_000
#: Space retained beyond the declared archive and maximum expansion for
#: filesystem metadata, terminal manifests, and publication temporaries.
PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES = 512 * 1024 * 1024


class TaskEvaluationSceneConfigurationVastError(RuntimeError):
    """The canonical scene-configuration Vast lane refused an unsafe input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_vast_json_invalid"
        ) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_vast_json_invalid"
        )
    return dict(value)


def _publish_provider_output_archive(
    *,
    output_zip: Path,
    job: Path,
    receipt: Mapping[str, Any],
    publisher: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Make the exact provider ZIP durable before local retention may reap it."""

    publish = publisher or publish_configured_scene_artifact
    reference = publish(path=output_zip, artifact_kind="provider-output")
    index_path = job / "scene_artifact_remote_index.v1.json"
    index = seal_scene_artifact_remote_index(
        destination=index_path,
        run_id=str(receipt["run_id"]),
        source_commit=str(receipt["source_commit"]),
        bundle_digest=str(receipt["bundle_sha256"]),
        artifact_references=[reference],
    )
    return reference, index, index_path


def _seal_provider_bundle_remote_index(
    *,
    job: Path,
    receipt: Mapping[str, Any],
    staging: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Bind the staged CAS bundle to a small immutable local index."""

    raw_reference = staging.get("provider_bundle_remote_reference")
    reference = dict(raw_reference) if isinstance(raw_reference, Mapping) else {}
    if not reference:
        # Compatibility for an older staging implementation during a rolling
        # deploy: still require durable CAS publication before allocation.
        reference = publish_configured_scene_artifact(
            path=Path(str(receipt["bundle_path"])),
            artifact_kind="provider-bundle",
        )
    index_path = job / "scene_provider_bundle_remote_index.v1.json"
    index = seal_scene_artifact_remote_index(
        destination=index_path,
        run_id=str(receipt["run_id"]),
        source_commit=str(receipt["source_commit"]),
        bundle_digest=str(receipt["bundle_sha256"]),
        artifact_references=[reference],
    )
    return reference, index, index_path


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


def _provider_output_disk_capacity(
    *,
    destination_directory: Path,
    required_free_bytes: int,
    phase: str,
    disk_usage_provider: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Measure the destination filesystem and fail closed on unknown capacity."""

    return observe_provider_output_disk_capacity(
        destination_directory=destination_directory,
        required_free_bytes=required_free_bytes,
        phase=phase,
        schema_version="scene_configuration_provider_output_disk_capacity.v1",
        blocker_prefix="scene_configuration_provider_output",
        disk_usage_provider=disk_usage_provider,
    )


def extract_provider_output_with_capacity_guard(
    archive_path: Path,
    destination: Path,
    *,
    maximum_archive_bytes: int,
    extractor: Callable[..., tuple[dict[str, Any], list[str]]],
    diagnostic_only: bool = False,
    disk_usage_provider: Callable[[Path], Any] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Re-stat free bytes before invoking the caller's unchanged extractor."""

    requirements = _provider_output_disk_requirements(maximum_archive_bytes)
    capacity = _provider_output_disk_capacity(
        destination_directory=destination.parent,
        required_free_bytes=requirements[
            "required_free_bytes_before_extraction"
        ],
        phase="before_extraction",
        disk_usage_provider=disk_usage_provider,
    )
    if capacity["status"] != "ready":
        return {}, list(capacity["blockers"]), capacity
    result, blockers = extractor(
        archive_path,
        destination,
        maximum_archive_bytes=maximum_archive_bytes,
        diagnostic_only=diagnostic_only,
    )
    return result, blockers, capacity


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


__all__ = [
    "ARTIFIXER_PINNED_WHEEL_DOWNLOAD_FLOOR_BYTES",
    "PROVIDER_OUTPUT_MAXIMUM_EXPANSION_RATIO",
    "PROVIDER_OUTPUT_MAXIMUM_MEMBER_COUNT",
    "PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES",
    "PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER",
    "PROVIDER_OUTPUT_UPLOAD_MINIMUM_BYTES",
    "PROVISIONING_DOWNLOAD_OVERHEAD_BYTES",
    "TaskEvaluationSceneConfigurationVastError",
    "_provider_output_disk_capacity",
    "_provider_output_disk_requirements",
    "_provider_transfer_byte_budget",
    "_publish_provider_output_archive",
    "_read",
    "_seal_provider_bundle_remote_index",
    "_sha256",
    "extract_provider_output_with_capacity_guard",
]
