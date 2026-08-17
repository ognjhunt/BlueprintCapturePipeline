"""Seal one exact-support ArtiFixer -> ArtiFixer3D -> ArtiFixer3D+ bundle.

The bundle contains only private-derived calibrated frames, masks, and a
deterministic COLMAP seed derived from the retained splat.  It deliberately
does not contain the publisher's raw dataset bytes or multi-gigabyte model
weights.  Released model artifacts are fetched on the worker at immutable
revisions and verified against this module's exact byte inventory before any
inference is permitted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from typing import Any, Mapping, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest, canonical_json
from .image_editor_backend_registry import (
    ARTIFIXER_DIRECT_CAPABILITY,
    NO_DIRECT_EDITOR as REGISTRY_NO_DIRECT_EDITOR,
    registered_backend_ids,
)
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint
from .public_scene_artifixer3d_candidate_inputs import (
    SCHEMA_VERSION as CANDIDATE_INPUT_SCHEMA,
)
from .public_scene_artifixer3d_dual_target_inputs import (
    SCHEMA_VERSION as DUAL_TARGET_INPUT_SCHEMA,
)


SCHEMA_VERSION = "public_scene_artifixer3d_bundle.v1"
RUNTIME_REQUEST_SCHEMA_VERSION = "public_scene_artifixer3d_runtime_request.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_runtime_result.v1"
ENTRYPOINT = "provider_runtime/run_public_scene_artifixer3d.sh"
RUNNER = "provider_runtime/public_scene_artifixer3d_runner.py"
RUNTIME_PACKAGE_INIT = "provider_runtime/blueprint_pipeline/__init__.py"
RUNTIME_EDITOR_REGISTRY = (
    "provider_runtime/blueprint_pipeline/image_editor_backend_registry.py"
)
RUNTIME_EDITOR_REGISTRY_MANIFEST = (
    "docs/arm_decision_proof_v1/manifests/image_editor_backends.v1.json"
)
MANIFEST = "provider_runtime/artifixer3d_bundle_manifest.json"
RUNTIME_REQUEST = "provider_runtime/artifixer3d_runtime_request.json"
INPUT_RECEIPT = "provider_runtime/input/public_scene_artifixer3d_candidate_inputs.v3.json"
DUAL_TARGET_INPUT_RECEIPT = (
    "provider_runtime/input/public_scene_artifixer3d_dual_target_inputs.v1.json"
)
FULL_PIPELINE_MODE = "full_artifixer3d_plus"
DUAL_TARGET_PIPELINE_MODE = "dual_target_artifixer3d_only"
DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE = "dual_target_artifixer3d_render_only"
CHECKPOINT_REUSE_SCHEMA_VERSION = "public_scene_artifixer3d_checkpoint_reuse.v1"
DEFAULT_REMOVAL_PIPELINE_POLICY = "candidate_schema_resolved.v1"

ARTIFIXER_REPOSITORY = "https://github.com/nv-tlabs/ArtiFixer.git"
ARTIFIXER_COMMIT = "a392c4dfe17459ef9952407accdb9fcdcdddba98"
ARTIFIXER_TREE = "f9283bfe5e3a6cc160fd418f4e66412746a19a07"
ARTIFIXER_SUBMODULE_REPOSITORY = "https://github.com/nv-tlabs/3DGRUT-ArtiFixer.git"
ARTIFIXER_SUBMODULE_COMMIT = "62e1038b74b2edc01440fd4ddf5f080109b6faba"
ARTIFIXER_SUBMODULE_TREE = "494ecc2dd0834fcf71bf0124de152940e0c6d845"
ARTIFIXER_MODEL_REPOSITORY = "nvidia/ArtiFixer"
ARTIFIXER_MODEL_REVISION = "f96352ad72c84a628d5844b6543e94ae8c4479b3"
WAN_MODEL_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
VIBE_IMAGE_EDIT_REPOSITORY = "iitolstykh/VIBE-Image-Edit"
VIBE_IMAGE_EDIT_REVISION = "d670fc5220d9806dd95e3d27508aaa66e7415c45"
VIBE_IMAGE_EDIT_LICENSE = "Apache-2.0"
VIBE_SOURCE_REPOSITORY = "https://github.com/ai-forever/VIBE.git"
VIBE_SOURCE_COMMIT = "7f0f01f9a6f66d55aa0fec2bf2562c332bba262b"
VIBE_SOURCE_TREE = "208f31e15a70de8a8b58e20acd6aba465ac1fcbc"
VIBE_SOURCE_LICENSE_SHA256 = (
    "sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
# Read from the registry rather than pinned here: the best image-editing models
# change every few months, and a frozenset in code meant adopting one required a
# release. The registry records each backend's terms alongside its name, so the
# seam stays open for the next model without opening it for a model nobody
# checked the license on. See `image_editor_backend_registry`.
DIRECT_EDITOR_BACKENDS = registered_backend_ids(capability=ARTIFIXER_DIRECT_CAPABILITY)
NO_DIRECT_EDITOR = REGISTRY_NO_DIRECT_EDITOR


def resolve_default_removal_pipeline(candidate_schema_version: str) -> dict[str, str]:
    """Resolve the removal pipeline without exposing training internals to callers."""

    if candidate_schema_version == DUAL_TARGET_INPUT_SCHEMA:
        return {
            "policy": DEFAULT_REMOVAL_PIPELINE_POLICY,
            "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
            "direct_editor_backend": NO_DIRECT_EDITOR,
        }
    if candidate_schema_version == CANDIDATE_INPUT_SCHEMA:
        return {
            "policy": DEFAULT_REMOVAL_PIPELINE_POLICY,
            "pipeline_mode": FULL_PIPELINE_MODE,
            "direct_editor_backend": "artifixer",
        }
    raise ValueError("artifixer3d_candidate_schema_unsupported")


MODEL_LICENSE_URL = (
    "https://developer.download.nvidia.com/licenses/"
    "NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf"
)
MODEL_LICENSE_SHA256 = "sha256:4ff203c3f7997c7fed287a463d733f794934a79cfabb2936008fca0bcc8ad3d6"
USE_ATTESTATION_SCHEMA_VERSION = "public_scene_artifixer3d_noncommercial_use_attestation.v1"
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/pytorch@sha256:0981807f1a51a156563e28b59dc2e7a9b5c1c7d85d1169d4965c5fd91fa38bcb"
)
ARTIFIXER_MODEL_FILES = (
    {
        "path": "artifixer-1.3b.pt",
        "size_bytes": 6_715_346_651,
        "sha256": "sha256:23e909fb4232c6a74a1c59eaf0ebfd419dd188e601aa0ab0145b9aaea821e059",
    },
)
WAN_RUNTIME_FILES = (
    {
        "path": "scheduler/scheduler_config.json",
        "size_bytes": 751,
        "sha256": "sha256:3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e",
    },
    {
        "path": "transformer/config.json",
        "size_bytes": 465,
        "sha256": "sha256:0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81",
    },
    {
        "path": "vae/config.json",
        "size_bytes": 724,
        "sha256": "sha256:f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527",
    },
    {
        "path": "vae/diffusion_pytorch_model.safetensors",
        "size_bytes": 507_591_892,
        "sha256": "sha256:d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
    },
)
VIBE_IMAGE_EDIT_LARGE_FILES = (
    {
        "path": "text_encoder/model.safetensors",
        "size_bytes": 4_255_140_312,
        "sha256": "sha256:7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00001-of-00001.safetensors",
        "size_bytes": 3_765_399_232,
        "sha256": "sha256:ac81406ec9f673a120695f83db41745f1b7b0e8a0b1f45a8de66aa420fc52c54",
    },
    {
        "path": "vae/diffusion_pytorch_model.safetensors",
        "size_bytes": 1_249_044_836,
        "sha256": "sha256:dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb",
    },
)
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_MEMBER_COUNT = 20_000


class ArtiFixer3DBundleError(ValueError):
    """Stable fail-closed bundle materialization errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtiFixer3DBundleError([code]) from exc
    if not isinstance(value, dict):
        raise ArtiFixer3DBundleError([code])
    return value


def _file(value: str | Path, *, code: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ArtiFixer3DBundleError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise ArtiFixer3DBundleError([code])
    return path


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    key = "relative_path" if root is not None else "path"
    value = path.relative_to(root).as_posix() if root is not None else str(path)
    return {
        key: value,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _portable(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_member_path_invalid"])
    return path


def _candidate_receipt(path: Path) -> dict[str, Any]:
    value = _read(path, code="artifixer3d_bundle_candidate_receipt_unreadable")
    tasks = value.get("tasks")
    if value.get("schema_version") == DUAL_TARGET_INPUT_SCHEMA:
        if (
            value.get("status") != "paired_target_inputs_prepared_no_model_no_execution"
            or value.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
            or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")
            or not isinstance(tasks, list)
            or not 1 <= len(tasks) <= 5
            or value.get("replacement_object_count") != len(tasks)
            or value.get("execution", {}).get("provider_mutations_performed") != 0
            or value.get("claim_boundary", {}).get("semantic_teacher_object_absence_reviewed")
            is not False
            or value.get("claim_boundary", {}).get("appearance_repair_qualified") is not False
        ):
            raise ArtiFixer3DBundleError(
                ["artifixer3d_bundle_dual_target_candidate_receipt_invalid"]
            )
        transition = value.get("transition_support")
        if (
            not isinstance(transition, Mapping)
            or not isinstance(transition.get("radius_pixels"), int)
            or isinstance(transition.get("radius_pixels"), bool)
            or transition["radius_pixels"] < 0
            or transition.get("anchor_loss_mask_outside_support") != 255
            or transition.get("anchor_loss_mask_inside_support") != 0
            or transition.get("semantic_teacher_loss_mask") is not None
        ):
            raise ArtiFixer3DBundleError(["artifixer3d_bundle_dual_target_transition_invalid"])
        for task in tasks:
            physical = task.get("physical_camera_count") if isinstance(task, Mapping) else None
            training = task.get("training_record_count") if isinstance(task, Mapping) else None
            frames = task.get("frames") if isinstance(task, Mapping) else None
            anchors = task.get("selected_anchor_indices") if isinstance(task, Mapping) else None
            teachers = task.get("semantic_teacher_indices") if isinstance(task, Mapping) else None
            if (
                not isinstance(task, Mapping)
                or not isinstance(physical, int)
                or isinstance(physical, bool)
                or physical <= 0
                or training != 2 * physical
                or not isinstance(frames, list)
                or len(frames) != physical
                or anchors != list(range(0, training, 2))
                or teachers != list(range(1, training, 2))
                or set(anchors) & set(teachers)
                or task.get("loss_contract", {}).get("same_pose_and_intrinsics_per_pair")
                is not True
                or task.get("loss_contract", {}).get("direct_artifixer_required") is not False
                or task.get("loss_contract", {}).get("artifixer3d_plus_required_for_this_packet")
                is not False
            ):
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_bundle_dual_target_candidate_task_invalid"]
                )
            for index, frame in enumerate(frames):
                if (
                    not isinstance(frame, Mapping)
                    or frame.get("physical_camera_index") != index
                    or frame.get("anchor_training_index") != 2 * index
                    or frame.get("semantic_teacher_training_index") != 2 * index + 1
                    or frame.get("teacher_loss_mask_materialized") is not False
                    or frame.get("pair_pose_and_intrinsics_exactly_equal") is not True
                    or not isinstance(frame.get("anchor_loss_mask"), Mapping)
                    or not isinstance(frame.get("semantic_teacher_rgb"), Mapping)
                ):
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_bundle_dual_target_candidate_frame_invalid"]
                    )
        return value
    semantics = value.get("repair_target_semantics")
    if (
        value.get("schema_version") != CANDIDATE_INPUT_SCHEMA
        or value.get("status") != "candidate_inputs_prepared_no_model_no_execution"
        or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= 5
        or value.get("replacement_object_count") != len(tasks)
        or not isinstance(semantics, Mapping)
        or semantics.get("source_washer_or_notebook_restoration_permitted") is not False
        or semantics.get("black_unknown_placeholder_preservation_permitted") is not False
        or semantics.get("outside_exact_support_changed_pixels_permitted") != 0
        or value.get("execution", {}).get("provider_mutations_performed") != 0
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_candidate_receipt_invalid"])
    for task in tasks:
        if (
            not isinstance(task, Mapping)
            or task.get("artifixer3d_distillation", {}).get("camera_partition_eligible") is not True
            or task.get("artifixer3d_distillation", {}).get("execution_eligible") is not False
            or task.get("direct_prediction_coverage_indices")
            != list(range(int(task.get("camera_count") or 0)))
            or any(
                frame.get("outside_support_changed_pixels") != 0
                for frame in task.get("frames") or []
                if isinstance(frame, Mapping)
            )
        ):
            raise ArtiFixer3DBundleError(["artifixer3d_bundle_candidate_task_invalid"])
    return value


def _absolute_bound(record: Any, *, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ArtiFixer3DBundleError([code])
    path = _file(record.get("path") or "", code=code)
    if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
        raise ArtiFixer3DBundleError([code])
    return path, _read(path, code=code)


def materialize_artifixer3d_use_attestation(
    *,
    candidate_inputs_receipt_path: str | Path,
    output_path: str | Path,
    authorized_by: str,
    authorization_kind: str = "explicit_user_direction_in_current_goal",
) -> dict[str, Any]:
    """Turn explicit noncommercial direction into a digest-bound authority file."""

    receipt_path = _file(
        candidate_inputs_receipt_path,
        code="artifixer3d_attestation_candidate_receipt_missing",
    )
    candidate = _candidate_receipt(receipt_path)
    authority_path, authority = _absolute_bound(
        candidate.get("execution_authority"),
        code="artifixer3d_attestation_execution_authority_invalid",
    )
    if (
        not authorized_by.strip()
        or authorization_kind != "explicit_user_direction_in_current_goal"
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_attestation_authority_invalid"])
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ArtiFixer3DBundleError(["artifixer3d_attestation_output_exists"])
    value: dict[str, Any] = {
        "schema_version": USE_ATTESTATION_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": candidate["publisher_scene_id"],
        "authorization_kind": authorization_kind,
        "authorized_by": authorized_by.strip(),
        "candidate_input_receipt": {
            **_record(receipt_path),
            "receipt_digest": candidate["receipt_digest"],
        },
        "parent_execution_authority": {
            **_record(authority_path),
            "authority_digest": authority["authority_digest"],
        },
        "artifixer_source_license": "Apache-2.0",
        "artifixer_model_license": {
            "name": "NVIDIA One-Way Noncommercial License",
            "url": MODEL_LICENSE_URL,
            "sha256": MODEL_LICENSE_SHA256,
        },
        "internal_noncommercial_research_and_development_only": True,
        "private_derived_input_upload_authorized": True,
        "raw_dataset_bytes_upload_authorized": False,
        "provider_training_authorized": False,
        "commercial_use_authorized": False,
        "redistribution_authorized": False,
        "publication_authorized": False,
        "simulator_or_generated_output_is_physical_evidence": False,
        "attestation_digest": "",
    }
    value["attestation_digest"] = canonical_digest(value, digest_field="attestation_digest")
    _write_json(output, value)
    return value


def _validated_use_attestation(path: Path, *, candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = _read(path, code="artifixer3d_use_attestation_unreadable")
    license_row = value.get("artifixer_model_license")
    if (
        value.get("schema_version") != USE_ATTESTATION_SCHEMA_VERSION
        or value.get("attestation_digest")
        != canonical_digest(value, digest_field="attestation_digest")
        or value.get("publisher_scene_id") != candidate.get("publisher_scene_id")
        or value.get("candidate_input_receipt", {}).get("receipt_digest")
        != candidate.get("receipt_digest")
        or not isinstance(license_row, Mapping)
        or license_row.get("url") != MODEL_LICENSE_URL
        or license_row.get("sha256") != MODEL_LICENSE_SHA256
        or value.get("internal_noncommercial_research_and_development_only") is not True
        or value.get("private_derived_input_upload_authorized") is not True
        or value.get("raw_dataset_bytes_upload_authorized") is not False
        or value.get("provider_training_authorized") is not False
        or value.get("commercial_use_authorized") is not False
        or value.get("redistribution_authorized") is not False
        or value.get("publication_authorized") is not False
        or value.get("simulator_or_generated_output_is_physical_evidence") is not False
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_use_attestation_invalid"])
    _absolute_bound(
        value.get("candidate_input_receipt"),
        code="artifixer3d_use_attestation_candidate_unbound",
    )
    _absolute_bound(
        value.get("parent_execution_authority"),
        code="artifixer3d_use_attestation_parent_unbound",
    )
    return value


def _git_value(root: Path, *args: str, code: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtiFixer3DBundleError([code]) from exc


def _repository_identity(root: Path) -> dict[str, Any]:
    commit = _git_value(root, "rev-parse", "HEAD", code="artifixer3d_repository_invalid")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}", code="artifixer3d_repository_invalid")
    dirty = _git_value(root, "status", "--porcelain=v1", code="artifixer3d_repository_invalid")
    if dirty or len(commit) != 40 or len(tree) != 40:
        raise ArtiFixer3DBundleError(["artifixer3d_repository_not_clean_immutable"])
    return {"commit": commit, "tree": tree, "tracked_files_clean": True}


def _copy_source_release(source: Path, destination: Path) -> list[dict[str, Any]]:
    if (
        _git_value(source, "rev-parse", "HEAD", code="artifixer3d_source_invalid")
        != ARTIFIXER_COMMIT
        or _git_value(source, "rev-parse", "HEAD^{tree}", code="artifixer3d_source_invalid")
        != ARTIFIXER_TREE
        or _git_value(source, "status", "--porcelain=v1", code="artifixer3d_source_invalid")
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_source_invalid"])
    names = _git_value(source, "ls-files", code="artifixer3d_source_index_invalid").splitlines()
    records: list[dict[str, Any]] = []
    for name in sorted(names):
        if name == "thirdparty/3DGRUT-ArtiFixer":
            continue
        relative = _portable(name)
        item = source.joinpath(*relative.parts)
        if not item.is_file() or item.is_symlink():
            raise ArtiFixer3DBundleError(["artifixer3d_source_member_invalid"])
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)
        if _sha256(item) != _sha256(target):
            raise ArtiFixer3DBundleError(["artifixer3d_source_copy_invalid"])
        records.append(_record(target, root=destination))
    if not records:
        raise ArtiFixer3DBundleError(["artifixer3d_source_empty"])
    return records


def _copy_candidate_tree(source: Path, destination: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in sorted(source.rglob("*")):
        if item.is_dir() and not item.is_symlink():
            continue
        if not item.is_file() or item.is_symlink():
            raise ArtiFixer3DBundleError(["artifixer3d_candidate_tree_member_invalid"])
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, target)
        except OSError:
            shutil.copyfile(item, target)
        if _sha256(item) != _sha256(target):
            raise ArtiFixer3DBundleError(["artifixer3d_candidate_tree_copy_invalid"])
        records.append(_record(target, root=destination))
    if not records:
        raise ArtiFixer3DBundleError(["artifixer3d_candidate_tree_empty"])
    return records


def _zip_tree(source: Path, destination: Path) -> None:
    total = 0
    count = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_dir() and not item.is_symlink():
                continue
            if not item.is_file() or item.is_symlink():
                raise ArtiFixer3DBundleError(["artifixer3d_bundle_archive_member_invalid"])
            relative = item.relative_to(source).as_posix()
            size = item.stat().st_size
            total += size
            count += 1
            if size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES or count > MAX_MEMBER_COUNT:
                raise ArtiFixer3DBundleError(["artifixer3d_bundle_archive_limit_exceeded"])
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with item.open("rb") as input_stream, archive.open(info, "w") as output:
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _same_file_record(left: Any, right: Any) -> bool:
    """Compare immutable file identity while deliberately ignoring local paths."""

    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("size_bytes") == right.get("size_bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _checkpoint_reuse_source(
    *,
    candidate: Mapping[str, Any],
    source_provider_output_zip_path: str | Path,
    source_provider_zero_path: str | Path,
    artifixer3d_steps: int,
) -> dict[str, Any]:
    """Re-open one zero-closed training attempt and locate its exact checkpoints.

    The provider ZIP is the byte authority for both the runtime result and each
    checkpoint.  The zero receipt closes the paid-attempt lineage.  Callers do
    not identify checkpoint paths independently, which prevents mixing a valid
    closeout with bytes from some other training run.
    """

    provider_zip = _file(
        source_provider_output_zip_path,
        code="artifixer3d_checkpoint_reuse_provider_zip_missing",
    )
    provider_zero_path = _file(
        source_provider_zero_path,
        code="artifixer3d_checkpoint_reuse_provider_zero_missing",
    )
    provider_zero = _read(
        provider_zero_path,
        code="artifixer3d_checkpoint_reuse_provider_zero_unreadable",
    )
    authority_path, authority = _absolute_bound(
        provider_zero.get("attempt_authority"),
        code="artifixer3d_checkpoint_reuse_authority_invalid",
    )
    attempt_result_path, attempt_result = _absolute_bound(
        provider_zero.get("attempt_result"),
        code="artifixer3d_checkpoint_reuse_attempt_result_invalid",
    )
    inventory = provider_zero.get("inventory")
    authority_digest = authority.get("authorization_digest")
    if (
        provider_zero.get("schema_version") != "artifixer3d_postblocked_provider_zero.v1"
        or provider_zero.get("receipt_digest")
        != canonical_digest(provider_zero, digest_field="receipt_digest")
        or provider_zero.get("provider_zero_confirmed") is not True
        or provider_zero.get("continuing_spend_from_attempt") is not False
        or provider_zero.get("all_staged_objects_absent") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
        or authority.get("schema_version") != "public_scene_artifixer3d_paid_attempt_authority.v1"
        or authority_digest != canonical_digest(authority, digest_field="authorization_digest")
        or provider_zero.get("attempt_authority_digest") != authority_digest
        or not _same_file_record(provider_zero.get("attempt_authority"), _record(authority_path))
        or not _same_file_record(provider_zero.get("attempt_result"), _record(attempt_result_path))
        or attempt_result.get("schema_version") != "public_scene_artifixer3d_vast_run.v1"
        or attempt_result.get("status") not in {"blocked", "completed"}
        or provider_zero.get("attempt_terminal_status") != attempt_result.get("status")
        or attempt_result.get("authorization_consumption", {}).get("status") != "consumed"
        or attempt_result.get("authorization_consumption", {}).get("authorization_digest")
        != authority_digest
        or attempt_result.get("continuing_spend_from_this_run") is not False
        or attempt_result.get("all_staged_objects_absent") is not True
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_checkpoint_reuse_source_attempt_invalid"])

    runtime_result_path = _file(
        attempt_result.get("execution_result_path") or "",
        code="artifixer3d_checkpoint_reuse_runtime_result_missing",
    )
    runtime_result_bytes = runtime_result_path.read_bytes()
    try:
        with zipfile.ZipFile(provider_zip) as archive:
            if archive.testzip() is not None:
                raise ArtiFixer3DBundleError(["artifixer3d_checkpoint_reuse_provider_zip_invalid"])
            try:
                archived_runtime_bytes = archive.read(
                    "public_scene_artifixer3d_runtime_result.json"
                )
            except KeyError as exc:
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_checkpoint_reuse_runtime_result_missing"]
                ) from exc
            if archived_runtime_bytes != runtime_result_bytes:
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_checkpoint_reuse_runtime_result_mismatch"]
                )
            try:
                runtime_result = json.loads(runtime_result_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_checkpoint_reuse_runtime_result_invalid"]
                ) from exc
            if not isinstance(runtime_result, Mapping):
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_checkpoint_reuse_runtime_result_invalid"]
                )
            task_ids = [str(task.get("task_id") or "") for task in candidate["tasks"]]
            runtime_tasks = runtime_result.get("tasks")
            if (
                runtime_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
                or runtime_result.get("status")
                != "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
                or runtime_result.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
                or runtime_result.get("candidate_input_receipt_digest")
                != candidate.get("receipt_digest")
                or runtime_result.get("task_ids") != task_ids
                or runtime_result.get("artifixer3d_distillation_executed") is not True
                or runtime_result.get("artifixer_direct_inference_executed") is not False
                or runtime_result.get("semantic_editor_inference_executed") is not False
                or runtime_result.get("artifixer3d_plus_inference_executed") is not False
                or not isinstance(runtime_tasks, list)
                or len(runtime_tasks) != len(task_ids)
                or attempt_result.get("manifest_digest") != runtime_result.get("manifest_digest")
                or attempt_result.get("runtime_request_digest")
                != runtime_result.get("runtime_request_digest")
            ):
                raise ArtiFixer3DBundleError(
                    ["artifixer3d_checkpoint_reuse_runtime_result_invalid"]
                )

            checkpoints: list[dict[str, Any]] = []
            for index, (task_id, task) in enumerate(zip(task_ids, runtime_tasks)):
                if not isinstance(task, Mapping):
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_checkpoint_reuse_checkpoint_invalid"]
                    )
                checkpoint = task.get("artifixer3d_checkpoint")
                provider_path = str(
                    checkpoint.get("path") if isinstance(checkpoint, Mapping) else ""
                ).replace("\\", "/")
                marker = "/runtime_output/"
                if (
                    task.get("task_id") != task_id
                    or task.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
                    or marker not in provider_path
                    or not isinstance(checkpoint, Mapping)
                ):
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_checkpoint_reuse_checkpoint_invalid"]
                    )
                member = provider_path.split(marker, 1)[1]
                if Path(member).name != f"ckpt_{artifixer3d_steps}.pt":
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_checkpoint_reuse_checkpoint_invalid"]
                    )
                try:
                    info = archive.getinfo(member)
                except KeyError as exc:
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_checkpoint_reuse_checkpoint_missing"]
                    ) from exc
                if (
                    info.is_dir()
                    or info.file_size != checkpoint.get("size_bytes")
                    or info.file_size <= 0
                    or info.file_size > MAX_MEMBER_BYTES
                ):
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_checkpoint_reuse_checkpoint_invalid"]
                    )
                checkpoints.append(
                    {
                        "task_id": task_id,
                        "source_member": member,
                        "source_record": dict(checkpoint),
                        "archive_index": index,
                    }
                )
    except zipfile.BadZipFile as exc:
        raise ArtiFixer3DBundleError(["artifixer3d_checkpoint_reuse_provider_zip_invalid"]) from exc

    return {
        "provider_output_zip_path": provider_zip,
        "provider_zero_path": provider_zero_path,
        "authority_path": authority_path,
        "attempt_result_path": attempt_result_path,
        "runtime_result_path": runtime_result_path,
        "provider_zero": provider_zero,
        "authority": authority,
        "attempt_result": attempt_result,
        "runtime_result": runtime_result,
        "checkpoints": checkpoints,
    }


def build_artifixer3d_bundle(
    *,
    candidate_inputs_receipt_path: str | Path,
    use_attestation_path: str | Path,
    artifixer_source_directory: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    allowed_active_instance_ids: Sequence[int] = (),
    artifixer3d_steps: int = 30_000,
    random_seed: int = 840_920,
    direct_editor_backend: str | None = None,
    semantic_editor_only: bool = False,
    pipeline_mode: str | None = None,
    reused_checkpoint_provider_output_zip_path: str | Path | None = None,
    reused_checkpoint_source_provider_zero_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and locally rehearse one immutable no-upload provider bundle."""

    receipt_path = _file(
        candidate_inputs_receipt_path,
        code="artifixer3d_bundle_candidate_receipt_missing",
    )
    candidate = _candidate_receipt(receipt_path)
    default_pipeline = resolve_default_removal_pipeline(str(candidate.get("schema_version") or ""))
    if pipeline_mode is None:
        pipeline_mode = default_pipeline["pipeline_mode"]
    if direct_editor_backend is None:
        direct_editor_backend = (
            default_pipeline["direct_editor_backend"]
            if pipeline_mode == default_pipeline["pipeline_mode"]
            else (
                NO_DIRECT_EDITOR
                if pipeline_mode
                in {DUAL_TARGET_PIPELINE_MODE, DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE}
                else "artifixer"
            )
        )
    attestation_path = _file(
        use_attestation_path, code="artifixer3d_bundle_use_attestation_missing"
    )
    attestation = _validated_use_attestation(attestation_path, candidate=candidate)
    source = Path(artifixer_source_directory).expanduser().resolve()
    repo = Path(repository_root).expanduser().resolve()
    render_only = pipeline_mode == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    reuse_inputs_present = (
        reused_checkpoint_provider_output_zip_path is not None,
        reused_checkpoint_source_provider_zero_path is not None,
    )
    if (
        source.is_symlink()
        or not source.is_dir()
        or repo.is_symlink()
        or not (repo / "scripts" / "run_public_scene_artifixer3d.sh").is_file()
        or not (repo / "scripts" / "public_scene_artifixer3d_runner.py").is_file()
        or not (repo / "src" / "blueprint_pipeline" / "__init__.py").is_file()
        or not (
            repo / "src" / "blueprint_pipeline" / "image_editor_backend_registry.py"
        ).is_file()
        or not (repo / RUNTIME_EDITOR_REGISTRY_MANIFEST).is_file()
        or not isinstance(artifixer3d_steps, int)
        or isinstance(artifixer3d_steps, bool)
        or not 1 <= artifixer3d_steps <= 30_000
        or not isinstance(random_seed, int)
        or isinstance(random_seed, bool)
        or pipeline_mode
        not in {
            FULL_PIPELINE_MODE,
            DUAL_TARGET_PIPELINE_MODE,
            DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
        }
        or (
            pipeline_mode == FULL_PIPELINE_MODE
            and direct_editor_backend not in DIRECT_EDITOR_BACKENDS
        )
        or (
            pipeline_mode in {DUAL_TARGET_PIPELINE_MODE, DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE}
            and direct_editor_backend != NO_DIRECT_EDITOR
        )
        or not isinstance(semantic_editor_only, bool)
        or (
            pipeline_mode == FULL_PIPELINE_MODE
            and semantic_editor_only
            and direct_editor_backend == "artifixer"
        )
        or (
            pipeline_mode == FULL_PIPELINE_MODE
            and direct_editor_backend == "vibe_image_edit"
            and not semantic_editor_only
        )
        or (
            pipeline_mode in {DUAL_TARGET_PIPELINE_MODE, DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE}
            and semantic_editor_only
        )
        or (
            pipeline_mode == FULL_PIPELINE_MODE
            and candidate.get("schema_version") != CANDIDATE_INPUT_SCHEMA
        )
        or (
            pipeline_mode in {DUAL_TARGET_PIPELINE_MODE, DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE}
            and (
                candidate.get("schema_version") != DUAL_TARGET_INPUT_SCHEMA
                or candidate.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
            )
        )
        or (render_only and not all(reuse_inputs_present))
        or (not render_only and any(reuse_inputs_present))
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in allowed_active_instance_ids
        )
    ):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_configuration_invalid"])
    checkpoint_reuse_source = (
        _checkpoint_reuse_source(
            candidate=candidate,
            source_provider_output_zip_path=str(reused_checkpoint_provider_output_zip_path),
            source_provider_zero_path=str(reused_checkpoint_source_provider_zero_path),
            artifixer3d_steps=artifixer3d_steps,
        )
        if render_only
        else None
    )
    repository_identity = _repository_identity(repo)
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "stage"
    runtime = stage / "provider_runtime"
    input_root = runtime / "input"
    source_root = runtime / "ArtiFixer_official"
    runtime.mkdir(parents=True)
    candidate_files = _copy_candidate_tree(receipt_path.parent, input_root)
    source_files = _copy_source_release(source, source_root)
    shutil.copyfile(repo / "scripts" / Path(ENTRYPOINT).name, runtime / Path(ENTRYPOINT).name)
    shutil.copyfile(repo / "scripts" / Path(RUNNER).name, runtime / Path(RUNNER).name)
    runtime_package = runtime / "blueprint_pipeline"
    runtime_package.mkdir()
    shutil.copyfile(
        repo / "src" / "blueprint_pipeline" / "__init__.py",
        runtime_package / "__init__.py",
    )
    shutil.copyfile(
        repo / "src" / "blueprint_pipeline" / "image_editor_backend_registry.py",
        runtime_package / "image_editor_backend_registry.py",
    )
    registry_manifest = stage / RUNTIME_EDITOR_REGISTRY_MANIFEST
    registry_manifest.parent.mkdir(parents=True)
    shutil.copyfile(repo / RUNTIME_EDITOR_REGISTRY_MANIFEST, registry_manifest)
    shutil.copyfile(attestation_path, runtime / "artifixer3d_use_attestation.json")

    checkpoint_reuse: dict[str, Any] | None = None
    if checkpoint_reuse_source is not None:
        reuse_root = input_root / "checkpoint_reuse"
        reuse_root.mkdir()
        copied_receipts: dict[str, Path] = {}
        for name, source_path in (
            ("source_attempt_authority", checkpoint_reuse_source["authority_path"]),
            ("source_attempt_result", checkpoint_reuse_source["attempt_result_path"]),
            ("source_provider_zero", checkpoint_reuse_source["provider_zero_path"]),
            ("source_runtime_result", checkpoint_reuse_source["runtime_result_path"]),
        ):
            destination = reuse_root / f"{name}.json"
            shutil.copyfile(source_path, destination)
            copied_receipts[name] = destination
        checkpoint_rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(checkpoint_reuse_source["provider_output_zip_path"]) as archive:
            for source_row in checkpoint_reuse_source["checkpoints"]:
                index = int(source_row["archive_index"])
                destination = reuse_root / f"checkpoint_{index:05d}.pt"
                with archive.open(source_row["source_member"]) as source_stream:
                    with destination.open("wb") as destination_stream:
                        shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
                source_record = source_row["source_record"]
                if destination.stat().st_size != source_record.get("size_bytes") or _sha256(
                    destination
                ) != source_record.get("sha256"):
                    raise ArtiFixer3DBundleError(
                        ["artifixer3d_checkpoint_reuse_checkpoint_mismatch"]
                    )
                checkpoint_rows.append(
                    {
                        "task_id": source_row["task_id"],
                        "steps": artifixer3d_steps,
                        "checkpoint": _record(destination, root=input_root),
                        "source_provider_zip_member": source_row["source_member"],
                    }
                )
        checkpoint_reuse = {
            "schema_version": CHECKPOINT_REUSE_SCHEMA_VERSION,
            "source_pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
            "source_candidate_input_receipt_digest": candidate["receipt_digest"],
            "source_manifest_digest": checkpoint_reuse_source["runtime_result"]["manifest_digest"],
            "source_runtime_request_digest": checkpoint_reuse_source["runtime_result"][
                "runtime_request_digest"
            ],
            "source_attempt_terminal_status": checkpoint_reuse_source["attempt_result"]["status"],
            "source_attempt_authority_digest": checkpoint_reuse_source["authority"][
                "authorization_digest"
            ],
            "source_provider_output_zip": _record(
                checkpoint_reuse_source["provider_output_zip_path"]
            ),
            "source_attempt_authority": _record(
                copied_receipts["source_attempt_authority"], root=input_root
            ),
            "source_attempt_result": _record(
                copied_receipts["source_attempt_result"], root=input_root
            ),
            "source_provider_zero": _record(
                copied_receipts["source_provider_zero"], root=input_root
            ),
            "source_runtime_result": _record(
                copied_receipts["source_runtime_result"], root=input_root
            ),
            "checkpoints": checkpoint_rows,
            "training_reexecution_permitted": False,
            "direct_inference_permitted": False,
            "artifixer3d_plus_permitted": False,
            "provider_zero_confirmed_before_reuse": True,
            "reuse_digest": "",
        }
        checkpoint_reuse["reuse_digest"] = canonical_digest(
            checkpoint_reuse, digest_field="reuse_digest"
        )

    runtime_request: dict[str, Any] = {
        "schema_version": RUNTIME_REQUEST_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": candidate["publisher_scene_id"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "replacement_object_count": candidate["replacement_object_count"],
        "task_ids": [task["task_id"] for task in candidate["tasks"]],
        "repair_target": ("plausible_object_free_background_inside_exact_support_only"),
        "source_object_restoration_permitted": False,
        "outside_exact_support_changed_pixels_permitted": 0,
        "artifixer": {
            "repository": ARTIFIXER_REPOSITORY,
            "commit": ARTIFIXER_COMMIT,
            "tree": ARTIFIXER_TREE,
            "submodule_repository": ARTIFIXER_SUBMODULE_REPOSITORY,
            "submodule_commit": ARTIFIXER_SUBMODULE_COMMIT,
            "submodule_tree": ARTIFIXER_SUBMODULE_TREE,
        },
        "model": {
            "repository": ARTIFIXER_MODEL_REPOSITORY,
            "revision": ARTIFIXER_MODEL_REVISION,
            "files": list(ARTIFIXER_MODEL_FILES),
            "license": "NVIDIA One-Way Noncommercial License",
            "use_scope": "internal_noncommercial_research_and_development_only",
        },
        "wan_base": {
            "repository": WAN_MODEL_REPOSITORY,
            "revision": WAN_MODEL_REVISION,
            "files": list(WAN_RUNTIME_FILES),
            "omitted_unneeded_components": ["text_encoder", "tokenizer", "transformer_weights"],
        },
        "container_image": DEFAULT_IMAGE,
        "blueprint_source_identity": repository_identity,
        "use_attestation": {
            "schema_version": USE_ATTESTATION_SCHEMA_VERSION,
            "attestation_digest": attestation["attestation_digest"],
        },
        "direct_inference": {
            "checkpoint_filename": "artifixer-1.3b.pt",
            "inference_pipeline": "kv_cache",
            "num_inference_steps": 4,
            "frames_per_block": 7,
            "neighbor_selection_mode": "evenly_spaced",
            "fold_policy": "two_complementary_target_folds",
            "prompt": "unconditioned_zero_embedding",
        },
        "artifixer3d": {
            "steps": artifixer3d_steps,
            "config_name": "apps/colmap_3dgut_sparse_mcmc_lpips",
            "use_wandb": False,
        },
        "random_seed": random_seed,
        "pipeline_mode": pipeline_mode,
        "direct_editor_backend": direct_editor_backend,
        "phases": [
            (
                "semantic_editor_inference"
                if direct_editor_backend != "artifixer"
                else "direct_inference"
            ),
            "exact_support_composite",
            "artifixer3d_distillation",
            "artifixer3d_render",
            "artifixer3d_plus_inference",
            "artifixer3d_plus_exact_support_composite",
        ],
        "runtime_request_digest": "",
    }
    if pipeline_mode in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }:
        runtime_request.update(
            {
                "repair_target": (
                    "whole_frame_semantic_empty_scene_distillation_with_"
                    "original_outside_support_anchors"
                ),
                "outside_exact_support_changed_pixels_permitted": (
                    "unconstrained_for_raw_representation_review"
                ),
                "outside_support_invariance_gate": ("deferred_until_final_soft_composite"),
                "semantic_editor_only": False,
                "model": None,
                "wan_base": None,
                "direct_inference": None,
                "semantic_editor": None,
                "direct_model_weights_required": False,
                "phases": [
                    "dual_target_input_validation",
                    "artifixer3d_distillation",
                    "artifixer3d_review_render",
                    "native_appearance_export",
                    "external_visual_and_multiview_review",
                ],
            }
        )
        runtime_request["artifixer3d"].update(
            {
                "loss_overrides": {
                    "loss.use_l1": True,
                    "loss.lambda_l1": 1.0,
                    "loss.use_l2": False,
                    "loss.use_ssim": False,
                    "loss.use_lpips_override": True,
                    "loss.lambda_lpips_override": 0.1,
                    "loss.lambda_reconlosses_override": 0.0,
                },
                "whole_semantic_teacher_unmasked": True,
                "anchor_mask_reduction": "full_frame_mean",
                "direct_artifixer_bypassed": True,
                "hard_exact_support_composite_bypassed": True,
                "artifixer3d_plus_bypassed": True,
            }
        )
        if render_only:
            runtime_request.update(
                {
                    "repair_target": (
                        "render_only_replay_of_zero_closed_dual_target_artifixer3d_checkpoint"
                    ),
                    "phases": [
                        "reused_checkpoint_validation",
                        "deterministic_distillation_input_replay",
                        "artifixer3d_review_render",
                        "native_appearance_export",
                        "external_visual_and_multiview_review",
                    ],
                }
            )
            runtime_request["artifixer3d"].update(
                {
                    "checkpoint_reuse": checkpoint_reuse,
                    "training_permitted": False,
                    "distillation_input_replay_only": True,
                }
            )
    if pipeline_mode == FULL_PIPELINE_MODE and direct_editor_backend != "artifixer":
        semantic_editor = {
            "backend": direct_editor_backend,
            "input_role": "black_exact_mask_hole_with_original_context",
            "prompt_policy": "generic_object_absent_background_completion_v1",
            "output_must_be_exact_support_composited": True,
        }
        semantic_editor.update(
            {
                "repository": VIBE_IMAGE_EDIT_REPOSITORY,
                "revision": VIBE_IMAGE_EDIT_REVISION,
                "license": VIBE_IMAGE_EDIT_LICENSE,
                "large_files": list(VIBE_IMAGE_EDIT_LARGE_FILES),
                "source": {
                    "repository": VIBE_SOURCE_REPOSITORY,
                    "commit": VIBE_SOURCE_COMMIT,
                    "tree": VIBE_SOURCE_TREE,
                    "license_sha256": VIBE_SOURCE_LICENSE_SHA256,
                },
                "torch_version": "2.6.0",
                "diffusers_version": "0.33.1",
                "transformers_version": "4.57.1",
                "num_inference_steps": 20,
                "image_guidance_scale": 1.2,
                "guidance_scale": 4.5,
                "enable_model_cpu_offload": False,
                "maximum_expected_vram_gib": 24,
            }
        )
        runtime_request["semantic_editor"] = semantic_editor
        runtime_request["semantic_editor_only"] = semantic_editor_only
        if semantic_editor_only:
            runtime_request["phases"] = [
                "semantic_editor_inference",
                "exact_support_composite",
                "external_visual_and_multiview_review",
            ]
    runtime_request["runtime_request_digest"] = canonical_digest(
        runtime_request, digest_field="runtime_request_digest"
    )
    _write_json(runtime / Path(RUNTIME_REQUEST).name, runtime_request)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_no_upload_no_execution",
        "entrypoint": ENTRYPOINT,
        "runner": RUNNER,
        "runtime_request": {
            **_record(runtime / Path(RUNTIME_REQUEST).name, root=stage),
            "runtime_request_digest": runtime_request["runtime_request_digest"],
        },
        "candidate_input_receipt": {
            **_record(input_root / receipt_path.name, root=stage),
            "receipt_digest": candidate["receipt_digest"],
        },
        "candidate_files": candidate_files,
        "source_files": source_files,
        "source_identity": runtime_request["artifixer"],
        "model_identity": runtime_request["model"],
        "wan_base_identity": runtime_request["wan_base"],
        "direct_model_weights_required": pipeline_mode == FULL_PIPELINE_MODE,
        "direct_editor_backend": direct_editor_backend,
        "semantic_editor_only": semantic_editor_only,
        "pipeline_mode": pipeline_mode,
        "semantic_editor_model_identity": runtime_request.get("semantic_editor"),
        "container_image": DEFAULT_IMAGE,
        "blueprint_source_identity": repository_identity,
        "use_attestation": {
            **_record(runtime / "artifixer3d_use_attestation.json", root=stage),
            "attestation_digest": attestation["attestation_digest"],
        },
        "allowed_active_instance_ids": sorted(set(allowed_active_instance_ids)),
        "contains_raw_dataset_bytes": False,
        "contains_private_derived_inputs": True,
        "contains_model_weights": render_only,
        "contains_reused_private_derived_3dgrut_checkpoint": render_only,
        "contains_released_direct_model_weights": False,
        "provider_mutations_performed": 0,
        "expected_runtime_result_schema": RUNTIME_RESULT_SCHEMA_VERSION,
        "manifest_digest": "",
    }
    if pipeline_mode in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }:
        manifest["model_identity"] = None
        manifest["wan_base_identity"] = None
    if render_only:
        manifest["checkpoint_reuse"] = checkpoint_reuse
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_json(runtime / Path(MANIFEST).name, manifest)
    bundle_path = output / "public_scene_artifixer3d_provider_bundle.zip"
    _zip_tree(stage, bundle_path)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path=ENTRYPOINT,
        evidence_path=output / "artifixer3d_exact_bundle_rehearsal.json",
    )
    if rehearsal.get("status") != "passed" or rehearsal.get("provider_mutations_performed") != 0:
        raise ArtiFixer3DBundleError(["artifixer3d_bundle_rehearsal_failed"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_rehearsal_passed_no_upload_no_execution",
        "bundle": _record(bundle_path),
        "manifest_digest": manifest["manifest_digest"],
        "runtime_request_digest": runtime_request["runtime_request_digest"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "replacement_object_count": candidate["replacement_object_count"],
        "task_ids": runtime_request["task_ids"],
        "task_camera_counts": {
            str(task["task_id"]): int(task.get("physical_camera_count") or task.get("camera_count"))
            for task in candidate["tasks"]
        },
        "task_training_record_counts": {
            str(task["task_id"]): int(task.get("training_record_count") or task.get("camera_count"))
            for task in candidate["tasks"]
        },
        "direct_editor_backend": direct_editor_backend,
        "semantic_editor_only": semantic_editor_only,
        "pipeline_mode": pipeline_mode,
        "checkpoint_reuse_digest": (checkpoint_reuse["reuse_digest"] if checkpoint_reuse else None),
        "container_image": DEFAULT_IMAGE,
        "blueprint_source_identity": repository_identity,
        "use_attestation_digest": attestation["attestation_digest"],
        "allowed_active_instance_ids": manifest["allowed_active_instance_ids"],
        "local_rehearsal": rehearsal,
        "provider_mutations_performed": 0,
        "private_derived_upload_performed": False,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    receipt = output / "public_scene_artifixer3d_bundle_receipt.json"
    _write_json(receipt, result)
    return result


__all__ = [
    "ARTIFIXER_COMMIT",
    "ARTIFIXER_MODEL_FILES",
    "DEFAULT_REMOVAL_PIPELINE_POLICY",
    "ARTIFIXER_MODEL_REVISION",
    "ARTIFIXER_SUBMODULE_COMMIT",
    "ARTIFIXER_SUBMODULE_TREE",
    "ARTIFIXER_TREE",
    "ArtiFixer3DBundleError",
    "DEFAULT_IMAGE",
    "DUAL_TARGET_INPUT_RECEIPT",
    "DUAL_TARGET_PIPELINE_MODE",
    "DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE",
    "CHECKPOINT_REUSE_SCHEMA_VERSION",
    "DIRECT_EDITOR_BACKENDS",
    "FULL_PIPELINE_MODE",
    "NO_DIRECT_EDITOR",
    "SCHEMA_VERSION",
    "VIBE_IMAGE_EDIT_LARGE_FILES",
    "VIBE_IMAGE_EDIT_REPOSITORY",
    "VIBE_IMAGE_EDIT_REVISION",
    "VIBE_SOURCE_COMMIT",
    "VIBE_SOURCE_REPOSITORY",
    "VIBE_SOURCE_TREE",
    "WAN_MODEL_REVISION",
    "WAN_RUNTIME_FILES",
    "build_artifixer3d_bundle",
    "materialize_artifixer3d_use_attestation",
    "resolve_default_removal_pipeline",
]

def main(argv: list[str] | None = None) -> int:
    """Build the immutable paired ArtiFixer3D provider bundle.

    The allocator refuses a bundle whose commit is not the one the control
    plane is running, and every deploy moves that commit. A bundle that can
    only be produced by calling a Python function is therefore launchable
    exactly once and then never again -- which is the state this was in, with a
    `status: ready` receipt pinned to a commit the host had long since left.

    The first cut of this entry point offered eight flags for a builder that
    takes thirteen, which quietly fixed five decisions at their defaults. Three
    of them are not incidental: the editor backend and `--semantic-editor-only`
    choose *which* removal pipeline is built, so the CLI could only ever
    produce whatever the input schema happens to default to; the two
    checkpoint-reuse paths are how a rebuild avoids paying for a checkpoint it
    already has; and the instance allowlist decides whether an already-running
    instance is one this bundle may proceed alongside.

    Performs no provider mutation and rents nothing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build the immutable paired ArtiFixer3D provider bundle.")
    parser.add_argument("--candidate-inputs-receipt", required=True)
    parser.add_argument("--use-attestation", required=True)
    parser.add_argument("--artifixer-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--artifixer3d-steps", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--pipeline-mode")
    parser.add_argument(
        "--direct-editor-backend",
        choices=sorted(DIRECT_EDITOR_BACKENDS | {NO_DIRECT_EDITOR}),
        help="Defaults to whatever the candidate input schema implies.",
    )
    parser.add_argument(
        "--semantic-editor-only",
        action="store_true",
        default=None,
        help="Pairs with the editor backend; the builder refuses invalid pairs.",
    )
    parser.add_argument("--reused-checkpoint-provider-output-zip")
    parser.add_argument("--reused-checkpoint-source-provider-zero")
    parser.add_argument(
        "--allow-active-instance",
        action="append",
        type=int,
        default=[],
        help="Repeatable. Instances that may already be running.",
    )
    args = parser.parse_args(argv)
    optional = {
        "artifixer3d_steps": args.artifixer3d_steps,
        "random_seed": args.random_seed,
        "pipeline_mode": args.pipeline_mode,
        "direct_editor_backend": args.direct_editor_backend,
        "semantic_editor_only": args.semantic_editor_only,
        "reused_checkpoint_provider_output_zip_path": args.reused_checkpoint_provider_output_zip,
        "reused_checkpoint_source_provider_zero_path": args.reused_checkpoint_source_provider_zero,
    }
    try:
        receipt = build_artifixer3d_bundle(
            candidate_inputs_receipt_path=args.candidate_inputs_receipt,
            use_attestation_path=args.use_attestation,
            artifixer_source_directory=args.artifixer_source,
            output_root=args.output_root,
            repository_root=args.repository_root,
            allowed_active_instance_ids=tuple(args.allow_active_instance or ()),
            **{key: value for key, value in optional.items() if value is not None},
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return (
        0
        if receipt.get("status")
        in {"ready", "sealed", "sealed_rehearsal_passed_no_upload_no_execution"}
        else 2
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
