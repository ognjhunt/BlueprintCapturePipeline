"""Secret-free post-training checkpoint for one ArtiFixer visual-review continuation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_artifixer_failure_evidence import (
    ARTIFIXER_RUNTIME_ACCEPTED_STATUS,
)


SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_artifixer_post_training_checkpoint.v1"
)
STATUS = "ready_for_artifixer_visual_review_only"
_DIGEST = __import__("re").compile(r"sha256:[0-9a-f]{64}\Z")
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "api_key_file",
        "authorization",
        "credential",
        "credentials",
        "secret",
        "token",
    }
)


class ArtifixerPostTrainingCheckpointError(ValueError):
    """The reusable ArtiFixer outputs were incomplete or changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _SECRET_FIELDS
            or _contains_secret_material(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_material(child) for child in value)
    return isinstance(value, str) and value.startswith(("sk-", "Bearer "))


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifixerPostTrainingCheckpointError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise ArtifixerPostTrainingCheckpointError(code)
    return dict(value)


def _input_file(value: str | Path, *, code: str) -> Path:
    unresolved = Path(value).expanduser()
    path = unresolved.resolve()
    if unresolved.is_symlink() or not path.is_file():
        raise ArtifixerPostTrainingCheckpointError(code)
    return path


def artifixer_post_training_binding_digest(bindings: Mapping[str, Any]) -> str:
    """Bind only immutable scientific/runtime identities, never scheduling paths."""

    value = json.loads(json.dumps(dict(bindings)))
    if (
        not value
        or _contains_secret_material(value)
        or any(
            not str(key).strip()
            or str(key).lower().endswith(("_path", "_file", "_url"))
            for key in value
        )
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_binding_invalid"
        )
    return canonical_digest(value)


def _copy_inventory_file(
    *, source: Path, root: Path, relative: str, role: str, mode: int
) -> dict[str, Any]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short ArtiFixer checkpoint write")
                    view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    return {
        "role": role,
        "relative_path": relative,
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "mode": mode,
    }


def materialize_artifixer_post_training_checkpoint(
    *,
    source_diagnostic_checkpoint: Mapping[str, Any],
    bindings: Mapping[str, Any],
    runtime_result_path: str | Path,
    review_frames: Sequence[Mapping[str, Any]],
    native_appearance_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Seal only the exact bytes needed to avoid repeating 30k training steps."""

    source_checkpoint_digest = str(
        source_diagnostic_checkpoint.get("checkpoint_digest") or ""
    )
    scientific_binding_digest = str(
        (source_diagnostic_checkpoint.get("scientific_bindings") or {}).get(
            "binding_digest"
        )
        or ""
    )
    if (
        _DIGEST.fullmatch(source_checkpoint_digest) is None
        or _DIGEST.fullmatch(scientific_binding_digest) is None
        or source_diagnostic_checkpoint.get("completed_stage_prefix_count") != 0
        or source_diagnostic_checkpoint.get("completed_stage_results") != []
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_source_checkpoint_invalid"
        )
    binding_digest = artifixer_post_training_binding_digest(bindings)
    runtime_path = _input_file(
        runtime_result_path,
        code="scene_configuration_artifixer_warm_runtime_result_invalid",
    )
    runtime_result = _read(
        runtime_path,
        code="scene_configuration_artifixer_warm_runtime_result_invalid",
    )
    if (
        runtime_result.get("schema_version")
        != "public_scene_artifixer3d_runtime_result.v1"
        or runtime_result.get("status") != ARTIFIXER_RUNTIME_ACCEPTED_STATUS
        or _contains_secret_material(runtime_result)
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_runtime_result_invalid"
        )
    if (
        len(review_frames) != 8
        or any(not isinstance(row, Mapping) for row in review_frames)
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_review_frame_set_invalid"
        )
    camera_ids = [str(row.get("camera_id") or "") for row in review_frames]
    if not all(camera_ids) or len(set(camera_ids)) != 8:
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_review_frame_set_invalid"
        )
    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_checkpoint_output_invalid"
        )
    root.mkdir(parents=True, mode=0o750)
    inventory: list[dict[str, Any]] = []
    portable_frames: list[dict[str, Any]] = []
    try:
        inventory.append(
            _copy_inventory_file(
                source=runtime_path,
                root=root,
                relative="runtime/runtime_result.json",
                role="artifixer_runtime_result",
                mode=0o440,
            )
        )
        for index, row in enumerate(review_frames):
            final = row.get("final_frame")
            path_value = final.get("path") if isinstance(final, Mapping) else None
            source = _input_file(
                str(path_value or ""),
                code="scene_configuration_artifixer_warm_review_frame_set_invalid",
            )
            role = f"review_frame:{camera_ids[index]}"
            relative = f"review/frames/{index:05d}{source.suffix.lower() or '.bin'}"
            inventory.append(
                _copy_inventory_file(
                    source=source,
                    root=root,
                    relative=relative,
                    role=role,
                    mode=0o440,
                )
            )
            portable_frames.append(
                {
                    "frame_index": row.get("frame_index"),
                    "camera_id": camera_ids[index],
                    "checkpoint_role": role,
                }
            )
        native = _input_file(
            native_appearance_path,
            code="scene_configuration_artifixer_warm_native_appearance_invalid",
        )
        inventory.append(
            _copy_inventory_file(
                source=native,
                root=root,
                relative="native/configured_appearance.usdz",
                role="native_appearance_usdz",
                mode=0o440,
            )
        )
        checkpoint: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "continuation_kind": "artifixer_visual_review_only",
            "source_diagnostic_checkpoint_digest": source_checkpoint_digest,
            "scientific_binding_digest": scientific_binding_digest,
            "bindings": dict(bindings),
            "binding_digest": binding_digest,
            # The canonical ArtiFixer runtime result is not self-digested. Its
            # integrity boundary is the exact provider-output member bytes, so
            # carry that same byte digest into the portable checkpoint.
            "runtime_result_sha256": _sha256(runtime_path),
            "review_frames": portable_frames,
            "native_appearance_role": "native_appearance_usdz",
            "inventory": sorted(inventory, key=lambda row: row["relative_path"]),
            "visual_review_provider_call_started": False,
            "rerun_paid_model_stages": ["artifixer_visual_review"],
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "raw_secret_values_recorded": False,
            "checkpoint_digest": "",
        }
        checkpoint["checkpoint_digest"] = canonical_digest(
            checkpoint, digest_field="checkpoint_digest"
        )
        manifest = root / f"{SCHEMA_VERSION}.json"
        manifest.write_text(
            json.dumps(checkpoint, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest.chmod(0o440)
        return checkpoint
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def validate_artifixer_post_training_checkpoint(
    *,
    checkpoint_root: str | Path,
    expected_binding_digest: str | None = None,
    expected_source_checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    """Reopen every retained byte before any key is read or remote call begins."""

    unresolved = Path(checkpoint_root).expanduser()
    root = unresolved.resolve()
    manifest = root / f"{SCHEMA_VERSION}.json"
    checkpoint = _read(
        manifest, code="scene_configuration_artifixer_warm_checkpoint_invalid"
    )
    inventory = checkpoint.get("inventory")
    frames = checkpoint.get("review_frames")
    if (
        unresolved.is_symlink()
        or not root.is_dir()
        or checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("status") != STATUS
        or checkpoint.get("continuation_kind") != "artifixer_visual_review_only"
        or checkpoint.get("checkpoint_digest")
        != canonical_digest(checkpoint, digest_field="checkpoint_digest")
        or checkpoint.get("binding_digest")
        != artifixer_post_training_binding_digest(checkpoint.get("bindings") or {})
        or (
            expected_binding_digest is not None
            and checkpoint.get("binding_digest") != expected_binding_digest
        )
        or (
            expected_source_checkpoint_digest is not None
            and checkpoint.get("source_diagnostic_checkpoint_digest")
            != expected_source_checkpoint_digest
        )
        or checkpoint.get("visual_review_provider_call_started") is not False
        or checkpoint.get("rerun_paid_model_stages") != ["artifixer_visual_review"]
        or checkpoint.get("diagnostic_only") is not True
        or checkpoint.get("qualification_eligible") is not False
        or checkpoint.get("configured_revision_publication_permitted") is not False
        or checkpoint.get("offering_publication_permitted") is not False
        or checkpoint.get("terminal_e2e_completion_permitted") is not False
        or checkpoint.get("raw_secret_values_recorded") is not False
        or _contains_secret_material(checkpoint)
        or _DIGEST.fullmatch(str(checkpoint.get("runtime_result_sha256") or ""))
        is None
        or not isinstance(inventory, list)
        or len(inventory) != 10
        or not isinstance(frames, list)
        or len(frames) != 8
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_checkpoint_invalid"
        )
    expected_paths: set[str] = set()
    roles: set[str] = set()
    for row in inventory:
        relative = str(row.get("relative_path") or "") if isinstance(row, Mapping) else ""
        role = str(row.get("role") or "") if isinstance(row, Mapping) else ""
        posix = PurePosixPath(relative)
        path = root.joinpath(*posix.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ArtifixerPostTrainingCheckpointError(
                "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
            ) from exc
        if (
            not role
            or role in roles
            or not relative
            or relative in expected_paths
            or posix.is_absolute()
            or ".." in posix.parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
            or stat.S_IMODE(path.stat().st_mode) != row.get("mode")
        ):
            raise ArtifixerPostTrainingCheckpointError(
                "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
            )
        expected_paths.add(relative)
        roles.add(role)
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    frame_roles = [
        str(row.get("checkpoint_role") or "")
        for row in frames
        if isinstance(row, Mapping)
    ]
    if (
        observed_paths != expected_paths
        or any(path.is_symlink() for path in root.rglob("*"))
        or len(frame_roles) != 8
        or len(set(frame_roles)) != 8
        or any(role not in roles for role in frame_roles)
        or checkpoint.get("native_appearance_role") not in roles
        or "artifixer_runtime_result" not in roles
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
        )
    return checkpoint


def _restore_inventory_modes_after_zip_extraction(*, checkpoint_root: Path) -> None:
    """Restore authenticated immutable modes that ``zipfile`` does not preserve."""

    root = checkpoint_root.resolve()
    manifest = root / f"{SCHEMA_VERSION}.json"
    checkpoint = _read(
        manifest, code="scene_configuration_artifixer_warm_checkpoint_invalid"
    )
    inventory = checkpoint.get("inventory")
    if (
        checkpoint_root.is_symlink()
        or not root.is_dir()
        or checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("checkpoint_digest")
        != canonical_digest(checkpoint, digest_field="checkpoint_digest")
        or not isinstance(inventory, list)
        or len(inventory) != 10
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
        )
    descriptors: list[tuple[int, int]] = []
    expected_paths: set[str] = set()
    try:
        for row in inventory:
            relative = (
                str(row.get("relative_path") or "")
                if isinstance(row, Mapping)
                else ""
            )
            mode = row.get("mode") if isinstance(row, Mapping) else None
            posix = PurePosixPath(relative)
            path = root.joinpath(*posix.parts).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ArtifixerPostTrainingCheckpointError(
                    "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
                ) from exc
            if (
                not relative
                or relative in expected_paths
                or posix.is_absolute()
                or ".." in posix.parts
                or isinstance(mode, bool)
                or mode != 0o440
            ):
                raise ArtifixerPostTrainingCheckpointError(
                    "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
                )
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                metadata = os.fstat(descriptor)
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size != row.get("size_bytes")
                    or "sha256:" + digest.hexdigest() != row.get("sha256")
                ):
                    raise ArtifixerPostTrainingCheckpointError(
                        "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
                    )
            except Exception:
                os.close(descriptor)
                raise
            descriptors.append((descriptor, mode))
            expected_paths.add(relative)
        observed_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest
        }
        if observed_paths != expected_paths:
            raise ArtifixerPostTrainingCheckpointError(
                "scene_configuration_artifixer_warm_checkpoint_inventory_invalid"
            )
        for descriptor, mode in descriptors:
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != mode:
                os.fchmod(descriptor, mode)
    finally:
        for descriptor, _mode in descriptors:
            os.close(descriptor)


def validated_artifixer_post_training_checkpoint_reference(
    *, extraction_root: Path, result: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and reopen a portable post-training checkpoint from provider output."""

    reference = result.get("artifixer_post_training_checkpoint")
    if not isinstance(reference, Mapping):
        return None, "scene_configuration_artifixer_warm_checkpoint_missing"
    relative_root = Path(str(reference.get("provider_output_relative_root") or ""))
    relative_manifest = Path(str(reference.get("manifest_relative_path") or ""))
    if (
        relative_root.is_absolute()
        or relative_manifest.is_absolute()
        or not relative_root.parts
        or not relative_manifest.parts
        or ".." in relative_root.parts
        or ".." in relative_manifest.parts
    ):
        return None, "scene_configuration_artifixer_warm_checkpoint_unsafe"
    root = (extraction_root / relative_root).resolve()
    manifest = (extraction_root / relative_manifest).resolve()
    try:
        root.relative_to(extraction_root)
        manifest.relative_to(root)
    except ValueError:
        return None, "scene_configuration_artifixer_warm_checkpoint_unsafe"
    if (
        manifest != root / f"{SCHEMA_VERSION}.json"
        or manifest.is_symlink()
        or not manifest.is_file()
        or _sha256(manifest) != reference.get("manifest_sha256")
    ):
        return None, "scene_configuration_artifixer_warm_checkpoint_invalid"
    try:
        _restore_inventory_modes_after_zip_extraction(checkpoint_root=root)
        checkpoint = validate_artifixer_post_training_checkpoint(checkpoint_root=root)
    except (OSError, RuntimeError, ValueError):
        return None, "scene_configuration_artifixer_warm_checkpoint_invalid"
    files = [path for path in root.rglob("*") if path.is_file()]
    if (
        checkpoint.get("checkpoint_digest") != reference.get("checkpoint_digest")
        or checkpoint.get("source_diagnostic_checkpoint_digest")
        != reference.get("source_diagnostic_checkpoint_digest")
        or checkpoint.get("binding_digest") != reference.get("binding_digest")
        or len(files) != reference.get("file_count")
        or sum(path.stat().st_size for path in files) != reference.get("total_bytes")
    ):
        return None, "scene_configuration_artifixer_warm_checkpoint_invalid"
    return {
        **dict(reference),
        "checkpoint_root": str(root),
        "manifest_path": str(manifest),
    }, None


def hydrate_artifixer_post_training_checkpoint(
    *, checkpoint_root: str | Path, expected_binding_digest: str
) -> dict[str, Any]:
    checkpoint = validate_artifixer_post_training_checkpoint(
        checkpoint_root=checkpoint_root,
        expected_binding_digest=expected_binding_digest,
    )
    root = Path(checkpoint_root).expanduser().resolve()
    by_role = {
        str(row["role"]): root / str(row["relative_path"])
        for row in checkpoint["inventory"]
    }
    runtime_result = _read(
        by_role["artifixer_runtime_result"],
        code="scene_configuration_artifixer_warm_runtime_result_invalid",
    )
    if _sha256(by_role["artifixer_runtime_result"]) != checkpoint[
        "runtime_result_sha256"
    ]:
        raise ArtifixerPostTrainingCheckpointError(
            "scene_configuration_artifixer_warm_runtime_result_invalid"
        )
    return {
        "checkpoint": checkpoint,
        "runtime_result": runtime_result,
        "review_frames": [
            {
                "frame_index": row["frame_index"],
                "camera_id": row["camera_id"],
                "final_frame": {
                    "path": str(by_role[row["checkpoint_role"]].resolve()),
                    "sha256": _sha256(by_role[row["checkpoint_role"]]),
                },
            }
            for row in checkpoint["review_frames"]
        ],
        "native_appearance_path": str(
            by_role[checkpoint["native_appearance_role"]].resolve()
        ),
    }


__all__ = [
    "ArtifixerPostTrainingCheckpointError",
    "SCHEMA_VERSION",
    "STATUS",
    "artifixer_post_training_binding_digest",
    "hydrate_artifixer_post_training_checkpoint",
    "materialize_artifixer_post_training_checkpoint",
    "validated_artifixer_post_training_checkpoint_reference",
    "validate_artifixer_post_training_checkpoint",
]
