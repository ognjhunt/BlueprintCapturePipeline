"""Derive first-run controls inputs from the published revision instead of prior runs.

A configured-controls autostart intent must be registered before the scene
configuration is activated, yet two of its inputs only exist after that run:
the native rigid trajectory plan (which needs the statically qualified
replacement, its native import, and the registered support) and the overview
image shown to the placement reviewer.  The 839873 rehearsal bridged that gap
by rebinding a plan and an image from an earlier diagnostic, which a fresh
scene cannot do.

An intent may instead declare those inputs *deferred*.  After publication the
autostart resolves them here from the exact revision documents, fetched by
their published references and verified byte for byte, through the same native
adapter and construction-plan materializer the runtime applies.  Resolution is
idempotent and immutable; it executes nothing and screens nothing itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest
from .native_task_construction_plan import (
    NativeTaskConstructionPlanError,
    materialize_native_task_construction_phase_plan,
)
from .task_evaluation_native_arena_episode_compiler import _runtime_subject_task_spec
from .task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION_CONTRACT_PATH,
    EXECUTION_CONTRACT_PATH,
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
    SOURCE_OBJECT_CONTRACT_PATH,
    STATIC_QUALIFICATION_CONTRACT_PATH,
    SUCCESS_CONTRACT_PATH,
    SUPPORT_PLANE_CONTRACT_PATH,
    TaskEvaluationRigidRelocationNativeAdapterError,
    adapt_rigid_relocation_task_template,
)
from .task_evaluation_robot_placement_trajectory import (
    RobotPlacementTrajectoryError,
    placement_trajectory_from_native_plan,
)


TRAJECTORY_MODE = "derive_from_configured_revision"
OVERVIEW_MODE = "configured_task_thumbnail"
SCENE_BUNDLE_MODE = "configured_scene_bundle"
DEFERRED_KEY = "deferred"
DEFERRABLE_MODES = {
    "native_trajectory_plan_path": TRAJECTORY_MODE,
    "overview_image_paths": OVERVIEW_MODE,
}
DEFERRED_DIRECTORY = "deferred-inputs"
TRAJECTORY_FILE_NAME = "native_trajectory_plan.v1.json"
THUMBNAIL_FILE_NAME = "configured_task_thumbnail.png"
RUNTIME_BINDING_FILE_NAME = "runtime_binding.v1.json"
RIGID_PLAN_SCHEMA_VERSION = "native_rigid_construction_phase_plan.v1"
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
REVISION_DOCUMENTS: dict[str, tuple[str, str]] = {
    DEFINITION_CONTRACT_PATH: ("task_template", "definition"),
    SUCCESS_CONTRACT_PATH: ("task_template", "success_criteria"),
    EXECUTION_CONTRACT_PATH: ("task_template", "execution"),
    SUPPORT_PLANE_CONTRACT_PATH: ("registration", "support_plane"),
    SOURCE_OBJECT_CONTRACT_PATH: ("replacement", "source_object"),
    STATIC_QUALIFICATION_CONTRACT_PATH: ("replacement", "static_qualification"),
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH: (
        "replacement",
        "native_import_qualification",
    ),
}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

ReferenceFetcher = Callable[[Mapping[str, Any]], bytes]


class ConfiguredControlsDeferredInputError(RuntimeError):
    """A deferred controls input could not be derived from exact published bytes."""


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reference(value: Any, *, blocker: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not str(value.get("uri") or "").strip()
        or _DIGEST.fullmatch(str(value.get("digest") or "")) is None
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] < 1
    ):
        raise ConfiguredControlsDeferredInputError(blocker)
    return {
        "uri": str(value["uri"]),
        "digest": str(value["digest"]),
        "size_bytes": int(value["size_bytes"]),
    }


def _write_immutable_bytes(path: Path, payload: bytes, *, conflict: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".deferred-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o440)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != payload:
                    raise ConfiguredControlsDeferredInputError(conflict) from None
        finally:
            temporary.unlink(missing_ok=True)
    return path


# ------------------------------------------------------------------ declarations


def deferred_declarations(paths: Any) -> dict[str, str]:
    """Return ``{input_name: mode}`` for every deferred input the intent declares."""

    if not isinstance(paths, Mapping):
        return {}
    declared: dict[str, str] = {}
    for name, value in paths.items():
        if not isinstance(value, Mapping):
            continue
        mode = DEFERRABLE_MODES.get(str(name))
        if (
            mode is None
            or set(value) != {DEFERRED_KEY}
            or value.get(DEFERRED_KEY) != mode
        ):
            raise ConfiguredControlsDeferredInputError(
                f"configured_controls_deferred_declaration_invalid:{name}"
            )
        declared[str(name)] = mode
    return declared


def concrete_paths(paths: Mapping[str, Any]) -> dict[str, Any]:
    """Return the intent paths without their deferred declarations."""

    declared = deferred_declarations(paths)
    return {name: value for name, value in paths.items() if name not in declared}


# ------------------------------------------------------------------ trajectory


def _document_references(revision: Mapping[str, Any]) -> dict[str, Any]:
    refs = {path: (revision.get(section) or {}).get(key)
            for path, (section, key) in REVISION_DOCUMENTS.items()}
    destination = (revision.get("task_template") or {}).get("destination")
    if destination is not None:
        refs["task.destination.geometry"] = destination.get("geometry")
    return refs


def _materialized_references(
    *, revision: Mapping[str, Any], documents: Mapping[str, Path]
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for contract_path, declared_reference in _document_references(revision).items():
        reference = _reference(
            declared_reference,
            blocker=f"configured_controls_deferred_revision_reference_invalid:{contract_path}",
        )
        local = documents.get(contract_path)
        if local is None:
            raise ConfiguredControlsDeferredInputError(
                f"configured_controls_deferred_document_missing:{contract_path}"
            )
        path = Path(local)
        if path.is_symlink() or not path.is_file():
            raise ConfiguredControlsDeferredInputError(
                f"configured_controls_deferred_document_missing:{contract_path}"
            )
        payload = path.read_bytes()
        if _digest(payload) != reference["digest"] or len(payload) != reference["size_bytes"]:
            raise ConfiguredControlsDeferredInputError(
                f"configured_controls_deferred_document_mismatch:{contract_path}"
            )
        references[contract_path] = {
            "contract_path": contract_path,
            **reference,
            "materialized_path": str(path),
            "full_byte_service_account_readback_passed": True,
        }
    return references


def derive_native_trajectory_plan(
    *, revision: Mapping[str, Any], documents: Mapping[str, Path]
) -> dict[str, Any]:
    """Return the rigid construction phase plan the runtime derives from this revision.

    The same native adapter and construction-plan materializer the arena applies
    at compile time run here on CPU, so the plan placement is screened against is
    the plan the construction launch will execute.
    """

    references = _materialized_references(revision=revision, documents=documents)
    try:
        adapted = adapt_rigid_relocation_task_template(
            configured_revision=revision, materialized_references=references
        )
    except TaskEvaluationRigidRelocationNativeAdapterError as exc:
        raise ConfiguredControlsDeferredInputError(
            f"configured_controls_deferred_adapter_failed:{exc}"
        ) from exc
    definition = adapted["native_task_definition"]
    task_spec = dict(definition["task_spec"])
    task_spec["subject_asset_id"] = str(revision["replacement"]["identity"]["id"])
    task_spec["manipulation_strategy"] = str(
        task_spec.get("manipulation_strategy") or adapted.get("strategy") or ""
    )
    task_spec["success_criteria"] = adapted["native_success_criteria"].get("criteria")
    task_spec = _runtime_subject_task_spec(task_spec)
    destination = revision["task_template"].get("destination")
    if destination is not None:
        from .task_evaluation_rigid_destination_geometry import (
            RigidDestinationGeometryError, bind_destination_trajectory, destination_trajectory_geometry,
        )
        try:
            geometry = json.loads(documents["task.destination.geometry"].read_text())
        except (OSError, ValueError) as exc:
            raise ConfiguredControlsDeferredInputError("configured_controls_deferred_destination_geometry_invalid") from exc
        if (not isinstance(geometry, Mapping) or geometry.get("subject_identity") != revision["replacement"]["identity"]
                or geometry.get("subject_static_qualification_digest") != revision["replacement"]["static_qualification"]["digest"]
                or geometry.get("destination_static_qualification_digest") != destination["static_qualification"]["digest"]):
            raise ConfiguredControlsDeferredInputError("configured_controls_deferred_destination_binding_invalid")
        try:
            task_spec = bind_destination_trajectory(task_spec, destination_trajectory_geometry(destination, geometry))
        except RigidDestinationGeometryError as exc:
            raise ConfiguredControlsDeferredInputError("configured_controls_deferred_destination_geometry_invalid") from exc
    scene_plan = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "task_kind": "rigid_pick_place",
        "task_spec": task_spec,
        "objects": [
            {
                "semantic_role": "replacement",
                "asset_id": task_spec["subject_asset_id"],
                "source_asset_id": task_spec["source_subject_identity"],
                "task_subject": True,
                "object_type": "RIGID",
                "reset_state": {
                    "root_pose_world": definition["task_object_pose_world"],
                    "joint_positions": {},
                },
            }
        ],
        "cadence": {"maximum_action_steps": task_spec["maximum_action_steps"]},
        "plan_digest": "",
    }
    scene_plan["plan_digest"] = canonical_digest(scene_plan, digest_field="plan_digest")
    try:
        plan = materialize_native_task_construction_phase_plan(scene_plan)
    except NativeTaskConstructionPlanError as exc:
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_plan_failed:" + ",".join(str(item) for item in exc.errors)
        ) from exc
    if plan.get("schema_version") != RIGID_PLAN_SCHEMA_VERSION:
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_plan_schema_invalid"
        )
    try:
        placement_trajectory_from_native_plan(plan)
    except RobotPlacementTrajectoryError as exc:
        raise ConfiguredControlsDeferredInputError(
            f"configured_controls_deferred_plan_unprojectable:{exc}"
        ) from exc
    return plan


# ------------------------------------------------------------------ fetching


def default_reference_fetcher(reference: Mapping[str, Any]) -> bytes:
    """Read one published object with the service account's ambient credentials."""

    bound = _reference(reference, blocker="configured_controls_deferred_reference_invalid")
    parsed = urlsplit(bound["uri"])
    limit = min(MAX_DOCUMENT_BYTES, bound["size_bytes"]) + 1
    if parsed.scheme == "s3":
        # Revisions retain original task references on the legacy store while
        # new private qualification documents can live in the artifact store.
        # Import lazily: preparation also imports controls-related contracts.
        from .task_evaluation_launch_preparation_worker import (
            TaskEvaluationLaunchPreparationWorkerError,
            _s3_client,
        )

        try:
            client = _s3_client(parsed.netloc)
        except TaskEvaluationLaunchPreparationWorkerError as exc:
            raise ConfiguredControlsDeferredInputError(
                f"configured_controls_deferred_fetch_configuration_invalid:{exc}"
            ) from exc
        try:
            response = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            body = response["Body"]
            try:
                payload = body.read(limit)
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
        except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
            raise ConfiguredControlsDeferredInputError(
                "configured_controls_deferred_fetch_failed"
            ) from exc
        return payload
    if parsed.scheme == "https":
        request = urllib.request.Request(bound["uri"], method="GET")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # nosec B310
                if response.geturl() != bound["uri"]:
                    raise ConfiguredControlsDeferredInputError(
                        "configured_controls_deferred_fetch_redirect_refused"
                    )
                return response.read(limit)
        except OSError as exc:
            raise ConfiguredControlsDeferredInputError(
                "configured_controls_deferred_fetch_failed"
            ) from exc
    raise ConfiguredControlsDeferredInputError(
        "configured_controls_deferred_fetch_scheme_unsupported"
    )


def _fetched(reference: Mapping[str, Any], *, fetcher: ReferenceFetcher) -> bytes:
    bound = _reference(reference, blocker="configured_controls_deferred_reference_invalid")
    payload = bytes(fetcher(bound))
    if _digest(payload) != bound["digest"] or len(payload) != bound["size_bytes"]:
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_fetch_mismatch"
        )
    return payload


def _retained_matches(path: Path, *, reference: Mapping[str, Any]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    payload = path.read_bytes()
    return _digest(payload) == reference.get("digest") and len(payload) == reference.get(
        "size_bytes"
    )


def _document_name(contract_path: str) -> str:
    return contract_path.removeprefix("scene.configured_revision.").replace(".", "-") + ".json"


# ------------------------------------------------------------------ runtime binding


def resolve_runtime_binding(
    *, runtime_binding_path: str | Path, revision: Mapping[str, Any], output_root: str | Path
) -> Path:
    """Bind the construction runtime's scene mount to the exact published bundle.

    The episode request mounts the configured scene bundle the run published,
    a reference that does not exist when the intent is registered.  A binding
    may leave that one mount source deferred; every other field is copied
    unchanged.  A binding with no deferred source is returned as-is.
    """

    source_path = Path(runtime_binding_path).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_runtime_binding_invalid"
        )
    try:
        binding = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_runtime_binding_invalid"
        ) from exc
    runtime = binding.get("runtime") if isinstance(binding, Mapping) else None
    mounts = runtime.get("mounts") if isinstance(runtime, Mapping) else None
    if not isinstance(mounts, list) or not mounts:
        # No deferred marker can exist without mounts; a concrete binding keeps
        # its exact bytes and is validated by the request contract at staging.
        return source_path
    deferred_positions = [
        index
        for index, mount in enumerate(mounts)
        if isinstance(mount, Mapping)
        and isinstance(mount.get("source"), Mapping)
        and DEFERRED_KEY in mount["source"]
    ]
    if not deferred_positions:
        return source_path
    first = mounts[0] if isinstance(mounts[0], Mapping) else {}
    if (
        deferred_positions != [0]
        or first.get("source") != {DEFERRED_KEY: SCENE_BUNDLE_MODE}
        or first.get("mode") != "read_only"
        or first.get("container_path") != "/inputs"
    ):
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_runtime_binding_invalid"
        )
    bundle = _reference(
        revision.get("configured_scene_bundle"),
        blocker="configured_controls_deferred_scene_bundle_reference_invalid",
    )
    resolved = json.loads(json.dumps(binding))
    resolved["runtime"]["mounts"][0]["source"] = bundle
    binding_digest = canonical_digest(resolved).removeprefix("sha256:")
    destination = (Path(output_root).expanduser() / DEFERRED_DIRECTORY /
                   f"{binding_digest}-{RUNTIME_BINDING_FILE_NAME}")
    _write_immutable_bytes(
        destination,
        (json.dumps(resolved, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        conflict="configured_controls_deferred_runtime_binding_conflict",
    )
    return destination


# ------------------------------------------------------------------ resolution


def resolve_deferred_inputs(
    *,
    intent: Mapping[str, Any],
    revision: Mapping[str, Any],
    output_root: str | Path,
    fetcher: ReferenceFetcher = default_reference_fetcher,
) -> dict[str, Any]:
    """Return concrete intent paths, deriving each deferred one once from the revision."""

    paths = dict(intent.get("paths") or {})
    declared = deferred_declarations(paths)
    if isinstance(paths.get("runtime_binding_path"), str):
        paths["runtime_binding_path"] = str(
            resolve_runtime_binding(
                runtime_binding_path=paths["runtime_binding_path"],
                revision=revision,
                output_root=output_root,
            )
        )
    if not declared:
        return paths
    identity = canonical_digest({"configured_revision_digest": revision.get("revision_digest"),
                                 "execution_commit": intent.get("expected_production_commit")})
    root = Path(output_root).expanduser() / DEFERRED_DIRECTORY / identity.removeprefix("sha256:")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if "native_trajectory_plan_path" in declared:
        plan_path = root / TRAJECTORY_FILE_NAME
        documents: dict[str, Path] = {}
        for contract_path, declared_reference in _document_references(revision).items():
            reference = _reference(declared_reference,
                blocker=f"configured_controls_deferred_revision_reference_invalid:{contract_path}")
            document_path = root / "documents" / _document_name(contract_path)
            if not _retained_matches(document_path, reference=reference):
                _write_immutable_bytes(document_path, _fetched(reference, fetcher=fetcher),
                    conflict="configured_controls_deferred_document_conflict")
            documents[contract_path] = document_path
        # Re-derive from the bound retained inputs on every restart. A self-sealed
        # cache alone cannot prove it belongs to this revision or execution code.
        plan = derive_native_trajectory_plan(revision=revision, documents=documents)
        _write_immutable_bytes(plan_path,
            (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            conflict="configured_controls_deferred_plan_conflict")
        paths["native_trajectory_plan_path"] = str(plan_path)
    if "overview_image_paths" in declared:
        presentation = revision.get("presentation")
        reference = _reference(
            presentation.get("task_thumbnail") if isinstance(presentation, Mapping) else None,
            blocker="configured_controls_deferred_thumbnail_reference_invalid",
        )
        thumbnail_path = root / THUMBNAIL_FILE_NAME
        if not _retained_matches(thumbnail_path, reference=reference):
            _write_immutable_bytes(
                thumbnail_path,
                _fetched(reference, fetcher=fetcher),
                conflict="configured_controls_deferred_thumbnail_conflict",
            )
        paths["overview_image_paths"] = [str(thumbnail_path)]
    return paths


__all__ = [
    "ConfiguredControlsDeferredInputError",
    "DEFERRABLE_MODES",
    "OVERVIEW_MODE",
    "SCENE_BUNDLE_MODE",
    "TRAJECTORY_MODE",
    "concrete_paths",
    "default_reference_fetcher",
    "deferred_declarations",
    "derive_native_trajectory_plan",
    "resolve_deferred_inputs",
    "resolve_runtime_binding",
]
