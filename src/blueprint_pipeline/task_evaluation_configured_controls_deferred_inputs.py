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

from .task_evaluation_deferred_controls_contract import (
    TRAJECTORY_MODE as TRAJECTORY_MODE,
    OVERVIEW_MODE as OVERVIEW_MODE,
    SCENE_BUNDLE_MODE as SCENE_BUNDLE_MODE,
    DEFERRED_KEY as DEFERRED_KEY,
    DEFERRABLE_MODES as DEFERRABLE_MODES,
    ConfiguredControlsDeferredInputError as ConfiguredControlsDeferredInputError,
    deferred_declarations as deferred_declarations,
    concrete_paths as concrete_paths,
)

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
    _quaternion_rotate_xyzw,
    materialize_native_task_construction_phase_plan,
)
from .native_franka_action_math import grasp_orientation_contact_xyzw
from .task_evaluation_articulated_open_close_native_adapter import (
    TaskEvaluationArticulatedOpenCloseNativeAdapterError,
    adapt_articulated_open_close_task_template,
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
    ARTICULATED_PLACEMENT_PLAN_SCHEMA_VERSION,
    RobotPlacementTrajectoryError,
    placement_trajectory_from_native_plan,
)


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
    """Return the task-bound path used to place the robot before execution.

    The same native adapter and construction-plan materializer the arena applies
    at compile time run here on CPU, so the plan placement is screened against is
    the plan the construction launch will execute.
    """

    references = _materialized_references(revision=revision, documents=documents)
    try:
        task_definition = json.loads(documents[DEFINITION_CONTRACT_PATH].read_text())
    except (OSError, ValueError, KeyError) as exc:
        raise ConfiguredControlsDeferredInputError("configured_controls_deferred_task_definition_invalid") from exc
    if task_definition.get("strategy") == "articulated_open_close":
        return _articulated_placement_plan(revision=revision, documents=documents, references=references)
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


def _articulated_placement_plan(
    *, revision: Mapping[str, Any], documents: Mapping[str, Path],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Project the qualified handle and passive joint into placement-only poses.

    This path screens robot reach. It never commands the drawer or supplies a
    scripted success to policy evaluation.
    """
    try:
        adapted = adapt_articulated_open_close_task_template(
            configured_revision=revision, materialized_references=references,
        )
    except TaskEvaluationArticulatedOpenCloseNativeAdapterError as exc:
        raise ConfiguredControlsDeferredInputError(
            f"configured_controls_deferred_adapter_failed:{exc}"
        ) from exc
    definition = adapted["native_task_definition"]
    spec = definition["task_spec"]
    graph = spec["articulation_graph"]
    target = next((row for row in graph["joints"] if row["role"] == "target"), None)
    if target is None or target["joint_type"] != "prismatic":
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_articulated_placement_joint_unsupported"
        )
    asset_reference = _reference(
        (revision.get("replacement") or {}).get("asset"),
        blocker="configured_controls_deferred_articulated_asset_reference_invalid",
    )
    asset = documents.get("scene.configured_revision.replacement.asset")
    if asset is None or Path(asset).is_symlink() or not Path(asset).is_file():
        raise ConfiguredControlsDeferredInputError("configured_controls_deferred_articulated_asset_missing")
    payload = Path(asset).read_bytes()
    if _digest(payload) != asset_reference["digest"] or len(payload) != asset_reference["size_bytes"]:
        raise ConfiguredControlsDeferredInputError("configured_controls_deferred_articulated_asset_mismatch")
    affordance = spec["interaction_affordance"]
    root_pose = definition["task_object_pose_world"]
    root_position = root_pose["position_world_m"]
    root_orientation = root_pose["orientation_xyzw"]
    try:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(asset))
        root_prim = stage.GetPrimAtPath("/Asset") if stage is not None else None
        link_path = next(
            path for path in affordance["contact_body_prim_paths"]
            if path.endswith("/" + affordance["contact_link_id"])
        )
        link_prim = stage.GetPrimAtPath(link_path) if stage is not None else None
        if not root_prim or not root_prim.IsValid() or not link_prim or not link_prim.IsValid():
            raise ValueError("contact_link_missing")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        point_stage = cache.GetLocalToWorldTransform(link_prim).Transform(
            Gf.Vec3d(*affordance["contact_point_link_m"])
        )
        point_asset = cache.GetLocalToWorldTransform(root_prim).GetInverse().Transform(point_stage)
        point_world = [
            float(root_position[i]) + value
            for i, value in enumerate(_quaternion_rotate_xyzw(root_orientation, list(point_asset)))
        ]
        approach = _quaternion_rotate_xyzw(root_orientation, affordance["approach_unit_asset_root"])
        jaw = _quaternion_rotate_xyzw(root_orientation, affordance["jaw_unit_asset_root"])
        axis = _quaternion_rotate_xyzw(root_orientation, target["axis"])
        orientation = grasp_orientation_contact_xyzw(approach_axis=approach, jaw_axis=jaw)
    except (ImportError, ValueError, StopIteration, RuntimeError, TypeError, KeyError) as exc:
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_articulated_handle_projection_failed"
        ) from exc
    clearance = float(affordance["precontact_clearance_m"])
    opening = float(spec["executable_opening_threshold"]["success_interval"][0])
    if not 0.0 < opening <= float(target["limits"][1]):
        raise ConfiguredControlsDeferredInputError(
            "configured_controls_deferred_articulated_opening_invalid"
        )

    def phase(name: str, point: list[float], state: str) -> dict[str, Any]:
        return {
            "phase_id": name, "position_world_m": point,
            "orientation_world_xyzw": orientation, "gripper_state": state,
            "gate_ids": ["native_ik", "native_collision_readback"],
        }

    precontact = [point_world[i] - approach[i] * clearance for i in range(3)]
    opened = [point_world[i] + axis[i] * opening for i in range(3)]
    phases = [
        phase("approach", precontact, "open"),
        phase("handle_contact", point_world, "open"),
        phase("minimum_opening_reach", opened, "closed"),
    ]
    plan = {
        "schema_version": ARTICULATED_PLACEMENT_PLAN_SCHEMA_VERSION,
        "task_kind": "articulated_open_close",
        "manipulation_strategy": "articulated_open_close",
        "configured_scene_revision_digest": revision["revision_digest"],
        "adapter_digest": adapted["adapter_digest"],
        "qualified_asset_digest": asset_reference["digest"],
        "qualified_static_digest": revision["replacement"]["static_qualification"]["digest"],
        "target_joint_id": target["joint_id"],
        "minimum_opening_m": opening,
        "phases": phases, "phase_count": len(phases),
        "execution_parameters": {
            "arrival_tolerance_m": 0.02,
            "arrival_orientation_tolerance_rad": 0.08,
            "maximum_steps_per_phase": 64,
        },
        "claim_boundary": {
            "placement_only_no_task_joint_command": True,
            "policy_action_and_native_readback_required_for_success": True,
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    placement_trajectory_from_native_plan(plan)
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
        try:
            definition = json.loads(documents[DEFINITION_CONTRACT_PATH].read_text())
        except (OSError, ValueError) as exc:
            raise ConfiguredControlsDeferredInputError("configured_controls_deferred_task_definition_invalid") from exc
        if definition.get("strategy") == "articulated_open_close":
            asset_reference = _reference(
                (revision.get("replacement") or {}).get("asset"),
                blocker="configured_controls_deferred_articulated_asset_reference_invalid",
            )
            asset_path = root / "documents" / "articulated-replacement.usdz"
            if not _retained_matches(asset_path, reference=asset_reference):
                _write_immutable_bytes(
                    asset_path, _fetched(asset_reference, fetcher=fetcher),
                    conflict="configured_controls_deferred_articulated_asset_conflict",
                )
            documents["scene.configured_revision.replacement.asset"] = asset_path
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
