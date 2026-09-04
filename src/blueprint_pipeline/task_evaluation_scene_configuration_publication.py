"""Publish one provider-built scene as an immutable configured revision."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from .task_evaluation_configured_scene_public_projection import (
    ConfiguredScenePublicProjectionError,
    build_public_display_projection,
)
from .task_evaluation_scene_configuration_disclosure import (
    render_inputs_disclosure_is_coherent,
    renders_on_provider,
)
from .task_evaluation_rigid_destination_geometry import (
    RigidDestinationGeometryError,
    derive_rigid_destination_geometry,
)
from .task_evaluation_scene_configuration_appearance_review import (
    AppearanceReviewContractError,
    PAUSED_UNGRADED_MODE,
    PAUSED_UNGRADED_WARNING,
    REQUIRED_MODE,
    appearance_review_mode,
    paused_review_receipt_valid,
)


MAX_TASK_THUMBNAIL_SIZE_BYTES = 16 * 1024 * 1024


Publisher = Callable[..., Mapping[str, Any]]
RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_publication.v1"


class TaskEvaluationSceneConfigurationPublicationError(RuntimeError):
    """The configured scene could not be published and read back exactly."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _artifact(
    stage_results: Sequence[Mapping[str, Any]], *, role: str
) -> tuple[dict[str, Any], Path]:
    matches = [
        dict(row)
        for result in stage_results
        for row in result.get("output_artifacts") or []
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationPublicationError(
            f"scene_configuration_publication_artifact_missing:{role}"
        )
    row = matches[0]
    path = Path(str(row.get("path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            f"scene_configuration_publication_artifact_invalid:{role}"
        )
    return row, path


def _thumbnail_selection(
    *,
    review_receipt_path: Path,
    thumbnail_path: Path,
    removal_receipt_path: Path | None = None,
    review_mode: str = REQUIRED_MODE,
    minimum_frame_count: int = 8,
) -> dict[str, Any]:
    try:
        receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
        removal = (
            json.loads(removal_receipt_path.read_text(encoding="utf-8"))
            if (
                removal_receipt_path is not None
                and review_mode == PAUSED_UNGRADED_MODE
            )
            else {}
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_thumbnail_selection_invalid"
        ) from exc
    if not isinstance(receipt, Mapping) or not isinstance(removal, Mapping):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_thumbnail_selection_invalid"
        )
    selection = receipt.get("task_thumbnail_selection")
    reviewer = receipt.get("reviewer")
    thumbnail_digest, thumbnail_size = _sha256_and_size(thumbnail_path)
    if review_mode == PAUSED_UNGRADED_MODE:
        selector = receipt.get("selector")
        if (
            removal.get("status")
            != "completed_ungraded_generated_appearance_edit"
            or removal.get("visual_review_mode") != PAUSED_UNGRADED_MODE
            or removal.get("visual_review_receipt_digest")
            != receipt.get("receipt_digest")
            or removal.get("warning_label") != PAUSED_UNGRADED_WARNING
            or not paused_review_receipt_valid(
                receipt,
                publisher_instance_id=str(
                    removal.get("publisher_instance_id") or ""
                ),
                minimum_frame_count=minimum_frame_count,
                thumbnail_digest=thumbnail_digest,
            )
            or not isinstance(selector, Mapping)
            or thumbnail_size < 1
            or thumbnail_size > MAX_TASK_THUMBNAIL_SIZE_BYTES
        ):
            raise TaskEvaluationSceneConfigurationPublicationError(
                "scene_configuration_publication_thumbnail_selection_invalid"
            )
        return {
            "camera_id": selection["camera_id"],
            "frame_digest": selection["frame_sha256"],
            "rationale": str(selection["rationale"]).strip(),
            "reviewer": {
                key: selector[key]
                for key in ("kind", "identity", "runtime", "model")
            },
            "appearance_review_status": PAUSED_UNGRADED_MODE,
        }
    if (
        review_mode != REQUIRED_MODE
        or receipt.get("schema_version")
        != "task_evaluation_artifixer_ai_visual_review.v1"
        or receipt.get("status") != "accepted"
        or receipt.get("review_frame_count") != 8
        or receipt.get("task_thumbnail_is_exact_review_frame") is not True
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or not isinstance(selection, Mapping)
        or set(selection) != {"camera_id", "frame_sha256", "rationale"}
        or not str(selection.get("camera_id") or "")
        or selection.get("frame_sha256") != thumbnail_digest
        or not str(selection.get("rationale") or "").strip()
        or not isinstance(reviewer, Mapping)
        or reviewer.get("kind") != "ai"
        or not str(reviewer.get("identity") or "")
        or not str(reviewer.get("runtime") or "")
        or not str(reviewer.get("model") or "")
        or thumbnail_size < 1
        or thumbnail_size > MAX_TASK_THUMBNAIL_SIZE_BYTES
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_thumbnail_selection_invalid"
        )
    return {
        "camera_id": selection["camera_id"],
        "frame_digest": selection["frame_sha256"],
        "rationale": str(selection["rationale"]).strip(),
        "reviewer": {
            key: reviewer[key]
            for key in ("kind", "identity", "runtime", "model")
        },
        "appearance_review_status": "accepted",
    }


def _publish(
    *,
    publisher: Publisher,
    path: Path,
    object_name: str,
) -> dict[str, Any]:
    observed = dict(publisher(path=path, object_name=object_name))
    expected_digest, expected_size = _sha256_and_size(path)
    reference = {
        key: observed.get(key) for key in ("uri", "digest", "size_bytes")
    }
    if (
        not isinstance(reference["uri"], str)
        or not reference["uri"].startswith(("gs://", "s3://", "https://"))
        or reference["digest"] != expected_digest
        or reference["size_bytes"] != expected_size
        or observed.get("full_byte_service_account_readback_passed") is not True
        or observed.get("readback_digest") != expected_digest
        or observed.get("readback_size_bytes") != expected_size
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_readback_invalid"
        )
    return {
        **reference,
        "full_byte_service_account_readback_passed": True,
        "readback_digest": expected_digest,
        "readback_size_bytes": expected_size,
    }


def _deterministic_bundle(
    *, files: Sequence[tuple[str, Path]], destination: Path
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for name, path in sorted(files):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100440 << 16
                with archive.open(info, "w") as output, path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
        os.chmod(temporary, 0o440)
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _materialized_reference_file(
    envelope: Mapping[str, Any], *, contract_path: str
) -> tuple[dict[str, Any], Path]:
    matches = [
        dict(row)
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping) and row.get("contract_path") == contract_path
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationPublicationError(
            f"scene_configuration_publication_reference_missing:{contract_path}"
        )
    row = matches[0]
    path = Path(str(row.get("materialized_path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            f"scene_configuration_publication_reference_invalid:{contract_path}"
        )
    return row, path


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationPublicationError(code) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationPublicationError(code)
    return dict(value)


_DESTINATION_STAGE_ROLES = (
    "statically_qualified_destination_asset",
    "destination_static_qualification_receipt",
    "destination_static_requalification_receipt",
    "native_qualified_destination_asset",
    "destination_native_import_qualification_receipt",
)


def _supplemental_destination_publication(
    *,
    envelope: Mapping[str, Any],
    request: Mapping[str, Any],
    recipe: Mapping[str, Any],
    stage_results: Sequence[Mapping[str, Any]],
    subject_static_receipt_path: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Return the destination files to publish, deriving its task geometry.

    The run produced the destination's native-import receipt; publication is
    where the subject's stage-4 static bounds and the destination's SimReady
    interior first coexist, so the geometry is derived here and nowhere else.
    """

    task = request.get("task") if isinstance(request, Mapping) else None
    destination = task.get("destination") if isinstance(task, Mapping) else None
    supplemental = recipe.get("supplemental_destination")
    declared_roles = {
        str(row.get("role") or "")
        for result in stage_results
        for row in result.get("output_artifacts") or []
        if isinstance(row, Mapping)
    }
    if destination is None:
        if supplemental is not None or any(
            role in declared_roles for role in _DESTINATION_STAGE_ROLES
        ):
            raise TaskEvaluationSceneConfigurationPublicationError(
                "scene_configuration_publication_destination_undeclared"
            )
        return {}
    if (
        not isinstance(destination, Mapping)
        or not isinstance(supplemental, Mapping)
        or supplemental.get("identity") != destination.get("identity")
        or supplemental.get("relation") != destination.get("relation")
        or supplemental.get("asset") != destination.get("asset")
        or supplemental.get("static_qualification")
        != destination.get("static_qualification")
        or "native_import_qualification" in destination
        or "geometry" in destination
        or "placement_qualification" in destination
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_destination_binding_invalid"
        )
    identity = dict(destination["identity"])
    _asset_record, asset = _artifact(stage_results, role="native_qualified_destination_asset")
    _static_asset_record, static_asset = _artifact(
        stage_results, role="statically_qualified_destination_asset"
    )
    _static_record, static_path = _artifact(
        stage_results, role="destination_static_qualification_receipt"
    )
    _requalification_record, requalification_path = _artifact(
        stage_results, role="destination_static_requalification_receipt"
    )
    _native_record, native_path = _artifact(
        stage_results, role="destination_native_import_qualification_receipt"
    )
    asset_digest, asset_size = _sha256_and_size(asset)
    static_digest, static_size = _sha256_and_size(static_path)
    static = _read_json(
        static_path, code="scene_configuration_publication_destination_static_invalid"
    )
    native = _read_json(
        native_path, code="scene_configuration_publication_destination_native_invalid"
    )
    if (
        _sha256_and_size(static_asset) != (asset_digest, asset_size)
        or destination["asset"].get("digest") != asset_digest
        or destination["asset"].get("size_bytes") != asset_size
        or destination["static_qualification"].get("digest") != static_digest
        or destination["static_qualification"].get("size_bytes") != static_size
        or static.get("replacement_identity") != identity
        or native.get("status") != "qualified"
        or native.get("replacement_identity") != identity
        or native.get("asset_digest") != asset_digest
        or native.get("static_qualification_digest") != static_digest
        or native.get("native_simulator_import_qualified") is not True
        or native.get("result_digest")
        != canonical_digest(native, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_destination_binding_invalid"
        )
    _definition_row, definition_path = _materialized_reference_file(
        envelope, contract_path="task.definition"
    )
    _simready_row, simready_path = _materialized_reference_file(
        envelope,
        contract_path="construction.recipe.supplemental_destination.simready_result",
    )
    definition = _read_json(
        definition_path,
        code="scene_configuration_publication_task_definition_invalid",
    )
    task_spec = definition.get("task_spec")
    affordance = (
        task_spec.get("interaction_affordance") if isinstance(task_spec, Mapping) else None
    )
    transform = (
        affordance.get("asset_root_from_scoring_frame")
        if isinstance(affordance, Mapping)
        else None
    )
    if not isinstance(transform, Mapping):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_task_definition_invalid"
        )
    probe = destination.get("native_probe")
    limits = probe.get("qualification_limits") if isinstance(probe, Mapping) else None
    if not isinstance(limits, Mapping):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_destination_probe_invalid"
        )
    try:
        geometry = derive_rigid_destination_geometry(
            subject_identity=recipe["subject_identity"],
            destination_identity=identity,
            relation=str(destination["relation"]),
            pose_world=destination["pose_world"],
            subject_static_qualification=_read_json(
                subject_static_receipt_path,
                code="scene_configuration_publication_subject_static_invalid",
            ),
            subject_static_qualification_digest=_sha256_and_size(
                subject_static_receipt_path
            )[0],
            subject_scoring_transform=transform,
            destination_static_qualification=static,
            destination_static_qualification_digest=static_digest,
            destination_simready_result=_read_json(
                simready_path,
                code="scene_configuration_publication_destination_simready_invalid",
            ),
            qualification_limits=limits,
        )
    except RigidDestinationGeometryError as exc:
        raise TaskEvaluationSceneConfigurationPublicationError(
            f"scene_configuration_publication_destination_geometry_failed:{exc}"
        ) from exc
    geometry_path = output_root / "destination_geometry.v1.json"
    if geometry_path.exists() or geometry_path.is_symlink():
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_destination_geometry_conflict"
        )
    geometry_path.write_text(canonical_json(geometry) + "\n", encoding="utf-8")
    return {
        "destination_asset": asset,
        "destination_static_qualification": static_path,
        "destination_static_requalification": requalification_path,
        "destination_native_import_qualification": native_path,
        "destination_geometry": geometry_path,
        "destination_simready_result": simready_path,
    }


def publish_configured_scene_revision(
    *,
    envelope: Mapping[str, Any],
    stage_results: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    publisher: Publisher,
) -> dict[str, Any]:
    """Publish every reusable scene byte, read it back, and seal the revision."""

    if any(
        result.get("diagnostic_only") is True
        or result.get("qualification_eligible") is False
        or result.get("executed_inside_one_parent_provider_run") is False
        or result.get("configured_revision_publication_permitted") is False
        or result.get("offering_publication_permitted") is False
        for result in stage_results
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_diagnostic_result_publication_forbidden"
        )

    request = envelope.get("request")
    recipe = envelope.get("recipe")
    render_inputs = envelope.get("render_inputs_result")
    disclosure_receipt = envelope.get("provider_disclosure_receipt")
    raw_source_sent = (
        disclosure_receipt.get("raw_interiorgs_bytes_in_provider_bundle")
        if isinstance(disclosure_receipt, Mapping)
        else render_inputs.get("raw_interiorgs_bytes_in_provider_packet")
        if isinstance(render_inputs, Mapping)
        else None
    )
    disclosure_decision = (
        render_inputs.get("disclosure_decision")
        if isinstance(render_inputs, Mapping)
        else None
    )
    production_semantic_reuse = (
        isinstance(render_inputs, Mapping)
        and render_inputs.get("production_semantic_input_reuse") is True
    )
    expected_raw_source_sent = renders_on_provider(disclosure_decision or {}) and not (
        production_semantic_reuse
    )
    if (
        not isinstance(recipe, Mapping)
        or not isinstance(request, Mapping)
        or not isinstance(render_inputs, Mapping)
        or not render_inputs_disclosure_is_coherent(render_inputs)
        or not isinstance(raw_source_sent, bool)
        or raw_source_sent is not expected_raw_source_sent
        or (
            production_semantic_reuse
            and (
                render_inputs.get("provider_render_skipped") is not True
                or render_inputs.get("raw_interiorgs_bytes_in_provider_packet")
                is not False
            )
        )
        or request.get("run_mode") != "scene_configuration"
        or request.get("scene", {}).get("mode") != "configure_source_scene"
        or request.get("construction", {}).get("mode") != "production_recipe"
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_envelope_invalid"
        )
    try:
        review_mode = appearance_review_mode(
            request, allow_historical_paused=True
        )
    except AppearanceReviewContractError as exc:
        raise TaskEvaluationSceneConfigurationPublicationError(str(exc)) from exc
    root = Path(output_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_output_root_invalid"
        )
    namespace = str(request["publication"]["input_namespace"])
    artifacts = {}
    for role in (
        "configured_appearance_without_source_object",
        "appearance_removal_receipt",
        "configured_collision_without_source_object",
        "collision_excision_receipt",
        "statically_qualified_replacement_asset",
        "static_qualification_receipt",
        "native_qualified_replacement_asset",
        "native_import_qualification_receipt",
        "configured_scene_bundle_candidate_manifest",
        "scene_assembly_receipt",
        "appearance_visual_review_receipt",
        "configured_task_thumbnail",
    ):
        artifacts[role] = _artifact(stage_results, role=role)[1]
    thumbnail_selection = _thumbnail_selection(
        review_receipt_path=artifacts["appearance_visual_review_receipt"],
        removal_receipt_path=artifacts["appearance_removal_receipt"],
        thumbnail_path=artifacts["configured_task_thumbnail"],
        review_mode=review_mode,
        minimum_frame_count=8,
    )
    bundle = root / "configured_scene_bundle.v1.zip"
    _deterministic_bundle(
        files=[
            ("appearance" + artifacts["configured_appearance_without_source_object"].suffix, artifacts["configured_appearance_without_source_object"]),
            ("collision" + artifacts["configured_collision_without_source_object"].suffix, artifacts["configured_collision_without_source_object"]),
            ("replacement" + artifacts["native_qualified_replacement_asset"].suffix, artifacts["native_qualified_replacement_asset"]),
            ("configured_scene_bundle_candidate.v1.json", artifacts["configured_scene_bundle_candidate_manifest"]),
        ],
        destination=bundle,
    )
    publish_roles = {
        "configured_appearance": artifacts[
            "configured_appearance_without_source_object"
        ],
        "appearance_removal_result": artifacts["appearance_removal_receipt"],
        "configured_collision": artifacts[
            "configured_collision_without_source_object"
        ],
        "collision_excision_result": artifacts["collision_excision_receipt"],
        "replacement_asset": artifacts["native_qualified_replacement_asset"],
        "static_qualification": artifacts["static_qualification_receipt"],
        "native_import_qualification": artifacts[
            "native_import_qualification_receipt"
        ],
        "bundle_manifest": artifacts[
            "configured_scene_bundle_candidate_manifest"
        ],
        "configured_scene_bundle": bundle,
        "task_thumbnail": artifacts["configured_task_thumbnail"],
        "thumbnail_selection_receipt": artifacts[
            "appearance_visual_review_receipt"
        ],
    }
    destination_files = _supplemental_destination_publication(
        envelope=envelope,
        request=request,
        recipe=recipe,
        stage_results=stage_results,
        subject_static_receipt_path=artifacts["static_qualification_receipt"],
        output_root=root,
    )
    publish_roles.update(destination_files)
    published = {
        role: _publish(
            publisher=publisher,
            path=path,
            object_name=f"{namespace}/{role}/{path.name}",
        )
        for role, path in publish_roles.items()
    }
    publication_receipt: dict[str, Any] = {
        "schema_version": "task_evaluation_configured_scene_publication_receipt.v1",
        "status": "published_and_read_back",
        "configuration_run_id": envelope["run_id"],
        "team_namespace": envelope["team_namespace"],
        "objects": [
            {"role": role, **record}
            for role, record in sorted(published.items())
        ],
        "object_count": len(published),
        "full_byte_service_account_readback_passed": True,
        "receipt_digest": "",
    }
    publication_receipt["receipt_digest"] = canonical_digest(
        publication_receipt, digest_field="receipt_digest"
    )
    publication_receipt_path = root / "configured_scene_publication_receipt.v1.json"
    publication_receipt_path.write_text(
        canonical_json(publication_receipt) + "\n", encoding="utf-8"
    )
    published_receipt = _publish(
        publisher=publisher,
        path=publication_receipt_path,
        object_name=f"{namespace}/publication/{publication_receipt_path.name}",
    )
    scene = request["scene"]
    task = request["task"]
    revision: dict[str, Any] = {
        "schema_version": "task_evaluation_configured_scene_revision.v1",
        "status": "configured",
        "configuration_run_id": envelope["run_id"],
        "team_namespace": envelope["team_namespace"],
        "scene_identity": dict(recipe["scene_identity"]),
        "source_commit": envelope["expected_production_commit"],
        "source": {
            "manifest": dict(scene["source_manifest"]),
            "rights_admission": dict(scene["rights"]["admission"]),
            "rights_evidence": [
                {
                    "role": row["role"],
                    "artifact": dict(row["artifact"]),
                }
                for row in scene["rights"]["evidence"]
            ],
            "raw_source_sent_to_external_provider": raw_source_sent,
            **(
                {"provider_disclosure_decision": dict(disclosure_decision)}
                if isinstance(disclosure_decision, Mapping)
                else {}
            ),
            **(
                {"production_semantic_input_reuse": True}
                if production_semantic_reuse
                else {}
            ),
        },
        "appearance": {
            "observed_source": dict(scene["appearance"]["representation"]),
            "object_removal_result": {
                key: published["appearance_removal_result"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "configured_representation": {
                key: published["configured_appearance"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "appearance_truth_source": "interiorgs_observed_plus_labeled_generated_edit",
            "visual_review_status": thumbnail_selection[
                "appearance_review_status"
            ],
            **(
                {"warning_label": PAUSED_UNGRADED_WARNING}
                if review_mode == PAUSED_UNGRADED_MODE
                else {}
            ),
        },
        "geometry": {
            "candidate_collision_source": dict(scene["geometry"]["collision"]),
            "object_excision_result": {
                key: published["collision_excision_result"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "configured_collision": {
                key: published["configured_collision"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "validation": dict(scene["geometry"]["validation"]),
            "observed_source_truth_claimed": False,
        },
        "replacement": {
            "identity": dict(recipe["subject_identity"]),
            "source_object": dict(task["subject"]["source_object"]),
            "asset": {
                key: published["replacement_asset"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "static_qualification": {
                key: published["static_qualification"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "native_import_qualification": {
                key: published["native_import_qualification"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "physics_authority": "qualified_replacement_asset",
        },
        "registration": {
            "metric": dict(scene["registration"]["metric_registration"]),
            "support_plane": dict(scene["registration"]["support_plane"]),
            "robot_mount_interface": dict(
                scene["registration"]["robot_mount_interface"]
            ),
            "camera_calibration": dict(
                scene["registration"]["camera_calibration"]
            ),
            "workspace_clearance": dict(
                scene["registration"]["workspace_clearance"]
            ),
        },
        "configured_scene_bundle": {
            key: published["configured_scene_bundle"][key]
            for key in ("uri", "digest", "size_bytes")
        },
        "task_template": {
            "identity": dict(recipe["task_identity"]),
            "definition": dict(task["definition"]),
            "success_criteria": dict(task["success_criteria"]),
            "execution": dict(task["execution"]),
            **(
                {
                    "destination": {
                        **dict(task["destination"]),
                        "native_import_qualification": {
                            key: published["destination_native_import_qualification"][key]
                            for key in ("uri", "digest", "size_bytes")
                        },
                        "geometry": {
                            key: published["destination_geometry"][key]
                            for key in ("uri", "digest", "size_bytes")
                        },
                    }
                }
                if destination_files
                else {}
            ),
        },
        "presentation": {
            "task_thumbnail": {
                key: published["task_thumbnail"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "selection_receipt": {
                key: published["thumbnail_selection_receipt"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "selection": thumbnail_selection,
            "appearance_review_status": thumbnail_selection[
                "appearance_review_status"
            ],
            "selected_from_exact_reviewed_frame_count": (
                0 if review_mode == PAUSED_UNGRADED_MODE else 8
            ),
            **(
                {"warning_label": PAUSED_UNGRADED_WARNING}
                if review_mode == PAUSED_UNGRADED_MODE
                else {}
            ),
            "derived_appearance_evidence": True,
            "capture_or_physical_evidence": False,
            "image_bytes_modified_after_selection": False,
        },
        "robot_team_interface": {
            "scene_construction_repeated_per_evaluation": False,
            "configuration_run_executed_episode": False,
            "configuration_run_purpose": "build_and_publish_reusable_robot_neutral_scene",
            "episode_run_purpose": "evaluate_one_robot_or_policy_against_configured_scene",
            "episode_packet_compiled_by_production": True,
            "team_supplied_components": [
                "robot_configuration",
                "kinematics_and_joint_bounds",
                "robot_to_scene_registration",
                "controller_or_policy",
                "camera_and_sensor_configuration",
                "task_binding",
                "episode_runtime",
            ],
            "configured_scene_components": [
                "appearance",
                "collision_geometry",
                "replacement_assets",
                "metric_registration",
                "support_plane",
                "robot_mount_interface",
                "workspace_clearance",
                "scene_camera_calibration",
                "rights_and_provenance",
                "task_templates",
                "configured_scene_bundle",
            ],
            "production_route": "authenticated_webapp_to_task_evaluation_dispatcher",
        },
        "publication": {
            "bundle_manifest": {
                key: published["bundle_manifest"][key]
                for key in ("uri", "digest", "size_bytes")
            },
            "receipt": {
                key: published_receipt[key]
                for key in ("uri", "digest", "size_bytes")
            },
            "full_byte_service_account_readback_passed": True,
        },
        "evaluation_admission": {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_admitted": False,
        },
        "revision_digest": "",
    }
    revision["revision_digest"] = canonical_digest(
        revision, digest_field="revision_digest"
    )
    validate_configured_scene_revision(revision)
    revision_path = root / "configured_scene_revision.v1.json"
    revision_path.write_text(canonical_json(revision) + "\n", encoding="utf-8")
    published_revision = _publish(
        publisher=publisher,
        path=revision_path,
        object_name=f"{namespace}/revision/{revision_path.name}",
    )
    offering: dict[str, Any] = {
        "schema_version": "task_evaluation_configured_scene_offering.v1",
        "status": "configured_controls_pending",
        "configuration_run_id": envelope["run_id"],
        "team_namespace": envelope["team_namespace"],
        "catalog_visibility": "team_only",
        "scene_identity": dict(revision["scene_identity"]),
        "task": {
            "identity": dict(revision["task_template"]["identity"]),
            "kind": task["kind"],
            "strategy": task["strategy"],
            "subject_identity": dict(revision["replacement"]["identity"]),
            **(
                {"destination": dict(revision["task_template"]["destination"])}
                if isinstance(revision["task_template"].get("destination"), Mapping)
                else {}
            ),
        },
        "presentation": dict(revision["presentation"]),
        "evaluation_preparation_binding": {
            "scene_mode": "reuse_configured_revision",
            "construction_mode": "reuse_configured_scene",
            "task_binding_mode": "reuse_configured_template",
            # Provenance of the configuration build.  A future evaluation run
            # must bind its own currently deployed evaluator commit instead of
            # treating this historical commit as executable release authority.
            "configuration_source_commit": revision["source_commit"],
            "configured_scene_revision": {
                key: published_revision[key]
                for key in ("uri", "digest", "size_bytes")
            },
            "configured_scene_revision_digest": revision["revision_digest"],
            "configured_scene_bundle": dict(revision["configured_scene_bundle"]),
        },
        "proof_boundary": {
            "thumbnail_is_derived_appearance_evidence": True,
            "thumbnail_is_capture_or_physical_evidence": False,
            "appearance_visual_review_completed": review_mode == REQUIRED_MODE,
            "appearance_quality_graded": review_mode == REQUIRED_MODE,
            "appearance_review_status": thumbnail_selection[
                "appearance_review_status"
            ],
            **(
                {"appearance_warning_label": PAUSED_UNGRADED_WARNING}
                if review_mode == PAUSED_UNGRADED_MODE
                else {}
            ),
            "configuration_is_policy_evaluation": False,
            "configuration_is_deployment_or_safety_approval": False,
        },
        "evaluation_admission": {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_evaluation_admitted": False,
        },
        "offering_digest": "",
    }
    source_offering_digest = canonical_digest(
        offering, digest_field="offering_digest"
    )
    try:
        public_display = build_public_display_projection(
            request=request,
            revision=revision,
            offering=offering,
            source_offering_digest=source_offering_digest,
            diagnostic_only=any(
                result.get("diagnostic_only") is True
                for result in stage_results
            ),
        )
    except ConfiguredScenePublicProjectionError as exc:
        raise TaskEvaluationSceneConfigurationPublicationError(str(exc)) from exc
    if public_display is not None:
        offering["public_display"] = public_display
    offering["offering_digest"] = canonical_digest(
        offering, digest_field="offering_digest"
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "configured_scene_published",
        "configuration_run_id": envelope["run_id"],
        "configured_scene_revision": {
            "role": "configured_scene_revision",
            "path": str(revision_path),
            "digest": _sha256_and_size(revision_path)[0],
            "size_bytes": _sha256_and_size(revision_path)[1],
        },
        "configured_scene_revision_reference": {
            key: published_revision[key]
            for key in ("uri", "digest", "size_bytes")
        },
        "configured_scene_revision_digest": revision["revision_digest"],
        "configured_scene_bundle_reference": revision[
            "configured_scene_bundle"
        ],
        "task_thumbnail_reference": revision["presentation"]["task_thumbnail"],
        "task_thumbnail_selection": revision["presentation"]["selection"],
        "task_thumbnail_selection_receipt_reference": revision["presentation"][
            "selection_receipt"
        ],
        "configured_scene_offering": offering,
        "publication_receipt_digest": publication_receipt["receipt_digest"],
        "full_byte_service_account_readback_passed": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationPublicationError",
    "publish_configured_scene_revision",
]
