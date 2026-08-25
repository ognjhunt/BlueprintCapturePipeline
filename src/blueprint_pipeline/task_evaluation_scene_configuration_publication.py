"""Publish one provider-built scene as an immutable configured revision."""

from __future__ import annotations

import hashlib
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


def publish_configured_scene_revision(
    *,
    envelope: Mapping[str, Any],
    stage_results: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    publisher: Publisher,
) -> dict[str, Any]:
    """Publish every reusable scene byte, read it back, and seal the revision."""

    request = envelope.get("request")
    recipe = envelope.get("recipe")
    if (
        not isinstance(recipe, Mapping)
        or not isinstance(request, Mapping)
        or request.get("run_mode") != "scene_configuration"
        or request.get("scene", {}).get("mode") != "configure_source_scene"
        or request.get("construction", {}).get("mode") != "production_recipe"
        or recipe.get("provider_disclosure", {}).get(
            "raw_source_bytes_to_external_provider"
        )
        is not False
    ):
        raise TaskEvaluationSceneConfigurationPublicationError(
            "scene_configuration_publication_envelope_invalid"
        )
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
    ):
        artifacts[role] = _artifact(stage_results, role=role)[1]
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
    }
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
            "raw_source_sent_to_external_provider": False,
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
