from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_bundle as bundle_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    TaskEvaluationSceneConfigurationBundleError,
    build_scene_configuration_provider_bundle,
)
from blueprint_pipeline.task_evaluation_scene_configuration_source_preflight import (
    TaskEvaluationSceneConfigurationSourcePreflightError,
    validate_scene_configuration_source_preflight,
)
from tests.test_task_evaluation_scene_configuration_bundle import (
    _bound,
    _envelope,
    _sha256,
)


def _bind_source_evidence(
    envelope_path: Path,
    *,
    exact_target_prim: str,
    stage_three_metric_envelope: dict | None = None,
    expected_target_max_xyz_m: list[float] | None = None,
) -> dict:
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    inputs = envelope_path.parent / "inputs"
    manifest_path = inputs / "source-manifest.json"
    validation_path = inputs / "collision-validation.json"
    appearance = envelope["materialized_references"][0]
    collision = envelope["materialized_references"][1]
    appearance["uri"] = "s3://fixture/source.ply"
    collision["uri"] = "s3://fixture/collision.usda"
    expected = {
        "aabb_min_xyz_m": [0.0, 0.0, 0.0],
        "aabb_max_xyz_m": expected_target_max_xyz_m or [0.2, 0.2, 0.2],
        "point_count": 12,
        "face_count": 20,
    }
    source_object = {
        "publisher_instance_id": "104",
        "aabb_min_xyz_m": [0.0, 0.0, 0.0],
        "aabb_max_xyz_m": [0.2, 0.2, 0.2],
    }
    manifest = {
        "schema_version": "task_evaluation_scene_source_manifest.v1",
        "status": "candidate_source_bytes_retained",
        "scene_id": "839873",
        "publisher_scene_id": "839873",
        "artifacts": [
            {
                "role": "interiorgs_source_splat",
                "sha256": appearance["digest"],
                "size_bytes": appearance["size_bytes"],
            },
            {
                "role": "sage_collision_source",
                "sha256": collision["digest"],
                "size_bytes": collision["size_bytes"],
            },
        ],
        "source_task_object": {
            "publisher_instance_id": "104",
            "source_aabb_min_xyz_m": source_object["aabb_min_xyz_m"],
            "source_aabb_max_xyz_m": source_object["aabb_max_xyz_m"],
        },
        "source_collision_object": {"prim_path": "/Root/Target", **expected},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = {
        "schema_version": "interiorgs_sage_collision_identity.v1",
        "source_files": {
            "sage_collision_usd": {
                "sha256": collision["digest"],
                "size_bytes": collision["size_bytes"],
            }
        },
        "whole_object_collision_identity_passed": True,
        "whole_object_matches": [
            {
                "prim_path": "/Root/Target",
                "point_count": expected["point_count"],
                "face_count": expected["face_count"],
            }
        ],
        "receipt_digest": "",
    }
    validation["receipt_digest"] = canonical_digest(
        validation, digest_field="receipt_digest"
    )
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    manifest_row = _bound(
        manifest_path,
        contract_path="scene.source_manifest",
        uri="s3://fixture/source-manifest.json",
    )
    validation_row = _bound(
        validation_path,
        contract_path="scene.geometry.validation",
        uri="s3://fixture/collision-validation.json",
    )
    envelope["materialized_references"].extend([manifest_row, validation_row])
    envelope["recipe"].update(
        {
            "source_manifest_digest": manifest_row["digest"],
            "scene_identity": {"id": "interiorgs-839873", "version": "v1"},
        }
    )
    envelope["request"] = {
        "scene": {
            "identity": envelope["recipe"]["scene_identity"],
            "source_manifest": {
                key: manifest_row[key] for key in ("uri", "digest", "size_bytes")
            },
            "appearance": {
                "representation": {
                    key: appearance[key] for key in ("uri", "digest", "size_bytes")
                }
            },
            "geometry": {
                "collision": {
                    key: collision[key] for key in ("uri", "digest", "size_bytes")
                },
                "validation": {
                    key: validation_row[key]
                    for key in ("uri", "digest", "size_bytes")
                },
            },
        }
    }
    envelope["render_inputs_result"].update(
        {
            "source_splat_digest": appearance["digest"],
            "source_object_masks": {
                "source_object_identity": {"publisher_instance_id": "104"}
            },
        }
    )
    stage_one_path = Path(
        envelope["stage_configuration_references"][0]["materialized_path"]
    )
    stage_two_path = Path(
        envelope["stage_configuration_references"][1]["materialized_path"]
    )
    stage_three_path = Path(
        envelope["stage_configuration_references"][2]["materialized_path"]
    )
    stage_one = {"source_object": source_object}
    stage_two = {
        "collision_source_digest": collision["digest"],
        "exact_target_prim": exact_target_prim,
        "expected_target": expected,
    }
    stage_three = {
        "metric_envelope": (
            stage_three_metric_envelope
            if stage_three_metric_envelope is not None
            else {
                "minimum_xyz_m": source_object["aabb_min_xyz_m"],
                "maximum_xyz_m": source_object["aabb_max_xyz_m"],
                "maximum_dimension_relative_error": 0.05,
            }
        )
    }
    stage_one_path.write_text(json.dumps(stage_one), encoding="utf-8")
    stage_two_path.write_text(json.dumps(stage_two), encoding="utf-8")
    stage_three_path.write_text(json.dumps(stage_three), encoding="utf-8")
    envelope["recipe"]["stage_sequence"][0][
        "capability"
    ] = "observed_appearance_object_removal"
    envelope["recipe"]["stage_sequence"][1][
        "capability"
    ] = "collision_object_excision"
    envelope["recipe"]["stage_sequence"][2][
        "capability"
    ] = "rigid_replacement_authoring"
    for index, path in (
        (0, stage_one_path),
        (1, stage_two_path),
        (2, stage_three_path),
    ):
        envelope["stage_configuration_references"][index].update(
            {"digest": _sha256(path), "size_bytes": path.stat().st_size}
        )
    envelope["render_inputs_result"]["result_digest"] = canonical_digest(
        envelope["render_inputs_result"], digest_field="result_digest"
    )
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    return {
        "envelope": envelope,
        "configurations": {
            "stage-1": stage_one,
            "stage-2": stage_two,
            "stage-3": stage_three,
        },
    }


def test_bundle_refuses_collision_prim_not_proven_by_bound_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in the exact SAGE prim must fail before a GPU is rented."""

    commit = "a" * 40
    source = tmp_path / "source-binding"
    source.mkdir()
    envelope_path = _envelope(source, commit)
    fixture = _bind_source_evidence(envelope_path, exact_target_prim="/Root/Typo")
    with pytest.raises(
        TaskEvaluationSceneConfigurationSourcePreflightError,
        match="scene_configuration_source_preflight_collision_target_invalid",
    ):
        validate_scene_configuration_source_preflight(**fixture)

    monkeypatch.setattr(
        bundle_module, "validate_immutable_stage_configurations", lambda **_: None
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="scene_configuration_source_preflight_collision_target_invalid",
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=envelope_path,
            toolchain_root=tmp_path / "toolchain-was-not-needed",
            repository_root=tmp_path / "repo-was-not-needed",
            output_root=output,
            expected_source_commit=commit,
        )
    assert not output.exists()


def test_exact_bound_collision_prim_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "source-binding"
    source.mkdir()
    envelope_path = _envelope(source, "a" * 40)
    fixture = _bind_source_evidence(envelope_path, exact_target_prim="/Root/Target")

    validate_scene_configuration_source_preflight(**fixture)


@pytest.mark.parametrize(
    ("stage_three_metric_envelope", "expected_target_max_xyz_m"),
    (
        (
            {
                "minimum_xyz_m": [10.0, 10.0, 10.0],
                "maximum_xyz_m": [11.0, 11.0, 11.0],
                "maximum_dimension_relative_error": 0.05,
            },
            [0.2, 0.2, 0.2],
        ),
        (None, [1.0, 1.0, 1.0]),
    ),
)
def test_bundle_refuses_replacement_envelope_unbound_from_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_three_metric_envelope: dict | None,
    expected_target_max_xyz_m: list[float],
) -> None:
    """A stale replacement envelope must refuse before Content Agents spend."""

    commit = "a" * 40
    source = tmp_path / "source-binding"
    source.mkdir()
    envelope_path = _envelope(source, commit)
    fixture = _bind_source_evidence(
        envelope_path,
        exact_target_prim="/Root/Target",
        stage_three_metric_envelope=stage_three_metric_envelope,
        expected_target_max_xyz_m=expected_target_max_xyz_m,
    )
    with pytest.raises(
        TaskEvaluationSceneConfigurationSourcePreflightError,
        match="scene_configuration_source_preflight_replacement_envelope_invalid",
    ):
        validate_scene_configuration_source_preflight(**fixture)

    monkeypatch.setattr(
        bundle_module, "validate_immutable_stage_configurations", lambda **_: None
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        TaskEvaluationSceneConfigurationBundleError,
        match="scene_configuration_source_preflight_replacement_envelope_invalid",
    ):
        build_scene_configuration_provider_bundle(
            construction_envelope_path=envelope_path,
            toolchain_root=tmp_path / "toolchain-was-not-needed",
            repository_root=tmp_path / "repo-was-not-needed",
            output_root=output,
            expected_source_commit=commit,
        )
    assert not output.exists()
