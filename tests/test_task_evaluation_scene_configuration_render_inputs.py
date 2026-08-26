from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    materialize_scene_configuration_render_inputs,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_renders_derived_views_without_disclosing_raw_splat(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    splat = inputs / "scene.ply"
    splat.write_bytes(b"raw-interiorgs-source")
    source = {
        "artifacts": [
            {
                "role": "interiorgs_source_splat",
                "sha256": _sha256(splat),
                "size_bytes": splat.stat().st_size,
                "splat_count": 1024,
                "provider_upload_allowed": False,
            }
        ]
    }
    source_path = inputs / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    plan = {
        "schema_version": "task_evaluation_renderer_qualification_plan.v1",
        "status": "execute_during_scene_configuration_run",
        "appearance_source": "InteriorGS",
        "browser_preview_qualifies": False,
        "debug_sage_render_qualifies_as_appearance": False,
    }
    plan_path = inputs / "renderer-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def reference(contract_path: str, path: Path) -> dict:
        return {
            "contract_path": contract_path,
            "materialized_path": str(path),
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "full_byte_service_account_readback_passed": True,
        }

    envelope = {
        # The real queue envelope has no top-level run_id -- it lives in the
        # request. Mirroring that here is what makes this fixture able to catch
        # a materializer reading it from the wrong place.
        "request": {
            "run_id": "configure-scene-839873-v1",
            "scene": {"registration": {"metric_registration": {"digest": "sha256:" + "a" * 64}}}
        },
        "materialized_references": [
            reference("scene.appearance.representation", splat),
            reference("scene.source_manifest", source_path),
            reference("scene.appearance.renderer_qualification", plan_path),
        ],
    }
    observed_splat: Path | None = None

    def render(**kwargs):
        nonlocal observed_splat
        observed_splat = Path(kwargs["splat_path"])
        output = Path(kwargs["output_dir"])
        frames = output / "frames"
        frames.mkdir(parents=True)
        rows = []
        for camera in kwargs["cameras"]:
            frame = frames / f"{camera['camera_id']}.png"
            Image.new("RGB", (1024, 1024), color=(90, 80, 70)).save(frame)
            rows.append(
                {
                    "camera_id": camera["camera_id"],
                    "relative_path": f"frames/{frame.name}",
                    "digest": _sha256(frame),
                    "width": 1024,
                    "height": 1024,
                }
            )
        result = {
            "schema_version": "sealed_camera_render_manifest.v1",
            "status": "rendered_exact_cameras",
            "authorization_class": "method_input",
            "splat_digest": kwargs["source_splat_digest"],
            "renders": rows,
            "render_count": len(rows),
            "sealed_camera_render_manifest_digest": "",
        }
        result["sealed_camera_render_manifest_digest"] = canonical_digest(
            result, digest_field="sealed_camera_render_manifest_digest"
        )
        return result

    result = materialize_scene_configuration_render_inputs(
        envelope=envelope,
        stage_one_configuration={
            "schema_version": "observed_appearance_object_removal_configuration.v1",
            "production_render_required": True,
            "source_object": {
                "publisher_instance_id": "104",
                "aabb_min_xyz_m": [2.91, -6.83, 0.754],
                "aabb_max_xyz_m": [3.04, -6.69, 0.884],
            },
            "gaussian_cutout": {
                "selection_rule": ("gaussian_center_inside_registered_source_object_aabb"),
                "aabb_padding_m": 0.0,
                "retained_rows_must_remain_byte_exact": True,
            },
            "required_views": {
                "minimum": 8,
                "lossless_inputs": True,
                "mask_source": "registered_source_object_bounds_projection",
            },
            "provider_disclosure": {
                "raw_interiorgs_bytes": False,
                "derived_rendered_views": True,
            },
            "human_authority": {
                "accepted_by": "project-owner",
                "accepted_on": "2026-08-25",
                "authority_reference": "website-scene-configuration-consent-v1",
                "private_derived_frame_disclosure_authorized": True,
                "provider_retention_terms_accepted": True,
                "provider_training_terms_accepted": True,
                "provider_training_authorized": False,
            },
        },
        output_root=tmp_path / "output",
        renderer=render,
        runtime_resolver=lambda **_kwargs: {
            "node": "/runtime/node",
            "browser_executable": "/runtime/chrome",
            "renderer_root": "/runtime/renderer",
            "identity": {
                "mode": "immutable_host_runtime",
                "runtime_digest": "sha256:" + "d" * 64,
                "source_commit": "e" * 40,
                "full_byte_service_account_readback_passed": True,
            },
        },
        splat_decoder=lambda _source, destination, **_kwargs: (
            write_standard_3dgs_ply(
                SplatData(
                    count=4,
                    xyz=np.asarray(
                        [
                            [2.95, -6.75, 0.82],
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [4.0, -4.0, 0.5],
                        ],
                        dtype=np.float32,
                    ),
                    opacity=np.zeros(4, dtype=np.float32),
                    f_dc=np.zeros((4, 3), dtype=np.float32),
                    scales=np.zeros((4, 3), dtype=np.float32),
                    quats=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32),
                    properties=(),
                ),
                destination,
            )
            and {"status": "completed"}
        ),
    )

    assert observed_splat == splat
    assert result["derived_frame_count"] == 8
    assert result["raw_interiorgs_bytes_in_provider_packet"] is False
    assert result["provider_disclosure_scope"] == "derived_rendered_views_only"
    assert result["renderer_runtime"]["mode"] == "immutable_host_runtime"
    assert result["source_object_masks"]["count"] == 8
    assert result["source_object_masks"]["source_object_identity"] == {
        "publisher_instance_id": "104"
    }
    assert result["source_object_masks"]["observed_segmentation_truth"] is False
    assert result["derived_gaussian_cutout"]["removed_count"] == 1
    assert result["derived_gaussian_cutout"]["retained_count"] == 3
    assert result["derived_gaussian_cutout"]["retained_rows_byte_exact"] is True
    assert all(
        Path(row["source_object_mask"]["path"]).is_file()
        and row["source_object_mask"]["digest"].startswith("sha256:")
        and row["source_object_mask"]["foreground_pixel_count"] > 0
        for row in result["derived_frames"]
    )
    assert all(
        Path(row["path"]).read_bytes() != splat.read_bytes() for row in result["derived_frames"]
    )


def test_rejects_scene_specific_mask_source(tmp_path: Path) -> None:
    try:
        materialize_scene_configuration_render_inputs(
            envelope={},
            stage_one_configuration={
                "schema_version": "observed_appearance_object_removal_configuration.v1",
                "production_render_required": True,
                "source_object": {
                    "publisher_instance_id": "104",
                    "aabb_min_xyz_m": [0.0, 0.0, 0.0],
                    "aabb_max_xyz_m": [1.0, 1.0, 1.0],
                },
                "gaussian_cutout": {
                    "selection_rule": ("gaussian_center_inside_registered_source_object_aabb"),
                    "aabb_padding_m": 0.0,
                    "retained_rows_must_remain_byte_exact": True,
                },
                "required_views": {
                    "minimum": 8,
                    "lossless_inputs": True,
                    "mask_source": ("publisher_instance_104_projected_from_registered_bounds"),
                },
                "provider_disclosure": {
                    "raw_interiorgs_bytes": False,
                    "derived_rendered_views": True,
                },
                "human_authority": {
                    "accepted_by": "project-owner",
                    "accepted_on": "2026-08-25",
                    "authority_reference": "website-scene-configuration-consent-v1",
                    "private_derived_frame_disclosure_authorized": True,
                    "provider_retention_terms_accepted": True,
                    "provider_training_terms_accepted": True,
                    "provider_training_authorized": False,
                },
            },
            output_root=tmp_path / "output",
        )
    except ValueError as exc:
        assert str(exc) == "scene_configuration_render_stage_configuration_invalid"
    else:  # pragma: no cover - fail-closed regression guard
        raise AssertionError("scene-specific mask source was accepted")


def test_rights_admitted_scene_defers_the_render_to_the_provider(tmp_path: Path) -> None:
    """The expensive step moves; nothing else about the packet does.

    When the scene's rights admit uploading source appearance bytes, the
    already-rented configuration GPU renders the same exact cameras. The
    control plane must still produce the cutout and the calibrated camera ring
    -- those are seconds of work and they bind the render -- but it must not
    spend an hour software-rasterising frames it is about to be handed.
    """

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    splat = inputs / "scene.ply"
    splat.write_bytes(b"raw-interiorgs-source")
    source = {
        "artifacts": [
            {
                "role": "interiorgs_source_splat",
                "sha256": _sha256(splat),
                "size_bytes": splat.stat().st_size,
                "splat_count": 1024,
                "provider_upload_allowed": False,
            }
        ]
    }
    source_path = inputs / "source-manifest.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    plan_path = inputs / "renderer-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "task_evaluation_renderer_qualification_plan.v1",
                "status": "execute_during_scene_configuration_run",
                "appearance_source": "InteriorGS",
                "browser_preview_qualifies": False,
                "debug_sage_render_qualifies_as_appearance": False,
            }
        ),
        encoding="utf-8",
    )
    rights_path = inputs / "rights.json"
    rights_path.write_text(
        json.dumps(
            {
                "provider_disclosure": {
                    "raw_interiorgs_downloaded_bytes_may_be_uploaded": True,
                    "provider_training_allowed": False,
                    "public_redistribution_allowed": False,
                    "provider_retention_rule": (
                        "bounded_to_the_exact_run_then_governed_teardown"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    def reference(contract_path: str, path: Path) -> dict:
        return {
            "contract_path": contract_path,
            "materialized_path": str(path),
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "full_byte_service_account_readback_passed": True,
        }

    envelope = {
        # The real queue envelope has no top-level run_id -- it lives in the
        # request. Mirroring that here is what makes this fixture able to catch
        # a materializer reading it from the wrong place.
        "request": {
            "run_id": "configure-scene-839873-v1",
            "scene": {"registration": {"metric_registration": {"digest": "sha256:" + "a" * 64}}}
        },
        "materialized_references": [
            reference("scene.appearance.representation", splat),
            reference("scene.source_manifest", source_path),
            reference("scene.appearance.renderer_qualification", plan_path),
            reference("scene.rights.admission", rights_path),
        ],
    }

    def refuse_to_render(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("control plane rendered a provider-admitted scene")

    result = materialize_scene_configuration_render_inputs(
        envelope=envelope,
        stage_one_configuration={
            "schema_version": "observed_appearance_object_removal_configuration.v1",
            "production_render_required": True,
            "source_object": {
                "publisher_instance_id": "104",
                "aabb_min_xyz_m": [2.91, -6.83, 0.754],
                "aabb_max_xyz_m": [3.04, -6.69, 0.884],
            },
            "gaussian_cutout": {
                "selection_rule": ("gaussian_center_inside_registered_source_object_aabb"),
                "aabb_padding_m": 0.0,
                "retained_rows_must_remain_byte_exact": True,
            },
            "required_views": {
                "minimum": 8,
                "lossless_inputs": True,
                "mask_source": "registered_source_object_bounds_projection",
            },
            "provider_disclosure": {
                "raw_interiorgs_bytes": True,
                "derived_rendered_views": True,
            },
            "human_authority": {
                "accepted_by": "project-owner",
                "accepted_on": "2026-08-26",
                "authority_reference": "website-scene-configuration-consent-v1",
                "private_derived_frame_disclosure_authorized": True,
                "provider_retention_terms_accepted": True,
                "provider_training_terms_accepted": True,
                "provider_training_authorized": False,
            },
        },
        output_root=tmp_path / "output",
        renderer=refuse_to_render,
        runtime_resolver=lambda **_kwargs: {
            "node": "/runtime/node",
            "browser_executable": "/runtime/chrome",
            "renderer_root": "/runtime/renderer",
            "identity": {
                "mode": "immutable_host_runtime",
                "runtime_digest": "sha256:" + "d" * 64,
                "source_commit": "e" * 40,
                "full_byte_service_account_readback_passed": True,
            },
        },
        splat_decoder=lambda _source, destination, **_kwargs: (
            write_standard_3dgs_ply(
                SplatData(
                    count=4,
                    xyz=np.asarray(
                        [
                            [2.95, -6.75, 0.82],
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [4.0, -4.0, 0.5],
                        ],
                        dtype=np.float32,
                    ),
                    opacity=np.zeros(4, dtype=np.float32),
                    f_dc=np.zeros((4, 3), dtype=np.float32),
                    scales=np.zeros((4, 3), dtype=np.float32),
                    quats=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32),
                    properties=(),
                ),
                destination,
            )
            and {"status": "completed"}
        ),
    )

    assert result["status"] == "derived_method_inputs_pending_provider_render"
    assert result["render_execution_site"] == "provider_gpu"
    assert result["provider_render_required"] is True
    assert result["raw_interiorgs_bytes_in_provider_packet"] is True
    assert result["source_splat_bytes_retained_on_control_plane"] is False
    assert result["provider_disclosure_scope"] == (
        "source_appearance_bytes_and_derived_views"
    )
    assert result["render_manifest"] is None
    assert result["derived_frames"] == []
    assert result["derived_frame_count"] == 0
    # The cheap, binding work still happened locally.
    assert Path(result["camera_calibration"]["path"]).is_file()
    assert result["derived_gaussian_cutout"]["removed_count"] == 1
    assert result["derived_gaussian_cutout"]["retained_count"] == 3
    assert result["derived_gaussian_cutout"]["retained_rows_byte_exact"] is True
    assert result["source_appearance"]["digest"] == _sha256(splat)
    assert result["disclosure_decision"]["refusals"] == []
    assert result["result_digest"] == canonical_digest(
        result, digest_field="result_digest"
    )


def test_provider_completes_the_owed_render_into_the_same_packet(tmp_path: Path) -> None:
    """The GPU finishes what the control plane deferred, in the same shape.

    Whatever the rest of the chain consumed before must still be there
    afterwards: frames, masks, a render manifest, and a materialized status --
    so stage one cannot tell whether the pixels were rasterised locally or on
    the provider that already holds the scene.
    """

    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        complete_provider_render_inputs,
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
        resolve_scene_configuration_disclosure,
    )

    splat = tmp_path / "scene.ply"
    splat.write_bytes(b"raw-interiorgs-source")
    calibration = tmp_path / "cameras.json"
    # Derive the ring exactly as the control plane does, so the poses really see
    # the object and the mask projection is exercised for real.
    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        _target_camera_ring,
    )

    ring = _target_camera_ring(
        minimum_xyz=[2.91, -6.83, 0.754], maximum_xyz=[3.04, -6.69, 0.884]
    )
    cameras = [
        {
            "id": row["camera_id"],
            "spec": {
                "pose": {"T_world_camera_opencv": row["T_world_camera_provider_frame"]},
                "intrinsics": row["intrinsics"],
            },
        }
        for row in ring
    ]
    calibration.write_text(json.dumps(cameras), encoding="utf-8")

    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration={
            "provider_disclosure": {
                "raw_interiorgs_bytes": True,
                "derived_rendered_views": True,
            },
            "human_authority": {
                "authority_reference": "operator-2026-08-26",
                "provider_retention_terms_accepted": True,
                "provider_training_authorized": False,
            },
        },
        rights_admission={
            "provider_disclosure": {
                "raw_interiorgs_downloaded_bytes_may_be_uploaded": True,
                "provider_training_allowed": False,
                "public_redistribution_allowed": False,
                "provider_retention_rule": "bounded_then_teardown",
            }
        },
    )
    pending = {
        "schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": "derived_method_inputs_pending_provider_render",
        "source_splat_digest": _sha256(splat),
        "raw_interiorgs_bytes_in_provider_packet": True,
        "disclosure_decision": decision,
        "render_execution_site": "provider_gpu",
        "source_appearance": {"path": str(splat), "digest": _sha256(splat)},
        "camera_calibration": {"path": str(calibration)},
        "source_object_masks": {"source": "registered_source_object_bounds_projection"},
        "derived_frames": [],
        "derived_frame_count": 0,
        "render_manifest": None,
        "result_digest": "sha256:" + "c" * 64,
    }

    observed_backend: dict[str, str] = {}

    def render(**kwargs):
        observed_backend["graphics_backend"] = kwargs["graphics_backend"]
        output = Path(kwargs["output_dir"])
        frames = output / "frames"
        frames.mkdir(parents=True)
        rows = []
        for camera in kwargs["cameras"]:
            frame = frames / f"{camera['camera_id']}.png"
            Image.new("RGB", (1024, 1024), color=(12, 34, 56)).save(frame)
            rows.append(
                {
                    "camera_id": camera["camera_id"],
                    "relative_path": f"frames/{frame.name}",
                    "digest": _sha256(frame),
                    "width": 1024,
                    "height": 1024,
                }
            )
        result = {
            "schema_version": "sealed_camera_render_manifest.v1",
            "status": "rendered_exact_cameras",
            "authorization_class": "method_input",
            "splat_digest": kwargs["source_splat_digest"],
            "renders": rows,
            "render_count": len(rows),
            "sealed_camera_render_manifest_digest": "",
        }
        result["sealed_camera_render_manifest_digest"] = canonical_digest(
            result, digest_field="sealed_camera_render_manifest_digest"
        )
        return result

    completed = complete_provider_render_inputs(
        render_inputs=pending,
        appearance_path=splat,
        source_object={
            "aabb_min_xyz_m": [2.91, -6.83, 0.754],
            "aabb_max_xyz_m": [3.04, -6.69, 0.884],
        },
        output_root=tmp_path / "provider-out",
        renderer=render,
        runtime_resolver=lambda **_kwargs: {
            "node": "/runtime/node",
            "browser_executable": "/runtime/chrome",
            "renderer_root": "/runtime/renderer",
            "identity": {"mode": "provider_runtime"},
        },
    )

    # A real GPU is demanded; the renderer refuses software fallback itself.
    assert observed_backend["graphics_backend"] == "egl"
    assert completed["status"] == "derived_method_inputs_materialized"
    assert completed["derived_frame_count"] == 8
    assert completed["render_completed_on_provider"] is True
    assert completed["render_manifest"]["manifest_digest"].startswith("sha256:")
    assert completed["control_plane_result_digest"] == pending["result_digest"]
    assert completed["result_digest"] == canonical_digest(
        completed, digest_field="result_digest"
    )
    assert all(
        Path(row["source_object_mask"]["path"]).is_file()
        for row in completed["derived_frames"]
    )
    # The completed packet is coherent for every downstream consumer.
    from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
        render_inputs_disclosure_is_coherent,
    )

    assert render_inputs_disclosure_is_coherent(pending)
    assert render_inputs_disclosure_is_coherent(
        {**completed, "raw_interiorgs_bytes_in_provider_packet": False}
    )


def test_provider_completion_refuses_an_unauthorized_or_swapped_source(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        TaskEvaluationSceneConfigurationRenderInputsError,
        complete_provider_render_inputs,
    )

    splat = tmp_path / "scene.ply"
    splat.write_bytes(b"real-source")

    def never(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("rendered without authorization")

    with pytest.raises(TaskEvaluationSceneConfigurationRenderInputsError):
        complete_provider_render_inputs(
            render_inputs={"status": "derived_method_inputs_materialized"},
            appearance_path=splat,
            source_object={},
            output_root=tmp_path / "a",
            renderer=never,
        )

    with pytest.raises(TaskEvaluationSceneConfigurationRenderInputsError):
        complete_provider_render_inputs(
            render_inputs={
                "status": "derived_method_inputs_pending_provider_render",
                "disclosure_decision": {},
            },
            appearance_path=splat,
            source_object={},
            output_root=tmp_path / "b",
            renderer=never,
        )
