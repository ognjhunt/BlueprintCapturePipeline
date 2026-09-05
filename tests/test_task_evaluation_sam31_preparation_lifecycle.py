"""No-paid-resource rehearsal of the fresh scene source-preparation contract chain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import task_evaluation_sam31_preparation_cpu_stages as module
from blueprint_pipeline import task_evaluation_sam31_preparation_review_stages as reviews
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.scene_placement.sam31_source_track_provider import execute_sam31_source_track_request
from blueprint_pipeline.scene_placement.semantic_source_track_import import import_semantic_source_tracks
from tests.test_task_evaluation_sam31_preparation_cpu_stages import _fixture, _record
from tests.test_sam31_source_track_provider import _request as sam_request, FakePredictor
from tests.test_public_scene_calibrated_object_masks import _run_production_ai_review
from tests.test_adp009d_gaussian_excision_audit import POLICY
from tests.test_task_evaluation_scene_configuration_submission import SHA


def _write(path, value, digest_field=None):
    if digest_field:
        value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))
    return path


def test_real_cpu_sam_review_mask_freeze_cutout_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _fixture(tmp_path)
    task_path = Path(job["plan"]["host_inputs"]["task_request"]["path"])
    task_value = json.loads(task_path.read_text())
    task_value["human_authority"] = {
        "accepted_by": "nijelhunt_1", "accepted_on": "2026-09-05",
        "authority_reference": "hermetic-current-task-authority",
        "sam31_visual_review_authorized": True, "sam31_visual_review_maximum_cost_usd": 1.,
        "private_derived_frame_disclosure_authorized": True,
        "provider_retention_terms_accepted": True, "provider_training_terms_accepted": True,
        "provider_training_authorized": False,
    }
    _write(task_path, task_value, "request_digest")
    job["plan"]["host_inputs"]["task_request"] = _record(task_path)
    preparation = Path(job["plan"]["host_inputs"]["source_preparation_receipt"]["path"])
    prepared_value = json.loads(preparation.read_text())
    frame_path = preparation.parent / "shared_frame_candidate.json"
    frame = json.loads(frame_path.read_text())
    frame.update(provider_transform={"source_to_collision": "identity"},
                 shared_frame_status="provider_declared_not_independently_validated",
                 metric_scale_status="provider_declared_not_independently_validated")
    _write(frame_path, frame, "receipt_digest")
    for row in prepared_value["artifacts"]:
        path = preparation.parent / row["relative_path"]
        row.update(sha256=module.sha(path), size_bytes=path.stat().st_size)
    _write(preparation, prepared_value, "receipt_digest")
    job["plan"]["host_inputs"]["source_preparation_receipt"] = _record(preparation)
    calls = []

    def convert(**kwargs):
        request = json.loads(Path(kwargs["request_path"]).read_text())
        assert request["rights"]["raw_private_upload_authorized"] is False
        assert request["rights"]["conversion_execution_location"] == "local_only"
        assert kwargs["production_runtime_root"] == Path(job["runtime_root"])
        output = Path(kwargs["output_root"])
        output.mkdir()
        ply = output / request["output_filename"]
        from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
        xyz = np.array([[-2.05, -3.50, .285], [-1.95, -3.50, .285],
                        [-1.95, -3.40, .285], [-2.05, -3.40, .285],
                        [-2., -3., .25], [-2., -4., .25]], dtype=np.float32)
        write_standard_3dgs_ply(SplatData(count=6, xyz=xyz, opacity=np.ones(6, dtype=np.float32),
            f_dc=np.zeros((6,3), dtype=np.float32), scales=np.full((6,3), -6., dtype=np.float32),
            quats=np.tile(np.array([[1.,0.,0.,0.]], dtype=np.float32), (6,1)), properties=()), ply)
        receipt = {"schema_version": "standard_splat_conversion_receipt.v1",
                   "status": "standard_splat_conversion_materialized", "raw_source_uploaded": False,
                   "source": _record(Path(job["inputs"]["source_appearance"]["path"])),
                   "output": {"relative_path": ply.name, "sha256": module.sha(ply),
                              "size_bytes": ply.stat().st_size,
                              "standard_3dgs_schema_validated": True, "gaussian_count_preserved": True}}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        (output / "standard_splat_conversion_receipt.v1.json").write_text(canonical_json(receipt))
        calls.append("conversion")
        return receipt

    def render(**kwargs):
        request = json.loads(Path(kwargs["request_path"]).read_text())
        assert request["scene"]["source_adapter"] == module.ADAPTER
        assert len(request["camera_policy"]["views"]) == 16
        assert kwargs["production_runtime_root"] == Path(job["runtime_root"])
        selection = json.loads(Path(request["scene"]["task_freeze_path"]).read_text())
        output = Path(kwargs["output_root"])
        output.mkdir()
        cameras, images = [], []
        for index, view in enumerate(request["camera_policy"]["views"]):
            path = output / (view["camera_id"] + ".png")
            Image.new("RGB", (8, 6), (index + 20, 30, 40)).save(path)
            images.append({"camera_id": view["camera_id"], "relative_path": path.name,
                           "sha256": module.sha(path), "size_bytes": path.stat().st_size})
            angle = np.deg2rad(index - 7.5)
            cameras.append({
                "camera_id": view["camera_id"],
                "T_world_camera_provider_frame": [[float(np.cos(angle)), 0., float(np.sin(angle)), -2. + .01 * (index - 7.5)], [0., 1., 0., -3.44],
                                                  [-float(np.sin(angle)), 0., float(np.cos(angle)), -1.], [0., 0., 0., 1.]],
                "intrinsics": {"model": "PINHOLE", "fx": 4., "fy": 4., "cx": 4., "cy": 3.,
                               "width": 8, "height": 6},
            })
        camera_path = output / "cameras.v1.json"
        camera_path.write_text(canonical_json(cameras))
        sealed_path = output / "images" / "sealed_camera_render_manifest.v1.json"
        sealed = {"schema_version": "sealed_camera_render_manifest.v1", "status": "rendered_exact_cameras",
            "authorization_class": "method_input", "splat_digest": module.sha(Path(request["scene"]["standard_splat_path"])),
            "calibrated_camera_file": {"digest": module.sha(camera_path), "binding": "caller_file_exact_match"},
            "render_count": 16, "renders": [{"camera_id": row["camera_id"], "digest": row["sha256"]} for row in images]}
        _write(sealed_path, sealed, "sealed_camera_render_manifest_digest")
        conversion = json.loads(Path(request["scene"]["standard_splat_conversion_receipt_path"]).read_text())
        receipt = {
            "schema_version": "public_scene_interiorgs_edit_input_receipt.v2",
            "status": "render_derived_input_packet_materialized",
            "scene": {"task_id": selection["task_id"], "publisher_scene_id": "841757",
                      "target_instance_id": selection["source_object"]["instance_id"]},
            "renderer": {"authorization_class": "method_input", "purpose_bound": True,
                         "render_manifest_digests": {"images": sealed["sealed_camera_render_manifest_digest"]}},
            "source_admission": {"adapter": module.ADAPTER,
                                 "standard_splat_conversion_receipt_digest": conversion["receipt_digest"],
                                 "task_freeze_digest": selection["task_freeze_digest"],
                                 "scene_freeze_digest": selection["scene_freeze_digest"]},
            "derived_artifacts": {"cameras": {"relative_path": camera_path.name,
                                             "sha256": module.sha(camera_path),
                                             "size_bytes": camera_path.stat().st_size},
                                  "images": images},
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        (output / "public_scene_interiorgs_edit_input_receipt.v2.json").write_text(canonical_json(receipt))
        calls.append("calibration")
        return receipt

    def encode(*, output_path: Path, **kwargs):
        output_path.write_bytes(b"synthetic-ffv1-exact-input-sequence")
        return ["fixture-ffv1"]

    monkeypatch.setattr(module, "materialize_standard_splat_conversion", convert)
    monkeypatch.setattr(module, "materialize_public_scene_inpainting_inputs", render)
    monkeypatch.setattr("blueprint_pipeline.public_scene_sam31_task_inputs._encode_lossless_sequence", encode)
    profile = tmp_path / "sam-profile.json"
    profile.write_text(canonical_json(sam_request()["provider_profile"]))
    ffmpeg = tmp_path / "ffmpeg-fixture"
    ffmpeg.write_bytes(b"fixture-not-executed")
    job["ffmpeg_executable"] = str(ffmpeg)
    job["inputs"]["sam31_provider_profile"] = _record(profile)
    for index, stage in enumerate(("source_selections", "standard_splat_conversion",
                                   "calibrated_views", "sam31_inputs")):
        result = module.execute_cpu_stage({**job, "stage_id": stage,
                                          "output_root": str(tmp_path / f"stage-{index}")})
        assert result["status"] == "completed"
        assert result["provider_mutation_performed"] is False
        assert result["candidate_policy_queried"] is False
        for record in result["artifacts"].values():
            path = Path(record["path"])
            assert record["sha256"] == module.sha(path)
            assert record["size_bytes"] == path.stat().st_size
        job["inputs"].update(result["artifacts"])
    packet = json.loads(Path(job["inputs"]["sam31_task_input_packet"]["path"]).read_text())
    request = json.loads(Path(job["inputs"]["sam31_run_request"]["path"]).read_text())
    assert packet["camera_count"] == 16
    assert len({row["source_frame_digest"] for row in request["frame_registry"]}) == 16
    assert request["prompts"][0]["text"] == "configured open notebook"
    assert packet["paid_execution_started"] is False
    assert calls == ["conversion", "calibration"]


    # SAM's expensive predictor is replaced; request validation, multiplex API,
    # encoded masks, and the canonical source-track importer are all real.
    frames = tmp_path / "predictor-frames"
    frames.mkdir()
    outputs = {}
    for index, row in enumerate(request["frame_artifacts"]):
        (frames / f"{index:06d}.jpg").write_bytes(Path(row["path"]).read_bytes())
        mask = np.zeros((1, 6, 8), dtype=bool)
        mask[:, 2:4, 2:6] = True
        if index == 0:
            mask[:, :, 3:] = False
        outputs[index] = {"out_obj_ids": np.array([1]), "out_probs": np.array([0.99]),
                          "out_binary_masks": mask}
    predictor = FakePredictor(outputs)
    tracked = execute_sam31_source_track_request(request,
        predictor_factory=lambda profile: predictor, materialized_frame_directory=frames)
    assert tracked["status"] == "completed", tracked
    tracks = import_semantic_source_tracks(tracked["source_track_import_request"], tracked["provider_result"])
    assert tracks["status"] == "completed", tracks
    tracks_path = _write(tmp_path / "sam-results" / "source-tracks.json", tracks)
    job["inputs"]["sam31_source_tracks"] = _record(tracks_path)

    from blueprint_pipeline.task_evaluation_sam31_preparation_review_authority import (
        TERMS, materialize_sam31_review_authority,
    )
    from blueprint_pipeline.public_scene_sam31_track_selection_review import AI_RIGHTS_SCHEMA_VERSION
    terms = _write(tmp_path / "terms.json", {
        "schema_version": AI_RIGHTS_SCHEMA_VERSION, "status": "accepted_for_private_derived_visual_review",
        **TERMS, "source_candidate_digest": "sha256:" + "a" * 64,
        "review_media_digest": "sha256:" + "b" * 64,
        "accepted_by": "nijelhunt_1", "accepted_on": "2026-09-05",
        "human_authority_reference": "fixture-terms-only",
    }, "attestation_digest")
    standing = tmp_path / "standing-review.json"
    materialize_sam31_review_authority(task_request_path=task_path,
        provider_terms_evidence_path=terms, output_path=standing)
    scope = _write(tmp_path / "sdk-scope.json", {"fixture": "checked-only-before-stubbed-provider"})
    admin = tmp_path / "admin-secret"
    admin.write_text("fixture-only")
    admin.chmod(0o600)
    profile = {"source_commit": SHA, "gaussian_excision_policy": POLICY,
               "sam31_visual_review": {"rights_attestation": _record(standing),
                 "openai_cost_scope_attestation": _record(scope), "openai_admin_api_key_file": str(admin),
                 "openai_project_id": "fixture", "openai_api_key_id": "fixture"}}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    job["server_profile"] = profile

    def sdk_boundary(**kwargs):
        # Reuse real SDK execution/receipt/official-cost contracts, with only
        # SDK transport and cost-API transport replaced by deterministic data.
        result, _ = _run_production_ai_review(monkeypatch=monkeypatch,
            candidate_path=kwargs["candidate_path"], output_root=kwargs["output_root"], decision="accepted")
        return result

    monkeypatch.setattr(reviews, "run_sam31_ai_visual_review", sdk_boundary)
    for index, stage in enumerate(("sam31_review", "calibrated_masks"), start=5):
        outcome = reviews.execute_review_stage({**job, "stage_id": stage,
            "output_root": str(tmp_path / f"stage-{index}")})
        assert outcome["status"] == "completed", outcome
        job["inputs"].update(outcome["artifacts"])
    masks = json.loads(Path(job["inputs"]["calibrated_mask_set"]["path"]).read_text())
    assert masks["selection_authority"]["all_selected_tracks_ai_visual_review_accepted"] is True
    assert masks["selection_authority"]["all_selected_tracks_human_review_accepted"] is False
    assert len(masks["tasks"][0]["masks"]) == 16

    # Substitute the native USD reader at its I/O boundary. Every geometric
    # projection, mask binding, freeze, and retained-PLY operation remains real.
    from tests.test_task_evaluation_scene_configuration_submission import BOOK_CORNERS, BOOK_PRIM
    vertices = np.array([[row[k] for k in ("x", "y", "z")] for row in BOOK_CORNERS])
    faces = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                      [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])

    def native_mesh(path, prim):
        assert prim == BOOK_PRIM
        assert path == Path(job["inputs"]["source_collision"]["path"])
        return vertices, faces, {"collision_stage_meters_per_unit": 1., "up_axis": "Z",
            "target_point_count": len(vertices), "target_triangle_count": len(faces),
            "target_world_aabb_min_m": vertices.min(axis=0).tolist(),
            "target_world_aabb_max_m": vertices.max(axis=0).tolist()}

    monkeypatch.setattr("blueprint_pipeline.public_scene_gaussian_excision_audit._load_target_mesh", native_mesh)
    real_freezes = reviews.materialize_fresh_scene_removal_freezes

    def freezes_with_diagnostics(**kwargs):
        try:
            return real_freezes(**kwargs)
        except Exception as exc:
            pytest.fail(f"Real freeze materializer failed: {exc}")

    monkeypatch.setattr(reviews, "materialize_fresh_scene_removal_freezes", freezes_with_diagnostics)
    outcome = reviews.execute_review_stage({**job, "stage_id": "removal_freezes",
        "output_root": str(tmp_path / "stage-7")})
    assert outcome["status"] == "completed", outcome
    job["inputs"].update(outcome["artifacts"])
    sweep = json.loads(Path(job["inputs"]["segment_sweep_freeze"]["path"]).read_text())
    assert sweep["camera_split"]["camera_count"] == 16
    assert sweep["camera_split"]["heldout_camera_ids"] == []
    assert any(row["registered_core_added_pixel_count"] > 0 for row in sweep["masks"])

    # GPU contribution execution is replaced by deterministic numeric arrays;
    # the production cutout materializer must reopen their exact bytes.
    from blueprint_pipeline.public_scene_gaussian_excision_audit import CONTRIBUTION_CLASS_ORDER, CONTRIBUTION_EVIDENCE_SCHEMA
    contribution_root = tmp_path / "contribution-runtime"
    contribution_root.mkdir()
    repetition_rows = []
    for repeat in range(2):
        evidence = np.zeros((16, len(CONTRIBUTION_CLASS_ORDER), 6), dtype=np.float32)
        evidence[:, 1, :4] = .25
        evidence[:, 0, 4:] = .25
        path = contribution_root / f"contribution_repetition_{repeat}.npz"
        np.savez_compressed(path, per_view_class_contribution=evidence)
        repetition_rows.append({"relative_path": path.name, "sha256": module.sha(path),
                                "size_bytes": path.stat().st_size})
    manifest = _write(contribution_root / "contributions.json", {
        "schema_version": CONTRIBUTION_EVIDENCE_SCHEMA, "freeze_digest": sweep["freeze_digest"],
        "class_order": list(CONTRIBUTION_CLASS_ORDER),
        "camera_ids": sweep["camera_split"]["calibration_camera_ids"],
        "method": {**sweep["contribution_method"], "released_code_executed": True},
        "repetitions": repetition_rows, "heldout_cameras_accessed_for_classification": False,
    }, "manifest_digest")
    job["inputs"]["gaussian_contribution_evidence"] = _record(manifest)
    outcome = reviews.execute_review_stage({**job, "stage_id": "segment_cutout",
        "output_root": str(tmp_path / "stage-9")})
    assert outcome["status"] == "completed", outcome
    cutout = json.loads(Path(outcome["artifacts"]["segment_cutout_set"]["path"]).read_text())
    assert cutout["shared_scene_union"]["counts"] == {"source": 6, "deleted_total": 4, "retained_total": 2}

    # The final production consumer must accept the real freeze's SAM lineage,
    # including no legacy render_input_packet/AABB-mask join.
    from tests.test_task_evaluation_scene_configuration_sam31_inputs import _consume
    from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import SAM31_SELECTION_RULE, SAM31_MASK_SOURCE
    job["inputs"].update(outcome["artifacts"])
    evidence = {name: job["inputs"][name] for name in (
        "selection_inputs", "track_selection_review", "calibrated_mask_set", "segment_cutout_set")}
    evidence["standard_splat_conversion"] = job["inputs"]["standard_splat_conversion_receipt"]
    config = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "production_render_required": True, "sam31_review_kind": "ai",
        "source_object": {"scene_id": "841757", "publisher_instance_id": "115",
                          "collision_candidate_prim": BOOK_PRIM},
        "gaussian_cutout": {"selection_rule": SAM31_SELECTION_RULE, "retained_rows_must_remain_byte_exact": True},
        "required_views": {"minimum": 16, "lossless_inputs": True, "mask_source": SAM31_MASK_SOURCE},
        "provider_disclosure": {"raw_interiorgs_bytes": False, "derived_rendered_views": True},
        "human_authority": task_value["human_authority"],
        "sam31_exact_mask_evidence": {name + "_digest": json.loads(Path(row["path"]).read_text())["receipt_digest"]
                                     for name, row in evidence.items()},
    }
    rights = _write(tmp_path / "processing-rights.json", {"private_provider_processing_allowed": True,
        "provider_training_allowed": False, "public_redistribution_allowed": False})
    raw = Path(job["inputs"]["source_appearance"]["path"])
    envelope = {"request": {"run_id": "hermetic-source-preparation"}, "sam31_exact_mask_inputs": evidence,
        "materialized_references": [{"contract_path": contract, "materialized_path": str(path),
            "digest": module.sha(path), "size_bytes": path.stat().st_size}
            for contract, path in (("scene.appearance.representation", raw), ("scene.rights.admission", rights))]}
    result = _consume(envelope, config, tmp_path / "final-render-inputs")
    assert result["derived_frame_count"] == 16
    assert result["derived_gaussian_cutout"]["removed_count"] == 4
    assert result["derived_gaussian_cutout"]["retained_rows_byte_exact"] is True
    assert result["raw_interiorgs_bytes_in_provider_packet"] is False
