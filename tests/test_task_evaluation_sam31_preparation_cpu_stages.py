from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline import task_evaluation_sam31_preparation_cpu_stages as module
from tests.test_public_scene_removal_selection import _source_fixture
from tests.test_task_evaluation_scene_configuration_submission import SHA
from tests.test_sam31_source_track_provider import _request as sam_request


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": module.sha(path), "size_bytes": path.stat().st_size}


def _fixture(root: Path) -> dict:
    fixture = _source_fixture(root)
    runtime, repo = root / "runtime", root / "repo"
    runtime.mkdir()
    repo.mkdir()
    task = json.loads(fixture["task_request"].read_text())
    task["subject"]["review_label"] = "configured open notebook"
    fixture["task_request"].write_text(canonical_json(task))
    return {
        "request": {"expected_production_commit": SHA},
        "plan": {
            "source_commit": SHA,
            "host_inputs": {
                name: _record(fixture[key]) for name, key in (
                    ("task_request", "task_request"), ("installation_receipt", "installation_receipt"),
                    ("publisher_intake", "publisher_intake"),
                    ("source_preparation_receipt", "source_preparation"),
                )
            } | {"interiorgs_terms": _record(fixture["rights_evidence"]["interiorgs_terms"])},
            "camera_policy": {
                "generator": "translated_target_coverage_v1", "orbit_only_forbidden": True,
                "views": [{"camera_id": f"view-{index:02d}",
                           "position_offset_m": [0.03 * index, 0.7 + 0.05 * index, 0.3 + 0.03 * index],
                           "target_offset_m": [0.0, 0.0, 0.0]} for index in range(16)],
            },
            "rendering": {"width": 1280, "height": 1280},
        },
        "inputs": {}, "server_data_root": str(root), "runtime_root": str(runtime),
        "repo_root": str(repo),
    }


def test_cpu_stages_reuse_typed_producers_and_never_invoke_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _fixture(tmp_path)
    calls = []

    def convert(**kwargs):
        request = json.loads(Path(kwargs["request_path"]).read_text())
        assert request["rights"]["raw_private_upload_authorized"] is False
        assert request["rights"]["conversion_execution_location"] == "local_only"
        assert kwargs["production_runtime_root"] == Path(job["runtime_root"])
        output = Path(kwargs["output_root"])
        output.mkdir()
        ply = output / request["output_filename"]
        ply.write_bytes(b"synthetic-standard-ply")
        receipt = {"schema_version": "standard_splat_conversion_receipt.v1",
                   "status": "standard_splat_conversion_materialized", "raw_source_uploaded": False,
                   "output": {"relative_path": ply.name, "sha256": module.sha(ply),
                              "size_bytes": ply.stat().st_size}}
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
            cameras.append({
                "camera_id": view["camera_id"],
                "T_world_camera_provider_frame": [[1., 0., 0., float(index)], [0., 1., 0., 0.],
                                                  [0., 0., 1., 0.], [0., 0., 0., 1.]],
                "intrinsics": {"model": "PINHOLE", "fx": 4., "fy": 4., "cx": 4., "cy": 3.,
                               "width": 8, "height": 6},
            })
        camera_path = output / "cameras.v1.json"
        camera_path.write_text(canonical_json(cameras))
        receipt = {
            "schema_version": "public_scene_interiorgs_edit_input_receipt.v2",
            "status": "render_derived_input_packet_materialized",
            "scene": {"task_id": selection["task_id"],
                      "target_instance_id": selection["source_object"]["instance_id"]},
            "source_admission": {"adapter": module.ADAPTER,
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


def test_cpu_stage_rejects_output_outside_operator_root_before_writes(tmp_path: Path) -> None:
    job = _fixture(tmp_path)
    with pytest.raises(module.Sam31PreparationCPUStageError, match="path_outside_server_data_root"):
        module.execute_cpu_stage({**job, "stage_id": "source_selections",
                                  "output_root": str(tmp_path.parent / "not-authorized")})


def test_cpu_stage_rejects_changed_host_bytes_before_producers(tmp_path: Path) -> None:
    job = _fixture(tmp_path)
    Path(job["plan"]["host_inputs"]["task_request"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input_bytes_mismatch"):
        module.execute_cpu_stage({**job, "stage_id": "source_selections",
                                  "output_root": str(tmp_path / "not-created")})
    assert not (tmp_path / "not-created").exists()


def test_calibrated_producer_forwards_verified_published_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import public_scene_inpainting_inputs as inputs
    from tests.test_public_scene_inpainting_inputs import _write_v2_fixture

    paths = _write_v2_fixture(tmp_path)
    published = paths["data"] / "published-runtime"
    identity = {"runtime_digest": "sha256:" + "f" * 64}
    calls = []

    def validate(**kwargs):
        assert kwargs == {"runtime_root": published, "repo_root": paths["repo"]}
        return {"node": "/verified/node", "browser_executable": "/verified/browser",
                "renderer_root": "/verified/renderer", "identity": identity}

    def renderer(**kwargs):
        calls.append(kwargs)
        assert kwargs["node"] == "/verified/node"
        assert kwargs["browser_executable"] == "/verified/browser"
        assert kwargs["renderer_runtime_root"] == "/verified/renderer"
        assert kwargs["renderer_runtime_identity"] == identity
        assert kwargs["authorization_class"] == "method_input"
        raise RuntimeError("synthetic-render-boundary-no-real-render")

    monkeypatch.setattr(inputs, "validate_splat_render_runtime", validate)
    monkeypatch.setattr(inputs, "render_splat_at_exact_cameras", renderer)
    with pytest.raises(RuntimeError, match="synthetic-render-boundary-no-real-render"):
        inputs.materialize_public_scene_inpainting_inputs(
            request_path=paths["repo"] / "request.json", repo_root=paths["repo"],
            data_root=paths["data"], output_root=paths["output"], production_runtime_root=published,
        )
    assert len(calls) == 1
