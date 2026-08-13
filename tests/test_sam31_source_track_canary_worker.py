import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.common import write_json
from blueprint_pipeline.sam31_source_track_canary_worker import (
    BUNDLE_MANIFEST_SCHEMA_VERSION,
    Sam31CanaryWorkerError,
    _safe_extract_bundle,
    build_sam31_source_track_input_bundle,
    run_sam31_source_track_canary_worker,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)


SHA = "sha256:" + "a" * 64


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _request(tmp_path: Path) -> dict:
    frames = []
    artifacts = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", (4, 2), color=color).save(path, format="JPEG")
        digest = _sha(path)
        frames.append(
            {
                "source_frame_id": f"frame-{index}",
                "model_frame_index": index,
                "source_frame_digest": digest,
                "retained_video_digest": SHA,
                "decoded_pts_seconds": float(index + 1),
                "sync_map_row_digest": SHA,
                "camera_record_digest": SHA,
                "encoder_retained": True,
                "width": 4,
                "height": 2,
                "analysis_jpeg_digest": digest,
            }
        )
        artifacts.append(
            {
                "source_frame_id": f"frame-{index}",
                "path": str(path.resolve()),
                "media_type": "image/jpeg",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "semantic_sam31_source_track_run_request.v1",
        "bindings": {
            "capture_digest": SHA,
            "retained_video_digest": SHA,
            "camera_solution_digest": SHA,
            "frame_registry_digest": canonical_json_digest(frames),
        },
        "frame_registry": frames,
        "frame_artifacts": artifacts,
        "provider_profile": {"profile_digest": SHA},
        "prompts": [{"prompt_id": "chair", "text": "chair", "output_label": "chair"}],
        "allowed_evidence_uses": ["semantic_analysis"],
    }


def _build(tmp_path: Path, *, suffix: str = "") -> tuple[Path, Path, dict]:
    request_path = tmp_path / f"request{suffix}.json"
    request_path.write_text(json.dumps(_request(tmp_path), sort_keys=True), encoding="utf-8")
    bundle = tmp_path / f"bundle{suffix}.zip"
    receipt = tmp_path / f"receipt{suffix}.json"
    result = build_sam31_source_track_input_bundle(
        request_path=request_path, bundle_path=bundle, receipt_path=receipt
    )
    return bundle, receipt, result


def test_bundle_is_deterministic_path_portable_and_source_bound(tmp_path: Path) -> None:
    first, _, first_receipt = _build(tmp_path, suffix="-a")
    second, _, second_receipt = _build(tmp_path, suffix="-b")
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["bundle"]["sha256"] == second_receipt["bundle"]["sha256"]
    assert first_receipt["frame_count"] == 2
    assert first_receipt["source_frame_bytes_returned_by_worker"] is False
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "request.json",
            "frames/000000.jpg",
            "frames/000001.jpg",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        request = json.loads(archive.read("request.json"))
    assert manifest["schema_version"] == BUNDLE_MANIFEST_SCHEMA_VERSION
    assert manifest["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert request["frame_artifacts"][0]["path"] == "frames/000000.jpg"
    assert request["frame_artifacts"][1]["path"] == "frames/000001.jpg"


def test_bundle_builder_rejects_tampered_frame(tmp_path: Path) -> None:
    request = _request(tmp_path)
    Path(request["frame_artifacts"][0]["path"]).write_bytes(b"not-a-jpeg")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(Sam31CanaryWorkerError, match="frame_artifact_size_invalid"):
        build_sam31_source_track_input_bundle(
            request_path=request_path,
            bundle_path=tmp_path / "bundle.zip",
            receipt_path=tmp_path / "receipt.json",
        )


def test_safe_extract_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("request.json", "{}")
        archive.writestr("../escape.jpg", b"x")
    with pytest.raises(Sam31CanaryWorkerError, match="input_bundle_member_unsafe"):
        _safe_extract_bundle(traversal, tmp_path / "traversal-out")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("frames/000000.jpg")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("request.json", "{}")
        archive.writestr(info, "target")
    with pytest.raises(Sam31CanaryWorkerError, match="input_bundle_member_unsafe"):
        _safe_extract_bundle(symlink, tmp_path / "symlink-out")


def _runtime_environment(monkeypatch: pytest.MonkeyPatch, *, bundle: Path, request_digest: str):
    values = {
        "BLUEPRINT_SAM31_CANARY_REQUEST_DIGEST": SHA,
        "BLUEPRINT_SAM31_BOUND_REQUEST_DIGEST": "sha256:" + "b" * 64,
        "BLUEPRINT_CONTAINER_IMAGE_DIGEST": "registry.test/sam31@sha256:" + "c" * 64,
        "BLUEPRINT_SAM31_INPUT_BUNDLE_DIGEST": _sha(bundle),
        "BLUEPRINT_SAM31_SOURCE_TRACK_REQUEST_DIGEST": request_digest,
        "BLUEPRINT_SAM31_EXPECTED_CHECKPOINT_DIGEST": "sha256:" + "d" * 64,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_worker_returns_only_bound_semantic_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, receipt = _build(tmp_path)
    _runtime_environment(
        monkeypatch,
        bundle=bundle,
        request_digest=receipt["source_track_run_request_digest"],
    )

    def fake_stage(**kwargs):
        provider = {
            "schema_version": "semantic_source_track_provider_result.v1",
            "tracks": [],
        }
        import_request = {"schema_version": "semantic_source_track_import_request.v1"}
        write_json(Path(kwargs["provider_result_path"]), provider)
        write_json(Path(kwargs["import_request_path"]), import_request)
        return {
            "schema_version": "semantic_sam31_source_track_run_result.v1",
            "status": "abstained",
            "blockers": [],
            "warnings": ["sam31_returned_no_tracks"],
            "metric_box_ready": False,
            "collision_ready": False,
            "physics_ready": False,
            "physical_task_success_established": False,
            "model_self_grading_permitted": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker.run_sam31_source_track_stage",
        fake_stage,
    )
    normalized = {
        "schema_version": "semantic_source_track_import_result.v1",
        "status": "abstained",
        "bindings": {},
        "track_registry": [],
        "frame_masks": [],
        "blockers": [],
        "warnings": ["provider_returned_no_tracks"],
        "claim_ceiling": "no_source_tracks_detected",
        "result_digest": "",
    }
    normalized["result_digest"] = canonical_json_digest(
        {key: value for key, value in normalized.items() if key != "result_digest"}
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker.import_semantic_source_tracks",
        lambda *_args, **_kwargs: normalized,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker._runtime_facts",
        lambda: {
            "torch_version": "fixture",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "fixture-gpu",
        },
    )
    result = run_sam31_source_track_canary_worker(
        input_bundle=bundle, output_path=tmp_path / "result.json"
    )
    assert result["status"] == "passed"
    assert result["stage_run_result"]["status"] == "abstained"
    assert result["source_frame_bytes_returned"] is False
    assert result["normalized_source_tracks"] == normalized
    assert result["metric_box_ready"] is False
    assert result["physics_ready"] is False
    assert result["physical_task_success_established"] is False
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"


def test_worker_fails_closed_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, _, receipt = _build(tmp_path)
    _runtime_environment(
        monkeypatch,
        bundle=bundle,
        request_digest=receipt["source_track_run_request_digest"],
    )

    def fake_stage(**kwargs):
        write_json(Path(kwargs["provider_result_path"]), {"tracks": []})
        write_json(Path(kwargs["import_request_path"]), {})
        return {"status": "abstained", "blockers": []}

    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker.run_sam31_source_track_stage",
        fake_stage,
    )
    normalized = {
        "schema_version": "semantic_source_track_import_result.v1",
        "status": "abstained",
        "bindings": {},
        "track_registry": [],
        "frame_masks": [],
        "blockers": [],
        "warnings": ["provider_returned_no_tracks"],
        "claim_ceiling": "no_source_tracks_detected",
    }
    normalized["result_digest"] = canonical_json_digest(normalized)
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker.import_semantic_source_tracks",
        lambda *_args, **_kwargs: normalized,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker._runtime_facts",
        lambda: {"cuda_available": False},
    )
    result = run_sam31_source_track_canary_worker(
        input_bundle=bundle, output_path=tmp_path / "result.json"
    )
    assert result["status"] == "failed"
    assert result["blockers"] == ["sam31_gpu_cuda_runtime_unavailable"]


def test_worker_rejects_wrong_bundle_digest_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, receipt = _build(tmp_path)
    _runtime_environment(
        monkeypatch,
        bundle=bundle,
        request_digest=receipt["source_track_run_request_digest"],
    )
    monkeypatch.setenv("BLUEPRINT_SAM31_INPUT_BUNDLE_DIGEST", SHA)
    with pytest.raises(Sam31CanaryWorkerError, match="input_bundle_digest_mismatch"):
        run_sam31_source_track_canary_worker(
            input_bundle=bundle, output_path=tmp_path / "result.json"
        )
