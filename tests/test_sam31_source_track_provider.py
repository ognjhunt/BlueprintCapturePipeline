"""Contract tests for the authorized Meta SAM 3.1 source-track adapter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from blueprint_pipeline.sam31_source_track_provider_stage import (
    _adapt_multiplex_start_session,
    run_sam31_source_track_stage,
)
from blueprint_pipeline.scene_placement.sam31_source_track_provider import (
    CHECKPOINT_FAMILY,
    FRAME_INPUT_MODE,
    RUN_REQUEST_SCHEMA_VERSION,
    RUNTIME_API,
    execute_sam31_source_track_request,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)
from blueprint_pipeline.scene_placement.semantic_source_track_import import (
    MASK_ENCODING,
    import_semantic_source_tracks,
)
from blueprint_pipeline.semantic_source_track_stage import (
    run_semantic_source_track_stage,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
REVISION = "9" * 40


class FakePredictor:
    def __init__(
        self,
        outputs: Mapping[int, Mapping[str, Any]],
        *,
        duplicate_frame_zero: bool = True,
    ) -> None:
        self.outputs = outputs
        self.duplicate_frame_zero = duplicate_frame_zero
        self.requests: list[dict[str, Any]] = []

    def handle_request(self, *, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(request))
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt":
            return {"frame_index": 0, "outputs": self.outputs[0]}
        return {"is_success": True}

    def handle_stream_request(self, *, request: Mapping[str, Any]):
        self.requests.append(dict(request))
        for frame_index in sorted(self.outputs):
            if frame_index == 0 and not self.duplicate_frame_zero:
                continue
            yield {"frame_index": frame_index, "outputs": self.outputs[frame_index]}


class CleanupFailingPredictor(FakePredictor):
    def handle_request(self, *, request: Mapping[str, Any]) -> dict[str, Any]:
        if request["type"] == "close_session":
            raise RuntimeError("simulated cleanup failure")
        return super().handle_request(request=request)


class OfficialMultiplexSignatureModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def init_state(
        self,
        resource_path: str,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        input_is_mp4: bool = False,
    ) -> dict[str, Any]:
        call = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "async_loading_frames": async_loading_frames,
            "use_torchcodec": use_torchcodec,
            "use_cv2": use_cv2,
            "input_is_mp4": input_is_mp4,
        }
        self.calls.append(call)
        return call


class OfficialBasePredictorShape:
    def __init__(self) -> None:
        self.model = OfficialMultiplexSignatureModel()
        self.async_loading_frames = False

    def start_session(
        self,
        *,
        resource_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, Any]:
        state = self.model.init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=self.async_loading_frames,
        )
        return {"session_id": "official-session", "state": state}


def _outputs(*, empty: bool = False) -> dict[int, dict[str, Any]]:
    if empty:
        return {
            index: {
                "out_obj_ids": np.zeros(0, dtype=np.int64),
                "out_probs": np.zeros(0, dtype=np.float32),
                "out_binary_masks": np.zeros((0, 2, 4), dtype=bool),
            }
            for index in range(2)
        }
    first = np.zeros((1, 2, 4), dtype=bool)
    first[0, 0, :2] = True
    second = np.zeros((1, 2, 4), dtype=bool)
    second[0, 1, 1:4] = True
    return {
        0: {
            "out_obj_ids": np.array([7], dtype=np.int64),
            "out_probs": np.array([0.9], dtype=np.float32),
            "out_binary_masks": first,
        },
        1: {
            "out_obj_ids": np.array([7], dtype=np.int64),
            "out_probs": np.array([0.8], dtype=np.float32),
            "out_binary_masks": second,
        },
    }


def _request() -> dict[str, Any]:
    frames = [
        {
            "source_frame_id": f"frame-{index}",
            "model_frame_index": index,
            "source_frame_digest": "sha256:" + str(index + 1) * 64,
            "retained_video_digest": SHA_B,
            "decoded_pts_seconds": float(index),
            "sync_map_row_digest": SHA_C,
            "camera_record_digest": "sha256:" + str(index + 3) * 64,
            "encoder_retained": True,
            "width": 4,
            "height": 2,
            "analysis_jpeg_digest": "sha256:" + str(index + 5) * 64,
        }
        for index in range(2)
    ]
    profile = {
        "method_id": "meta.sam3.1.object_multiplex",
        "method_version": "2026-03-27",
        "runtime_api": RUNTIME_API,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "frame_input_mode": FRAME_INPUT_MODE,
        "mask_encoding": MASK_ENCODING,
        "execution_mode": "local",
        "official_code_revision": REVISION,
        "runtime_digest": SHA_C,
        "model_digest": SHA_D,
        "checkpoint_digest": SHA_D,
        "license_terms_digest": SHA_C,
        "license_use_authorization_digest": SHA_A,
        "privacy_use_authorization_digest": SHA_B,
        "trade_controls_review_digest": SHA_C,
        "execution_authorization_digest": SHA_B,
        "checkpoint_access_authorized": True,
        "commercial_evidence_use_authorized": True,
        "persistent_track_ids": True,
        "model_self_grading_forbidden": True,
        "source_frames_are_hash_verified": True,
        "network_access_during_inference_forbidden": True,
        "customer_data_training_allowed": False,
        "output_probability_threshold": 0.5,
        "max_num_objects": 16,
        "multiplex_count": 16,
        "use_fa3": False,
        "compile": False,
        "warm_up": False,
        "async_loading_frames": False,
    }
    profile["profile_digest"] = canonical_json_digest(profile)
    return {
        "schema_version": RUN_REQUEST_SCHEMA_VERSION,
        "bindings": {
            "capture_digest": SHA_A,
            "retained_video_digest": SHA_B,
            "camera_solution_digest": SHA_C,
            "frame_registry_digest": canonical_json_digest(frames),
        },
        "frame_registry": frames,
        "provider_profile": profile,
        "prompts": [{"prompt_id": "chair", "text": "chair", "output_label": "chair"}],
        "allowed_evidence_uses": ["semantic_analysis"],
    }


def _frame_directory(tmp_path: Path) -> Path:
    root = tmp_path / "frames"
    root.mkdir(exist_ok=True)
    for index in range(2):
        (root / f"{index:06d}.jpg").write_bytes(f"jpeg-{index}".encode())
    return root


def _factory(predictor: FakePredictor):
    def build(_: Mapping[str, Any]) -> FakePredictor:
        return predictor

    return build


def _attach_stage_frames(request: dict[str, Any], tmp_path: Path) -> None:
    artifacts = []
    for index, frame in enumerate(request["frame_registry"]):
        path = tmp_path / f"source-{index}.jpg"
        path.write_bytes(f"real-jpeg-{index}".encode())
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        frame["analysis_jpeg_digest"] = digest
        artifacts.append(
            {
                "source_frame_id": frame["source_frame_id"],
                "path": str(path),
                "media_type": "image/jpeg",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    request["bindings"]["frame_registry_digest"] = canonical_json_digest(request["frame_registry"])
    request["frame_artifacts"] = artifacts


def test_executes_official_multiplex_shape_into_source_bound_tracks(tmp_path: Path) -> None:
    request = _request()
    predictor = FakePredictor(_outputs())

    result = execute_sam31_source_track_request(
        request,
        predictor_factory=_factory(predictor),
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    assert result["status"] == "completed"
    provider = result["provider_result"]
    assert provider["tracks"][0]["track_id"] == "sam31-chair-7"
    assert provider["tracks"][0]["label_source"] == "model_inferred"
    assert provider["tracks"][0]["observations"][0]["runs"] == [
        {"start": 0, "length": 2, "probability": pytest.approx(0.9)}
    ]
    assert provider["provider_metadata"]["run_probability"] == (
        "object_detection_score_on_binary_support"
    )
    assert provider["provider_metadata"]["cross_prompt_instance_deduplication_performed"] is False
    assert result["metric_box_ready"] is False
    assert result["collision_ready"] is False
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert predictor.requests[-1]["type"] == "close_session"

    imported = import_semantic_source_tracks(result["source_track_import_request"], provider)
    assert imported["status"] == "completed"
    assert imported["track_registry"][0]["semantic_authority"] == "inferred_candidate"


def test_replay_is_deterministic_and_empty_model_result_abstains(tmp_path: Path) -> None:
    request = _request()
    frame_root = _frame_directory(tmp_path)
    first = execute_sam31_source_track_request(
        request,
        predictor_factory=_factory(FakePredictor(_outputs(empty=True))),
        materialized_frame_directory=frame_root,
    )
    second = execute_sam31_source_track_request(
        copy.deepcopy(request),
        predictor_factory=_factory(FakePredictor(_outputs(empty=True))),
        materialized_frame_directory=frame_root,
    )

    assert first == second
    assert first["status"] == "abstained"
    assert first["provider_result"]["tracks"] == []
    assert first["warnings"] == ["sam31_returned_no_tracks"]


def test_multiple_prompts_keep_namespaced_tracks_without_claiming_deduplication(
    tmp_path: Path,
) -> None:
    request = _request()
    request["prompts"].append({"prompt_id": "seat", "text": "seat", "output_label": "seat"})

    result = execute_sam31_source_track_request(
        request,
        predictor_factory=_factory(FakePredictor(_outputs())),
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    provider = result["provider_result"]
    assert [track["track_id"] for track in provider["tracks"]] == [
        "sam31-chair-7",
        "sam31-seat-7",
    ]
    assert provider["provider_metadata"]["cross_prompt_instance_deduplication_performed"] is False


def test_fails_closed_without_license_execution_and_checkpoint_authority(tmp_path: Path) -> None:
    request = _request()
    profile = request["provider_profile"]
    profile["commercial_evidence_use_authorized"] = False
    profile["checkpoint_access_authorized"] = False
    profile["execution_authorization_digest"] = ""
    profile["profile_digest"] = canonical_json_digest(
        {key: value for key, value in profile.items() if key != "profile_digest"}
    )

    result = execute_sam31_source_track_request(
        request,
        predictor_factory=_factory(FakePredictor(_outputs())),
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    assert result["status"] == "blocked"
    assert "provider_profile_commercial_evidence_use_authorized_required" in result["blockers"]
    assert "provider_profile_checkpoint_access_authorized_required" in result["blockers"]
    assert "provider_profile_execution_authorization_digest_invalid" in result["blockers"]


def test_session_cleanup_failure_erases_candidate_provider_output(tmp_path: Path) -> None:
    result = execute_sam31_source_track_request(
        _request(),
        predictor_factory=_factory(CleanupFailingPredictor(_outputs())),
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["sam31_session_cleanup_failed"]
    assert result["provider_result"] is None
    assert result["source_track_import_request"] is None


def test_untrusted_runtime_exception_text_is_redacted(tmp_path: Path) -> None:
    def leaking_factory(_: Mapping[str, Any]) -> FakePredictor:
        raise ValueError("sam31_token_supersecret")

    result = execute_sam31_source_track_request(
        _request(),
        predictor_factory=leaking_factory,
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "sam31_runtime_failed:predictor_construction:ValueError"
    ]
    assert "supersecret" not in json.dumps(result)


def test_adapts_exact_official_multiplex_start_session_signature() -> None:
    predictor = _adapt_multiplex_start_session(OfficialBasePredictorShape())

    started = predictor.start_session(
        resource_path="/frames",
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    )

    assert started["session_id"] == "official-session"
    assert predictor.model.calls == [
        {
            "resource_path": "/frames",
            "offload_video_to_cpu": False,
            "async_loading_frames": False,
            "use_torchcodec": False,
            "use_cv2": False,
            "input_is_mp4": False,
        }
    ]
    with pytest.raises(ValueError, match="sam31_runtime_state_offload_unsupported"):
        predictor.start_session(
            resource_path="/frames",
            offload_state_to_cpu=True,
        )


def test_runtime_failure_reports_bounded_phase_without_exception_text(
    tmp_path: Path,
) -> None:
    class SessionFailingPredictor:
        def handle_request(self, *, request: Mapping[str, Any]) -> dict[str, Any]:
            if request["type"] == "start_session":
                raise TypeError("sam31_token_supersecret")
            return {"is_success": True}

    result = execute_sam31_source_track_request(
        _request(),
        predictor_factory=lambda _: SessionFailingPredictor(),
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    assert result["blockers"] == ["sam31_runtime_failed:session_start:TypeError"]
    assert "supersecret" not in json.dumps(result)


def test_rejects_checkpoint_incompatible_multiplex_count_before_runtime(
    tmp_path: Path,
) -> None:
    request = _request()
    request["provider_profile"]["multiplex_count"] = 5
    request["provider_profile"]["profile_digest"] = canonical_json_digest(
        {
            key: value
            for key, value in request["provider_profile"].items()
            if key != "profile_digest"
        }
    )
    called = False

    def forbidden_factory(_: Mapping[str, Any]) -> FakePredictor:
        nonlocal called
        called = True
        return FakePredictor(_outputs())

    result = execute_sam31_source_track_request(
        request,
        predictor_factory=forbidden_factory,
        materialized_frame_directory=_frame_directory(tmp_path),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "provider_profile_multiplex_count_checkpoint_mismatch"
    ]
    assert called is False


def test_fails_closed_on_missing_frame_or_wrong_mask_dimensions(tmp_path: Path) -> None:
    request = _request()
    missing = {0: _outputs()[0]}
    missing_result = execute_sam31_source_track_request(
        request,
        predictor_factory=_factory(FakePredictor(missing)),
        materialized_frame_directory=_frame_directory(tmp_path),
    )
    assert missing_result["status"] == "blocked"
    assert missing_result["blockers"] == ["sam31_outputs_missing_retained_frames"]

    wrong = _outputs()
    wrong[1] = {**wrong[1], "out_binary_masks": np.zeros((1, 3, 4), dtype=bool)}
    wrong_result = execute_sam31_source_track_request(
        request,
        predictor_factory=_factory(FakePredictor(wrong)),
        materialized_frame_directory=_frame_directory(tmp_path),
    )
    assert wrong_result["status"] == "blocked"
    assert wrong_result["blockers"] == ["sam31_output_frame_dimensions_mismatch"]


def test_stage_hash_verifies_frames_and_feeds_existing_import_stage(tmp_path: Path) -> None:
    request = _request()
    _attach_stage_frames(request, tmp_path)
    request_path = tmp_path / "run-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    run_path = tmp_path / "run-result.json"
    provider_path = tmp_path / "provider-result.json"
    import_path = tmp_path / "import-request.json"

    result = run_sam31_source_track_stage(
        request_path=request_path,
        run_result_path=run_path,
        provider_result_path=provider_path,
        import_request_path=import_path,
        predictor_factory=_factory(FakePredictor(_outputs())),
    )

    assert result["status"] == "completed"
    assert run_path.is_file() and provider_path.is_file() and import_path.is_file()
    assert result["provider_result_artifact"]["sha256"] == (
        "sha256:" + hashlib.sha256(provider_path.read_bytes()).hexdigest()
    )
    normalized_path = tmp_path / "normalized.json"
    normalized = run_semantic_source_track_stage(
        request_path=import_path,
        provider_result_path=provider_path,
        output_path=normalized_path,
    )
    assert normalized["status"] == "completed"
    assert normalized["claim_ceiling"] == "source_bound_2d_mask_tracks_only"


def test_stage_rejects_tampered_frame_before_runtime(tmp_path: Path) -> None:
    request = _request()
    _attach_stage_frames(request, tmp_path)
    Path(request["frame_artifacts"][0]["path"]).write_bytes(b"tampered")
    request_path = tmp_path / "run-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    called = False

    def forbidden_factory(_: Mapping[str, Any]) -> FakePredictor:
        nonlocal called
        called = True
        return FakePredictor(_outputs())

    result = run_sam31_source_track_stage(
        request_path=request_path,
        run_result_path=tmp_path / "run-result.json",
        provider_result_path=tmp_path / "provider-result.json",
        import_request_path=tmp_path / "import-request.json",
        predictor_factory=forbidden_factory,
    )

    assert result["status"] == "blocked"
    assert called is False
    assert any("frame_artifact_" in blocker for blocker in result["blockers"])
    assert not (tmp_path / "provider-result.json").exists()


def test_stage_denies_unconfigured_gated_runtime_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    _attach_stage_frames(request, tmp_path)
    request_path = tmp_path / "run-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.delenv("BLUEPRINT_SAM31_OFFICIAL_CODE_REVISION", raising=False)
    monkeypatch.delenv("BLUEPRINT_SAM31_RUNTIME_DIGEST", raising=False)
    monkeypatch.delenv("BLUEPRINT_SAM31_CHECKPOINT_PATH", raising=False)

    result = run_sam31_source_track_stage(
        request_path=request_path,
        run_result_path=tmp_path / "run-result.json",
        provider_result_path=tmp_path / "provider-result.json",
        import_request_path=tmp_path / "import-request.json",
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["sam31_installed_code_revision_mismatch"]
    assert not (tmp_path / "provider-result.json").exists()
    assert "HF_HUB_OFFLINE" not in result


def test_stage_outputs_are_immutable_and_distinct(tmp_path: Path) -> None:
    request = _request()
    _attach_stage_frames(request, tmp_path)
    request_path = tmp_path / "run-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable_output_already_exists"):
        run_sam31_source_track_stage(
            request_path=request_path,
            run_result_path=existing,
            provider_result_path=tmp_path / "provider.json",
            import_request_path=tmp_path / "import.json",
            predictor_factory=_factory(FakePredictor(_outputs())),
        )
    with pytest.raises(ValueError, match="output_paths_must_be_distinct"):
        run_sam31_source_track_stage(
            request_path=request_path,
            run_result_path=tmp_path / "same.json",
            provider_result_path=tmp_path / "same.json",
            import_request_path=tmp_path / "other.json",
            predictor_factory=_factory(FakePredictor(_outputs())),
        )
