"""R025: success-label provenance must be explicit and gate success_rate.

A headline ``success_rate`` for a WAM run is a VLM judgment over GENERATED
rollout video, not physics or captured truth. These tests pin that:

- the WAM generated-video VLM labelers stamp every label with
  ``success_label_provenance = generated_video_vlm`` plus the disclosure
  boundary, and
- the buyer-facing Task Evaluation Run scorecard threads that provenance into
  each row and refuses to present a generated-video-VLM (or unknown) rate as
  physics/real-world success.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import wam_generated_video_success_label_gemini as gemini_labeler
from blueprint_pipeline import wam_generated_video_success_label_openai as openai_labeler
from blueprint_pipeline.success_claim_contracts import build_success_claim_ledger
from blueprint_pipeline.task_eval_run_report import (
    SUCCESS_LABEL_PROVENANCE_VOCABULARY,
    SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY,
    SUCCESS_RATE_UNKNOWN_PROVENANCE_CLAIM_BOUNDARY,
    build_task_eval_run_report,
    build_task_eval_scorecard,
)


def _attempts(
    n_success: int,
    n_fail: int,
    *,
    provenance: str | None,
    task_id: str = "open-drawer",
    scenario_id: str = "clean-path",
) -> list[dict]:
    rows: list[dict] = []
    for i in range(n_success + n_fail):
        row = {
            "attempt_id": f"{task_id}_{i}",
            "task_id": task_id,
            "scenario_id": scenario_id,
            "success": i < n_success,
        }
        if provenance is not None:
            row["success_label_provenance"] = provenance
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Buyer report scorecard: presentation gate                                    #
# --------------------------------------------------------------------------- #


def test_generated_video_vlm_row_is_tagged_and_carries_boundary() -> None:
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(8, 2, provenance="generated_video_vlm"),
        evidence_level="review_task_success",
    )
    row = scorecard["conditions"][0]
    # provenance is explicit and drawn from the controlled vocabulary
    assert row["success_label_provenance"] == "generated_video_vlm"
    assert row["success_label_provenances"] == ["generated_video_vlm"]
    # the rate is still published (rates_published) but is NOT physics/captured
    # truth and must carry the disclosure boundary
    assert row["success_rate"]["point"] == 0.8
    assert row["success_rate_is_physics_or_captured_truth"] is False
    assert (
        row["success_rate_claim_boundary"]
        == SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )
    boundary = scorecard["success_label_provenance_boundary"]
    assert boundary["any_success_rate_from_generated_video_vlm"] is True
    assert boundary["all_success_rates_are_physics_or_captured_truth"] is False
    assert boundary["generated_video_vlm_boundary"] == (
        SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )
    assert "generated_video_vlm" in scorecard["observed_success_label_provenances"]
    assert (
        scorecard["success_label_provenance_vocabulary"]
        == list(SUCCESS_LABEL_PROVENANCE_VOCABULARY)
    )


def test_simulator_physics_row_is_truth_and_carries_no_boundary() -> None:
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(6, 4, provenance="simulator_physics"),
        evidence_level="review_task_success",
    )
    row = scorecard["conditions"][0]
    assert row["success_label_provenance"] == "simulator_physics"
    assert row["success_rate_is_physics_or_captured_truth"] is True
    assert row["success_rate_claim_boundary"] is None
    boundary = scorecard["success_label_provenance_boundary"]
    assert boundary["any_success_rate_from_generated_video_vlm"] is False
    assert boundary["all_success_rates_are_physics_or_captured_truth"] is True


def test_missing_and_unrecognized_provenance_defaults_to_unknown() -> None:
    # No provenance declared -> conservative ``unknown``, never fabricated.
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(5, 5, provenance=None),
        evidence_level="review_task_success",
    )
    row = scorecard["conditions"][0]
    assert row["success_label_provenance"] == "unknown"
    assert row["success_rate_is_physics_or_captured_truth"] is False
    assert (
        row["success_rate_claim_boundary"]
        == SUCCESS_RATE_UNKNOWN_PROVENANCE_CLAIM_BOUNDARY
    )

    # An out-of-vocabulary label is also treated as ``unknown``.
    scorecard_bogus = build_task_eval_scorecard(
        attempts=_attempts(5, 5, provenance="marketing_claim"),
        evidence_level="review_task_success",
    )
    assert scorecard_bogus["conditions"][0]["success_label_provenance"] == "unknown"


def test_mixed_condition_never_presents_a_single_trusted_provenance() -> None:
    attempts = _attempts(4, 0, provenance="simulator_physics") + _attempts(
        0, 4, provenance="generated_video_vlm"
    )
    scorecard = build_task_eval_scorecard(
        attempts=attempts, evidence_level="review_task_success"
    )
    row = scorecard["conditions"][0]
    assert row["success_label_provenances"] == [
        "generated_video_vlm",
        "simulator_physics",
    ]
    # a generated-video-VLM contributor wins the disclosure and blocks truth
    assert row["success_label_provenance"] == "generated_video_vlm"
    assert row["success_rate_is_physics_or_captured_truth"] is False
    assert (
        row["success_rate_claim_boundary"]
        == SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )


def test_report_claim_boundary_discloses_generated_video_vlm_success_rate() -> None:
    ledger = build_success_claim_ledger(
        task_metadata={"task_id": "open-drawer"},
        media_validity={"status": "PASS", "blockers": []},
        review_task_success={"status": "PASS", "blockers": []},
    )
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(20, 5, provenance="generated_video_vlm")},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
    )
    assert report["evidence_level"] == "review_task_success"
    boundary = report["claim_boundary"]
    assert boundary["success_rate_provenance_disclosed"] is True
    assert (
        boundary["generated_video_vlm_success_rate_is_not_physics_or_captured_truth"]
        is True
    )
    assert boundary["any_success_rate_from_generated_video_vlm"] is True
    assert boundary["all_success_rates_are_physics_or_captured_truth"] is False
    row = report["scorecard"]["conditions"][0]
    assert (
        row["success_rate_claim_boundary"]
        == SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )


# --------------------------------------------------------------------------- #
# Producer path: labelers stamp generated_video_vlm provenance                 #
# --------------------------------------------------------------------------- #


def test_provenance_constant_is_shared_between_producer_and_report() -> None:
    # The thread only holds if the producer emits exactly the string the buyer
    # report gates on.
    assert gemini_labeler.GENERATED_VIDEO_VLM_PROVENANCE == "generated_video_vlm"
    assert (
        gemini_labeler.SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
        == SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )
    assert (
        openai_labeler.GENERATED_VIDEO_VLM_PROVENANCE
        == gemini_labeler.GENERATED_VIDEO_VLM_PROVENANCE
    )


def _wam_request(tmp_path: Path) -> Path:
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"mp4")
    request = tmp_path / "wam_success_label_request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "wam_success_label_request.v1",
                "inference_input_manifest_sha256": "a" * 64,
                "rollouts": [
                    {
                        "rollout_id": "rollout_1",
                        "scenario_eval_run_id": "run_1",
                        "policy_id": "policy",
                        "generated_video_path": str(video),
                    }
                ],
                "task_prompts": [
                    {
                        "scenario_eval_run_id": "run_1",
                        "task_prompt": "Open the articulated target.",
                        "success_check_plan": {
                            "success_requires": [
                                "robot hand reaches the pull",
                                "target visibly opens",
                            ],
                            "common_failure_modes": ["target_already_open_in_first_frame"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return request


def _configure_runtime_signing(tmp_path: Path, monkeypatch) -> None:
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    private_key_path = tmp_path / "wam-success-label-runtime.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv(
        gemini_labeler.RUNTIME_SIGNING_PRIVATE_KEY_FILE_ENV,
        str(private_key_path),
    )


def test_gemini_label_stamps_generated_video_vlm_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_runtime_signing(tmp_path, monkeypatch)
    monkeypatch.setenv(gemini_labeler.GATE_ENV, "true")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setattr(
        gemini_labeler,
        "_sample_video_frames",
        lambda **_kwargs: (
            [
                {
                    "frame_index": 0,
                    "jpeg_bytes": b"frame-zero",
                    "mime_type": "image/jpeg",
                    "evidence_ref": "rollout.mp4#frame=0",
                }
            ],
            [],
        ),
    )

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, *, response_mime_type):
            assert response_mime_type == "application/json"

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "success": True,
                        "confidence": 0.9,
                        "rationale": "reaches target",
                        "end_effector_reaches_target": True,
                        "target_state_change_visible": True,
                        "robot_caused_target_motion": True,
                    }
                )
            )

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = FakeModels()

    types_module.Part = FakePart
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    genai_module.Client = FakeClient
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    result = gemini_labeler.build_gemini_wam_success_labels(
        input_path=_wam_request(tmp_path),
        output_path=tmp_path / "labels.json",
        model="gemini-test-flash-model",
        max_frames=1,
    )
    assert result["status"] == "completed"
    label = result["labels"][0]
    assert label["success_label_provenance"] == "generated_video_vlm"
    assert (
        label["success_label_claim_boundary"]
        == SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )
    assert label["success_label_is_physics_or_captured_truth"] is False
    assert result["success_label_provenance"] == "generated_video_vlm"
    assert (
        result["claim_boundary"][
            "success_rate_from_generated_video_vlm_is_not_physics_or_captured_truth"
        ]
        is True
    )


def test_openai_label_stamps_generated_video_vlm_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_runtime_signing(tmp_path, monkeypatch)
    monkeypatch.setenv(openai_labeler.GATE_ENV, "true")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    monkeypatch.setattr(
        openai_labeler,
        "_sample_video_frames",
        lambda **_kwargs: (
            [
                {
                    "frame_index": 0,
                    "image_url": "data:image/jpeg;base64,ZmFrZQ==",
                    "evidence_ref": "rollout.mp4#frame=0",
                }
            ],
            [],
        ),
    )

    class FakeResponses:
        def create(self, *, model, input, max_output_tokens):
            return types.SimpleNamespace(
                output_text=json.dumps(
                    {
                        "success": True,
                        "confidence": 0.8,
                        "rationale": "hand reaches target",
                        "end_effector_reaches_target": True,
                        "target_state_change_visible": True,
                        "robot_caused_target_motion": True,
                    }
                )
            )

    class FakeOpenAI:
        def __init__(self, *, api_key):
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    result = openai_labeler.build_openai_wam_success_labels(
        input_path=_wam_request(tmp_path),
        output_path=tmp_path / "labels.json",
        model="gpt-test-vision",
    )
    assert result["status"] == "completed"
    label = result["labels"][0]
    assert label["success_label_provenance"] == "generated_video_vlm"
    assert (
        label["success_label_claim_boundary"]
        == SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY
    )
    assert result["success_label_provenance"] == "generated_video_vlm"
    assert (
        result["claim_boundary"][
            "success_rate_from_generated_video_vlm_is_not_physics_or_captured_truth"
        ]
        is True
    )
