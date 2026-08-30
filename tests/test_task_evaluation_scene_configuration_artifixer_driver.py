from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_driver as driver
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver import (
    _VISUAL_REVIEW_COST_SCOPE,
    _artifixer_tuning,
    _materialize_preflight,
    _materialize_selected_task_thumbnail,
    _materialize_ungraded_task_thumbnail_and_receipt,
    _semantic_rights_and_request,
    _write_execution_authority,
)
from blueprint_pipeline.task_evaluation_scene_configuration_appearance_review import (
    PAUSED_UNGRADED_WARNING,
    paused_review_receipt_valid,
)
from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    _target_camera_ring,
)
from tests.test_task_evaluation_scene_configuration_diagnostic_checkpoint import (
    _materialize as _materialize_diagnostic_checkpoint,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[dict, dict]:
    rights = {
        "schema_version": "task_evaluation_scene_rights_admission.v1",
        "status": "admitted_for_internal_development",
        "scene_id": "839873",
        "publisher_scene_id": "839873",
        "private_provider_processing_allowed": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
    }
    rights_path = tmp_path / "rights.json"
    rights_path.write_text(json.dumps(rights), encoding="utf-8")
    frame = tmp_path / "camera-0.png"
    mask = tmp_path / "camera-0-mask.png"
    Image.new("RGB", (1024, 1024), color=(90, 80, 70)).save(frame)
    Image.new("L", (1024, 1024), color=255).save(mask)
    retained = tmp_path / "retained.ply"
    write_standard_3dgs_ply(
        SplatData(
            count=2,
            xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            opacity=np.zeros(2, dtype=np.float32),
            f_dc=np.zeros((2, 3), dtype=np.float32),
            scales=np.zeros((2, 3), dtype=np.float32),
            quats=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
            properties=(),
        ),
        retained,
    )
    calibration = tmp_path / "cameras.json"
    generated_camera = _target_camera_ring(
        minimum_xyz=[2.91, -6.83, 0.754],
        maximum_xyz=[3.04, -6.69, 0.884],
    )[0]
    calibration.write_text(
        json.dumps(
            [
                {
                    "id": "camera-0",
                    "spec": {
                        "pose": {
                            "T_world_camera_opencv": generated_camera[
                                "T_world_camera_provider_frame"
                            ]
                        },
                        "intrinsics": generated_camera["intrinsics"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    envelope = {
        "materialized_references": [
            {
                "contract_path": "scene.rights.admission",
                "materialized_path": str(rights_path),
                "digest": _sha256(rights_path),
                "size_bytes": rights_path.stat().st_size,
                "full_byte_service_account_readback_passed": True,
            }
        ],
        "render_inputs_result": {
            "camera_calibration": {"path": str(calibration)},
            "derived_frames": [
                {
                    "camera_id": "camera-0",
                    "path": str(frame),
                    "source_object_mask": {"path": str(mask)},
                }
            ],
            "derived_gaussian_cutout": {
                "retained_scene_without_source_object": {"path": str(retained)},
                "retained_count": 2,
            },
        },
    }
    configuration = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "source_object": {"publisher_instance_id": "104"},
        "human_authority": {
            "accepted_by": "project-owner",
            "accepted_on": "2026-08-25",
            "authority_reference": "website-scene-configuration-consent-v1",
            "private_derived_frame_disclosure_authorized": True,
            "provider_retention_terms_accepted": True,
            "provider_training_terms_accepted": True,
            "provider_training_authorized": False,
        },
    }
    return envelope, configuration


def test_nullable_production_tuning_resolves_before_paid_semantic_edits() -> None:
    assert _artifixer_tuning(
        {
            "transition_radius_pixels": None,
            "artifixer3d_steps": None,
            "random_seed": None,
        }
    ) == {
        "transition_radius_pixels": 3,
        "artifixer3d_steps": 30_000,
        "random_seed": 839_873,
    }


def test_post_training_binding_survives_a_diagnostic_code_overlay() -> None:
    base = {
        "source_commit": "a" * 40,
        "run_id": "base-run",
        "toolchain_digest": "sha256:" + "1" * 64,
        "configuration_sha256": "sha256:" + "2" * 64,
        "construction_envelope": {
            "control_plane_envelope_digest": "sha256:" + "3" * 64
        },
    }
    overlay = {
        **base,
        "source_commit": "b" * 40,
        "run_id": "warm-iteration-run",
        "toolchain_digest": "sha256:" + "4" * 64,
    }
    inputs = {
        "package_manifest": {"package_digest": "sha256:" + "5" * 64},
        "semantic_teacher_receipt_digest": "sha256:" + "6" * 64,
        "tuning": {
            "transition_radius_pixels": 3,
            "artifixer3d_steps": 30_000,
            "random_seed": 839_873,
        },
    }

    assert driver._post_training_bindings(stage_input=base, **inputs) == (
        driver._post_training_bindings(stage_input=overlay, **inputs)
    )
    assert driver._post_training_bindings(stage_input=base, **inputs) != (
        driver._post_training_bindings(
            stage_input=base,
            **{
                **inputs,
                "semantic_teacher_receipt_digest": "sha256:" + "7" * 64,
            },
        )
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("transition_radius_pixels", -1),
        ("artifixer3d_steps", 30_001),
        ("random_seed", True),
    ],
)
def test_invalid_production_tuning_fails_closed_before_paid_semantic_edits(
    name: str, value: object
) -> None:
    configuration = {
        "transition_radius_pixels": None,
        "artifixer3d_steps": None,
        "random_seed": None,
        name: value,
    }
    with pytest.raises(
        RuntimeError, match="scene_configuration_artifixer_tuning_invalid"
    ):
        _artifixer_tuning(configuration)


def test_diagnostic_driver_hydrates_render_and_semantic_without_paid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root, _checkpoint, fixture = _materialize_diagnostic_checkpoint(
        tmp_path
    )
    stage_input = json.loads(Path(fixture["stage_path"]).read_text(encoding="utf-8"))
    stage_input["stage"] = {
        "stage_id": "stage-1",
        "adapter": {"id": "artifixer3d_observed_object_removal"},
    }
    stage_input["construction_envelope"]["render_inputs_result"] = fixture[
        "render_result"
    ]
    stage_path = tmp_path / "driver-stage-input.json"
    stage_path.write_text(json.dumps(stage_input), encoding="utf-8")
    dependencies_path = tmp_path / "dependencies.json"
    dependencies_path.write_text("[]\n", encoding="utf-8")
    output_root = tmp_path / "driver-output"
    output_root.mkdir()
    package_root = tmp_path / "package"
    package_root.mkdir()
    semantic_request = json.loads(
        Path(fixture["request_path"]).read_text(encoding="utf-8")
    )
    locality_receipt = tmp_path / "locality-receipt.json"
    locality_receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        driver,
        "materialize_semantic_locality_seal",
        lambda **kwargs: {
            "receipt": {"receipt_digest": "sha256:" + "1" * 64},
            "receipt_path": str(locality_receipt),
            "semantic_teacher_frames_root": str(kwargs["semantic_output_root"] / "tasks" / "remove-source-object-104"),
        },
    )

    monkeypatch.setattr(
        driver,
        "complete_provider_render_inputs",
        lambda **_kwargs: pytest.fail("diagnostic retry rerendered frames"),
    )
    monkeypatch.setattr(
        driver,
        "execute_semantic_teacher_image_edits",
        lambda **_kwargs: pytest.fail("diagnostic retry called semantic provider"),
    )
    monkeypatch.setattr(
        driver,
        "_stage_openai_token",
        lambda *_args, **_kwargs: pytest.fail("diagnostic retry read semantic key"),
    )
    monkeypatch.setattr(
        driver,
        "_write_execution_authority",
        lambda **_kwargs: ({}, tmp_path / "rights.json", "839873"),
    )
    monkeypatch.setattr(driver, "materialize_provider_render_handoff", lambda **_kwargs: {})
    monkeypatch.setattr(
        driver,
        "_materialize_preflight",
        lambda **_kwargs: ({}, "remove-source-object-104"),
    )
    monkeypatch.setattr(
        driver,
        "materialize_artifixer3d_candidate_inputs",
        lambda **_kwargs: {"receipt_digest": "sha256:" + "2" * 64},
    )
    monkeypatch.setattr(
        driver,
        "_semantic_rights_and_request",
        lambda **_kwargs: tmp_path / "packet",
    )

    def write_request(**_kwargs):
        packet = tmp_path / "packet"
        packet.mkdir(exist_ok=True)
        path = packet / "semantic_teacher_image_edit_runtime_request.v1.json"
        path.write_text(json.dumps(semantic_request), encoding="utf-8")
        return path

    monkeypatch.setattr(driver, "_semantic_runtime_request", write_request)

    class Hydrated(Exception):
        pass

    def stop_after_hydration(**kwargs):
        frames = Path(kwargs["semantic_teacher_frames_root"])
        assert len(list(frames.glob("*.png"))) == 8
        raise Hydrated

    monkeypatch.setattr(
        driver, "materialize_whole_frame_semantic_teacher_receipt", stop_after_hydration
    )

    with pytest.raises(Hydrated):
        driver.execute_artifixer_component(
            environment={
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(stage_path),
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(
                    dependencies_path
                ),
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output_root),
                "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(
                    output_root / "result.json"
                ),
                "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT": str(package_root),
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT": str(
                    checkpoint_root
                ),
            }
        )


def test_diagnostic_driver_repairs_only_rejected_semantic_frame_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retained render/semantic checkpoint feeds one bounded repair loop."""

    checkpoint_root, _checkpoint, fixture = _materialize_diagnostic_checkpoint(
        tmp_path
    )
    stage_input = json.loads(Path(fixture["stage_path"]).read_text(encoding="utf-8"))
    stage_input["stage"] = {
        "stage_id": "stage-1",
        "adapter": {"id": "artifixer3d_observed_object_removal"},
    }
    stage_input["construction_envelope"]["render_inputs_result"] = fixture[
        "render_result"
    ]
    stage_path = tmp_path / "driver-stage-input.json"
    stage_path.write_text(json.dumps(stage_input), encoding="utf-8")
    dependencies_path = tmp_path / "dependencies.json"
    dependencies_path.write_text("[]\n", encoding="utf-8")
    output_root = tmp_path / "driver-output"
    output_root.mkdir()
    package_root = tmp_path / "package"
    package_root.mkdir()
    semantic_request = json.loads(
        Path(fixture["request_path"]).read_text(encoding="utf-8")
    )
    locality_receipt = tmp_path / "locality-receipt.json"
    locality_receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        driver,
        "materialize_semantic_locality_seal",
        lambda **kwargs: {
            "receipt": {"receipt_digest": "sha256:" + "1" * 64},
            "receipt_path": str(locality_receipt),
            "semantic_teacher_frames_root": str(kwargs["semantic_output_root"] / "tasks" / "remove-source-object-104"),
        },
    )
    monkeypatch.setattr(
        driver,
        "complete_provider_render_inputs",
        lambda **_kwargs: pytest.fail("diagnostic retry rerendered frames"),
    )
    monkeypatch.setattr(
        driver,
        "_write_execution_authority",
        lambda **_kwargs: ({}, tmp_path / "rights.json", "839873"),
    )
    monkeypatch.setattr(driver, "materialize_provider_render_handoff", lambda **_kwargs: {})
    monkeypatch.setattr(
        driver,
        "_materialize_preflight",
        lambda **_kwargs: ({}, "remove-source-object-104"),
    )
    monkeypatch.setattr(
        driver,
        "materialize_artifixer3d_candidate_inputs",
        lambda **_kwargs: {
            "receipt_digest": "sha256:" + "2" * 64,
            "tasks": [{"task_id": "remove-source-object-104", "frames": []}],
        },
    )
    monkeypatch.setattr(
        driver,
        "_semantic_rights_and_request",
        lambda **_kwargs: tmp_path / "packet",
    )

    def write_request(**_kwargs):
        packet = tmp_path / "packet"
        packet.mkdir(exist_ok=True)
        path = packet / "semantic_teacher_image_edit_runtime_request.v1.json"
        path.write_text(json.dumps(semantic_request), encoding="utf-8")
        return path

    monkeypatch.setattr(driver, "_semantic_runtime_request", write_request)
    teacher_calls: list[dict[str, object]] = []

    def write_teacher(**kwargs):
        teacher_calls.append(kwargs)
        Path(kwargs["output_path"]).write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        driver, "materialize_whole_frame_semantic_teacher_receipt", write_teacher
    )
    native = tmp_path / "configured.usdz"
    native.write_bytes(b"configured")
    review_frames = []
    for index in range(8):
        frame = tmp_path / f"trained-{index}.png"
        frame.write_bytes(f"trained-{index}".encode())
        review_frames.append(
            {
                "frame_index": index,
                "camera_id": f"camera-{index}",
                "final_frame": {
                    "path": str(frame),
                    "sha256": _sha256(frame),
                },
            }
        )
    training_calls: list[dict[str, object]] = []

    def train(**kwargs):
        training_calls.append(kwargs)
        return {
            "review_frames": review_frames,
            "native_appearance_source": native,
            "post_training_binding_digest": "sha256:" + "3" * 64,
            "runtime_result": {},
        }

    monkeypatch.setattr(driver, "_run_artifixer_training_round", train)
    review_execution = tmp_path / "review-execution.json"
    review_execution.write_text(
        json.dumps({"usage": {"projected_max_cost_usd": 0.1}}),
        encoding="utf-8",
    )
    review_receipt = tmp_path / "accepted-review.json"
    review_receipt.write_text("{}\n", encoding="utf-8")
    review_calls: list[dict[str, object]] = []

    def review_round(**kwargs):
        review_calls.append(kwargs)
        if len(review_calls) == 1:
            return {
                "review": {
                    "decision": "rejected",
                    "review_receipt": None,
                    "execution_receipt": {
                        "path": str(review_execution),
                        "execution_digest": "sha256:" + "4" * 64,
                    },
                },
                "review_input_path": tmp_path / "initial-review-input.json",
            }
        return {
            "review": {
                "decision": "accepted",
                "review_receipt": {
                    "path": str(review_receipt),
                    "receipt_digest": "sha256:" + "5" * 64,
                },
                "execution_receipt": {
                    "path": str(review_execution),
                    "execution_digest": "sha256:" + "6" * 64,
                },
            },
            "review_input_path": tmp_path / "repaired-review-input.json",
        }

    monkeypatch.setattr(driver, "_run_artifixer_visual_review_round", review_round)
    repair_request = tmp_path / "repair-request.json"
    repair_request.write_text("{}\n", encoding="utf-8")
    selective_calls: list[dict[str, object]] = []

    def stage_selective(**kwargs):
        selective_calls.append(kwargs)
        return {
            "plan": {
                "selected_frame_count": 1,
                "plan_digest": "sha256:" + "7" * 64,
                "remaining_stage_cost_usd": 1.52,
            },
            "plan_path": str(tmp_path / "repair-plan.json"),
            "repair_request_path": str(repair_request),
        }

    monkeypatch.setattr(driver, "materialize_selective_repair_request", stage_selective)

    class Gate:
        def reserve(self):
            return {}

        def complete(self, **_kwargs):
            return {}

    monkeypatch.setattr(
        driver, "scene_configuration_openai_stage_gate", lambda **_kwargs: Gate()
    )
    paid_semantic_calls: list[dict[str, object]] = []

    def execute_repair(**kwargs):
        paid_semantic_calls.append(kwargs)
        return {
            "backend_id": "openai-gpt-image",
            "model_snapshot": "gpt-image-2-2026-04-21",
            "result_digest": "sha256:" + "8" * 64,
        }

    monkeypatch.setattr(driver, "execute_semantic_teacher_image_edits", execute_repair)
    merged_frames = tmp_path / "merged-frames"
    merged_frames.mkdir()
    monkeypatch.setattr(
        driver,
        "merge_selective_repair_outputs",
        lambda **_kwargs: {
            "semantic_teacher_frames_root": str(merged_frames),
            "receipt": {"merge_digest": "sha256:" + "9" * 64},
        },
    )
    key_reads: list[str] = []

    def read_key(_environment, *, stage):
        key_reads.append(stage)
        return "test-file-token"

    monkeypatch.setattr(driver, "_stage_openai_token", read_key)

    class SelectiveLoopCompleted(RuntimeError):
        pass

    monkeypatch.setattr(
        driver,
        "_materialize_selected_task_thumbnail",
        lambda **_kwargs: (_ for _ in ()).throw(SelectiveLoopCompleted()),
    )
    with pytest.raises(SelectiveLoopCompleted):
        driver.execute_artifixer_component(
            environment={
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(stage_path),
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(
                    dependencies_path
                ),
                "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output_root),
                "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(
                    output_root / "result.json"
                ),
                "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT": str(package_root),
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT": str(
                    checkpoint_root
                ),
                "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD": "2.4",
                "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD": "0.33",
                "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": "16",
            }
        )

    assert len(training_calls) == 2
    assert training_calls[0]["post_training_checkpoint_root"] is None
    assert training_calls[1]["post_training_checkpoint_root"] is None
    assert len(review_calls) == 2
    assert [call["review_round"] for call in review_calls] == [0, 1]
    assert review_calls[1]["max_cost_usd"] == pytest.approx(0.23)
    assert len(selective_calls) == 1
    assert selective_calls[0]["semantic_runtime_result"]["request_count"] == 8
    assert len(paid_semantic_calls) == 1
    assert key_reads == ["artifixer_semantic_teacher"]
    assert len(teacher_calls) == 2
    assert teacher_calls[1]["prompt_policy"] == (
        driver.STRICT_LOCALITY_PROMPT_POLICY
    )


def test_ungraded_thumbnail_is_exact_deterministic_render_with_pause_receipt(
    tmp_path: Path,
) -> None:
    frames = []
    for camera_id in ("camera-z", "camera-a", "camera-m"):
        path = tmp_path / f"{camera_id}.png"
        path.write_bytes(camera_id.encode())
        frames.append(
            {
                "camera_id": camera_id,
                "final_frame": {"path": str(path), "sha256": _sha256(path)},
            }
        )
    thumbnail = tmp_path / "thumbnail.png"
    receipt_path = tmp_path / "pause.json"

    selection, receipt = _materialize_ungraded_task_thumbnail_and_receipt(
        review_frames=frames,
        publisher_instance_id="104",
        minimum_frame_count=3,
        thumbnail_destination=thumbnail,
        receipt_destination=receipt_path,
    )

    assert selection["camera_id"] == "camera-a"
    assert thumbnail.read_bytes() == b"camera-a"
    assert receipt["decision"] == "not_reviewed"
    assert receipt["ai_visual_review_completed"] is False
    assert receipt["review_provider_call_performed"] is False
    assert receipt["warning_label"] == PAUSED_UNGRADED_WARNING
    assert paused_review_receipt_valid(
        receipt,
        publisher_instance_id="104",
        minimum_frame_count=3,
        thumbnail_digest=_sha256(thumbnail),
    )


def test_paused_mode_finishes_stage_without_calling_visual_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root, _checkpoint, fixture = _materialize_diagnostic_checkpoint(
        tmp_path
    )
    stage_input = json.loads(Path(fixture["stage_path"]).read_text(encoding="utf-8"))
    stage_input["stage"] = {
        "stage_id": "stage-1",
        "adapter": {"id": "artifixer3d_observed_object_removal"},
    }
    stage_input["construction_envelope"]["render_inputs_result"] = fixture[
        "render_result"
    ]
    stage_input["construction_envelope"]["request"] = {
        "appearance_review_override": {
            "mode": "paused_ungraded",
            "scope": "artifixer_appearance_only",
            "ungraded_publication_acknowledged": True,
            "review_provider_call_permitted": False,
            "warning_label": PAUSED_UNGRADED_WARNING,
        }
    }
    stage_path = tmp_path / "stage-input.json"
    stage_path.write_text(json.dumps(stage_input), encoding="utf-8")
    dependencies_path = tmp_path / "dependencies.json"
    dependencies_path.write_text("[]\n", encoding="utf-8")
    output_root = tmp_path / "output"
    output_root.mkdir()
    package_root = tmp_path / "package"
    package_root.mkdir()
    semantic_request = json.loads(
        Path(fixture["request_path"]).read_text(encoding="utf-8")
    )
    locality_receipt = tmp_path / "locality.json"
    locality_receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        driver,
        "materialize_semantic_locality_seal",
        lambda **kwargs: {
            "receipt": {"receipt_digest": "sha256:" + "1" * 64},
            "receipt_path": str(locality_receipt),
            "semantic_teacher_frames_root": str(
                kwargs["semantic_output_root"]
                / "tasks"
                / "remove-source-object-104"
            ),
        },
    )
    monkeypatch.setattr(
        driver,
        "_write_execution_authority",
        lambda **_kwargs: ({}, tmp_path / "rights.json", "839873"),
    )
    monkeypatch.setattr(
        driver, "materialize_provider_render_handoff", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        driver,
        "_materialize_preflight",
        lambda **_kwargs: ({}, "remove-source-object-104"),
    )
    monkeypatch.setattr(
        driver,
        "materialize_artifixer3d_candidate_inputs",
        lambda **_kwargs: {"receipt_digest": "sha256:" + "2" * 64},
    )
    monkeypatch.setattr(
        driver, "_semantic_rights_and_request", lambda **_kwargs: tmp_path / "packet"
    )

    def write_request(**_kwargs):
        packet = tmp_path / "packet"
        packet.mkdir(exist_ok=True)
        path = packet / "semantic_teacher_image_edit_runtime_request.v1.json"
        path.write_text(json.dumps(semantic_request), encoding="utf-8")
        return path

    monkeypatch.setattr(driver, "_semantic_runtime_request", write_request)
    monkeypatch.setattr(
        driver,
        "materialize_whole_frame_semantic_teacher_receipt",
        lambda **kwargs: Path(kwargs["output_path"]).write_text(
            "{}\n", encoding="utf-8"
        ),
    )
    native = tmp_path / "configured.usdz"
    native.write_bytes(b"configured")
    frames = []
    for index in range(8):
        path = tmp_path / f"frame-{index}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(
            {
                "camera_id": f"camera-{index}",
                "final_frame": {"path": str(path), "sha256": _sha256(path)},
            }
        )
    monkeypatch.setattr(
        driver,
        "_run_artifixer_training_round",
        lambda **_kwargs: {
            "review_frames": frames,
            "native_appearance_source": native,
            "post_training_binding_digest": "sha256:" + "3" * 64,
            "runtime_result": {},
        },
    )
    monkeypatch.setattr(
        driver,
        "_run_artifixer_visual_review_round",
        lambda **_kwargs: pytest.fail("paused mode called the visual reviewer"),
    )

    result = driver.execute_artifixer_component(
        environment={
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(stage_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(
                dependencies_path
            ),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output_root),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(
                output_root / "result.json"
            ),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT": str(package_root),
            "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT": str(
                checkpoint_root
            ),
        }
    )

    assert result["status"] == "completed"
    assert result["appearance_review_status"] == "paused_ungraded"
    assert result["appearance_quality_graded"] is False
    removal = json.loads(
        (output_root / "appearance_removal_receipt.v1.json").read_text()
    )
    assert removal["status"] == "completed_ungraded_generated_appearance_edit"
    assert removal["review_provider_call_performed"] is False


def test_generic_render_contract_feeds_released_artifixer_inputs(tmp_path: Path) -> None:
    envelope, configuration = _inputs(tmp_path)
    authority_path = tmp_path / "authority.json"
    authority, _rights_path, scene_id = _write_execution_authority(
        envelope=envelope,
        configuration=configuration,
        destination=authority_path,
    )
    preflight_path = tmp_path / "preflight.json"
    preflight, task_id = _materialize_preflight(
        envelope=envelope,
        configuration=configuration,
        authority=authority,
        authority_path=authority_path,
        output_path=preflight_path,
    )
    candidate_root = tmp_path / "candidate"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight_path,
        output_root=candidate_root,
    )

    assert scene_id == "839873"
    assert task_id == "remove-source-object-104"
    assert preflight["replacement_object_count"] == 1
    assert candidate["publisher_scene_id"] == "839873"
    assert candidate["tasks"][0]["camera_count"] == 1
    assert candidate["receipt_digest"] == canonical_digest(candidate, digest_field="receipt_digest")


def test_generic_candidate_feeds_existing_semantic_teacher_packet(tmp_path: Path) -> None:
    envelope, configuration = _inputs(tmp_path)
    authority_path = tmp_path / "authority.json"
    authority, _rights_path, scene_id = _write_execution_authority(
        envelope=envelope,
        configuration=configuration,
        destination=authority_path,
    )
    preflight_path = tmp_path / "preflight.json"
    _materialize_preflight(
        envelope=envelope,
        configuration=configuration,
        authority=authority,
        authority_path=authority_path,
        output_path=preflight_path,
    )
    candidate_root = tmp_path / "candidate"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight_path,
        output_root=candidate_root,
    )
    candidate_path = candidate_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    packet_root = _semantic_rights_and_request(
        candidate=candidate,
        candidate_path=candidate_path,
        registry_path=(
            Path(__file__).resolve().parents[1]
            / "docs/arm_decision_proof_v1/manifests/image_editor_backends.v1.json"
        ),
        configuration=configuration,
        publisher_scene_id=scene_id,
        output_root=tmp_path,
    )
    packet = json.loads(
        (packet_root / "fresh_scene_semantic_teacher_image_edit_packet.v1.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_request_path = driver._semantic_runtime_request(
        packet_root=packet_root,
        source_commit="a" * 40,
        maximum_cost_usd=2.4,
        expected_request_cost_usd=0.22,
    )
    runtime_request = json.loads(runtime_request_path.read_text(encoding="utf-8"))

    assert packet["task_count"] == 1
    assert packet["request_count"] == 1
    assert packet["backend"]["registry_entry"]["backend_id"].startswith("openai_gpt_image_2")
    assert packet["raw_nonredistributable_source_bytes_included"] is False
    assert runtime_request["max_parallel_requests"] == 4
    assert runtime_request["maximum_cost_usd"] == 2.4
    assert runtime_request["expected_request_cost_usd"] == 0.22


def test_visual_review_uses_the_scene_lanes_exclusive_cost_scope() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver",
            fromlist=["__file__"],
        ).__file__
    ).read_text(encoding="utf-8")

    assert _VISUAL_REVIEW_COST_SCOPE == (
        "task_evaluation_scene_configuration_artifixer_visual_review"
    )
    assert "cost_lane_id=_VISUAL_REVIEW_COST_SCOPE" in source
    assert "paid_resource_class=_VISUAL_REVIEW_COST_SCOPE" in source
    assert "require_zero_baseline=False" in source
    assert "review_attestation_path = materialize_stage_scope_attestation(" in source
    assert "openai_cost_scope_attestation_path=review_attestation_path" in source
    assert 'review_scope["attestation_file"]' not in source


def test_selected_task_thumbnail_is_an_exact_reviewed_frame_copy(
    tmp_path: Path,
) -> None:
    frames: list[dict[str, object]] = []
    for index in range(8):
        path = tmp_path / f"camera-{index}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(
            {
                "camera_id": f"camera-{index}",
                "final_frame": {
                    "path": str(path),
                    "sha256": _sha256(path),
                },
            }
        )
    selected = frames[5]
    selected_frame = selected["final_frame"]
    assert isinstance(selected_frame, dict)
    receipt = {
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {
            "camera_id": selected["camera_id"],
            "frame_sha256": selected_frame["sha256"],
            "rationale": "The configured task surface is clearly visible.",
        },
        "reviewer": {
            "kind": "ai",
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
    }
    destination = tmp_path / "configured_task_thumbnail.png"

    selection = _materialize_selected_task_thumbnail(
        review_receipt=receipt,
        review_frames=frames,
        destination=destination,
    )

    assert destination.read_bytes() == (tmp_path / "camera-5.png").read_bytes()
    assert _sha256(destination) == selected_frame["sha256"]
    assert selection["camera_id"] == "camera-5"
    assert selection["derived_appearance_evidence"] is True
    assert selection["capture_or_physical_evidence"] is False


def test_failed_artifixer_runtime_streams_survive_into_the_stage_log(
    tmp_path: Path, capsys
) -> None:
    """A paid failure must carry its own cause.

    Run ...-f1e07c7f-...-171647Z failed the runtime acceptance check and the
    exported receipts said only scene_configuration_artifixer_runtime_failed:
    the subprocess's streams were captured by the driver and discarded, so the
    $0.74 GPU run left nothing to diagnose with. The driver's stderr feeds
    stage_producer.log, which is proven to survive into the exported zip.
    """

    opaque_file_secret = "opaque-file-credential-value-839873"
    completed = types.SimpleNamespace(
        returncode=1,
        stdout=(
            "optimizer step 40\n"
            "token=sk-proj-supersecret123456\n"
            f"file_token={opaque_file_secret}\n"
        ),
        stderr=(
            "CUDA error: out of memory at step 41\n"
            f"credential={opaque_file_secret}\n"
        ),
    )

    retained = tmp_path / "retained"
    retained.mkdir()
    driver._emit_artifixer_runtime_diagnostics(
        completed=completed,
        runtime_result_path=tmp_path / "public_scene_artifixer3d_runtime_result.json",
        retained_root=retained,
        secret_values=(opaque_file_secret,),
    )

    err = capsys.readouterr().err
    assert "scene_configuration_artifixer_runtime_diagnostics" in err
    assert '"returncode": 1' in err
    assert '"runtime_result_present": false' in err
    assert "CUDA error: out of memory at step 41" in err
    assert "optimizer step 40" in err
    assert "sk-proj-supersecret123456" not in err
    assert opaque_file_secret not in err
    assert "REDACTED_SECRET" in err
    # The full streams survive as files even when the inline relay truncates.
    full_stderr = (retained / "artifixer_runtime_stderr.log").read_text(
        encoding="utf-8"
    )
    assert "CUDA error: out of memory at step 41" in full_stderr
    assert "sk-proj-supersecret123456" not in full_stderr
    assert opaque_file_secret not in full_stderr
    assert "REDACTED_SECRET" in full_stderr
    full_stdout = (retained / "artifixer_runtime_stdout.log").read_text(
        encoding="utf-8"
    )
    assert opaque_file_secret not in full_stdout


def test_download_progress_noise_is_filtered_from_the_inline_tail(
    tmp_path: Path, capsys
) -> None:
    """wget progress rows crowded the marker out of run ...-183325Z's log."""

    progress = "\n".join(
        f" {50000 + 50 * i}K .......... .......... .......... 83%  266M 0s"
        for i in range(2_000)
    )
    completed = types.SimpleNamespace(
        returncode=1,
        stdout="",
        stderr=progress + "\nRuntimeError: optimizer diverged at step 7\n",
    )

    driver._emit_artifixer_runtime_diagnostics(
        completed=completed,
        runtime_result_path=tmp_path / "missing.json",
        retained_root=tmp_path,
    )

    err = capsys.readouterr().err
    assert "scene_configuration_artifixer_runtime_diagnostics" in err
    assert "RuntimeError: optimizer diverged at step 7" in err
    # The full unfiltered stream is still on disk.
    assert "266M" in (tmp_path / "artifixer_runtime_stderr.log").read_text(
        encoding="utf-8"
    )


def test_successful_artifixer_runtime_emits_no_diagnostics(
    tmp_path: Path, capsys
) -> None:
    result_path = tmp_path / "public_scene_artifixer3d_runtime_result.json"
    result_path.write_text(
        json.dumps({"status": driver.ARTIFIXER_RUNTIME_ACCEPTED_STATUS}),
        encoding="utf-8",
    )
    completed = types.SimpleNamespace(returncode=0, stdout="fine", stderr="")

    driver._emit_artifixer_runtime_diagnostics(
        completed=completed, runtime_result_path=result_path, retained_root=tmp_path
    )

    assert capsys.readouterr().err == ""
    assert not (tmp_path / "artifixer_runtime_stderr.log").exists()
