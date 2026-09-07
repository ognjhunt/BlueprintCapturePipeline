import copy
import math

import pytest

import blueprint_pipeline.semantic_target_training_selection as selection
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def inputs(rejected=(7,), *, isolated=False):
    frames, decisions, poses = [], [], []
    for i in range(16):
        cam = f"camera-{i:02d}"
        digest = "sha256:" + f"{i:064x}"
        frames.append({"camera_id": cam, "final_frame": {"sha256": digest}})
        decisions.append(
            {
                "camera_id": cam,
                "frame_sha256": digest,
                "orientation_is_upright": True,
                "decision": "rejected" if i in rejected else "accepted",
                "source_object_absent": True,
                "repair_is_locally_plausible": i not in rejected,
                "preserves_non_target_content": True,
                "rationale": "wrong material" if i in rejected else "minor seam; usable target",
            }
        )
        angle = math.pi if isolated and i in rejected else 0
        poses.append(
            {
                "camera_id": cam,
                "transform_matrix": [
                    [math.cos(angle), 0, math.sin(angle), i * 0.1],
                    [0, 1, 0, 0],
                    [-math.sin(angle), 0, math.cos(angle), 2],
                    [0, 0, 0, 1],
                ],
            }
        )
    review = _seal(
        {"review_phase": "pre_training_semantic_targets", "tasks": [{"frames": frames}]},
        "receipt_digest",
    )
    execution = _seal(
        {
            "review_phase": "pre_training_semantic_targets",
            "provider_called": True,
            "schema_version": "task_evaluation_artifixer_ai_visual_review_execution.v1",
            "status": "completed",
            "all_frames_upright": True,
            "decision": "rejected",
            "response_store": False,
            "tracing_disabled": True,
            "raw_secret_values_recorded": False,
            "frames": decisions,
            "final_composite_receipt_digest": review["receipt_digest"],
        },
        "execution_digest",
    )
    return dict(
        review_input=review,
        review_execution=execution,
        transforms={"frames": poses},
        minimum_views=8,
    )


def test_one_bad_view_admitted_as_exclusion_with_all_final_views_retained():
    kwargs = inputs()
    v = selection.build_selection(**kwargs)
    assert v["excluded_camera_ids"] == ["camera-07"]
    assert len(v["approved_camera_ids"]) == 15
    assert len(v["final_review_camera_ids"]) == 16
    assert v["appearance_repair_qualified"] is False
    teachers = [
        {"camera_id": r["camera_id"], "whole_frame_semantic_teacher": r["final_frame"]}
        for r in kwargs["review_input"]["tasks"][0]["frames"]
    ]
    assert selection.validate_selection(
        v, transforms=kwargs["transforms"], teacher_frames=teachers
    ) == {"camera-07"}
    forged = copy.deepcopy(v)
    forged["approved_camera_ids"].append("camera-07")
    forged["excluded_camera_ids"] = []
    _seal(forged, "selection_digest")
    with pytest.raises(ValueError, match="review_selection_mismatch"):
        selection.validate_selection(
            forged, transforms=kwargs["transforms"], teacher_frames=teachers
        )


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (inputs(rejected=tuple(range(5))), "insufficient_approved_views"),
        (inputs(isolated=True), "excluded_view_uncovered"),
    ],
)
def test_insufficient_or_uncovered_views_still_block(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        selection.build_selection(**kwargs)


def test_duplicate_poses_cannot_manufacture_coverage():
    kwargs = inputs()
    for row in kwargs["transforms"]["frames"]:
        row["transform_matrix"][0][3] = 0
    with pytest.raises(ValueError, match="distinct_approved_poses"):
        selection.build_selection(**kwargs)


def test_rejected_orientation_is_not_repaired_by_pixel_exclusion():
    kwargs = inputs()
    kwargs["review_execution"]["frames"][7]["orientation_is_upright"] = False
    _seal(kwargs["review_execution"], "execution_digest")
    with pytest.raises(ValueError, match="orientation_invalid"):
        selection.build_selection(**kwargs)


@pytest.mark.parametrize("second_rejected", [(), (7,), (0, 1, 2, 3, 4)])
def test_pretraining_recovery_is_bounded_and_remaining_bad_view_can_be_excluded(
    tmp_path, monkeypatch, second_rejected
):
    import json
    from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_driver as driver
    from blueprint_pipeline import public_scene_artifixer3d_dual_target_inputs as dual

    first = inputs(rejected=(7,))
    second = inputs(rejected=second_rejected)
    calls, repairs, teachers = [], [], []

    def review(**kwargs):
        data = first if not calls else second
        calls.append(kwargs)
        root = kwargs["round_root"]
        root.mkdir(parents=True)
        path = root / "execution.json"
        path.write_text(json.dumps(data["review_execution"]))
        accepted = not (data is first or second_rejected)
        return {
            "review_input": data["review_input"],
            "review_input_path": root / "input.json",
            "review": {
                "decision": "accepted" if accepted else "rejected",
                "review_receipt": {"path": "accepted"} if accepted else None,
                "execution_receipt": {"path": str(path)},
            },
        }

    frames = first["review_input"]["tasks"][0]["frames"]

    def repair(**kwargs):
        repairs.append(kwargs)
        return (
            {},
            {"result_digest": "sha256:" + "b" * 64},
            {
                "semantic_teacher_frames_root": str(tmp_path / "merged/tasks/task"),
                "receipt_path": str(tmp_path / "merged/receipt.json"),
                "receipt": {
                    "merge_digest": "sha256:" + "c" * 64,
                    "frame_inventory": [
                        {
                            "camera_id": row["camera_id"],
                            "relative_path": f"tasks/task/{i:05d}.png",
                            **row["final_frame"],
                        }
                        for i, row in enumerate(frames)
                    ],
                },
            },
        )

    monkeypatch.setattr(driver, "_review_semantic_targets_before_training", review)
    monkeypatch.setattr(driver, "_execute_bounded_semantic_target_repair", repair)
    monkeypatch.setattr(dual, "_source_task_frames", lambda task: [])
    monkeypatch.setattr(
        dual, "_validated_transforms", lambda task, frames: (first["transforms"], None)
    )
    monkeypatch.setattr(
        driver,
        "materialize_whole_frame_semantic_teacher_receipt",
        lambda **kwargs: teachers.append(kwargs),
    )
    kwargs = dict(
        locality_seal={"semantic_teacher_frames_root": "original", "receipt": {"frames": frames}},
        work=tmp_path,
        output_root=tmp_path / "output",
        publisher_scene_id="fixture",
        task_id="task",
        rights_path=tmp_path / "rights",
        configuration={"required_views": {"minimum": 8}},
        stage_input={},
        values={},
        visual_review_cap=0.96,
        semantic_request=tmp_path / "request",
        semantic_result={"result_digest": "sha256:" + "a" * 64},
        expected_frame_cost=0.3,
        semantic_cap=4.8,
        token="",
        candidate={"tasks": [{}]},
        candidate_path=tmp_path / "candidate",
        teacher_receipt_path=tmp_path / "teacher",
    )
    if len(second_rejected) > 4:
        with pytest.raises(ValueError, match="insufficient_approved_views"):
            driver._admit_semantic_training_targets(**kwargs)
        assert not teachers
    else:
        result = driver._admit_semantic_training_targets(**kwargs)
        assert result["remaining_visual_review_cap"] == pytest.approx(0.32)
        assert result["semantic_repair_used"] is True
        chosen = teachers[0]["training_view_selection"]
        assert (chosen is None) == (not second_rejected)
        if chosen:
            assert chosen["excluded_camera_ids"] == ["camera-07"]
    assert len(calls) == 2 and len(repairs) == 1
    assert all(c["max_cost_usd"] == pytest.approx(0.32) for c in calls)
