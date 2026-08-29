from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.semantic_teacher_image_edit_worker import (
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_artifixer_ai_visual_review import (
    DUAL_TARGET_REVIEW_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION as REVIEW_EXECUTION_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_selective_repair import (
    STRICT_LOCALITY_PROMPT_POLICY,
    TaskEvaluationArtifixerSelectiveRepairError,
    materialize_selective_repair_request,
    merge_selective_repair_outputs,
)
from blueprint_pipeline.task_evaluation_scene_configuration_semantic_locality import (
    SEMANTIC_LOCALITY_POLICY,
    materialize_semantic_locality_seal,
    seal_semantic_teacher_frame,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        record["path"] = str(path)
    else:
        record["relative_path"] = path.relative_to(root).as_posix()
    return record


def _fixture(
    tmp_path: Path,
    *,
    rejected_count: int = 1,
    frame_count: int = 4,
    implausible_repair: bool = False,
) -> dict[str, object]:
    request_root = tmp_path / "request"
    source_root = tmp_path / "source-output"
    request_root.mkdir(parents=True)
    source_root.mkdir(parents=True)
    task_id = "remove-source-object-104"
    request_frames = []
    review_frames = []
    decisions = []
    semantic_frames = []
    for index in range(frame_count):
        camera_id = f"camera-{index}"
        source = tmp_path / f"source-{index}.png"
        mask = tmp_path / f"mask-{index}.png"
        final = tmp_path / f"final-{index}.png"
        Image.new("RGB", (8, 8), color=(40 + index, 50, 60)).save(source)
        exact = Image.new("L", (8, 8), color=0)
        exact.putpixel((3, 3), 255)
        exact.putpixel((4, 3), 255)
        mask.parent.mkdir(parents=True, exist_ok=True)
        exact.save(mask)
        Image.new("RGB", (8, 8), color=(70 + index, 80, 90)).save(final)
        staged_source = request_root / f"source-{index}.png"
        staged_mask = request_root / f"encoded-mask-{index}.png"
        Image.open(source).convert("RGB").save(staged_source)
        rgba = Image.new("RGBA", (8, 8), color=(255, 255, 255, 255))
        rgba.putpixel((3, 3), (255, 255, 255, 0))
        rgba.putpixel((4, 3), (255, 255, 255, 0))
        rgba.save(staged_mask)
        request_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "input_rgb": _record(staged_source, root=request_root),
                "edit_mask": _record(staged_mask, root=request_root),
            }
        )
        review_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "source_frame": _record(source),
                "exact_repair_mask": _record(mask),
                "final_frame": _record(final),
            }
        )
        rejected = index < rejected_count
        decisions.append(
            {
                "task_id": task_id,
                "camera_id": camera_id,
                "frame_sha256": _sha256(final),
                "orientation_is_upright": True,
                "source_object_absent": True,
                "repair_is_locally_plausible": not (
                    rejected and implausible_repair
                ),
                "preserves_non_target_content": (
                    True if implausible_repair else not rejected
                ),
                "decision": "rejected" if rejected else "accepted",
                "rationale": (
                    "Table veining changed outside the mask."
                    if rejected
                    else "The scene outside the target is preserved."
                ),
            }
        )
        semantic_path = source_root / "tasks" / task_id / f"{index:05d}.png"
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic = Image.open(source).convert("RGB")
        semantic.putpixel((3, 3), (100 + index, 110, 120))
        semantic.putpixel((4, 3), (100 + index, 110, 120))
        if index == 1:
            semantic = Image.new("RGB", (8, 8), color=(220, 10, 10))
        semantic.save(semantic_path)
        semantic_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "source_rgb_sha256": request_frames[-1]["input_rgb"]["sha256"],
                "edit_mask_sha256": request_frames[-1]["edit_mask"]["sha256"],
                "terminal_state": "completed_unreviewed_candidate",
                "semantic_teacher_frame": _record(
                    semantic_path, root=source_root
                ),
            }
        )
    request = {
        "schema_version": RUNTIME_REQUEST_SCHEMA_VERSION,
        "source_commit_sha": "a" * 40,
        "source_packet_digest": "sha256:" + "1" * 64,
        "backend": {
            "registry_entry": {
                "backend_id": "openai-gpt-image",
                "capability": "semantic_teacher_image_edit",
            },
            "backend_entry_digest": "sha256:" + "2" * 64,
            "execution": {
                "mask_encoding": "rgba_alpha_zero_edit_region_png",
            },
        },
        "prompt_policy": "first-pass",
        "prompt": "Remove the masked object.",
        "tasks": [
            {
                "task_id": task_id,
                "camera_count": frame_count,
                "frames": request_frames,
            }
        ],
        "max_parallel_requests": 4,
        "maximum_cost_usd": 2.4,
        "expected_request_cost_usd": 0.22,
        "retry_count": 0,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path = request_root / f"{RUNTIME_REQUEST_SCHEMA_VERSION}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    review = {
        "schema_version": DUAL_TARGET_REVIEW_SCHEMA_VERSION,
        "status": "paired_target_frames_pending_independent_visual_review",
        "publisher_scene_id": "839873",
        "review_scope": "source_anchor_exact_mask_and_generated_full_frame_comparison",
        "tasks": [
            {
                "task_id": task_id,
                "physical_camera_count": frame_count,
                "frames": review_frames,
            }
        ],
        "outside_support_invariance_proven": False,
        "outside_support_invariance_claimed": False,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    review["receipt_digest"] = canonical_digest(
        review, digest_field="receipt_digest"
    )
    review_path = tmp_path / "review-input.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    execution = {
        "schema_version": REVIEW_EXECUTION_SCHEMA_VERSION,
        "status": "completed",
        "decision": "rejected",
        "task_id": task_id,
        "final_composite_receipt_digest": review["receipt_digest"],
        "provider_called": True,
        "response_store": False,
        "tracing_disabled": True,
        "raw_secret_values_recorded": False,
        "frames": decisions,
        "execution_digest": "",
    }
    execution["execution_digest"] = canonical_digest(
        execution, digest_field="execution_digest"
    )
    execution_path = tmp_path / "review-execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    source_result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "completed_unreviewed_semantic_teacher_candidates",
        "source_runtime_request_digest": request["request_digest"],
        "backend_id": "openai-gpt-image",
        "model_snapshot": "gpt-image-2-2026-04-21",
        "request_count": frame_count,
        "computed_editor_cost_usd": 0.22 * frame_count,
        "tasks": [
            {
                "task_id": task_id,
                "camera_count": frame_count,
                "frames": semantic_frames,
            }
        ],
        "result_digest": "",
    }
    source_result["result_digest"] = canonical_digest(
        source_result, digest_field="result_digest"
    )
    locality = materialize_semantic_locality_seal(
        semantic_runtime_request_path=request_path,
        semantic_runtime_result=source_result,
        semantic_output_root=source_root,
        output_root=tmp_path / "locality-sealed",
    )
    return {
        "task_id": task_id,
        "review_path": review_path,
        "execution_path": execution_path,
        "request_path": request_path,
        "source_root": source_root,
        "source_result": source_result,
        "locality": locality,
    }


def test_semantic_locality_seal_restores_false_accepted_non_target_pixels(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    sealed = fixture["locality"]

    receipt = sealed["receipt"]
    assert receipt["policy"] == SEMANTIC_LOCALITY_POLICY
    assert receipt["raw_outside_support_change_frame_count"] == 1
    assert receipt["deterministic_selective_repair_frame_count"] == 1
    assert receipt["all_non_target_source_pixels_preserved_exactly"] is True
    sealed_root = Path(sealed["semantic_teacher_frames_root"])
    request_root = Path(fixture["request_path"]).parent
    raw_root = Path(fixture["source_root"])
    for index in range(4):
        with Image.open(request_root / f"source-{index}.png") as image:
            source = image.convert("RGB")
        with Image.open(
            raw_root / "tasks" / fixture["task_id"] / f"{index:05d}.png"
        ) as image:
            raw = image.convert("RGB")
        with Image.open(sealed_root / f"{index:05d}.png") as image:
            candidate = image.convert("RGB")
        for y in range(8):
            for x in range(8):
                if (x, y) in {(3, 3), (4, 3)}:
                    assert candidate.getpixel((x, y)) == raw.getpixel((x, y))
            else:
                assert candidate.getpixel((x, y)) == source.getpixel((x, y))


def test_semantic_locality_feathers_inside_mask_without_changing_outside(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    raw_path = tmp_path / "raw.png"
    output_path = tmp_path / "sealed.png"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(source_path)
    mask = Image.new("L", (64, 64), color=0)
    for y in range(8, 56):
        for x in range(8, 56):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(raw_path)

    result = seal_semantic_teacher_frame(
        source_path=source_path,
        mask_path=mask_path,
        raw_teacher_path=raw_path,
        mask_encoding="binary_white_edit_region_png",
        output_path=output_path,
    )

    with Image.open(output_path) as image:
        sealed = image.convert("RGB")
    assert result["inner_feather_radius_pixels"] > 0
    assert sealed.getpixel((7, 32)) == (100, 100, 100)
    assert 100 <= sealed.getpixel((8, 32))[0] < 110
    assert sealed.getpixel((32, 32)) == (200, 200, 200)


def test_locality_only_rejection_repairs_one_camera_and_reuses_the_rest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    staged = materialize_selective_repair_request(
        review_input_path=fixture["review_path"],
        review_execution_path=fixture["execution_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result=fixture["source_result"],
        semantic_locality_receipt_path=fixture["locality"]["receipt_path"],
        expected_request_cost_usd=0.22,
        maximum_stage_cost_usd=2.4,
        output_root=tmp_path / "repair-request",
    )

    assert staged["plan"]["selected_frame_count"] == 2
    assert staged["plan"]["selected_frames"][0]["camera_id"] == "camera-0"
    assert staged["plan"]["selected_frames"][1]["camera_id"] == "camera-1"
    assert staged["plan"]["selected_frames"][1]["selection_reasons"] == [
        "deterministic_gross_outside_mask_change"
    ]
    assert staged["plan"]["additional_provider_request_cap"] == 2
    assert staged["plan"]["second_repair_round_permitted"] is False
    assert staged["repair_request"]["prompt_policy"] == STRICT_LOCALITY_PROMPT_POLICY
    assert len(staged["repair_request"]["tasks"][0]["frames"]) == 2
    assert staged["repair_request"]["tasks"][0]["frames"][0]["frame_index"] == 0

    repair_root = tmp_path / "repair-output"
    repaired_frame = repair_root / "tasks" / fixture["task_id"] / "00000.png"
    repaired_frame.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(201, 202, 203)).save(repaired_frame)
    repair_result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "completed_unreviewed_semantic_teacher_candidates",
        "source_runtime_request_digest": staged["repair_request"]["request_digest"],
        "request_count": 2,
        "attempted_request_count": 2,
        "successful_request_count": 2,
        "failed_request_count": 0,
        "billing_qualified": True,
        "retry_count": 0,
        "tasks": [
            {
                "task_id": fixture["task_id"],
                "camera_count": 2,
                "frames": [
                    {
                        "frame_index": 0,
                        "camera_id": "camera-0",
                        "terminal_state": "completed_unreviewed_candidate",
                        "semantic_teacher_frame": _record(
                            repaired_frame, root=repair_root
                        ),
                    },
                    {
                        "frame_index": 1,
                        "camera_id": "camera-1",
                        "terminal_state": "completed_unreviewed_candidate",
                        "semantic_teacher_frame": _record(
                            repaired_frame, root=repair_root
                        ),
                    },
                ],
            }
        ],
        "result_digest": "",
    }
    repair_result["result_digest"] = canonical_digest(
        repair_result, digest_field="result_digest"
    )
    (repair_root / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json").write_text(
        json.dumps(repair_result), encoding="utf-8"
    )
    merged = merge_selective_repair_outputs(
        plan_path=staged["plan_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_locality_receipt_path=fixture["locality"]["receipt_path"],
        source_semantic_output_root=Path(
            fixture["locality"]["semantic_teacher_frames_root"]
        ).parents[1],
        source_semantic_result=fixture["source_result"],
        repair_output_root=repair_root,
        output_root=tmp_path / "merged",
    )

    receipt = merged["receipt"]
    assert receipt["repair_round"] == 1
    assert receipt["repaired_frame_count"] == 2
    assert receipt["reused_frame_count"] == 2
    assert receipt["all_non_target_source_pixels_preserved_exactly"] is True
    merged_root = Path(merged["semantic_teacher_frames_root"])
    request_root = Path(fixture["request_path"]).parent
    with Image.open(repaired_frame) as image:
        repaired_pixels = image.convert("RGB")
    for index in (0, 1):
        with Image.open(request_root / f"source-{index}.png") as image:
            source_pixels = image.convert("RGB")
        with Image.open(merged_root / f"{index:05d}.png") as image:
            merged_pixels = image.convert("RGB")
        assert merged_pixels.getpixel((3, 3)) == repaired_pixels.getpixel((3, 3))
        assert merged_pixels.getpixel((0, 0)) == source_pixels.getpixel((0, 0))
    for index in range(2, 4):
        assert _sha256(merged_root / f"{index:05d}.png") == _sha256(
            Path(fixture["source_root"])
            / "tasks"
            / fixture["task_id"]
            / f"{index:05d}.png"
        )


def test_non_locality_rejection_is_not_repaired(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    execution_path = Path(fixture["execution_path"])
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["frames"][0]["source_object_absent"] = False
    execution["execution_digest"] = canonical_digest(
        execution, digest_field="execution_digest"
    )
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    with pytest.raises(
        TaskEvaluationArtifixerSelectiveRepairError,
        match="scene_configuration_artifixer_selective_repair_ineligible_rejection",
    ):
        materialize_selective_repair_request(
            review_input_path=fixture["review_path"],
            review_execution_path=execution_path,
            semantic_runtime_request_path=fixture["request_path"],
            semantic_runtime_result=fixture["source_result"],
            semantic_locality_receipt_path=fixture["locality"]["receipt_path"],
            expected_request_cost_usd=0.22,
            maximum_stage_cost_usd=2.4,
            output_root=tmp_path / "repair-request",
        )


def test_all_eight_implausible_fills_get_one_bounded_repair_round(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path / "all-eight",
        rejected_count=8,
        frame_count=8,
        implausible_repair=True,
    )
    staged = materialize_selective_repair_request(
        review_input_path=fixture["review_path"],
        review_execution_path=fixture["execution_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result=fixture["source_result"],
        semantic_locality_receipt_path=fixture["locality"]["receipt_path"],
        expected_request_cost_usd=0.22,
        maximum_stage_cost_usd=4.0,
        output_root=tmp_path / "all-eight-repair",
    )

    assert staged["plan"]["selected_frame_count"] == 8
    assert staged["plan"]["additional_provider_request_cap"] == 8
    assert all(
        "independent_visual_review_local_plausibility_rejection"
        in row["selection_reasons"]
        for row in staged["plan"]["selected_frames"]
    )


def test_repair_refuses_insufficient_remaining_cost(tmp_path: Path) -> None:

    insufficient = _fixture(tmp_path / "cost")
    with pytest.raises(
        TaskEvaluationArtifixerSelectiveRepairError,
        match="scene_configuration_artifixer_selective_repair_cost_insufficient",
    ):
        materialize_selective_repair_request(
            review_input_path=insufficient["review_path"],
            review_execution_path=insufficient["execution_path"],
            semantic_runtime_request_path=insufficient["request_path"],
            semantic_runtime_result=insufficient["source_result"],
            semantic_locality_receipt_path=insufficient["locality"]["receipt_path"],
            expected_request_cost_usd=0.22,
            maximum_stage_cost_usd=1.0,
            output_root=tmp_path / "insufficient-repair",
        )


def test_repair_packet_is_self_contained_for_the_worker_root(
    tmp_path: Path,
) -> None:
    """Every referenced frame must resolve beside the repair request itself.

    ``execute_semantic_teacher_image_edits`` resolves each frame's
    ``relative_path`` against ``request_path.parent``.  A production run
    (scene 839873, 2026-08-29) staged the repair request into a directory
    holding only JSON, so the first bounded repair round -- the round the
    corrected review budget had finally made reachable -- died on
    ``semantic_teacher_runtime_input_invalid`` after the GPU was already paid
    for.  The staged packet has to carry the bytes it names.
    """

    fixture = _fixture(tmp_path, rejected_count=2, implausible_repair=True)
    output_root = tmp_path / "repair-request"
    staged = materialize_selective_repair_request(
        review_input_path=fixture["review_path"],
        review_execution_path=fixture["execution_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result=fixture["source_result"],
        semantic_locality_receipt_path=fixture["locality"]["receipt_path"],
        expected_request_cost_usd=0.22,
        maximum_stage_cost_usd=2.4,
        output_root=output_root,
    )

    request_path = Path(staged["repair_request_path"])
    # The worker's own rule, not a convenience path.
    root = request_path.parent
    frames = staged["repair_request"]["tasks"][0]["frames"]
    assert frames
    for frame in frames:
        for member in ("input_rgb", "edit_mask"):
            record = frame[member]
            staged_path = root / record["relative_path"]
            assert staged_path.is_file(), f"{member} missing from repair packet"
            assert not staged_path.is_symlink()
            assert staged_path.stat().st_size == record["size_bytes"]
            assert _sha256(staged_path) == record["sha256"]


def test_repair_packet_bytes_match_the_original_semantic_packet(
    tmp_path: Path,
) -> None:
    """The repair edits the same source pixels the first pass was given."""

    fixture = _fixture(tmp_path, rejected_count=2, implausible_repair=True)
    staged = materialize_selective_repair_request(
        review_input_path=fixture["review_path"],
        review_execution_path=fixture["execution_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result=fixture["source_result"],
        semantic_locality_receipt_path=fixture["locality"]["receipt_path"],
        expected_request_cost_usd=0.22,
        maximum_stage_cost_usd=2.4,
        output_root=tmp_path / "repair-request",
    )

    source_root = Path(fixture["request_path"]).parent
    root = Path(staged["repair_request_path"]).parent
    for frame in staged["repair_request"]["tasks"][0]["frames"]:
        for member in ("input_rgb", "edit_mask"):
            relative = frame[member]["relative_path"]
            assert (root / relative).read_bytes() == (
                source_root / relative
            ).read_bytes()
