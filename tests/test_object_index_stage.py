from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blueprint_pipeline.materialization import materialize_capture_bundle
from blueprint_pipeline.object_index_stage import run_object_index_stage
from blueprint_pipeline.capture_orchestrator import run_capture_pipeline
from blueprint_pipeline.swap_orchestrator import OrchestratorConfig


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _make_video(path: Path, *, duration_seconds: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=640x480:d={duration_seconds}",
            "-vf",
            "format=yuv420p",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)


def _build_raw_capture(
    tmp_path: Path,
    *,
    scene_id: str = "scene_idx",
    capture_id: str = "cap_idx",
    workflow_name: str = "Open drawer and organize storage bins",
    privacy_limits: list[str] | None = None,
    include_arkit: bool = True,
) -> Path:
    capture_root = tmp_path / "bucket" / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_source": "iphone",
            "capture_modality": "iphone_arkit_lidar",
            "capture_tier_hint": "tier1_iphone",
            "has_lidar": include_arkit,
            "video_uri": "raw/walkthrough.mov",
            "intended_space_type": "office",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": workflow_name,
            "taskSteps": ["approach workspace", "open drawer", "organize storage bins"],
            "zone": "office workstation",
            "owner": "operator",
            "privacySecurityLimits": privacy_limits or [],
        },
    )
    _write_json(
        raw_root / "capture_context.json",
        {
            "sceneId": scene_id,
            "captureId": capture_id,
            "captureSource": "iphoneVideo",
            "captureModality": "iphone_arkit_lidar" if include_arkit else "glasses_video_only",
        },
    )
    _write_json(raw_root / "capture_upload_complete.json", {"sceneId": scene_id, "captureId": capture_id})
    _make_video(raw_root / "walkthrough.mov")

    if include_arkit:
        frames_path = raw_root / "arkit" / "frames.jsonl"
        poses_path = raw_root / "arkit" / "poses.jsonl"
        intrinsics_path = raw_root / "arkit" / "intrinsics.json"
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        poses = []
        for idx in range(4):
            timestamp = idx * 0.4
            frames.append(
                {
                    "frameIndex": idx,
                    "timestamp": timestamp,
                    "imageResolution": [640, 480],
                    "intrinsics": [600.0, 0.0, 0.0, 0.0, 600.0, 0.0, 320.0, 240.0, 1.0],
                    "cameraTransform": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        idx * 0.2,
                        0.0,
                        0.0,
                        1.0,
                    ],
                }
            )
            poses.append(
                {
                    "frameIndex": idx,
                    "timestamp": timestamp,
                    "transform": [
                        [1.0, 0.0, 0.0, idx * 0.2],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            )
        frames_path.write_text("".join(json.dumps(item) + "\n" for item in frames), encoding="utf-8")
        poses_path.write_text("".join(json.dumps(item) + "\n" for item in poses), encoding="utf-8")
        intrinsics_path.write_text("{}", encoding="utf-8")
    return capture_root


def _materialize_capture(capture_root: Path) -> None:
    materialize_capture_bundle(
        bucket="bucket",
        scene_id=capture_root.parts[-3],
        capture_id=capture_root.parts[-1],
        gcs_root=capture_root.parents[4],
    )


def _write_fake_backend(path: Path, *, detections: list[dict]) -> None:
    payload = json.dumps({"detections": detections})
    _write_executable(
        path,
        f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

input_payload = json.loads(Path(sys.argv[1]).read_text())
keyframes = input_payload.get("keyframes", [])
frame_indexes = [item.get("frame_index", 0) for item in keyframes]
detections = {payload}["detections"]
for item in detections:
    if item.get("frame_index") == "FIRST":
        item["frame_index"] = frame_indexes[0]
    elif item.get("frame_index") == "SECOND":
        item["frame_index"] = frame_indexes[min(1, len(frame_indexes)-1)]
    elif item.get("frame_index") == "THIRD":
        item["frame_index"] = frame_indexes[min(2, len(frame_indexes)-1)]
Path(sys.argv[2]).write_text(json.dumps({{"detections": detections}}, indent=2), encoding="utf-8")
""",
    )


def test_object_index_stage_builds_canonical_index_with_external_backends(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_raw_capture(tmp_path)
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo"
    grounding_script = tmp_path / "bin" / "fake-grounding"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "storage bin", "score": 0.82, "bbox_xyxy": [60, 90, 260, 320], "source_prompt": "bin"},
            {"frame_index": "SECOND", "label": "storage bin", "score": 0.79, "bbox_xyxy": [70, 95, 270, 325], "source_prompt": "storage bin"},
        ],
    )
    _write_fake_backend(
        grounding_script,
        detections=[
            {"frame_index": "THIRD", "label": "drawer", "score": 0.91, "bbox_xyxy": [300, 120, 520, 300], "source_prompt": "drawer"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {grounding_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    result = run_object_index_stage(capture_root=capture_root, force_rebuild=True)

    manifest = json.loads((capture_root / "raw" / "object_index.json").read_text(encoding="utf-8"))
    report = json.loads((capture_root / "raw" / "object_index_build_report.json").read_text(encoding="utf-8"))
    hints = json.loads((capture_root / "raw" / "object_grounding_hints.json").read_text(encoding="utf-8"))
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((capture_root / "raw" / "manifest.json").read_text(encoding="utf-8"))

    assert result["object_count"] == 2
    assert report["backend_summary"]["object_count"] == 2
    assert descriptor["object_index_uri"].endswith("/raw/object_index.json")
    assert raw_manifest["object_index_uri"] == "object_index.json"
    labels = {item["label"] for item in manifest["objects"]}
    assert labels == {"storage bin", "drawer"}
    drawer = next(item for item in manifest["objects"] if item["label"] == "drawer")
    assert drawer["reference_crop"]
    assert drawer["boundingBox"]["center"][0] > 0.0
    assert drawer["articulation_hints"]["interactive"] is True
    assert drawer["task_relevance"]["score"] >= 0.45
    assert hints["grounded_objects"]
    assert hints["manipulation_candidates"]
    assert hints["articulation_hints"]
    assert hints["tasks"][0]["task_id"] == "open_close_primary"


def test_object_index_stage_applies_llm_enrichment(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_raw_capture(tmp_path)
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo-llm"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "storage bin", "score": 0.7, "bbox_xyxy": [60, 90, 260, 320], "source_prompt": "bin"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", raising=False)
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    def fake_runner(skill_name: str, payload: dict) -> dict:
        if skill_name == "prompt_bank_expander":
            return {"additional_prompts": ["handle"], "resolved_task_nouns": ["bin"], "notes": "expanded"}
        if skill_name == "task_relevance_ranker":
            return {"scores": [{"object_id": "container_0001", "score": 0.97, "matched_terms": ["bin"], "reason": "inventory task"}]}
        if skill_name == "workflow_target_resolver":
            return {
                "manipulation_candidates": [{"instance_id": "container_0001", "label": "storage bin", "confidence": 0.96}],
                "articulation_hints": [],
                "tasks": [{"task_id": "organize_bin", "target_object_ids": ["container_0001"]}],
                "open_questions": [],
            }
        if skill_name == "articulation_prior_writer":
            return {"articulation_priors": []}
        return {}

    monkeypatch.setattr("blueprint_pipeline.object_index_stage.build_capture_enrichment_runner", lambda repo_root: fake_runner)

    run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    manifest = json.loads((capture_root / "raw" / "object_index.json").read_text(encoding="utf-8"))
    report = json.loads((capture_root / "raw" / "object_index_build_report.json").read_text(encoding="utf-8"))
    hints = json.loads((capture_root / "raw" / "object_grounding_hints.json").read_text(encoding="utf-8"))

    assert "handle" in report["prompt_bank"]["task_specific"]
    assert manifest["objects"][0]["task_relevance"]["score"] == 0.97
    assert hints["tasks"][0]["task_id"] == "organize_bin"
    assert report["llm_enrichment"]["workflow_target_resolver"]["tasks"][0]["task_id"] == "organize_bin"


def test_object_index_stage_supports_rgb_only_video_capture(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_raw_capture(tmp_path, include_arkit=False)
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo-rgb"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "box", "score": 0.8, "bbox_xyxy": [120, 100, 300, 320], "source_prompt": "box"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", raising=False)
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    result = run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    manifest = json.loads((capture_root / "raw" / "object_index.json").read_text(encoding="utf-8"))

    assert result["object_count"] == 1
    assert manifest["objects"][0]["label"] == "box"
    assert manifest["objects"][0]["reference_crop"]


def test_object_index_stage_keeps_vague_tasks_low_relevance(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_raw_capture(tmp_path, workflow_name="General site walkthrough")
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo-vague"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "box", "score": 0.82, "bbox_xyxy": [120, 100, 300, 320], "source_prompt": "box"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", raising=False)
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    manifest = json.loads((capture_root / "raw" / "object_index.json").read_text(encoding="utf-8"))

    assert manifest["objects"][0]["task_relevance"]["score"] < 0.6


def test_object_index_stage_degrades_confidence_when_privacy_limited(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_raw_capture(
        tmp_path,
        privacy_limits=["cabinet faces partially redacted"],
    )
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo-private"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "cabinet", "score": 0.88, "bbox_xyxy": [100, 80, 320, 360], "source_prompt": "cabinet"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", raising=False)
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    manifest = json.loads((capture_root / "raw" / "object_index.json").read_text(encoding="utf-8"))

    assert manifest["objects"][0]["mean_confidence"] < 0.88
    assert manifest["objects"][0]["provenance"]["privacy_penalty_applied"] is True


def test_qualification_auto_builds_object_index_and_task_targets(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_raw_capture(tmp_path)
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo-qualification"
    grounding_script = tmp_path / "bin" / "fake-grounding-qualification"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "storage bin", "score": 0.82, "bbox_xyxy": [60, 90, 260, 320], "source_prompt": "storage bin"},
        ],
    )
    _write_fake_backend(
        grounding_script,
        detections=[
            {"frame_index": "SECOND", "label": "drawer", "score": 0.91, "bbox_xyxy": [300, 120, 520, 300], "source_prompt": "drawer"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {grounding_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    run_capture_pipeline(
        descriptor_gcs_uri=f"gs://bucket/scenes/{capture_root.parts[-3]}/captures/{capture_root.parts[-1]}/capture_descriptor.json",
        lane="qualification",
        config=OrchestratorConfig(
            gcs_root=capture_root.parents[4],
            fail_on_commit_mismatch=False,
            expected_blueprintpipeline_commit="",
        ),
    )

    scorecard = json.loads((capture_root / "pipeline" / "capture_qa_scorecard.json").read_text(encoding="utf-8"))
    task_targets = json.loads((capture_root / "pipeline" / "task_targets.json").read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in scorecard["checks"]}

    assert checks["object_index_present"]["passed"] is True
    assert checks["object_index_populated"]["passed"] is True
    assert task_targets["target_object_ids"]
    assert task_targets["articulation_required_ids"]


def test_qualification_writes_llm_weakness_and_recapture_artifacts(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_raw_capture(tmp_path)
    _materialize_capture(capture_root)

    yolo_script = tmp_path / "bin" / "fake-yolo-qual-llm"
    _write_fake_backend(
        yolo_script,
        detections=[
            {"frame_index": "FIRST", "label": "drawer", "score": 0.9, "bbox_xyxy": [250, 100, 520, 320], "source_prompt": "drawer"},
        ],
    )
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {yolo_script} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.delenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", raising=False)
    monkeypatch.delenv("OBJECT_INDEX_SAM3_COMMAND", raising=False)

    def fake_runner(skill_name: str, payload: dict) -> dict:
        if skill_name == "prompt_bank_expander":
            return {"additional_prompts": [], "resolved_task_nouns": [], "notes": "none"}
        if skill_name == "task_relevance_ranker":
            return {"scores": []}
        if skill_name == "workflow_target_resolver":
            return {"manipulation_candidates": [], "articulation_hints": [], "tasks": [], "open_questions": []}
        if skill_name == "articulation_prior_writer":
            return {"articulation_priors": []}
        if skill_name == "qualification_weakness_summarizer":
            return {"summary": "Coverage is still weak around the task zone.", "top_gaps": ["missing route coverage"], "recommended_focus": ["capture the aisle approach"]}
        if skill_name == "recapture_instruction_writer":
            return {"operator_brief": "Do a second pass focused on the drawer and aisle.", "instructions": [{"order": 1, "detail": "Re-capture the aisle approach."}]}
        return {}

    monkeypatch.setattr("blueprint_pipeline.object_index_stage.build_capture_enrichment_runner", lambda repo_root: fake_runner)
    monkeypatch.setattr("blueprint_pipeline.qualification.build_capture_enrichment_runner", lambda repo_root: fake_runner)

    run_capture_pipeline(
        descriptor_gcs_uri=f"gs://bucket/scenes/{capture_root.parts[-3]}/captures/{capture_root.parts[-1]}/capture_descriptor.json",
        lane="qualification",
        config=OrchestratorConfig(
            gcs_root=capture_root.parents[4],
            fail_on_commit_mismatch=False,
            expected_blueprintpipeline_commit="",
        ),
    )

    weakness = json.loads((capture_root / "pipeline" / "qualification_weakness_summary.json").read_text(encoding="utf-8"))
    recapture = json.loads((capture_root / "pipeline" / "recapture_instructions.json").read_text(encoding="utf-8"))
    human_actions = json.loads((capture_root / "pipeline" / "human_actions_required.json").read_text(encoding="utf-8"))

    assert weakness["summary"] == "Coverage is still weak around the task zone."
    assert recapture["instructions"][0]["detail"] == "Re-capture the aisle approach."
    assert human_actions["llm_recapture_instructions"][0]["detail"] == "Re-capture the aisle approach."
