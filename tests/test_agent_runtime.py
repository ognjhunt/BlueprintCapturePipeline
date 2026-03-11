from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.agent_runtime.orchestrator import run_agent_review
from blueprint_pipeline.agent_runtime.openai_phase2 import OpenAIPhase2Config
from blueprint_pipeline.capture_orchestrator import run_capture_pipeline
from blueprint_pipeline.materialization import materialize_capture_bundle
from blueprint_pipeline.run_e2e import run_end_to_end
from blueprint_pipeline.swap_orchestrator import OrchestratorConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _raw_capture_root(tmp_path: Path, scene_id: str = "scene_agent", capture_id: str = "cap_agent") -> Path:
    return tmp_path / "bucket/scenes" / scene_id / "captures" / capture_id


def _build_raw_capture(
    tmp_path: Path,
    *,
    scene_id: str = "scene_agent",
    capture_id: str = "cap_agent",
    source: str = "iphone",
    modality: str = "iphone_arkit_lidar",
    has_lidar: bool = True,
) -> Path:
    capture_root = _raw_capture_root(tmp_path, scene_id=scene_id, capture_id=capture_id)
    raw_root = capture_root / "raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": scene_id,
            "capture_source": source,
            "capture_tier_hint": "tier1_iphone" if source == "iphone" else "tier2_glasses",
            "has_lidar": has_lidar,
            "pose_match_rate": 0.96 if has_lidar else 0.35,
            "video_uri": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/raw/walkthrough.mov",
            "object_point_cloud_index": "arkit/objects/index.json",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Material handoff",
            "taskSteps": ["entry", "handoff"],
            "targetKPI": "throughput",
            "zone": "dock_lane_a",
            "owner": "ops_manager",
        },
    )
    _write_json(raw_root / "capture_context.json", {"captureModality": modality})
    _write_json(raw_root / "capture_upload_complete.json", {"scene_id": scene_id, "capture_id": capture_id})
    (raw_root / "walkthrough.mov").write_bytes(b"mov")
    if has_lidar:
        (raw_root / "arkit/objects").mkdir(parents=True, exist_ok=True)
        _write_json(
            raw_root / "arkit/objects/index.json",
            {
                "objects": [
                    {
                        "id": "lane_1",
                        "label": "aisle",
                        "pointCloudFile": "lane_1.ply",
                        "boundingBox": {
                            "center": [0.0, 0.0, 0.0],
                            "extents": [1.4, 2.0, 4.0],
                            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "orientationQuaternion": [1, 0, 0, 0],
                        },
                    }
                ]
            },
        )
        (raw_root / "arkit/poses.jsonl").write_text("{}\n", encoding="utf-8")
        (raw_root / "arkit/intrinsics.json").write_text("{}", encoding="utf-8")
    return capture_root


def _run_qualification_for_raw_capture(capture_root: Path) -> None:
    scene_id = capture_root.parts[-3]
    capture_id = capture_root.parts[-1]
    materialize_capture_bundle(
        bucket="bucket",
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=capture_root.parents[4],
    )
    run_capture_pipeline(
        descriptor_gcs_uri=f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/capture_descriptor.json",
        lane="qualification",
        config=OrchestratorConfig(
            gcs_root=capture_root.parents[4],
            fail_on_commit_mismatch=False,
            expected_blueprintpipeline_commit="",
        ),
    )


def test_agent_review_keeps_standards_deterministic_even_when_provider_offers_override(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path)
    _run_qualification_for_raw_capture(capture_root)

    payload = run_agent_review(
        capture_root=capture_root,
        provider_name="claude",
        skill_runner=lambda skill_name, payload: {"entries": [{"title": "Override"}]} if skill_name == "standards_retriever" else None,
    )

    bundle = json.loads(Path(payload["final_bundle_path"]).read_text(encoding="utf-8"))
    standards = json.loads((capture_root / "pipeline/standards_notes.json").read_text(encoding="utf-8"))
    assert bundle["provider"] == "claude"
    assert any(step["skill_name"] == "standards_retriever" and step["source"] == "local_deterministic" for step in bundle["steps"])
    assert standards["entries"] != [{"title": "Override"}]


def test_agent_review_allows_provider_override_for_openai(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_openai", capture_id="cap_openai")
    _run_qualification_for_raw_capture(capture_root)

    payload = run_agent_review(
        capture_root=capture_root,
        provider_name="openai",
        skill_runner=lambda skill_name, payload: {"summary": "mock"} if skill_name == "oem_handoff_writer" else None,
    )

    bundle = json.loads(Path(payload["final_bundle_path"]).read_text(encoding="utf-8"))
    oem = json.loads((capture_root / "pipeline/oem_handoff_summary.json").read_text(encoding="utf-8"))
    assert bundle["provider"] == "openai"
    assert any(step["skill_name"] == "oem_handoff_writer" and step["source"] == "provider_override" for step in bundle["steps"])
    assert oem["summary"] == "mock"


def test_agent_review_allows_provider_override_for_recapture_plan(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_recap", capture_id="cap_recap", source="glasses", modality="glasses_video_only", has_lidar=False)
    _run_qualification_for_raw_capture(capture_root)

    payload = run_agent_review(
        capture_root=capture_root,
        provider_name="openai",
        skill_runner=lambda skill_name, payload: {"required": True, "steps": [{"order": 1, "detail": "custom"}]} if skill_name == "recapture_planner" else None,
    )

    bundle = json.loads(Path(payload["final_bundle_path"]).read_text(encoding="utf-8"))
    recapture = json.loads((capture_root / "pipeline/recapture_plan.json").read_text(encoding="utf-8"))
    assert any(step["skill_name"] == "recapture_planner" and step["source"] == "provider_override" for step in bundle["steps"])
    assert recapture["steps"] == [{"order": 1, "detail": "custom"}]


def test_agent_review_uses_codex_phase2_runner_when_enabled(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_codex", capture_id="cap_codex")
    _run_qualification_for_raw_capture(capture_root)
    fake_codex = tmp_path / "bin/fake-codex"
    _write_executable(
        fake_codex,
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
prompt = sys.stdin.read()
if "Skill: oem_handoff_writer" in prompt:
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_codex",
        "capture_id": "cap_codex",
        "recommended_lane": "qualification",
        "target_robot_team": {},
        "summary": "Codex override summary"
    }
else:
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_codex",
        "capture_id": "cap_codex",
        "entries": []
    }
output_path.write_text(json.dumps(payload), encoding="utf-8")
""",
    )

    payload = run_agent_review(
        capture_root=capture_root,
        provider_name="openai",
        openai_phase2_config=OpenAIPhase2Config(
            mode="codex_cli",
            model="gpt-5.1",
            codex_bin=str(fake_codex),
            timeout_seconds=30,
        ),
    )

    bundle = json.loads(Path(payload["final_bundle_path"]).read_text(encoding="utf-8"))
    oem = json.loads((capture_root / "pipeline/oem_handoff_summary.json").read_text(encoding="utf-8"))
    assert any(step["skill_name"] == "oem_handoff_writer" and step["source"] == "provider_override" for step in bundle["steps"])
    assert bundle["runtime"]["openai_phase2_transport"] == "codex_exec"
    assert oem["summary"] == "Codex override summary"


def test_agent_review_falls_back_when_codex_phase2_runner_fails(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_fallback", capture_id="cap_fallback")
    _run_qualification_for_raw_capture(capture_root)
    fake_codex = tmp_path / "bin/failing-codex"
    _write_executable(
        fake_codex,
        """#!/usr/bin/env bash
exit 2
""",
    )

    payload = run_agent_review(
        capture_root=capture_root,
        provider_name="openai",
        openai_phase2_config=OpenAIPhase2Config(
            mode="codex_cli",
            model="gpt-5.1",
            codex_bin=str(fake_codex),
            timeout_seconds=30,
        ),
    )

    bundle = json.loads(Path(payload["final_bundle_path"]).read_text(encoding="utf-8"))
    assert any(step["skill_name"] == "oem_handoff_writer" and step["source"] == "local_deterministic" for step in bundle["steps"])


def test_run_end_to_end_writes_final_bundle_and_memo(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_e2e", capture_id="cap_e2e")

    result = run_end_to_end(capture_root=str(capture_root), provider="openai")

    assert Path(result["final_memo_path"]).is_file()
    assert Path(result["final_bundle_path"]).is_file()
    bundle = json.loads(Path(result["final_bundle_path"]).read_text(encoding="utf-8"))
    assert bundle["provider"] == "openai"
    assert Path(bundle["artifacts"]["human_actions_required"]).is_file()


def test_run_end_to_end_optionally_builds_simready_workcell(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_simready", capture_id="cap_simready")

    result = run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        run_simready=True,
    )

    assert result["simready"] is not None
    assert Path(result["simready"]["scene_path"]).is_file()
    assert Path(result["simready"]["manifest_path"]).is_file()
    manifest = json.loads(Path(result["simready"]["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["runtime"] == "isaac_sim"


def test_run_end_to_end_optionally_builds_evaluation_prep(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(tmp_path, scene_id="scene_evalprep", capture_id="cap_evalprep")

    result = run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        run_evaluation_prep=True,
    )

    assert result["evaluation_prep"] is not None
    manifest_path = Path(result["evaluation_prep"]["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["qualified_opportunity_handoff"] == "qualified_opportunity_handoff.json"


def test_video_only_capture_generates_recapture_plan_and_human_actions(tmp_path: Path) -> None:
    capture_root = _build_raw_capture(
        tmp_path,
        scene_id="scene_video",
        capture_id="cap_video",
        source="glasses",
        modality="glasses_video_only",
        has_lidar=False,
    )

    result = run_end_to_end(capture_root=str(capture_root), provider="claude")

    bundle = json.loads(Path(result["final_bundle_path"]).read_text(encoding="utf-8"))
    recapture = json.loads((capture_root / "pipeline/recapture_plan.json").read_text(encoding="utf-8"))
    human_actions = json.loads((capture_root / "pipeline/human_actions_required.json").read_text(encoding="utf-8"))
    assert bundle["readiness_state"] != "ready"
    assert recapture["required"] is True
    assert human_actions["actions"]
