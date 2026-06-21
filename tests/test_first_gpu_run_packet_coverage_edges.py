from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import first_gpu_run_packet as fg


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_first_gpu_run_packet_small_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result_path = tmp_path / "gpu_vm_runtime_preflight_result.json"
    result_path.write_text("[]", encoding="utf-8")
    invalid_payload = fg._gpu_vm_runtime_preflight_result_summary(result_path)
    assert invalid_payload["blockers"] == [
        "gpu_vm_runtime_preflight_result_invalid_payload:list"
    ]
    _write_json(result_path, {"status": "blocked", "blockers": ["missing_driver"]})
    blocked_payload = fg._gpu_vm_runtime_preflight_result_summary(result_path)
    assert "gpu_vm_runtime_preflight_result_blocker:missing_driver" in blocked_payload[
        "blockers"
    ]
    assert "gpu_vm_runtime_preflight_result_status:blocked" in blocked_payload["blockers"]

    capture_root = tmp_path / "capture"
    assert fg._default_output_dir(capture_root) == (
        capture_root / "pipeline" / "first_gpu_e2e_run_packet"
    )
    env_preflight = tmp_path / "forwarding.json"
    monkeypatch.setenv(fg.FORWARD_PREFLIGHT_REPORT_ENV, str(env_preflight))
    assert fg._selected_webapp_forwarding_preflight_path(capture_root, None) == (
        env_preflight.resolve()
    )
    monkeypatch.delenv(fg.FORWARD_PREFLIGHT_REPORT_ENV)
    default_preflight = capture_root / "pipeline" / "webapp_forwarding_preflight.json"
    _write_json(default_preflight, {"status": "ready"})
    assert fg._selected_webapp_forwarding_preflight_path(capture_root, None) == (
        default_preflight.resolve()
    )

    assert fg._default_owner_command("isaac_lab_arena").endswith(
        "run_isaac_lab_arena_gpu_proof.sh"
    )
    assert fg._default_owner_command("other").endswith("run_owner_gpu_proof.sh")
    assert fg._sha_file(tmp_path / "missing.bin") is None
    entries: list[dict[str, object]] = []
    missing = tmp_path / "missing.txt"
    fg._append_file_entry(entries, missing, role="missing")
    fg._append_file_entry(entries, missing, role="duplicate")
    assert len(entries) == 1
    assert entries[0]["blockers"] == ["missing_required_sync_file:missing"]

    raw_dir = capture_root / "raw"
    _write_json(raw_dir / "manifest.json", {"video_uri": "walkthrough-relative.mov"})
    (raw_dir / "walkthrough-relative.mov").write_bytes(b"video")
    assert fg._raw_video_path(capture_root) == raw_dir / "walkthrough-relative.mov"
    (raw_dir / "walkthrough-relative.mov").unlink()
    (capture_root / "walkthrough-relative.mov").write_bytes(b"video")
    assert fg._raw_video_path(capture_root) == capture_root / "walkthrough-relative.mov"
    assert fg._capture_root_by_site_json(site_slug="", capture_root=capture_root) == "{}"

    env_example = fg._env_example(
        capture_root=capture_root,
        packet_dir=tmp_path / "packet",
        webapp_site_slug="site",
        webapp_staged_inputs_path=tmp_path / "staged.json",
        webapp_forwarding_preflight_path=None,
        simulator="other",
        provisioner="runpod",
        owner_command="echo owner",
    )
    assert 'OWNER_SIMULATOR_COMMAND="bash $OWNER_DEFAULT_SMOKE_COMMAND_BINDING"' in env_example

    monkeypatch.setattr(fg, "_repo_root", lambda: tmp_path / "missing-repo")
    assert "regenerate the packet" in fg._mujoco_unitree_g1_smoke_script()


def test_first_gpu_run_packet_manifest_and_markdown_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = tmp_path / "capture"
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    bootstrap = fg._gpu_provider_bootstrap_manifest(
        capture_root=capture_root,
        simulator="newton",
        provisioner="vast",
        owner_command="echo owner",
        owner_command_location="remote",
    )
    assert bootstrap["first_smoke_path"]["requires_paid_gpu_for_owner_runtime"] is True
    assert bootstrap["gpu_guidance"]["minimum_vram_gb"] == 16

    matrix_markdown = fg._simulator_path_matrix_markdown(
        {
            "status": "ready",
            "selected_simulator": "mujoco",
            "selected_provisioner": "local",
            "paths": ["bad", {"framework": "mujoco"}],
            "nvidia_nim_boundary": {},
        }
    )
    assert "| `mujoco` |" in matrix_markdown

    plan = fg._gpu_vm_runtime_preflight_plan_manifest(
        capture_root=capture_root,
        packet_dir=packet_dir,
        script_path=packet_dir / "preflight.sh",
        result_path=packet_dir / "result.json",
        sync_manifest_path=packet_dir / "sync.json",
        readiness={"blockers": []},
        simulator="isaac_sim",
        provisioner="runpod",
        owner_command="",
        owner_command_location="remote",
        owner_command_supplied=True,
        vm_sync_manifest={"status": "blocked"},
    )
    assert "gpu_vm_sync_manifest_status:blocked" in plan["hard_stop_blockers"]

    assert fg._blocker_details_matching(
        ["bad", {"blocker_id": "one"}, {"severity": "hard"}],
        ["one"],
        severities=["hard"],
    ) == [{"blocker_id": "one"}, {"severity": "hard"}]

    _write_json(
        capture_root / "pipeline" / "source_video_preflight_manifest.json",
        {
            "status": "blocked",
            "blockers": ["source_blocked"],
            "candidates": [
                "bad",
                {
                    "staging_blockers": ["stage_blocked"],
                    "worldlabs_blockers": ["world_blocked"],
                    "warnings": ["candidate_warning"],
                },
            ],
        },
    )
    source_category = fg._source_video_category(capture_root)
    assert {"source_blocked", "stage_blocked", "world_blocked"}.issubset(
        set(source_category["blockers"])
    )
    assert source_category["warnings"] == ["candidate_warning"]

    blocker_md = fg._blocker_resolution_markdown(
        {
            "status": "blocked",
            "resolution_actions": [],
            "categories": ["bad", {"title": "Source", "status": "blocked"}],
        }
    )
    assert "### Source" in blocker_md

    launch_md = fg._launch_order_markdown(
        {
            "status": "blocked",
            "gpu_execution_allowed": False,
            "selected_simulator": "mujoco",
            "next_action_step_ids": [],
            "steps": ["bad", {"title": "Sync", "step_id": "sync", "phase_group": "prep"}],
        }
    )
    assert "### Sync" in launch_md

    raw_dir = capture_root / "raw"
    _write_json(raw_dir / "manifest.json", {"video_uri": "gs://remote/video.mov"})
    _write_json(raw_dir / "capture_context.json", {})
    _write_json(raw_dir / "capture_upload_complete.json", {})
    sync_manifest = fg._gpu_vm_sync_manifest(
        capture_root=capture_root,
        packet_dir=packet_dir,
        generated_files={},
        readiness={"blockers": []},
    )
    assert any(item["role"] == "raw_walkthrough_video" for item in sync_manifest["files"])

    pipeline = capture_root / "pipeline"
    collider = pipeline / "collider.glb"
    materialized_asset = pipeline / "materialized.glb"
    collider.parent.mkdir(parents=True, exist_ok=True)
    collider.write_bytes(b"glb")
    materialized_asset.write_bytes(b"asset")
    _write_json(
        pipeline / "worldlabs_export_manifest.json",
        {"output_collider_mesh_path": "collider.glb", "collider_mesh_glb_url": ""},
    )
    _write_json(
        pipeline / "worldlabs_assets" / "materialized_assets_manifest.json",
        {
            "downloads": [
                "bad",
                {"local_path": ""},
                {"kind": "", "local_path": str(materialized_asset)},
            ]
        },
    )
    candidates = fg._scene_asset_candidates(capture_root)
    assert candidates[0]["exists"] is True
    assert candidates[-1]["role"] == "worldlabs_materialized_asset"

    _write_json(
        pipeline / "source_video_preflight_manifest.json",
        {"status": "ready", "ready_for_worldlabs_first_clip_count": 1},
    )
    monkeypatch.setenv(fg.WORLDLABS_API_KEY_ENV, "key")
    monkeypatch.setenv(fg.WORLDLABS_PROVIDER_SUBMISSION_GATE_ENV, "true")
    scene_manifest = fg._scene_asset_acquisition_manifest(
        capture_root=capture_root,
        webapp_site_slug="site",
        provider_submission_script_path=packet_dir / "submit.sh",
    )
    assert scene_manifest["provider_submission"]["status"] == "ready_to_submit_worldlabs_request"
    _write_json(pipeline / "worldlabs_request_manifest.json", {"status": "submitted"})
    assert fg._scene_asset_acquisition_manifest(
        capture_root=capture_root,
        webapp_site_slug="site",
        provider_submission_script_path=packet_dir / "submit.sh",
    )["provider_submission"]["status"] == "waiting_for_worldlabs_world_manifest"
    _write_json(pipeline / "worldlabs_world_manifest.json", {"status": "ready"})
    (pipeline / "worldlabs_assets" / "materialized_assets_manifest.json").unlink()
    assert fg._scene_asset_acquisition_manifest(
        capture_root=capture_root,
        webapp_site_slug="site",
        provider_submission_script_path=packet_dir / "submit.sh",
    )["provider_submission"]["status"] == "ready_to_materialize_worldlabs_assets"


def test_first_gpu_run_packet_webapp_and_build_edges(tmp_path: Path) -> None:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene" / "captures" / "capture"
    _write_json(capture_root / "capture_descriptor.json", {"scene_id": "scene", "capture_id": "capture"})
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene", "capture_id": "capture"})
    readiness = {"stages": {"webapp_forwarding": {"blockers": ["not_ready"]}}}
    handoff = fg._webapp_handoff_manifest(
        capture_root=capture_root,
        webapp_site_slug="site",
        webapp_staged_inputs_path=capture_root / "pipeline" / "staged.json",
        verification_script_path=tmp_path / "verify.sh",
        verification_result_path=tmp_path / "verify.json",
        readiness=readiness,
        allow_local_webapp_rehearsal=False,
        simulator="mujoco",
        provisioner="local",
        owner_command_location="bad-location",
    )
    assert "webapp_upstream_truth:stage_missing" in handoff["blockers"]
    assert "webapp_staged_request:stage_missing" in handoff["blockers"]

    md = fg._webapp_handoff_markdown(
        {
            "status": "blocked",
            "capture_root": str(capture_root),
            "webapp_site_slug": "site",
            "blockers": ["blocked"],
            "warnings": ["warning"],
            "upstream_id_requirements": [{"field": "capture_id"}],
            "ordered_next_steps": ["verify"],
        }
    )
    assert "## Warnings" in md
    assert "`capture_id`" in md

    result = fg.build_first_gpu_run_packet(
        capture_root=capture_root,
        simulator="mujoco",
        provisioner="local",
        owner_command_location="unsupported",
        output_dir=tmp_path / "packet",
        require_webapp_forwarding=False,
        require_webapp_staged_request=False,
        require_gpu_gates=False,
    )
    packet = json.loads(Path(result["packet_path"]).read_text(encoding="utf-8"))
    assert packet["owner_command_location"] == "remote"
