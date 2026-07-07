from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.arena_fixture_smoke import build_arena_fixture_smoke, main

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_arena_fixture_smoke_surfaces_buyer_readout_blockers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "preexisting")

    result = build_arena_fixture_smoke(
        output_dir=tmp_path / "arena-smoke",
        scenario_count=20,
        shard_size=5,
    )

    package_dir = Path(str(result["package_dir"]))
    schedule = _read_json(package_dir / "arena_eval_schedule.json")
    trace = _read_json(package_dir / "normalized_attempt_trace.json")
    clips = _read_json(package_dir / "clips_manifest.json")
    vision = _read_json(package_dir / "rollout_vision_labels.json")
    signed_access = _read_json(package_dir / "signed_access_manifest.json")
    policy = _read_json(package_dir / "policy_adapter_manifest.json")
    operators = _read_json(package_dir / "live_operator_ledger.json")
    audit = _read_json(package_dir / "arena_package_proof_boundary_audit.json")

    assert result["status"] == "blocked"
    assert result["ingest_exit_code"] == 0
    assert "post_training_data_package_not_export_ready_review_required" in result["blockers"]
    assert schedule["scenario_count"] == 20
    assert schedule["shard_count"] == 4
    assert trace["attempt_count"] == 3
    assert clips["clip_count"] == 3
    assert any(clip["status"] == "blocked_missing_video" for clip in clips["clips"])
    assert vision["status"] == "completed_review_required"
    assert vision["labels"][0]["status"] == "review_required"
    assert signed_access["status"] == "local_delivery_ready_review_required"
    assert policy["status"] == "ready_for_owner_launch_review"
    assert operators["status"] == "completed"
    assert operators["agents_sdk_operator_performed"] is True
    assert operators["codex_sdk_operator_performed"] is True
    assert audit["status"] == "blocked"
    assert "post_training_data_package_not_export_ready_review_required" in audit["blockers"]
    assert result["proof_boundary"]["simulator_execution_proven"] is False
    assert result["proof_boundary"]["owner_system_arena_execution_proven"] is False
    assert result["proof_boundary"]["webapp_upstream_truth_proven"] is False
    assert (package_dir / "archives" / "post_training_data_package.tar.gz").is_file()
    assert (Path(str(result["output_dir"])) / "arena_fixture_smoke_manifest.json").is_file()
    assert os.environ["BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"] == "preexisting"


def test_arena_fixture_smoke_module_cli(tmp_path: Path) -> None:
    output_dir = tmp_path / "arena-smoke-cli"
    env = os.environ.copy()
    src_root = Path.cwd() / "src"
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src_root)
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.arena_fixture_smoke",
            "--output-dir",
            str(output_dir),
            "--scenario-count",
            "10",
            "--shard-size",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 1, completed.stderr
    manifest = _read_json(output_dir / "arena_fixture_smoke_manifest.json")
    assert manifest["status"] == "blocked"
    assert "post_training_data_package_not_export_ready_review_required" in manifest["blockers"]
    assert manifest["scenario_count"] == 10
    assert manifest["shard_count"] == 2


def test_arena_fixture_smoke_main_reports_status(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "arena-smoke-main"

    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--scenario-count",
            "8",
            "--shard-size",
            "4",
            "--retry-budget",
            "1",
        ]
    )

    captured = capsys.readouterr()
    manifest = _read_json(output_dir / "arena_fixture_smoke_manifest.json")
    assert exit_code == 1
    assert manifest["status"] == "blocked"
    assert "post_training_data_package_not_export_ready_review_required" in manifest["blockers"]
    assert "[arena-fixture-smoke] manifest=" in captured.out
    assert "[arena-fixture-smoke] status=blocked" in captured.out
