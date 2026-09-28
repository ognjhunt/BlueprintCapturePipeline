from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import adp_content_agents_vast as lane


@pytest.mark.parametrize(("status", "exit_code"), [("ready", 0), ("blocked", 2)])
def test_content_agents_module_cli_preserves_builder_arguments_and_exit_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    exit_code: int,
) -> None:
    observed: list[dict[str, object]] = []

    def build(**kwargs):
        observed.append(kwargs)
        return {"status": status, "receipt_digest": "sha256:" + "a" * 64}

    monkeypatch.setattr(lane, "build_content_agents_vast_bundle", build)
    repo = tmp_path / "repo"
    content_agents = tmp_path / "content-agents"
    job = tmp_path / "job"
    reference_a = tmp_path / "reference-a.png"
    reference_b = tmp_path / "reference-b.png"

    result = lane.main(
        [
            "--repo-root",
            str(repo),
            "--content-agents-root",
            str(content_agents),
            "--reference-image",
            str(reference_a),
            "--reference-image",
            str(reference_b),
            "--job-dir",
            str(job),
            "--input-variant",
            "paired_target_registered_v1",
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--agent-cad-output-manifest",
            str(tmp_path / "cad.json"),
            "--agent-mesh-projection-receipt",
            str(tmp_path / "projection.json"),
            "--paired-target-construction-bindings",
            str(tmp_path / "bindings.json"),
            "--paired-target-task-id",
            "task-fixture",
            "--reference-rights-authority",
            str(tmp_path / "rights.json"),
            "--content-agents-execution-route",
            str(tmp_path / "route.json"),
            "--historical-replay-only",
        ]
    )

    assert result == exit_code
    assert observed == [
        {
            "repo_root": str(repo),
            "content_agents_root": str(content_agents),
            "reference_image_paths": [str(reference_a), str(reference_b)],
            "job_dir": str(job),
            "input_variant": "paired_target_registered_v1",
            "evidence_root": str(tmp_path / "evidence"),
            "agent_cad_output_manifest_path": str(tmp_path / "cad.json"),
            "agent_mesh_projection_receipt_path": str(tmp_path / "projection.json"),
            "paired_target_construction_bindings_path": str(
                tmp_path / "bindings.json"
            ),
            "paired_target_task_id": "task-fixture",
            "reference_rights_authority_path": str(tmp_path / "rights.json"),
            "content_agents_execution_route_path": str(tmp_path / "route.json"),
            "historical_replay_only": True,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "status": status,
        "receipt_digest": "sha256:" + "a" * 64,
    }
