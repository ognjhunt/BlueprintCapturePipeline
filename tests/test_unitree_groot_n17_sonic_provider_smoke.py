from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

from blueprint_pipeline import unitree_groot_n17_sonic_provider_smoke as smoke


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\x81\xb3\x1c\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _frame(path: Path) -> Path:
    path.write_bytes(PNG_1X1)
    return path


def test_groot_n17_sonic_provider_bundle_contains_runtime_contract(tmp_path: Path) -> None:
    manifest = smoke.build_unitree_groot_n17_sonic_policy_provider_bundle(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        policy_command="python run_groot_sonic_policy.py",
        n17_checkpoint="nvidia/GR00T-N1.7-3B",
        sonic_checkpoint="/weights/g1_sonic/checkpoint-20000",
        groot_root="/workspace/Isaac-GR00T",
        wbc_root="/workspace/GR00T-WholeBodyControl",
        policy_server_url="tcp://127.0.0.1:5550",
        sim2sim_command="python gear_sonic/scripts/run_sim_loop.py",
    )

    bundle = Path(manifest["bundle_path"])
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        runner_text = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_provider_runner.py"
        ).decode("utf-8")
    assert "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" in names
    assert "provider_runtime/unitree_groot_n17_sonic_policy_provider_manifest.json" in names
    assert "provider_runtime/policy_input.json" in names
    ast.parse(runner_text)
    assert "run_unitree_groot_n17_sonic_policy" in runner_text
    assert manifest["env_contract"]["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] == (
        "<configured>"
    )
    assert manifest["ready_for_fresh_model_execution"] is True
    assert manifest["runtime_execution_blockers"] == []
    assert (
        manifest["truth_boundary"]["unitree_groot_n17_sonic_policy_action_command_ran"]
        is False
    )
    assert manifest["truth_boundary"]["physical_robot_readiness_proven"] is False


def test_import_groot_n17_sonic_provider_output_completed(tmp_path: Path) -> None:
    output_zip = tmp_path / "provider_output.zip"
    provider_payload = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_output.v1",
        "status": "completed",
        "unitree_groot_n17_sonic_model_executed": True,
        "unitree_groot_n17_sonic_policy_action_command_ran": True,
        "policy_action_model_command_ran": True,
        "action": {"action_type": "unitree_g1_sonic_action_chunk"},
    }
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(provider_payload),
        )

    imported = smoke.import_unitree_groot_n17_sonic_provider_output(
        provider_output_zip=output_zip,
        extraction_dir=tmp_path / "extracted",
        output_path=tmp_path / "import.json",
    )

    assert imported["status"] == "completed"
    assert imported["unitree_groot_n17_sonic_model_executed"] is True
    assert imported["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert imported["action"]["action_type"] == "unitree_g1_sonic_action_chunk"
    assert imported["truth_boundary"]["physical_robot_readiness_proven"] is False
    assert (
        imported["truth_boundary"]["provider_output_import_is_not_fresh_local_policy_execution"]
        is True
    )


def test_groot_n17_sonic_provider_smoke_dry_run(tmp_path: Path) -> None:
    summary = smoke.run_unitree_groot_n17_sonic_policy_provider_smoke(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        dry_run=True,
    )

    assert summary["status"] == "dry_run_ready"
    assert summary["unitree_groot_n17_sonic_model_executed"] is False
    assert summary["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert summary["ready_for_fresh_model_execution"] is False
    assert "blocked_missing_unitree_groot_n17_sonic_policy_command" in summary[
        "runtime_execution_blockers"
    ]
    assert "blocked_missing_unitree_groot_n17_checkpoint" in summary[
        "runtime_execution_blockers"
    ]
    assert "blocked_missing_unitree_g1_sonic_checkpoint" in summary[
        "runtime_execution_blockers"
    ]
    assert Path(summary["bundle_manifest_path"]).is_file()
