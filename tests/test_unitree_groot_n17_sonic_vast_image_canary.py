from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline import unitree_groot_n17_sonic_vast_image_canary as canary


def test_canary_bundle_contains_runtime_contract(tmp_path: Path) -> None:
    manifest = canary.build_unitree_groot_n17_sonic_vast_image_canary_bundle(
        job_dir=tmp_path / "bundle",
        generated_at="now",
    )

    bundle_path = Path(str(manifest["bundle_path"]))
    assert manifest["status"] == "canary_bundle_ready"
    assert manifest["canary_only"] is True
    assert manifest["ready_for_fresh_model_execution"] is False
    assert manifest["local_bundle_ready_for_remote_staging"] is True
    assert manifest["truth_boundary"]["canary_bundle_is_not_model_execution"] is True
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        assert "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" in names
        assert "provider_runtime/unitree_groot_n17_sonic_provider_runner.py" in names
        assert "provider_runtime/input_frame.png" in names
        runner = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_provider_runner.py"
        ).decode()
    assert canary.CANARY_MARKER in runner
    assert "nvidia-smi" in runner


def test_run_canary_requires_vast_markers_and_min_gpu(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_EXCLUDED_MACHINE_ID", "140330")
    monkeypatch.setenv("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_PUBLIC_IMAGE", "image:test")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_vast(**kwargs):
        captured.update(kwargs)
        run_dir = Path(kwargs["job_dir"])
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                canary.PROVIDER_OUTPUT_FILENAME,
                json.dumps(
                    {
                        "schema_version": canary.OUTPUT_SCHEMA_VERSION,
                        "status": "completed",
                        "canary_only": True,
                        "canary_marker": canary.CANARY_MARKER,
                        "checks": {"nvidia_smi": {"returncode": 0}},
                        "unitree_groot_n17_sonic_model_executed": False,
                        "unitree_groot_n17_sonic_policy_action_command_ran": False,
                        "policy_action_model_command_ran": False,
                        "blockers": [],
                    }
                ),
            )
        for name, payload in {
            "vast_startup_probe_manifest.json": {
                "status": "completed",
                "heartbeat_completed": True,
                "blockers": [],
            },
            "vast_gpu_sanity_report.json": {"status": "completed", "blockers": []},
            "vast_provider_command_result.json": {
                "status": "completed",
                "provider_command_path_remote_proven": True,
                "blueprint_provider_bundle_execution_proven": True,
                "provider_output_upload_ok": True,
                "provider_runtime_output_zip_produced": True,
                "blockers": [],
            },
            "vast_teardown_manifest.json": {
                "status": "completed",
                "continuing_spend_from_this_run": False,
                "runner_gpu_teardown_completed": True,
            },
            "vast_offer_selection_manifest.json": {
                "status": "selected",
                "selected_offer": {
                    "machine_id": 123,
                    "gpu_ram_mb": 49140,
                    "hourly_rate_usd": 0.2,
                },
                "min_gpu_ram_mb": 48000,
            },
            "vast_budget_ledger.json": {
                "status": "completed",
                "selected_hourly_rate_usd": 0.2,
                "actual_live_runtime_seconds_observed_by_adapter": 30.0,
                "estimated_cost_usd": 0.002,
                "estimated_spend_under_hard_cap": True,
                "continuing_spend_from_this_run": False,
            },
        }.items():
            (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        return {
            "status": "completed",
            "blockers": [],
            "vast_instance_ids": [456],
            "estimated_cost_usd": 0.01,
        }

    monkeypatch.setattr(canary, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(canary, "run_vast_provider_adapter", fake_vast)

    result = canary.run_unitree_groot_n17_sonic_vast_image_canary(
        job_dir=tmp_path / "canary",
        generated_at="now",
    )

    assert result["status"] == "completed"
    assert result["public_image"] == "image:test"
    assert result["min_gpu_ram_mb"] == 48000
    assert result["heartbeat_completed"] is True
    assert result["gpu_sanity_completed"] is True
    assert result["provider_output_upload_ok"] is True
    assert result["canary_marker_observed"] is True
    assert result["selected_hourly_rate_usd"] == 0.2
    assert result["actual_live_runtime_seconds"] == 30.0
    assert result["claim_boundary"]["canary_is_not_policy_inference"] is True
    assert captured["min_gpu_ram_mb"] == 48000
    assert captured["provider_bundle_kind"] == "unitree_groot_n17_sonic"
    assert captured["public_image"] == "image:test"
    avoidlist = json.loads(Path(str(captured["machine_avoidlist_path"])).read_text())
    assert 140330 in avoidlist["machine_ids"]
