from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

from blueprint_pipeline import openvla_policy_provider_smoke as smoke


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\x81\xb3\x1c\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _frame(path: Path) -> Path:
    path.write_bytes(PNG_1X1)
    return path


def test_openvla_policy_provider_bundle_contains_runtime_contract(tmp_path: Path) -> None:
    manifest = smoke.build_openvla_policy_provider_bundle(
        job_dir=tmp_path,
        frame_path=_frame(tmp_path / "frame.png"),
        task_id="contact_or_push_light_object",
        task_prompt="touch the block",
    )

    bundle = Path(manifest["bundle_path"])
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        runner_text = archive.read("provider_runtime/openvla_provider_runner.py").decode("utf-8")
    assert "provider_runtime/run_wam_provider_runtime.sh" in names
    assert "provider_runtime/wam_provider_runtime_runner.py" in names
    assert "provider_runtime/wam_rollout_input_manifest.json" in names
    assert "provider_runtime/openvla_policy_provider_manifest.json" in names
    assert "provider_runtime/policy_input.json" in names
    ast.parse(runner_text)
    for dependency in smoke.OPENVLA_RUNTIME_DEPENDENCY_PINS:
        assert dependency in runner_text
    assert "transformers>=4.40.1,<4.47" not in runner_text
    assert 'or "eager"' in runner_text
    assert "model_load_attempt_errors" in runner_text
    assert "_supports_sdpa" in runner_text
    assert "_force_openvla_eager_attention_compat" in runner_text
    assert "model_sdpa_compatibility_patch" in runner_text
    assert "supports_sdpa_missing_patched" in runner_text
    assert "?\\nOut:" in runner_text
    assert "?\\n Out:" not in runner_text
    assert "openvla_empty_token_id" in runner_text
    assert "torch.full" in runner_text
    assert "torch.ones" in runner_text
    assert "attention_mask_shape_after" in runner_text
    assert "openvla_input_diagnostics" in runner_text
    assert "traceback_tail" in runner_text
    assert "openvla_model_loaded" in runner_text
    assert "openvla_runtime_dependency_version_mismatch" in runner_text
    assert "_runtime_dependency_version_issues" in runner_text
    assert "initial_dependency_version_issues" in runner_text
    assert manifest["truth_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False
    assert manifest["raw_credentials_written_to_artifacts"] is False


def test_import_openvla_provider_output_completed(tmp_path: Path) -> None:
    output_zip = tmp_path / "provider_output.zip"
    provider_payload = {
        "schema_version": "openvla_policy_provider_output.v1",
        "status": "completed",
        "openvla_model_executed": True,
        "openvla_policy_action_command_ran": True,
        "action": {"action_type": "manipulation_contact"},
    }
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr("openvla_policy_provider_output.json", json.dumps(provider_payload))

    imported = smoke.import_openvla_provider_output(
        provider_output_zip=output_zip,
        extraction_dir=tmp_path / "extracted",
        output_path=tmp_path / "import.json",
    )

    assert imported["status"] == "completed"
    assert imported["openvla_model_executed"] is True
    assert imported["openvla_policy_action_command_ran"] is True
    assert imported["action"]["action_type"] == "manipulation_contact"
    assert imported["truth_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_openvla_policy_provider_smoke_dry_run_without_paid_launch(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_stage(*, job_dir: str | Path, bundle_path: str | Path, **_kwargs):
        job = Path(job_dir)
        job.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "wam_provider_object_store_staging.v1",
            "status": "completed",
            "provider_bundle_url_file": {"path": str(job / "provider_bundle_url.txt")},
            "provider_output_put_url_file": {"path": str(job / "provider_output_put_url.txt")},
            "provider_output_get_url_file": {"path": str(job / "provider_output_get_url.txt")},
            "raw_secret_values_recorded": False,
            "blockers": [],
        }
        (job / "wam_provider_object_store_staging_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(smoke, "stage_wam_provider_bundle_object_store", fake_stage)
    summary = smoke.run_openvla_policy_provider_smoke(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        dry_run=True,
        allow_paid_vast_launch=False,
    )

    assert summary["status"] == "dry_run_ready"
    assert summary["openvla_model_executed"] is False
    assert summary["blockers"] == []
    assert Path(summary["bundle_manifest_path"]).is_file()


def test_openvla_policy_provider_smoke_passes_machine_avoidlist(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_stage(*, job_dir: str | Path, bundle_path: str | Path, **_kwargs):
        job = Path(job_dir)
        job.mkdir(parents=True, exist_ok=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (job / name).write_text(f"https://example.test/{name}", encoding="utf-8")
        return {
            "schema_version": "wam_provider_object_store_staging.v1",
            "status": "completed",
            "provider_bundle_url_file": {"path": str(job / "provider_bundle_url.txt")},
            "provider_output_put_url_file": {"path": str(job / "provider_output_put_url.txt")},
            "provider_output_get_url_file": {"path": str(job / "provider_output_get_url.txt")},
            "raw_secret_values_recorded": False,
            "blockers": [],
        }

    captured: dict[str, object] = {}

    def fake_vast_provider(**kwargs):
        captured.update(kwargs)
        captured["forward_secret_env"] = smoke.os.environ.get(
            smoke.VAST_FORWARD_SECRET_ENV_VARS_ENV
        )
        captured["hf_token_present"] = bool(smoke.os.environ.get("HF_TOKEN"))
        return {"status": "blocked", "blockers": ["fake_provider_stop"]}

    avoidlist = tmp_path / "avoid.json"
    avoidlist.write_text(
        '{"schema_version":"vast_machine_avoidlist.v1","machine_ids":[93686]}',
        encoding="utf-8",
    )
    hf_file = tmp_path / "hf_token"
    hf_file.write_text("hf-test-token\n", encoding="utf-8")
    hf_file.chmod(0o600)
    monkeypatch.setenv("HF_TOKEN_FILE", str(hf_file))
    monkeypatch.setattr(smoke, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(smoke, "run_vast_provider_adapter", fake_vast_provider)

    summary = smoke.run_openvla_policy_provider_smoke(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        allow_paid_vast_launch=True,
        machine_avoidlist_path=avoidlist,
    )

    assert summary["status"] == "blocked"
    assert captured["machine_avoidlist_path"] == avoidlist
    assert captured["hf_token_present"] is True
    assert "HF_TOKEN" in str(captured["forward_secret_env"])
    assert "HUGGINGFACE_HUB_TOKEN" in str(captured["forward_secret_env"])
    assert "hf-test-token" not in json.dumps(summary)
    assert summary["model_access_secret_status"]["huggingface"]["auth_ready"] is True


def test_openvla_policy_provider_smoke_imports_policy_output_when_wam_video_smoke_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_stage(*, job_dir: str | Path, bundle_path: str | Path, **_kwargs):
        job = Path(job_dir)
        job.mkdir(parents=True, exist_ok=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (job / name).write_text(f"https://example.test/{name}", encoding="utf-8")
        return {
            "schema_version": "wam_provider_object_store_staging.v1",
            "status": "completed",
            "provider_bundle_url_file": {"path": str(job / "provider_bundle_url.txt")},
            "provider_output_put_url_file": {"path": str(job / "provider_output_put_url.txt")},
            "provider_output_get_url_file": {"path": str(job / "provider_output_get_url.txt")},
            "raw_secret_values_recorded": False,
            "blockers": [],
        }

    def fake_vast_provider(**kwargs):
        provider_job_dir = Path(kwargs["job_dir"])
        provider_job_dir.mkdir(parents=True, exist_ok=True)
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        provider_payload = {
            "schema_version": "openvla_policy_provider_output.v1",
            "status": "completed",
            "openvla_model_executed": True,
            "openvla_policy_action_command_ran": True,
            "action": {"action_type": "manipulation_contact"},
        }
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                "openvla_policy_provider_output.json",
                json.dumps(provider_payload),
            )
        (provider_job_dir / "vast_provider_command_result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "provider_runtime_output_zip_received": True,
                    "provider_output_upload_ok": True,
                    "provider_command_path_remote_proven": True,
                    "blockers": [],
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "blockers": ["mp4_count_below_expected_video_smoke_camera_count"],
        }

    monkeypatch.setattr(smoke, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(smoke, "run_vast_provider_adapter", fake_vast_provider)

    summary = smoke.run_openvla_policy_provider_smoke(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        allow_paid_vast_launch=True,
    )

    assert summary["status"] == "completed"
    assert summary["openvla_model_executed"] is True
    assert summary["openvla_policy_action_command_ran"] is True
    assert summary["blockers"] == []
