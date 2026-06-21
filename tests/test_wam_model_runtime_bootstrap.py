from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from blueprint_pipeline import wam_model_runtime_bootstrap as bootstrap


def test_wam_model_runtime_bootstrap_writes_blocked_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", str(tmp_path / "missing-source"))
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(tmp_path / "missing-checkpoint"))

    summary = bootstrap.build_bootstrap_package(
        candidate_id="oscar_wam",
        output_dir=tmp_path / "bootstrap",
        job_root=tmp_path / "isolated-jobs",
        generated_at="now",
    )

    assert summary["status"] == "blocked"
    assert "blocked_missing_model_source_runtime" in summary["blockers"]
    assert "blocked_missing_model_checkpoint" in summary["blockers"]
    assert "blocked_missing_runnable_adapter_command" not in summary["blockers"]
    for artifact_path in summary["artifact_paths"].values():
        assert Path(str(artifact_path)).is_file()

    manifest = json.loads(
        (tmp_path / "bootstrap" / "wam_model_runtime_bootstrap_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["candidate"]["model_repo_id"] == "zywu2115/OSCAR-2B"
    assert manifest["candidate"]["source_repo_url"] == "https://github.com/wuzy2115/oscar-public"
    assert manifest["claim_boundary"]["bootstrap_package_is_not_model_execution"] is True
    assert manifest["claim_boundary"]["raw_credentials_written_to_artifacts"] is False

    download_plan = json.loads(
        (tmp_path / "bootstrap" / "wam_model_checkpoint_download_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert download_plan["download_not_started_by_this_artifact"] is True
    assert "snapshot_download" in download_plan["download_command"]

    env_template = (tmp_path / "bootstrap" / "wam_model_runtime_env_template.sh").read_text(
        encoding="utf-8"
    )
    assert "BLUEPRINT_OSCAR_WAM_COMMAND" in env_template
    assert "BLUEPRINT_OSCAR_WAM_CHECKPOINT" in env_template
    assert "blueprint_pipeline.oscar_wam_command_adapter" in env_template
    assert "hf_" not in env_template.lower()


def test_wam_model_runtime_bootstrap_ready_when_command_and_paths_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    checkpoint_root = tmp_path / "checkpoint"
    source_root.mkdir()
    checkpoint_root.mkdir()
    command = tmp_path / "adapter.py"
    command.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        bootstrap,
        "_disk_status",
        lambda path, *, required_bytes: {
            "path": str(path),
            "free_bytes": 1,
            "required_bytes": required_bytes,
            "free_gib": 0.0,
            "required_gib": 0.5,
            "has_required_space": False,
        },
    )

    summary = bootstrap.build_bootstrap_package(
        candidate_id="openvla_policy",
        output_dir=tmp_path / "bootstrap",
        source_root=source_root,
        checkpoint_root=checkpoint_root,
        adapter_command=str(command),
        generated_at="now",
    )

    assert summary["status"] == "ready_for_wam_evaluator_configuration"
    assert summary["blockers"] == []
    manifest = json.loads(
        (tmp_path / "bootstrap" / "wam_model_runtime_bootstrap_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["source_status"]["exists"] is True
    assert manifest["checkpoint_status"]["exists"] is True
    assert manifest["adapter_command"]["available"] is True
    assert (
        manifest["provider_gate_status"]["providers"][2]["provider_id"]
        == "digitalocean_gpu"
    )
    assert "blocked_no_digitalocean_gpu_adapter_implemented" in manifest[
        "provider_gate_status"
    ]["providers"][2]["blockers"]
    image_plan = json.loads(
        (tmp_path / "bootstrap" / "wam_provider_reusable_image_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert image_plan["daily_reusable_image_recommended"] is True
    assert image_plan["claim_boundary"]["cpu_droplet_is_not_gpu_wam_runtime"] is True
    assert (tmp_path / "bootstrap" / "Dockerfile.wam-provider").is_file()


def test_wam_model_runtime_bootstrap_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code = bootstrap.main(["--candidate", "cosmos_wam", "--output-dir", str(tmp_path / "cli")])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_id"] == "cosmos_wam"
    assert (tmp_path / "cli" / "wam_model_provider_launch_request.json").is_file()


def test_wam_model_runtime_bootstrap_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bootstrap._timestamp().endswith("Z")
    assert bootstrap._mapping({"key": "value"}) == {"key": "value"}
    assert bootstrap._path_status(None) == {
        "configured": False,
        "path": None,
        "exists": False,
        "is_dir": False,
    }
    oscar_source = tmp_path / "oscar-source"
    (oscar_source / "inference").mkdir(parents=True)
    (oscar_source / "inference" / "inference_oscar.py").write_text("print('ok')\n", encoding="utf-8")
    assert bootstrap._candidate_source_ready("oscar_wam", oscar_source) is True
    other_source = tmp_path / "other-source"
    other_source.mkdir()
    assert bootstrap._candidate_source_ready("openvla_policy", other_source) is True
    checkpoint_file = tmp_path / "model.safetensors"
    checkpoint_file.write_text("weights", encoding="utf-8")
    assert bootstrap._candidate_checkpoint_ready(checkpoint_file) is True
    unsupported_file = tmp_path / "model.txt"
    unsupported_file.write_text("not weights", encoding="utf-8")
    assert bootstrap._candidate_checkpoint_ready(unsupported_file) is False
    missing_checkpoint_dir = tmp_path / "missing-checkpoint-dir"
    assert bootstrap._candidate_checkpoint_ready(missing_checkpoint_dir) is False
    empty_checkpoint_dir = tmp_path / "empty-checkpoint-dir"
    empty_checkpoint_dir.mkdir()
    assert bootstrap._candidate_checkpoint_ready(empty_checkpoint_dir) is False
    nested_checkpoint_dir = tmp_path / "nested-checkpoint-dir"
    (nested_checkpoint_dir / "shards").mkdir(parents=True)
    nested_shard = nested_checkpoint_dir / "shards" / "model.bin"
    nested_shard.write_text("weights", encoding="utf-8")
    assert bootstrap._candidate_checkpoint_ready(nested_checkpoint_dir) is True
    old_ready = tmp_path / "old-ready"
    new_ready = tmp_path / "new-ready"
    old_ready.mkdir()
    new_ready.mkdir()
    (old_ready / "marker").write_text("old", encoding="utf-8")
    (new_ready / "marker").write_text("new", encoding="utf-8")
    old_ready.touch()
    new_ready.touch()
    assert bootstrap._newest_ready_path(
        [old_ready, new_ready],
        ready=lambda path: (path / "marker").is_file(),
    ) in {old_ready, new_ready}
    assert bootstrap._newest_ready_path([old_ready], ready=lambda _path: False) is None
    monkeypatch.setenv("BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT", str(other_source))
    env_source, env_selection = bootstrap._discover_runtime_path(
        candidate_id="openvla_policy",
        candidate=bootstrap.BOOTSTRAP_CANDIDATES["openvla_policy"],
        job_root=tmp_path / "jobs",
        explicit_path=None,
        env_name="BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT",
        leaf="source",
        default_path=tmp_path / "default-source",
    )
    assert env_source == other_source.resolve()
    assert env_selection["selection_source"] == "BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT"
    assert bootstrap._command_available(None) is False
    assert bootstrap._command_available("'unterminated") is False
    assert bootstrap._command_available("   ") is False
    assert bootstrap._image_ref_is_versioned("") is False
    assert bootstrap._image_ref_is_versioned("registry.example/blueprint/wam:latest") is False
    assert bootstrap._image_ref_is_versioned("registry.example/blueprint/wam:20260621") is True

    command = tmp_path / "adapter"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    assert bootstrap._command_available(str(command)) is True

    source = tmp_path / "generic-source"
    source.mkdir()
    assert bootstrap._candidate_source_ready("openvla_policy", source) is True

    checkpoint_file = tmp_path / "model.safetensors"
    checkpoint_file.write_text("weights", encoding="utf-8")
    assert bootstrap._candidate_checkpoint_ready(checkpoint_file) is True
    non_checkpoint_file = tmp_path / "notes.txt"
    non_checkpoint_file.write_text("not weights", encoding="utf-8")
    assert bootstrap._candidate_checkpoint_ready(non_checkpoint_file) is False
    assert bootstrap._candidate_checkpoint_ready(tmp_path / "missing-checkpoint") is False
    checkpoint_dir = tmp_path / "checkpoint-dir"
    checkpoint_dir.mkdir()
    assert bootstrap._candidate_checkpoint_ready(checkpoint_dir) is False
    nested_checkpoint = checkpoint_dir / "nested" / "model.bin"
    nested_checkpoint.parent.mkdir()
    nested_checkpoint.write_text("weights", encoding="utf-8")
    assert bootstrap._candidate_checkpoint_ready(checkpoint_dir) is True

    env_source = tmp_path / "env-source"
    env_source.mkdir()
    monkeypatch.setenv("BOOTSTRAP_TEST_SOURCE_ROOT", str(env_source))
    discovered_source, source_selection = bootstrap._discover_runtime_path(
        candidate_id="openvla_policy",
        candidate=bootstrap.BOOTSTRAP_CANDIDATES["openvla_policy"],
        job_root=tmp_path / "jobs",
        explicit_path=None,
        env_name="BOOTSTRAP_TEST_SOURCE_ROOT",
        leaf="source",
        default_path=tmp_path / "default-source",
    )
    assert discovered_source == env_source.resolve()
    assert source_selection["selection_source"] == "BOOTSTRAP_TEST_SOURCE_ROOT"

    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="abc123\tHEAD\n", stderr=""),
    )
    reachable = bootstrap._git_ls_remote_probe("https://example.com/repo.git")
    assert reachable["status"] == "reachable"
    assert reachable["head_sha"] == "abc123"

    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    )
    blocked = bootstrap._git_ls_remote_probe("https://example.com/repo.git")
    assert blocked["status"] == "blocked"
    assert blocked["stderr_omitted"] is True

    class FakeHfApi:
        def model_info(self, repo_id: str, *, files_metadata: bool):
            assert repo_id == "repo/model"
            assert files_metadata is True
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="small.json", size=10),
                    SimpleNamespace(rfilename="large.safetensors", size=100),
                ]
            )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi))
    metadata = bootstrap._hf_repo_metadata("repo/model")
    assert metadata["status"] == "completed"
    assert metadata["total_bytes"] == 110
    assert metadata["largest_files"][0]["rfilename"] == "large.safetensors"

    class RaisingHfApi:
        def model_info(self, repo_id: str, *, files_metadata: bool):
            del repo_id, files_metadata
            raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=RaisingHfApi))
    blocked_metadata = bootstrap._hf_repo_metadata("repo/model")
    assert blocked_metadata["status"] == "blocked"
    assert blocked_metadata["error_type"] == "RuntimeError"

    with pytest.raises(ValueError):
        bootstrap.build_bootstrap_package(candidate_id="missing_candidate", output_dir=tmp_path / "bad")

    missing_adapter = bootstrap.build_bootstrap_package(
        candidate_id="openvla_policy",
        output_dir=tmp_path / "missing-adapter",
        source_root=tmp_path / "missing-source",
        checkpoint_root=tmp_path / "missing-checkpoint",
        adapter_command=str(tmp_path / "missing-command"),
        generated_at="now",
    )
    assert "blocked_missing_runnable_adapter_command" in missing_adapter["blockers"]

    monkeypatch.setenv("BLUEPRINT_WAM_PROVIDER_IMAGE_REF", "registry.example/blueprint/oscar-wam:latest")
    unversioned_image_plan = bootstrap._provider_image_plan(
        candidate_id="oscar_wam",
        output_dir=tmp_path / "unversioned-image-plan",
    )
    assert "blocked_wam_provider_image_ref_not_versioned" in unversioned_image_plan["blockers"]


def test_wam_model_runtime_bootstrap_provider_image_plan_with_file_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    username = tmp_path / "docker-username"
    pat = tmp_path / "docker-pat"
    username.write_text("docker-user\n", encoding="utf-8")
    pat.write_text("docker-token-should-not-leak\n", encoding="utf-8")
    username.chmod(0o600)
    pat.chmod(0o600)
    monkeypatch.setenv("BLUEPRINT_WAM_PROVIDER_IMAGE_REF", "registry.example/blueprint/oscar-wam:20260621")
    monkeypatch.setenv("DOCKER_USERNAME_FILE", str(username))
    monkeypatch.setenv("DOCKER_PAT_FILE", str(pat))

    summary = bootstrap.build_bootstrap_package(
        candidate_id="oscar_wam",
        output_dir=tmp_path / "image-plan",
        job_root=tmp_path / "jobs",
        generated_at="now",
    )

    plan_path = Path(summary["artifact_paths"]["provider_image_plan"])
    image_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert image_plan["status"] == "ready_for_manual_image_build_and_push"
    assert image_plan["configured_image_ref_is_versioned"] is True
    assert image_plan["registry_auth"]["docker_pat_file"]["mode_is_0600"] is True
    assert "docker-token-should-not-leak" not in plan_path.read_text(encoding="utf-8")
    dockerfile = Path(summary["artifact_paths"]["provider_dockerfile"]).read_text(encoding="utf-8")
    assert "blueprint_pipeline.oscar_wam_command_adapter" in dockerfile
