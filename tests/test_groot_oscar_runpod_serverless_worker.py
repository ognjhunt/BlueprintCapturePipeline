import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_runpod_serverless_worker as worker


def test_handler_rejects_unknown_operation() -> None:
    result = worker.handler({"input": {"operation": "shell"}})

    assert result["status"] == "blocked"
    assert result["blockers"] == ["serverless_operation_not_allowed"]
    assert result["raw_secret_values_recorded"] is False


def test_handler_routes_allowed_job_through_isolated_oscar_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def fake_run(job_input):
        observed.update(job_input)
        return {"status": "completed", "runtime_present": True}

    monkeypatch.setattr(worker, "_run_in_oscar_runtime", fake_run)
    result = worker.handler({"input": {"operation": "startup"}})

    assert result["status"] == "completed"
    assert observed == {"operation": "startup"}


def test_safe_volume_path_rejects_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(worker, "NETWORK_VOLUME_ROOT", tmp_path)

    with pytest.raises(ValueError, match="network_volume_relative_path_invalid"):
        worker._safe_volume_path("../secret")


def test_startup_keeps_execution_claims_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / worker.MODEL_CACHE_RELATIVE).mkdir(parents=True)
    monkeypatch.setattr(worker, "NETWORK_VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(worker, "RUNTIME_CACHE_LINK", tmp_path / "runtime-cache-link")
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir

    def fake_is_file(path: Path) -> bool:
        if str(path) in {
            "/opt/oscar-venv/bin/python",
            "/opt/gr00t-venv/bin/python",
            "/isaac-sim/python.sh",
        }:
            return True
        return original_is_file(path)

    def fake_is_dir(path: Path) -> bool:
        if str(path) in {"/opt/gr00t", "/opt/OSCAR"}:
            return True
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA A40"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=FakeCuda(), __version__="test"),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_model_cache.activate_model_cache",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "blockers": [],
            "model_manifest_digest": "sha256:" + "a" * 64,
            "verified_file_count": 30,
            "verified_size_bytes": 16_791_338_353,
            "runtime_links_activated": True,
        },
    )
    result = worker._startup()

    assert result["status"] == "completed"
    assert result["runtime_present"] is True
    assert result["model_execution_proven"] is False
    assert result["simulator_execution_proven"] is False
    assert result["semantic_task_success"] is None


def test_robot_eval_requires_manifest_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "jobs" / "smoke" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(worker, "NETWORK_VOLUME_ROOT", tmp_path)

    result = worker._robot_eval(
        {
            "manifest_relative_path": "jobs/smoke/manifest.json",
            "manifest_sha256": "0" * 64,
            "output_relative_path": "jobs/smoke/output",
            "timeout_seconds": 300,
        }
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["worker_manifest_sha256_mismatch"]


def test_robot_eval_runs_only_hashed_network_volume_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "jobs" / "smoke" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"job_id":"smoke"}', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "NETWORK_VOLUME_ROOT", tmp_path)

    def fake_run(**kwargs):
        assert kwargs["manifest_uri"] == str(manifest.resolve())
        assert kwargs["timeout_seconds"] == 300
        assert kwargs["allowed_simulators"] == ("isaac_sim",)
        assert kwargs["artifact_output_uri"].startswith(str(tmp_path.resolve()))
        return {
            "schema_version": "robot_eval_worker_runtime.v1",
            "status": "completed",
            "job_id": "smoke",
            "simulator": "isaac_sim",
            "job_status": "completed",
            "evaluation_status": "completed",
            "simulator_execution_proven": True,
            "policy_execution_proven": True,
            "semantic_task_success": False,
            "blockers": [],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.robot_eval_worker.run_robot_eval_worker", fake_run
    )
    result = worker._robot_eval(
        {
            "manifest_relative_path": "jobs/smoke/manifest.json",
            "manifest_sha256": digest,
            "output_relative_path": "jobs/smoke/output",
            "timeout_seconds": 300,
        }
    )

    assert result["status"] == "completed"
    assert result["simulator_execution_proven"] is True
    assert result["policy_execution_proven"] is True
    assert result["semantic_task_success"] is False


def test_kitchen_campaign_rechecks_startup_and_uses_fixed_bounded_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(worker, "NETWORK_VOLUME_ROOT", tmp_path)
    monkeypatch.setenv("BLUEPRINT_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setenv("BLUEPRINT_WORKER_IMAGE_DIGEST", "image@sha256:" + "d" * 64)
    monkeypatch.setenv(
        "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST",
        "sha256:" + "e" * 64,
    )
    monkeypatch.setattr(
        worker,
        "_startup",
        lambda: {"status": "completed", "runtime_present": True},
    )
    observed = {}

    def fake_campaign(**kwargs):
        observed.update(kwargs)
        return {
            "status": "completed",
            "blockers": [],
            "semantic_task_success_not_inferred_from_execution": True,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_serverless_campaign_worker.run_kitchen_campaign",
        fake_campaign,
    )
    result = worker._kitchen_campaign(
        {
            "expected_runtime_worker_identity_sha256": worker._base_result(
                "test"
            )["runtime_worker_identity_sha256"],
            "campaign_manifest_relative_path": "campaign/input.json",
            "campaign_manifest_sha256": "f" * 64,
            "output_relative_path": "campaign/output",
            "timeout_seconds": 999999,
        }
    )

    assert result["status"] == "completed"
    assert observed["network_volume_root"] == tmp_path
    assert observed["source_commit"] == "c" * 40
    assert observed["image_ref"] == "image@sha256:" + "d" * 64
    assert observed["model_manifest_digest"] == "sha256:" + "e" * 64
    assert worker._isolated_timeout({"operation": "kitchen-campaign"}) == 3_630


def test_kitchen_campaign_does_not_run_when_cache_recheck_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "_startup",
        lambda: {"status": "blocked", "runtime_present": False},
    )

    result = worker._kitchen_campaign(
        {
            "expected_runtime_worker_identity_sha256": worker._base_result(
                "test"
            )["runtime_worker_identity_sha256"]
        }
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["campaign_runtime_or_model_cache_not_ready"]


def test_kitchen_campaign_requires_same_worker_as_strict_probe() -> None:
    result = worker._kitchen_campaign(
        {"expected_runtime_worker_identity_sha256": "0" * 64}
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["campaign_runtime_worker_identity_mismatch"]
