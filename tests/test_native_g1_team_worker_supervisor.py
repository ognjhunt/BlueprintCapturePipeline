"""The selected-policy controller needs terminal child proof, not a wait timeout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_team_worker_supervisor as supervisor
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_policy_worker import FILENAME, PRECLOSE_FILENAME
from blueprint_pipeline.native_g1_team_paid_output import verify_g1_team_paid_output
from tests.test_native_g1_team_paid_output import _evidence


def _exit(root: Path) -> None:
    result = supervisor.run_g1_team_worker_process(
        command=[sys.executable, "-c", "pass"],
        diagnostics_dir=root / "child-diagnostics", timeout_seconds=5,
    )
    (root / "worker.exit.json").write_text(json.dumps(result))


def test_clean_isaac_exit_recovers_only_bound_preclose_and_verified_media(
    tmp_path: Path, monkeypatch,
) -> None:
    args, episode = _evidence(tmp_path, monkeypatch)
    root = args["output_dir"]
    _exit(root)
    # The production child can exit natively after retaining preclose.
    (root / FILENAME).unlink()
    recovered = supervisor.recover_g1_team_worker_result(
        worker_output_dir=root, execution_packet=args["execution_packet"],
        returncode=0,
    )
    assert recovered["status"] == "completed_development_only"
    assert recovered["teardown"]["simulator"] == "process_exited_after_close_request"
    verified = verify_g1_team_paid_output(**args)
    assert verified["policy_query_count"] == episode["policy_query_count"]


@pytest.mark.parametrize("returncode", [1, -9, True])
def test_failed_child_never_recovers_as_completed(
    tmp_path: Path, monkeypatch, returncode,
) -> None:
    args, _ = _evidence(tmp_path, monkeypatch)
    root = args["output_dir"]
    _exit(root)
    (root / FILENAME).unlink()
    with pytest.raises(ValueError, match="child_exit_invalid"):
        supervisor.recover_g1_team_worker_result(
            worker_output_dir=root, execution_packet=args["execution_packet"],
            returncode=returncode,
        )
    assert not (root / FILENAME).exists()


def test_foreign_preclose_and_missing_video_cannot_be_delivered(
    tmp_path: Path, monkeypatch,
) -> None:
    args, _ = _evidence(tmp_path, monkeypatch)
    root = args["output_dir"]
    _exit(root)
    worker = json.loads((root / FILENAME).read_text())
    (root / FILENAME).unlink()
    preclose = json.loads((root / PRECLOSE_FILENAME).read_text())
    preclose["execution_packet_digest"] = "sha256:" + "0" * 64
    preclose["preclose_digest"] = canonical_digest(preclose, digest_field="preclose_digest")
    (root / PRECLOSE_FILENAME).write_text(json.dumps(preclose))
    with pytest.raises(ValueError, match="preclose_binding_invalid"):
        supervisor.recover_g1_team_worker_result(
            worker_output_dir=root, execution_packet=args["execution_packet"], returncode=0,
        )
    preclose["execution_packet_digest"] = worker["execution_packet_digest"]
    preclose["preclose_digest"] = canonical_digest(preclose, digest_field="preclose_digest")
    (root / PRECLOSE_FILENAME).write_text(json.dumps(preclose))
    supervisor.recover_g1_team_worker_result(
        worker_output_dir=root, execution_packet=args["execution_packet"], returncode=0,
    )
    verified = verify_g1_team_paid_output(**args)
    (root / verified["media"]["review_videos"]["head"]["relative_path"]).unlink()
    with pytest.raises(ValueError, match="g1_pair_media"):
        verify_g1_team_paid_output(**args)


def test_supervisor_retains_real_child_exit_without_exposing_stderr(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    result = supervisor.run_g1_team_worker_process(
        command=[sys.executable, "-c", "import sys; print('private endpoint detail'); sys.exit(7)"],
        diagnostics_dir=diagnostics, timeout_seconds=5,
    )
    assert result["status"] == "exited"
    assert result["returncode"] == 7
    assert "private endpoint detail" not in json.dumps(result)
    assert "private endpoint detail" in (diagnostics / "worker.log").read_text()
    assert (diagnostics / "worker.log").stat().st_mode & 0o077 == 0
    assert json.loads((diagnostics / "worker.exit.json").read_text()) == result


def test_timeout_kills_child_and_records_timeout_not_terminal_success(tmp_path: Path) -> None:
    result = supervisor.run_g1_team_worker_process(
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
        diagnostics_dir=tmp_path / "diagnostics", timeout_seconds=0.05,
    )
    assert result["status"] == "timed_out"
    assert result["returncode"] is not None
    assert result["process_group_kill_requested"] is True


def test_launch_failure_has_typed_private_receipt(tmp_path: Path) -> None:
    result = supervisor.run_g1_team_worker_process(
        command=[str(tmp_path / "missing-launcher")],
        diagnostics_dir=tmp_path / "diagnostics", timeout_seconds=5,
    )
    assert result["status"] == "launch_failed"
    assert result["error_type"] == "FileNotFoundError"
    assert result["returncode"] is None


def test_recovery_requires_retained_exit_bytes_not_just_a_zero_argument(tmp_path: Path, monkeypatch):
    args, _ = _evidence(tmp_path, monkeypatch)
    root = args["output_dir"]
    (root / FILENAME).unlink()
    with pytest.raises(ValueError, match="receipt_unavailable"):
        supervisor.recover_g1_team_worker_result(
            worker_output_dir=root, execution_packet=args["execution_packet"], returncode=0,
        )
    _exit(root)
    receipt = json.loads((root / "worker.exit.json").read_text())
    receipt["status"] = "timed_out"
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (root / "worker.exit.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="child_exit_invalid"):
        supervisor.recover_g1_team_worker_result(
            worker_output_dir=root, execution_packet=args["execution_packet"], returncode=0,
        )


@pytest.mark.parametrize("native_close", [False, True])
def test_selected_runner_drives_child_verification_and_preserves_task_score(
    tmp_path: Path, monkeypatch, native_close: bool,
) -> None:
    args, episode = _evidence(tmp_path, monkeypatch)
    source = args["output_dir"]
    if native_close:
        (source / FILENAME).unlink()
    packet = args["execution_packet"]
    monkeypatch.setattr(supervisor, "_execution_packet", lambda path, commit: packet)
    monkeypatch.setattr(supervisor, "_verify_packet", lambda root: {
        "receipt_digest": args["scene_packet_receipt_digest"],
        "arena_scene_plan_digest": args["scene_plan_digest"],
    })
    real_run = supervisor.run_g1_team_worker_process
    destination = tmp_path / "selected-run" / "worker"

    def staged_child(**kwargs):
        assert "blueprint_pipeline.native_g1_team_policy_worker" in kwargs["command"]
        command = [sys.executable, "-c",
                   "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2])",
                   str(source), str(destination)]
        return real_run(**{**kwargs, "command": command})

    monkeypatch.setattr(supervisor, "run_g1_team_worker_process", staged_child)
    result = supervisor.run_supervised_g1_team_worker(
        worker_arguments={
            "execution_packet_path": tmp_path / "execution.json",
            "expected_implementation_commit": "a" * 40,
            "scene_packet_root": tmp_path / "scene",
            "runtime_provisioning_receipt_path": tmp_path / "provision.json",
            "sonic_provider_source": tmp_path / "sonic.py",
            "sonic_encoder": tmp_path / "encoder.onnx",
            "sonic_encoder_sha256": "sha256:" + "a" * 64,
            "sonic_decoder": tmp_path / "decoder.onnx",
            "sonic_decoder_sha256": "sha256:" + "b" * 64,
            "max_steps": 2,
        },
        output_dir=destination.parent, worker_launcher=Path(sys.executable),
        timeout_seconds=5,
    )
    assert result["status"] == "completed_development_only"
    assert result["verified_output"]["score"] == episode["score"]
    assert result["verified_output"]["policy_query_count"] == 2
    assert result["recovered_native_close"] is native_close
    assert result["provider_teardown_verified"] is False
    assert result["official_billing_reconciled"] is False
    assert json.loads((destination.parent / supervisor.RESULT_FILENAME).read_text()) == result


def test_supervisor_seals_redacted_input_failure_without_starting_child(tmp_path: Path, monkeypatch):
    def invalid_packet(path, commit):
        raise ValueError("private endpoint response contains credential")

    monkeypatch.setattr(supervisor, "_execution_packet", invalid_packet)

    def must_not_run(**kwargs):
        raise AssertionError("unapproved input reached a child")

    monkeypatch.setattr(supervisor, "run_g1_team_worker_process", must_not_run)
    result = supervisor.run_supervised_g1_team_worker(
        worker_arguments={
            "execution_packet_path": tmp_path / "execution.json",
            "expected_implementation_commit": "a" * 40,
            "scene_packet_root": tmp_path / "scene",
            "runtime_provisioning_receipt_path": tmp_path / "provision.json",
            "sonic_provider_source": tmp_path / "sonic.py",
            "sonic_encoder": tmp_path / "encoder.onnx",
            "sonic_encoder_sha256": "sha256:" + "a" * 64,
            "sonic_decoder": tmp_path / "decoder.onnx",
            "sonic_decoder_sha256": "sha256:" + "b" * 64,
            "max_steps": 2,
        },
        output_dir=tmp_path / "selected-run", worker_launcher=Path(sys.executable),
    )
    assert result["status"] == "blocked"
    assert result["stage_reached"] == "input_verification"
    assert result["blocker_code"] == "g1_team_supervised_worker_validation_failed"
    assert result["child_exit_receipt_digest"] is None
    assert "private endpoint" not in json.dumps(result)
