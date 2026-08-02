from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "postshot_windows_worker" / "launch_postshot_worker.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("postshot_windows_worker_launcher", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_template_encodes_curated_pulses_and_two_watchdogs() -> None:
    launcher = _load_launcher()
    template = launcher.BOOTSTRAP_TEMPLATE
    assert "Start-Transcript" not in template
    assert "worker_pulse.v2" in template
    assert "pulse-series.jsonl" in template
    assert "New-ScheduledTaskTrigger" in template
    assert "Start-ScheduledTask" in template
    assert "Start-Process -FilePath PowerShell.exe" in template
    assert "Stop-Computer -Force" in template
    assert 'InstanceInitiatedShutdownBehavior="terminate"' in Path(SCRIPT).read_text(encoding="utf-8")
    assert '"--login",$lic["POSTSHOT_LOGIN_EMAIL"],"--password",$lic["POSTSHOT_LOGIN_PASSWORD"],"train"' in template
    assert '"--no-recenter-points"' in template
    assert '"--train-steps-limit","1"' in template
    assert '"--max-num-splats","100"' in template
    assert 'C0_canary_splat3.psht' in template
    assert 'C0_canary_splat3.ply' in template
    assert '"--max-steps"' not in template


def test_rendered_user_data_keeps_transport_urls_out_of_local_receipt() -> None:
    launcher = _load_launcher()
    urls = {key: f"https://example.invalid/{key}?signature=secret" for key in launcher.URL_EXPIRIES_SECONDS}
    rendered = launcher._render_user_data(run_id="postshot-20260801T215521Z", staging={"staging_digest": "sha256:" + "0" * 64, "keys": {"license": "blueprint-postshot-bakeoff/postshot-20260801T215521Z/license.env"}}, urls=urls, instance_type="g6.xlarge")
    assert "https://example.invalid" in rendered  # sent as worker user-data only
    assert "POSTSHOT_LOGIN_PASSWORD" in rendered
    assert "url_expiry_seconds" not in rendered


def test_admission_fails_closed_without_exact_authorization(tmp_path: Path) -> None:
    launcher = _load_launcher()
    run_id = "postshot-20260801T215521Z"
    state_dir = tmp_path / "provider_packets" / "postshot" / run_id
    state_dir.mkdir(parents=True)
    (state_dir / "staging.json").write_text(json.dumps({"staging_digest": "sha256:" + "0" * 64}), encoding="utf-8")
    packet = {
        "source_capture_digest": "sha256:" + "1" * 64,
        "pose_only_dataset_digest": "sha256:" + "2" * 64,
        "frozen_split_digest": "sha256:" + "3" * 64,
        "hidden_images_included": False,
        "provider_sees_hidden_views": False,
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    receipt = launcher.admit(Namespace(proxy_root=str(tmp_path), run_id=run_id, execution_packet=str(packet_path), authorization_file=None, focused_test_receipt=None))
    assert receipt["launch_allowed"] is False
    assert receipt["authorization"]["authorized"] is False
    assert "AUTHORIZE_POSTSHOT_ATTEMPT_5" in receipt["authorization"]["line_required"]


def test_candidate_archive_digest_is_stable_across_file_mtime_changes(tmp_path: Path) -> None:
    launcher = _load_launcher()
    root = tmp_path / "dataset"
    root.mkdir()
    file_path = root / "images" / "frame 01.jpg"
    file_path.parent.mkdir()
    file_path.write_bytes(b"candidate-frame")
    first = launcher._deterministic_zip_bytes(root)
    file_path.touch()
    second = launcher._deterministic_zip_bytes(root)
    assert first == second


def test_ledger_operation_is_local_and_does_not_launch(tmp_path: Path) -> None:
    launcher = _load_launcher()
    run_id = "postshot-20260801T215521Z"
    launcher.ledger(Namespace(proxy_root=str(tmp_path), run_id=run_id))
    path = tmp_path / "provider_packets" / "postshot" / "attempt_ledger.v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "postshot_attempt_ledger.v1"
    assert len(value["attempts"]) == 4
    assert not (tmp_path / "provider_packets" / "postshot" / run_id / "launch.json").exists()


def test_cli_help_exposes_required_control_operations() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=False, capture_output=True, text=True)
    assert result.returncode == 0
    for operation in ("stage", "admit", "launch", "watch", "status", "collect", "abort", "teardown", "inventory", "reconcile"):
        assert operation in result.stdout


def test_terminate_targets_only_the_tag_verified_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()

    class FakeEc2:
        def __init__(self, tag_value: str) -> None:
            self.tag_value = tag_value
            self.terminated: list[str] = []

        def describe_instances(self, *, InstanceIds: list[str]):
            return {"Reservations": [{"Instances": [{"InstanceId": InstanceIds[0], "State": {"Name": "running"}, "InstanceType": "g6.xlarge", "Tags": [{"Key": "blueprint-run", "Value": self.tag_value}], "BlockDeviceMappings": []}]}]}

        def terminate_instances(self, *, InstanceIds: list[str]):
            self.terminated.extend(InstanceIds)
            return {"TerminatingInstances": []}

    class FakeSession:
        def __init__(self, ec2: FakeEc2) -> None:
            self.ec2 = ec2

        def client(self, name: str):
            assert name == "ec2"
            return self.ec2

    ec2 = FakeEc2("postshot-20260801T215521Z")
    monkeypatch.setattr(launcher, "_aws_session", lambda: FakeSession(ec2))
    result = launcher._terminate_exact(run_id="postshot-20260801T215521Z", launch={"instance_id": "i-target"})
    assert result["requested"] is True
    assert ec2.terminated == ["i-target"]

    mismatched = FakeEc2("postshot-other-run")
    monkeypatch.setattr(launcher, "_aws_session", lambda: FakeSession(mismatched))
    result = launcher._terminate_exact(run_id="postshot-20260801T215521Z", launch={"instance_id": "i-unrelated"})
    assert result["requested"] is False
    assert result["reason"] == "exact_instance_tag_not_verified"
    assert mismatched.terminated == []
