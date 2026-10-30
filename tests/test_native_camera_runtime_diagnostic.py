"""CPU scheduling and evidence tests only; SDK camera data is injected."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_camera_runtime_diagnostic as diagnostic
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


class Camera:
    native = False
    closed = False

    def __init__(self, app):
        self.calls = 0

    def frame(self):
        self.calls += 1
        return {"shape": [64, 64, 3], "device": "cuda:0", "dtype": "torch.uint8", "finite": True,
                "minimum": 0., "maximum": 100., "sha256": "sha256:" + "0" * 64}

    def close(self):
        self.closed = True


def test_frame_evidence_is_sealed_without_task_or_site_claim(tmp_path):
    camera = Camera(None)
    result = diagnostic.run_camera_diagnostic(None, tmp_path, deadline_epoch=1010.,
        backend_factory=lambda _: camera, wall=lambda: 1000.)
    assert result["status"] == "passed" and camera.closed
    assert result["native_isaac_executed"] is False
    assert result["task_executed"] is result["policy_executed"] is result["site_appearance_exercised"] is False
    assert json.loads((tmp_path / diagnostic.FILENAME).read_text()) == result
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")


@pytest.mark.parametrize("bad", [{"device": "cpu"}, {"finite": False}, {"maximum": 0.}, {"shape": [32, 32, 3]}])
def test_invalid_camera_frame_cannot_pass(tmp_path, bad):
    camera = Camera(None)
    original = camera.frame
    camera.frame = lambda: {**original(), **bad}
    result = diagnostic.run_camera_diagnostic(None, tmp_path, deadline_epoch=1010.,
        backend_factory=lambda _: camera, wall=lambda: 1000.)
    assert result["status"] == "failed" and camera.calls == diagnostic.MAX_TICKS


def test_inherited_deadline_reserves_teardown_and_never_resets_after_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "1600")
    result = diagnostic.run_camera_diagnostic(None, tmp_path, deadline_epoch=2000., wall=lambda: 1000.,
        backend_factory=lambda _: pytest.fail("no remaining provider time"))
    assert result["status"] == "failed" and result["ticks"] == 0
    assert result["deadline_epoch"] == 1000.


def test_frame_returning_after_deadline_cannot_pass(tmp_path):
    clock = SimpleNamespace(value=1000.)
    camera = Camera(None)
    original = camera.frame

    def late():
        clock.value = 1011.
        return original()

    camera.frame = late
    result = diagnostic.run_camera_diagnostic(None, tmp_path, deadline_epoch=1010.,
        backend_factory=lambda _: camera, wall=lambda: clock.value)
    assert result["status"] == "failed" and camera.closed


def test_parent_bounds_even_blocking_kit_startup(tmp_path):
    def stuck(command, **options):
        assert options["timeout"] == 90.
        assert command[command.index("--deadline-epoch") + 1] == "1085.0"
        raise subprocess.TimeoutExpired(command, options["timeout"])

    result = diagnostic.run_bounded_camera_process(provisioning_receipt=tmp_path / "receipt.json", output=tmp_path,
        deadline_epoch=2000., wall=lambda: 1000., run=stuck)
    assert result["status"] == "failed" and "TimeoutExpired" in result["reason"]


def test_parent_never_reuses_an_old_passed_receipt(tmp_path):
    diagnostic.run_camera_diagnostic(None, tmp_path, deadline_epoch=1010., backend_factory=Camera, wall=lambda: 1000.)
    result = diagnostic.run_bounded_camera_process(provisioning_receipt=tmp_path / "receipt.json", output=tmp_path,
        deadline_epoch=2000., wall=lambda: 1000., run=lambda *_a, **_kw: pytest.fail("existing result"))
    assert result["reason"] == "camera_diagnostic_result_already_exists"


def test_nonzero_child_cannot_be_an_overall_pass(tmp_path):
    def fail_after_frame(_command, **_options):
        diagnostic.run_camera_diagnostic(None, tmp_path, deadline_epoch=1085., backend_factory=Camera, wall=lambda: 1000.)
        return SimpleNamespace(returncode=2)

    result = diagnostic.run_bounded_camera_process(provisioning_receipt=tmp_path / "receipt.json", output=tmp_path,
        deadline_epoch=2000., wall=lambda: 1000., run=fail_after_frame)
    assert result["status"] == "failed" and result["returncode"] == 2
    assert json.loads((tmp_path / diagnostic.FILENAME).read_text())["status"] == "passed"


def test_real_shell_child_is_terminated_by_the_bound(tmp_path):
    import os
    from pathlib import Path
    import sys
    from blueprint_pipeline.scene_component_process import run_component_process
    marker = tmp_path / "child.pid"
    child = "import os,time,pathlib; pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_component_process(["bash", "-c", '"$1" -c "$2" & wait', "development-only", sys.executable, child], timeout=1.)
    pid = int(marker.read_text())
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text().split()[2]
    except FileNotFoundError:
        state = None  # Reaped between parent exit and this read.
    assert state in (None, "Z")
    assert pid != os.getpid()


def test_cli_retains_actual_failed_launch_readback(tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_isaaclab_launch as launch
    observed = {"runtime_readback": {"modules": {"torch": {"version": "wrong-observed-version"}}}}

    def fail(*_args, **_kwargs):
        raise launch.NativeTaskIsaacLabLaunchError(["native_task_isaaclab_post_start_identity_failed"], diagnostics=observed)

    monkeypatch.setattr(launch, "launch_native_task_isaaclab", fail)
    assert diagnostic.main(["--provisioning-receipt", "development-only.json", "--output", str(tmp_path),
                            "--deadline-epoch", "9999999999"]) == 1
    retained = json.loads((tmp_path / "launch.json").read_text())
    assert retained["status"] == "failed" and retained["diagnostics"] == observed
    assert retained["task_executed"] is retained["policy_executed"] is False


def test_cli_seals_camera_result_before_process_terminating_close(tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_isaaclab_launch as launch

    def terminate():
        assert json.loads((tmp_path / diagnostic.FILENAME).read_text())["status"] == "passed"
        raise SystemExit(0)

    monkeypatch.setattr(launch, "launch_native_task_isaaclab", lambda *_a, **_kw: (SimpleNamespace(close=terminate), {"development_only": True}))
    original = diagnostic.run_camera_diagnostic
    monkeypatch.setattr(diagnostic, "run_camera_diagnostic", lambda app, output, **kw: original(app, output, backend_factory=Camera, **kw))
    with pytest.raises(SystemExit):
        diagnostic.main(["--provisioning-receipt", "development-only.json", "--output", str(tmp_path),
                         "--deadline-epoch", "9999999999"])
    assert json.loads((tmp_path / diagnostic.FILENAME).read_text())["native_isaac_executed"] is False
