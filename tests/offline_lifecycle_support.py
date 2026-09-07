"""Opt-in fixture isolation: load with pytest -p tests.offline_lifecycle_support.

The operating-system sandbox remains required for subprocess containment.
"""
from functools import partial
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def offline_edges(tmp_path, monkeypatch):
    import socket
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as preparation
    from blueprint_pipeline import control_plane_evidence_offload as offload

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline_lifecycle_external_network_forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(tmp_path / "disk"))
    # Optional credential discovery must see test-owned absence, never home defaults.
    for name in ("HF_TOKEN_FILE", "NGC_API_KEY_FILE", "DOCKER_USERNAME_FILE", "DOCKER_PAT_FILE"):
        monkeypatch.setenv(name, str(tmp_path / ("absent-" + name.lower())))

    for module in (preparation, offload):
        monkeypatch.setattr(module, "reserve_control_plane_disk", partial(
            module.reserve_control_plane_disk,
            disk_usage=lambda _path: SimpleNamespace(total=512 * 2**30, used=128 * 2**30, free=384 * 2**30),
        ))


# Audit open events add a reviewable process-level ledger to the OS policy.
# They are attempted writes, not a claim to observe every C-extension syscall.
def pytest_sessionstart(session):
    import os
    import sys
    from pathlib import Path
    root = Path(str(session.config.rootpath)).resolve()
    session.config._offline_write_paths = set()
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        _path, mode, flags = args
        if not ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))):
            return
        session.config._offline_write_paths.add(str(Path(os.fsdecode(_path)).resolve()))
    sys.addaudithook(audit)
    session.config._offline_root = root


def pytest_sessionfinish(session, exitstatus):
    import json
    import os
    from pathlib import Path
    root = session.config._offline_root
    paths = sorted(session.config._offline_write_paths)
    outside = [path for path in paths if path != "/dev/null" and not Path(path).is_relative_to(root)]
    destination = root / "audit/evidence" / f"containment-{os.getpid()}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "pytest_exitstatus": int(exitstatus), "root": str(root),
        "python_attempted_write_paths": paths,
        "python_outside_root_write_attempts": outside,
        "process_environment": {key: value for key, value in os.environ.items()
            if key in {"HOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "TMPDIR"}},
        "network_fixture_guard": "socket connect/connect_ex/create_connection refuse",
        "os_policy_required": "audit/offline.sb: deny network and writes outside root",
        "inventory_scope": "synthetic_only",
    }, indent=2) + "\n")
