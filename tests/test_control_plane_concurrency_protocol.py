"""Only live original storage holds establish the requested N-way milestone."""

import json
import os

import pytest


def test_overlap_requires_every_live_child_and_its_original_positive_hold(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from scripts.control_plane_concurrency_protocol import observe_overlap

    reservations = tmp_path / "reservations"
    reservations.mkdir()
    ready = tmp_path / "ready"
    ready.mkdir()
    script = """import json,os,sys
from pathlib import Path
from collections import namedtuple
from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
root,ready,key=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
usage=namedtuple('Usage','total used free')(200*1024**3,50*1024**3,150*1024**3)
hold=reserve_control_plane_disk('policy_canary_output',target_root=root,expected_bytes=4096,
    reservation_root=root,disk_usage=lambda _:usage)
(ready/(key+'.json')).write_text(json.dumps({'scene_key':key,'pid':os.getpid(),
    'lease_path':str(hold.path),'expected_bytes':hold.expected_bytes}))
print('ready',flush=True)
try:sys.stdin.readline()
finally:hold.release()
"""
    processes = []
    try:
        for key in ("one", "two"):
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(reservations), str(ready), key],
                env={
                    "PATH": os.environ["PATH"],
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                },
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            assert process.stdout.readline().strip() == "ready"
        children = [
            {"scene_key": key, "pid": process.pid, "running": process.poll() is None}
            for key, process in zip(("one", "two"), processes)
        ]
        assert (
            observe_overlap(children=children, ready_root=ready, reservation_root=reservations) == 2
        )
        children[0]["running"] = False
        with pytest.raises(ValueError, match="overlap_child_not_running"):
            observe_overlap(children=children, ready_root=ready, reservation_root=reservations)
        children[0]["running"] = True
        lease = Path(json.loads((ready / "one.json").read_text())["lease_path"])
        value = json.loads(lease.read_text())
        value["expected_bytes"] = 0
        lease.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="overlap_reservation_invalid"):
            observe_overlap(children=children, ready_root=ready, reservation_root=reservations)
    finally:
        for process in processes:
            process.communicate("\n", timeout=5)


def test_overlap_rejects_a_foreign_or_expired_reservation(tmp_path):
    from scripts.control_plane_concurrency_protocol import observe_overlap

    ready = tmp_path / "ready"
    ready.mkdir()
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}")
    (ready / "one.json").write_text(
        json.dumps(
            {
                "scene_key": "one",
                "pid": os.getpid(),
                "lease_path": str(foreign),
                "expected_bytes": 4096,
            }
        )
    )
    with pytest.raises(ValueError, match="overlap_reservation_not_owned"):
        observe_overlap(
            children=[{"scene_key": "one", "pid": os.getpid(), "running": True}],
            ready_root=ready,
            reservation_root=reservations,
        )
