"""A reconstruction stage replays on its retained splat and mesh, writes only under its replay root, and goes offline."""
# Covers: src/blueprint_pipeline/atlas_reconstruction_stage_replay.py
from __future__ import annotations

import socket

from blueprint_pipeline import atlas_reconstruction_stage_replay as stage_replay
from blueprint_pipeline.deterministic_refusal import unchanged
from tests.test_atlas_mesh_registration import exported_glb, grid, splat_root


def _tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _registration(tmp_path, *, x0):
    (tmp_path / "in/splat").mkdir(parents=True)
    splat_root(tmp_path / "in/splat", grid(-2.0, extent=0.1)[0])
    exported_glb(tmp_path / "in/mesh.glb", grid(-2.0, extent=0.1, x0=x0), sign=(1.0, 1.0, 1.0))
    return {"splat_root": tmp_path / "in/splat", "mesh": tmp_path / "in/mesh.glb"}


def test_a_mesh_in_no_declared_frame_continues_on_the_named_estimate_and_its_inputs_are_only_read(tmp_path):
    inputs = _registration(tmp_path, x0=0.16)
    before = _tree(tmp_path / "in")
    report = stage_replay.replay(stage="mesh_registration", inputs=inputs, replay_root=tmp_path / "replays")
    assert report["first_stop"] == {"kind": "estimate", "stage": "mesh_registration",
                                    "code": "scene_estimate:collider_registration"}
    assert report["evidence"]["unchanged"] and _tree(tmp_path / "in") == before
    assert report["network_attempts"] == [] and report["new_paid_calls"] == 0
    assert (tmp_path / "replays").is_dir() and not (tmp_path / "in/registered").exists()


def test_a_mesh_beside_the_splat_in_every_frame_is_a_deterministic_refusal_with_its_key(tmp_path):
    report = stage_replay.replay(stage="mesh_registration", inputs=_registration(tmp_path, x0=5.0),
                                 replay_root=tmp_path / "replays")
    stop = report["first_stop"]
    assert (stop["kind"], stop["code"]) == ("refusal", "atlas_mesh_not_registered_to_splat")
    assert stop["deterministic"]["step"] == "atlas_mesh_registration" and unchanged(stop["deterministic"])


def test_a_stage_that_reaches_for_the_network_is_stopped_and_reported(tmp_path, monkeypatch):
    def calls_out(_paths, _output_root):
        socket.create_connection(("provider.invalid", 443), timeout=1)
        raise AssertionError("the connection was made")

    monkeypatch.setitem(stage_replay._RUNNERS, "mesh_registration", calls_out)
    report = stage_replay.replay(stage="mesh_registration", inputs=_registration(tmp_path, x0=0.16),
                                 replay_root=tmp_path / "replays")
    assert report["first_stop"]["kind"] == "error"
    assert report["first_stop"]["code"].startswith("unstubbed_network_call:")
    assert report["network_attempts"] and socket.getaddrinfo("localhost", 80)
    assert "connect" not in vars(socket.socket) and "connect_ex" not in vars(socket.socket)  # nothing outlives it
