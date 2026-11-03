"""Every reader of a website room collider reads a USDA text one once per process, into one binary copy.

A collider normalized before 2026-11-03 is USDA text (capture-447's: 292 MB). OpenUSD parses it again at
each open that finds no stage of it alive, at about 3.6 GB a parse for a 280 MB room; the website
preparation's verdict and fit table parsed it three times, and a stance search or G1 preparation outside
the team packet once per search. Each now reads it once, its outputs byte for byte what the text gave.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import usd_binary_read_cache as cache  # noqa: E402
from tests.test_beta_policy_screening_g1_prepare import prepared_scene  # noqa: E402, F401 - a fixture

NEVER = 1 << 62


def collider_reads(monkeypatch: Any, path: Path) -> dict[str, list[str]]:
    """The callers of every read of text collider ``path``'s bytes (any file of its name holding them), as it happens.

    ``openusd``: an ``Usd.Stage.Open`` that finds no layer of it alive, or an anonymous open of it, each a
    whole-text parse. ``numpy``: the binary copy's own read (``usd_binary_read_cache._fast_layer``).
    """
    from pxr import Sdf, Usd

    size, digest = path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()

    def same(value: Any) -> bool:
        try:
            candidate = Path(os.fspath(value))
            # A packet's evidence copy is another file; a fixture's appearance may hold the same bytes.
            return (candidate.name == path.name and candidate.stat().st_size == size
                    and hashlib.sha256(candidate.read_bytes()).hexdigest() == digest)
        except (OSError, TypeError):
            return False

    reads: dict[str, list[str]] = {"openusd": [], "numpy": []}
    stage_open, anonymous, fast = Usd.Stage.Open, Sdf.Layer.OpenAsAnonymous, cache._fast_layer

    def opening(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], str) and same(args[0]) and Sdf.Layer.Find(args[0]) is None:
            reads["openusd"].append(sys._getframe(1).f_code.co_name)
        return stage_open(*args, **kwargs)

    def opening_anonymous(source: Any, *args: Any, **kwargs: Any) -> Any:
        if same(source):
            reads["openusd"].append(sys._getframe(1).f_code.co_name)
        return anonymous(source, *args, **kwargs)

    def reading(source: Any, *args: Any, **kwargs: Any) -> Any:
        if same(source):
            reads["numpy"].append(sys._getframe(1).f_code.co_name)
        return fast(source, *args, **kwargs)

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(opening))
    monkeypatch.setattr(Sdf.Layer, "OpenAsAnonymous", staticmethod(opening_anonymous))
    monkeypatch.setattr(cache, "_fast_layer", reading)
    return reads


@pytest.fixture
def copies_small_rooms(tmp_path, monkeypatch):
    """Fixture rooms are small: copy any text collider, read its arrays with numpy, in a temporary dir we see."""
    monkeypatch.setattr(cache, "LAZY_COPY_BYTES", 1)
    monkeypatch.setattr(cache, "LARGE_ARRAY_BYTES", 1)
    temporary = tmp_path / "system-tmp"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    yield temporary
    assert cache._SCOPES == [] and cache._COPIES == {}
    assert list(temporary.iterdir()) == []  # every copy and its directory are gone


def test_the_capability_verdict_and_fit_table_read_a_text_room_once(tmp_path, monkeypatch, copies_small_rooms):
    from blueprint_pipeline import website_scene_handoff as handoff
    from tests.test_website_rigid_feasibility import _room
    from tests.test_website_robot_fit import _job

    (tmp_path / "room").mkdir()
    room = _room(tmp_path / "room")
    # A carry that clears: the verdict's stance and carry searches, then the table's own station search.
    job = _job(destination={"relation": "on", "position_world_m": [0.3, 0.35, 0.0]},
               subject=([0.2675, -0.0325, 0.0], [0.3325, 0.0325, 0.065]))

    def run(name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
        native = tmp_path / name
        native.mkdir()
        verdict, fit = handoff._rigid_verdict_and_fit(preparation=job, collision_path=room, native_root=native)
        assert fit.pop("path") == str(native / "robot_fit_table.json")
        return verdict, fit, {path.name: path.read_bytes() for path in sorted(native.iterdir())}

    with monkeypatch.context() as text_only:
        text_only.setattr(cache, "LAZY_COPY_BYTES", NEVER)
        parsed = collider_reads(text_only, room)
        from_text = run("text")
    # Each search opened the file afresh: three whole-text parses of a 292 MB room, 3.6 GB each.
    assert parsed["openusd"] == ["ground_stance", "collision_kept_loaded", "ground_stance"]
    reads = collider_reads(monkeypatch, room)
    from_copy = run("copy")
    assert reads == {"openusd": [], "numpy": ["binary_copy"]}
    verdict, fit, files = from_copy
    assert verdict["status"] == "clear" and fit["counts"]
    assert json.loads(files["robot_fit_table.json"])["inputs"]["stations"]["source"] == "ground_stance_search"
    # Byte for byte what the text gave, the verdict naming the published file by its digest.
    assert from_copy == from_text and set(files) == {"capability_verdict.json", "robot_fit_table.json"}
    assert verdict["collision_sha256"] == "sha256:" + hashlib.sha256(room.read_bytes()).hexdigest()


def test_a_stance_search_on_its_own_reads_a_text_room_once(tmp_path, monkeypatch, copies_small_rooms):
    # The G1 screening preparation and a quick team launch each search once, with no compile around them.
    from blueprint_pipeline.team_embodiment_stance import ground_stance
    from tests.test_team_embodiment_stance_room_mesh import _search, write_scanned_room

    room = write_scanned_room(tmp_path / "collision.usda")
    with monkeypatch.context() as text_only:
        text_only.setattr(cache, "LAZY_COPY_BYTES", NEVER)
        parsed = collider_reads(text_only, room)
        from_text = _search(room, ranked=24)
    assert parsed["openusd"] == ["ground_stance"]
    reads = collider_reads(monkeypatch, room)
    assert _search(room, ranked=24) == from_text
    assert reads == {"openusd": [], "numpy": ["binary_copy"]}
    assert ground_stance.__wrapped__  # its own read scope, or its caller's


def test_the_g1_preparation_reads_the_base_packets_text_room_once(tmp_path, monkeypatch, prepared_scene,  # noqa: F811
                                                                  copies_small_rooms):
    from blueprint_pipeline import beta_policy_screening_g1_prepare as prepare

    args, _, _ = prepared_scene
    base_root = Path(args["base_packet_record"]["path"]).parent
    [collision] = [path for path in (base_root / "assets").iterdir() if path.name.startswith("collision")]
    assert collision.read_bytes()[:5] == b"#usda"

    def prepared() -> dict[str, str]:
        # Into the same root each time (its paths are recorded), removed after.
        import shutil

        result = prepare.prepare_native_g1_beta_packet(**args)
        root = Path(result["scene_packet"]["path"]).parent
        files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(root.rglob("*")) if path.is_file()}
        shutil.rmtree(Path(args["output_root"]) / "native-g1")
        return files

    with monkeypatch.context() as text_only:
        text_only.setattr(cache, "LAZY_COPY_BYTES", NEVER)
        parsed = collider_reads(text_only, collision)
        from_text = prepared()
    assert parsed["openusd"][0] == "ground_stance" and len(parsed["openusd"]) > 2
    reads = collider_reads(monkeypatch, collision)
    from_copy = prepared()
    # The stance search reads the base packet's file; the packet's evidence and staged copies of it are other
    # files holding the same bytes, which open the same copy. OpenUSD parses none of them.
    assert reads == {"openusd": [], "numpy": ["binary_copy"]}
    assert from_copy == from_text and "native_task_arena_packet_receipt.v1.json" in from_copy
