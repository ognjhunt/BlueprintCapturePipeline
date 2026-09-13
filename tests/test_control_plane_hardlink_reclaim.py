"""The ext4 hard-link reclaim tool never replaces bytes whose identity moved after hashing."""
import importlib.util
import os
from pathlib import Path


def _load(name):
    script = Path(__file__).parents[1] / "scripts/control_plane_hardlink_reclaim.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hardlink_reclaim_must_not_replace_bytes_if_keeper_changes_after_hash(tmp_path, monkeypatch):
    dedup = _load("control_plane_hardlink_reclaim_race")
    keeper, victim = tmp_path / "a-keeper", tmp_path / "b-victim"
    keeper.write_bytes(b"original")
    victim.write_bytes(b"original")
    live = [(str(p), p.stat()) for p in [keeper, victim]]
    original = dedup.digest

    def mutate_after_hash(path):
        digest = original(path)
        if path == str(victim):
            keeper.write_bytes(b"modified")  # a concurrent writer after verification
        return digest

    monkeypatch.setattr(dedup, "digest", mutate_after_hash)
    skipped = []
    dedup.dedup_partition(live, True, skipped, minimum_age_seconds=0)
    assert victim.read_bytes() == b"original"
    assert any("changed during verification" in row for row in skipped), skipped
    assert not os.path.samefile(keeper, victim)


def test_hardlink_reclaim_skips_files_inside_the_quiescence_window_and_links_quiet_ones(tmp_path):
    dedup = _load("control_plane_hardlink_reclaim_quiet")
    keeper, victim = tmp_path / "a-keeper", tmp_path / "b-victim"
    keeper.write_bytes(b"same bytes")
    victim.write_bytes(b"same bytes")
    live = [(str(p), p.stat()) for p in [keeper, victim]]
    skipped = []
    assert dedup.dedup_partition(list(live), True, skipped) == (0, 0)  # default window: just-written files wait
    assert not os.path.samefile(keeper, victim) and len(skipped) == 2
    freed, links = dedup.dedup_partition(list(live), True, [], minimum_age_seconds=0)
    assert links == 1 and freed == len(b"same bytes") and os.path.samefile(keeper, victim)
    assert victim.read_bytes() == b"same bytes"
