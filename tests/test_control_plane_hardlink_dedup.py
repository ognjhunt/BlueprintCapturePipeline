"""The historical command delegates to safe extent sharing, never inode aliasing."""
import importlib.util
from pathlib import Path

from blueprint_pipeline import control_plane_file_dedup


def test_compatibility_command_keeps_quiescence_and_delegates_without_aliasing(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/control_plane_hardlink_dedup.py'
    spec = importlib.util.spec_from_file_location('dedup_command', path)
    command = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command)
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.write_bytes(b'same')
    right.write_bytes(b'same')
    live = [(str(p), p.stat()) for p in [left, right]]
    skipped = []
    assert command.dedup_partition(live, True, skipped) == (0, 0)
    assert len(skipped) == 2
    seen = []
    def shared(source, target, *, apply):
        seen.append((source, target, apply))
        return {'status': 'deduplicated', 'bytes_deduplicated': 4}
    monkeypatch.setattr(control_plane_file_dedup, 'deduplicate_pair', shared)
    assert command.dedup_partition(live, True, [], minimum_age_seconds=0) == (4, 1)
    assert seen == [(str(left), str(right), True)]
    assert left.stat().st_ino != right.stat().st_ino
