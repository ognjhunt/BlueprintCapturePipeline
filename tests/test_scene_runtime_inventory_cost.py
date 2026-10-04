# Covers (for impacted-test selection):
#   scripts/install_scene_retirement_runtime.py
"""SDK inventory verifies every byte with linear aggregate-size accounting."""
import hashlib
from pathlib import Path
import time

import pytest

from tests.test_scene_retirement_runtime_installation import fixture, tmp_path  # noqa: F401


class CostedRows(dict):
    visits = 0

    def values(self):
        for row in super().values():
            self.visits += 1
            yield row


def test_nested_sdk_byte_accounting_does_not_rescan_prior_files(tmp_path, monkeypatch):  # noqa: F811
    module, _, dependencies = fixture(tmp_path, monkeypatch)
    for branch in ('first', 'second'):
        directory = dependencies / branch
        directory.mkdir()
        for index in range(16):
            (directory / f'{index}.dat').write_bytes(b'original bytes')
    rows = CostedRows({'earlier/source.py': {'size': 7}})
    module._tree(dependencies, Path('dependencies'), rows, {}, time.monotonic() + 30)
    assert rows.visits <= 2 * len(rows), 'aggregate inventory work must stay linear'
    assert len(rows) == 34
    expected = hashlib.sha256(b'original bytes').hexdigest()
    assert all(row['sha256'] == expected and row['size'] == 14
               for name, row in rows.items() if name.endswith('.dat'))


def test_sdk_byte_cap_includes_prior_roots_and_nested_members(tmp_path, monkeypatch):  # noqa: F811
    module, _, dependencies = fixture(tmp_path, monkeypatch)
    (dependencies / 'nested').mkdir()
    (dependencies / 'nested/payload').write_bytes(b'owned bytes')
    rows = {'earlier/source.py': {'size': 9}}
    monkeypatch.setattr(module, '_MAX_BYTES', 20)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        module._tree(dependencies, Path('dependencies'), rows, {}, time.monotonic() + 30)
