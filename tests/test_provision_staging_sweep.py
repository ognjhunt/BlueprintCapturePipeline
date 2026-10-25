from __future__ import annotations

import os
from pathlib import Path

from blueprint_pipeline import provision_staging_sweep as sweep
from blueprint_pipeline.provision_staging_sweep import STALE_STAGING_SECONDS, sweep_stale_staging


def test_staging_a_killed_provision_left_behind_is_swept_before_the_next_one(tmp_path):
    # 2026-10-23: three stopped deploys each left ~11 GB of staging and filled the root disk.
    now = 1_800_000_000.0
    old = now - STALE_STAGING_SECONDS - 60
    stale = [tmp_path / "scene-configuration-components-dead1", tmp_path / "blueprint-artisan-component-dead2"]
    fresh = tmp_path / "scene-configuration-components-live"
    other = tmp_path / "unrelated-tool-cache"
    for path in [*stale, fresh, other]:
        (path / "nested").mkdir(parents=True)
        (path / "nested" / "weights.bin").write_bytes(b"x")
        (path / "nested").chmod(0o500)  # read-only staging still goes
    for path in [*stale, other]:
        os.utime(path, (old, old))
    os.utime(fresh, (now - 60, now - 60))
    assert sweep_stale_staging(tmp_path, now=now) == sorted(path.name for path in stale)
    assert not any(path.exists() for path in stale)
    assert fresh.is_dir() and other.is_dir()  # a running provision and other tools are untouched
    for path in (fresh, other):
        (path / "nested").chmod(0o700)


def test_half_published_runtime_trees_of_dead_runs_are_swept(tmp_path):
    # 2026-10-23: three stopped publications left .<commit>.<random> folders (7.5 GB) that
    # blocked the release retention plan.
    now = 1_800_000_000.0
    old = now - STALE_STAGING_SECONDS - 60
    runtimes = tmp_path / "system-runtimes"
    stale = [runtimes / "scene-configuration" / f".{'a' * 40}.8ouxnrg0",
             runtimes / "splat-render" / f".{'b' * 40}.4x0u_xhk"]
    fresh = runtimes / "scene-configuration" / f".{'c' * 40}.yx03wuck"
    published = runtimes / "scene-configuration" / ("d" * 40)
    receipt = runtimes / "scene-configuration" / f"{'d' * 40}.publication.v1.json"
    for path in [*stale, fresh, published]:
        (path / "components").mkdir(parents=True)
        (path / "components" / "file").write_bytes(b"x")
        (path / "components").chmod(0o555)
    receipt.write_text("{}")
    for path in [*stale, published, receipt]:
        os.utime(path, (old, old))
    os.utime(fresh, (now - 60, now - 60))
    removed = sweep_stale_staging(tmp_path / "tmp", runtime_root=runtimes, now=now)
    assert removed == [f"scene-configuration/{stale[0].name}", f"splat-render/{stale[1].name}"]
    assert not any(path.exists() for path in stale)
    assert fresh.is_dir() and published.is_dir() and receipt.is_file()
    for path in (fresh, published):
        (path / "components").chmod(0o700)


def test_an_entry_that_vanishes_mid_sweep_is_skipped_and_others_are_never_inspected(tmp_path, monkeypatch):
    # Review of #2821: another tool's temporary file removed between listing and
    # lstat aborted the whole provision.
    now = 1_800_000_000.0
    old = now - STALE_STAGING_SECONDS - 60
    stale = tmp_path / "scene-configuration-components-dead"
    vanished = tmp_path / "scene-configuration-components-vanished"
    unrelated = tmp_path / "tmpq2x8unrelated"
    for path in (stale, vanished):
        path.mkdir()
        os.utime(path, (old, old))
    unrelated.write_bytes(b"x")
    real_lstat = Path.lstat

    def lstat(self, *args, **kwargs):
        if self in (vanished, unrelated):  # removed by its owner after the listing
            raise FileNotFoundError(self)
        return real_lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    assert sweep.sweep_stale_staging(tmp_path, now=now) == [stale.name]
    assert not stale.exists() and vanished.is_dir() and unrelated.is_file()

