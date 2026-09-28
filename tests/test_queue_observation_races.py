"""Covers: descriptor anchored selected queue reads, churn and FD cleanup faults."""
import errno
import importlib
import os
import stat

import pytest


def _module():
    return importlib.import_module("blueprint_pipeline.control_plane_queue_observation")


def _observe(module, roots, states=("pending", "processing"), clock=lambda: 0):
    return module.observe_queue_states([module.QueueRootContract(str(root), states) for root in roots],
                                       observed_at_epoch=10, monotonic=clock)


def _row(root, name="a.json"):
    directory = root / "pending"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b'{"path":"/work/a"}')
    return path


@pytest.mark.parametrize("location", ["ancestor", "root", "state", "row"])
def test_symlink_at_every_selected_level_is_unknown(tmp_path, location):
    module = _module()
    root = tmp_path / "ancestor" / "queue"
    row = _row(root)
    target = {"ancestor": root.parent, "root": root, "state": row.parent, "row": row}[location]
    saved = target.with_name(target.name + "-actual")
    target.rename(saved)
    target.symlink_to(saved, target_is_directory=location != "row")
    result = _observe(module, [root])
    assert not result.complete and result.blockers and result.rows == ()


@pytest.mark.parametrize("kind", ["directory", "fifo", "temporary"])
def test_selected_special_and_unpublished_entries_never_read(tmp_path, kind):
    module = _module()
    (tmp_path / "pending").mkdir()
    target = tmp_path / "pending" / ("staged.tmp" if kind == "temporary" else "item.json")
    if kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b"{}")
    result = _observe(module, [tmp_path])
    assert not result.complete and result.rows == () and result.blockers


def test_move_pending_to_processing_during_observation_is_unknown(tmp_path, monkeypatch):
    module = _module()
    row = _row(tmp_path)
    (tmp_path / "processing").mkdir()
    real = module._Scan.read_row

    def read(self, *args):
        result = real(self, *args)
        row.rename(tmp_path / "processing" / row.name)
        return result

    monkeypatch.setattr(module._Scan, "read_row", read)
    result = _observe(module, [tmp_path])
    assert not result.complete and "queue_directory_changed" in result.blockers


@pytest.mark.parametrize("change", ["remove", "modify", "replace", "symlink"])
def test_row_changed_after_read_is_unknown_and_retains_positive(tmp_path, monkeypatch, change):
    module = _module()
    row = _row(tmp_path)
    real = module._Scan.read_row

    def read(self, *args):
        result = real(self, *args)
        if change == "modify":
            row.write_bytes(b'{"path":"/work/b"}')
        else:
            row.unlink()
            if change == "replace":
                row.write_bytes(b'{"path":"/work/b"}')
            elif change == "symlink":
                foreign = tmp_path / "foreign"
                foreign.write_bytes(b'{"path":"/secret"}')
                row.symlink_to(foreign)
        return result

    monkeypatch.setattr(module._Scan, "read_row", read)
    result = _observe(module, [tmp_path])
    assert not result.complete and len(result.rows) == 1
    assert result.rows[0].raw_text == '{"path":"/work/a"}'


@pytest.mark.parametrize("change", ["root", "state", "missing_state"])
def test_directory_replacement_or_missing_state_creation_is_unknown(tmp_path, monkeypatch, change):
    module = _module()
    root = tmp_path / "queue"
    row = _row(root)
    real = module._Scan.read_row

    def read(self, *args):
        result = real(self, *args)
        if change == "missing_state":
            (root / "processing").mkdir()
        else:
            target = root if change == "root" else row.parent
            target.rename(target.with_name(target.name + "-old"))
            target.mkdir()
            (target / "foreign.json").write_bytes(b'{"secret":1}')
        return result

    monkeypatch.setattr(module._Scan, "read_row", read)
    result = _observe(module, [root])
    assert not result.complete and result.blockers
    assert all("secret" not in row.raw_text for row in result.rows)


@pytest.mark.parametrize("change", ["row", "state", "root"])
def test_first_root_changed_during_later_root_requires_global_final_check(tmp_path, monkeypatch, change):
    module = _module()
    first, second = tmp_path / "a-root", tmp_path / "b-root"
    row = _row(first)
    _row(second)
    real = module._Scan.read_row

    def read(self, *args):
        result = real(self, *args)
        if args[0] == str(second):
            target = row if change == "row" else row.parent if change == "state" else first
            if change == "row":
                target.write_bytes(b'{"path":"/work/b"}')
            else:
                target.rename(target.with_name(target.name + "-old"))
                target.mkdir()
        return result

    monkeypatch.setattr(module._Scan, "read_row", read)
    result = _observe(module, [first, second])
    assert not result.complete and result.blockers
    assert len(result.roots) == 2 and len(result.rows) == 2


def test_missing_first_root_retains_scope_and_later_partial_positive(tmp_path):
    module = _module()
    missing, second = tmp_path / "a-missing", tmp_path / "b-root"
    _row(second)
    result = _observe(module, [missing, second])
    assert not result.complete and len(result.rows) == 1 and len(result.roots) == 2
    assert result.roots[0].root_path == str(missing)
    assert result.roots[0].root_identity is None and result.roots[0].missing_states == ()


def test_unreadable_later_row_keeps_earlier_positive(tmp_path, monkeypatch):
    module = _module()
    _row(tmp_path)
    _row(tmp_path, "b.json")
    real = module.os.open

    def opened(name, *args, **kwargs):
        if name == "b.json":
            raise PermissionError("private path must not appear")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", opened)
    result = _observe(module, [tmp_path])
    assert not result.complete and len(result.rows) == 1
    assert "private" not in str(result.blockers)


def test_all_owned_descriptors_close_on_post_open_clock_failure(tmp_path, monkeypatch):
    module = _module()
    opened = []
    real = module.os.open

    def tracked(*args, **kwargs):
        fd = real(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(module.os, "open", tracked)
    result = _observe(module, [tmp_path], clock=lambda: 6 if opened else 0)
    assert not result.complete and opened
    for fd in opened:
        with pytest.raises(OSError) as error:
            os.fstat(fd)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("fault", ["fstat", "clock"])
def test_acquisition_fault_combined_with_one_shot_close_failure_closes_owned_fd(tmp_path, monkeypatch, fault):
    module = _module()
    real_open, real_fstat, real_close = module.os.open, module.os.fstat, module.os.close
    opened = []
    stat_failed = close_failed = False

    def tracked(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fstat(fd):
        nonlocal stat_failed
        if fault == "fstat" and not stat_failed and fd in opened:
            stat_failed = True
            raise OSError(errno.EIO, "private")
        return real_fstat(fd)

    def close(fd):
        nonlocal close_failed
        if fd in opened and not close_failed:
            close_failed = True
            raise OSError(errno.EIO, "definite one shot")
        return real_close(fd)

    monkeypatch.setattr(module.os, "open", tracked)
    monkeypatch.setattr(module.os, "fstat", fstat)
    monkeypatch.setattr(module.os, "close", close)
    result = _observe(module, [tmp_path], clock=lambda: 6 if fault == "clock" and opened else 0)
    assert not result.complete and "queue_descriptor_close_failed" in result.blockers
    for fd in opened:
        with pytest.raises(OSError):
            real_fstat(fd)


def test_ambiguous_close_reusing_foreign_fd_never_closes_foreign(tmp_path, monkeypatch):
    module = _module()
    foreign_path = tmp_path / "foreign"
    foreign_path.write_bytes(b"foreign")
    real_open, real_close, real_fstat = module.os.open, module.os.close, module.os.fstat
    foreign = []
    opened = []

    def tracked(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def close(fd):
        if fd in opened and not foreign:
            real_close(fd)
            replacement = real_open(foreign_path, os.O_RDONLY)
            if replacement != fd:
                os.dup2(replacement, fd)
                real_close(replacement)
            foreign.append(fd)
            raise OSError(errno.EINTR, "ambiguous close")
        return real_close(fd)

    monkeypatch.setattr(module.os, "open", tracked)
    monkeypatch.setattr(module.os, "close", close)
    try:
        result = _observe(module, [tmp_path], clock=lambda: 6 if opened else 0)
        assert not result.complete and "queue_descriptor_changed" in result.blockers
        assert stat.S_ISREG(real_fstat(foreign[0]).st_mode)
    finally:
        for fd in foreign:
            real_close(fd)
