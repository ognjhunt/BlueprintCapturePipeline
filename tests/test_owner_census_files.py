"""Hermetic no-follow descriptor ownership and finite protected metadata IO."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
import os

import pytest

from blueprint_pipeline import control_plane_lane_owner_consents as c
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def files(**kw):
    return c._Files(ReferenceCollectionBudget(monotonic=lambda: 0), **kw)


def test_bounded_read_rechecks_named_inode_and_closes_every_owned_descriptor(tmp_path):
    p = tmp_path / "small.json"
    p.write_bytes(b"{}")
    f = files()
    raw, value = f.read(p, cap=16)
    assert raw == b"{}" and value.info.st_size == 2
    f.verify()
    owned = list(f.owned)
    f.finish()
    for fd in owned:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("kind", ["file", "ancestor"])
def test_symlink_components_never_open(tmp_path, kind):
    p = tmp_path / "real"
    p.mkdir()
    (p / "x").write_bytes(b"{}")
    target = tmp_path / "alias"
    target.symlink_to(p if kind == "ancestor" else p / "x")
    f = files()
    try:
        with pytest.raises(c.OwnerCensusConsentError):
            f.read(target / "x" if kind == "ancestor" else target, cap=16)
    finally:
        f.finish()


def test_raw_cap_is_checked_before_each_read_including_sentinel(tmp_path, monkeypatch):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    f = files(raw_cap=2)
    calls = []
    old = os.read

    def read(fd, size):
        calls.append(size)
        assert size <= 2
        return old(fd, size)

    monkeypatch.setattr(os, "read", read)
    try:
        with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_resource_exhausted"):
            f.read(p, cap=8)
        assert calls == [2]
    finally:
        f.finish()


def test_named_replacement_refuses_final_verification(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"{}")
    f = files()
    try:
        f.read(p, cap=8)
        p.unlink()
        p.write_bytes(b"{}")
        with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_record_changed"):
            f.verify()
    finally:
        f.finish()


def test_known_close_failure_retains_ownership_for_safe_retry(tmp_path, monkeypatch):
    p = tmp_path / "x"
    p.write_bytes(b"{}")
    f = files()
    f.read(p, cap=8)
    owned = set(f.owned)
    old = os.close
    failed = []

    def close(fd):
        if not failed:
            failed.append(fd)
            raise OSError("injected")
        old(fd)

    monkeypatch.setattr(os, "close", close)
    f.finish()
    assert not f.owned
    for fd in owned:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_unknown_initial_descriptor_identity_never_blindly_closes(monkeypatch):
    f = files()
    closed = []
    monkeypatch.setattr(os, "fstat", lambda fd: (_ for _ in ()).throw(OSError("initial")))
    monkeypatch.setattr(os, "close", closed.append)
    with pytest.raises(
        c.OwnerCensusConsentError, match="owner_consent_descriptor_ownership_unproven"
    ):
        f.adopt(777)
    with pytest.raises(
        c.OwnerCensusConsentError, match="owner_consent_descriptor_ownership_unproven"
    ):
        f.finish()
    assert closed == [] and f.unresolved == 1


def test_regular_private_file_requires_root_mode_and_single_link(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"{}")
    p.chmod(0o644)
    f = files()
    try:
        with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_store_unsafe"):
            f.read(p, cap=8, protected=True, mode=0o600)
    finally:
        f.finish()


def test_budget_enabled_existing_reader_charges_once_and_none_keeps_old_bytes(tmp_path):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as retained

    path = tmp_path / "input"
    path.write_bytes(b"{}")
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    assert retained.read_census_input(path, _work_budget=budget) == b"{}"
    assert budget.counts["raw_bytes"] == 2
    assert retained.read_census_input(path) == b"{}"


def test_existing_reader_rejects_wrong_registry_before_open(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as retained

    a = ReferenceCollectionBudget(monotonic=lambda: 0)
    b = ReferenceCollectionBudget(monotonic=lambda: 0)
    f = c._Files(a)
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("opened wrong registry"))
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_options_invalid"):
        retained.read_census_input(tmp_path / "x", _work_budget=b, _descriptors=f)


def test_shared_reader_retains_ancestors_until_invocation_finalcheck(tmp_path):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as retained

    path = tmp_path / "input"
    path.write_bytes(b"{}")
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    f = c._Files(budget)
    try:
        assert retained.read_census_input(path, _work_budget=budget, _descriptors=f) == b"{}"
        path.unlink()
        path.write_bytes(b"{}")
        with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_record_changed"):
            f.verify()
    finally:
        f.finish()


def test_descriptor_cap_refuses_before_open(monkeypatch):
    f = files()
    f.owned = {i: (1, i) for i in range(c.MAX_DESCRIPTOR_COUNT)}
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("over-cap open"))
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_resource_exhausted"):
        f.open("/", os.O_RDONLY)


def test_reused_numeric_descriptor_is_not_closed_but_others_finish(tmp_path, monkeypatch):
    p = tmp_path / "x"
    p.write_bytes(b"{}")
    f = files()
    f.read(p, cap=8)
    fd = list(f.owned)[0]
    old = os.fstat
    closed = []
    oldclose = os.close

    def identity(value):
        info = old(value)
        if value == fd:
            return type("Info", (), {"st_dev": info.st_dev, "st_ino": info.st_ino + 1})()
        return info

    def close(value):
        closed.append(value)
        oldclose(value)

    monkeypatch.setattr(os, "fstat", identity)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(
        c.OwnerCensusConsentError, match="owner_consent_descriptor_ownership_unproven"
    ):
        f.finish()
    assert fd not in closed and not f.owned
    # Restore the actual still-owned test descriptor: the production registry
    # correctly cannot infer that an injected foreign token belongs to this test.
    oldclose(fd)


def test_protected_ancestor_same_inode_permission_drift_refuses(tmp_path, monkeypatch):
    import stat

    path = tmp_path / "private" / "x"
    path.parent.mkdir(mode=0o700)
    path.write_bytes(b"{}")
    path.chmod(0o600)
    original = c._protected

    def simulated_root(info, **kwargs):
        # UID/GID are supplied by a hermetic root fixture; mode/type are real.
        from types import SimpleNamespace

        data = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        data.update(st_uid=0, st_gid=0)
        if stat.S_ISDIR(info.st_mode) and info.st_ino != path.parent.stat().st_ino:
            data["st_mode"] = stat.S_IFDIR | 0o755
        original(SimpleNamespace(**data), **kwargs)

    monkeypatch.setattr(c, "_protected", simulated_root)
    f = files()
    try:
        f.read(path, cap=8, protected=True, mode=0o600)
        path.parent.chmod(0o777)
        with pytest.raises(c.OwnerCensusConsentError):
            f.verify()
    finally:
        f.finish()


def test_registry_rejects_invalid_budget_before_io(monkeypatch):
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("invalid budget open"))
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_options_invalid"):
        c._Files(object())


def test_initial_root_named_security_is_rechecked(tmp_path, monkeypatch):
    from types import SimpleNamespace

    p = tmp_path / "x"
    p.write_bytes(b"{}")
    f = files()
    f.read(p, cap=8)
    old = os.stat

    def changed(path, *args, **kw):
        info = old(path, *args, **kw)
        if path == "/":
            data = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            data["st_ino"] += 1
            return SimpleNamespace(**data)
        return info

    monkeypatch.setattr(os, "stat", changed)
    try:
        with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_record_changed"):
            f.verify()
    finally:
        f.finish()


def test_root_owned_blueprint_group_ancestor_is_safe_but_private_store_requires_root_group():
    import stat
    from types import SimpleNamespace

    ancestor = SimpleNamespace(st_uid=0, st_gid=42, st_mode=stat.S_IFDIR | 0o2750, st_nlink=2)
    c._protected(ancestor, directory=True)
    with pytest.raises(c.OwnerCensusConsentError):
        c._protected(ancestor, directory=True, mode=0o700)
    c._protected(SimpleNamespace(st_uid=0, st_gid=42, st_mode=stat.S_IFREG | 0o640, st_nlink=1))
