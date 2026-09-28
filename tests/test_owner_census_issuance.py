"""Protected fixture issuance and immutable publication; never real host paths."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_lane_owner_consents as c
from tests.test_owner_census_consent_contract import policy
from tests.test_owner_census_budget import payloads
from tests.test_lane_scratch_decisions import _json


@pytest.fixture
def issuer(tmp_path, monkeypatch):
    census, annotations = payloads()
    paths = [tmp_path / "census.json", tmp_path / "annotations.json", tmp_path / "policy.json"]
    for p, raw in zip(paths, (census, annotations, _json(policy()))):
        p.write_bytes(raw)
        p.chmod(0o600)
    store = tmp_path / "store"
    store.mkdir(mode=0o700)
    (store / ".owner-consents.lock").write_bytes(b"")
    (store / ".owner-consents.lock").chmod(0o600)
    cfg = SimpleNamespace(
        owner_census_decisions_enabled=1,
        lane_owner_policy_file=str(paths[2]),
        owner_consent_store=str(store),
        lane_scratch_work_root="/work/lanes",
        lane_scratch_inputs_root="/inputs/lanes",
    )
    monkeypatch.setattr(c, "_installed_config", lambda files, path: cfg)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    actual = c._protected

    def fixture_protection(info, *, directory=False, mode=None):
        # Local fixtures lack root ownership. Preserve every other leaf check;
        # actual root/ancestor and bridge validation has independent tests.
        if directory:
            c._require(info.st_mode & 0o170000 == 0o040000, "owner_consent_store_unsafe")
            if mode is not None:
                c._require(info.st_mode & 0o777 == mode, "owner_consent_store_unsafe")
        else:
            c._require(
                info.st_mode & 0o170000 == 0o100000 and info.st_nlink == 1,
                "owner_consent_store_unsafe",
            )
            if mode is not None:
                c._require(info.st_mode & 0o777 == mode, "owner_consent_store_unsafe")

    monkeypatch.setattr(c, "_protected", fixture_protection)
    args = dict(
        census_sha256="sha256:" + hashlib.sha256(census).hexdigest(),
        census_size_bytes=len(census),
        annotations_sha256="sha256:" + hashlib.sha256(annotations).hexdigest(),
        annotations_size_bytes=len(annotations),
        principal="operator",
        selected_paths=["/work/sample"],
        expires_at_epoch=1050,
        installed_config_path=tmp_path / "door.json",
        now=1000,
        monotonic=lambda: 0,
    )
    return paths, store, args, actual


def test_issuance_only_publishes_private_immutable_metadata(issuer):
    paths, store, args, _ = issuer
    result = c.issue_owner_consent(paths[0], paths[1], **args)
    assert result["consent_metadata_published"] is True and result["mutations"] == 0
    p = store / (result["consent_id"] + ".json")
    assert p.stat().st_mode & 0o777 == 0o600 and p.stat().st_nlink == 1
    raw = p.read_bytes()
    assert result["expected_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert result["expected_size_bytes"] == len(raw)
    assert json.loads(raw)["execution_authorized"] is False
    assert sorted(x.name for x in store.iterdir()) == [".owner-consents.lock", p.name]


def test_root_required_before_config_or_input_read(issuer, monkeypatch):
    paths, _, args, _ = issuer
    monkeypatch.setattr(os, "geteuid", lambda: 501)
    monkeypatch.setattr(c, "_installed_config", lambda *a: pytest.fail("nonroot config"))
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_issuer_required"):
        c.issue_owner_consent(paths[0], paths[1], **args)


@pytest.mark.parametrize("entry", ["unrecognized", "staging.tmp", "a" * 32 + ".json"])
def test_unknown_or_linked_store_entries_refuse_before_temp_creation(issuer, entry, monkeypatch):
    paths, store, args, _ = issuer
    dest = store / entry
    if entry.endswith(".json"):
        dest.symlink_to(paths[0])
    else:
        dest.write_bytes(b"")
    before = sorted(x.name for x in store.iterdir())
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_store_unsafe"):
        c.issue_owner_consent(paths[0], paths[1], **args)
    assert sorted(x.name for x in store.iterdir()) == before


def test_capacity_proof_precedes_temp_creation(issuer, monkeypatch):
    paths, store, args, _ = issuer
    monkeypatch.setattr(c, "MAX_STORE_RECORDS", 0)
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_store_full"):
        c.issue_owner_consent(paths[0], paths[1], **args)
    assert [x.name for x in store.iterdir()] == [".owner-consents.lock"]


def test_existing_consent_is_never_overwritten(issuer, monkeypatch):
    paths, store, args, _ = issuer
    monkeypatch.setattr(c.secrets, "token_hex", lambda n: "a" * (n * 2))
    first = c.issue_owner_consent(paths[0], paths[1], **args)
    raw = (store / (first["consent_id"] + ".json")).read_bytes()
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_publication_failed"):
        c.issue_owner_consent(paths[0], paths[1], **args)
    assert (store / (first["consent_id"] + ".json")).read_bytes() == raw


def test_protected_metadata_validation_rejects_nonroot_mode_and_hardlinks(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"{}")
    with pytest.raises(c.OwnerCensusConsentError):
        c._protected(p.stat(), mode=0o600)


def test_owned_temp_is_removed_after_write_failure(issuer, monkeypatch):
    paths, store, args, _ = issuer
    monkeypatch.setattr(os, "write", lambda *a: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_publication_failed"):
        c.issue_owner_consent(paths[0], paths[1], **args)
    assert [p.name for p in store.iterdir()] == [".owner-consents.lock"]


def test_foreign_replaced_temp_is_preserved_and_never_published(issuer, monkeypatch):
    paths, store, args, _ = issuer
    old = os.fsync
    replacement = []

    def sync(fd):
        old(fd)
        if not replacement:
            p = next(store.glob(".consent-*.tmp"))
            p.unlink()
            p.write_bytes(b"foreign")
            replacement.append(p)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_publication_failed"):
        c.issue_owner_consent(paths[0], paths[1], **args)
    assert replacement[0].read_bytes() == b"foreign" and not list(store.glob("*.json"))


def test_parent_security_change_during_publication_refuses_and_cleans_own_temp(issuer, monkeypatch):
    paths, store, args, _ = issuer
    old = os.fsync
    changed = []

    def sync(fd):
        old(fd)
        if not changed:
            store.chmod(0o777)
            changed.append(True)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_record_changed"):
        c.issue_owner_consent(paths[0], paths[1], **args)
    assert [p.name for p in store.iterdir()] == [".owner-consents.lock"]
