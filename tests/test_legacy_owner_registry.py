# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""Only a committed protected head can expose a historical owner label."""

import pytest


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.control_plane_lane_legacy_owner import LegacyOwnerStore

    root = tmp_path / "registry"
    root.mkdir(mode=0o700)
    (root / ".legacy-owner.lock").write_bytes(b"")
    (root / ".legacy-owner.lock").chmod(0o600)
    true_protected = owners._protected

    def fixture_protection(info, *, directory=False, mode=None):
        import stat
        if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
            raise owners.OwnerCensusConsentError("owner_consent_store_unsafe")
        if mode is not None and stat.S_IMODE(info.st_mode) != mode:
            raise owners.OwnerCensusConsentError("owner_consent_store_unsafe")

    monkeypatch.setattr(owners, "_protected", fixture_protection)
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    files = owners._Files(budget)
    store = LegacyOwnerStore(files, root)
    yield store, root
    files.finish()
    budget.close()
    monkeypatch.setattr(owners, "_protected", true_protected)


def test_registry_commits_only_after_full_receipt_and_head(registry):
    from blueprint_pipeline.control_plane_lane_legacy_owner import LegacyOwnerError

    store, root = registry
    packet_id = "a" * 32
    path = "/work/lanes/diagnostics/old-1"
    packet = {"schema_version": "packet", "path": path, "digest": "sha256:" + "a" * 64}
    approval = {"schema_version": "approval", "packet_digest": packet["digest"]}
    registration = {"schema_version": "registration", "path": path, "owner": "owner"}
    store.publish(packet_id, "packet", packet)
    store.publish(packet_id, "approval", approval)
    store.publish(packet_id, "registration", registration)
    assert store.committed_heads() == []
    store.publish(packet_id, "receipt", {"registration": registration})
    assert store.committed_heads() == []
    store.publish_head(path, packet_id, registration)
    assert len(store.committed_heads()) == 1
    assert store.read(packet_id, "packet") == packet
    store.publish(packet_id, "packet", packet)  # exact retry
    with pytest.raises(LegacyOwnerError, match="legacy_owner_record_conflict"):
        store.publish(packet_id, "packet", packet | {"path": "/other"})
    assert not (root / ("b" * 32 + ".head.json")).exists()


def test_registry_rejects_foreign_link_and_unknown_entry(registry):
    from blueprint_pipeline.control_plane_lane_legacy_owner import LegacyOwnerError

    store, root = registry
    outside = root.parent / "outside"
    outside.write_bytes(b"foreign")
    (root / ("a" * 32 + ".packet.json")).symlink_to(outside)
    with pytest.raises(LegacyOwnerError):
        store.publish("a" * 32, "packet", {"x": 1})
    with pytest.raises(LegacyOwnerError):
        store.committed_heads()


def test_registry_reads_multiple_committed_heads_without_parent_fd_exhaustion(registry):
    from blueprint_pipeline.control_plane_lane_legacy_owner import LegacyOwnerStore
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    store, root = registry
    for index in range(6):
        packet_id = f"{index + 1:032x}"
        path = f"/work/lanes/diagnostics/old-{index}"
        registration = {"path": path, "owner": "reviewer"}
        receipt = {"registration": registration}
        head = {"schema_version": "control_plane_lane_legacy_owner_head.v1",
                "packet_id": packet_id, "path": path,
                "registration_digest": canonical_digest(registration),
                "gc_eligible": False, "references_clear": False, "mutations": 0}
        for name, record in ((f"{packet_id}.registration.json", registration),
                             (f"{packet_id}.receipt.json", receipt),
                             (LegacyOwnerStore._head_name(path, packet_id), head)):
            entry = root / name
            entry.write_bytes(LegacyOwnerStore._payload(record))
            entry.chmod(0o600)
    assert len(store.committed_heads()) == 6


def test_crash_between_no_replace_link_and_temp_unlink_recovers_owned_link(registry, monkeypatch):
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    store, root = registry
    monkeypatch.setattr(legacy, "_root_owned_publication", lambda info: True)
    packet_id = "b" * 32
    temporary = root / (".consent-" + "c" * 32 + ".tmp")
    final = root / (packet_id + ".packet.json")
    temporary.write_bytes(b'{"stage":"packet"}\n')
    temporary.chmod(0o600)
    import os
    os.link(temporary, final)
    assert final.stat().st_nlink == 2
    assert final.name in store._scan()
    store.recover_publication_links()
    assert not temporary.exists()
    assert final.stat().st_nlink == 1


def test_unfinished_hardlinked_head_is_not_committed_until_writer_recovery(registry, monkeypatch):
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    import os

    store, root = registry
    monkeypatch.setattr(legacy, "_root_owned_publication", lambda info: True)
    packet_id = "d" * 32
    path = "/work/lanes/diagnostics/old-1"
    registration = {"path": path, "owner": "owner"}
    store.publish(packet_id, "registration", registration)
    store.publish(packet_id, "receipt", {"registration": registration})
    head = dict(schema_version="control_plane_lane_legacy_owner_head.v1",
                packet_id=packet_id, path=path,
                registration_digest=canonical_digest(registration),
                gc_eligible=False, references_clear=False, mutations=0)
    temporary = root / (".consent-" + "e" * 32 + ".tmp")
    temporary.write_bytes(legacy.LegacyOwnerStore._payload(head))
    temporary.chmod(0o600)
    final = root / store._head_name(path, packet_id)
    os.link(temporary, final)
    with pytest.raises(legacy.LegacyOwnerError, match="legacy_owner_record_invalid"):
        store.committed_heads()
    store.recover_publication_links()
    assert len(store.committed_heads()) == 1
