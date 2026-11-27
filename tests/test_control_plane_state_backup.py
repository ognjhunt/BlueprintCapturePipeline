"""The control-plane state backup: class (a) only, encrypted, bounded, restorable byte for byte, purged after a
withdrawal, and a restore that holds paid admission until the money records have caught up."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_restore_money_hold as hold
from blueprint_pipeline import control_plane_state_backup as backup
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.spend_admission_lock import build_spend_admission_lock

KEY = bytes(range(32))
CP = "var/lib/blueprint/pipeline-control-plane"
SITE = "var/lib/blueprint/pubsub-handoffs/bucket-a/scenes/site-req1"
VIDEO = b"SITE-VIDEO-BYTES-" * 64


class FakeStore:
    """An S3 bucket with object versions, in memory."""

    def __init__(self) -> None:
        self.objects: dict[str, list[tuple[str, bytes]]] = {}
        self.counter = 0
        self.unlistable = False

    def put(self, key: str, data: bytes) -> None:
        self.counter += 1
        self.objects.setdefault(key, []).append((f"v{self.counter}", bytes(data)))

    def get(self, key: str) -> bytes:
        return self.objects[key][-1][1]

    def versions(self, prefix: str) -> list[tuple[str, str]]:
        if self.unlistable:
            raise ConnectionError("bucket unreachable")
        return sorted((key, version) for key, rows in self.objects.items() if key.startswith(prefix)
                      for version, _ in rows)

    def delete(self, key: str, version: str) -> None:
        self.objects[key] = [row for row in self.objects[key] if row[0] != version]
        if not self.objects[key]:
            del self.objects[key]

    def generations(self) -> set[str]:
        return {key.split("/")[2] for key in self.objects}


def _write(root: Path, relative: str, data: bytes, mode: int = 0o640) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    return path


STATE = {
    f"{CP}/gpu_spend_guard/spend_ledger.json": b'{"spend": [1, 2, 3]}',
    f"{CP}/provider-locks/vast_paid_launch.lock": b"",
    f"{CP}/task-evaluation-launches/queue/job-1.json": b'{"job": "job-1", "status": "running"}',
    f"{CP}/beta-policy-screening/ledger.jsonl": b'{"hold": 12.5}\n',
    "etc/blueprint/pipeline-control-plane.env": b"BLUEPRINT_FLAG=1\n",
    "etc/blueprint/website-withdrawal-deletion-approval.json": b'{"approved_by": "owner"}',
    "mnt/blueprint-work/beta-policy-screening-work/attempt-1/receipt.json": b'{"settled": true}',
    f"{SITE}/website_withdrawal/tombstone.json": b'{"digest": "sha256:x"}',
}
SITE_DATA = {
    "var/lib/blueprint/capture-intakes/walkthrough-req1/video.mp4": VIDEO,
    f"{SITE}/captures/walkthrough-req1/raw/video.mp4": VIDEO,
    f"{SITE}/captures/walkthrough-req1/raw/meta.json": b'{"site": "data"}',
    f"{SITE}/website_withdrawal/quarantine/abc/video.mp4": VIDEO,
    f"{CP}/task-evaluation-launch-runs/run-1/receipt.json": b'{"run": 1}',
    "var/lib/blueprint/task-evaluation-inputs/team-scene-halves/site-req1/half.json": b'{"half": 1}',
}
NOT_BACKED_UP = {
    "etc/blueprint/provider-secrets/vast_api_key": b"vast-secret",
    "mnt/blueprint-work/beta-policy-screening-work/attempt-1/bundle.tar": b"bundle-bytes",
    f"{CP}/result-artifact-cache/blob.bin": b"cache",
}


@pytest.fixture()
def host(tmp_path: Path) -> Path:
    root = tmp_path / "host"
    for table in (STATE, SITE_DATA, NOT_BACKED_UP):
        for relative, data in table.items():
            _write(root, relative, data)
    (root / f"{CP}/gpu_spend_guard").chmod(0o750)
    return root


def _run(host: Path, store: FakeStore, tmp_path: Path, **kwargs):
    return backup.run_backup(fs_root=host, store=store, key=KEY, report_root=tmp_path / "report",
                             site_roots=[host / "var/lib/blueprint/pubsub-handoffs"], part_bytes=4096, **kwargs)


def test_a_restore_into_a_temporary_root_reproduces_the_backed_up_tree_byte_for_byte(host, tmp_path):
    store = FakeStore()
    report = _run(host, store, tmp_path)
    assert report["status"] == "complete", report["reason"]
    assert report["files"] >= len(STATE) and report["parts"] >= 1
    target = tmp_path / "drill"
    receipt = backup.restore(store=store, key=KEY, fs_root=target)
    assert receipt["files_restored_and_verified"] == report["files"]
    for relative, data in STATE.items():
        restored = target / relative
        assert restored.read_bytes() == data, relative
        assert restored.stat().st_mode & 0o777 == (host / relative).stat().st_mode & 0o777
        assert int(restored.stat().st_mtime) == int((host / relative).stat().st_mtime)
    assert (target / f"{CP}/gpu_spend_guard").stat().st_mode & 0o777 == 0o750
    # The owner's standing approval comes back as it was; nothing recreates it.
    assert (target / "etc/blueprint/website-withdrawal-deletion-approval.json").read_bytes() == STATE[
        "etc/blueprint/website-withdrawal-deletion-approval.json"]
    # A drill on a temporary root writes no money hold; a second restore of the same generation is a no-op.
    assert receipt["money_hold_marker"] is None
    assert backup.restore(store=store, key=KEY, fs_root=target)["already_restored"] is True
    # The directories above a restored journal come back as they were, not as the restore made them.
    site_dir = host / SITE
    site_dir.chmod(0o711)
    again = tmp_path / "drill-2"
    _run(host, store, tmp_path)
    backup.restore(store=store, key=KEY, fs_root=again)
    assert (again / SITE).stat().st_mode & 0o777 == 0o711
    assert not (again / SITE / "captures").exists()


def test_a_runs_reports_are_readable_like_the_ledger_and_the_stream_is_packed_fast(host, tmp_path):
    store = FakeStore()
    report = _run(host, store, tmp_path)
    assert report["status"] == "complete"
    # The first run on the host packed at tarfile's default, level 9, and ran out of its time bound. The
    # gzip header says how the stream was packed: its extra-flags byte is 4 for the fastest level, 2 for 9.
    first = backup._open(KEY, store.get(f"{backup.KEY_PREFIX}{report['generation']}/part-000000.bin"),
                         f"{report['generation']}:part:0")
    assert first[:2] == b"\x1f\x8b" and first[8] == 4 and backup.GZIP_LEVEL == 1
    # An operator reads a run through the door, which is not root: the reports are readable, as the ledger is.
    for name in ("latest.json", "enabled_units.json", "ledger.json"):
        assert (tmp_path / "report" / name).stat().st_mode & 0o777 == 0o644, name
    # The stream restores byte for byte.
    drill = tmp_path / "drill"
    backup.restore(store=store, key=KEY, fs_root=drill)
    for relative, data in STATE.items():
        assert (drill / relative).read_bytes() == data, relative


def test_site_data_is_never_in_a_backup_even_when_present_on_disk(host, tmp_path):
    store = FakeStore()
    report = _run(host, store, tmp_path)
    target = tmp_path / "drill"
    backup.restore(store=store, key=KEY, fs_root=target)
    restored = {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()}
    for relative in (*SITE_DATA, *NOT_BACKED_UP):
        assert relative not in restored, relative
    # Not even encrypted: no stored object carries the video's bytes, in clear or as a decrypted part.
    gen = report["generation"]
    seal = json.loads(store.get(f"{backup.KEY_PREFIX}{gen}/seal.json"))
    plain = b"".join(backup._open(KEY, store.get(part["key"]), f"{gen}:part:{number}")
                     for number, part in enumerate(seal["parts"]))
    import gzip
    assert VIDEO[:32] not in gzip.decompress(plain)
    assert all(VIDEO[:32] not in blob for rows in store.objects.values() for _, blob in rows)
    # Only the seal is plain, and it names no path.
    assert "site-req1" not in json.dumps(seal)
    excluded = report["roots"]["/mnt/blueprint-work/beta-policy-screening-work"]["excluded"]
    assert excluded == {"not_a_record": 1}
    skipped = {row["path"]: row["reason"] for row in report["skipped_roots"]}
    assert skipped["/var/lib/blueprint/capture-intakes"] == "site_data_never_backed_up"
    assert skipped["/var/lib/blueprint/pubsub-handoffs/*/scenes/*"] == "site_data_never_backed_up"
    assert skipped["/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs"] == "site_data_never_backed_up"
    assert skipped["/var/lib/blueprint/pipeline-control-plane/result-artifact-cache"] == "site_data_never_backed_up"
    assert skipped["/opt/blueprint/task-evaluation-control-plane-releases"] == "rebuildable_release"


def test_the_billing_reconcilers_evidence_trail_is_left_out_and_counted_and_the_ledger_beside_it_is_kept(host, tmp_path):
    guard = f"{CP}/gpu_spend_guard"
    for stamp in ("20261127T004422.843839Z", "20261127T005556.990412Z"):
        _write(host, f"{guard}/billing-audit/{stamp}/response-001-vast.json", b'{"invoices": []}' * 4096)
        _write(host, f"{guard}/billing-audit/{stamp}/provider_billing_source_receipt.json", b'{"receipt": 1}')
    _write(host, f"{guard}/metered-call-events/4fdd11dd/14f6a3f9.attempted.json", b'{"call": 1}')
    _write(host, f"{guard}/paid_spend_admission_lock.json", b'{"status": "open"}')
    store = FakeStore()
    report = _run(host, store, tmp_path)
    assert report["status"] == "complete"
    # Each of the two folders is left out whole, however much it holds, and the report says so.
    counts = report["roots"][f"/{guard}"]
    assert counts["excluded"] == {"excluded_by_allowlist": 2}
    drill = tmp_path / "drill"
    backup.restore(store=store, key=KEY, fs_root=drill)
    assert (drill / guard / "spend_ledger.json").read_bytes() == STATE[f"{guard}/spend_ledger.json"]
    assert (drill / guard / "paid_spend_admission_lock.json").read_bytes() == b'{"status": "open"}'
    assert not (drill / guard / "billing-audit").exists() and not (drill / guard / "metered-call-events").exists()
    root = next(item for item in backup.ALLOWLIST if item.path == f"/{guard}")
    assert root.exclude == ("billing-audit", "metered-call-events") and "not backed up" in root.reason


def test_the_sam_queues_records_are_kept_and_the_watchdogs_sweep_receipts_beside_them_are_left_out(host, tmp_path):
    queue = f"{CP}/sam31-preparation-executions"
    _write(host, f"{queue}/completed/sam31-e41bd19c.json", b'{"status": "completed"}')
    _write(host, f"{queue}/results/sam31-e41bd19c.json", b'{"result": 1}')
    for index in range(40):
        _write(host, f"{queue}/watchdog-cgroup-cleanup/dispatcher-cgroup-cleanup-{index}.json", b'{"swept": 0}')
    store = FakeStore()
    report = _run(host, store, tmp_path)
    assert report["status"] == "complete"
    counts = report["roots"][f"/{queue}"]
    # The whole folder is one exclusion, however many receipts it holds.
    assert counts["files"] == 2 and counts["excluded"] == {"excluded_by_allowlist": 1}
    drill = tmp_path / "drill"
    backup.restore(store=store, key=KEY, fs_root=drill)
    assert (drill / queue / "completed/sam31-e41bd19c.json").read_bytes() == b'{"status": "completed"}'
    assert not (drill / queue / "watchdog-cgroup-cleanup").exists()
    # Listed once, as a records-only queue.
    listed = [item for item in backup.ALLOWLIST if item.path == f"/{queue}"]
    assert len(listed) == 1 and listed[0].records_only and listed[0].exclude == ("watchdog-cgroup-cleanup",)


def test_no_allowlisted_root_is_site_data():
    for root in backup.ALLOWLIST:
        if "website_withdrawal" in root.path or root.path.endswith(".retired.v1.json"):
            continue
        assert not backup._site_data(root.path), root.path


def test_a_run_over_its_byte_bound_is_incomplete_and_leaves_nothing_behind(host, tmp_path):
    store = FakeStore()
    report = _run(host, store, tmp_path, max_bytes=100)
    assert report["status"] == "incomplete"
    assert report["reason"] == "control_plane_backup_byte_bound_reached"
    assert store.objects == {}
    ledger = backup.read_ledger(backup.ledger_path(tmp_path / "report"))
    assert [row["status"] for row in ledger["generations"]] == ["deleted"]
    with pytest.raises(backup.StateBackupError, match="no_sealed_generation"):
        backup.restore(store=store, key=KEY, fs_root=tmp_path / "drill")


def test_a_run_over_its_time_bound_is_incomplete(host, tmp_path):
    ticks = iter(range(0, 10_000, 100))
    report = _run(host, FakeStore(), tmp_path, max_seconds=50, clock=lambda: float(next(ticks)))
    assert report["status"] == "incomplete" and report["reason"] == "control_plane_backup_time_bound_reached"


def test_the_wrong_key_cannot_restore(host, tmp_path):
    store = FakeStore()
    _run(host, store, tmp_path)
    with pytest.raises(backup.StateBackupError, match="key_does_not_match"):
        backup.restore(store=store, key=bytes(32), fs_root=tmp_path / "drill")


def test_seven_days_are_kept_and_the_newest_generation_always(host, tmp_path):
    store = FakeStore()
    old = _run(host, store, tmp_path, now=time.time() - 9 * 86400)
    recent = _run(host, store, tmp_path, now=time.time() - 3 * 86400)
    assert recent["retention"]["generations"] == []  # nine days old is within seven of three days ago
    newest = _run(host, store, tmp_path)
    assert newest["retention"]["status"] == "deleted"
    assert newest["retention"]["generations"] == [old["generation"]]
    assert store.generations() == {recent["generation"], newest["generation"]}


def _owner_key(tmp_path: Path, name: str = "owner", *, bits: int = 3072) -> tuple[Path, Path]:
    """A test key pair: the private PEM (0600) and the recipient file a release would carry."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    private_path = tmp_path / f"{name}-private.pem"
    private_path.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                   serialization.NoEncryption()))
    private_path.chmod(0o600)
    public = private.public_key()
    recipient_path = tmp_path / f"{name}-recipient.json"
    recipient_path.write_text(json.dumps({
        "schema_version": backup.RECIPIENT_SCHEMA,
        "public_key_pem": public.public_bytes(serialization.Encoding.PEM,
                                              serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii"),
        "fingerprint": backup._public_fingerprint(public)}))
    return private_path, recipient_path


def test_a_backup_sealed_to_the_owners_public_key_restores_with_the_private_key_alone(host, tmp_path):
    private_path, recipient_path = _owner_key(tmp_path)
    recipient = backup.load_recipient(recipient_path)
    key, envelope = backup.new_generation_key(recipient)
    store = FakeStore()
    report = backup.run_backup(fs_root=host, store=store, key=key, key_envelope=envelope,
                               report_root=tmp_path / "report",
                               site_roots=[host / "var/lib/blueprint/pubsub-handoffs"], part_bytes=4096)
    assert report["status"] == "complete" and report["recipient_fingerprint"] == recipient["fingerprint"]
    seal = json.loads(store.get(f"{backup.KEY_PREFIX}{report['generation']}/seal.json"))
    assert seal["key_envelope"] == envelope and seal["digest"] == canonical_digest(seal, digest_field="digest")
    # Nothing in the bucket holds the generation's key in the clear, and the host kept none.
    assert all(key not in data for rows in store.objects.values() for _, data in rows)
    assert not list((tmp_path / "report").rglob("*key*"))
    drill = tmp_path / "drill"
    receipt = backup.restore(store=store, private_key_path=private_path, fs_root=drill)
    assert receipt["generation"] == report["generation"]
    for relative, data in STATE.items():
        assert (drill / relative).read_bytes() == data, relative
    # Each generation gets its own key.
    assert backup.new_generation_key(recipient)[0] != key


def test_only_the_private_key_the_generation_was_sealed_to_can_open_it(host, tmp_path):
    private_path, recipient_path = _owner_key(tmp_path)
    other_private, _ = _owner_key(tmp_path, "other")
    key, envelope = backup.new_generation_key(backup.load_recipient(recipient_path))
    store = FakeStore()
    backup.run_backup(fs_root=host, store=store, key=key, key_envelope=envelope, report_root=tmp_path / "report",
                      site_roots=[host / "var/lib/blueprint/pubsub-handoffs"], part_bytes=4096)
    with pytest.raises(backup.StateBackupError, match="private_key_does_not_match_generation"):
        backup.restore(store=store, private_key_path=other_private, fs_root=tmp_path / "drill")
    with pytest.raises(backup.StateBackupError, match="private_key_required"):
        backup.restore(store=store, fs_root=tmp_path / "drill")
    # The private key file must be the owner's alone, a regular file, and a key.
    private_path.chmod(0o640)
    with pytest.raises(backup.StateBackupError, match="private_key_file_unsafe"):
        backup.open_generation_key(envelope, private_path)
    private_path.chmod(0o600)
    assert backup.open_generation_key(envelope, private_path) == key
    link = tmp_path / "link.pem"
    link.symlink_to(private_path)
    with pytest.raises(backup.StateBackupError, match="private_key_file_unsafe"):
        backup.open_generation_key(envelope, link)
    with pytest.raises(backup.StateBackupError, match="private_key_unreadable"):
        backup.open_generation_key(envelope, tmp_path / "absent.pem")
    for broken in ({**envelope, "scheme": "none"}, {**envelope, "wrapped_key": "not base64!"},
                   {**envelope, "wrapped_key": envelope["wrapped_key"][:-8] + "AAAAAAA="}, None):
        with pytest.raises(backup.StateBackupError, match="key_envelope_invalid"):
            backup.open_generation_key(broken, private_path)
    assert not (tmp_path / "drill").exists()


def test_the_recipient_file_is_refused_unless_it_is_one_strong_key_with_its_own_fingerprint(tmp_path):
    _, recipient_path = _owner_key(tmp_path)
    good = json.loads(recipient_path.read_text())
    assert backup.load_recipient(recipient_path)["fingerprint"] == good["fingerprint"]
    _, other = _owner_key(tmp_path, "other")
    _, weak = _owner_key(tmp_path, "weak", bits=2048)
    cases = (({**good, "fingerprint": json.loads(other.read_text())["fingerprint"]}, "fingerprint_mismatch"),
             ({**good, "public_key_pem": json.loads(other.read_text())["public_key_pem"]}, "fingerprint_mismatch"),
             (json.loads(weak.read_text()), "key_unsupported"),
             ({**good, "schema_version": "v0"}, "recipient_invalid"),
             ({**good, "public_key_pem": "-----BEGIN PUBLIC KEY-----\nnot a key\n-----END PUBLIC KEY-----\n"},
              "recipient_invalid"))
    for value, reason in cases:
        recipient_path.write_text(json.dumps(value))
        with pytest.raises(backup.StateBackupError, match=reason):
            backup.load_recipient(recipient_path)
    recipient_path.write_text("{")
    with pytest.raises(backup.StateBackupError, match="recipient_unreadable"):
        backup.load_recipient(recipient_path)
    with pytest.raises(backup.StateBackupError, match="recipient_unreadable"):
        backup.load_recipient(tmp_path / "absent.json")
    link = tmp_path / "link.json"
    link.symlink_to(other)
    with pytest.raises(backup.StateBackupError, match="recipient_file_unsafe"):
        backup.load_recipient(link)


def test_the_release_carries_the_owners_public_key_and_no_private_one():
    path = backup.RELEASE_RECIPIENT_PATH
    assert path == Path(__file__).resolve().parents[1] / "config" / backup.RECIPIENT_FILENAME
    recorded = json.loads(path.read_text())
    recipient = backup.load_recipient(path)
    assert recipient["fingerprint"] == recorded["fingerprint"] and recipient["public_key"].key_size >= 3072
    assert recorded["decision"]["by"] == "owner"
    text = path.read_text()
    assert "PRIVATE KEY" not in text and "@" not in text
    # A generation sealed to it carries the envelope a restore opens.
    _, envelope = backup.new_generation_key(recipient)
    assert envelope["scheme"] == backup.ENVELOPE_SCHEME and envelope["recipient_fingerprint"] == recorded["fingerprint"]


def test_the_timers_run_seals_to_the_release_key_and_reads_no_key_file(host, tmp_path, monkeypatch):
    private_path, recipient_path = _owner_key(tmp_path)
    store = FakeStore()
    seen: dict = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return {"status": "complete", "withdrawal_purge": {"outcome": None}}

    monkeypatch.setattr(backup.S3BackupStore, "configured", classmethod(lambda cls: store))
    monkeypatch.setattr(backup, "run_backup", fake_run)
    monkeypatch.setenv(backup.RECIPIENT_FILE_ENV, str(recipient_path))
    monkeypatch.setenv(backup.REPORT_ROOT_ENV, str(tmp_path / "report"))
    monkeypatch.delenv(backup.PRIVATE_KEY_FILE_ENV, raising=False)
    assert backup.main(["run"]) == 0
    assert backup.open_generation_key(seen["key_envelope"], private_path) == seen["key"]
    # A restore names the owner's private key or refuses; it never guesses one.
    with pytest.raises(SystemExit, match=backup.PRIVATE_KEY_FILE_ENV):
        backup.main(["restore", "--fs-root", str(tmp_path / "drill")])


# --------------------------------------------------------------------------- withdrawal purge


def _deletion_receipt(host: Path, applied_at: float) -> dict:
    receipt = {"schema_version": backup.DELETION_RECEIPT_SCHEMA, "scene_id": "site-req1",
               "applied_at_iso": datetime.fromtimestamp(applied_at, timezone.utc).isoformat(),
               "deletion_status": "complete"}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    _write(host, f"{SITE}/website_withdrawal/deletion_receipts/{receipt['digest'][7:]}.json",
           json.dumps(receipt).encode())
    return receipt


def test_a_deletion_purges_every_older_backup_after_a_fresh_one(host, tmp_path):
    store = FakeStore()
    earlier = _run(host, store, tmp_path, now=time.time() - 3600)
    receipt = _deletion_receipt(host, time.time() - 600)
    ledger = backup.ledger_path(tmp_path / "report")
    site = host / SITE
    assert backup.backup_copies_status(site, receipt["digest"], receipt["applied_at_iso"],
                                       ledger=ledger)["status"] == "purge_pending"
    fresh = _run(host, store, tmp_path)
    assert fresh["withdrawal_purge"]["purged"] == 1
    assert store.generations() == {fresh["generation"]}
    purge = backup.read_purge_receipt(site, receipt["digest"])
    assert purge["outcome"] == "purged" and purge["generations_deleted"] == [earlier["generation"]]
    assert purge["proven_absent_by_listing"] is True and purge["physical_erasure_proven"] is False
    assert backup.backup_copies_status(site, receipt["digest"], receipt["applied_at_iso"],
                                       ledger=ledger)["status"] == "purged"


def test_a_purge_with_an_unlistable_bucket_is_unknown_and_never_clears(host, tmp_path):
    store = FakeStore()
    _run(host, store, tmp_path, now=time.time() - 3600)
    receipt = _deletion_receipt(host, time.time() - 600)
    ledger = backup.ledger_path(tmp_path / "report")
    gen = "g-20991231T000000000000Z-00000000"
    backup._write_ledger(ledger, [*backup.read_ledger(ledger)["generations"],
                                  {"id": gen, "started_at_epoch": time.time(), "status": "sealed"}])
    store.unlistable = True
    result = backup.purge_withdrawn(store, site_roots=[host / "var/lib/blueprint/pubsub-handoffs"], ledger=ledger,
                                    fresh={"id": gen, "started_at_epoch": time.time()}, now=time.time())
    assert result["outcome"] == "unknown" and result["unknown"] == 1
    purge = backup.read_purge_receipt(host / SITE, receipt["digest"])
    assert purge["outcome"] == "unknown" and "listing" in purge["reason"]
    status = backup.backup_copies_status(host / SITE, receipt["digest"], receipt["applied_at_iso"], ledger=ledger)
    assert status["status"] == "unknown"
    assert status["status"] not in backup.BACKUP_CLEARED_STATUSES


def test_an_unreadable_ledger_is_unknown_and_no_ledger_means_no_backup(tmp_path):
    ledger = tmp_path / "ledger.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    assert backup.backup_copies_status(tmp_path, "sha256:" + "0" * 64, now_iso, ledger=ledger)["status"] == \
        "no_backup_taken"
    ledger.write_text("{broken")
    assert backup.backup_copies_status(tmp_path, "sha256:" + "0" * 64, now_iso, ledger=ledger)["status"] == "unknown"


# --------------------------------------------------------------------------- money after a restore


def _lock(restore_hold):
    now = datetime.now(timezone.utc)
    return build_spend_admission_lock(
        fleet_budget={"status": "passed", "total_spend_usd": 0.0, "blockers": []},
        billing_reconciliation={"status": "reconciled", "required": True, "currency": "USD",
                                "billing_export_schema_version": "blueprint.provider_billing_export.v1",
                                "scope": "blueprint_beta_100_user_cohort", "generated_at": now.isoformat(),
                                "billing_export_sha256": "sha256:" + "a" * 64, "billing_export_mode_octal": "0600",
                                "provider_totals_usd": {"vast": 0.0, "digitalocean": 0.0}},
        instances=[], reap_results=[],
        inventory_results=[{"provider": p, "required": True, "status": "succeeded", "credential_present": True,
                            "api_confirmed": True, "row_count": 0, "blockers": []} for p in ("vast", "digitalocean")],
        override_path=None, now=now, restore_hold=restore_hold)


def test_a_restore_blocks_paid_admission_until_the_reap_and_the_billing_feed_postdate_it(host, tmp_path):
    store = FakeStore()
    _run(host, store, tmp_path)
    marker = tmp_path / "drill" / CP / "gpu_spend_guard" / hold.MARKER_NAME
    restored_at = time.time()
    receipt = backup.restore(store=store, key=KEY, fs_root=tmp_path / "drill", hold_marker=marker, now=restored_at)
    assert receipt["money_hold_marker"] == str(marker)
    assert _lock(None)["admission_allowed"] is True  # the lock itself is otherwise open

    def billing(at: float) -> dict:
        return {"status": "reconciled", "generated_at": datetime.fromtimestamp(at, timezone.utc).isoformat()}

    # Billing from before the restore, or a run that did not reap: still blocked, the marker stays.
    for reading in (
        hold.evaluate(hold.read_hold(marker), billing=billing(restored_at - 60), reap_mode=True,
                      inventory_complete=True, all_reap_candidates_terminated=True, now=restored_at + 60),
        hold.evaluate(hold.read_hold(marker), billing=billing(restored_at + 30), reap_mode=False,
                      inventory_complete=True, all_reap_candidates_terminated=True, now=restored_at + 60),
        hold.evaluate(hold.read_hold(marker), billing=billing(restored_at + 30), reap_mode=True,
                      inventory_complete=False, all_reap_candidates_terminated=True, now=restored_at + 60),
    ):
        lock = _lock(reading)
        assert lock["admission_allowed"] is False
        assert hold.BLOCKER in lock["blockers"]
        assert hold.retire(marker, reading) is None and marker.exists()
    # Both after the restore: the same run clears it, retires the marker, and the next lock is open.
    cleared = hold.evaluate(hold.read_hold(marker), billing=billing(restored_at + 30), reap_mode=True,
                            inventory_complete=True, all_reap_candidates_terminated=True, now=restored_at + 60)
    assert cleared["status"] == "cleared" and _lock(cleared)["admission_allowed"] is True
    retired = hold.retire(marker, cleared, now=restored_at + 60)
    assert not marker.exists() and json.loads(retired.read_text())["retired_by"] == "gpu_spend_guard"
    assert hold.read_hold(marker) is None and _lock(None)["admission_allowed"] is True


def test_an_unreadable_hold_marker_still_blocks_and_still_clears_by_itself(tmp_path):
    marker = tmp_path / hold.MARKER_NAME
    marker.write_text("{not json")
    restored = os.lstat(marker).st_mtime
    reading = hold.read_hold(marker)
    assert reading["marker_unreadable"] is True
    blocked = hold.evaluate(reading, billing={"status": "reconciled", "generated_at": "1970-01-01T00:00:00+00:00"},
                            reap_mode=True, inventory_complete=True, all_reap_candidates_terminated=True,
                            now=restored + 60)
    assert blocked["blocking"] is True
    later = datetime.fromtimestamp(restored + 30, timezone.utc).isoformat()
    cleared = hold.evaluate(reading, billing={"status": "reconciled", "generated_at": later}, reap_mode=True,
                            inventory_complete=True, all_reap_candidates_terminated=True, now=restored + 60)
    assert cleared["status"] == "cleared"


def test_the_deletion_receipt_schema_is_the_deletion_tool_s():
    from blueprint_pipeline import website_capture_withdrawal_deletion as deletion

    assert backup.DELETION_RECEIPT_SCHEMA == deletion.RECEIPT_SCHEMA


# --------------------------------------------------------------------------- the stall alarm's backup check


def test_a_backup_that_stopped_pages_and_a_host_never_set_up_warns(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_capacity_controller as capacity
    from blueprint_pipeline import control_plane_stall_alarm as stall

    report = tmp_path / "report"
    monkeypatch.setenv(backup.REPORT_ROOT_ENV, str(report))
    now = time.time()
    # F-01: never set up is a standing warning, not silence and not a page.
    assert stall._check_state_backup(now=now) == (
        {"status": "not_configured"},
        [{"code": backup.NOT_CONFIGURED_CODE, "target": str(backup.ledger_path(report))}])
    assert backup.NOT_CONFIGURED_CODE not in stall.PAGE_CODES
    assert capacity._severity(backup.NOT_CONFIGURED_CODE) == "warn"
    report.mkdir()
    ledger = backup.ledger_path(report)
    # Hourly runs: one missed run (two hours since the last complete one) is not a page; four hours is.
    backup._write_ledger(ledger, [{"id": "g-a", "started_at_epoch": now - 2 * 3600, "status": "sealed"},
                                  {"id": "g-b", "started_at_epoch": now - 600, "status": "incomplete"}])
    assert stall._check_state_backup(now=now)[1] == []
    backup._write_ledger(ledger, [{"id": "g-a", "started_at_epoch": now - 4 * 3600, "status": "sealed"},
                                  {"id": "g-b", "started_at_epoch": now - 600, "status": "incomplete"}])
    section, alerts = stall._check_state_backup(now=now)
    assert section["status"] == "stale" and [row["code"] for row in alerts] == [backup.STALE_CODE]
    assert backup.STALE_CODE in stall.PAGE_CODES and "4.0h" in stall.describe(alerts[0])
    backup._write_ledger(ledger, [{"id": "g-c", "started_at_epoch": now - 900, "status": "sealed"}])
    assert stall._check_state_backup(now=now)[1] == []
    ledger.write_text("{broken")
    section, alerts = stall._check_state_backup(now=now)
    assert section["status"] == "unreadable"
    assert alerts == [{"code": stall.INCOMPLETE_CODE, "reason": "state_backup_ledger_unreadable"}]
