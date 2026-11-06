"""The G1 runtime source restore, against tmp paths and an in-memory object store."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/native_g1_runtime_source_restore.py
#   src/blueprint_pipeline/wam_provider_object_store.py

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import native_g1_runtime_source_restore as restore
from blueprint_pipeline import wam_provider_object_store as object_store
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_runtime_source_packet import MANIFEST_SCHEMA_VERSION, SCHEMA_VERSION

Usage = namedtuple("Usage", "total used free")
REGISTRY_RECEIPT = ("/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926/"
                    "native_task_runtime_source_packet.v1.json")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _packet() -> bytes:
    member = b"print('isaac lab source')\n" * 64
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "runtime_profile": "unitree_g1",
        "repositories": [{"files": [{"archive_path": "IsaacLab/source/a.py", "size_bytes": len(member),
                                     "sha256": _sha(member)}]}],
        "runtime_dependency_wheels": [],
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("native_task_runtime_source_manifest.v1.json", json.dumps(manifest))
        archive.writestr("IsaacLab/source/a.py", member)
    return buffer.getvalue()


class _Missing(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 404}, "Error": {"Code": "NoSuchKey"}}


class FakeStore:
    """HEAD and GET only, like the real retained copy; ``body`` replaces what a GET returns."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.body: bytes | None = None

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append(("head", Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise _Missing()
        data, metadata = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "Metadata": dict(metadata), "ETag": '"fixture-etag"'}

    def get_object(self, *, Bucket: str, Key: str, IfMatch: str | None = None) -> dict[str, Any]:
        self.calls.append(("get", Bucket, Key))
        assert IfMatch == '"fixture-etag"'
        data, _ = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(data if self.body is None else self.body)}


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    work = tmp_path / "work"
    work.mkdir()
    target = work / "g1-runtime-source"
    packet = _packet()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "runtime_profile": "unitree_g1",
        "packet_path": str(target / restore.PACKET_NAME), "packet_sha256": _sha(packet),
        "packet_size_bytes": len(packet),
        "manifest_digest": json.loads(zipfile.ZipFile(io.BytesIO(packet)).read(
            "native_task_runtime_source_manifest.v1.json"))["manifest_digest"],
        "runtime_dependency_wheels": [], "redistribution_permitted": True, "policy_bytes_included": False,
    }
    record["receipt_digest"] = canonical_digest(record, digest_field="receipt_digest")
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    build = work / "g1-runtime-source-build.json"
    build.write_bytes(raw)
    key = object_store.cached_dependency_object_key(restore.OBJECT_KEY_PREFIX, _sha(packet), "runtime_dependency")
    pins = restore.Pins(
        work_root=work, build_record=build, build_record_sha256=_sha(raw), build_record_size_bytes=len(raw),
        receipt_digest=record["receipt_digest"], target_dir=target, packet_sha256=_sha(packet),
        packet_size_bytes=len(packet), runtime_profile="unitree_g1",
        object_key_sha256=hashlib.sha256(key.encode()).hexdigest())
    monkeypatch.setattr(restore, "PINS", pins)
    store = FakeStore()
    store.objects[("fixture-bucket", key)] = (packet, {"sha256": _sha(packet).removeprefix("sha256:")})
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    for name, value in (("ACCESS_KEY_ID", "fixture-access"), ("SECRET_ACCESS_KEY", "fixture-secret-value"),
                        ("ENDPOINT_URL", "https://store.example.test"), ("BUCKET", "fixture-bucket"),
                        ("REGION", "fixture-region")):
        (secrets / name).write_text(value + "\n", encoding="utf-8")
        monkeypatch.setenv(f"BLUEPRINT_WAM_OBJECT_STORE_{name}_FILE", str(secrets / name))
        monkeypatch.delenv(f"BLUEPRINT_WAM_OBJECT_STORE_{name}", raising=False)
    clients: list[dict[str, str]] = []

    def client(**kwargs: str) -> FakeStore:
        clients.append(kwargs)
        return store

    monkeypatch.setattr(object_store, "_s3_client", client)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(tmp_path / "ledger"))
    monkeypatch.setattr(restore, "_disk_usage", lambda _path: Usage(10**15, 0, 10**14))
    return {"pins": pins, "store": store, "packet": packet, "raw": raw, "work": work, "target": target,
            "ledger": tmp_path / "ledger", "clients": clients, "key": key}


def _tree(root: Path) -> dict[str, bytes | None]:
    return {str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
            for path in sorted(root.rglob("*"))}


def _reservations(ledger: Path) -> list[str]:
    return sorted(path.name for path in ledger.glob("*.json")) if ledger.is_dir() else []


def test_production_pins_name_the_registry_binding_and_the_retained_cache_key() -> None:
    from blueprint_pipeline.adp_isaac_lab_arena_vast import DEFAULT_KEY_PREFIX

    pins = restore.PINS
    assert str(pins.receipt_path) == REGISTRY_RECEIPT
    assert pins.packet_path.parent == pins.receipt_path.parent
    assert pins.build_record.parent == pins.work_root == Path("/mnt/blueprint-work")
    assert restore.OBJECT_KEY_PREFIX == DEFAULT_KEY_PREFIX
    # The 09-27 run's cache record names the key only by its sha256, which starts 561630a4.
    assert pins.object_key_sha256.startswith("561630a4")
    assert hashlib.sha256(restore.object_key(pins).encode()).hexdigest() == pins.object_key_sha256
    assert restore.object_key(pins).endswith(pins.packet_sha256.removeprefix("sha256:") + ".zip")


def test_the_cached_dependency_key_is_the_staging_scheme() -> None:
    sha = "ab" * 32
    assert object_store.cached_dependency_object_key("/p/q/ ", "sha256:" + sha) == \
        f"p/q/runtime-dependencies/sha256/{sha}.zip"
    assert object_store.cached_dependency_object_key("", sha, "g1_checkpoint") == \
        f"blueprint/wam-provider/g1-checkpoints/sha256/{sha}.bin"
    with pytest.raises(ValueError, match="cached_dependency_artifact_kind_invalid"):
        object_store.cached_dependency_object_key("p", sha, "other")


def test_a_plan_checks_everything_and_writes_nothing(world: dict[str, Any]) -> None:
    before = _tree(world["work"])
    result = restore.restore()

    assert result["outcome"] == "planned" and result["code"] is None and result["mode"] == "plan"
    assert result["build_record"]["verified"] is True and result["restore_needed"] is True
    assert result["targets"] == {"directory": str(world["target"]), "directory_present": False,
                                 "receipt_path": str(world["target"] / restore.RECEIPT_NAME),
                                 "packet_path": str(world["target"] / restore.PACKET_NAME),
                                 "receipt": "absent", "packet": "absent"}
    assert result["object_store"]["present"] is True and result["object_store"]["size_matches"] is True
    assert result["disk"]["sufficient"] is True
    assert result["disk"]["need_bytes"] == len(world["packet"]) + len(world["raw"]) + restore.DISK_MARGIN_BYTES
    assert [call[0] for call in world["store"].calls] == ["head"]
    assert _tree(world["work"]) == before and _reservations(world["ledger"]) == []
    assert "runtime_source" not in result and result["provider_mutation_performed"] is False


def test_apply_restores_byte_identical_files_that_verify(world: dict[str, Any]) -> None:
    result = restore.restore(apply=True)

    target = world["target"]
    assert result["outcome"] == "restored", result
    assert (target / restore.PACKET_NAME).read_bytes() == world["packet"]
    assert (target / restore.RECEIPT_NAME).read_bytes() == world["raw"]
    assert oct(target.stat().st_mode & 0o777) == oct(0o750)
    for name in (restore.PACKET_NAME, restore.RECEIPT_NAME):
        assert oct((target / name).stat().st_mode & 0o777) == oct(0o640)
    assert sorted(os.listdir(target)) == sorted([restore.PACKET_NAME, restore.RECEIPT_NAME])
    assert result["verification"] == {"verified": True} and result["download"]["performed"] is True
    assert result["runtime_source"] == {"sha256": _sha(world["raw"]), "size_bytes": len(world["raw"])}
    assert result["receipt_sha256"] == _sha(world["raw"]) and result["packet_sha256"] == _sha(world["packet"])
    assert _reservations(world["ledger"]) == [], "the disk hold is released"
    assert [call[0] for call in world["store"].calls] == ["head", "get"]


def test_a_second_apply_is_already_restored_and_touches_nothing(world: dict[str, Any]) -> None:
    assert restore.restore(apply=True)["outcome"] == "restored"
    before = {path: path.stat().st_mtime_ns for path in world["target"].iterdir()}
    world["store"].calls.clear()

    result = restore.restore(apply=True)

    assert result["outcome"] == "already_restored" and result["restore_needed"] is False
    assert result["runtime_source"] == {"sha256": _sha(world["raw"]), "size_bytes": len(world["raw"])}
    assert {path: path.stat().st_mtime_ns for path in world["target"].iterdir()} == before
    assert world["store"].calls == [] and result["written"] == []
    plan = restore.restore()
    assert plan["outcome"] == "planned" and plan["restore_needed"] is False


def test_only_the_missing_receipt_is_written_beside_a_matching_packet(world: dict[str, Any]) -> None:
    world["target"].mkdir(mode=0o750)
    (world["target"] / restore.PACKET_NAME).write_bytes(world["packet"])

    result = restore.restore(apply=True)

    assert result["outcome"] == "restored" and result["download"]["performed"] is False
    assert result["written"] == [str(world["target"] / restore.RECEIPT_NAME)]
    assert [call[0] for call in world["store"].calls] == ["head"]


@pytest.mark.parametrize(("body", "code"), [
    ("corrupt", "g1_runtime_source_digest_mismatch"),
    ("short", "g1_runtime_source_object_size_mismatch"),
    ("long", "g1_runtime_source_object_size_mismatch"),
])
def test_a_bad_download_is_refused_and_leaves_no_partial_file(world: dict[str, Any], body: str, code: str) -> None:
    packet = world["packet"]
    world["store"].body = {"corrupt": packet[:-1] + bytes([packet[-1] ^ 1]), "short": packet[:-10],
                           "long": packet + b"x"}[body]
    before = _tree(world["work"])

    result = restore.restore(apply=True)

    assert result["outcome"] == "refused" and result["code"] == code
    assert _tree(world["work"]) == before, "no partial file and no new directory remain"
    assert "runtime_source" not in result and _reservations(world["ledger"]) == []


def test_a_cut_transfer_is_refused_and_leaves_no_partial_file(world: dict[str, Any]) -> None:
    class Cut(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise ConnectionResetError("cut")

    world["store"].get_object = lambda **_kwargs: {"Body": Cut()}
    world["target"].mkdir(mode=0o750)
    result = restore.restore(apply=True)
    assert result["code"] == "g1_runtime_source_download_failed"
    assert os.listdir(world["target"]) == []


@pytest.mark.parametrize("conflict", ["packet", "receipt", "symlink", "directory"])
def test_an_existing_target_with_other_bytes_is_refused_untouched(world: dict[str, Any], tmp_path: Path,
                                                                  conflict: str) -> None:
    target = world["target"]
    if conflict == "directory":
        target.write_bytes(b"not a directory")
    else:
        target.mkdir(mode=0o750)
        if conflict == "packet":
            (target / restore.PACKET_NAME).write_bytes(b"other bytes")
        elif conflict == "receipt":
            (target / restore.RECEIPT_NAME).write_bytes(world["raw"].replace(b"unitree_g1", b"unitree_h1"))
        else:
            (tmp_path / "elsewhere.zip").write_bytes(world["packet"])
            (target / restore.PACKET_NAME).symlink_to(tmp_path / "elsewhere.zip")
    before = _tree(world["work"])

    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_runtime_source_target_conflict"
    assert _tree(world["work"]) == before and world["store"].calls == []


@pytest.mark.parametrize("change", ["bytes", "missing", "symlink"])
def test_a_changed_build_record_is_refused(world: dict[str, Any], tmp_path: Path, change: str) -> None:
    build = world["pins"].build_record
    if change == "bytes":
        build.write_bytes(world["raw"].replace(b"unitree_g1", b"unitree_h1"))
    elif change == "missing":
        build.unlink()
    else:
        copy = tmp_path / "copy.json"
        copy.write_bytes(world["raw"])
        build.unlink()
        build.symlink_to(copy)

    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_runtime_source_build_record_changed"
    assert not world["target"].exists() and world["store"].calls == []


def test_a_pinned_digest_the_record_does_not_carry_is_refused(world: dict[str, Any],
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    monkeypatch.setattr(restore, "PINS", dataclasses.replace(world["pins"], receipt_digest="sha256:" + "0" * 64))
    assert restore.restore()["code"] == "g1_runtime_source_build_record_changed"


@pytest.mark.parametrize(("mutate", "code"), [
    ("remove", "g1_runtime_source_object_missing"),
    ("size", "g1_runtime_source_object_size_mismatch"),
    ("metadata", "g1_runtime_source_digest_mismatch"),
])
def test_the_retained_object_must_exist_with_its_exact_identity(world: dict[str, Any], mutate: str,
                                                                code: str) -> None:
    location = ("fixture-bucket", world["key"])
    data, metadata = world["store"].objects[location]
    if mutate == "remove":
        del world["store"].objects[location]
    elif mutate == "size":
        world["store"].objects[location] = (data + b"x", metadata)
    else:
        world["store"].objects[location] = (data, {"sha256": "0" * 64})
    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == code
    assert not world["target"].exists()


def test_missing_store_credentials_are_refused_without_a_client(world: dict[str, Any],
                                                                monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    monkeypatch.setenv("BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE", str(tmp_path / "absent"))
    monkeypatch.setattr(object_store, "DEFAULT_SECRET_KEY_FILES", ())
    result = restore.restore()
    assert result["code"] == "g1_runtime_source_object_store_unavailable" and world["clients"] == []


def test_too_little_disk_is_refused_before_any_download(world: dict[str, Any],
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(restore, "_disk_usage", lambda _path: Usage(10 * 1024**3, 0, 8 * 1024**3 + 100 * 1024**2))
    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_runtime_source_disk_insufficient"
        assert result["disk"]["sufficient"] is False
    assert not world["target"].exists() and "get" not in [call[0] for call in world["store"].calls]


def test_a_packet_that_does_not_verify_is_removed_with_its_receipt(world: dict[str, Any],
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from blueprint_pipeline.native_task_runtime_source_packet import NativeTaskRuntimeSourcePacketError

    def refuse(_path: Path) -> dict[str, Any]:
        raise NativeTaskRuntimeSourcePacketError(["native_task_runtime_source_manifest_invalid"])

    monkeypatch.setattr(restore, "verify_native_task_runtime_source_packet", refuse)
    before = _tree(world["work"])

    result = restore.restore(apply=True)

    assert result["outcome"] == "refused" and result["code"] == "g1_runtime_source_verification_failed"
    assert result["verification"]["verified"] is False
    assert _tree(world["work"]) == before, "this run's directory, packet and receipt are removed"
    assert "runtime_source" not in result


def test_the_result_never_records_a_key_url_or_credential(world: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    assert restore.main(["--apply", "--result", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["outcome"] == "restored" and document["schema_version"] == restore.SCHEMA_VERSION
    assert document["raw_secret_values_recorded"] is False and document["provider_mutation_performed"] is False
    for secret in (world["key"], "fixture-secret-value", "fixture-access", "store.example.test", "fixture-bucket"):
        assert secret not in text
    assert restore.main(["--apply"]) == 0


def test_a_refusal_exits_two_with_its_fixed_code(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    world["pins"].build_record.unlink()
    assert restore.main([]) == 2
    document = json.loads(capsys.readouterr().out)
    assert document["outcome"] == "refused" and document["code"] in restore.REFUSALS
