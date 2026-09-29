# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_output_member_view.py
#   src/blueprint_pipeline/provider_output_member_index.py
#   src/blueprint_pipeline/provider_output_range_ingestion.py
#   tests/provider_output_fixtures.py
"""A streamed attempt answers member digests from its index and bytes from the durable archive."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import provider_output_member_view as views
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_output_member_index import (
    build_member_index,
    build_member_selection,
    seal_durable_reference,
)
from blueprint_pipeline.provider_output_range_ingestion import CasArchiveSource, ingest_selected_members
from tests.provider_output_fixtures import (
    DEFLATED,
    SECRET,
    URL,
    Entry,
    RangeStore,
    Zeros,
    build_zip,
    no_disk_writes,
)

RESULT = "native_task_arena_policy_canary_session_result.v1.json"
FRAME = "cell_runs/00/episodes/media/e0/frames/external/000000.png"
VIDEO = "cell_runs/00/episodes/media/e0/external.mp4"
RECEIPT = "cell_runs/00/episodes/e0.score_receipt.json"
FRAME_BYTES = bytes(range(256)) * 64


def _reference(index):
    digest = index["archive"]["sha256"]
    return {"schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
            "artifact_kind": "policy-canary-provider-output",
            "uri": ("s3://blueprint-artifacts/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/"
                    f"policy-canary-provider-output/sha256/{digest.removeprefix('sha256:')}/"
                    "vast_provider_runtime_output.zip"),
            "digest": digest, "size_bytes": index["archive"]["size"], "content_addressed_key": True,
            "remote_identity_verified": True, "full_byte_service_account_readback_passed": True}


class Streamed(SimpleNamespace):
    def view(self, path=None, **options):
        options.setdefault("presign", self.presign)
        options.setdefault("opener", self.store.opener)
        return views.open_member_view(path or self.evidence, **options)


def _streamed(tmp_path: Path) -> Streamed:
    archive = build_zip([
        Entry(VIDEO, Zeros(1024**2)),
        Entry(FRAME, FRAME_BYTES),
        Entry(RECEIPT, json.dumps({"episode_id": "e0", "task_success": True}).encode(), method=DEFLATED),
        Entry(RESULT, json.dumps({"status": "completed", "episodes": list(range(50))}).encode(), method=DEFLATED),
    ])
    store = RangeStore(archive)
    index = build_member_index(store.reader(block_bytes=128 * 1024), maximum_expanded_bytes=64 * 1024**2)
    index = seal_durable_reference(index, _reference(index))
    attempt = tmp_path / "attempt_001"
    attempt.mkdir(parents=True)
    index_path = attempt / "provider_output_member_index.v1.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    presigns: list[int] = []

    def presign():
        presigns.append(1)
        return URL

    selection = build_member_selection(index, [RESULT, RECEIPT], selection_version="test-contract.v1")
    source = CasArchiveSource(index["archive"]["durable_reference"], presign=presign, opener=store.opener,
                              block_bytes=128 * 1024)
    receipt = ingest_selected_members(
        source=source, index=index, selection=selection, members_root=attempt / "immutable_execution",
        metadata_root=attempt / ".provider_output_ingestion", reserve=lambda needed: None,
        disk_usage_provider=lambda path: SimpleNamespace(free=10**12))
    assert receipt["status"] == "materialized", receipt["blockers"]
    presigns.clear()
    descriptor = views.write_member_view_descriptor(
        evidence_root=attempt / "immutable_execution", index_path=index_path,
        ingestion_receipt_path=attempt / ".provider_output_ingestion" / "receipt.json")
    rows = {row["path"]: row for row in index["members"]}
    return Streamed(archive=archive, store=store, index=index, rows=rows, attempt=attempt,
                    evidence=attempt / "immutable_execution", descriptor=descriptor, presign=presign,
                    presigns=presigns, index_path=index_path)


def _listing(root: Path):
    return sorted((path.relative_to(root).as_posix(), path.stat().st_size) for path in root.rglob("*"))


def test_view_answers_digests_from_the_index_without_bytes(tmp_path):
    streamed = _streamed(tmp_path)
    before = len(streamed.store.requests)

    # Discovered from a remote member's path, which has no file of its own.
    view = streamed.view(streamed.evidence / FRAME)

    assert view.evidence_root == streamed.evidence.resolve()
    frame = streamed.rows[FRAME]
    assert view.member(FRAME) == frame and view.digest(FRAME) == frame["sha256"]
    assert view.verify(FRAME, sha256=frame["sha256"], size_bytes=frame["size"]) is True
    assert view.verify(FRAME, sha256=frame["sha256"], size_bytes=frame["size"] + 1) is False
    assert view.verify(FRAME, sha256="sha256:" + "0" * 64, size_bytes=frame["size"]) is False
    assert view.digest("cell_runs/00/missing.json") is None and view.member("cell_runs/00/missing.json") is None
    assert view.digest("cell_runs/00/episodes") is None  # a directory is no member
    with pytest.raises(views.ProviderOutputMemberViewError, match="^provider_output_member_view_path_invalid$"):
        view.digest("../escape.json")
    # Digests come from the index alone: no presign, no request.
    assert streamed.presigns == [] and len(streamed.store.requests) == before
    # The deepest Quick-10 path (eight levels below the root) still finds it.
    assert streamed.view(streamed.evidence / FRAME).descriptor == streamed.descriptor
    assert streamed.descriptor["view_digest"] == canonical_digest(streamed.descriptor, digest_field="view_digest")
    assert SECRET not in json.dumps(streamed.descriptor) and "https:" not in json.dumps(streamed.descriptor)


def test_read_member_is_one_range_request_checked_by_crc_and_sha256(tmp_path):
    streamed = _streamed(tmp_path)
    view = streamed.view()
    for path, expected in ((FRAME, FRAME_BYTES),
                           (RESULT, json.dumps({"status": "completed", "episodes": list(range(50))}).encode())):
        row, before = streamed.rows[path], len(streamed.store.requests)
        assert view.read_member(path, maximum_bytes=row["size"]) == expected
        # The reader's one-byte probe pins the ETag; the member is one range.
        assert [entry["range"] for entry in streamed.store.requests[before:]] == [
            (0, 0), (row["data_offset"], row["data_offset"] + row["compressed_size"] - 1)]
    assert len(streamed.presigns) == 2

    before = len(streamed.store.requests)
    for path, maximum, code in ((FRAME, len(FRAME_BYTES) - 1, "provider_output_member_read_cap_exceeded"),
                                ("cell_runs/00/absent.png", 10, "provider_output_member_view_member_absent")):
        with pytest.raises(views.ProviderOutputMemberViewError, match=f"^{code}$"):
            view.read_member(path, maximum_bytes=maximum)
    assert len(streamed.store.requests) == before

    frame = streamed.rows[FRAME]
    tampered = RangeStore(streamed.archive.patched(frame["data_offset"] + 9, b"\xff"))
    with pytest.raises(views.ProviderOutputMemberViewError, match="^provider_output_member_digest_mismatch$"):
        streamed.view(opener=tampered.opener).read_member(FRAME, maximum_bytes=10**6)
    shorter = RangeStore(b"a different, shorter object")
    with pytest.raises(views.ProviderOutputMemberViewError, match="^provider_output_remote_size_mismatch$"):
        streamed.view(opener=shorter.opener).read_member(FRAME, maximum_bytes=10**6)


def _rewrite(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reseal(value, **changes):
    changed = {**copy.deepcopy(value), **changes}
    changed["view_digest"] = canonical_digest(changed, digest_field="view_digest")
    return changed


def test_view_refuses_a_descriptor_that_does_not_bind_its_index(tmp_path):
    streamed = _streamed(tmp_path)
    path = streamed.attempt / ("immutable_execution" + views.DESCRIPTOR_SUFFIX)
    original = json.loads(path.read_text(encoding="utf-8"))
    other_index = copy.deepcopy(streamed.index)
    other_index["limits"]["maximum_members"] = 9_999
    other_index["index_digest"] = canonical_digest(other_index, digest_field="index_digest")
    (streamed.attempt / "other_index.json").write_text(json.dumps(other_index), encoding="utf-8")
    other_record = {"path": "other_index.json", "size_bytes": (streamed.attempt / "other_index.json").stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256((streamed.attempt / "other_index.json").read_bytes()).hexdigest()}
    cases = {
        "unsealed_change": ({**original, "archive_sha256": "sha256:" + "0" * 64},
                            "provider_output_member_view_descriptor_invalid"),
        "another_evidence_root": (_reseal(original, evidence_root="immutable_execution_2"),
                                  "provider_output_member_view_descriptor_invalid"),
        "index_record_changed": (_reseal(original, member_index={**original["member_index"], "size_bytes": 1}),
                                 "provider_output_member_view_index_mismatch"),
        "index_of_another_view": (_reseal(original, member_index=other_record),
                                  "provider_output_member_view_index_mismatch"),
        "another_archive": (_reseal(original, archive_sha256="sha256:" + "0" * 64),
                            "provider_output_member_view_index_mismatch"),
        "another_durable_copy": (_reseal(original, durable_reference={
            **original["durable_reference"], "uri": original["durable_reference"]["uri"].replace(
                "blueprint-artifacts", "another-bucket")}), "provider_output_member_view_index_mismatch"),
        "receipt_changed": (_reseal(original, ingestion_receipt_sha256="sha256:" + "0" * 64),
                            "provider_output_member_view_receipt_mismatch"),
        "url_recorded": (_reseal(original, private_url_recorded=True),
                         "provider_output_member_view_descriptor_invalid"),
    }
    for name, (value, code) in cases.items():
        _rewrite(path, value)
        with pytest.raises(views.ProviderOutputMemberViewError, match=f"^{code}$"):
            streamed.view()
    _rewrite(path, original)
    assert streamed.view() is not None
    # The index file itself changing after the view was written is refused.
    streamed.index_path.write_text(streamed.index_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(views.ProviderOutputMemberViewError, match="^provider_output_member_view_index_mismatch$"):
        streamed.view()
    # A descriptor that is a symlink is never followed.
    path.unlink()
    (tmp_path / "elsewhere.json").write_text(json.dumps(original), encoding="utf-8")
    path.symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(views.ProviderOutputMemberViewError, match="^provider_output_member_view_descriptor_invalid$"):
        streamed.view()


def test_view_is_absent_without_a_descriptor(tmp_path):
    evidence = tmp_path / "download_mode" / "attempt_001" / "immutable_execution"
    (evidence / "cell_runs/00").mkdir(parents=True)
    (evidence / RESULT).write_text("{}", encoding="utf-8")
    assert views.open_member_view(evidence / RESULT) is None
    assert views.open_member_view(evidence / FRAME) is None
    assert views.open_member_view(evidence) is None

    streamed = _streamed(tmp_path / "streamed")
    # Only eight ancestors are searched: a path nine levels below is not in the view.
    assert streamed.view(streamed.evidence / "a/b/c/d/e/f/g/h/i.json") is None
    # A descriptor is written only for a materialized ingestion of a sealed index.
    receipt_path = streamed.attempt / ".provider_output_ingestion" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    blocked = streamed.attempt / "blocked_receipt.json"
    _rewrite(blocked, {**receipt, "status": "blocked"})
    unsealed = streamed.attempt / "unsealed_index.json"
    unsealed_index = copy.deepcopy(streamed.index)
    unsealed_index["archive"]["durable_reference"] = None
    unsealed_index["index_digest"] = canonical_digest(unsealed_index, digest_field="index_digest")
    _rewrite(unsealed, unsealed_index)
    for index_path, ingestion, code in (
        (streamed.index_path, blocked, "provider_output_member_view_ingestion_not_materialized"),
        (unsealed, receipt_path, "provider_output_member_view_index_not_durable"),
    ):
        with pytest.raises(views.ProviderOutputMemberViewError, match=f"^{code}$"):
            views.build_member_view_descriptor(evidence_root=streamed.evidence, index_path=index_path,
                                               ingestion_receipt_path=ingestion)
    # Written once: writing the same descriptor again returns it unchanged.
    assert views.write_member_view_descriptor(
        evidence_root=streamed.evidence, index_path=streamed.index_path,
        ingestion_receipt_path=receipt_path) == streamed.descriptor


def test_view_never_writes_into_the_evidence_root(tmp_path, monkeypatch):
    streamed = _streamed(tmp_path)
    before = _listing(streamed.evidence)
    monkeypatch.chdir(tmp_path)

    with no_disk_writes(monkeypatch) as attempts:
        view = streamed.view(streamed.evidence / FRAME)
        assert view.digest(FRAME) == streamed.rows[FRAME]["sha256"]
        assert view.read_member(FRAME, maximum_bytes=10**6) == FRAME_BYTES
    assert attempts == []

    for destination in (streamed.evidence / "copy.png", streamed.evidence, streamed.evidence / FRAME):
        with pytest.raises(views.ProviderOutputMemberViewError,
                           match="^provider_output_member_view_write_inside_evidence_root$"):
            view.fetch_to(FRAME, destination)
    scratch = tmp_path / "scratch" / "cell_00" / "000000.png"
    record = view.fetch_to(FRAME, scratch)
    assert scratch.read_bytes() == FRAME_BYTES and stat.S_IMODE(scratch.stat().st_mode) == 0o440
    assert record == {"path": str(scratch), "size_bytes": len(FRAME_BYTES), "sha256": streamed.rows[FRAME]["sha256"]}
    assert sorted(path.name for path in scratch.parent.iterdir()) == ["000000.png"]  # no partial left
    with pytest.raises(views.ProviderOutputMemberViewError, match="^provider_output_member_view_destination_exists$"):
        view.fetch_to(FRAME, scratch)
    video = tmp_path / "scratch" / "external.mp4"
    assert view.fetch_to(VIDEO, video)["size_bytes"] == 1024**2 and video.read_bytes() == bytes(1024**2)
    assert _listing(streamed.evidence) == before


def _retained_zip(path: Path, members: dict[str, bytes]) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    path.write_bytes(buffer.getvalue())
    return path


def test_plan_prints_bytes_by_disposition_for_a_retained_archive(tmp_path, capsys):
    members = {RESULT: b'{"status": "completed"}', RECEIPT: b'{"episode_id": "e0"}',
               "cell_runs/00/episodes/media/e0/policy-requests/0001.json": b'{"observation": [0]}' * 50,
               FRAME: FRAME_BYTES, VIDEO: b"\0" * 5000, "cell_runs/00/worker.log": b"step ok\n" * 10}
    archive = _retained_zip(tmp_path / "vast_provider_runtime_output.zip", members)

    assert views.main(["plan", "--archive", str(archive), "--contract",
                       "policy_canary_output_member_contract.v1"]) == 0

    plan = json.loads(capsys.readouterr().out)
    needed = [RESULT, RECEIPT]
    assert plan["schema_version"] == "provider_output_member_plan.v1"
    assert plan["contract"] == "policy_canary_output_member_contract.v1"
    assert (plan["members"], plan["files"], plan["entry_rule_refusal"]) == (6, 6, None)
    assert plan["dispositions"]["materialized"]["members"] == len(needed)
    assert plan["dispositions"]["materialized"]["bytes"] == sum(len(members[name]) for name in needed)
    assert plan["dispositions"]["remote"]["bytes"] == sum(
        len(data) for name, data in members.items() if name not in needed)
    assert plan["bytes_by_class"] == {"bulk": len(FRAME_BYTES) + 5000,
                                      "small": sum(map(len, members.values())) - len(FRAME_BYTES) - 5000}
    assert [row["path"] for row in plan["largest_materialized"]] == sorted(
        needed, key=lambda name: (-len(members[name]), name))

    # An entry the index would refuse is reported, not hidden, and the plan still runs.
    refused = _retained_zip(tmp_path / "refused.zip", {**members, "a\\b.json": b"{}"})
    assert views.main(["plan", "--archive", str(refused), "--contract",
                       "policy_canary_output_member_contract.v1"]) == 0
    assert json.loads(capsys.readouterr().out)["entry_rule_refusal"] == "provider_output_archive_path_invalid"
    (tmp_path / "not.zip").write_bytes(b"not a zip")
    assert views.main(["plan", "--archive", str(tmp_path / "not.zip"), "--contract",
                       "policy_canary_output_member_contract.v1"]) == 1
    with pytest.raises(SystemExit):
        views.main(["plan", "--archive", str(archive), "--contract", "unknown.v1"])


def test_view_reads_bytes_only_through_an_explicitly_configured_artifact_store(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_scene_object_store as scene_store

    streamed = _streamed(tmp_path)
    for name in scene_store._ARTIFACT_STORE_FILE_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(scene_store, "_artifact_object_store_client",
                        lambda: pytest.fail("the staging store's credentials may not be borrowed"))
    before = len(streamed.store.requests)
    view = views.open_member_view(streamed.evidence / FRAME, opener=streamed.store.opener)

    assert view.digest(FRAME) == streamed.rows[FRAME]["sha256"]  # digests need no store
    with pytest.raises(views.ProviderOutputMemberViewError,
                       match="^provider_output_member_view_artifact_store_not_configured$"):
        view.read_member(FRAME, maximum_bytes=10**6)
    assert len(streamed.store.requests) == before
