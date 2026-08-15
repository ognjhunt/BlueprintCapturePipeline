from __future__ import annotations

import hashlib
import io
import json
import os
import pwd
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_host_input_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_with_members(members: dict[str, bytes]) -> io.BytesIO:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in members.items():
            archive.writestr(intake._zip_entry(name), content)
    stream.seek(0)
    return stream


def _valid_packet_archive(
    *,
    rights_schema: str = intake.RIGHTS_RECEIPT_SCHEMA,
    agent_accepted_terms: bool = False,
    source_commit_sha: str | None = None,
) -> io.BytesIO:
    collision = b"#usda 1.0\ndef Xform \"Scene\" {}\n"
    registration = b'{"transform": [1, 0, 0, 1]}\n'
    collision_digest = "sha256:" + hashlib.sha256(collision).hexdigest()
    registration_digest = "sha256:" + hashlib.sha256(registration).hexdigest()
    rights = (
        json.dumps(
            {
                "schema_version": rights_schema,
                "reviewer_status": "approved_for_declared_use",
                "agent_accepted_terms": agent_accepted_terms,
                "authorized_source_sha256": [
                    collision_digest,
                    registration_digest,
                ],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    rights_digest = "sha256:" + hashlib.sha256(rights).hexdigest()
    packet = {
        "schema_version": intake.PACKET_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "scene_id": "new-scene-001",
        "packet_id": "new-scene-001-public-source-v1",
        "source_commit_sha": source_commit_sha or intake._verified_checkout_head(),
        "rights_receipts": [
            {
                "receipt_id": "scene-rights",
                "relative_path": "rights/00-rights.json",
                "sha256": rights_digest,
                "size_bytes": len(rights),
            }
        ],
        "files": [
            {
                "role": "collision_usd",
                "task_id": None,
                "rights_receipt_ids": ["scene-rights"],
                "relative_path": "inputs/scene.usd",
                "sha256": collision_digest,
                "size_bytes": len(collision),
            },
            {
                "role": "shared_frame_registration",
                "task_id": None,
                "rights_receipt_ids": ["scene-rights"],
                "relative_path": "inputs/shared-frame.json",
                "sha256": registration_digest,
                "size_bytes": len(registration),
            },
        ],
        "claim_ceiling": "rights_bound_public_scene_source_bytes_only",
        "provider_mutation_performed": False,
        "paid_resource_used": False,
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    return _archive_with_members(
        {
            "packet.json": (json.dumps(packet, sort_keys=True) + "\n").encode(),
            "rights/00-rights.json": rights,
            "inputs/scene.usd": collision,
            "inputs/shared-frame.json": registration,
        }
    )


def _local_limit_request(tmp_path: Path, sizes: tuple[int, int, int]) -> Path:
    paths = [tmp_path / "rights.json", tmp_path / "scene.usd", tmp_path / "frame.json"]
    for path, size in zip(paths, sizes):
        path.write_bytes(b"x" * size)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": intake.REQUEST_SCHEMA,
                "scene_id": "new-scene-001",
                "packet_id": "new-scene-001-public-source-v1",
                "source_commit_sha": intake._verified_checkout_head(),
                "rights_receipts": [
                    {
                        "receipt_id": "scene-rights",
                        "path": str(paths[0]),
                        "sha256": _sha(paths[0]),
                    }
                ],
                "files": [
                    {
                        "role": "collision_usd",
                        "path": str(paths[1]),
                        "sha256": _sha(paths[1]),
                    },
                    {
                        "role": "shared_frame_registration",
                        "path": str(paths[2]),
                        "sha256": _sha(paths[2]),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return request


@pytest.mark.parametrize(
    ("sizes", "constant", "limit", "blocker"),
    [
        (
            (3, 1, 1),
            "MAX_ARCHIVE_MEMBER_BYTES",
            2,
            "local_input_member_size_exceeds_limit",
        ),
        (
            (1, 2, 2),
            "MAX_ARCHIVE_UNCOMPRESSED_BYTES",
            4,
            "local_input_aggregate_size_exceeds_limit",
        ),
        (
            (1, 1, 1),
            "MAX_ARCHIVE_MEMBERS",
            3,
            "local_input_member_count_exceeds_limit",
        ),
    ],
)
def test_local_limits_fail_before_any_source_read(
    sizes: tuple[int, int, int],
    constant: str,
    limit: int,
    blocker: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _local_limit_request(tmp_path, sizes)
    monkeypatch.setattr(intake, constant, limit)
    monkeypatch.setattr(
        intake,
        "_sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("source hashed before limits")),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(AssertionError("source read before limits")),
    )
    with pytest.raises(intake.PublicSceneHostInputError, match=blocker):
        intake._load_request(request)


def test_archive_rejects_oversized_member_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intake, "MAX_ARCHIVE_MEMBER_BYTES", 4)
    with _archive_with_members({"packet.json": b"12345"}) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError,
                match="packet_archive_member_size_exceeds_limit",
            ):
                intake._validated_archive(archive)


def test_archive_rejects_oversized_aggregate_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intake, "MAX_ARCHIVE_MEMBER_BYTES", 4)
    monkeypatch.setattr(intake, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 5)
    with _archive_with_members(
        {"packet.json": b"123", "inputs/scene.usd": b"456"}
    ) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError,
                match="packet_archive_uncompressed_size_exceeds_limit",
            ):
                intake._validated_archive(archive)


@pytest.mark.parametrize("member_type", [stat.S_IFLNK, stat.S_IFCHR, stat.S_IFIFO])
def test_archive_rejects_nonregular_member_before_read(member_type: int) -> None:
    stream = io.BytesIO()
    entry = zipfile.ZipInfo("packet.json")
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = (member_type | 0o440) << 16
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(entry, b"not-a-regular-file")
    stream.seek(0)
    with zipfile.ZipFile(stream, "r") as archive:
        with pytest.raises(
            intake.PublicSceneHostInputError, match="packet_archive_member_unsafe"
        ):
            intake._validated_archive(archive)


def test_archive_rejects_member_count_compressed_size_and_ratio_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _archive_with_members({"packet.json": b"123", "inputs/a": b"456"}) as stream:
        monkeypatch.setattr(intake, "MAX_ARCHIVE_MEMBERS", 1)
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError, match="packet_archive_members_invalid"
            ):
                intake._validated_archive(archive)

    monkeypatch.setattr(intake, "MAX_ARCHIVE_MEMBERS", 16)
    monkeypatch.setattr(intake, "MAX_ARCHIVE_COMPRESSED_BYTES", 1)
    with _archive_with_members({"packet.json": b"123"}) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError,
                match="packet_archive_compressed_size_exceeds_limit",
            ):
                intake._validated_archive(archive)

    monkeypatch.setattr(intake, "MAX_ARCHIVE_COMPRESSED_BYTES", 1024)
    monkeypatch.setattr(intake, "MAX_ARCHIVE_COMPRESSION_RATIO", 1.0)
    with _archive_with_members({"packet.json": b"A" * 100}) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError,
                match="packet_archive_compression_ratio_exceeds_limit",
            ):
                intake._validated_archive(archive)


def test_archive_rejects_input_bound_before_zip_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intake, "MAX_ARCHIVE_INPUT_BYTES", 4)
    monkeypatch.setattr(intake, "PRODUCTION_ROOTS", (tmp_path,))
    with pytest.raises(
        intake.PublicSceneHostInputError,
        match="packet_archive_input_size_exceeds_limit",
    ):
        intake.install_packet_archive(io.BytesIO(b"12345"), destination_root=tmp_path)


@pytest.mark.parametrize(
    ("rights_schema", "agent_accepted_terms"),
    [("unexpected_rights.v1", False), (intake.RIGHTS_RECEIPT_SCHEMA, True)],
)
def test_installer_rejects_untrusted_rights_receipt_in_archive(
    rights_schema: str, agent_accepted_terms: bool
) -> None:
    with _valid_packet_archive(
        rights_schema=rights_schema,
        agent_accepted_terms=agent_accepted_terms,
    ) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError, match="packet_archive_rights_invalid"
            ):
                intake._validated_archive(archive)


def test_installer_rejects_packet_for_different_checkout() -> None:
    with _valid_packet_archive(source_commit_sha="0" * 40) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            with pytest.raises(
                intake.PublicSceneHostInputError, match="source_commit_sha_mismatch"
            ):
                intake._validated_archive(archive)


def test_service_readback_timeout_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def timeout(*_args: object, **kwargs: object) -> None:
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(cmd="runuser", timeout=int(kwargs["timeout"]))

    monkeypatch.setattr(intake.subprocess, "run", timeout)
    with pytest.raises(
        intake.PublicSceneHostInputError,
        match="service_readback_failed:input.usd",
    ):
        intake._consumer_digest(
            tmp_path / "input.usd", account="blueprint", uid=os.getuid() + 1
        )
    assert observed["timeout"] == intake.SERVICE_READBACK_TIMEOUT_SECONDS == 30


@pytest.mark.parametrize(
    "field",
    [
        "packet_digest",
        "authoritative_request_digest",
        "scene_id",
        "packet_id",
        "source_commit_sha",
        "destination_root",
        "service_account",
    ],
)
def test_upload_rejects_self_digested_receipt_with_wrong_binding(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "scene_id": "new-scene-001",
        "packet_id": "new-scene-001-public-source-v1",
        "source_commit_sha": intake._verified_checkout_head(),
    }
    destination = tmp_path / "inputs"
    monkeypatch.setattr(intake, "PRODUCTION_ROOTS", (destination,))
    monkeypatch.setattr(
        intake,
        "_archive_for_request",
        lambda _request: (io.BytesIO(b"packet"), packet),
    )
    receipt = {
        "schema_version": intake.RECEIPT_SCHEMA,
        "status": "installed",
        "packet_digest": packet["packet_digest"],
        "authoritative_request_digest": packet["packet_digest"],
        "scene_id": packet["scene_id"],
        "packet_id": packet["packet_id"],
        "source_commit_sha": packet["source_commit_sha"],
        "destination_root": str(destination / packet["packet_id"]),
        "service_account": intake.DEFAULT_SERVICE_ACCOUNT,
    }
    receipt[field] = "wrong"
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    completed = subprocess.CompletedProcess(
        args=["ssh"],
        returncode=0,
        stdout=json.dumps(receipt).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(intake.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(
        intake.PublicSceneHostInputError,
        match="host_input_upload_receipt_binding_mismatch",
    ):
        intake.upload_packet(
            request_path=tmp_path / "request.json",
            host="paperclip-prod-01",
            destination_root=destination,
        )


@pytest.mark.parametrize("failure", ["timeout", "oserror"])
def test_upload_transport_is_bounded_and_typed(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = {
        "packet_digest": "sha256:" + "a" * 64,
        "scene_id": "new-scene-001",
        "packet_id": "new-scene-001-public-source-v1",
        "source_commit_sha": intake._verified_checkout_head(),
    }
    destination = tmp_path / "inputs"
    monkeypatch.setattr(intake, "PRODUCTION_ROOTS", (destination,))
    monkeypatch.setattr(
        intake,
        "_archive_for_request",
        lambda _request: (io.BytesIO(b"packet"), packet),
    )
    observed: dict[str, object] = {}

    def fail(argv: list[str], **kwargs: object) -> None:
        observed["argv"] = argv
        observed.update(kwargs)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, int(kwargs["timeout"]))
        raise OSError("simulated ssh launch failure")

    monkeypatch.setattr(intake.subprocess, "run", fail)
    with pytest.raises(
        intake.PublicSceneHostInputError,
        match="host_input_upload_failed:transport",
    ):
        intake.upload_packet(
            request_path=tmp_path / "request.json",
            host="paperclip-prod-01",
            destination_root=destination,
        )
    assert observed["timeout"] == intake.HOST_UPLOAD_TIMEOUT_SECONDS == 1800
    assert "ConnectTimeout=30" in observed["argv"]


def test_cli_installs_rights_bound_scene_inputs_for_service_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "client-source"
    source.mkdir()
    collision = source / "scene_collision.usd"
    collision.write_bytes(b"#usda 1.0\ndef Xform \"Scene\" {}\n")
    registration = source / "shared_frame.json"
    registration.write_text('{"transform": [1, 0, 0, 1]}\n', encoding="utf-8")
    support_a = source / "task-a.json"
    support_a.write_text('{"task_id": "task-a"}\n', encoding="utf-8")
    support_b = source / "task-b.png"
    support_b.write_bytes(b"rights-cleared-lossless-frame")
    rights = source / "rights.json"
    rights.write_text(
        json.dumps(
            {
                "schema_version": "public_scene_rights_authority.v1",
                "reviewer_status": "approved_for_declared_use",
                "declared_use_scope": "internal_noncommercial_evaluation",
                "agent_accepted_terms": False,
                "authorized_source_sha256": [
                    _sha(collision),
                    _sha(registration),
                    _sha(support_a),
                    _sha(support_b),
                ],
            }
        ),
        encoding="utf-8",
    )
    request = source / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": intake.REQUEST_SCHEMA,
                "adp_item": "ADP-009B",
                "scene_id": "new-scene-001",
                "packet_id": "new-scene-001-public-source-v1",
                "source_commit_sha": intake._verified_checkout_head(),
                "rights_receipts": [
                    {
                        "receipt_id": "scene-rights",
                        "path": str(rights),
                        "sha256": _sha(rights),
                    }
                ],
                "files": [
                    {
                        "role": "collision_usd",
                        "path": str(collision),
                        "sha256": _sha(collision),
                        "rights_receipt_ids": ["scene-rights"],
                    },
                    {
                        "role": "shared_frame_registration",
                        "path": str(registration),
                        "sha256": _sha(registration),
                        "rights_receipt_ids": ["scene-rights"],
                    },
                    {
                        "role": "task_support",
                        "task_id": "task-a",
                        "path": str(support_a),
                        "sha256": _sha(support_a),
                        "rights_receipt_ids": ["scene-rights"],
                    },
                    {
                        "role": "task_support",
                        "task_id": "task-b",
                        "path": str(support_b),
                        "sha256": _sha(support_b),
                        "rights_receipt_ids": ["scene-rights"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "var-lib-inputs"
    monkeypatch.setattr(intake, "PRODUCTION_ROOTS", (destination,))
    account = pwd.getpwuid(os.getuid())

    assert (
        intake.main(
            [
                "stage",
                "--request",
                str(request),
                "--destination-root",
                str(destination),
                "--service-account",
                account.pw_name,
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    installed = destination / "new-scene-001-public-source-v1"
    assert receipt["schema_version"] == intake.RECEIPT_SCHEMA
    assert receipt["status"] == "installed"
    assert receipt["scene_id"] == "new-scene-001"
    assert receipt["service_readable"] is True
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_used"] is False
    assert receipt["secret_pattern_scan_scope"] == "bounded_patterns_only"
    assert receipt["raw_secret_values_recorded_in_receipt"] is False
    assert "raw_secret_values_recorded" not in receipt
    assert "secrets_retained" not in receipt
    assert receipt["authoritative_request_digest"] == receipt["packet_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert (installed / "inputs/scene_collision.usd").read_bytes() == collision.read_bytes()
    assert (installed / "inputs/shared_frame.json").read_bytes() == registration.read_bytes()
    assert len([row for row in receipt["files"] if row.get("role") == "task_support"]) == 2
    assert str(source) not in json.dumps(receipt)
    assert all(path.stat().st_mode & 0o777 == 0o440 for path in installed.rglob("*") if path.is_file())

    with pytest.raises(intake.PublicSceneHostInputError, match="destination_already_exists"):
        intake.main(
            [
                "stage",
                "--request",
                str(request),
                "--destination-root",
                str(destination),
                "--service-account",
                account.pw_name,
            ]
        )
    with pytest.raises(SystemExit) as help_exit:
        intake.main(["--help"])
    assert help_exit.value.code == 0
