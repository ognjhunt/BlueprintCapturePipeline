from __future__ import annotations

import hashlib
import io
import json
import os
import pwd
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
            archive.writestr(name, content)
    stream.seek(0)
    return stream


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
                "source_commit_sha": "491c3619cfba4b3024ffbc219f1270a3ef1203a6",
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
    assert receipt["secrets_retained"] is False
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
