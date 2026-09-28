from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from blueprint_pipeline.native_g1_packet_stream import receive_packet_stream, send_packet_stream
from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
from tests.test_native_task_arena_bundle import _packet


def test_packet_stream_roundtrip_is_receipt_exact_without_appledouble(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="g1-stream")
    _, original, rows = verify_native_task_arena_packet(packet)
    stream = io.BytesIO()
    sent = send_packet_stream(packet, stream)
    assert sent["packet_receipt_digest"] == original["receipt_digest"]
    with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode="r:") as archive:
        assert {member.name for member in archive} == {row["relative_path"] for row in rows}
        assert all(not member.name.startswith("._") for member in archive.getmembers())
    output = tmp_path / "received"
    result = receive_packet_stream(
        output,
        io.BytesIO(stream.getvalue()),
        expected_receipt_digest=original["receipt_digest"],
    )
    assert result["status"] == "verified_for_private_staging_not_executed"
    assert result["packet_receipt_digest"] == original["receipt_digest"]
    assert verify_native_task_arena_packet(output)[1] == original


def test_packet_stream_rejects_wrong_receipt_without_promoting(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="g1-mismatch")
    stream = io.BytesIO()
    send_packet_stream(packet, stream)
    output = tmp_path / "received"
    with pytest.raises(ValueError, match="g1_packet_stream_receipt_mismatch"):
        receive_packet_stream(
            output,
            io.BytesIO(stream.getvalue()),
            expected_receipt_digest="sha256:" + "0" * 64,
        )
    assert not output.exists()


def test_packet_stream_rejects_links_and_appledouble(tmp_path: Path) -> None:
    for name, link in (
        ("assets/link", True),
        ("._packet.json", False),
        ("../escape", False),
    ):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:") as archive:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            else:
                member.size = 1
            archive.addfile(member, None if link else io.BytesIO(b"x"))
        output = tmp_path / name.replace("/", "_").replace(".", "dot")
        with pytest.raises(ValueError, match="g1_packet_stream_member_invalid"):
            receive_packet_stream(
                output,
                io.BytesIO(stream.getvalue()),
                expected_receipt_digest="sha256:" + "0" * 64,
            )
        assert not output.exists()
