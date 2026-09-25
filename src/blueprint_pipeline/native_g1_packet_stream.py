"""Stream a sealed task packet to the controller without local archive copies.

The sender emits only receipt-listed regular files. Python's tar writer does
not add macOS AppleDouble metadata files. The receiver rejects unsafe members,
verifies the complete packet, and promotes it atomically under its expected
receipt digest. Neither side needs enough free disk for an extra local archive.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .native_task_arena_bundle import verify_native_task_arena_packet


MAX_STREAM_BYTES = 8 * 1024**3
MAX_MEMBERS = 10000


def send_packet_stream(packet_dir: Path, stream: BinaryIO) -> dict[str, Any]:
    """Write only verified packet files to a nonseekable stream."""

    root, receipt, rows = verify_native_task_arena_packet(packet_dir)
    with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for row in rows:
            relative = row["relative_path"]
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("g1_packet_stream_file_missing_or_symlink")
            info = tarfile.TarInfo(relative)
            info.size = row["size_bytes"]
            info.mode = 0o640
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return {
        "status": "streamed_not_received",
        "packet_receipt_digest": receipt["receipt_digest"],
        "file_count": len(rows),
    }


def receive_packet_stream(
    output_dir: Path, stream: BinaryIO, *, expected_receipt_digest: str
) -> dict[str, Any]:
    """Extract safely, verify all bytes, and atomically stage the packet."""

    output = Path(output_dir)
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or output.resolve() != output
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_receipt_digest) is None
    ):
        raise ValueError("g1_packet_stream_destination_invalid")
    with tempfile.TemporaryDirectory(prefix=".g1-packet-stream-", dir=output.parent) as raw:
        temporary = Path(raw)
        members: set[str] = set()
        total = 0
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or not relative.parts
                    or ".." in relative.parts
                    or any(part.startswith("._") for part in relative.parts)
                    or member.name in members
                    or len(members) >= MAX_MEMBERS
                    or member.size < 0
                    or member.size > MAX_STREAM_BYTES
                    or total + member.size > MAX_STREAM_BYTES
                ):
                    raise ValueError("g1_packet_stream_member_invalid")
                members.add(member.name)
                total += member.size
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("g1_packet_stream_member_unreadable")
                copied = 0
                with source, target.open("xb") as destination:
                    while block := source.read(1024 * 1024):
                        copied += len(block)
                        if copied > member.size:
                            raise ValueError("g1_packet_stream_member_size_mismatch")
                        destination.write(block)
                if copied != member.size:
                    raise ValueError("g1_packet_stream_member_size_mismatch")
                target.chmod(0o640)
        _, receipt, rows = verify_native_task_arena_packet(temporary)
        if receipt["receipt_digest"] != expected_receipt_digest or len(rows) != len(members):
            raise ValueError("g1_packet_stream_receipt_mismatch")
        temporary.chmod(0o750)
        temporary.rename(output)
    return {
        "status": "verified_for_private_staging_not_executed",
        "packet_receipt_digest": receipt["receipt_digest"],
        "scene_plan_digest": receipt["arena_scene_plan_digest"],
        "file_count": len(rows),
        "size_bytes": total,
        "output_dir": str(output),
        "gpu_allocated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    send = commands.add_parser("send")
    send.add_argument("--packet", type=Path, required=True)
    receive = commands.add_parser("receive")
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--expected-receipt-digest", required=True)
    args = parser.parse_args()
    if args.command == "send":
        send_packet_stream(args.packet, sys.stdout.buffer)
        return 0
    result = receive_packet_stream(
        args.output_dir,
        sys.stdin.buffer,
        expected_receipt_digest=args.expected_receipt_digest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
