"""Safe, symlink-preserving ZIP extraction for provider runtime bundles."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = "provider_archive_extraction.v1"


class ProviderArchiveError(ValueError):
    """An archive cannot be reproduced safely and byte-faithfully."""


def _member_mode(member: zipfile.ZipInfo) -> int:
    return (member.external_attr >> 16) & 0xFFFF


def _safe_member_name(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or "\x00" in name:
        raise ProviderArchiveError("provider_archive_member_path_invalid")
    normalized = PurePosixPath(posixpath.normpath(name))
    if normalized == PurePosixPath(".") or normalized.parts != value.parts:
        raise ProviderArchiveError("provider_archive_member_path_invalid")
    return value


def inspect_provider_archive(archive_path: str | Path) -> dict[str, Any]:
    """Return the frozen entry-type inventory used by extraction receipts."""

    archive_file = Path(archive_path).expanduser().resolve()
    if not archive_file.is_file():
        raise ProviderArchiveError("provider_archive_missing")
    try:
        with zipfile.ZipFile(archive_file) as archive:
            members = archive.infolist()
            names = [_safe_member_name(member.filename) for member in members]
            if len(set(names)) != len(names):
                raise ProviderArchiveError("provider_archive_duplicate_member")
            symlink_names = {
                name
                for name, member in zip(names, members, strict=True)
                if stat.S_ISLNK(_member_mode(member))
            }
            for name in names:
                if any(parent in symlink_names for parent in name.parents):
                    raise ProviderArchiveError("provider_archive_member_beneath_symlink")
            rows = []
            for name, member in zip(names, members, strict=True):
                mode = _member_mode(member)
                if member.is_dir() or stat.S_ISDIR(mode):
                    kind = "directory"
                elif stat.S_ISLNK(mode):
                    kind = "symlink"
                    try:
                        target = archive.read(member).decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ProviderArchiveError(
                            "provider_archive_symlink_target_invalid"
                        ) from exc
                    if not target or PurePosixPath(target).is_absolute():
                        raise ProviderArchiveError(
                            "provider_archive_symlink_target_invalid"
                        )
                    resolved = posixpath.normpath(
                        posixpath.join(name.parent.as_posix(), target)
                    )
                    if resolved == ".." or resolved.startswith("../"):
                        raise ProviderArchiveError(
                            "provider_archive_symlink_target_outside_root"
                        )
                elif stat.S_IFMT(mode) in {0, stat.S_IFREG}:
                    kind = "file"
                else:
                    raise ProviderArchiveError("provider_archive_special_member_forbidden")
                rows.append(
                    {
                        "name": name.as_posix(),
                        "kind": kind,
                        "size_bytes": member.file_size,
                        "unix_mode": mode,
                    }
                )
    except zipfile.BadZipFile as exc:
        raise ProviderArchiveError("provider_archive_invalid_zip") from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "member_count": len(rows),
        "file_count": sum(row["kind"] == "file" for row in rows),
        "directory_count": sum(row["kind"] == "directory" for row in rows),
        "symlink_count": sum(row["kind"] == "symlink" for row in rows),
        "members": rows,
    }


def extract_provider_archive(
    archive_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Extract without converting Unix symlinks into tiny text files."""

    archive_file = Path(archive_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ProviderArchiveError("provider_archive_destination_not_empty")
    inventory = inspect_provider_archive(archive_file)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_file) as archive:
            by_name = {member.filename: member for member in archive.infolist()}
            rows = inventory["members"]
            for row in rows:
                if row["kind"] == "directory":
                    (output / row["name"]).mkdir(parents=True, exist_ok=True)
            for row in rows:
                if row["kind"] != "file":
                    continue
                member = by_name[row["name"]]
                target = output / row["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
                if row["unix_mode"]:
                    target.chmod(stat.S_IMODE(row["unix_mode"]))
            for row in rows:
                if row["kind"] != "symlink":
                    continue
                member = by_name[row["name"]]
                target = output / row["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                link_target = archive.read(member).decode("utf-8")
                os.symlink(link_target, target)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProviderArchiveError("provider_archive_extraction_failed") from exc
    extracted_symlinks = sum(path.is_symlink() for path in output.rglob("*"))
    if extracted_symlinks != inventory["symlink_count"]:
        raise ProviderArchiveError("provider_archive_symlink_fidelity_mismatch")
    return {
        **inventory,
        "status": "passed",
        "archive_path": str(archive_file),
        "destination": str(output),
        "extracted_symlink_count": extracted_symlinks,
        "python_zipfile_extractall_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("destination")
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)
    receipt = extract_provider_archive(args.archive, args.destination)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        Path(args.receipt).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProviderArchiveError",
    "extract_provider_archive",
    "inspect_provider_archive",
]
