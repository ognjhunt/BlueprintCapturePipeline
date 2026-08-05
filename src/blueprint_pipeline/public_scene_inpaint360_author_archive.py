"""Probe the exact Inpaint360GS author-data archives without bulk download."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SOURCE_REPOSITORY = "https://github.com/dfki-av/Inpaint360GS"
SOURCE_COMMIT = "d54c893285c6cb27788e05cce607e7d3cca6388a"
SOURCE_TREE = "671626f4825cbf3d7c1ca37cc97a153d45e49b1c"
DEFAULT_AUTHOR_SCENE = "doppelherz"
EXPECTED_SCENES = (
    "bag",
    "car",
    "cone_red",
    "cone_yellow",
    "cube",
    "doppelherz",
    "fruits",
    "garden_toys",
    "redbull",
    "toys",
    "truck",
)
_ALLOWED_DOWNLOAD_HOST = "drive.usercontent.google.com"
_TAIL_BYTES = 131_072
_LICENSE_MARKERS = ("license", "licence", "terms", "copyright", "readme")


@dataclass(frozen=True)
class ArchiveSpec:
    role: str
    file_id: str
    filename: str
    root: str
    size_bytes: int
    last_modified: str
    entry_count: int
    central_directory_offset: int
    central_directory_size: int
    central_directory_sha256: str


ARCHIVES = (
    ArchiveSpec(
        role="author_input",
        file_id="1YLpop12JRbzglJfx0FUFUZ2GLaBfZX_x",
        filename="inpaint360.zip",
        root="inpaint360",
        size_bytes=8_020_567_705,
        last_modified="Fri, 16 Jan 2026 13:21:29 GMT",
        entry_count=6_566,
        central_directory_offset=8_019_750_855,
        central_directory_size=816_752,
        central_directory_sha256=(
            "sha256:01e983a66db22ce1d08e1415f1d7ff5db6d8fbcba68381c9cdc670b957197524"
        ),
    ),
    ArchiveSpec(
        role="published_evaluation_output",
        file_id="1SgB4grTSKhFeKp8-l1TGcjhpmtTkvx0W",
        filename="inpaint360gs.zip",
        root="inpaint360gs",
        size_bytes=4_307_622_065,
        last_modified="Fri, 16 Jan 2026 15:58:09 GMT",
        entry_count=1_751,
        central_directory_offset=4_307_411_053,
        central_directory_size=210_914,
        central_directory_sha256=(
            "sha256:5828e3c7b2709d1d1c17765ed2a44aa3e2d152c61f21249a76d00bde14bdcf52"
        ),
    ),
)


class Inpaint360AuthorArchiveError(ValueError):
    """Raised when publisher bytes do not match the frozen archive identity."""


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    compression_method: int


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _download_url(file_id: str) -> str:
    query = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return f"https://{_ALLOWED_DOWNLOAD_HOST}/download?{query}"


def _header(response: BinaryIO, name: str) -> str:
    headers = getattr(response, "headers", {})
    value = headers.get(name) if hasattr(headers, "get") else None
    return str(value or "")


def _get_range(
    spec: ArchiveSpec,
    start: int,
    end: int,
    *,
    urlopen: Callable[..., BinaryIO],
) -> tuple[bytes, Mapping[str, str]]:
    if start < 0 or end < start or end >= spec.size_bytes:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_range_invalid")
    url = _download_url(spec.file_id)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_DOWNLOAD_HOST:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_host_not_allowlisted")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=60) as response:  # nosec B310 - fixed HTTPS publisher host
        payload = response.read()
        headers = {
            "content_range": _header(response, "Content-Range"),
            "content_disposition": _header(response, "Content-Disposition"),
            "last_modified": _header(response, "Last-Modified"),
        }
        status = int(getattr(response, "status", 206))
    expected_range = f"bytes {start}-{end}/{spec.size_bytes}"
    if status != 206 or headers["content_range"] != expected_range:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_range_response_invalid")
    if len(payload) != end - start + 1:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_range_size_mismatch")
    return payload, headers


def _zip64_values(extra: bytes, values: list[int]) -> tuple[int, int, int]:
    uncompressed, compressed, offset = values
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, cursor)
        data = extra[cursor + 4 : cursor + 4 + size]
        cursor += 4 + size
        if header_id != 1:
            continue
        data_cursor = 0
        if uncompressed == 0xFFFFFFFF:
            uncompressed = struct.unpack_from("<Q", data, data_cursor)[0]
            data_cursor += 8
        if compressed == 0xFFFFFFFF:
            compressed = struct.unpack_from("<Q", data, data_cursor)[0]
            data_cursor += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", data, data_cursor)[0]
        break
    return uncompressed, compressed, offset


def _parse_central_directory(raw: bytes) -> list[ZipEntry]:
    cursor = 0
    entries: list[ZipEntry] = []
    while cursor < len(raw):
        if cursor + 46 > len(raw) or raw[cursor : cursor + 4] != b"PK\x01\x02":
            raise Inpaint360AuthorArchiveError("inpaint360_central_directory_invalid")
        values = struct.unpack_from("<4s6H3I5H2I", raw, cursor)
        name_size, extra_size, comment_size = values[10:13]
        name_start = cursor + 46
        extra_start = name_start + name_size
        extra_end = extra_start + extra_size
        name = raw[name_start:extra_start].decode("utf-8")
        uncompressed, compressed, offset = _zip64_values(
            raw[extra_start:extra_end], [values[9], values[8], values[16]]
        )
        if 0xFFFFFFFF in (uncompressed, compressed, offset):
            raise Inpaint360AuthorArchiveError("inpaint360_zip64_entry_incomplete")
        entries.append(
            ZipEntry(
                name=name,
                compressed_size=compressed,
                uncompressed_size=uncompressed,
                local_header_offset=offset,
                compression_method=values[4],
            )
        )
        cursor = extra_end + comment_size
    return entries


def _parse_directory_location(tail: bytes, *, tail_offset: int) -> tuple[int, int, int]:
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise Inpaint360AuthorArchiveError("inpaint360_zip_eocd_missing")
    values = struct.unpack_from("<4s4H2IH", tail, eocd)
    entry_count, directory_size, directory_offset = values[4], values[5], values[6]
    if 0xFFFFFFFF not in (directory_size, directory_offset) and entry_count != 0xFFFF:
        return entry_count, directory_offset, directory_size
    locator = tail.rfind(b"PK\x06\x07", 0, eocd)
    if locator < 0:
        raise Inpaint360AuthorArchiveError("inpaint360_zip64_locator_missing")
    zip64_offset = struct.unpack_from("<4sIQI", tail, locator)[2]
    relative = zip64_offset - tail_offset
    if relative < 0 or relative + 56 > len(tail):
        raise Inpaint360AuthorArchiveError("inpaint360_zip64_eocd_outside_probe")
    zip64 = struct.unpack_from("<4sQ2H2I4Q", tail, relative)
    if zip64[0] != b"PK\x06\x06":
        raise Inpaint360AuthorArchiveError("inpaint360_zip64_eocd_invalid")
    return zip64[7], zip64[9], zip64[8]


def _scene_rows(entries: list[ZipEntry], root: str) -> dict[str, list[ZipEntry]]:
    scenes: dict[str, list[ZipEntry]] = defaultdict(list)
    for entry in entries:
        parts = entry.name.split("/")
        if len(parts) >= 3 and parts[0] == root and parts[1]:
            scenes[parts[1]].append(entry)
    return dict(scenes)


def _scene_summaries(
    entries: list[ZipEntry], *, root: str, central_offset: int
) -> list[dict[str, Any]]:
    scenes = _scene_rows(entries, root)
    offsets = sorted(
        (min(row.local_header_offset for row in rows), scene)
        for scene, rows in scenes.items()
    )
    boundaries = {
        scene: (start, offsets[index + 1][0] - 1 if index + 1 < len(offsets) else central_offset - 1)
        for index, (start, scene) in enumerate(offsets)
    }
    summaries: list[dict[str, Any]] = []
    for scene in sorted(scenes):
        rows = scenes[scene]
        start, end = boundaries[scene]
        summaries.append(
            {
                "scene": scene,
                "entry_count": len(rows),
                "compressed_payload_bytes": sum(row.compressed_size for row in rows),
                "uncompressed_payload_bytes": sum(row.uncompressed_size for row in rows),
                "contiguous_archive_range": {
                    "start": start,
                    "end_inclusive": end,
                    "size_bytes": end - start + 1,
                },
                "entry_manifest_digest": canonical_digest(
                    {
                        "entries": [
                            {
                                "name": row.name,
                                "compressed_size": row.compressed_size,
                                "uncompressed_size": row.uncompressed_size,
                                "local_header_offset": row.local_header_offset,
                                "compression_method": row.compression_method,
                            }
                            for row in sorted(rows, key=lambda item: item.name)
                        ]
                    }
                ),
            }
        )
    return summaries


def _probe_archive(
    spec: ArchiveSpec, *, urlopen: Callable[..., BinaryIO]
) -> dict[str, Any]:
    _, identity_headers = _get_range(spec, 0, 0, urlopen=urlopen)
    if spec.filename not in identity_headers["content_disposition"]:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_filename_mismatch")
    if identity_headers["last_modified"] != spec.last_modified:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_last_modified_mismatch")
    tail_start = max(0, spec.size_bytes - _TAIL_BYTES)
    tail, _ = _get_range(spec, tail_start, spec.size_bytes - 1, urlopen=urlopen)
    entry_count, directory_offset, directory_size = _parse_directory_location(
        tail, tail_offset=tail_start
    )
    if (
        entry_count != spec.entry_count
        or directory_offset != spec.central_directory_offset
        or directory_size != spec.central_directory_size
    ):
        raise Inpaint360AuthorArchiveError("inpaint360_archive_directory_identity_mismatch")
    directory, _ = _get_range(
        spec,
        directory_offset,
        directory_offset + directory_size - 1,
        urlopen=urlopen,
    )
    if _sha256_bytes(directory) != spec.central_directory_sha256:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_directory_digest_mismatch")
    entries = _parse_central_directory(directory)
    if len(entries) != spec.entry_count:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_entry_count_mismatch")
    summaries = _scene_summaries(
        entries, root=spec.root, central_offset=spec.central_directory_offset
    )
    if tuple(sorted(row["scene"] for row in summaries)) != EXPECTED_SCENES:
        raise Inpaint360AuthorArchiveError("inpaint360_archive_scene_set_mismatch")
    license_candidates = sorted(
        entry.name
        for entry in entries
        if any(marker in entry.name.lower() for marker in _LICENSE_MARKERS)
    )
    selected = next(row for row in summaries if row["scene"] == DEFAULT_AUTHOR_SCENE)
    return {
        "role": spec.role,
        "publisher": "Google Drive link declared by the official source repository",
        "file_id": spec.file_id,
        "filename": spec.filename,
        "size_bytes": spec.size_bytes,
        "last_modified": spec.last_modified,
        "whole_archive_sha256": None,
        "whole_archive_materialized": False,
        "central_directory": {
            "offset": directory_offset,
            "size_bytes": directory_size,
            "sha256": spec.central_directory_sha256,
            "entry_count": entry_count,
        },
        "scene_summaries": summaries,
        "selected_author_scene": selected,
        "license_or_terms_filename_candidates": license_candidates,
        "raw_bytes_inspected": 1 + len(tail) + len(directory),
    }


def materialize_author_archive_probe(
    *,
    repo_root: str | Path | None = None,
    source_root: str | Path,
    output_path: str | Path,
    generated_at: str | None = None,
    urlopen: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Bind the exact default author case and retain the unresolved rights gate."""

    repo = Path(repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    execution_commit = _git(repo, "rev-parse", "HEAD")
    execution_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if _git(repo, "status", "--porcelain"):
        raise Inpaint360AuthorArchiveError("inpaint360_materializer_source_dirty")
    source = Path(source_root).expanduser().resolve()
    if (
        _git(source, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(source, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(source, "status", "--porcelain")
    ):
        raise Inpaint360AuthorArchiveError("inpaint360_source_identity_mismatch")
    readme = source / "README.md"
    license_path = source / "LICENSE.txt"
    if not readme.is_file() or not license_path.is_file():
        raise Inpaint360AuthorArchiveError("inpaint360_source_rights_files_missing")
    readme_text = readme.read_text(encoding="utf-8")
    if "Dataset" not in readme_text or "scripts/download_inpaint360gs_dataset.sh" not in readme_text:
        raise Inpaint360AuthorArchiveError("inpaint360_author_workflow_declaration_missing")
    archives = [_probe_archive(spec, urlopen=urlopen) for spec in ARCHIVES]
    generated = generated_at or utc_now_iso()
    receipt: dict[str, Any] = {
        "schema_version": "public_scene_inpaint360_author_archive_probe.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009A",
        "generated_at": generated,
        "status": "blocked",
        "blockers": ["inpaint360_author_dataset_license_authority_missing"],
        "materializer_source": {
            "repository_root": str(repo),
            "commit": execution_commit,
            "tree": execution_tree,
            "clean_before_write": True,
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
            "clean": True,
            "observed_checkout_path": str(source),
            "source_license": "Apache-2.0",
            "source_license_sha256": _sha256_file(license_path),
            "readme_sha256": _sha256_file(readme),
            "readme_declares_dataset_license": bool(
                re.search(r"dataset.{0,120}licen[cs]e|licen[cs]e.{0,120}dataset", readme_text, re.I | re.S)
            ),
        },
        "author_command": ["bash run_seg.sh", "bash run_remove.sh", "bash run_inpaint.sh"],
        "default_author_scene": DEFAULT_AUTHOR_SCENE,
        "archives": archives,
        "rights": {
            "author_dataset_license_authority_established": False,
            "source_code_license_not_inherited_by_dataset": True,
            "archive_license_or_terms_filename_candidates": sorted(
                {name for archive in archives for name in archive["license_or_terms_filename_candidates"]}
            ),
        },
        "execution": {
            "author_method_executed": False,
            "gpu_allocated": False,
            "publisher_archive_bytes_fully_downloaded": False,
            "range_only_retrieval_plan_bound": True,
            "selected_scene_total_range_bytes": sum(
                archive["selected_author_scene"]["contiguous_archive_range"]["size_bytes"]
                for archive in archives
            ),
        },
        "claim_ceiling": "archive_identity_and_range_plan_only_no_inpainting_result",
        "replay_command": [
            "python",
            "-m",
            "blueprint_pipeline.public_scene_inpaint360_author_archive",
            "--repo-root",
            str(repo),
            "--source-root",
            str(source),
            "--output",
            str(Path(output_path).expanduser().resolve()),
            "--generated-at",
            generated,
        ],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(output_path).expanduser().resolve(), receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    receipt = materialize_author_archive_probe(
        repo_root=args.repo_root,
        source_root=args.source_root,
        output_path=args.output,
        generated_at=args.generated_at,
    )
    print(json.dumps({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    return 2 if receipt["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
