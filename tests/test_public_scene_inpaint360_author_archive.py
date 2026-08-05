from __future__ import annotations

import hashlib
import io
import re
import struct
import urllib.parse
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_inpaint360_author_archive as archive
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, content_range: str, filename: str, modified: str):
        super().__init__(payload)
        self.status = 206
        self.headers = {
            "Content-Range": content_range,
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Last-Modified": modified,
        }

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _zip_bytes(root: str, *, include_license_name: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{root}/", b"")
        for index, scene in enumerate(archive.EXPECTED_SCENES):
            bundle.writestr(f"{root}/{scene}/", b"")
            bundle.writestr(f"{root}/{scene}/images/frame_{index:02d}.png", scene.encode())
            bundle.writestr(f"{root}/{scene}/sparse/0/cameras.bin", bytes([index]))
        if include_license_name:
            bundle.writestr(f"{root}/LICENSE.txt", "not reviewed")
    return output.getvalue()


def _spec(*, role: str, file_id: str, filename: str, root: str, payload: bytes) -> archive.ArchiveSpec:
    eocd = payload.rfind(b"PK\x05\x06")
    values = struct.unpack_from("<4s4H2IH", payload, eocd)
    count, size, offset = values[4], values[5], values[6]
    directory = payload[offset : offset + size]
    return archive.ArchiveSpec(
        role=role,
        file_id=file_id,
        filename=filename,
        root=root,
        size_bytes=len(payload),
        last_modified="fixed",
        entry_count=count,
        central_directory_offset=offset,
        central_directory_size=size,
        central_directory_sha256="sha256:" + hashlib.sha256(directory).hexdigest(),
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_license_name: bool = False,
) -> tuple[Path, dict[str, bytes]]:
    source = tmp_path / "Inpaint360GS"
    source.mkdir()
    (source / "README.md").write_text(
        "Dataset\nbash scripts/download_inpaint360gs_dataset.sh\n", encoding="utf-8"
    )
    (source / "LICENSE.txt").write_text("Apache License 2.0", encoding="utf-8")
    payloads = {
        "author": _zip_bytes("inpaint360", include_license_name=include_license_name),
        "results": _zip_bytes("inpaint360gs", include_license_name=include_license_name),
    }
    specs = (
        _spec(
            role="author_input",
            file_id="author",
            filename="inpaint360.zip",
            root="inpaint360",
            payload=payloads["author"],
        ),
        _spec(
            role="published_evaluation_output",
            file_id="results",
            filename="inpaint360gs.zip",
            root="inpaint360gs",
            payload=payloads["results"],
        ),
    )
    monkeypatch.setattr(archive, "ARCHIVES", specs)
    monkeypatch.setattr(
        archive,
        "_git",
        lambda _repo, *args: {
            ("rev-parse", "HEAD"): archive.SOURCE_COMMIT,
            ("rev-parse", "HEAD^{tree}"): archive.SOURCE_TREE,
            ("status", "--porcelain"): "",
        }[args],
    )
    return source, payloads


def _urlopen_for(payloads: dict[str, bytes]):
    def urlopen(request, timeout=60):  # noqa: ANN001, ARG001 - urllib-compatible fake
        file_id = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["id"][0]
        value = payloads[file_id]
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match
        start, end = (int(item) for item in match.groups())
        spec = next(item for item in archive.ARCHIVES if item.file_id == file_id)
        return _Response(
            value[start : end + 1],
            content_range=f"bytes {start}-{end}/{len(value)}",
            filename=spec.filename,
            modified=spec.last_modified,
        )

    return urlopen


def test_probe_binds_default_author_scene_without_bulk_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, payloads = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "receipt.json"
    receipt = archive.materialize_author_archive_probe(
        repo_root=source,
        source_root=source,
        output_path=output,
        generated_at="2026-08-04T00:00:00+00:00",
        urlopen=_urlopen_for(payloads),
    )
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["inpaint360_author_dataset_license_authority_missing"]
    assert receipt["default_author_scene"] == "doppelherz"
    assert receipt["execution"]["gpu_allocated"] is False
    assert receipt["execution"]["publisher_archive_bytes_fully_downloaded"] is False
    assert receipt["execution"]["selected_scene_total_range_bytes"] < sum(
        len(value) for value in payloads.values()
    )
    assert all(
        item["selected_author_scene"]["scene"] == "doppelherz"
        for item in receipt["archives"]
    )
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert receipt["replay_command"][-2:] == [
        "--generated-at",
        "2026-08-04T00:00:00+00:00",
    ]
    assert output.is_file()


def test_license_named_archive_entry_cannot_self_clear_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, payloads = _fixture(tmp_path, monkeypatch, include_license_name=True)
    receipt = archive.materialize_author_archive_probe(
        repo_root=source,
        source_root=source,
        output_path=tmp_path / "receipt.json",
        urlopen=_urlopen_for(payloads),
    )
    assert receipt["status"] == "blocked"
    assert receipt["rights"]["author_dataset_license_authority_established"] is False
    assert receipt["rights"]["archive_license_or_terms_filename_candidates"]


def test_probe_rejects_server_ignoring_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, payloads = _fixture(tmp_path, monkeypatch)

    def ignored_range(request, timeout=60):  # noqa: ANN001, ARG001
        file_id = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["id"][0]
        spec = next(item for item in archive.ARCHIVES if item.file_id == file_id)
        return _Response(
            payloads[file_id],
            content_range=f"bytes 0-{len(payloads[file_id]) - 1}/{len(payloads[file_id])}",
            filename=spec.filename,
            modified=spec.last_modified,
        )

    with pytest.raises(
        archive.Inpaint360AuthorArchiveError,
        match="inpaint360_archive_range_response_invalid",
    ):
        archive.materialize_author_archive_probe(
            repo_root=source,
            source_root=source,
            output_path=tmp_path / "receipt.json",
            urlopen=ignored_range,
        )


def test_probe_rejects_changed_central_directory_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, payloads = _fixture(tmp_path, monkeypatch)
    specs = list(archive.ARCHIVES)
    specs[0] = archive.ArchiveSpec(
        **{
            **specs[0].__dict__,
            "central_directory_sha256": "sha256:" + "0" * 64,
        }
    )
    monkeypatch.setattr(archive, "ARCHIVES", tuple(specs))
    with pytest.raises(
        archive.Inpaint360AuthorArchiveError,
        match="inpaint360_archive_directory_digest_mismatch",
    ):
        archive.materialize_author_archive_probe(
            repo_root=source,
            source_root=source,
            output_path=tmp_path / "receipt.json",
            urlopen=_urlopen_for(payloads),
        )


def test_probe_rejects_dirty_materializer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, payloads = _fixture(tmp_path, monkeypatch)
    original = archive._git

    def dirty_git(repo: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return " M run_seg.sh"
        return original(repo, *args)

    monkeypatch.setattr(archive, "_git", dirty_git)
    with pytest.raises(
        archive.Inpaint360AuthorArchiveError, match="inpaint360_materializer_source_dirty"
    ):
        archive.materialize_author_archive_probe(
            repo_root=source,
            source_root=source,
            output_path=tmp_path / "receipt.json",
            urlopen=_urlopen_for(payloads),
        )
