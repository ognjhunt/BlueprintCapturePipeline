from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.provider_archive import (
    ProviderArchiveError,
    extract_provider_archive,
)
from blueprint_pipeline.provider_bundle_rehearsal import (
    provider_bundle_rehearsal_blockers,
)


def test_provider_archive_preserves_unix_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        regular = zipfile.ZipInfo("lib/libcore.so.12")
        regular.create_system = 3
        regular.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(regular, b"real-shared-library")
        link = zipfile.ZipInfo("lib/libcore.so")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "libcore.so.12")

    receipt = extract_provider_archive(archive_path, tmp_path / "expanded")

    extracted = tmp_path / "expanded/lib/libcore.so"
    assert extracted.is_symlink()
    assert extracted.readlink() == Path("libcore.so.12")
    assert extracted.read_bytes() == b"real-shared-library"
    assert receipt["symlink_count"] == 1
    assert receipt["extracted_symlink_count"] == 1
    assert receipt["python_zipfile_extractall_used"] is False


@pytest.mark.parametrize(
    ("name", "payload", "mode", "error"),
    [
        ("../escape", b"x", stat.S_IFREG | 0o644, "member_path_invalid"),
        ("link", "../../escape", stat.S_IFLNK | 0o777, "target_outside_root"),
    ],
)
def test_provider_archive_rejects_escape_paths(
    tmp_path: Path, name: str, payload: bytes | str, mode: int, error: str
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = mode << 16
        archive.writestr(info, payload)
    with pytest.raises(ProviderArchiveError, match=error):
        extract_provider_archive(archive_path, tmp_path / "expanded")


def test_exact_bundle_rehearsal_is_bound_to_bundle_and_entrypoint() -> None:
    receipt = {
        "status": "passed",
        "bundle_sha256": "sha256:" + "1" * 64,
        "entrypoint_relative_path": "provider_runtime/run.sh",
        "returncode": 0,
        "gpu_runtime_started": False,
        "paid_inference_performed": False,
        "provider_mutations_performed": 0,
    }
    assert provider_bundle_rehearsal_blockers(
        receipt,
        bundle_sha256="sha256:" + "1" * 64,
        entrypoint_relative_path="provider_runtime/run.sh",
    ) == []
    assert provider_bundle_rehearsal_blockers(
        receipt,
        bundle_sha256="sha256:" + "2" * 64,
        entrypoint_relative_path="provider_runtime/run.sh",
    ) == ["exact_bundle_entrypoint_rehearsal_invalid"]
    assert provider_bundle_rehearsal_blockers(
        None,
        bundle_sha256="sha256:" + "1" * 64,
        entrypoint_relative_path="provider_runtime/run.sh",
    ) == ["exact_bundle_entrypoint_rehearsal_missing"]
