"""The provider toolchain is compressed once per exact content and spliced raw."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import (
    task_evaluation_scene_configuration_toolchain_archive as archive_module,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
    validate_scene_configuration_toolchain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    build_scene_configuration_provider_bundle,
    load_scene_configuration_provider_bundle_receipt,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    TaskEvaluationSceneConfigurationStageProducerError,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)
from tests.test_build_task_evaluation_scene_configuration_toolchain import _component_packages
from tests.test_task_evaluation_scene_configuration_bundle import (  # noqa: F401 - autouse fixtures
    _ceiling_output_admission,
    _envelope,
    _repo,
    _stub_api_preparation_for_bundle_contracts,
)

A = "a" * 40
B = "b" * 40


def _published_toolchain(parent: Path, commit: str, **package_options: bool) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / "toolchains" / commit
    build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=root,
        readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test",
        component_packages=_component_packages(parent, **package_options),
    )
    return root


def _bundle(tmp_path: Path, name: str, *, toolchain: Path, commit: str, cache: Path | None) -> dict:
    source = tmp_path / f"source-{commit[:1]}"
    source.mkdir(exist_ok=True)
    envelope = source / "envelope.json"
    if not envelope.exists():
        envelope = _envelope(source, commit)
    repo = tmp_path / "repo"
    if not repo.exists():
        _repo(repo)
    return build_scene_configuration_provider_bundle(
        construction_envelope_path=envelope,
        toolchain_root=toolchain,
        repository_root=repo,
        output_root=tmp_path / name,
        expected_source_commit=commit,
        toolchain_archive_cache_root=cache,
    )


def _toolchain_members(bundle: str) -> dict[str, bytes]:
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        return {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if info.filename.startswith(archive_module.ARCHIVE_PREFIX) and not info.is_dir()
        }


@pytest.fixture(autouse=True)
def _no_ambient_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(archive_module.CACHE_ROOT_ENV, raising=False)
    monkeypatch.setattr(archive_module, "DEFAULT_CACHE_PARENT", tmp_path / "absent-inputs-volume")


def test_reused_archive_gives_the_same_bundle_and_every_toolchain_byte(tmp_path: Path) -> None:
    toolchain = _published_toolchain(tmp_path / "release-a", A)
    cache = tmp_path / "cache"

    first = _bundle(tmp_path, "first", toolchain=toolchain, commit=A, cache=cache)
    second = _bundle(tmp_path, "second", toolchain=toolchain, commit=A, cache=cache)
    uncached = _bundle(tmp_path, "uncached", toolchain=toolchain, commit=A, cache=None)

    assert first["toolchain_archive"]["reused"] is False
    assert second["toolchain_archive"]["reused"] is True
    assert uncached["toolchain_archive"]["persistent_cache"] is False
    assert first["bundle_sha256"] == second["bundle_sha256"] == uncached["bundle_sha256"]
    assert not any(path.name.startswith(".toolchain-archive") for path in (tmp_path / "uncached").iterdir())
    # The receipt still reopens against the bundle and its sealed manifest.
    receipt_path = tmp_path / "second" / "task_evaluation_scene_configuration_provider_bundle.v1.receipt.json"
    assert load_scene_configuration_provider_bundle_receipt(receipt_path)["bundle_sha256"] == second["bundle_sha256"]

    manifest = json.loads((toolchain / f"{TOOLCHAIN_SCHEMA_VERSION}.json").read_text())
    members = _toolchain_members(second["bundle_path"])
    expected = {
        archive_module.ARCHIVE_PREFIX + row["relative_path"]: row for row in manifest["files"]
    }
    manifest_member = archive_module.ARCHIVE_PREFIX + f"{TOOLCHAIN_SCHEMA_VERSION}.json"
    assert set(members) == set(expected) | {manifest_member}
    assert members[manifest_member] == (toolchain / f"{TOOLCHAIN_SCHEMA_VERSION}.json").read_bytes()
    for name, row in expected.items():
        assert "sha256:" + hashlib.sha256(members[name]).hexdigest() == row["sha256"]

    # The provider's own validation of the unpacked toolchain still passes.
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(second["bundle_path"]) as archive:
        archive.extractall(unpacked)
        modes = {info.filename.rstrip("/"): (info.external_attr >> 16) & 0o777
                 for info in archive.infolist()}
    for path in sorted(unpacked.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(modes[path.relative_to(unpacked).as_posix()] & ~0o222)  # the provider seals read-only
    try:
        validate_scene_configuration_toolchain(
            root=unpacked / "provider_runtime/toolchain", expected_source_commit=A
        )
    finally:
        for path in unpacked.rglob("*"):
            path.chmod(0o700)


def test_a_release_with_unchanged_toolchain_content_reuses_the_archive(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    release_a = _published_toolchain(tmp_path / "release-a", A)
    release_b = _published_toolchain(tmp_path / "release-b", B)

    first = _bundle(tmp_path, "a", toolchain=release_a, commit=A, cache=cache)
    second = _bundle(tmp_path, "b", toolchain=release_b, commit=B, cache=cache)

    assert first["toolchain_digest"] != second["toolchain_digest"]  # binds source_commit
    assert second["toolchain_archive"]["reused"] is True
    assert first["toolchain_archive"]["archive_key"] == second["toolchain_archive"]["archive_key"]
    first_members = _toolchain_members(first["bundle_path"])
    second_members = _toolchain_members(second["bundle_path"])
    manifest_member = archive_module.ARCHIVE_PREFIX + f"{TOOLCHAIN_SCHEMA_VERSION}.json"
    assert json.loads(second_members[manifest_member])["source_commit"] == B
    first_members.pop(manifest_member)
    second_members.pop(manifest_member)
    assert first_members == second_members


def test_changed_toolchain_content_builds_a_new_archive_and_bounds_the_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    plain = _published_toolchain(tmp_path / "plain", A)
    with_bytecode = _published_toolchain(tmp_path / "bytecode", A, bundled_bytecode=True)
    with_empty = _published_toolchain(tmp_path / "empty", A, empty_file=True)

    keys = [
        _bundle(tmp_path, name, toolchain=root, commit=A, cache=cache)["toolchain_archive"]
        for name, root in (("out-plain", plain), ("out-bytecode", with_bytecode),
                           ("out-empty", with_empty))
    ]

    assert [row["reused"] for row in keys] == [False, False, False]
    assert len({row["archive_key"] for row in keys}) == 3
    retained = sorted(path.name for path in cache.iterdir())
    assert len(retained) == archive_module.RETAINED_ENTRIES
    assert keys[-1]["archive_key"].removeprefix("sha256:") in retained


def test_a_corrupted_cached_archive_is_rebuilt_not_shipped(tmp_path: Path) -> None:
    toolchain = _published_toolchain(tmp_path / "release-a", A)
    cache = tmp_path / "cache"
    clean = _bundle(tmp_path, "clean", toolchain=toolchain, commit=A, cache=cache)
    entry = cache / clean["toolchain_archive"]["archive_key"].removeprefix("sha256:")
    fragment = entry / archive_module.FRAGMENT_NAME
    entry.chmod(0o755)
    fragment.chmod(0o644)
    data = bytearray(fragment.read_bytes())
    data[40] ^= 0xFF  # inside the first member's stored bytes; size unchanged
    fragment.write_bytes(bytes(data))

    rebuilt = _bundle(tmp_path, "rebuilt", toolchain=toolchain, commit=A, cache=cache)

    assert rebuilt["bundle_sha256"] == clean["bundle_sha256"]
    assert rebuilt["toolchain_archive"]["reused"] is False
    assert archive_module._file_sha256(entry / archive_module.FRAGMENT_NAME) == (
        clean["toolchain_archive"]["fragment_sha256"]
    )


def test_source_bytes_that_differ_from_the_inventory_are_never_archived(tmp_path: Path) -> None:
    toolchain = _published_toolchain(tmp_path / "release-a", A)
    manifest = validate_scene_configuration_toolchain(root=toolchain, expected_source_commit=A)
    tampered = json.loads(json.dumps(manifest))
    tampered["files"][0]["sha256"] = "sha256:" + "0" * 64
    cache = tmp_path / "cache"

    with pytest.raises(archive_module.TaskEvaluationSceneConfigurationToolchainArchiveError,
                       match="source_mismatch"):
        archive_module.prepare_toolchain_archive(
            toolchain_root=toolchain, manifest=tampered,
            compress_type_for=lambda path: zipfile.ZIP_DEFLATED, deflate_level=1,
            stored_suffixes=(), scratch_root=tmp_path, cache_root=cache,
        )
    assert list(cache.iterdir()) == []


def test_a_toolchain_tampered_after_publication_still_refuses_the_bundle(tmp_path: Path) -> None:
    toolchain = _published_toolchain(tmp_path / "release-a", A)
    victim = next(path for path in sorted(toolchain.rglob("run")) if path.is_file())
    victim.chmod(0o755)
    victim.write_text("#!/bin/sh\nexit 0\n")
    victim.chmod(0o555)

    with pytest.raises(TaskEvaluationSceneConfigurationStageProducerError,
                       match="toolchain_file_invalid"):
        _bundle(tmp_path, "tampered", toolchain=toolchain, commit=A, cache=tmp_path / "cache")


def test_spliced_members_keep_zip64_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Production bundles pass 4 GiB, so spliced members sit past the ZIP64
    # offset limit; a tiny limit exercises the same central-directory path.
    toolchain = _published_toolchain(tmp_path / "release-a", A)
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 64)
    receipt = _bundle(tmp_path, "zip64", toolchain=toolchain, commit=A, cache=tmp_path / "cache")
    monkeypatch.undo()

    manifest = json.loads((toolchain / f"{TOOLCHAIN_SCHEMA_VERSION}.json").read_text())
    members = _toolchain_members(receipt["bundle_path"])
    for row in manifest["files"]:
        data = members[archive_module.ARCHIVE_PREFIX + row["relative_path"]]
        assert "sha256:" + hashlib.sha256(data).hexdigest() == row["sha256"]


def test_cache_root_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert archive_module.resolve_cache_root() is None
    volume = tmp_path / "inputs"
    volume.mkdir()
    monkeypatch.setattr(archive_module, "DEFAULT_CACHE_PARENT", volume)
    assert archive_module.resolve_cache_root() == volume / archive_module.DEFAULT_CACHE_NAME
    monkeypatch.setenv(archive_module.CACHE_ROOT_ENV, str(tmp_path / "explicit"))
    assert archive_module.resolve_cache_root() == (tmp_path / "explicit").resolve()
    assert archive_module.resolve_cache_root(tmp_path / "flag") == (tmp_path / "flag").resolve()
    assert os.environ[archive_module.CACHE_ROOT_ENV]
