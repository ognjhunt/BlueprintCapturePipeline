from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from blueprint_pipeline import immutable_file_digest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.immutable_file_digest import immutable_file_digest_cache
from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    SCHEMA_VERSION as COMPONENT_PACKAGE_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    validate_scene_configuration_toolchain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def publication_digest_scope(monkeypatch):
    """Exercise full-byte readback and refusal cases under the production scope."""
    clock = immutable_file_digest._now_ns
    monkeypatch.setattr(immutable_file_digest, "_now_ns", lambda: clock() + 10_000_000_000)
    with immutable_file_digest_cache():
        yield


def _component_packages(tmp_path: Path, *, empty_file: bool = False) -> dict[str, Path]:
    packages: dict[str, Path] = {}
    for identity in ADMITTED_PRODUCER_IDENTITIES:
        root = tmp_path / "component-packages" / identity.adapter_id
        root.mkdir(parents=True)
        driver = root / "run"
        driver.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        driver.chmod(0o555)
        package_files = [
            {
                "relative_path": "run",
                "sha256": _sha256(driver),
                "size_bytes": driver.stat().st_size,
                "executable": True,
            }
        ]
        if empty_file and identity.adapter_id == "content_agents_rigid_replacement":
            marker = root / "py.typed"
            marker.write_bytes(b"")
            marker.chmod(0o444)
            package_files.append({"relative_path": "py.typed", "sha256": _sha256(marker),
                                  "size_bytes": 0, "executable": False})
        if identity.adapter_id == "artifixer3d_observed_object_removal":
            wheelhouse = root / "python_wheelhouse"
            wheels = wheelhouse / "wheels"
            wheels.mkdir(parents=True)
            wheel = wheels / "openai_agents-1.0.0-py3-none-any.whl"
            usd_wheel = wheels / "usd_core-1.0.0-py3-none-any.whl"
            wheel.write_bytes(b"fixture-agents-wheel")
            usd_wheel.write_bytes(b"fixture-usd-wheel")
            python_manifest = {
                "schema_version": (
                    "task_evaluation_scene_configuration_python_wheelhouse.v1"
                ),
                "status": "ready",
                "python_version": "3.12",
                "implementation": "cpython",
                "platform": "linux-x86_64",
                "platform_tags": ["manylinux_2_17_x86_64"],
                "lockfile_sha256": "sha256:" + "1" * 64,
                "root_distributions": ["openai-agents", "usd-core"],
                "requirements": [
                    {"name": "openai-agents", "version": "1.0.0"},
                    {"name": "usd-core", "version": "1.0.0"},
                ],
                "wheels": [
                    {
                        "distribution": "openai-agents",
                        "version": "1.0.0",
                        "filename": wheel.name,
                        "sha256": _sha256(wheel),
                        "size_bytes": wheel.stat().st_size,
                    },
                    {
                        "distribution": "usd-core",
                        "version": "1.0.0",
                        "filename": usd_wheel.name,
                        "sha256": _sha256(usd_wheel),
                        "size_bytes": usd_wheel.stat().st_size,
                    },
                ],
                "sdists_allowed": False,
                "provider_network_install_required": False,
                "manifest_digest": "",
            }
            python_manifest["manifest_digest"] = canonical_digest(
                python_manifest, digest_field="manifest_digest"
            )
            python_manifest_path = wheelhouse / (
                "task_evaluation_scene_configuration_python_wheelhouse.v1.json"
            )
            python_manifest_path.write_text(
                json.dumps(python_manifest), encoding="utf-8"
            )
            python_manifest_path.chmod(0o444)
            wheel.chmod(0o444)
            usd_wheel.chmod(0o444)
            wheels.chmod(0o555)
            wheelhouse.chmod(0o555)
            for path in (python_manifest_path, wheel, usd_wheel):
                package_files.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                        "executable": False,
                    }
                )
        manifest = {
            "schema_version": COMPONENT_PACKAGE_SCHEMA_VERSION,
            "status": "immutable_component_ready",
            "adapter_id": identity.adapter_id,
            "adapter_version": identity.version,
            "capability": identity.capability,
            "source_identity": {
                "repository": "https://example.test/public-component",
                "commit": "c" * 40,
                "license": "Apache-2.0",
                "scene_specific_source": False,
            },
            "driver_protocol": (
                "task_evaluation_scene_configuration_component_driver.v1"
            ),
            "driver_entrypoint": "run",
            "network_policy": (
                "disabled"
                if identity.adapter_id
                == "simready_native_import_qualification"
                else "provider_and_openai_api"
            ),
            "secrets_via_files_only": True,
            "raw_secret_values_in_argv_or_logs": False,
            "files": package_files,
            "package_digest": "",
        }
        manifest["package_digest"] = canonical_digest(
            manifest, digest_field="package_digest"
        )
        manifest_path = root / f"{COMPONENT_PACKAGE_SCHEMA_VERSION}.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_path.chmod(0o444)
        root.chmod(0o555)
        packages[identity.adapter_id] = root
    return packages


@pytest.mark.parametrize("private_umask", [False, True])
def test_builds_exclusive_read_only_full_byte_readback_toolchain(
    tmp_path: Path, private_umask: bool,
) -> None:
    commit = "a" * 40
    output = tmp_path / "runtime" / commit
    observed: list[Path] = []
    component_packages = _component_packages(tmp_path)
    output.parent.mkdir(mode=0o755)

    def readback(path: Path) -> bytes:
        publication_root = next(
            parent for parent in path.parents if parent.parent == output.parent
        )
        assert publication_root.stat().st_mode & 0o001
        assert path.stat().st_mode & 0o004
        observed.append(path)
        return path.read_bytes()

    previous = os.umask(0o077 if private_umask else 0o022)
    try:
        receipt = build_published_scene_configuration_toolchain(
            source_commit=commit,
            output_root=output,
            readback=readback,
            readback_actor="service-account:test-runner",
            component_packages=component_packages,
        )
    finally:
        os.umask(previous)

    publication = output.parent / f"{commit}.publication.v1.json"
    assert stat.S_IMODE(publication.stat().st_mode) == 0o444
    assert json.loads(publication.read_text()) == receipt

    manifest = validate_scene_configuration_toolchain(
        root=output,
        expected_source_commit=commit,
    )
    assert receipt["toolchain_digest"] == manifest["toolchain_digest"]
    assert receipt["full_byte_service_account_readback_passed"] is True
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_allocated"] is False
    assert len(observed) == 3 * len(ADMITTED_PRODUCER_IDENTITIES) + 4
    assert not output.stat().st_mode & 0o222
    assert all(not path.stat().st_mode & 0o222 for path in output.rglob("*"))
    for identity in ADMITTED_PRODUCER_IDENTITIES:
        executable = output / "stages" / identity.adapter_id
        assert executable.stat().st_mode & 0o111
        assert (
            "blueprint_pipeline.task_evaluation_scene_configuration_stage_tool"
            in executable.read_text(encoding="utf-8")
        )
        component = output / "components" / identity.adapter_id / "package" / "run"
        source_component = component_packages[identity.adapter_id] / "run"
        assert component.stat().st_ino == source_component.stat().st_ino
        assert component.stat().st_mode & 0o111
        assert component.read_text(encoding="utf-8") == "#!/bin/sh\nexit 99\n"


def test_toolchain_publication_fails_closed_on_existing_or_bad_readback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime"
    output.mkdir()
    with pytest.raises(ValueError, match="output_exists"):
        build_published_scene_configuration_toolchain(
            source_commit="b" * 40,
            output_root=output,
            readback=lambda path: path.read_bytes(),
            readback_actor="service-account:test-runner",
            component_packages=_component_packages(tmp_path / "existing"),
        )

    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="service_readback_failed"):
        build_published_scene_configuration_toolchain(
            source_commit="b" * 40,
            output_root=failed,
            readback=lambda _path: b"tampered",
            readback_actor="service-account:test-runner",
            component_packages=_component_packages(tmp_path / "bad"),
        )
    assert not failed.exists()


def test_a_stopped_toolchain_publication_leaves_no_staging(tmp_path: Path) -> None:
    def stopped(_path: Path) -> bytes:
        raise KeyboardInterrupt

    root = tmp_path / "scene-configuration"
    with pytest.raises(KeyboardInterrupt):
        build_published_scene_configuration_toolchain(
            source_commit="b" * 40,
            output_root=root / ("b" * 40),
            readback=stopped,
            readback_actor="service-account:test-runner",
            component_packages=_component_packages(tmp_path / "packages"),
        )
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("case,reuses_previous", [
    ("identical", True), ("changed_bytes", False), ("writable_source", False),
    ("writable_previous", False), ("different_mode", False), ("symlink", False),
    ("retired_during_link", False),
])
def test_package_copy_only_reuses_identical_readonly_files(tmp_path, monkeypatch, case, reuses_previous):
    from scripts import build_task_evaluation_scene_configuration_toolchain as builder
    source_root = tmp_path / "new-package"
    source_root.mkdir()
    source = source_root / "source.zip"
    source.write_bytes(b"a" * 65536)
    source.chmod(0o644 if case == "writable_source" else 0o444)
    prior_root = tmp_path / ("a" * 40)
    package_relative = Path("components/example/package")
    previous = prior_root / package_relative / "source.zip"
    previous.parent.mkdir(parents=True)
    previous.write_bytes((b"b" if case == "changed_bytes" else b"a") * 65536)
    previous.chmod(0o644 if case == "writable_previous" else 0o555 if case == "different_mode" else 0o444)
    if case == "symlink":
        previous.unlink()
        previous.symlink_to(source)
    if case == "retired_during_link":
        real_link = builder.os.link
        def link(origin, target, **kwargs):
            if Path(origin) == previous:
                raise FileNotFoundError("old release retired")
            return real_link(origin, target, **kwargs)
        monkeypatch.setattr(builder.os, "link", link)
    target = tmp_path / "installed.zip"
    shared = {"file_count": 0, "bytes": 0}
    builder._reuse_published_package_file(
        str(source), str(target), package_source=source_root,
        package_relative=package_relative, previous_roots=[prior_root], shared=shared,
    )
    assert target.read_bytes() == source.read_bytes() == b"a" * 65536
    assert target.stat().st_mode == source.stat().st_mode
    expected = previous if reuses_previous else source
    assert target.stat().st_ino == expected.stat().st_ino
    assert shared == {"file_count": int(reuses_previous), "bytes": 65536 if reuses_previous else 0}


def test_new_release_shares_regenerated_immutable_payload_after_full_readback(tmp_path):
    releases = tmp_path / "releases"
    installed = []
    for commit in ("a" * 40, "b" * 40):
        packages = _component_packages(tmp_path / commit)
        adapter, package = next(iter(packages.items()))
        package.chmod(0o755)
        payload = package / "retained-library.zip"
        payload.write_bytes(b"immutable vendor source\n" * 4096)
        payload.chmod(0o444)
        manifest_path = package / f"{COMPONENT_PACKAGE_SCHEMA_VERSION}.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].append({"relative_path": payload.name, "sha256": _sha256(payload),
                                  "size_bytes": payload.stat().st_size, "executable": False})
        manifest["package_digest"] = canonical_digest(manifest, digest_field="package_digest")
        manifest_path.chmod(0o644)
        manifest_path.write_text(json.dumps(manifest))
        manifest_path.chmod(0o444)
        package.chmod(0o555)
        result = build_published_scene_configuration_toolchain(
            source_commit=commit, output_root=releases / commit,
            readback=lambda path: path.read_bytes(), readback_actor="test-service",
            component_packages=packages,
        )
        installed.append(releases / commit / "components" / adapter / "package" / payload.name)
        assert result["full_byte_service_account_readback_passed"] is True
    assert installed[0].stat().st_ino == installed[1].stat().st_ino
    assert result["shared_immutable_file_count"] == 1
    assert result["shared_immutable_file_bytes"] == payload.stat().st_size
    assert payload.stat().st_ino != installed[1].stat().st_ino


def test_per_file_readback_streams_through_a_digest_capable_reader(tmp_path: Path) -> None:
    """The deploy's reader hashes each file as the service account streams it; nothing is held whole."""
    from scripts.build_task_evaluation_scene_configuration_toolchain import _readback_digest

    path = tmp_path / "large.so"
    path.write_bytes(b"x" * 1024)

    class StreamingReader:
        def __init__(self) -> None:
            self.digests: list[Path] = []

        def __call__(self, _path: Path) -> bytes:
            raise AssertionError("per-file verification must not hold the whole file")

        def digest(self, target: Path) -> tuple[int, str]:
            self.digests.append(target)
            return 1024, "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()

    reader = StreamingReader()
    assert _readback_digest(reader, path) == (1024, "sha256:" + hashlib.sha256(b"x" * 1024).hexdigest())
    assert reader.digests == [path]
    # A plain bytes reader (the builder's own CLI and tests) still works.
    assert _readback_digest(lambda target: target.read_bytes(), path)[0] == 1024


def test_an_empty_component_file_publishes_and_validates(tmp_path: Path) -> None:
    """2026-10-13: the ARTISAN runtime's 675 empty files (py.typed, REQUESTED) refused the toolchain."""
    commit = "b" * 40
    output = tmp_path / "runtime" / commit
    output.parent.mkdir(mode=0o755)
    build_published_scene_configuration_toolchain(
        source_commit=commit, output_root=output, readback=lambda path: path.read_bytes(),
        readback_actor="service-account:test-runner",
        component_packages=_component_packages(tmp_path, empty_file=True))
    manifest = validate_scene_configuration_toolchain(root=output, expected_source_commit=commit)
    empty = [row for row in manifest["files"] if row["size_bytes"] == 0]
    assert [row["relative_path"].rsplit("/", 1)[-1] for row in empty] == ["py.typed"]
    assert empty[0]["sha256"] == "sha256:" + hashlib.sha256(b"").hexdigest()


class _RecordingServiceAccount:
    """The production reader process (as the test user), recording what it read and what it only opened."""

    def __init__(self) -> None:
        from blueprint_pipeline.service_account_reader import ServiceAccountReader

        self.reader = ServiceAccountReader("blueprint", command_prefix=())
        self.read: list[str] = []
        self.opened: list[str] = []

    def __call__(self, path: Path) -> bytes:
        self.read.append(path.name)
        return self.reader(path)

    def digest(self, path: Path) -> tuple[int, str]:
        self.read.append(path.name)
        return self.reader.digest(path)

    def identity(self, path: Path) -> tuple[int, ...]:
        self.opened.append(path.name)
        return self.reader.identity(path)


_PAYLOAD = b"vendored library bytes\n" * 4096  # above the 64 KiB package reuse floor


def _publish_with_payload(tmp_path, releases, commit, *, payload=_PAYLOAD,
                          actor="service-account:blueprint"):
    """One deploy: its own process-wide digest publication, its own service-account reader."""
    import contextvars

    packages = _component_packages(tmp_path / f"packages-{commit[:1]}")
    adapter, package = next(iter(packages.items()))
    package.chmod(0o755)
    (package / "libvendor.so").write_bytes(payload)
    (package / "libvendor.so").chmod(0o444)
    manifest_path = package / f"{COMPONENT_PACKAGE_SCHEMA_VERSION}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append({"relative_path": "libvendor.so", "sha256": _sha256(package / "libvendor.so"),
                              "size_bytes": len(payload), "executable": False})
    manifest["package_digest"] = canonical_digest(manifest, digest_field="package_digest")
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest))
    manifest_path.chmod(0o444)
    package.chmod(0o555)
    account = _RecordingServiceAccount()

    def deploy():
        with immutable_file_digest_cache():
            return build_published_scene_configuration_toolchain(
                source_commit=commit, output_root=releases / commit, readback=account,
                readback_actor=actor, component_packages=packages)

    try:
        receipt = contextvars.Context().run(deploy)
    finally:
        account.reader.close()
    return receipt, account, releases / commit / "components" / adapter / "package" / "libvendor.so"


def test_unchanged_inode_read_back_by_the_previous_release_is_opened_not_read_again(tmp_path):
    from blueprint_pipeline.readback_ledger import ledger_path

    releases = tmp_path / "scene-configuration"
    first, first_account, first_payload = _publish_with_payload(tmp_path, releases, "a" * 40)
    assert "libvendor.so" in first_account.read and first_account.opened == []
    assert first["service_account_readback"]["previously_read_inode_file_count"] == 0
    ledger = first["service_account_readback"]["ledger"]
    assert ledger["status"] == "written" and ledger["file_count"] == first["file_count"]
    assert stat.S_IMODE(ledger_path(releases / ("a" * 40)).stat().st_mode) == 0o444

    second, account, payload = _publish_with_payload(tmp_path, releases, "b" * 40)
    assert payload.stat().st_ino == first_payload.stat().st_ino
    assert account.opened == ["libvendor.so"] and "libvendor.so" not in account.read
    assert len(account.read) == second["file_count"]  # every other file, plus the manifest
    readback = second["service_account_readback"]
    assert readback["previously_read_inode_file_count"] == 1
    assert readback["previously_read_inode_bytes"] == len(_PAYLOAD)
    assert readback["full_byte_read_file_count"] == second["file_count"] - 1
    assert readback["proven_by_previous_toolchains"] == {"a" * 40: first["file_count"]}
    assert readback["ledger"]["file_count"] == second["file_count"]
    # The release's own identity is untouched: same manifest shape, same claims, validates.
    manifest = validate_scene_configuration_toolchain(root=releases / ("b" * 40), expected_source_commit="b" * 40)
    first_manifest = json.loads((releases / ("a" * 40) / f"{manifest['schema_version']}.json").read_text())
    assert set(manifest) == set(first_manifest)
    assert manifest["full_byte_service_account_readback_passed"] is True
    assert second["toolchain_digest"] == manifest["toolchain_digest"]
    assert payload.read_bytes() == _PAYLOAD


@pytest.mark.parametrize("case", ("changed_bytes", "rewritten_in_place", "touched", "replaced",
                                  "ledger_edited", "ledger_writable", "other_reader"))
def test_changed_or_tampered_inode_is_read_back_in_full(tmp_path, case):
    from blueprint_pipeline.readback_ledger import ledger_path

    releases = tmp_path / "scene-configuration"
    _first, _account, first_payload = _publish_with_payload(tmp_path, releases, "a" * 40)
    payload = _PAYLOAD
    ledger = ledger_path(releases / ("a" * 40))
    if case == "changed_bytes":
        payload = _PAYLOAD.replace(b"vendored", b"upgraded")
    elif case == "rewritten_in_place":
        payload = _PAYLOAD.replace(b"vendored", b"tampered")
        first_payload.chmod(0o644)
        first_payload.write_bytes(payload)
        first_payload.chmod(0o444)
    elif case == "touched":
        before = first_payload.stat()
        os.utime(first_payload, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    elif case == "replaced":
        before = first_payload.stat()
        replacement = first_payload.with_name("replacement")
        first_payload.parent.chmod(0o755)
        replacement.write_bytes(_PAYLOAD)
        replacement.chmod(0o444)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, first_payload)
        first_payload.parent.chmod(0o555)
    elif case == "ledger_edited":
        value = json.loads(ledger.read_text())
        relative = next(name for name in value["inodes"] if name.endswith("libvendor.so"))
        value["inodes"][relative][1] += 1
        ledger.chmod(0o644)
        ledger.write_text(json.dumps(value))
        ledger.chmod(0o444)
    elif case == "ledger_writable":
        ledger.chmod(0o644)
    second, account, installed = _publish_with_payload(
        tmp_path, releases, "b" * 40, payload=payload,
        actor="service-account:other" if case == "other_reader" else "service-account:blueprint")
    assert account.opened == [] and "libvendor.so" in account.read
    assert second["service_account_readback"]["previously_read_inode_file_count"] == 0
    assert installed.read_bytes() == payload
    validate_scene_configuration_toolchain(root=releases / ("b" * 40), expected_source_commit="b" * 40)


def test_retiring_the_previous_release_keeps_inodes_a_retained_release_still_links(tmp_path):
    from blueprint_pipeline import control_plane_release_retirement as retirement
    from blueprint_pipeline.readback_ledger import ledger_path

    releases = tmp_path / "scene-configuration"
    _first, _account, first_payload = _publish_with_payload(tmp_path, releases, "a" * 40)
    _second, _account, payload = _publish_with_payload(tmp_path, releases, "b" * 40)
    # Moving a directory to a new parent rewrites its "..": root may through
    # 0555, the test user needs write on the published root.
    (releases / ("a" * 40)).chmod(0o755)
    retiring = retirement._stage_aside(releases / ("a" * 40), "test")
    for directory in [retiring, *(path for path in retiring.rglob("*") if path.is_dir())]:
        directory.chmod(0o700)  # root deletes through 0555 directories; the test user cannot
    deleted = retirement.delete_retiring_trees([releases])
    assert deleted["failed"] == [] and deleted["shared_bytes"] >= len(_PAYLOAD)
    assert not first_payload.exists() and payload.read_bytes() == _PAYLOAD
    validate_scene_configuration_toolchain(root=releases / ("b" * 40), expected_source_commit="b" * 40)

    third, account, third_payload = _publish_with_payload(tmp_path, releases, "c" * 40)
    assert third_payload.stat().st_ino == payload.stat().st_ino
    assert account.opened == ["libvendor.so"]
    assert list(third["service_account_readback"]["proven_by_previous_toolchains"]) == ["b" * 40]
    assert not ledger_path(releases / ("a" * 40)).exists()  # pruned with its retired tree
