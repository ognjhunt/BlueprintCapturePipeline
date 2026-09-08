from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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


def _component_packages(tmp_path: Path) -> dict[str, Path]:
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


def test_builds_exclusive_read_only_full_byte_readback_toolchain(tmp_path: Path) -> None:
    commit = "a" * 40
    output = tmp_path / "runtime" / commit
    observed: list[Path] = []
    component_packages = _component_packages(tmp_path)

    def readback(path: Path) -> bytes:
        publication_root = next(
            parent for parent in path.parents if parent.parent == output.parent
        )
        assert publication_root.stat().st_mode & 0o001
        assert path.stat().st_mode & 0o004
        observed.append(path)
        return path.read_bytes()

    receipt = build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=output,
        readback=readback,
        readback_actor="service-account:test-runner",
        component_packages=component_packages,
    )

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
