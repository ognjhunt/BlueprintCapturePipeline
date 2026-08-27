from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_python_wheelhouse import (
    ROOT_DISTRIBUTIONS,
)
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
            # Derived from the shipped root list, not restated. When
            # ``usd-core`` was added so the provider could import ``pxr``, a
            # hardcoded copy here made nine unrelated tests fail on a manifest
            # mismatch instead of on anything real.
            built_wheels = []
            for distribution in ROOT_DISTRIBUTIONS:
                built = wheels / f"{distribution.replace('-', '_')}-1.0.0-py3-none-any.whl"
                built.write_bytes(f"fixture-wheel:{distribution}".encode())
                built_wheels.append((distribution, built))
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
                "root_distributions": list(ROOT_DISTRIBUTIONS),
                "requirements": [
                    {"name": distribution, "version": "1.0.0"}
                    for distribution, _ in built_wheels
                ],
                "wheels": [
                    {
                        "distribution": distribution,
                        "version": "1.0.0",
                        "filename": built.name,
                        "sha256": _sha256(built),
                        "size_bytes": built.stat().st_size,
                    }
                    for distribution, built in built_wheels
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
            for _distribution, built in built_wheels:
                built.chmod(0o444)
            wheels.chmod(0o555)
            wheelhouse.chmod(0o555)
            for path in [python_manifest_path, *(b for _d, b in built_wheels)]:
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
        component_packages=_component_packages(tmp_path),
    )

    manifest = validate_scene_configuration_toolchain(
        root=output,
        expected_source_commit=commit,
    )
    assert receipt["toolchain_digest"] == manifest["toolchain_digest"]
    assert receipt["full_byte_service_account_readback_passed"] is True
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_allocated"] is False
    # Two per producer plus its driver, then the toolchain manifest, the
    # wheelhouse manifest, and one wheel per shipped root distribution.
    assert len(observed) == 3 * len(ADMITTED_PRODUCER_IDENTITIES) + 2 + len(
        ROOT_DISTRIBUTIONS
    )
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
