from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import native_task_runtime_source_packet as source_packet
from blueprint_pipeline.native_task_runtime_source_packet import (
    ISAACLAB_PACKAGE_NAMES,
    RUNTIME_DEPENDENCY_WHEELS,
    NativeTaskRuntimeSourcePacketError,
    materialize_native_task_runtime_source_packet,
    verify_native_task_runtime_source_packet,
)
from blueprint_pipeline.native_task_runtime_source_provision import (
    provision_native_task_runtime_sources,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(
    root: Path, *, arena: bool, isaaclab_commit: str | None = None
) -> tuple[Path, str, str]:
    repo = root / ("arena" if arena else "isaaclab")
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    if arena:
        assert isaaclab_commit is not None
        files = {
            ".gitmodules": (
                '[submodule "submodules/IsaacLab"]\n'
                "\tpath = submodules/IsaacLab\n"
                "\turl = git@github.com:isaac-sim/IsaacLab.git\n"
            ),
            "LICENSE.md": "Apache-2.0 fixture\n",
            "docker/Dockerfile.isaaclab_arena": (
                "ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1\nFROM ${BASE_IMAGE}\n"
            ),
            "setup.py": "from setuptools import setup; setup(name='isaaclab_arena')\n",
            "pyproject.toml": "[build-system]\nrequires=['setuptools']\n",
            "extension.toml": "[package]\nversion='fixture'\n",
            "isaaclab_arena/__init__.py": "VERSION = 'fixture'\n",
            "ignored.txt": "must not be packaged\n",
        }
    else:
        files = {
            "LICENSE": "BSD-3-Clause fixture\n",
            "apps/isaaclab.python.kit": (
                "[dependencies]\n"
                '"isaacsim.core.simulation_manager" = {}\n'
                '"omni.physx.bundle" = {}\n'
                "[settings.app.extensions]\n"
                'excluded = ["omni.warp.core"]\n'
            ),
            "apps/isaaclab.python.headless.kit": (
                '[dependencies]\n"omni.physx" = {}\n'
                '[settings]\napp.extensions.excluded = ["omni.warp.core"]\n'
            ),
            "apps/isaaclab.python.headless.rendering.kit": (
                '[dependencies]\n"isaaclab.python.headless" = {}\n'
            ),
            "apps/rendering_modes/quality.kit": "[settings]\nquality = true\n",
            "ignored.txt": "omit\n",
        }
        for name in ISAACLAB_PACKAGE_NAMES:
            files[f"source/{name}/setup.py"] = (
                "from setuptools import setup; "
                f"setup(name='{name}', install_requires="
                f"{['warp-lang==1.13.0'] if name == 'isaaclab' else []!r})\n"
            )
            files[f"source/{name}/pyproject.toml"] = "[build-system]\nrequires=['setuptools']\n"
            files[f"source/{name}/config/extension.toml"] = "[package]\nversion='fixture'\n"
            files[f"source/{name}/{name}/__init__.py"] = "VERSION = 'fixture'\n"
    for relative, value in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    _git(repo, "add", ".")
    if arena:
        _git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{isaaclab_commit},submodules/IsaacLab",
        )
    _git(repo, "commit", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")


def _packet(tmp_path: Path, *, output_name: str = "packet") -> dict:
    isaaclab, isaaclab_commit, isaaclab_tree = _repository(tmp_path / "lab-root", arena=False)
    arena, arena_commit, arena_tree = _repository(
        tmp_path / "arena-root", arena=True, isaaclab_commit=isaaclab_commit
    )
    return materialize_native_task_runtime_source_packet(
        output_dir=tmp_path / output_name,
        isaaclab_repo=isaaclab,
        arena_repo=arena,
        generated_at="fixed",
        isaaclab_commit=isaaclab_commit,
        isaaclab_tree=isaaclab_tree,
        isaaclab_runtime_compatibility_commit=isaaclab_commit,
        isaaclab_runtime_compatibility_tree=isaaclab_tree,
        arena_commit=arena_commit,
        arena_tree=arena_tree,
    )


def _wheelhouse(root: Path) -> Path:
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    for contract in RUNTIME_DEPENDENCY_WHEELS:
        path = wheelhouse / contract["filename"]
        if path.is_file():
            continue
        distribution = contract["filename"].split("-", 1)[0]
        dist_info = f"{distribution}-{contract['version']}.dist-info"
        pure_python = bool(contract.get("pure_python", True))
        wheel_tag = str(contract.get("wheel_tag", "py3-none-any"))
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{dist_info}/WHEEL",
                "Wheel-Version: 1.0\n"
                f"Root-Is-Purelib: {str(pure_python).lower()}\n"
                f"Tag: {wheel_tag}\n",
            )
            archive.writestr(f"{dist_info}/METADATA", "Metadata-Version: 2.1\n")
            archive.writestr(f"{distribution}/__init__.py", "FIXTURE = True\n")
    return wheelhouse


def test_source_packet_binds_exact_revisions_licenses_and_minimum_closure(
    tmp_path: Path,
) -> None:
    receipt = _packet(tmp_path)
    verified = verify_native_task_runtime_source_packet(
        tmp_path / "packet/native_task_runtime_source_packet.v1.json"
    )

    assert verified["packet_sha256"] == receipt["packet_sha256"]
    assert verified["redistribution_permitted"] is True
    assert {row["license"]["spdx_id"] for row in verified["repositories"]} == {
        "Apache-2.0",
        "BSD-3-Clause",
    }
    assert verified["install_roots"][-1] == "runtime_sources/arena"
    assert verified["runtime_dependency_basis"]["requirement"] == ("warp-lang==1.13.0")
    assert verified["runtime_dependency_basis"]["source_repository"] == (
        source_packet.ISAACLAB_REPOSITORY
    )
    assert (
        verified["runtime_dependency_basis"]["source_revision"]
        == (verified["repositories"][0]["commit"])
    )
    assert verified["runtime_dependency_basis"]["relative_path"] == (
        "runtime_sources/isaaclab/source/isaaclab/setup.py"
    )
    assert verified["paired_stack"]["isaaclab_revision"] == (verified["repositories"][0]["commit"])
    assert verified["paired_stack"]["simulator_base_image"] == ("nvcr.io/nvidia/isaac-sim:6.0.1")
    assert verified["paired_stack"]["simulator_base_runtime_image"] == (
        "nvcr.io/nvidia/isaac-sim:6.0.1@"
        "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
    )
    assert verified["paired_stack"]["simulator_base_amd64_manifest_digest"] == (
        "sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
    )
    assert verified["paired_stack"]["simulator_runtime_image"] == (
        "nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1@"
        "sha256:ae9c938a16df856effad6dab92115ee0dce2a8813f56847eeeccbebc008d02c4"
    )
    assert verified["paired_stack"]["runtime_image_amd64_manifest_digest"] == (
        "sha256:ef451c70084abdf17af3e65fafbb2a8eae1c25d356b418efa285b942f783703f"
    )
    assert verified["paired_stack"]["runtime_image_config_digest"] == (
        "sha256:663d4a37ac3019ae3b19418df062a72032db9ce4c6dfb82e894c0f0931807978"
    )
    assert verified["paired_stack"]["runtime_image_base_layer_prefix_count"] == 19
    assert verified["paired_stack"]["runtime_image_layer_count"] == 40
    assert verified["runtime_dependency_wheels"] == []
    assert verified["runtime_dependency_basis"]["runtime_owner"] == (
        "official_isaac_lab_complete_runtime"
    )
    assert (
        verified["runtime_dependency_basis"]["runtime_image"]
        == (verified["paired_stack"]["simulator_runtime_image"])
    )
    assert verified["runtime_dependency_basis"]["packet_overlay_required"] is False
    assert verified["runtime_dependency_basis"]["qualification_gate"] == (
        "native_task_pre_app_dependency_matrix.v1"
    )
    assert any(path.endswith("isaaclab_teleop") for path in verified["install_roots"])
    assert any(path.endswith("isaaclab_contrib") for path in verified["install_roots"])
    assert any(path.endswith("isaaclab_newton") for path in verified["install_roots"])
    with zipfile.ZipFile(verified["packet_path"]) as archive:
        names = set(archive.namelist())
        assert "runtime_sources/isaaclab/ignored.txt" not in names
        assert "runtime_sources/arena/ignored.txt" not in names
        assert "runtime_sources/isaaclab/apps/isaaclab.python.kit" in names
        assert "runtime_sources/isaaclab/apps/rendering_modes/quality.kit" in names
        assert (
            "runtime_sources/isaaclab/source/isaaclab_contrib/isaaclab_contrib/__init__.py" in names
        )
        assert (
            "runtime_sources/isaaclab/source/isaaclab_newton/isaaclab_newton/__init__.py" in names
        )
        manifest = json.loads(archive.read("native_task_runtime_source_manifest.v1.json"))
        assert manifest["source_file_count"] == verified["source_file_count"]
        assert len(manifest["runtime_dependency_wheels"]) == len(RUNTIME_DEPENDENCY_WHEELS)


def test_source_packet_rejects_revision_and_archive_tamper(tmp_path: Path) -> None:
    receipt = _packet(tmp_path)
    receipt_path = tmp_path / "packet/native_task_runtime_source_packet.v1.json"
    packet_path = Path(receipt["packet_path"])
    packet_path.write_bytes(packet_path.read_bytes() + b"tamper")

    with pytest.raises(
        NativeTaskRuntimeSourcePacketError,
        match="native_task_runtime_source_packet_identity_mismatch",
    ):
        verify_native_task_runtime_source_packet(receipt_path)

    isaaclab, wrong_lab_commit, isaaclab_tree = _repository(tmp_path / "wrong-lab", arena=False)
    arena, arena_commit, arena_tree = _repository(
        tmp_path / "right-arena", arena=True, isaaclab_commit=wrong_lab_commit
    )
    with pytest.raises(
        NativeTaskRuntimeSourcePacketError,
        match="native_task_runtime_source_commit_mismatch:isaaclab",
    ):
        materialize_native_task_runtime_source_packet(
            output_dir=tmp_path / "wrong-packet",
            isaaclab_repo=isaaclab,
            arena_repo=arena,
            dependency_wheel_dir=_wheelhouse(tmp_path),
            generated_at="fixed",
            isaaclab_commit="0" * 40,
            isaaclab_tree=isaaclab_tree,
            isaaclab_runtime_compatibility_commit="0" * 40,
            isaaclab_runtime_compatibility_tree=isaaclab_tree,
            arena_commit=arena_commit,
            arena_tree=arena_tree,
        )


def test_source_packet_rejects_warp_wheel_not_bound_by_api_source(
    tmp_path: Path,
) -> None:
    isaaclab, _, _ = _repository(tmp_path / "lab", arena=False)
    (isaaclab / "source/isaaclab/setup.py").write_text(
        "from setuptools import setup; setup(name='isaaclab')\n",
        encoding="utf-8",
    )
    _git(isaaclab, "add", "source/isaaclab/setup.py")
    _git(isaaclab, "commit", "-m", "remove warp contract")
    isaaclab_commit = _git(isaaclab, "rev-parse", "HEAD")
    isaaclab_tree = _git(isaaclab, "rev-parse", "HEAD^{tree}")
    arena, arena_commit, arena_tree = _repository(
        tmp_path / "arena", arena=True, isaaclab_commit=isaaclab_commit
    )

    with pytest.raises(
        NativeTaskRuntimeSourcePacketError,
        match="native_task_runtime_dependency_source_contract_invalid:warp-lang",
    ):
        materialize_native_task_runtime_source_packet(
            output_dir=tmp_path / "bad-warp-contract",
            isaaclab_repo=isaaclab,
            arena_repo=arena,
            dependency_wheel_dir=_wheelhouse(tmp_path),
            generated_at="fixed",
            isaaclab_commit=isaaclab_commit,
            isaaclab_tree=isaaclab_tree,
            isaaclab_runtime_compatibility_commit=isaaclab_commit,
            isaaclab_runtime_compatibility_tree=isaaclab_tree,
            arena_commit=arena_commit,
            arena_tree=arena_tree,
        )


def test_source_packet_rejects_arena_pinned_to_another_isaaclab_revision(
    tmp_path: Path,
) -> None:
    isaaclab, first_commit, _ = _repository(tmp_path / "lab", arena=False)
    arena, arena_commit, arena_tree = _repository(
        tmp_path / "arena", arena=True, isaaclab_commit=first_commit
    )
    (isaaclab / "release-marker.txt").write_text("next\n", encoding="utf-8")
    _git(isaaclab, "add", "release-marker.txt")
    _git(isaaclab, "commit", "-m", "next compatible source fixture")
    isaaclab_commit = _git(isaaclab, "rev-parse", "HEAD")
    isaaclab_tree = _git(isaaclab, "rev-parse", "HEAD^{tree}")

    with pytest.raises(
        NativeTaskRuntimeSourcePacketError,
        match="native_task_runtime_arena_isaaclab_pair_mismatch",
    ):
        materialize_native_task_runtime_source_packet(
            output_dir=tmp_path / "unpaired",
            isaaclab_repo=isaaclab,
            arena_repo=arena,
            dependency_wheel_dir=_wheelhouse(tmp_path),
            generated_at="fixed",
            isaaclab_commit=isaaclab_commit,
            isaaclab_tree=isaaclab_tree,
            isaaclab_runtime_compatibility_commit=isaaclab_commit,
            isaaclab_runtime_compatibility_tree=isaaclab_tree,
            arena_commit=arena_commit,
            arena_tree=arena_tree,
        )


def test_relocated_packet_installs_all_sources_once_without_build_backend(
    tmp_path: Path,
) -> None:
    receipt = _packet(tmp_path)
    provider = tmp_path / "provider/runtime_sources"
    provider.mkdir(parents=True)
    relocated_receipt = provider / "native_task_runtime_source_packet.v1.json"
    relocated_packet = provider / "native_task_runtime_sources.zip"
    shutil.copy2(
        tmp_path / "packet/native_task_runtime_source_packet.v1.json",
        relocated_receipt,
    )
    shutil.copy2(receipt["packet_path"], relocated_packet)
    observed: list[list[str]] = []

    def fake_run(command, **kwargs):
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")

    simulator = tmp_path / "isaac-sim"
    simulator.mkdir()
    result = provision_native_task_runtime_sources(
        source_receipt_path=relocated_receipt,
        source_packet_path=relocated_packet,
        extraction_dir=tmp_path / "extracted",
        output_path=tmp_path / "provisioning.json",
        simulator_root=simulator,
        site_packages_dir=tmp_path / "site-packages",
        python_executable="/isaac-sim/python.sh",
        runtime_python_tag="cp312",
        runtime_platform_tags=(
            "manylinux_2_28_x86_64",
            "manylinux_2_26_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ),
        run_command=fake_run,
    )

    assert result["status"] == "completed"
    assert result["all_sources_verified_before_install"] is True
    assert result["source_packages_made_importable"] is True
    assert result["dependencies_installed"] is True
    assert result["runtime_dependency_owner"] == "official_isaac_lab_complete_runtime"
    assert result["runtime_dependency_overlay_required"] is False
    assert result["runtime_dependencies_installed"] == []
    assert result["runtime_dependency_target"] is None
    assert len(observed) == 1
    assert observed[0][1:3] == ["-I", "-c"]
    assert "pip" not in observed[0]
    assert "setuptools" not in observed[0]
    assert "torch" not in observed[0][-1]
    assert "packaging" not in observed[0][-1]
    assert result["runtime_import_probe_returncode"] == 0
    assert result["runtime_import_probes"] == []
    assert len(result["install_roots"]) == len(ISAACLAB_PACKAGE_NAMES) + 1
    assert Path(result["isaac_sim_link"]["path"]).readlink() == simulator
    path_lines = Path(result["path_file"]).read_text(encoding="utf-8").splitlines()
    assert len(path_lines) == 1
    assert path_lines[0].startswith("import sys;sys.path[:0]=[")
    assert all(path in path_lines[0] for path in result["install_roots"])


def test_dependency_imports_are_deferred_to_complete_pre_app_matrix(
    tmp_path: Path,
) -> None:
    receipt = _packet(tmp_path)
    observed: list[list[str]] = []

    def source_probe(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="found", stderr="")

    simulator = tmp_path / "isaac-sim"
    simulator.mkdir()
    result = provision_native_task_runtime_sources(
        source_receipt_path=tmp_path / "packet/native_task_runtime_source_packet.v1.json",
        source_packet_path=receipt["packet_path"],
        extraction_dir=tmp_path / "image-owned-dependencies",
        output_path=tmp_path / "image-owned-dependencies.json",
        simulator_root=simulator,
        site_packages_dir=tmp_path / "site-packages",
        python_executable="/isaac-sim/python.sh",
        runtime_python_tag="cp312",
        runtime_platform_tags=(
            "manylinux_2_28_x86_64",
            "manylinux_2_26_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ),
        run_command=source_probe,
    )

    assert result["status"] == "completed"
    assert len(observed) == 1
    assert all(name not in observed[0][-1] for name in ("torch", "warp", "packaging"))
    assert result["runtime_import_probes"] == []


def test_provisioner_uses_simulator_python_wrapper_for_all_runtime_probes(
    tmp_path: Path,
) -> None:
    receipt = _packet(tmp_path)
    simulator = tmp_path / "isaac-sim"
    simulator.mkdir()
    launcher = simulator / "python.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    observed: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="found", stderr="")

    result = provision_native_task_runtime_sources(
        source_receipt_path=tmp_path / "packet/native_task_runtime_source_packet.v1.json",
        source_packet_path=receipt["packet_path"],
        extraction_dir=tmp_path / "wrapper-probe",
        output_path=tmp_path / "wrapper-probe.json",
        simulator_root=simulator,
        site_packages_dir=tmp_path / "site-packages",
        runtime_python_tag="cp312",
        runtime_platform_tags=(
            "manylinux_2_28_x86_64",
            "manylinux_2_26_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ),
        run_command=fake_run,
    )

    assert result["status"] == "completed"
    assert result["python_executable"] == str(launcher)
    assert result["python_executable_source"] == "simulator_python_launcher"
    assert result["python_probe_flag"] == "-P"
    assert result["python_probe_mode"] == "simulator_wrapper_safe_path"
    assert len(observed) == 1
    assert all(command[0] == str(launcher) for command in observed)
    assert all(command[1:3] == ["-P", "-c"] for command in observed)


def test_explicit_interpreter_runtime_probes_remain_isolated(tmp_path: Path) -> None:
    receipt = _packet(tmp_path)
    simulator = tmp_path / "isaac-sim"
    simulator.mkdir()
    observed: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="found", stderr="")

    result = provision_native_task_runtime_sources(
        source_receipt_path=tmp_path / "packet/native_task_runtime_source_packet.v1.json",
        source_packet_path=receipt["packet_path"],
        extraction_dir=tmp_path / "explicit-probe",
        output_path=tmp_path / "explicit-probe.json",
        simulator_root=simulator,
        site_packages_dir=tmp_path / "site-packages",
        python_executable="/verified/python",
        runtime_python_tag="cp312",
        runtime_platform_tags=(
            "manylinux_2_28_x86_64",
            "manylinux_2_26_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ),
        run_command=fake_run,
    )

    assert result["status"] == "completed"
    assert result["python_probe_flag"] == "-I"
    assert result["python_probe_mode"] == "isolated_interpreter"
    assert all(command[1:3] == ["-I", "-c"] for command in observed)


def test_empty_overlay_does_not_apply_binary_platform_gate(
    tmp_path: Path,
) -> None:
    receipt = _packet(tmp_path)
    probe_called = False

    def source_probe(command, **kwargs):
        nonlocal probe_called
        probe_called = True
        return subprocess.CompletedProcess(command, 0, stdout="found", stderr="")

    simulator = tmp_path / "isaac-sim"
    simulator.mkdir()
    result = provision_native_task_runtime_sources(
        source_receipt_path=tmp_path / "packet/native_task_runtime_source_packet.v1.json",
        source_packet_path=receipt["packet_path"],
        extraction_dir=tmp_path / "wrong-platform",
        output_path=tmp_path / "wrong-platform.json",
        simulator_root=simulator,
        site_packages_dir=tmp_path / "site-packages",
        runtime_python_tag="cp311",
        runtime_platform_tags=("macosx_14_0_arm64",),
        run_command=source_probe,
    )

    assert result["status"] == "completed"
    assert result["runtime_dependency_overlay_required"] is False
    assert probe_called is True


def test_materialization_batches_each_repository_into_one_exact_git_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isaaclab, isaaclab_commit, isaaclab_tree = _repository(tmp_path / "lab", arena=False)
    arena, arena_commit, arena_tree = _repository(
        tmp_path / "arena", arena=True, isaaclab_commit=isaaclab_commit
    )
    commands: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(command, **kwargs):
        commands.append([str(value) for value in command])
        return real_run(command, **kwargs)

    monkeypatch.setattr(source_packet.subprocess, "run", recording_run)
    materialize_native_task_runtime_source_packet(
        output_dir=tmp_path / "batched",
        isaaclab_repo=isaaclab,
        arena_repo=arena,
        dependency_wheel_dir=_wheelhouse(tmp_path),
        generated_at="fixed",
        isaaclab_commit=isaaclab_commit,
        isaaclab_tree=isaaclab_tree,
        isaaclab_runtime_compatibility_commit=isaaclab_commit,
        isaaclab_runtime_compatibility_tree=isaaclab_tree,
        arena_commit=arena_commit,
        arena_tree=arena_tree,
    )

    assert sum("archive" in command for command in commands) == 2
    assert sum("show" in command for command in commands) == 2
