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


def _repository(root: Path, *, arena: bool) -> tuple[Path, str, str]:
    repo = root / ("arena" if arena else "isaaclab")
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    if arena:
        files = {
            "LICENSE.md": "Apache-2.0 fixture\n",
            "setup.py": "from setuptools import setup; setup(name='isaaclab_arena')\n",
            "pyproject.toml": "[build-system]\nrequires=['setuptools']\n",
            "extension.toml": "[package]\nversion='fixture'\n",
            "isaaclab_arena/__init__.py": "VERSION = 'fixture'\n",
            "ignored.txt": "must not be packaged\n",
        }
    else:
        files = {
            "LICENSE": "BSD-3-Clause fixture\n",
            "apps/isaaclab.python.kit": "[settings]\nfixture = true\n",
            "apps/rendering_modes/quality.kit": "[settings]\nquality = true\n",
            "ignored.txt": "omit\n",
        }
        for name in ISAACLAB_PACKAGE_NAMES:
            files[f"source/{name}/setup.py"] = (
                "from setuptools import setup; " f"setup(name='{name}')\n"
            )
            files[f"source/{name}/pyproject.toml"] = (
                "[build-system]\nrequires=['setuptools']\n"
            )
            files[f"source/{name}/config/extension.toml"] = (
                "[package]\nversion='fixture'\n"
            )
            files[f"source/{name}/{name}/__init__.py"] = "VERSION = 'fixture'\n"
    for relative, value in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")


def _packet(tmp_path: Path, *, output_name: str = "packet") -> dict:
    isaaclab, isaaclab_commit, isaaclab_tree = _repository(
        tmp_path / "lab-root", arena=False
    )
    arena, arena_commit, arena_tree = _repository(tmp_path / "arena-root", arena=True)
    return materialize_native_task_runtime_source_packet(
        output_dir=tmp_path / output_name,
        isaaclab_repo=isaaclab,
        arena_repo=arena,
        dependency_wheel_dir=_wheelhouse(tmp_path),
        generated_at="fixed",
        isaaclab_commit=isaaclab_commit,
        isaaclab_tree=isaaclab_tree,
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
    assert any(path.endswith("isaaclab_teleop") for path in verified["install_roots"])
    assert any(path.endswith("isaaclab_contrib") for path in verified["install_roots"])
    with zipfile.ZipFile(verified["packet_path"]) as archive:
        names = set(archive.namelist())
        assert "runtime_sources/isaaclab/ignored.txt" not in names
        assert "runtime_sources/arena/ignored.txt" not in names
        assert (
            "runtime_sources/isaaclab/apps/isaaclab.python.kit" in names
        )
        assert (
            "runtime_sources/isaaclab/apps/rendering_modes/quality.kit" in names
        )
        assert (
            "runtime_sources/isaaclab/source/isaaclab_contrib/"
            "isaaclab_contrib/__init__.py" in names
        )
        manifest = json.loads(
            archive.read("native_task_runtime_source_manifest.v1.json")
        )
        assert manifest["source_file_count"] == verified["source_file_count"]
        assert len(manifest["runtime_dependency_wheels"]) == len(
            RUNTIME_DEPENDENCY_WHEELS
        )


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

    isaaclab, _, isaaclab_tree = _repository(tmp_path / "wrong-lab", arena=False)
    arena, arena_commit, arena_tree = _repository(tmp_path / "right-arena", arena=True)
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
        runtime_platform_tags=("manylinux_2_28_x86_64",),
        run_command=fake_run,
    )

    assert result["status"] == "completed"
    assert result["all_sources_verified_before_install"] is True
    assert result["source_packages_made_importable"] is True
    assert result["dependencies_installed"] is True
    assert len(result["runtime_dependencies_installed"]) == len(
        RUNTIME_DEPENDENCY_WHEELS
    )
    assert next(
        row for row in result["runtime_dependencies_installed"] if row["package"] == "h5py"
    )["pure_python"] is False
    assert len(observed) == 1
    assert observed[0][1:3] == ["-I", "-c"]
    assert "pip" not in observed[0]
    assert "setuptools" not in observed[0]
    assert len(result["install_roots"]) == len(ISAACLAB_PACKAGE_NAMES) + 1
    assert Path(result["isaac_sim_link"]["path"]).readlink() == simulator
    path_lines = Path(result["path_file"]).read_text(encoding="utf-8").splitlines()
    assert path_lines == [result["runtime_dependency_target"], *result["install_roots"]]


def test_binary_runtime_dependency_rejects_wrong_python_or_platform_before_probe(
    tmp_path: Path,
) -> None:
    receipt = _packet(tmp_path)
    probe_called = False

    def forbidden_probe(*args, **kwargs):
        nonlocal probe_called
        probe_called = True
        raise AssertionError("runtime probe must not run after binary wheel mismatch")

    simulator = tmp_path / "isaac-sim"
    simulator.mkdir()
    result = provision_native_task_runtime_sources(
        source_receipt_path=tmp_path
        / "packet/native_task_runtime_source_packet.v1.json",
        source_packet_path=receipt["packet_path"],
        extraction_dir=tmp_path / "wrong-platform",
        output_path=tmp_path / "wrong-platform.json",
        simulator_root=simulator,
        site_packages_dir=tmp_path / "site-packages",
        runtime_python_tag="cp311",
        runtime_platform_tags=("macosx_14_0_arm64",),
        run_command=forbidden_probe,
    )

    assert result["status"] == "blocked"
    assert result["exception"] == (
        "native_task_runtime_dependency_binary_wheel_incompatible"
    )
    assert probe_called is False


def test_materialization_batches_each_repository_into_one_exact_git_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isaaclab, isaaclab_commit, isaaclab_tree = _repository(
        tmp_path / "lab", arena=False
    )
    arena, arena_commit, arena_tree = _repository(tmp_path / "arena", arena=True)
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
        arena_commit=arena_commit,
        arena_tree=arena_tree,
    )

    assert sum("archive" in command for command in commands) == 2
    assert not any("show" in command for command in commands)
