from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import groot_oscar_model_cache_s3_remote_executor as executor
from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    RUNTIME_ARCHIVE_ROOTS,
)


SOURCE_REF = "docker.io/blueprint/release@sha256:" + "1" * 64
CARRIER_REF = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" + "2" * 64


class FakeDocker:
    def __init__(
        self,
        *,
        verify_digest: bool = True,
        fail_copy_root: str = "",
        fail_validation_text: str = "",
        deferred_driver_sonames: tuple[str, ...] = (),
    ) -> None:
        self.calls: list[list[str]] = []
        self.verify_digest = verify_digest
        self.fail_copy_root = fail_copy_root
        self.fail_validation_text = fail_validation_text
        self.deferred_driver_sonames = deferred_driver_sonames

    def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        command = list(argv)
        self.calls.append(command)
        if command[1] == "pull":
            return SimpleNamespace(stdout="")
        if command[1:4] == ["image", "inspect", "--format"]:
            image_ref = command[-1]
            digest = image_ref.rpartition("@")[2] if self.verify_digest else "sha256:" + "f" * 64
            return SimpleNamespace(stdout=json.dumps(["docker.io/test/image@" + digest]))
        if command[1] == "create":
            return SimpleNamespace(stdout="a" * 64 + "\n")
        if command[1] == "cp":
            source = command[2].split(":", 1)[1]
            if source == f"/{self.fail_copy_root}":
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    stderr="credential-like diagnostic must not be persisted",
                )
            destination = Path(command[3])
            copied = destination / Path(source).name
            (copied / "bin").mkdir(parents=True)
            (copied / "bin/python3").write_text("runtime\n", encoding="utf-8")
            (copied / "bin/python").symlink_to("python3")
            if source == "/opt/blueprint":
                (copied / "ckpts/sonic").mkdir(parents=True)
                (copied / "ckpts/sonic/model.safetensors").write_bytes(b"model")
                (copied / "hf_home").mkdir()
                (copied / "hf_home/cached-model").write_bytes(b"model")
                (copied / "models").symlink_to("hf_home")
            return SimpleNamespace(stdout="")
        if command[1:3] == ["rm", "-f"]:
            return SimpleNamespace(stdout="")
        if command[1] == "run":
            if self.fail_validation_text and self.fail_validation_text in command[-1]:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    stderr="credential-like diagnostic must not be persisted",
                )
            stdout = (
                "".join(
                    f"BLUEPRINT_GPU_DRIVER_DEFERRED_SONAME {soname}\n"
                    for soname in self.deferred_driver_sonames
                )
                + "BLUEPRINT_ALL_ARCHIVED_ELF_LINKAGE_OK count=42\n"
                if "elf_count" in command[-1]
                else "runtime verified\n"
            )
            return SimpleNamespace(stdout=stdout)
        raise AssertionError(command)


def _request() -> dict:
    return {
        "enabled": True,
        "source_release_image_ref": SOURCE_REF,
        "carrier_image_ref": CARRIER_REF,
    }


def test_prepare_runtime_bundle_copies_allowlist_and_builds_verified_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")
    real_tarfile_open = tarfile.open
    archive_open_options: list[tuple[str, int | None]] = []

    def tracked_tarfile_open(*args, **kwargs):  # type: ignore[no-untyped-def]
        mode = str(args[1] if len(args) > 1 else kwargs.get("mode", "r"))
        archive_open_options.append((mode, kwargs.get("compresslevel")))
        return real_tarfile_open(*args, **kwargs)

    monkeypatch.setattr(executor.tarfile, "open", tracked_tarfile_open)
    docker = FakeDocker(deferred_driver_sonames=("libnvidia-ml.so.1", "libcuda.so.1"))
    runtime_root = tmp_path / "runtime"
    build_root = tmp_path / "build"

    result = executor.prepare_runtime_bundle(
        _request(),
        runtime_root=runtime_root,
        build_root=build_root,
        runner=docker,
        generated_at="2026-07-15T12:00:00Z",
    )

    assert result["status"] == "completed"
    assert result["source_and_carrier_registry_digests_verified"] is True
    assert result["carrier_validation"]["status"] == "passed_with_gpu_driver_deferred"
    assert result["carrier_validation"]["archived_elf_file_count"] == 42
    assert result["carrier_validation"]["gpu_driver_deferred_sonames"] == [
        "libcuda.so.1",
        "libnvidia-ml.so.1",
    ]
    assert result["runtime_imports_and_wbc_linkage_verified_in_exact_carrier"] is False
    assert result["runtime_cpu_compatible_with_gpu_driver_deferred_linkage"] is True
    assert result["gpu_driver_deferred_resolution_unproven"] is True
    assert ("w:gz", executor.RUNTIME_ARCHIVE_GZIP_COMPRESSLEVEL) in archive_open_options
    assert executor.RUNTIME_ARCHIVE_GZIP_COMPRESSLEVEL == 1
    assert len(result["additional_artifacts"]) == 2
    assert {row["remote_key"] for row in result["additional_artifacts"]} == {
        DEFAULT_RUNTIME_ARCHIVE_PATH.removeprefix("/workspace/"),
        DEFAULT_RUNTIME_MANIFEST_PATH.removeprefix("/workspace/"),
    }
    with tarfile.open(result["archive_path"], "r:gz") as archive:
        inventory = archive.getnames()
        for root in RUNTIME_ARCHIVE_ROOTS:
            assert root in inventory
            assert f"{root}/bin/python" in inventory
        assert not any(name.startswith("opt/blueprint/ckpts/") for name in inventory)
        assert not any(name.startswith("opt/blueprint/hf_home/") for name in inventory)
        assert "opt/blueprint/models" not in inventory
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["source_release_image_ref"] == SOURCE_REF
    assert manifest["carrier_image_ref"] == CARRIER_REF
    assert manifest["gpu_driver_deferred_sonames"] == [
        "libcuda.so.1",
        "libnvidia-ml.so.1",
    ]
    assert [call[:2] for call in docker.calls].count(["docker", "pull"]) == 2
    assert docker.calls[-1] == ["docker", "rm", "-f", "a" * 64]
    carrier_runs = [call for call in docker.calls if call[1] == "run"]
    assert len(carrier_runs) == 7
    assert all("--network" in call for call in carrier_runs)
    assert all("none" in call for call in carrier_runs)
    assert all("--env" in call for call in carrier_runs)
    assert all(SOURCE_REF not in call for call in carrier_runs)
    assert all(CARRIER_REF in call for call in carrier_runs)
    for key, value in executor.RUNTIME_CARRIER_ENV.items():
        assert all(f"{key}={value}" in call for call in carrier_runs)
    serverless_runs = [
        call
        for call in carrier_runs
        if "blueprint_pipeline.groot_oscar_runpod_serverless_worker" in call[-1]
    ]
    assert len(serverless_runs) == 1
    assert "--verify-serverless-runtime" in serverless_runs[0][-1]
    assert "import runpod" not in serverless_runs[0][-1]
    validation_scripts = [call[-1] for call in carrier_runs]
    elf_scan = next(script for script in validation_scripts if "elf_count" in script)
    assert all(f"/{root}" in elf_scan for root in RUNTIME_ARCHIVE_ROOTS)
    assert "ThreadPoolExecutor" in elf_scan
    assert "subprocess.run(" in elf_scan
    assert '["ldd", path]' in elf_scan
    assert "timeout=30" in elf_scan
    assert "BLUEPRINT_VALIDATION_DIAGNOSTIC elf_linkage_timeout" in elf_scan
    elf_python = elf_scan.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(elf_python, "<carrier-elf-audit>", "exec")
    assert "not found" in elf_scan
    assert "BLUEPRINT_GPU_DRIVER_DEFERRED_SONAME" in elf_scan
    assert "libcuda.so.*" in elf_scan
    assert any("import diffusers" in script for script in validation_scripts)
    isaac_scan = next(
        script for script in validation_scripts if "import carb; import isaacsim" in script
    )
    assert "from isaacsim import SimulationApp" in isaac_scan
    assert "import blueprint_pipeline.isaac_runtime_task_backend" in isaac_scan
    assert "app = SimulationApp" not in isaac_scan
    assert "isaacsim.core" not in isaac_scan
    isaac_inventory = next(
        script for script in validation_scripts if "ISAAC_CORE_EXTENSION_INVENTORY" in script
    )
    assert "-path '*/isaacsim/core/prims/__init__.py'" in isaac_inventory
    assert 'test -n "$prims_init"' in isaac_inventory
    assert "grep -R -Fq 'SingleArticulation'" in isaac_inventory
    assert Path(result["archive_path"]).parent == runtime_root
    assert not build_root.exists()

    serverless_healthcheck = [
        argv
        for argv in manifest["healthcheck_argv"]
        if "blueprint_pipeline.groot_oscar_runpod_serverless_worker" in argv
    ]
    assert serverless_healthcheck == [
        [
            "/opt/runpod-serverless-venv/bin/python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_serverless_worker",
            "--verify-serverless-runtime",
        ]
    ]
    isaac_healthcheck = [
        argv for argv in manifest["healthcheck_argv"] if argv[0] == "/isaac-sim/python.sh"
    ]
    assert len(isaac_healthcheck) == 1
    assert isaac_healthcheck[0][2].index("app = SimulationApp") < isaac_healthcheck[0][
        2
    ].index("from isaacsim.core.prims import SingleArticulation")
    assert isaac_healthcheck[0][2].endswith("app.close()")


def test_runtime_carrier_env_matches_foundation_runtime_contract() -> None:
    foundation = (
        Path(__file__).resolve().parents[1]
        / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile"
    ).read_text(encoding="utf-8")

    assert "PYTHONPATH=/opt/wbc:/opt/OSCAR" in foundation
    assert (
        "/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:"
        "/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
        in executor.RUNTIME_CARRIER_ENV["LD_LIBRARY_PATH"]
    )


def test_prepare_runtime_bundle_rejects_registry_digest_mismatch_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")
    docker = FakeDocker(verify_digest=False)

    with pytest.raises(RuntimeError, match="typed_runtime_bundle_image_digest_unverified"):
        executor.prepare_runtime_bundle(
            _request(),
            runtime_root=tmp_path / "runtime",
            build_root=tmp_path / "build",
            runner=docker,
        )

    assert not any(call[1] == "cp" for call in docker.calls)


def test_prepare_runtime_bundle_not_requested_never_calls_docker(tmp_path: Path) -> None:
    docker = FakeDocker()
    result = executor.prepare_runtime_bundle(
        {"enabled": False},
        runtime_root=tmp_path / "runtime",
        build_root=tmp_path / "build",
        runner=docker,
    )
    assert result["status"] == "not_requested"
    assert result["additional_artifacts"] == []
    assert docker.calls == []


def test_prepare_runtime_bundle_reports_secret_free_failed_copy_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")
    failed_root = RUNTIME_ARCHIVE_ROOTS[1]
    docker = FakeDocker(fail_copy_root=failed_root)
    progress: dict[str, object] = {}

    with pytest.raises(
        RuntimeError,
        match=f"typed_runtime_bundle_allowlisted_root_copy_failed:{failed_root}",
    ):
        executor.prepare_runtime_bundle(
            _request(),
            runtime_root=tmp_path / "runtime",
            build_root=tmp_path / "build",
            runner=docker,
            progress=progress,
        )

    assert progress == {
        "schema_version": "groot_oscar_runtime_bundle_progress.v1",
        "phase": "copying_allowlisted_root",
        "allowlisted_root": failed_root,
        "raw_secret_values_recorded": False,
    }
    assert "credential-like" not in json.dumps(progress)
    assert docker.calls[-1] == ["docker", "rm", "-f", "a" * 64]


def test_prepare_runtime_bundle_names_failed_carrier_check_without_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")
    docker = FakeDocker(fail_validation_text="import inference.inference_oscar")

    with pytest.raises(
        RuntimeError,
        match="typed_runtime_bundle_carrier_validation_failed:oscar_import_matrix",
    ) as raised:
        executor.prepare_runtime_bundle(
            _request(),
            runtime_root=tmp_path / "runtime",
            build_root=tmp_path / "build",
            runner=docker,
        )

    assert "credential-like" not in str(raised.value)
    assert docker.calls[-1] == ["docker", "rm", "-f", "a" * 64]


def test_runtime_carrier_validation_collects_all_failures_before_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")

    class MultiFailureDocker(FakeDocker):
        def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
            command = list(argv)
            if command[1] == "run" and (
                "elf_count" in command[-1] or "import carb; import isaacsim" in command[-1]
            ):
                self.calls.append(command)
                raise subprocess.CalledProcessError(returncode=1, cmd=command)
            return super().__call__(argv, **kwargs)

    docker = MultiFailureDocker()
    with pytest.raises(RuntimeError) as raised:
        executor.prepare_runtime_bundle(
            _request(),
            runtime_root=tmp_path / "runtime",
            build_root=tmp_path / "build",
            runner=docker,
        )

    message = str(raised.value)
    assert "all_archived_elf_linkage" in message
    assert "isaac_bootstrap_import" in message
    assert sum(call[1] == "run" for call in docker.calls) == 7


def test_runtime_carrier_validation_diagnostics_allowlist_only_missing_dependencies() -> None:
    failure = subprocess.CalledProcessError(
        returncode=1,
        cmd=["docker", "run"],
        output=(
            "libcuda_runtime.so.1 => not found\n"
            "ModuleNotFoundError: No module named 'omni.missing'\n"
            "BLUEPRINT_VALIDATION_DIAGNOSTIC elf_linkage_timeout\n"
            "credential-like text must not be copied\n"
        ),
        stderr="liboptional.so.2: cannot open shared object file\n",
    )

    assert executor._validation_diagnostic_tokens(failure) == [
        "elf_linkage_timeout",
        "libcuda_runtime.so.1",
        "liboptional.so.2",
        "omni.missing",
    ]


def test_runtime_carrier_validation_diagnostic_marker_is_strictly_bounded() -> None:
    failure = subprocess.CalledProcessError(
        returncode=1,
        cmd=["docker", "run"],
        output=(
            "BLUEPRINT_VALIDATION_DIAGNOSTIC elf_read_error\n"
            "BLUEPRINT_VALIDATION_DIAGNOSTIC ../../secret\n"
            "BLUEPRINT_VALIDATION_DIAGNOSTIC value with spaces\n"
        ),
    )

    assert executor._validation_diagnostic_tokens(failure) == ["elf_read_error"]


def test_runtime_carrier_import_failure_defers_only_pure_driver_sonames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")

    class DriverDeferredDocker(FakeDocker):
        def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
            command = list(argv)
            if command[1] == "run" and "import diffusers" in command[-1]:
                self.calls.append(command)
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    stderr="libcuda.so.1: cannot open shared object file\n",
                )
            return super().__call__(argv, **kwargs)

    result = executor.prepare_runtime_bundle(
        _request(),
        runtime_root=tmp_path / "runtime",
        build_root=tmp_path / "build",
        runner=DriverDeferredDocker(),
    )

    assert result["status"] == "completed"
    assert result["carrier_validation"]["status"] == "passed_with_gpu_driver_deferred"
    assert result["carrier_validation"]["gpu_driver_deferred_sonames"] == [
        "libcuda.so.1"
    ]
    assert any(
        row["name"] == "oscar_import_matrix"
        and row["status"] == "deferred_to_gpu_driver_bootstrap"
        for row in result["carrier_validation"]["checks"]
    )


def test_runtime_carrier_import_failure_rejects_mixed_driver_and_module_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/usr/bin/{name}")

    class MixedFailureDocker(FakeDocker):
        def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
            command = list(argv)
            if command[1] == "run" and "import diffusers" in command[-1]:
                self.calls.append(command)
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    stderr=(
                        "libcuda.so.1: cannot open shared object file\n"
                        "ModuleNotFoundError: No module named 'yaml'\n"
                    ),
                )
            return super().__call__(argv, **kwargs)

    with pytest.raises(RuntimeError, match="oscar_import_matrix"):
        executor.prepare_runtime_bundle(
            _request(),
            runtime_root=tmp_path / "runtime",
            build_root=tmp_path / "build",
            runner=MixedFailureDocker(),
        )


def test_runtime_carrier_validation_soname_extraction_is_bounded_and_linear() -> None:
    hostile = "-." * 100_000
    text = (
        f"{hostile}.so => not found\n"
        "loader: error while loading shared libraries: libyaml-cpp.so.0.8: "
        "cannot open shared object file\n"
        "/unsafe/path/libescape.so.1 => not found\n"
    )

    assert executor._missing_soname_tokens(text) == {"libyaml-cpp.so.0.8"}
