"""Fake driver/device observations never prove actual GPU qualification."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_team_archive_gpu as gpu
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

PROFILE = "sha256:" + "a" * 64
PACKET = "sha256:" + "b" * 64
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


@pytest.fixture
def artifact_fd(tmp_path):
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@pytest.fixture
def observed(monkeypatch):
    monkeypatch.setattr(
        gpu,
        "_gpu_identity",
        lambda: {"gpu_index": 0, "gpu_uuid": UUID, "driver_version": "580.65.06"},
    )
    monkeypatch.setattr(gpu, "_uvm_major", lambda: 239)
    monkeypatch.setattr(
        gpu,
        "_device_record",
        lambda path, major, minor: {
            "path": path,
            "major": major,
            "minor": minor,
            "uid": 0,
            "mode": 0o666,
            "inode": minor + 100,
            "filesystem_device": 5,
        },
    )
    return gpu.observe_archive_gpu_binding(profile_digest=PROFILE, execution_packet_digest=PACKET)


def test_observed_binding_is_exact_and_rechecked(observed):
    gpu.validate_archive_gpu_binding(
        observed, profile_digest=PROFILE, execution_packet_digest=PACKET
    )
    gpu.recheck_archive_gpu_binding(observed)
    assert [row["path"] for row in observed["devices"]] == [
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
    ]
    assert observed["guest_gpu_inference_verified"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "digest",
        "gpu",
        "uuid",
        "driver",
        "profile",
        "packet",
        "node",
        "major",
        "minor",
        "private",
        "extra",
        "boolean",
    ],
)
def test_forged_or_changed_binding_refused(observed, fault):
    row = copy.deepcopy(observed)
    if fault == "gpu":
        row["gpu_index"] = 1
    elif fault == "uuid":
        row["gpu_uuid"] = "MIG-unqualified"
    elif fault == "driver":
        row["driver_version"] = "550.54.14"
    elif fault == "profile":
        row["profile_digest"] = "sha256:" + "f" * 64
    elif fault == "packet":
        row["execution_packet_digest"] = "sha256:" + "f" * 64
    elif fault == "node":
        row["devices"][0]["path"] = "/dev/nvidia1"
    elif fault == "major":
        row["devices"][0]["major"] = 1
    elif fault == "minor":
        row["devices"][0]["minor"] = 1
    elif fault == "private":
        row["devices"][0]["mode"] = 0o600
    elif fault == "extra":
        row["devices"].append(copy.deepcopy(row["devices"][0]))
    elif fault == "boolean":
        row["gpu_index"] = False
    if fault != "digest":
        row["receipt_digest"] = canonical_digest(row, digest_field="receipt_digest")
    else:
        row["receipt_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError):
        gpu.validate_archive_gpu_binding(
            row, profile_digest=PROFILE, execution_packet_digest=PACKET
        )


def test_device_must_be_actual_root_owned_public_character_node(monkeypatch):
    monkeypatch.setattr(
        gpu.os,
        "lstat",
        lambda path: SimpleNamespace(
            st_mode=0o020666, st_uid=0, st_rdev=os.makedev(195, 0), st_ino=10, st_dev=5
        ),
    )
    assert gpu._device_record("/dev/nvidia0", 195, 0)["major"] == 195
    for mode, uid, rdev in [
        (0o100666, 0, os.makedev(195, 0)),
        (0o120777, 0, 0),
        (0o020600, 0, os.makedev(195, 0)),
        (0o020666, 1, os.makedev(195, 0)),
        (0o020666, 0, os.makedev(195, 1)),
    ]:
        monkeypatch.setattr(
            gpu.os,
            "lstat",
            lambda path: SimpleNamespace(
                st_mode=mode, st_uid=uid, st_rdev=rdev, st_ino=10, st_dev=5
            ),
        )
        with pytest.raises(ValueError):
            gpu._device_record("/dev/nvidia0", 195, 0)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "extra_gpu",
        "uuid",
        "memory",
        "context",
        "version",
        "boolean",
        "exit",
        "invalid_json",
        "timeout",
    ],
)
def test_same_sandbox_probe_must_prove_exact_device_memory_and_cleanup(
    observed, monkeypatch, tmp_path, fault, artifact_fd
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if fault == "timeout":
            raise gpu.subprocess.TimeoutExpired(command, 30)
        value = {
            "gpu_uuid_hex": UUID.removeprefix("GPU-").replace("-", ""),
            "visible_gpu_count": 1,
            "cuda_driver_api_version": 13000,
            "memory_roundtrip_verified": True,
            "memory_freed": True,
            "context_destroyed": True,
        }
        if fault == "extra_gpu":
            value["visible_gpu_count"] = 2
        elif fault == "uuid":
            value["gpu_uuid_hex"] = "f" * 32
        elif fault == "memory":
            value["memory_roundtrip_verified"] = False
        elif fault == "context":
            value["context_destroyed"] = False
        elif fault == "version":
            value["cuda_driver_api_version"] = 10000
        elif fault == "boolean":
            value["visible_gpu_count"] = True
        return SimpleNamespace(
            returncode=1 if fault == "exit" else 0,
            stdout="bad" if fault == "invalid_json" else json.dumps(value),
            stderr="private probe stderr",
        )

    monkeypatch.setattr(gpu.subprocess, "run", run)
    command = [
        "bwrap",
        "--unshare-all",
        "--clearenv",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "--cap-drop",
        "ALL",
        "--ro-bind-fd", str(artifact_fd), "/work",
    ]
    for row in observed["devices"]:
        command += ["--dev-bind", row["path"], row["path"]]
    command += ["--", "/work/policy/run"]
    if fault is not None:
        with pytest.raises(ValueError):
            gpu.probe_archive_gpu_namespace(command=command, binding=observed, output_dir=tmp_path)
    else:
        receipt = gpu.probe_archive_gpu_namespace(
            command=command, binding=observed, output_dir=tmp_path
        )
        gpu.validate_archive_gpu_probe(receipt, binding=observed)
        assert calls[0][0][:2] == command[:2]
        assert calls[0][0][-6:-3] == ["/usr/bin/python3", "-I", "-B"]
        assert "PYTHONPATH" not in calls[0][1]["env"]
        assert calls[0][1]["pass_fds"] == (artifact_fd,)
        assert calls[0][1]["close_fds"] is True
        assert receipt["guest_gpu_inference_verified"] is False
        assert (tmp_path / gpu.PRIVATE_LOG_FILENAME).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fault", [None, "set", "copy", "free", "destroy", "count", "memory"])
def test_actual_trusted_probe_abi_and_cleanup_without_a_driver(monkeypatch, capsys, fault):
    import ctypes

    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append(self.name)
            name = self.name
            if name == {
                "set": "cuMemsetD8_v2",
                "copy": "cuMemcpyDtoH_v2",
                "free": "cuMemFree_v2",
                "destroy": "cuCtxDestroy_v2",
            }.get(fault):
                return 7
            if name == "cuDeviceGetCount":
                args[0]._obj.value = 2 if fault == "count" else 1
            elif name == "cuDriverGetVersion":
                args[0]._obj.value = 13000
            elif name == "cuDeviceGet":
                args[0]._obj.value = 0
            elif name == "cuDeviceGetUuid_v2":
                args[0]._obj.bytes[:] = bytes.fromhex(UUID.removeprefix("GPU-").replace("-", ""))
            elif name in {"cuCtxCreate_v2", "cuMemAlloc_v2"}:
                args[0]._obj.value = 1234
            elif name == "cuMemcpyDtoH_v2":
                args[0][:] = (b"X" if fault == "memory" else b"Z") * 16
            return 0

    class Driver:
        def __getattr__(self, name):
            return Function(name)

    monkeypatch.setattr(ctypes, "CDLL", lambda name: Driver())
    if fault is None:
        exec(gpu.CUDA_PROBE, {})
        result = json.loads(capsys.readouterr().out)
        assert result["memory_freed"] is True and result["context_destroyed"] is True
        assert result["gpu_uuid_hex"] == UUID.removeprefix("GPU-").replace("-", "")
    else:
        with pytest.raises((RuntimeError, AssertionError)):
            exec(gpu.CUDA_PROBE, {})
    if fault != "count":
        assert calls[-2:] == ["cuMemFree_v2", "cuCtxDestroy_v2"]


@pytest.mark.parametrize("mutation", ["extra_device", "share_net", "capability", "duplicate_uid"])
def test_probe_refuses_widened_namespace_before_execution(
    observed, monkeypatch, tmp_path, mutation, artifact_fd
):
    command = [
        "bwrap",
        "--unshare-all",
        "--clearenv",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "--cap-drop",
        "ALL",
        "--ro-bind-fd", str(artifact_fd), "/work",
    ]
    for row in observed["devices"]:
        command += ["--dev-bind", row["path"], row["path"]]
    command += {
        "extra_device": ["--dev-bind", "/dev/nvidia1", "/dev/nvidia1"],
        "share_net": ["--share-net"],
        "capability": ["--cap-add", "SYS_ADMIN"],
        "duplicate_uid": ["--uid", "0"],
    }[mutation]
    command += ["--", "/work/policy/run"]
    monkeypatch.setattr(
        gpu.subprocess, "run", lambda *args, **kwargs: pytest.fail("widened GPU probe executed")
    )
    with pytest.raises(ValueError, match="namespace_command_invalid"):
        gpu.probe_archive_gpu_namespace(command=command, binding=observed, output_dir=tmp_path)
