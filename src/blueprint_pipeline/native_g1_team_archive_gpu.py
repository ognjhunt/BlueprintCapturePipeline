"""Observe and prove bounded CUDA access inside an archive policy namespace.

This is device/memory access proof, never learned inference or paid admission.
The selected VM and its watchdog/rights remain the canonical caller's concern.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION

BINDING_SCHEMA = "native_g1_team_archive_gpu_binding.v1"
PROBE_SCHEMA = "native_g1_team_archive_gpu_namespace_probe.v1"
PROBE_FILENAME = PROBE_SCHEMA + ".json"
PRIVATE_LOG_FILENAME = "archive_gpu_probe.private.stderr.log"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z", re.I)
_PATHS = ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")

# Stable versioned driver symbols: no Torch/model/site input in this probe.
CUDA_PROBE = '''
import ctypes as c, json
cuda = c.CDLL('libcuda.so.1')
def call(name, types, *arguments):
    function = getattr(cuda, name)
    function.argtypes, function.restype = types, c.c_int
    result = function(*arguments)
    if result != 0:
        raise RuntimeError('cuda_driver_error:' + name + ':' + str(result))
call('cuInit', [c.c_uint], 0)
count, version, device = c.c_int(), c.c_int(), c.c_int()
call('cuDeviceGetCount', [c.POINTER(c.c_int)], c.byref(count))
assert count.value == 1, 'cuda_probe_requires_one_visible_gpu'
call('cuDriverGetVersion', [c.POINTER(c.c_int)], c.byref(version))
call('cuDeviceGet', [c.POINTER(c.c_int), c.c_int], c.byref(device), 0)
class UUID(c.Structure):
    _fields_ = [('bytes', c.c_ubyte * 16)]
uuid = UUID()
call('cuDeviceGetUuid_v2', [c.POINTER(UUID), c.c_int], c.byref(uuid), device)
context, pointer = c.c_void_p(), c.c_uint64()
created = allocated = freed = destroyed = False
try:
    call('cuCtxCreate_v2', [c.POINTER(c.c_void_p), c.c_uint, c.c_int], c.byref(context), 0, device)
    created = True
    call('cuMemAlloc_v2', [c.POINTER(c.c_uint64), c.c_size_t], c.byref(pointer), 16)
    allocated = True
    call('cuMemsetD8_v2', [c.c_uint64, c.c_ubyte, c.c_size_t], pointer, 90, 16)
    result = (c.c_ubyte * 16)()
    call('cuMemcpyDtoH_v2', [c.c_void_p, c.c_uint64, c.c_size_t], result, pointer, 16)
    assert bytes(result) == b'Z' * 16, 'cuda_probe_memory_roundtrip_failed'
finally:
    try:
        if allocated:
            call('cuMemFree_v2', [c.c_uint64], pointer)
            freed = True
    finally:
        if created:
            call('cuCtxDestroy_v2', [c.c_void_p], context)
            destroyed = True
print(json.dumps({'visible_gpu_count': count.value, 'gpu_uuid_hex': bytes(uuid.bytes).hex(),
    'cuda_driver_api_version': version.value, 'memory_roundtrip_verified': True,
    'memory_freed': freed, 'context_destroyed': destroyed}))
'''


def _gpu_identity() -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False, timeout=20)
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise ValueError("g1_archive_gpu_identity_unavailable")
    lines = [row.strip().split(",") for row in result.stdout.splitlines() if row.strip()]
    if len(lines) != 1 or len(lines[0]) != 3 or lines[0][0].strip() != "0":
        raise ValueError("g1_archive_requires_one_full_gpu")
    return {"gpu_index": 0, "gpu_uuid": lines[0][1].strip(), "driver_version": lines[0][2].strip()}


def _uvm_major() -> int:
    rows = [line.split() for line in Path("/proc/devices").read_text().splitlines()]
    values = [int(row[0]) for row in rows if len(row) == 2 and row[0].isdigit() and row[1] == "nvidia-uvm"]
    if len(values) != 1 or not 1 <= values[0] <= 4095:
        raise ValueError("g1_archive_gpu_uvm_driver_unavailable")
    return values[0]


def _device_record(path: str, major: int, minor: int) -> dict[str, Any]:
    node = os.lstat(path)
    if (not stat.S_ISCHR(node.st_mode) or node.st_uid != 0
            or stat.S_IMODE(node.st_mode) & 0o006 != 0o006
            or os.major(node.st_rdev) != major or os.minor(node.st_rdev) != minor):
        raise ValueError("g1_archive_gpu_device_invalid")
    return {"path": path, "major": major, "minor": minor, "uid": node.st_uid,
            "mode": stat.S_IMODE(node.st_mode), "inode": node.st_ino, "filesystem_device": node.st_dev}


def observe_archive_gpu_binding(*, profile_digest: str, execution_packet_digest: str) -> dict[str, Any]:
    identity = _gpu_identity()
    uvm = _uvm_major()
    value = {"schema_version": BINDING_SCHEMA, "profile_digest": profile_digest,
             "execution_packet_digest": execution_packet_digest, **identity,
             "devices": [_device_record(path, major, minor) for path, major, minor in
                         zip(_PATHS, (195, 195, uvm), (0, 255, 0))],
             "guest_gpu_inference_verified": False, "claim_ceiling": "development_only"}
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    validate_archive_gpu_binding(value, profile_digest=profile_digest, execution_packet_digest=execution_packet_digest)
    return value


def validate_archive_gpu_binding(value: Mapping[str, Any], *, profile_digest: str,
                                 execution_packet_digest: str) -> None:
    if (not isinstance(value, Mapping) or set(value) != {"schema_version", "profile_digest", "execution_packet_digest",
            "gpu_index", "gpu_uuid", "driver_version", "devices", "guest_gpu_inference_verified", "claim_ceiling", "receipt_digest"}
            or value.get("schema_version") != BINDING_SCHEMA
            or not _SHA.fullmatch(str(profile_digest)) or not _SHA.fullmatch(str(execution_packet_digest))
            or value.get("profile_digest") != profile_digest or value.get("execution_packet_digest") != execution_packet_digest
            or type(value.get("gpu_index")) is not int or value["gpu_index"] != 0
            or not _UUID.fullmatch(str(value.get("gpu_uuid")))
            or not re.fullmatch(r"\d+\.\d+\.\d+", str(value.get("driver_version")))
            or tuple(int(x) for x in value["driver_version"].split(".")) < tuple(int(x) for x in NATIVE_TASK_ARENA_MINIMUM_DRIVER_VERSION.split("."))
            or value.get("guest_gpu_inference_verified") is not False or value.get("claim_ceiling") != "development_only"
            or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")):
        raise ValueError("g1_archive_gpu_binding_invalid")
    devices = value.get("devices")
    if not isinstance(devices, list) or len(devices) != 3:
        raise ValueError("g1_archive_gpu_binding_invalid")
    for index, row in enumerate(devices):
        if (not isinstance(row, dict) or set(row) != {"path", "major", "minor", "uid", "mode", "inode", "filesystem_device"}
                or row.get("path") != _PATHS[index]
                or any(type(row.get(key)) is not int for key in ("major", "minor", "uid", "mode", "inode", "filesystem_device"))
                or row["uid"] != 0 or row["mode"] & 0o006 != 0o006 or not 0 <= row["mode"] <= 0o777
                or row["inode"] <= 0 or row["filesystem_device"] < 0
                or row["minor"] != (0, 255, 0)[index]
                or (row["major"] != 195 if index < 2 else not 1 <= row["major"] <= 4095)):
            raise ValueError("g1_archive_gpu_binding_invalid")


def recheck_archive_gpu_binding(binding: Mapping[str, Any]) -> None:
    validate_archive_gpu_binding(binding, profile_digest=binding["profile_digest"], execution_packet_digest=binding["execution_packet_digest"])
    fresh = observe_archive_gpu_binding(profile_digest=binding["profile_digest"], execution_packet_digest=binding["execution_packet_digest"])
    if fresh != binding:
        raise ValueError("g1_archive_gpu_binding_changed")


def validate_archive_gpu_probe(value: Mapping[str, Any], *, binding: Mapping[str, Any]) -> None:
    validate_archive_gpu_binding(binding, profile_digest=binding["profile_digest"], execution_packet_digest=binding["execution_packet_digest"])
    expected = {"gpu_uuid_hex": binding["gpu_uuid"].removeprefix("GPU-").replace("-", "").lower(),
                "visible_gpu_count": 1, "memory_roundtrip_verified": True, "memory_freed": True, "context_destroyed": True}
    observed = value.get("observed") if isinstance(value, Mapping) else None
    if (not isinstance(value, Mapping) or set(value) != {"schema_version", "status", "binding_digest", "observed",
            "probe_source_sha256", "guest_gpu_inference_verified", "claim_ceiling", "receipt_digest"}
            or value.get("schema_version") != PROBE_SCHEMA or value.get("status") != "cuda_device_memory_access_observed"
            or value.get("binding_digest") != binding["receipt_digest"]
            or value.get("probe_source_sha256") != "sha256:" + hashlib.sha256(CUDA_PROBE.encode()).hexdigest()
            or value.get("guest_gpu_inference_verified") is not False or value.get("claim_ceiling") != "development_only"
            or not isinstance(observed, dict) or set(observed) != set(expected) | {"cuda_driver_api_version"}
            or any(type(observed.get(key)) is not type(item) or observed[key] != item for key, item in expected.items())
            or type(observed.get("cuda_driver_api_version")) is not int or observed["cuda_driver_api_version"] < 12080
            or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")):
        raise ValueError("g1_archive_gpu_namespace_probe_invalid")


def probe_archive_gpu_namespace(*, command: list[str], binding: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    recheck_archive_gpu_binding(binding)
    prefix = command[:command.index("--")]
    controls = {"--uid": "65534", "--gid": "65534", "--cap-drop": "ALL"}
    if (any(prefix.count(flag) != 1 for flag in ("--unshare-all", "--clearenv", *controls))
            or any(prefix.index(flag) + 1 >= len(prefix) or prefix[prefix.index(flag) + 1] != item
                   for flag, item in controls.items())
            or any(flag in prefix for flag in ("--share-net", "--share-user", "--cap-add", "--bind", "--bind-try"))):
        raise ValueError("g1_archive_gpu_namespace_command_invalid")
    observed_devices = [prefix[i + 1:i + 3] for i, item in enumerate(prefix) if item == "--dev-bind"]
    if (observed_devices != [[path, path] for path in _PATHS] or "--unshare-all" not in prefix
            or "--clearenv" not in prefix):
        raise ValueError("g1_archive_gpu_namespace_command_invalid")
    directory_mounts = [prefix[i + 1:i + 3] for i, item in enumerate(prefix) if item == "--ro-bind-fd"]
    try:
        if (len(directory_mounts) != 1 or len(directory_mounts[0]) != 2
                or directory_mounts[0][1] != "/work"
                or not re.fullmatch(r"[0-9]+", directory_mounts[0][0])):
            raise ValueError("g1_archive_gpu_namespace_command_invalid")
        artifact_fd = int(directory_mounts[0][0])
        if artifact_fd < 3 or not stat.S_ISDIR(os.fstat(artifact_fd).st_mode):
            raise ValueError("g1_archive_gpu_namespace_command_invalid")
    except OSError as exc:
        raise ValueError("g1_archive_gpu_namespace_command_invalid") from exc
    try:
        result = subprocess.run([*prefix, "--", "/usr/bin/python3", "-I", "-B", "-S", "-c", CUDA_PROBE],
            capture_output=True, text=True, check=False, timeout=30,
            pass_fds=(artifact_fd,), close_fds=True,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"})
        with os.fdopen(os.open(output_dir / PRIVATE_LOG_FILENAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as stream:
            stream.write(result.stderr)
        if result.returncode != 0 or len(result.stdout) > 8192:
            raise ValueError("g1_archive_gpu_namespace_probe_failed")
        observed = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ValueError("g1_archive_gpu_namespace_probe_failed") from exc
    value = {"schema_version": PROBE_SCHEMA, "status": "cuda_device_memory_access_observed",
             "binding_digest": binding["receipt_digest"], "observed": observed,
             "probe_source_sha256": "sha256:" + hashlib.sha256(CUDA_PROBE.encode()).hexdigest(),
             "guest_gpu_inference_verified": False, "claim_ceiling": "development_only"}
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    validate_archive_gpu_probe(value, binding=binding)
    with (output_dir / PROBE_FILENAME).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return value
