"""Stdlib transport for a sealed, selected G1 VM bundle (ADP-050 Day28).

This module is embedded in the canonical provider's on-start script. It observes
system prerequisites and transports immutable bytes; it does not install a VM
stack, allocate resources, grade an episode, or reconcile provider billing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
from urllib.parse import urlsplit
import zipfile


RECEIPT_FILENAME = "native_g1_team_vm_transport.v1.json"
ENVIRONMENT_KEYS = (
    "BLUEPRINT_EVAL_MANIFEST_URI", "BLUEPRINT_RUNTIME_DEPENDENCY_URI",
    "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
)
BWRAP_OPTIONS = (
    "cap-drop", "chdir", "clearenv", "dev", "dev-bind", "die-with-parent",
    "dir", "gid", "new-session", "proc", "ro-bind", "ro-bind-fd", "setenv",
    "tmpfs", "uid", "unshare-all", "unsetenv",
)
TRANSFER_SECONDS = 900
MAX_BUNDLE_BYTES = 64 * 1024**3
HOST_ENTRYPOINT = "provider_runtime/run_g1_team_vm_host.sh"
HOST_RECEIPT = "native_g1_team_vm_host_result.v1.json"
EXCLUDED_INPUT = "policy-host/runtime/artifact"


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("vm_transport_sha_invalid")
    return value


def _image(value):
    if (not isinstance(value, str) or
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", value) is None):
        raise ValueError("vm_transport_image_not_pinned")
    return value


def _regular(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("vm_transport_file_not_regular")
    return value


def verify_file(path, sha256, *, size_bytes=None):
    _sha(sha256)
    before = _regular(path)
    if size_bytes is not None and (type(size_bytes) is not int or size_bytes <= 0
                                   or before.st_size != size_bytes):
        raise ValueError("vm_transport_file_size_mismatch")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("vm_transport_file_replaced")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("vm_transport_file_changed")
    if "sha256:" + digest.hexdigest() != sha256:
        raise ValueError("vm_transport_file_sha_mismatch")


def read_environment(current, path=Path("/etc/environment")):
    """Parse allowlisted assignments as literal data, with no shell evaluation."""
    result = {}
    if path.exists() or path.is_symlink():
        metadata = _regular(path)
        if metadata.st_size > 1024 * 1024 or metadata.st_uid != os.geteuid():
            raise ValueError("vm_transport_environment_file_invalid")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key not in ENVIRONMENT_KEYS:
                continue
            words = shlex.split(value, comments=False, posix=True)
            if len(words) != 1 or key in result:
                raise ValueError("vm_transport_environment_assignment_invalid")
            result[key] = words[0]
    for key in ENVIRONMENT_KEYS:
        if key in current:
            result[key] = current[key]
    return result


def extract_bundle(path, destination):
    """Validate the entire member graph before creating a fresh private tree."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("vm_transport_zip_destination_exists")
    _regular(path)
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist()
        if not rows or len(rows) > 100_000 or sum(row.file_size for row in rows) > MAX_BUNDLE_BYTES:
            raise ValueError("vm_transport_zip_limits_exceeded")
        names = {}
        for row in rows:
            name = row.filename.rstrip("/") if row.is_dir() else row.filename
            parts = PurePosixPath(name).parts
            mode = row.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if (not name or name.startswith("/") or "\\" in name or "\x00" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or PurePosixPath(name).as_posix() != name or name in names
                    or row.flag_bits & 1 or row.file_size < 0
                    or kind not in ({0, stat.S_IFDIR} if row.is_dir() else {0, stat.S_IFREG})):
                raise ValueError("vm_transport_zip_member_invalid")
            names[name] = row.is_dir()
            if not parts:
                raise ValueError("vm_transport_zip_member_invalid")
        for name in names:
            for parent in PurePosixPath(name).parents:
                if parent.as_posix() in names and not names[parent.as_posix()]:
                    raise ValueError("vm_transport_zip_parent_collision")
        destination.mkdir(mode=0o700)
        for row in rows:
            target = destination / row.filename
            if row.is_dir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with archive.open(row) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            target.chmod(0o700 if row.external_attr >> 16 & 0o111 else 0o600)


def read_probe(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20,
                                env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("vm_transport_system_probe_unavailable") from exc
    return result.returncode, result.stdout.strip()


def observe_vm(*, pid1_path=Path("/proc/1/comm")):
    if os.geteuid() != 0 or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("vm_transport_host_identity_invalid")
    if pid1_path.read_text().strip() != "systemd":
        raise ValueError("vm_transport_pid1_not_systemd")
    container_rc, container = read_probe(["systemd-detect-virt", "--container"])
    vm_rc, kind = read_probe(["systemd-detect-virt", "--vm"])
    if container_rc != 1 or container != "none" or vm_rc != 0 or kind not in {"kvm", "qemu"}:
        raise ValueError("vm_transport_not_supported_virtual_machine")
    return {"vm_kind": kind, "pid1": "systemd", "architecture": "x86_64", "uid": 0}


def observe_system(mode):
    rc, text = read_probe(["docker", "info", "--format", '{{json .Runtimes}}'])
    try:
        runtimes = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError("vm_transport_docker_unavailable") from exc
    if rc != 0 or not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        raise ValueError("vm_transport_nvidia_container_runtime_missing")
    rc, driver = read_probe(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if (rc != 0 or re.fullmatch(r"\d+\.\d+\.\d+", driver) is None
            or tuple(map(int, driver.split("."))) < (580, 65, 6)):
        raise ValueError("vm_transport_driver_floor_failed")
    result = {"driver_version": driver, "nvidia_container_runtime_observed": True}
    if mode == "noncontainer_artifact":
        version_rc, version = read_probe(["bwrap", "--version"])
        help_rc, help_text = read_probe(["bwrap", "--help"])
        flags = set(re.findall(r"--([a-z][a-z-]*)\b", help_text))
        if version_rc != 0 or not version.startswith("bubblewrap ") or help_rc != 0 or not set(BWRAP_OPTIONS) <= flags:
            raise ValueError("vm_transport_archive_sandbox_features_missing")
        result["archive_sandbox_required_features"] = list(BWRAP_OPTIONS)
    return result


def run_private(argv, *, log_path, timeout):
    """Keep subprocess output private, and kill the child group on timeout."""
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "HOME": "/root"})
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=20)
            raise ValueError("vm_transport_child_timeout") from exc


def download(url, destination, *, sha256, size_bytes=None, log_path):
    code = run_private([
        "curl", "--fail", "--silent", "--show-error", "--http1.1",
        "--connect-timeout", "30", "--max-time", str(TRANSFER_SECONDS),
        "--retry", "2", "--retry-max-time", str(TRANSFER_SECONDS),
        "--output", str(destination), url,
    ], log_path=log_path, timeout=TRANSFER_SECONDS + 15)
    if code != 0:
        raise ValueError("vm_transport_download_failed")
    destination.chmod(0o600)
    verify_file(destination, sha256, size_bytes=size_bytes)


def archive_output(root, destination):
    if not stat.S_ISDIR(root.lstat().st_mode) or root.resolve() != root:
        raise ValueError("vm_transport_output_root_invalid")
    rows = []
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(directories):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative == EXCLUDED_INPUT:
                directories.remove(name)
            elif not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("vm_transport_output_directory_invalid")
        for name in files:
            path = parent / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("vm_transport_output_file_invalid")
            rows.append(path)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(rows):
            before = _regular(path)
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
                if (os.fstat(source.fileno()).st_ino, os.fstat(source.fileno()).st_dev) != (before.st_ino, before.st_dev):
                    raise ValueError("vm_transport_output_replaced")
                with archive.open(path.relative_to(root).as_posix(), "w", force_zip64=True) as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                after = os.fstat(source.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("vm_transport_output_changed")
    destination.chmod(0o600)


def publish_recovery_archive(archive, *, workspace=Path("/workspace")):
    """Preserve the canonical adapter's fixed SSH fallback, with no overwrite."""
    if not workspace.is_absolute() or workspace.parent.resolve() != workspace.parent or workspace.is_symlink():
        raise ValueError("vm_transport_recovery_workspace_invalid")
    workspace.mkdir(mode=0o755, exist_ok=True)
    metadata = workspace.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise ValueError("vm_transport_recovery_workspace_invalid")
    _regular(archive)
    target = workspace / "adp_arena_provider_runtime_output.zip"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output, os.fdopen(os.open(archive, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        shutil.copyfileobj(source, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())


def _json(path):
    if _regular(path).st_size > 4 * 1024 * 1024:
        raise ValueError("vm_transport_json_limits_exceeded")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("vm_transport_json_duplicate_key")
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("vm_transport_json_nonfinite")))
    if not isinstance(value, dict):
        raise ValueError("vm_transport_json_object_required")
    return value


def _inputs(bundle, simulator_image):
    runtime = bundle / "provider_runtime"
    manifest = _json(runtime / "native_g1_team_provider_manifest.json")
    packet = _json(runtime / "inputs/execution_packet.json")
    mode = manifest.get("delivery_mode")
    if (mode not in {"container", "noncontainer_artifact"} or manifest.get("policy_runtime_required") is not True
            or manifest.get("runtime_entrypoint") != HOST_ENTRYPOINT
            or manifest.get("container_image") != simulator_image
            or packet.get("delivery_mode") != mode
            or _sha(manifest.get("execution_packet_digest")) != packet.get("packet_digest")):
        raise ValueError("vm_transport_selected_bundle_binding_invalid")
    delivery = packet["request"]["policy_profile"]["delivery"]
    if delivery.get("mode") != mode:
        raise ValueError("vm_transport_profile_mode_mismatch")
    images = [simulator_image]
    if mode == "container":
        images.append(_image(delivery.get("image_ref")))
    _regular(bundle / HOST_ENTRYPOINT)
    dependency = _json(runtime / "native_task_runtime_sources/native_task_runtime_source_packet.v1.json")
    _sha(dependency.get("packet_sha256"))
    size = dependency.get("packet_size_bytes")
    if type(size) is not int or not 0 < size <= MAX_BUNDLE_BYTES:
        raise ValueError("vm_transport_dependency_size_invalid")
    return mode, images, dependency


def _urls(environment):
    for key in ENVIRONMENT_KEYS:
        value = environment.get(key)
        if not isinstance(value, str) or len(value) > 16384 or any(character.isspace() for character in value):
            raise ValueError("vm_transport_url_invalid")
        parts = urlsplit(value)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise ValueError("vm_transport_url_invalid")


def run_transport(*, root, expected_bundle_sha256, simulator_image, environment, recovery_workspace=None):
    _sha(expected_bundle_sha256)
    _image(simulator_image)
    _urls(environment)
    if not root.is_absolute() or root.exists() or root.is_symlink() or root.parent.resolve() != root.parent:
        raise ValueError("vm_transport_fresh_private_root_required")
    root.mkdir(mode=0o700)
    print("BLUEPRINT_VAST_WORK_DIR:" + str(root), flush=True)
    diagnostics = root / "private_diagnostics"
    diagnostics.mkdir(mode=0o700)
    bundle = root / "bundle"
    output = bundle / "runtime_output"
    result = {"schema_version": "native_g1_team_vm_transport.v1", "status": "blocked",
              "expected_bundle_sha256": expected_bundle_sha256, "gpu_runtime_qualified": False,
              "provider_teardown_verified": False, "official_billing_reconciled": False,
              "claim_ceiling": "development_only", "stage": "vm_identity"}
    try:
        result["vm_observation"] = observe_vm()
        print("BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED", flush=True)
        result["stage"] = "bundle_download"
        path = root / "bundle.zip"
        download(environment[ENVIRONMENT_KEYS[0]], path, sha256=expected_bundle_sha256,
                 log_path=diagnostics / "bundle.private.log")
        result["stage"] = "bundle_extract"
        extract_bundle(path, bundle)
        mode, images, dependency = _inputs(bundle, simulator_image)
        print("BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED", flush=True)
        result["stage"] = "runtime_dependency_download"
        download(environment[ENVIRONMENT_KEYS[1]],
                 bundle / "provider_runtime/native_task_runtime_sources/native_task_runtime_sources.zip",
                 sha256=dependency["packet_sha256"], size_bytes=dependency["packet_size_bytes"],
                 log_path=diagnostics / "dependency.private.log")
        result["stage"] = "system_prerequisites"
        result["system_observation"] = observe_system(mode)
        result["stage"] = "pinned_images"
        for index, image in enumerate(dict.fromkeys(images)):
            if run_private(["docker", "pull", image], log_path=diagnostics / f"image-{index}.private.log", timeout=900) != 0:
                raise ValueError("vm_transport_pinned_image_pull_failed")
        output.mkdir(mode=0o700, parents=True)
        result["stage"] = "vm_host"
        print("BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED", flush=True)
        code = run_private(["bash", str(bundle / HOST_ENTRYPOINT)],
                           log_path=output / "vm-host.private.log", timeout=2700)
        result["entrypoint_exit_code"] = code
        print("BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE:" + str(code), flush=True)
        if code != 0:
            raise ValueError("vm_transport_host_entrypoint_failed")
        _regular(output / HOST_RECEIPT)
        result["status"] = "host_exited"
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        # Keep URLs and arbitrary child messages out of the public receipt.
        code = str(exc)
        result["blocker_code"] = code if re.fullmatch(r"vm_transport_[a-z_]+", code) else "vm_transport_stage_failed"
        result["error_type"] = type(exc).__name__
        print("BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:" + result["blocker_code"], flush=True)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.move(str(diagnostics), output / "private_transport_diagnostics")
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["receipt_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    receipt = output / RECEIPT_FILENAME
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt.chmod(0o600)
    archive = root / "adp_arena_provider_runtime_output.zip"
    archive_output(output, archive)
    recovery_failed = False
    if recovery_workspace is not None:
        try:
            publish_recovery_archive(archive, workspace=recovery_workspace)
        except (ValueError, OSError):
            recovery_failed = True
            print("BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:vm_transport_recovery_publication_failed", flush=True)
    print("BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:" + str(archive.stat().st_size), flush=True)
    code = run_private(["curl", "--fail", "--silent", "--show-error", "--connect-timeout", "30",
                        "--max-time", str(TRANSFER_SECONDS), "--request", "PUT", "--upload-file",
                        str(archive), environment[ENVIRONMENT_KEYS[2]]],
                       log_path=root / "upload.private.log", timeout=TRANSFER_SECONDS + 15)
    if code != 0:
        raise ValueError("vm_transport_output_upload_failed")
    print("BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK", flush=True)
    print("BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED", flush=True)
    if recovery_failed:
        raise ValueError("vm_transport_recovery_publication_failed")
    return result


def vm_probe_script(expected_bundle_sha256, heartbeat_url):
    """The caller embeds this file; the VM imports no Blueprint dependencies."""
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

    _sha(expected_bundle_sha256)
    _urls(dict.fromkeys(ENVIRONMENT_KEYS, heartbeat_url))
    source = Path(__file__).read_text()
    return ("#!/usr/bin/env bash\nset -u\numask 077\n"
            "python3 -I -B -S - --expected-bundle-sha256 " + shlex.quote(expected_bundle_sha256)
            + " --simulator-image " + shlex.quote(NATIVE_TASK_ARENA_IMAGE)
            + " --heartbeat-url " + shlex.quote(heartbeat_url)
            + " <<'BLUEPRINT_G1_VM_TRANSPORT_PY'\n" + source
            + "\nBLUEPRINT_G1_VM_TRANSPORT_PY\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--simulator-image", required=True)
    parser.add_argument("--root", type=Path, default=Path("/root/blueprint-g1-runtime"))
    parser.add_argument("--heartbeat-url", required=True)
    args = parser.parse_args()
    try:
        print("BLUEPRINT_VAST_ONSTART_STARTED", flush=True)
        code, _ = read_probe(["curl", "--fail", "--silent", "--output", "/dev/null",
                              "--connect-timeout", "10", "--max-time", "15", args.heartbeat_url])
        print("BLUEPRINT_VAST_HEARTBEAT_OK" if code == 0 else "BLUEPRINT_VAST_HEARTBEAT_BLOCKED", flush=True)
        result = run_transport(root=args.root, expected_bundle_sha256=args.expected_bundle_sha256,
                               simulator_image=args.simulator_image,
                               environment=read_environment(os.environ), recovery_workspace=Path("/workspace"))
    except (ValueError, OSError) as exc:
        print("BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:vm_transport_failed_" + type(exc).__name__, flush=True)
        return 1
    return 0 if result["status"] == "host_exited" else 1


if __name__ == "__main__":
    raise SystemExit(main())
