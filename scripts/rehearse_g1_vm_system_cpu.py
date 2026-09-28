#!/usr/bin/env python3
"""ADP-050 Day28: bounded local CPU inspection of the exact Vast guest disk.

Local artifact writer and public immutable-asset reader. No provider allocation,
GPU, captured input, policy inference, package installation or cleanup.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request

IMAGE_SHA = "sha256:28dc36f977d4a078ee410caf08f595d91f95185a00e0d4e7970c2d11f7358738"
LAYER_SHA = "sha256:de25e09c332152ccf749d454abe78b777530344e99d25580d6d204d64bd00619"
LAYER_BYTES = 2_633_977_898
DISK_BYTES = 5_196_152_832
OVERLAY_LIMIT = 512 * 1024**2
LOG_LIMIT = 32 * 1024**2
FREE_FLOOR = 8_000_000_000
TERMINAL = "BLUEPRINT_G1_VM_CPU_RESULT:"


def file_sha(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("g1_vm_cpu_asset_not_regular")
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require_capacity(path, *, additional_bytes):
    if shutil.disk_usage(path).free < additional_bytes + FREE_FLOOR:
        raise ValueError("g1_vm_cpu_capacity_insufficient")


def extract_guest_disk(layer, destination, *, expected_sha256, expected_layer_bytes, expected_disk_bytes):
    if (destination.exists() or destination.is_symlink() or
            layer.lstat().st_size != expected_layer_bytes or file_sha(layer) != expected_sha256):
        raise ValueError("g1_vm_cpu_layer_binding_invalid")
    with tarfile.open(layer, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) - {"root", "root/images", "root/images/ubuntu.img"}:
            raise ValueError("g1_vm_cpu_layer_members_invalid")
        disks = [member for member in members if member.name == "root/images/ubuntu.img"]
        if (len(disks) != 1 or not disks[0].isfile() or disks[0].size != expected_disk_bytes
                or any(not member.isdir() for member in members if member.name != "root/images/ubuntu.img")):
            raise ValueError("g1_vm_cpu_guest_disk_invalid")
        require_capacity(destination.parent, additional_bytes=expected_disk_bytes + OVERLAY_LIMIT + LOG_LIMIT)
        with archive.extractfile(disks[0]) as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=4 * 1024**2)
        destination.chmod(0o400)
    if destination.stat().st_size != expected_disk_bytes:
        raise ValueError("g1_vm_cpu_guest_disk_size_mismatch")
    return file_sha(destination)


def _registry_request(path, token):
    return urllib.request.Request("https://registry-1.docker.io/v2/vastai/kvm/" + path,
                                  headers={"Authorization": "Bearer " + token,
                                           "Accept": "application/vnd.docker.distribution.manifest.v2+json"})


def download_layer(path):
    require_capacity(path.parent, additional_bytes=LAYER_BYTES + DISK_BYTES + OVERLAY_LIMIT + LOG_LIMIT)
    query = urllib.parse.urlencode({"service": "registry.docker.io", "scope": "repository:vastai/kvm:pull"})
    with urllib.request.urlopen("https://auth.docker.io/token?" + query, timeout=30) as response:
        token = json.load(response)["token"]
    with urllib.request.urlopen(_registry_request("manifests/" + IMAGE_SHA, token), timeout=30) as response:
        encoded = response.read(2 * 1024**2 + 1)
    if len(encoded) > 2 * 1024**2 or "sha256:" + hashlib.sha256(encoded).hexdigest() != IMAGE_SHA:
        raise ValueError("g1_vm_cpu_manifest_binding_invalid")
    manifest = json.loads(encoded)
    if not any(row.get("digest") == LAYER_SHA and row.get("size") == LAYER_BYTES for row in manifest["layers"]):
        raise ValueError("g1_vm_cpu_layer_not_in_manifest")
    (path.parent / "image-manifest.json").write_bytes(encoded)
    started, count, reported = time.monotonic(), 0, 0
    with urllib.request.urlopen(_registry_request("blobs/" + LAYER_SHA, token), timeout=45) as response, path.open("xb") as output:
        while True:
            chunk = response.read(4 * 1024**2)
            if not chunk:
                break
            count += len(chunk)
            if count > LAYER_BYTES or time.monotonic() - started > 900:
                raise ValueError("g1_vm_cpu_download_bound_exceeded")
            require_capacity(path.parent, additional_bytes=LAYER_BYTES - count + DISK_BYTES + OVERLAY_LIMIT + LOG_LIMIT)
            output.write(chunk)
            if count - reported >= 128 * 1024**2:
                print(json.dumps({"stage": "immutable_layer_download", "bytes": count, "expected": LAYER_BYTES}), flush=True)
                reported = count
    path.chmod(0o400)
    if count != LAYER_BYTES or file_sha(path) != LAYER_SHA:
        raise ValueError("g1_vm_cpu_download_binding_invalid")


def guest_probe_source():
    return '''import hashlib,json,os,platform,subprocess
from pathlib import Path
result={'schema_version':'g1_vm_guest_system_cpu_observation.v1','scope':'local_tcg_cpu_only',
 'gpu_runtime_qualified':False,'policy_inference_performed':False,'provider_mutation_performed':False,
 'claim_ceiling':'development_only','uid':os.geteuid(),'platform':platform.machine(),'probes':{}}
commands={'os_release':['cat','/etc/os-release'],'kernel':['uname','-r'],
 'packages':['dpkg-query','-W','-f=${Package} ${Version}\\n'],
 'docker_version':['docker','--version'],'docker_runtimes':['docker','info','--format','{{json .Runtimes}}'],
 'nvidia_toolkit':['nvidia-container-cli','--version'],
 'bwrap_version':['bwrap','--version'],'bwrap_features':['bwrap','--help']}
for name,argv in commands.items():
 try:
  child=subprocess.run(argv,capture_output=True,text=True,timeout=20,
   env={'PATH':'/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','LANG':'C'})
  result['probes'][name]={'exit_code':child.returncode,'stdout':child.stdout[:262144],'stderr':child.stderr[:4096]}
 except (OSError,subprocess.TimeoutExpired) as error:
  result['probes'][name]={'error_type':type(error).__name__}
value=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
result['receipt_digest']='sha256:'+hashlib.sha256(value.encode()).hexdigest()
with Path('/dev/ttyS0').open('w') as serial:
 serial.write('BLUEPRINT_G1_VM_CPU_RESULT:'+json.dumps(result,sort_keys=True)+'\\n');serial.flush()
subprocess.run(['systemctl','poweroff'],timeout=30,check=False)
'''


def qemu_command(executable, overlay, seed):
    return [executable, "-machine", "q35,accel=tcg", "-cpu", "max", "-m", "2048", "-smp", "2",
            "-drive", "file=" + str(overlay) + ",format=qcow2,if=virtio",
            "-drive", "file=" + str(seed) + ",format=raw,media=cdrom,readonly=on",
            "-nic", "none", "-nographic", "-no-reboot"]


def read_guest_result(text):
    rows = [line.split(TERMINAL, 1)[1].strip() for line in text.splitlines() if TERMINAL in line]
    if len(rows) != 1:
        raise ValueError("g1_vm_cpu_terminal_receipt_missing_or_duplicate")
    result = json.loads(rows[0])
    if (result.get("schema_version") != "g1_vm_guest_system_cpu_observation.v1"
            or result.get("scope") != "local_tcg_cpu_only" or result.get("gpu_runtime_qualified") is not False
            or result.get("policy_inference_performed") is not False or result.get("provider_mutation_performed") is not False):
        raise ValueError("g1_vm_cpu_terminal_scope_invalid")
    declared = result.pop("receipt_digest", None)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if declared != "sha256:" + hashlib.sha256(encoded).hexdigest():
        raise ValueError("g1_vm_cpu_terminal_digest_invalid")
    result["receipt_digest"] = declared
    return result


def _command(argv, log):
    with log.open("xb") as output:
        result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=120, check=False)
    if result.returncode != 0:
        raise ValueError("g1_vm_cpu_setup_command_failed")


def _stop(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=20)


def run(*, root, implementation_commit):
    if (not re.fullmatch(r"[0-9a-f]{40}", implementation_commit) or not root.is_absolute()
            or root.exists() or root.is_symlink() or root.parent.resolve() != root.parent):
        raise ValueError("g1_vm_cpu_fresh_stage_required")
    checkout = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip()
    if head != implementation_commit or dirty:
        raise ValueError("g1_vm_cpu_immutable_source_required")
    require_capacity(root.parent, additional_bytes=LAYER_BYTES + DISK_BYTES + OVERLAY_LIMIT + LOG_LIMIT)
    binaries = {name: shutil.which(name) for name in ("qemu-img", "qemu-system-x86_64", "hdiutil")}
    if not all(binaries.values()):
        raise ValueError("g1_vm_cpu_local_tools_missing")
    root.mkdir(mode=0o700)
    result = {"schema_version": "g1_vm_system_cpu_preflight.v1", "status": "blocked",
              "implementation_commit": implementation_commit, "image_ref": "docker.io/vastai/kvm@" + IMAGE_SHA,
              "scope": "local_tcg_cpu_only", "gpu_runtime_qualified": False,
              "policy_inference_performed": False, "provider_mutation_performed": False,
              "claim_ceiling": "development_only", "stage": "layer_download"}
    child = None
    try:
        layer, base = root / "guest-layer.tar.gz", root / "base.qcow2"
        download_layer(layer)
        result["stage"] = "guest_disk_extract"
        print(json.dumps({"stage": result["stage"]}), flush=True)
        result["guest_disk_sha256"] = extract_guest_disk(layer, base, expected_sha256=LAYER_SHA,
                                                       expected_layer_bytes=LAYER_BYTES, expected_disk_bytes=DISK_BYTES)
        result["layer_sha256"] = LAYER_SHA
        result["stage"] = "seed_and_overlay"
        seed_root = root / "seed"
        seed_root.mkdir(mode=0o700)
        source = guest_probe_source()
        encoded = base64.b64encode(source.encode()).decode()
        (seed_root / "meta-data").write_text("instance-id: g1-cpu-" + implementation_commit[:16] + "\nlocal-hostname: g1-cpu-inspection\n")
        (seed_root / "network-config").write_text("version: 2\nethernets: {}\n")
        (seed_root / "user-data").write_text(
            "#cloud-config\nnetwork: {config: disabled}\nwrite_files:\n"
            "  - path: /root/blueprint_cpu_probe.py\n    permissions: '0600'\n    encoding: b64\n    content: " + encoded
            + "\nruncmd:\n  - [python3, -I, -B, -S, /root/blueprint_cpu_probe.py]\n")
        seed, overlay = root / "seed.iso", root / "overlay.qcow2"
        _command([binaries["hdiutil"], "makehybrid", "-o", str(seed), "-iso", "-joliet",
                  "-default-volume-name", "CIDATA", str(seed_root)], root / "seed.private.log")
        _command([binaries["qemu-img"], "create", "-f", "qcow2", "-F", "qcow2", "-b", str(base), str(overlay)], root / "overlay.private.log")
        require_capacity(root, additional_bytes=OVERLAY_LIMIT + LOG_LIMIT)
        argv = qemu_command(binaries["qemu-system-x86_64"], overlay, seed)
        result["command"] = argv
        result["qemu_executable_sha256"] = file_sha(Path(binaries["qemu-system-x86_64"]).resolve())
        result["guest_probe_sha256"] = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
        result["stage"] = "guest_cpu_boot"
        log = root / "guest-serial.private.log"
        with log.open("xb") as output:
            child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
            result["pid"] = child.pid
            (root / "running.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"stage": result["stage"], "pid": child.pid}), flush=True)
            started = time.monotonic()
            while child.poll() is None:
                require_capacity(root, additional_bytes=0)
                if (time.monotonic() - started > 900 or overlay.stat().st_size > OVERLAY_LIMIT
                        or log.stat().st_size > LOG_LIMIT):
                    raise ValueError("g1_vm_cpu_child_resource_bound_exceeded")
                time.sleep(1)
            result["exit_code"] = child.returncode
        if child.returncode != 0:
            raise ValueError("g1_vm_cpu_guest_child_failed")
        result["guest_observation"] = read_guest_result(log.read_text(errors="replace"))
        if file_sha(base) != result["guest_disk_sha256"]:
            raise ValueError("g1_vm_cpu_base_image_changed")
        result["status"] = "guest_cpu_observed"
    except Exception as exc:
        message = str(exc)
        result["blocker_code"] = message if re.fullmatch(r"g1_vm_cpu_[a-z_]+", message) else "g1_vm_cpu_stage_failed"
        result["error_type"] = type(exc).__name__
    finally:
        if child is not None:
            _stop(child)
            result["child_terminal"] = child.poll() is not None
        result["available_bytes_after"] = shutil.disk_usage(root).free
        value = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result["receipt_digest"] = "sha256:" + hashlib.sha256(value).hexdigest()
        (root / "g1_vm_system_cpu_preflight.v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "stage", "receipt_digest")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--execute-local-cpu", action="store_true", required=True)
    args = parser.parse_args()
    result = run(root=args.root, implementation_commit=args.implementation_commit)
    return 0 if result["status"] == "guest_cpu_observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
