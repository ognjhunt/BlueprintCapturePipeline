"""ADP-050 Day28: offline CPU installation in the exact isolated guest only.

No provider/CLI entrypoint, GPU, Docker image pull, policy or approval grant.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

KERNEL = "5.15.0-1067-kvm"
BWRAP_OPTIONS = (
    "--cap-drop", "--chdir", "--clearenv", "--dev", "--dev-bind", "--die-with-parent",
    "--dir", "--gid", "--new-session", "--proc", "--ro-bind", "--ro-bind-fd", "--setenv",
    "--tmpfs", "--uid", "--unshare-all", "--unsetenv",
)
NAMESPACE_PROBE = """import json,os
from pathlib import Path
status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
interfaces=sorted(line.split(':',1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:])
print(json.dumps({'uid':os.geteuid(),'gid':os.getegid(),'cap_eff':status['CapEff'].strip(),'interfaces':interfaces},sort_keys=True))
"""
ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
       "LANG": "C", "DEBIAN_FRONTEND": "noninteractive"}


def validate_cpu_guest_context(value):
    if (value.get("system") != "Linux" or value.get("architecture") != "x86_64"
            or value.get("kernel") != KERNEL or value.get("uid") != 0):
        raise ValueError("g1_vm_system_install_guest_context_invalid")
    interfaces = value.get("interfaces")
    if interfaces not in (["lo"], ["docker0", "lo"]):
        raise ValueError("g1_vm_system_install_guest_network_invalid")
    if "docker0" in interfaces and value.get("docker_bridge") != {
            "is_bridge": True, "hardware_device": False, "carrier": "0", "operstate": "down"}:
        raise ValueError("g1_vm_system_install_guest_network_invalid")
    routes = value.get("routes")
    if not isinstance(routes, list) or any(
            not isinstance(row, dict) or row.get("dst") != "172.17.0.0/16"
            or row.get("dev") != "docker0" or "docker0" not in interfaces
            or "linkdown" not in row.get("flags", []) for row in routes):
        raise ValueError("g1_vm_system_install_guest_routes_invalid")
    if not any(row.get("device") == "/dev/vdb" and row.get("target") == "/run/blueprint-cpu-seed"
               and row.get("filesystem") == "iso9660"
               and {"ro", "nodev", "nosuid", "noexec"} <= set(row.get("flags", []))
               for row in value.get("seed_mounts", [])):
        raise ValueError("g1_vm_system_install_guest_seed_mount_invalid")


def require_cpu_guest():
    """Observe the exact offline guest; a disconnected Docker bridge is internal."""
    value = {"system": platform.system(), "architecture": platform.machine(),
             "kernel": platform.release(), "uid": os.geteuid()}
    try:
        if value != {"system": "Linux", "architecture": "x86_64", "kernel": KERNEL, "uid": 0}:
            raise ValueError("g1_vm_system_install_guest_context_invalid")
        value["interfaces"] = sorted(p.name for p in Path("/sys/class/net").iterdir())
        value["docker_bridge"] = None
        if "docker0" in value["interfaces"]:
            bridge = Path("/sys/class/net/docker0")
            value["docker_bridge"] = {"is_bridge": (bridge / "bridge").is_dir(),
                                      "hardware_device": (bridge / "device").exists(),
                                      "carrier": (bridge / "carrier").read_text().strip(),
                                      "operstate": (bridge / "operstate").read_text().strip()}
        child = subprocess.run(["ip", "-j", "route"], capture_output=True, text=True,
                               timeout=10, check=True, env=ENV)
        if len(child.stdout) > 65536:
            raise ValueError("g1_vm_system_install_guest_routes_invalid")
        value["routes"] = json.loads(child.stdout)
        mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
        value["seed_mounts"] = [{"device": row[0], "target": row[1], "filesystem": row[2],
                                 "flags": row[3].split(",")} for row in mounts
                                if len(row) >= 4 and row[1] == "/run/blueprint-cpu-seed"]
        validate_cpu_guest_context(value)
        return value
    except Exception as error:
        error.guest_context = value
        raise


def validate_namespace_observation(text):
    value = json.loads(text)
    if value != {"uid": 65534, "gid": 65534, "cap_eff": "0000000000000000", "interfaces": ["lo"]}:
        raise ValueError("g1_vm_system_install_namespace_invalid")


def _command(argv, *, timeout=30, pass_fds=()):
    with tempfile.TemporaryFile() as output:
        child = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=timeout,
                               check=False, env=ENV, pass_fds=pass_fds, stdin=subprocess.DEVNULL)
        size = output.tell()
        output.seek(max(0, size - 65536))
        return {"exit_code": child.returncode, "stdout": output.read(65536).decode(errors="replace"),
                "output_bytes": size, "output_truncated": size > 65536}


def _namespace_command(fd):
    argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--proc", "/proc",
            "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc", "--ro-bind-fd", str(fd), "/work",
            "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL", "--clearenv"]
    for name in ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache"):
        if Path(name).exists():
            argv.extend(["--ro-bind", name, name])
    argv.extend(["--chdir", "/work", "--setenv", "PATH", "/usr/bin:/bin", "--unsetenv", "PYTHONPATH",
                 "--", "/usr/bin/python3", "-I", "-B", "-S", "-c", NAMESPACE_PROBE])
    return argv


def rehearse_offline_installation(root, *, implementation_commit, system_helpers):
    result = {"schema_version": "g1_vm_system_cpu_installation.v1", "status": "blocked",
              "scope": "local_tcg_cpu_only", "stage": "guest_context", "probes": {},
              "runtime_installation_attempted": False, "gpu_runtime_qualified": False,
              "provider_mutation_performed": False, "policy_inference_performed": False,
              "claim_ceiling": "development_only"}

    def observe(stage, argv, **kwargs):
        result["stage"] = stage
        print("BLUEPRINT_G1_VM_CPU_STAGE:" + stage, flush=True)
        observed = _command(argv, **kwargs)
        result["probes"][stage] = observed
        if observed["exit_code"] != 0:
            raise ValueError("g1_vm_system_install_command_failed")
        return observed["stdout"]

    try:
        result["guest_context"] = require_cpu_guest()
        result["stage"] = "package_verification"
        argv = system_helpers["offline_apt_simulation_command"](
            root, expected_implementation_commit=implementation_commit)
        argv = [part for part in argv if part != "--simulate"]
        argv.insert(1, "--yes")
        result["runtime_installation_attempted"] = True
        observe("offline_install", argv, timeout=1800)
        packages = system_helpers["SYSTEM_PACKAGES"]
        text = observe("installed_packages", ["dpkg-query", "-W", "-f=${Status}\t${Package}\t${Version}\n",
                                              *(row["package"] for row in packages.values())])
        expected = {"install ok installed\t" + row["package"] + "\t" + row["version"] for row in packages.values()}
        if set(text.splitlines()) != expected or len(text.splitlines()) != len(expected):
            raise ValueError("g1_vm_system_install_package_versions_invalid")
        if observe("kernel_module", ["modinfo", "-k", KERNEL, "-F", "version", "nvidia"]).strip() != "580.65.06":
            raise ValueError("g1_vm_system_install_kernel_module_invalid")
        help_text = observe("bwrap_help", ["bwrap", "--help"])
        if not set(BWRAP_OPTIONS) <= set(re.findall(r"--[a-z-]+", help_text)):
            raise ValueError("g1_vm_system_install_bwrap_features_invalid")
        result["stage"] = "archive_namespace"
        with tempfile.TemporaryDirectory(prefix="blueprint-namespace-") as name:
            directory = Path(name)
            directory.chmod(0o755)
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                validate_namespace_observation(observe("archive_namespace", _namespace_command(fd), pass_fds=(fd,)))
            finally:
                os.close(fd)
        observe("toolkit_configure", ["nvidia-ctk", "runtime", "configure", "--runtime=docker"])
        observe("docker_restart", ["systemctl", "restart", "docker"], timeout=60)
        runtimes = json.loads(observe("docker_runtimes", ["docker", "info", "--format", "{{json .Runtimes}}"], timeout=60))
        if not isinstance(runtimes, dict) or not isinstance(runtimes.get("nvidia"), dict):
            raise ValueError("g1_vm_system_install_docker_runtime_invalid")
        result["status"] = "cpu_installation_observed"
    except Exception as error:
        if isinstance(getattr(error, "guest_context", None), dict):
            result["guest_context"] = error.guest_context
        code = str(error)
        result["blocker_code"] = code if re.fullmatch(r"g1_vm_system_[a-z_]+", code) else "g1_vm_system_install_probe_failed"
        result["error_type"] = type(error).__name__
    result["receipt_digest"] = "sha256:" + hashlib.sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result
