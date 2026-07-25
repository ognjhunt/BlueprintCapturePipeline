"""Namespace-independent GPU role attribution for the closed-loop residency proof.

``nvidia-smi`` reports compute-application PIDs from the ROOT pid namespace even
when it runs inside a container. Linux exposes the host-to-local translation via
``NSpid`` in ``/proc/<pid>/status``, but only on runtimes that leak the full
namespace chain outward; on runtimes that do not, a container cannot learn its
own host PIDs *in principle* — the information is absent, not merely unparsed.

Attempt 068 landed on such a host. ``nvidia-smi`` reported three compute apps
holding ~45 GiB on the single visible GPU, every PID unresolvable
(``process_name: "[Not Found]"``), and the residency proof concluded that none
of the four required roles were resident — a false negative that failed a sealed
run at step 1. The defect was the unmapped-PID fallback: an untranslatable host
PID was treated as a local PID, which is indistinguishable from a genuinely
foreign container's process, so no blocker described what had actually happened.

This module supplies the proof from the opposite direction. Rather than asking
the GPU which processes it holds — which requires host-PID attribution — it asks
our own processes whether they hold the GPU. A process with an open descriptor on
``/dev/nvidia*`` has an initialized driver context on a device visible to this
container; when the container's inventory contains exactly one GPU, the device is
unambiguous and same-GPU residency follows. Every lookup is namespace-local, so
the fallback works on hosts where translation is impossible.

Soundness is bounded deliberately: the fallback refuses to conclude anything when
more than one GPU is visible, because an open device handle then cannot identify
*which* GPU a role landed on.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Only a NUMBERED per-GPU node proves a compute context on that GPU. The driver's
# generic nodes -- /dev/nvidiactl, /dev/nvidia-uvm, /dev/nvidia-modeset,
# /dev/nvidia-caps/* -- are opened by any process that merely initializes CUDA or
# probes the driver, so accepting them would let four idle roles attest residency
# they never had. Requiring /dev/nvidia<N> keeps the fallback a proof rather than
# a liveness check.
NVIDIA_GPU_DEVICE_NODE_PATTERN = re.compile(r"^/dev/nvidia(\d+)$")
ATTRIBUTION_MODE_HOST_PID_NAMESPACE = "host_pid_namespace"
ATTRIBUTION_MODE_DEVICE_HANDLE_FALLBACK = "device_handle_fallback"
ATTRIBUTION_MODE_UNAVAILABLE = "unavailable"
PID_TRANSLATION_IDENTITY = "identity"
PID_TRANSLATION_NSPID = "nspid_host_to_local"

MULTI_GPU_BLOCKER = "oscar_gpu_residency_pid_attribution_unavailable_multi_gpu_inventory"
MISSING_HANDLE_BLOCKER = "oscar_gpu_residency_pid_attribution_roles_without_gpu_device_handle"


def positive_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def pid_ancestor_chain(
    pid: int,
    *,
    parent_pid: Callable[[int], int | None],
    max_depth: int = 128,
) -> list[int]:
    chain: list[int] = []
    current = positive_pid(pid)
    visited: set[int] = set()
    while current is not None and current not in visited and len(chain) < max_depth:
        chain.append(current)
        visited.add(current)
        try:
            current = positive_pid(parent_pid(current))
        except Exception:
            current = None
    return chain


def linux_nvidia_host_to_local_pid_map(proc_root: Path = Path("/proc")) -> dict[int, int]:
    """Map root-namespace NVIDIA PIDs to PIDs visible in this container.

    NVIDIA's driver reports compute-application PIDs from the host PID
    namespace even when ``nvidia-smi`` runs inside a container. Linux exposes
    the namespace chain in ``/proc/<local-pid>/status`` as ``NSpid``. The
    first value is the root-namespace PID and the last is the PID visible in
    the current namespace.

    Returns an empty or identity-only map on runtimes that never expose the
    outer namespace; callers must treat that as *unavailable* attribution
    rather than as evidence that no role is resident.
    """

    mapped: dict[int, int] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return mapped
    for entry in entries:
        if not entry.name.isdigit():
            continue
        local_pid = positive_pid(entry.name)
        if local_pid is None:
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for line in status.splitlines():
            if not line.startswith("NSpid:"):
                continue
            namespace_pids = [
                pid
                for value in line.split(":", 1)[1].split()
                if (pid := positive_pid(value)) is not None
            ]
            if namespace_pids:
                # ``/proc`` directory names are already expressed in the PID
                # namespace visible to this process.  Some container runtimes
                # expose the complete NSpid chain (host ... local), while
                # others expose only the outermost/host value even though the
                # directory name remains the local PID.  The prior
                # ``namespace_pids[-1] == local_pid`` check therefore dropped
                # every mapping on the retained Vast runtime.  Bind the
                # outermost PID to the authoritative local ``/proc`` entry in
                # both layouts.
                mapped[namespace_pids[0]] = local_pid
            break
    return mapped


def local_pid_visible(pid: Any, *, proc_root: Path = Path("/proc")) -> bool:
    """True when ``pid`` names a process in THIS namespace's ``/proc``."""

    resolved = positive_pid(pid)
    if resolved is None:
        return False
    try:
        return (proc_root / str(resolved)).is_dir()
    except OSError:
        return False


def process_gpu_device_minors(pid: Any, *, proc_root: Path = Path("/proc")) -> set[int]:
    """Minor numbers of the per-GPU device nodes this process holds open.

    Generic driver nodes are deliberately excluded; see
    ``NVIDIA_GPU_DEVICE_NODE_PATTERN``.
    """

    minors: set[int] = set()
    resolved = positive_pid(pid)
    if resolved is None:
        return minors
    try:
        entries = list((proc_root / str(resolved) / "fd").iterdir())
    except OSError:
        return minors
    for entry in entries:
        try:
            target = os.readlink(str(entry))
        except OSError:
            continue
        match = NVIDIA_GPU_DEVICE_NODE_PATTERN.match(target)
        if match is not None:
            minors.add(int(match.group(1)))
    return minors


def process_holds_nvidia_device(pid: Any, *, proc_root: Path = Path("/proc")) -> bool:
    """True when the process holds a concrete per-GPU device node open."""

    return bool(process_gpu_device_minors(pid, proc_root=proc_root))


def device_handle_role_pids(
    role_root_pids: Mapping[str, int | None],
    *,
    parent_pid: Callable[[int], int | None],
    proc_root: Path = Path("/proc"),
) -> dict[str, list[int]]:
    """Per role, the local PIDs under its root that hold an NVIDIA device handle."""

    roots = {role: positive_pid(pid) for role, pid in role_root_pids.items()}
    holders: dict[str, list[int]] = {role: [] for role in roots}
    try:
        names = sorted(entry.name for entry in proc_root.iterdir() if entry.name.isdigit())
    except OSError:
        return holders
    for name in names:
        local = positive_pid(name)
        if local is None or not process_holds_nvidia_device(local, proc_root=proc_root):
            continue
        chain = pid_ancestor_chain(local, parent_pid=parent_pid)
        for role, root in roots.items():
            if root is not None and root in chain:
                holders[role].append(local)
    return holders


def compute_app_attribution_unavailable(app: Mapping[str, Any]) -> bool:
    """True when a compute app's PID could not be resolved in this namespace.

    Distinguishes "a process we cannot see" from "another tenant's process":
    the former has no namespace translation AND no local ``/proc`` entry to walk,
    which is precisely the shape attempt 068 recorded on every sampled app.
    """

    if app.get("roles"):
        return False
    if app.get("pid_namespace_translation") != PID_TRANSLATION_IDENTITY:
        return False
    chain = app.get("ancestor_chain") or []
    return len(chain) <= 1


def device_handle_residency_fallback(
    *,
    role_root_pids: Mapping[str, int | None],
    required_roles: Sequence[str],
    inventory_uuids: Iterable[str],
    parent_pid: Callable[[int], int | None],
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Prove same-GPU residency without host-PID translation.

    Applies only when exactly one GPU is visible to the container, so that an
    open ``/dev/nvidia*`` handle identifies the device unambiguously.
    """

    uuids = sorted({str(uuid) for uuid in inventory_uuids if uuid})
    required = list(dict.fromkeys(str(role) for role in required_roles))
    result: dict[str, Any] = {
        "applied": False,
        "gpu_uuid": None,
        "roles": [],
        "role_device_handle_pids": {},
        "blockers": [],
    }
    if len(uuids) != 1:
        result["blockers"].append(f"{MULTI_GPU_BLOCKER}:visible_gpus={len(uuids)}")
        return result
    holders = device_handle_role_pids(role_root_pids, parent_pid=parent_pid, proc_root=proc_root)
    result["role_device_handle_pids"] = {
        role: sorted(pids) for role, pids in sorted(holders.items())
    }
    missing = [role for role in required if not holders.get(role)]
    if missing:
        result["blockers"].append(f"{MISSING_HANDLE_BLOCKER}:{','.join(sorted(missing))}")
        return result
    result["applied"] = True
    result["gpu_uuid"] = uuids[0]
    result["roles"] = required
    return result
