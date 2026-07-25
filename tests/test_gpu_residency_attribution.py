"""Pin the attempt-068 false negative: unattributable PIDs are not absent roles.

Attempt 068 was a sealed run that failed at step 1 with
``required_roles_not_simultaneously_resident_on_same_gpu`` while all four roles
were in fact resident, holding ~45 GiB on the single visible GPU. The host's
container runtime never exposed the outer ``NSpid`` chain, so nvidia-smi's
root-namespace PIDs (3263434, ...) could not be translated to container PIDs
(3746, ...); the code then treated each untranslatable PID as a local PID that
simply was not ours. These tests reconstruct that host from ``/proc`` fixtures
and require the device-handle fallback to reach the truthful verdict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from blueprint_pipeline.gpu_residency_attribution import (
    device_handle_residency_fallback,
    device_handle_role_pids,
    linux_nvidia_host_to_local_pid_map,
    process_gpu_device_minors,
    process_holds_nvidia_device,
)

# The four roles and their container PIDs, verbatim from attempt 068's
# oscar_gpu_residency_report.json.
ROLE_ROOT_PIDS = {"groot": 3746, "gear_sonic": 3747, "isaac_task": 3748, "oscar": 7883}
REQUIRED_ROLES = ("gear_sonic", "groot", "isaac_task", "oscar")
GPU_UUID = "GPU-9b984c33-61eb-8a48-44d1-b6dc457b87fd"


def _write_process(
    proc_root: Path,
    pid: int,
    *,
    ppid: int,
    nspid: str | None = None,
    device_fds: tuple[str, ...] = (),
    other_fds: tuple[str, ...] = ("/dev/null",),
) -> None:
    entry = proc_root / str(pid)
    entry.mkdir(parents=True, exist_ok=True)
    status = f"Name:\tproc{pid}\nPPid:\t{ppid}\n"
    if nspid is not None:
        status += f"NSpid:\t{nspid}\n"
    (entry / "status").write_text(status, encoding="utf-8")
    fd_dir = entry / "fd"
    fd_dir.mkdir(exist_ok=True)
    for index, target in enumerate((*device_fds, *other_fds)):
        link = fd_dir / str(index)
        real = proc_root / "_targets" / target.lstrip("/")
        real.parent.mkdir(parents=True, exist_ok=True)
        real.touch()
        # The fallback reads the symlink text, so point at a real file but keep
        # the recorded target string identical to the device node path.
        try:
            os.symlink(target, link)
        except OSError:  # pragma: no cover - platforms without symlink perms
            pytest.skip("symlinks unavailable")


@pytest.fixture()
def opaque_host_proc(tmp_path: Path) -> Path:
    """A container whose runtime hides host PIDs: NSpid carries only the local PID."""

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    for role_pid in ROLE_ROOT_PIDS.values():
        _write_process(proc_root, role_pid, ppid=1, nspid=str(role_pid))
        # Each role's CUDA-owning child, as torchrun and Isaac actually run.
        _write_process(
            proc_root,
            role_pid + 10000,
            ppid=role_pid,
            nspid=str(role_pid + 10000),
            device_fds=("/dev/nvidia0", "/dev/nvidiactl"),
        )
    _write_process(proc_root, 1, ppid=0, nspid="1")
    return proc_root


def _parent_pid(proc_root: Path):
    def lookup(pid: int) -> int | None:
        try:
            status = (proc_root / str(pid) / "status").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", 1)[1].strip())
        return None

    return lookup


def test_opaque_host_yields_no_usable_host_pid_translation(opaque_host_proc: Path) -> None:
    """Reproduce the root cause: the NSpid map cannot reach nvidia-smi's PIDs."""

    mapping = linux_nvidia_host_to_local_pid_map(opaque_host_proc)
    assert 3263434 not in mapping, "host PIDs are unknowable inside this namespace"
    assert all(host == local for host, local in mapping.items()), "identity-only map"


def test_device_handle_fallback_proves_residency_without_translation(
    opaque_host_proc: Path,
) -> None:
    """The verdict attempt 068 should have reached, on the host that failed it."""

    result = device_handle_residency_fallback(
        role_root_pids=ROLE_ROOT_PIDS,
        required_roles=REQUIRED_ROLES,
        inventory_uuids=[GPU_UUID],
        parent_pid=_parent_pid(opaque_host_proc),
        proc_root=opaque_host_proc,
    )
    assert result["applied"] is True
    assert result["gpu_uuid"] == GPU_UUID
    assert sorted(result["roles"]) == sorted(REQUIRED_ROLES)
    assert result["blockers"] == []
    for role, pid in ROLE_ROOT_PIDS.items():
        assert result["role_device_handle_pids"][role] == [pid + 10000]


def test_fallback_attributes_descendants_not_just_root_pids(
    opaque_host_proc: Path,
) -> None:
    """Roles hold the GPU through children; root PIDs themselves never do."""

    for pid in ROLE_ROOT_PIDS.values():
        assert not process_holds_nvidia_device(pid, proc_root=opaque_host_proc)
    holders = device_handle_role_pids(
        ROLE_ROOT_PIDS,
        parent_pid=_parent_pid(opaque_host_proc),
        proc_root=opaque_host_proc,
    )
    assert all(holders[role] for role in REQUIRED_ROLES)


def test_fallback_refuses_multi_gpu_inventory(opaque_host_proc: Path) -> None:
    """An open device handle cannot say WHICH GPU when several are visible."""

    result = device_handle_residency_fallback(
        role_root_pids=ROLE_ROOT_PIDS,
        required_roles=REQUIRED_ROLES,
        inventory_uuids=[GPU_UUID, "GPU-deadbeef-0000-1111-2222-333344445555"],
        parent_pid=_parent_pid(opaque_host_proc),
        proc_root=opaque_host_proc,
    )
    assert result["applied"] is False
    assert any("multi_gpu_inventory" in blocker for blocker in result["blockers"])


CONTROL_ONLY_NODES = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-modeset",
    "/dev/nvidia-caps/nvidia-cap1",
)


def test_generic_driver_nodes_alone_never_prove_residency(tmp_path: Path) -> None:
    """Four roles that only probed the driver must not attest a compute context.

    Every process that initializes CUDA opens the control nodes, so accepting
    them would turn the fallback from a residency proof into a liveness check
    and let a sealed report pass with no role doing GPU work at all.
    """

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(proc_root, 1, ppid=0, nspid="1")
    for pid in ROLE_ROOT_PIDS.values():
        _write_process(proc_root, pid, ppid=1, nspid=str(pid))
        _write_process(
            proc_root,
            pid + 10000,
            ppid=pid,
            nspid=str(pid + 10000),
            device_fds=CONTROL_ONLY_NODES,
        )
    for pid in ROLE_ROOT_PIDS.values():
        assert process_gpu_device_minors(pid + 10000, proc_root=proc_root) == set()
    result = device_handle_residency_fallback(
        role_root_pids=ROLE_ROOT_PIDS,
        required_roles=REQUIRED_ROLES,
        inventory_uuids=[GPU_UUID],
        parent_pid=_parent_pid(proc_root),
        proc_root=proc_root,
    )
    assert result["applied"] is False
    assert any("without_gpu_device_handle" in blocker for blocker in result["blockers"])


def test_numbered_gpu_node_is_what_counts(opaque_host_proc: Path) -> None:
    """The concrete per-GPU node is the descriptor that carries the proof."""

    for pid in ROLE_ROOT_PIDS.values():
        assert process_gpu_device_minors(pid + 10000, proc_root=opaque_host_proc) == {0}


class _Done:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _attempt_068_nvidia_smi(argv, **_kwargs):
    """nvidia-smi exactly as attempt 068 saw it: host PIDs, unresolvable names."""

    if any(str(arg).startswith("--query-gpu=") for arg in argv):
        return _Done(f"0, {GPU_UUID}, 46068, 45066, 392\n")
    return _Done(
        f"{GPU_UUID}, 3263434, [Not Found], 6680\n"
        f"{GPU_UUID}, 3263449, [Not Found], 7306\n"
        f"{GPU_UUID}, 3263501, [Not Found], 31080\n"
    )


def test_attempt_068_sample_now_proves_residency_end_to_end(
    opaque_host_proc: Path,
) -> None:
    """The sealed-run failure, replayed through the real sampler, now passes."""

    from blueprint_pipeline import oscar_isaac_closed_loop_eval as loop

    sample = loop.collect_oscar_gpu_residency_sample(
        query_run=_attempt_068_nvidia_smi,
        role_root_pids=ROLE_ROOT_PIDS,
        parent_pid=_parent_pid(opaque_host_proc),
        host_to_local_pid_map=lambda: {},
        proc_root=opaque_host_proc,
    )
    assert sample["all_required_roles_simultaneously_resident_on_same_gpu"] is True
    assert sample["role_attribution_mode"] == "device_handle_fallback"
    assert sample["blockers"] == []

    report = loop.summarize_oscar_gpu_residency_samples([sample], role_root_pids=ROLE_ROOT_PIDS)
    assert report["proof_passed"] is True


def test_working_host_still_uses_host_pid_attribution(opaque_host_proc: Path) -> None:
    """No behaviour change where translation works: the fallback stays dormant."""

    from blueprint_pipeline import oscar_isaac_closed_loop_eval as loop

    translation = {
        3263434: 3746 + 10000,
        3263449: 3747 + 10000,
        3263501: 3748 + 10000,
    }

    def with_oscar(argv, **kwargs):
        result = _attempt_068_nvidia_smi(argv, **kwargs)
        if any(str(arg).startswith("--query-gpu=") for arg in argv):
            return result
        return _Done(result.stdout + f"{GPU_UUID}, 3263777, python, 100\n")

    translation[3263777] = 7883 + 10000
    sample = loop.collect_oscar_gpu_residency_sample(
        query_run=with_oscar,
        role_root_pids=ROLE_ROOT_PIDS,
        parent_pid=_parent_pid(opaque_host_proc),
        host_to_local_pid_map=lambda: translation,
        proc_root=opaque_host_proc,
    )
    assert sample["all_required_roles_simultaneously_resident_on_same_gpu"] is True
    assert sample["role_attribution_mode"] == "host_pid_namespace"
    assert sample["device_handle_attribution"] == {}


def test_early_sample_without_cuda_contexts_does_not_poison_the_proof(
    tmp_path: Path,
) -> None:
    """A foreign tenant's app before our roles start must not fail the run.

    The fallback's negative result is diagnostic; only the summary rules, and
    only when NO sample ever achieved simultaneity.
    """

    from blueprint_pipeline import oscar_isaac_closed_loop_eval as loop

    bare = tmp_path / "proc"
    bare.mkdir()
    _write_process(bare, 1, ppid=0, nspid="1")
    for pid in ROLE_ROOT_PIDS.values():
        _write_process(bare, pid, ppid=1, nspid=str(pid))
    early = loop.collect_oscar_gpu_residency_sample(
        query_run=_attempt_068_nvidia_smi,
        role_root_pids=ROLE_ROOT_PIDS,
        parent_pid=_parent_pid(bare),
        host_to_local_pid_map=lambda: {},
        proc_root=bare,
        sample_index=0,
    )
    assert early["blockers"] == [], "a pre-CUDA sample must contribute no blockers"
    assert early["all_required_roles_simultaneously_resident_on_same_gpu"] is False


def test_fallback_still_fails_a_role_that_never_touched_the_gpu(tmp_path: Path) -> None:
    """The fallback must not manufacture residency for a genuinely absent role."""

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(proc_root, 1, ppid=0, nspid="1")
    for role, pid in ROLE_ROOT_PIDS.items():
        _write_process(proc_root, pid, ppid=1, nspid=str(pid))
        if role == "oscar":
            continue  # OSCAR never opened a device node.
        _write_process(
            proc_root,
            pid + 10000,
            ppid=pid,
            nspid=str(pid + 10000),
            device_fds=("/dev/nvidia0",),
        )
    result = device_handle_residency_fallback(
        role_root_pids=ROLE_ROOT_PIDS,
        required_roles=REQUIRED_ROLES,
        inventory_uuids=[GPU_UUID],
        parent_pid=_parent_pid(proc_root),
        proc_root=proc_root,
    )
    assert result["applied"] is False
    assert any(blocker.endswith(":oscar") for blocker in result["blockers"])
