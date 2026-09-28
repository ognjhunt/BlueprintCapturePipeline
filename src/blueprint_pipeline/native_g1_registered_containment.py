"""Installed source identities for the finite registered local G1 producer.

Only the fixed native participant set is eligible. The root bootstrap publisher
independently reads these same installed files before publishing any authority.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import select
import stat
import subprocess
import sys
import time

from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_reference_budget import ReferenceCollectionBudget
from .control_plane_lane_owner_target_versions import _require
from .decision_evidence_contracts import canonical_digest

SOURCE_MODULES = frozenset({
    "control_plane_lane_experiment_consumer", "control_plane_lane_experiment_completion",
    "native_g1_development_pair", "native_g1_development_worker", "native_g1_registered_containment",
    "native_g1_runtime_assembly", "native_g1_policy_server_supervisor", "native_g1_shared_scene_episode",
    "control_plane_scratch_lifetime", "control_plane_g1_lifetime_adapter",
})


def producer_source_identities():
    """Hash actual protected installed files under one bounded admission."""
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        result = {}
        for name in sorted(SOURCE_MODULES):
            raw, record = files.read(Path(__file__).parent / (name + ".py"), cap=1024 * 1024, protected=True)
            result[name] = retained._digest(raw, _work_budget=files.budget)
            files.verify_record(record)
        files.verify()
        return result
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def _unit_membership(intent_id):
    """Check this process's finite selected kernel cgroup, not global PID absence."""
    _require(sys.platform == "linux" and os.geteuid() != 0, "experiment_containment_required")
    path = Path("/proc/self/cgroup")
    with path.open("rb") as stream:
        raw = stream.read(4097)
    expected = "/system.slice/blueprint-experiment-" + intent_id + ".service"
    _require(len(raw) <= 4096 and raw.decode("ascii").splitlines() == ["0::" + expected],
             "experiment_containment_required")
    return expected


def _typed(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


class _ChildLifetime:
    """Parent-owned child and log; direct closure is not descendant clearance."""
    def __init__(self, use, *, process, parent, log, parent_proof, log_proof, log_name, handshake):
        self.use, self.process = use, process
        self.parent, self.log = parent, log
        self.parent_proof, self.log_proof = parent_proof, log_proof
        self.log_name, self.handshake, self.closed = log_name, handshake, False

    def check(self):
        self.use.check()
        self.proof()

    def proof(self):
        _require(_typed(os.fstat(self.parent)) == self.parent_proof
                 and _typed(os.stat(self.log_name, dir_fd=self.parent, follow_symlinks=False)) == self.log_proof
                 == _typed(os.fstat(self.log)), "experiment_child_log_changed")

    def finish(self):
        _require(not self.closed and self.process.poll() is not None, "experiment_child_closure_unproven")
        self.proof()
        failed = False
        for fd, proof in ((self.log, self.log_proof), (self.parent, self.parent_proof)):
            try:
                _require(_typed(os.fstat(fd)) == proof, "experiment_child_descriptor_changed")
                os.close(fd)
            except BaseException:
                failed = True
        _require(not failed, "experiment_child_closure_unproven")
        self.closed = True
        value = dict(schema_version="registered_g1_child_lifetime.v1", intent_id=self.use.entry["intent_id"],
            generation=self.use.entry["generation"], pid=self.process.pid, handshake=self.handshake,
            direct_child_exited=True, log_closed=True, descendant_clearance=False)
        value["lifetime_digest"] = canonical_digest(value, digest_field="lifetime_digest")
        return value


def launch_policy_child(use, *, argv, log_path):
    """Run pinned policy code only after the contained child proves its live SH."""
    from .control_plane_lane_experiment_consumer import RegisteredExperimentUse
    _require(type(use) is RegisteredExperimentUse and use._producer_bootstrap_record is not None
             and type(argv) is list and 2 <= len(argv) <= 32 and all(type(v) is str and len(v) <= 4096 for v in argv),
             "experiment_child_participation_unproven")
    use.check()
    _unit_membership(use.entry["intent_id"])
    _require(log_path.is_relative_to(use.path) and log_path.name == "g1_policy_server.log",
             "experiment_child_participation_unproven")
    # The selected public requests already retain the exact native policy path.
    matched = False
    for _, original, _, _ in use._producer_requests:
        raw = os.pread(original.fd, 65537, 0)
        request = json.loads(raw)
        if Path(request["policy_server_source"]) == Path(argv[1]):
            matched = True
            break
    _require(matched, "experiment_child_request_changed")
    use.check()
    parent, log, process = None, None, None
    read_fd, write_fd, proceed_read, proceed_write = None, None, None, None
    pipes = {}
    parent_proof, log_proof = None, None
    try:
        named = os.stat(log_path.parent, follow_symlinks=False)
        parent = os.open(log_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _require(_typed(os.fstat(parent)) == _typed(named), "experiment_child_descriptor_unproven")
        parent_proof = _typed(named)
        use.check()
        log = os.open(log_path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        named = os.stat(log_path.name, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISREG(named.st_mode) and named.st_nlink == 1 and named.st_uid == os.geteuid()
                 and stat.S_IMODE(named.st_mode) == 0o600 and _typed(os.fstat(log)) == _typed(named),
                 "experiment_child_descriptor_unproven")
        log_proof = _typed(named)
        read_fd, write_fd = os.pipe()
        proceed_read, proceed_write = os.pipe()
        for fd in (read_fd, write_fd, proceed_read, proceed_write):
            info = os.fstat(fd)
            _require(stat.S_ISFIFO(info.st_mode), "experiment_child_descriptor_unproven")
            pipes[fd] = _typed(info)
        bootstrap = use._public_root / (use.entry["intent_id"] + ".producer-bootstrap.json")
        child_argv = [argv[0], "-m", __name__, "--policy-child", str(bootstrap), str(use.fd),
                      str(write_fd), str(proceed_read), *argv[1:]]
        use.check()
        process = subprocess.Popen(child_argv, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, pass_fds=(use.fd, write_fd, proceed_read), start_new_session=False)
        _require(_typed(os.fstat(write_fd)) == pipes[write_fd], "experiment_child_descriptor_changed")
        os.close(write_fd)
        write_fd = None
        _require(_typed(os.fstat(proceed_read)) == pipes[proceed_read], "experiment_child_descriptor_changed")
        os.close(proceed_read)
        proceed_read = None
        deadline = time.monotonic() + 30
        raw = b""
        while b"\n" not in raw and time.monotonic() < deadline:
            use.check()
            ready, _, _ = select.select([read_fd], [], [], min(0.25, max(0, deadline-time.monotonic())))
            if ready:
                chunk = os.read(read_fd, 4097-len(raw))
                _require(bool(chunk) and len(raw)+len(chunk) <= 4096, "experiment_child_handshake_failed")
                raw += chunk
            _require(process.poll() is None, "experiment_child_handshake_failed")
        expected = dict(schema_version="registered_g1_child_ack.v1", intent_id=use.entry["intent_id"],
            generation=use.entry["generation"], birth=use.entry["birth"], target_identity=use.entry["target_identity"],
            pid=process.pid)
        _require(json.loads(raw) == expected, "experiment_child_handshake_failed")
        use.check()
        _require(_typed(os.fstat(proceed_write)) == pipes[proceed_write]
                 and os.write(proceed_write, b"proceed\n") == 8, "experiment_child_handshake_failed")
        os.close(proceed_write)
        proceed_write = None
        _require(_typed(os.fstat(read_fd)) == pipes[read_fd], "experiment_child_descriptor_changed")
        os.close(read_fd)
        read_fd = None
        lifetime = _ChildLifetime(use, process=process, parent=parent, log=log,
            parent_proof=parent_proof, log_proof=log_proof, log_name=log_path.name, handshake=expected)
        lifetime.check()
        return process, lifetime
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        # Unproven initial descriptors are deliberately not adopted for cleanup.
        for fd, proof in ((log, log_proof), (parent, parent_proof)):
            if fd is not None and proof is not None and _typed(os.fstat(fd)) == proof:
                os.close(fd)
        raise
    finally:
        for fd in (read_fd, write_fd, proceed_read, proceed_write):
            if fd is not None and fd in pipes and _typed(os.fstat(fd)) == pipes[fd]:
                os.close(fd)


def _policy_child(arguments):
    """Fixed bootstrap checks precede the selected pinned script's first read."""
    import runpy
    from .control_plane_lane_experiment_consumer import RegisteredExperimentUse, AUTHORITY_ROOT, LANE_ROOTS
    _require(5 <= len(arguments) <= 36, "experiment_child_configuration_invalid")
    bootstrap = Path(arguments[0])
    intent_id = bootstrap.name.removesuffix(".producer-bootstrap.json")
    _require(bootstrap == AUTHORITY_ROOT / (intent_id + ".producer-bootstrap.json")
             and len(intent_id) == 32 and all(c in "0123456789abcdef" for c in intent_id),
             "experiment_child_configuration_invalid")
    _unit_membership(intent_id)
    inherited, ack_fd, proceed_fd = int(arguments[1]), int(arguments[2]), int(arguments[3])
    _require(len({inherited, ack_fd, proceed_fd}) == 3 and min(inherited, ack_fd, proceed_fd) >= 3,
             "experiment_child_configuration_invalid")
    admitted = None
    for root in LANE_ROOTS:
        target = root / "g1" / ("registered-" + intent_id)
        if target.exists():
            _require(admitted is None, "experiment_child_configuration_invalid")
            admitted = RegisteredExperimentUse.admit(target, _producer_bootstrap_path=bootstrap)
    _require(admitted is not None, "experiment_child_configuration_invalid")
    inherited_proof, ack_proof, proceed_proof = None, None, None
    try:
        expected = admitted.entry["target_identity"]
        actual = os.fstat(inherited)
        _require((actual.st_dev, actual.st_ino, stat.S_ISDIR(actual.st_mode))
                 == (expected["dev"], expected["ino"], True), "experiment_child_descriptor_unproven")
        inherited_proof = _typed(actual)
        ack_actual = os.fstat(ack_fd)
        _require(stat.S_ISFIFO(ack_actual.st_mode), "experiment_child_descriptor_unproven")
        ack_proof = _typed(ack_actual)
        proceed_actual = os.fstat(proceed_fd)
        _require(stat.S_ISFIFO(proceed_actual.st_mode), "experiment_child_descriptor_unproven")
        proceed_proof = _typed(proceed_actual)
        script = Path(arguments[4])
        matched = any(Path(json.loads(os.pread(record.fd, 65537, 0))["policy_server_source"]) == script
                      for _, record, _, _ in admitted._producer_requests)
        _require(matched, "experiment_child_request_changed")
        admitted.check()
        ack = dict(schema_version="registered_g1_child_ack.v1", intent_id=intent_id,
            generation=admitted.entry["generation"], birth=admitted.entry["birth"],
            target_identity=expected, pid=os.getpid())
        payload = json.dumps(ack, sort_keys=True, separators=(",", ":")).encode()+b"\n"
        _require(len(payload) <= 4096 and os.write(ack_fd, payload) == len(payload), "experiment_child_handshake_failed")
        ready, _, _ = select.select([proceed_fd], [], [], 30)
        _require(ready == [proceed_fd] and _typed(os.fstat(proceed_fd)) == proceed_proof
                 and os.read(proceed_fd, 9) == b"proceed\n", "experiment_child_handshake_failed")
        admitted.check()
        # Native policy script and this bootstrap share one process/lifetime.
        sys.argv = arguments[4:]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        try:
            admitted.close()
        finally:
            for fd, proof in ((inherited, inherited_proof), (ack_fd, ack_proof), (proceed_fd, proceed_proof)):
                if proof is not None and _typed(os.fstat(fd)) == proof:
                    os.close(fd)


if __name__ == "__main__":
    _require(len(sys.argv) >= 2 and sys.argv[1] == "--policy-child", "experiment_child_configuration_invalid")
    _policy_child(sys.argv[2:])
