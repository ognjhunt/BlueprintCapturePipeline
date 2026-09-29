"""Installed source identities for the finite registered local G1 producer.

Only the fixed native participant set is eligible. The root bootstrap publisher
independently reads these same installed files before publishing any authority.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import select
import re
import shlex
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
    "control_plane_scratch_lifetime", "control_plane_g1_lifetime_adapter", "native_g1_registered_path_boundary",
})
_NATIVE_PYTHON = Path("/opt/blueprint/task-evaluation-control-plane/.venv/bin/python")
_SYSTEMCTL = "/usr/bin/systemctl"
_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _unit_arguments(intent_id, target, bootstrap):
    _require(type(intent_id) is str and re.fullmatch(r"[0-9a-f]{32}", intent_id)
             and isinstance(target, Path) and target.is_absolute() and target.name == "registered-" + intent_id
             and isinstance(bootstrap, Path) and bootstrap.is_absolute()
             and bootstrap.name == intent_id + ".producer-bootstrap.json", "experiment_unit_configuration_invalid")
    properties = ["User=blueprint", "Group=blueprint", "UMask=0077", "NoNewPrivileges=yes",
        "CapabilityBoundingSet=", "AmbientCapabilities=", "ProtectControlGroups=yes", "KillMode=control-group",
        "Delegate=no", "TasksMax=64", "TimeoutStopSec=30", "RemainAfterExit=yes", "ProtectSystem=strict",
        "ProtectHome=yes", "PrivateTmp=yes", "PrivateNetwork=yes", "RestrictNamespaces=yes",
        "RestrictSUIDSGID=yes", "ReadWritePaths=" + str(target), "ReadOnlyPaths=/"]
    return ["/usr/bin/systemd-run", "--no-block", "--quiet", "--unit=blueprint-experiment-" + intent_id,
        *("--property=" + value for value in properties), "--", str(_NATIVE_PYTHON), "-m", __name__,
        "--producer-bootstrap", str(bootstrap), "--target", str(target)]


def _native_control(arguments):
    value = subprocess.run(arguments, stdin=subprocess.DEVNULL, capture_output=True, timeout=5,
                           check=False, env={"PATH":"/usr/bin:/bin", "LC_ALL":"C"})
    _require(value.returncode == 0 and len(value.stdout) <= 65536 and len(value.stderr) <= 4096,
             "experiment_unit_observation_failed")
    return value.stdout.decode("utf-8", errors="strict")


_UNIT_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "InvocationID", "MainPID", "Result",
    "ExecMainStatus", "ExecStart", "ControlGroup", "User", "Group", "UMask", "NoNewPrivileges",
    "CapabilityBoundingSet", "AmbientCapabilities", "ProtectControlGroups", "KillMode", "Delegate",
    "TasksMax", "TimeoutStopUSec", "RemainAfterExit", "ProtectSystem", "ProtectHome", "PrivateTmp",
    "PrivateNetwork", "RestrictNamespaces", "RestrictSUIDSGID", "ReadWritePaths", "ReadOnlyPaths")


def _show_unit(intent_id):
    unit = "blueprint-experiment-" + intent_id + ".service"
    raw = _native_control([_SYSTEMCTL, "show", unit, "--no-pager", "--all", "--property=" + ",".join(_UNIT_PROPERTIES)])
    rows = raw.splitlines()
    _require(len(rows) <= len(_UNIT_PROPERTIES) and all("=" in row for row in rows), "experiment_unit_observation_failed")
    value = dict(row.split("=", 1) for row in rows)
    _require(len(value) == len(rows), "experiment_unit_observation_failed")
    # Native systemd omits an empty ExecStart array even with --all for a
    # genuinely absent unit. This finite NO COMMAND representation applies only
    # to the exact absent identity; loaded/active/incomplete units stay refused.
    if set(value) == set(_UNIT_PROPERTIES) - {"ExecStart"}:
        _require(value["Id"] == unit and value["LoadState"] == "not-found"
                 and value["ActiveState"] == "inactive" and value["SubState"] == "dead"
                 and value["MainPID"] == "0" and value["InvocationID"] == ""
                 and value["ControlGroup"] == "" and value["Result"] == "success"
                 and value["ExecMainStatus"] == "0", "experiment_unit_observation_failed")
        value["ExecStart"] = ""
    _require(set(value) == set(_UNIT_PROPERTIES), "experiment_unit_observation_failed")
    return value


def _check_unit(value, command, intent_id, *, started=None):
    unit = "blueprint-experiment-" + intent_id + ".service"
    group = "/system.slice/" + unit
    terminal = False
    if started is not None:
        _require(type(started) is dict, "experiment_unit_identity_changed")
        _check_unit(started, command, intent_id)
        _require(started["MainPID"].isdigit() and int(started["MainPID"]) > 1
                 and value["InvocationID"] == started["InvocationID"], "experiment_unit_identity_changed")
        terminal = (value["ActiveState"], value["SubState"]) in {("active", "exited"), ("inactive", "dead")}
        terminal = (terminal and value["MainPID"] == "0" and value["Result"] == "success"
                    and value["ExecMainStatus"] == "0")
    _require(value["Id"] == unit and value["LoadState"] == "loaded"
             and re.fullmatch(r"[0-9a-f]{32}", value["InvocationID"])
             and (value["ControlGroup"] == group or (value["ControlGroup"] == "" and terminal)),
             "experiment_unit_identity_changed")
    expected = {"User":"blueprint", "Group":"blueprint", "UMask":"0077", "NoNewPrivileges":"yes",
        "CapabilityBoundingSet":"", "AmbientCapabilities":"", "ProtectControlGroups":"yes",
        "KillMode":"control-group", "Delegate":"no", "TasksMax":"64", "TimeoutStopUSec":"30s",
        "RemainAfterExit":"yes", "ProtectSystem":"strict", "ProtectHome":"yes", "PrivateTmp":"yes",
        "PrivateNetwork":"yes", "RestrictNamespaces":"yes", "RestrictSUIDSGID":"yes",
        "ReadWritePaths":command[-1], "ReadOnlyPaths":"/"}
    _require(all(value[key] == selected for key, selected in expected.items()), "experiment_unit_sandbox_changed")
    start = value["ExecStart"]
    match = re.fullmatch(r"\{ path=(.*?) ; argv\[\]=(.*?) ; ignore_errors=no ; .* \}", start)
    argv = command[command.index("--") + 1:]
    _require(match is not None and match[1] == argv[0] and shlex.split(match[2]) == argv,
             "experiment_unit_command_changed")


def _check_finished_unit(value, command, intent_id, *, started, stopping, kernel, stop_command):
    """Raw post-stop disappearance is meaningful only in this exact chain."""
    _check_unit(started, command, intent_id)
    _check_unit(stopping, command, intent_id, started=started)
    unit = 'blueprint-experiment-' + intent_id + '.service'
    _require(stopping['ActiveState'] == 'active' and stopping['SubState'] == 'exited'
             and stopping['MainPID'] == '0' and stopping['Result'] == 'success'
             and stopping['ExecMainStatus'] == '0' and kernel['tasks'] == 0
             and stop_command == [_SYSTEMCTL, 'stop', unit], 'experiment_unit_closure_unproven')
    if value['LoadState'] == 'loaded':
        _check_unit(value, command, intent_id, started=started)
    else:
        _require(set(value) == set(_UNIT_PROPERTIES) and value['LoadState'] == 'not-found'
                 and value['Id'] == unit and value['ActiveState'] == 'inactive' and value['SubState'] == 'dead'
                 and value['MainPID'] == '0' and value['InvocationID'] == '' and value['ExecStart'] == ''
                 and value['ControlGroup'] == '' and value['Result'] == 'success'
                 and value['ExecMainStatus'] == '0', 'experiment_unit_closure_unproven')
    _require(value['ActiveState'] == 'inactive' and value['Result'] == 'success'
             and value['ExecMainStatus'] == '0', 'experiment_unit_closure_unproven')


def _started_group(selected):
    """Capture the actual owned kernel namespace while its producer is live."""
    _require(re.fullmatch(r'/system.slice/blueprint-experiment-[0-9a-f]{32}\.service', selected),
             'experiment_cgroup_unproven')
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        parent, name = files.parent(_CGROUP_ROOT / selected.lstrip('/'), protected=True)
        root = files.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
        files.verify()
        return {'path': selected, 'parent_identity': _typed(os.fstat(parent)),
                'root_identity': _typed(os.fstat(root))}
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def _empty_group(selected, *, started=None):
    """Observe only the owned root and its bounded cgroup-v2 descendants."""
    _require(type(selected) is str and selected.startswith("/system.slice/blueprint-experiment-")
             and selected.endswith(".service") and ".." not in selected, "experiment_cgroup_unproven")
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        if started is not None:
            _require(type(started) is dict and set(started) == {'path', 'parent_identity', 'root_identity'}
                     and started['path'] == selected, 'experiment_cgroup_unproven')
            parent, name = files.parent(_CGROUP_ROOT / selected.lstrip('/'), protected=True)
            _require(_typed(os.fstat(parent)) == tuple(started['parent_identity']), 'experiment_cgroup_identity_changed')
            files.location(parent)
            try:
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                # A cgroup-v2 directory disappears only after its processes and
                # descendant groups are gone. The child cannot escape the
                # authenticated read-only cgroup sandbox. This is absence of
                # our first observed exact group, never arbitrary absence.
                files.location(parent)
                files.verify()
                original = started['root_identity']
                return dict(root_identity={'dev': original[0], 'ino': original[1], 'type': 'directory'},
                            parent_identity=started['parent_identity'], groups=0, tasks=0,
                            observed_bytes=0, group_identities=[], absent_after_started=True)
            _require(_typed(named) == tuple(started['root_identity']), 'experiment_cgroup_identity_changed')
        root, _ = files.parent(_CGROUP_ROOT / selected.lstrip("/") / "cgroup.events", protected=True)
        root_info = os.fstat(root)
        pending, groups, tasks, raw_bytes = [root], 0, 0, 0
        identities = []
        while pending:
            parent = pending.pop()
            files.location(parent)
            groups += 1
            _require(groups <= 32, "experiment_cgroup_resource_exhausted")
            names = []
            with os.scandir(parent) as entries:
                for entry in entries:
                    files.budget.charge("entries")
                    info = os.stat(entry.name, dir_fd=parent, follow_symlinks=False)
                    _require(info.st_uid == 0 and not stat.S_IMODE(info.st_mode) & 0o022,
                             "experiment_cgroup_permissions_changed")
                    if stat.S_ISDIR(info.st_mode):
                        _require(len(names) + groups <= 32, "experiment_cgroup_resource_exhausted")
                        names.append(entry.name)
            for name in ("cgroup.events", "cgroup.procs", "cgroup.threads"):
                fd = files.open(name, os.O_RDONLY | os.O_NONBLOCK, parent=parent)
                try:
                    raw = files.read_bytes(fd, 65536)
                    raw_bytes += len(raw)
                    _require(raw_bytes <= 65536, "experiment_cgroup_resource_exhausted")
                    if name == "cgroup.events":
                        _require(b"populated 0\n" in raw, "experiment_cgroup_still_populated")
                    else:
                        tasks += len(raw.splitlines())
                        _require(tasks == 0, "experiment_cgroup_still_populated")
                finally:
                    files.close(fd)
            identities.append(_typed(os.fstat(parent)))
            for name in sorted(names):
                pending.append(files.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent))
        files.verify()
        return dict(root_identity={"dev":root_info.st_dev, "ino":root_info.st_ino, "type":"directory"},
                    groups=groups, tasks=tasks, observed_bytes=raw_bytes, group_identities=identities)
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


class _TerminatedUnitProof:
    """Created only by the root observer of its exact native unit invocation."""
    def __init__(self, *, started, stopping, finished, kernel, command, stop_command, observed_at):
        self.started, self.finished, self.kernel = started, finished, kernel
        self.command, self.observed_at = command, observed_at
        self.stopping, self.stop_command = stopping, stop_command


def run_registered_experiment(intent_id, *, expected_intent, installed_config_path="/etc/blueprint-operator-door/door.json",
                              now=time.time):
    """Root authentic launch; actual kernel closure precedes completion publication."""
    from . import control_plane_lane_experiment_actions as actions
    from . import control_plane_lane_experiment_retirement as issuance
    from . import control_plane_lane_owner_consents as owners
    from .control_plane_lane_experiment_authority import _read
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        _require(os.geteuid() == 0 and type(intent_id) is str and re.fullmatch(r"[0-9a-f]{32}", intent_id)
                 and type(expected_intent) is dict and set(expected_intent) == {"sha256", "size_bytes"},
                 "experiment_issuer_required")
        issued = now()
        config, gid = actions._context(files, installed_config_path, issued)
        _require(config.experiment_creation_enabled is True, "experiment_creation_disabled")
        public, current, entry = actions._selected(files, config, intent_id, issued, gid)
        _require(entry["state"] == "active" and entry["completion"] is None and entry["operation_id"] is None
                 and issued < entry["expires_at_epoch"], "experiment_producer_inactive")
        target, _ = actions._target(files, config, entry)
        lease, _ = actions._lease(files, target, entry)
        birth = actions._birth(files, public, entry, gid)
        _require(birth["participant_profile"] == "g1_local_contained_completed.v1"
                 and lease["released_at_epoch"] is None and issued < lease["expires_at_epoch"],
                 "experiment_producer_inactive")
        raw, _ = files.read(Path(config.experiment_record_store)/(intent_id+".json"), cap=32768, protected=True, mode=0o600)
        owners._identity(raw, expected_intent["sha256"], expected_intent["size_bytes"], files.budget)
        intent = retained._document(raw, 32768, _work_budget=files.budget)
        _require(intent["intent_digest"] == canonical_digest(intent, digest_field="intent_digest")
                 and intent["generation"] == entry["generation"] and intent["policy"] == current[1]["policy"]
                 and issued < intent["expires_at_epoch"], "experiment_producer_authority_changed")
        bootstrap = Path(config.experiment_authority_root)/(intent_id+".producer-bootstrap.json")
        raw, _ = _read(files, public, bootstrap.name, 32768, gid)
        selected = retained._document(raw, 32768, _work_budget=files.budget)
        _require(selected["intent"] == expected_intent and selected["bootstrap_digest"]
                 == canonical_digest(selected, digest_field="bootstrap_digest")
                 and selected["generation"] == entry["generation"] and selected["birth"] == entry["birth"]
                 and issued < selected["expires_at_epoch"]
                 and selected["installed_sources"] == producer_source_identities(), "experiment_producer_authority_changed")
        policy_raw, _ = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES, protected=True, mode=0o600)
        _require(issuance._selector(policy_raw, files.budget) == current[1]["policy"], "experiment_policy_changed")
        files.verify()
        command = _unit_arguments(intent_id, target, bootstrap)
        expiry = min(entry["expires_at_epoch"], selected["expires_at_epoch"], current[1]["expires_at_epoch"])
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
    _require(sys.platform == "linux", "experiment_containment_required")
    _require(_show_unit(intent_id)["LoadState"] == "not-found", "experiment_unit_name_consumed")
    _native_control(command)
    started, started_group, origin = None, None, time.monotonic()
    while time.monotonic() - origin <= 4*3600 and now() < expiry:
        value = _show_unit(intent_id)
        _check_unit(value, command, intent_id, started=started)
        if started is None and value["MainPID"] == "0":
            _require(value["ActiveState"] == "activating", "experiment_unit_start_unproven")
            time.sleep(0.1)
            continue
        if started is None:
            _require(value["MainPID"].isdigit() and int(value["MainPID"]) > 1, "experiment_unit_start_unproven")
            started_group = _started_group(value["ControlGroup"])
            started = value
        _require(value["InvocationID"] == started["InvocationID"], "experiment_unit_identity_changed")
        if value["ActiveState"] == "active" and value["SubState"] == "exited":
            _require(value["Result"] == "success" and value["ExecMainStatus"] == "0", "experiment_producer_failed")
            kernel = _empty_group(started['ControlGroup'], started=started_group)
            current_unit = _show_unit(intent_id)
            _check_unit(current_unit, command, intent_id, started=started)
            _require(current_unit['ActiveState'] == 'active' and current_unit['SubState'] == 'exited'
                     and current_unit['MainPID'] == '0' and current_unit['Result'] == 'success'
                     and current_unit['ExecMainStatus'] == '0'
                     and time.monotonic() - origin <= 4*3600 and now() < expiry,
                     'experiment_unit_closure_unproven')
            stop_command = [_SYSTEMCTL, 'stop', 'blueprint-experiment-' + intent_id + '.service']
            _native_control(stop_command)
            finished = _show_unit(intent_id)
            _check_finished_unit(finished, command, intent_id, started=started, stopping=current_unit,
                                 kernel=kernel, stop_command=stop_command)
            _require(time.monotonic() - origin <= 4*3600 and now() < expiry,
                     'experiment_unit_closure_unproven')
            proof = _TerminatedUnitProof(started=started, stopping=current_unit, finished=finished,
                kernel=kernel, command=command, stop_command=stop_command, observed_at=now())
            from .control_plane_lane_experiment_completion import publish_contained_completion
            completion = publish_contained_completion(intent_id, expected_intent=expected_intent,
                proof=proof, installed_config_path=installed_config_path, now=now)
            return {"status":"completed", "completion":completion, "unit_invocation_id":started["InvocationID"]}
        _require(value["ActiveState"] in ("activating", "active"), "experiment_producer_failed")
        time.sleep(0.1)
    raise ValueError("experiment_producer_deadline")


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
    def __init__(self, use, *, process, parent, log, parent_proof, log_proof, parent_path, log_name, handshake):
        self.use, self.process = use, process
        self.parent, self.log = parent, log
        self.parent_proof, self.log_proof = parent_proof, log_proof
        self.parent_path = parent_path
        self.log_name, self.handshake, self.closed = log_name, handshake, False
        parent_info, log_info = os.fstat(parent), os.fstat(log)
        self.parent_security = (parent_info.st_uid, parent_info.st_gid, stat.S_IMODE(parent_info.st_mode))
        self.log_security = (log_info.st_uid, log_info.st_gid, stat.S_IMODE(log_info.st_mode), log_info.st_nlink)
        self.proof()

    def check(self):
        self.use.check()
        self.proof()

    def proof(self):
        parent_info, log_info = os.fstat(self.parent), os.fstat(self.log)
        named_parent = os.stat(self.parent_path, follow_symlinks=False)
        named_log = os.stat(self.log_name, dir_fd=self.parent, follow_symlinks=False)
        _require(_typed(parent_info) == self.parent_proof == _typed(named_parent)
                 and _typed(named_log) == self.log_proof == _typed(log_info)
                 and (parent_info.st_uid, parent_info.st_gid, stat.S_IMODE(parent_info.st_mode)) == self.parent_security
                 == (named_parent.st_uid, named_parent.st_gid, stat.S_IMODE(named_parent.st_mode))
                 and (log_info.st_uid, log_info.st_gid, stat.S_IMODE(log_info.st_mode), log_info.st_nlink)
                 == self.log_security == (named_log.st_uid, named_log.st_gid, stat.S_IMODE(named_log.st_mode), named_log.st_nlink)
                 and self.log_security[2:] == (0o600, 1), "experiment_child_log_changed")

    def finish(self):
        _require(not self.closed and self.process.poll() is not None, "experiment_child_closure_unproven")
        failed = False
        try:
            self.proof()
        except BaseException:
            failed = True
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
            parent_proof=parent_proof, log_proof=log_proof, parent_path=log_path.parent,
            log_name=log_path.name, handshake=expected)
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


def _producer_main(bootstrap, target):
    from .control_plane_lane_experiment_consumer import RegisteredExperimentUse
    from .native_g1_development_pair import run_g1_development_pair
    use = RegisteredExperimentUse.admit(target, _producer_bootstrap_path=bootstrap)
    try:
        _unit_membership(use.entry["intent_id"])
        result = run_g1_development_pair(request_paths=[Path(path) for path, *_ in use._producer_requests],
                                       output_dir=target, mode="local", _registered_use=use)
        _require(result["status"] == "completed_development_only", "experiment_producer_failed")
        return result
    finally:
        if not use._closed:
            use.close()


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
    if len(sys.argv) == 5 and sys.argv[1] == "--producer-bootstrap" and sys.argv[3] == "--target":
        _producer_main(Path(sys.argv[2]), Path(sys.argv[4]))
    else:
        _require(len(sys.argv) >= 2 and sys.argv[1] == "--policy-child", "experiment_child_configuration_invalid")
        _policy_child(sys.argv[2:])
