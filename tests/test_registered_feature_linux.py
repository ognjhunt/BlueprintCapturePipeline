"""ADP-009D real root/blueprint UID acceptance on a disposable hosted Linux runner.

Requires explicit BLUEPRINT_DISPOSABLE_LINUX_TEST=1. A Mac skip is NOT proof.
No production host, remote provider, cloud SDK, or actual network is contacted.
Root-owned installation lives under /var/lib, with actual blueprint group modes.
"""

from __future__ import annotations

import base64
import errno
import functools
import grp
import hashlib
import io
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _cache_fixture_disk_reservation(root):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    # A real root fixture must not create the production ledger: its surviving
    # /var/lib/blueprint parent would enable residency checks for later tests.
    return functools.partial(
        reserve_control_plane_disk,
        reservation_root=root / "actual-cache-reservation-ledger",
    )


def test_cache_fixture_reservation_keeps_admission_and_history_in_its_installation(
    tmp_path, monkeypatch
):
    from blueprint_pipeline.control_plane_disk_budget import ControlPlaneDiskBudgetError
    from blueprint_pipeline.host_resident_launch_inputs import configured_launch_input_roots

    roots_before = configured_launch_input_roots(env={})
    # Use the real small sandbox filesystem; retain the proportional floor.
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", "0")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reserve = _cache_fixture_disk_reservation(tmp_path)
    reservation = reserve(
        "g1_checkpoint_cache",
        target_root=workspace,
        expected_bytes=4096,
        minimum_bytes=4096,
        workspace=workspace,
        fresh=True,
        evictor=None,
    )
    with reservation:
        assert reservation.path.parent == tmp_path / "actual-cache-reservation-ledger"
        entry = json.loads(reservation.path.read_bytes())
        assert entry["role"] == "g1_checkpoint_cache"
        assert entry["expected_bytes"] == 4096
        assert entry["device"] == workspace.stat().st_dev
        reservation.renew()
        (workspace / "payload").write_bytes(b"cache bytes")
        with pytest.raises(ControlPlaneDiskBudgetError, match="disk_budget_exceeded"):
            reserve(
                "g1_checkpoint_cache",
                target_root=workspace,
                expected_bytes=shutil.disk_usage(workspace).total + 1,
                evictor=None,
            )
    assert not reservation.path.exists()
    samples = (reservation.reservation_root / "history/g1_checkpoint_cache.jsonl").read_text()
    sample = json.loads(samples.strip())
    assert sample["outcome"] == "completed"
    assert sample["observed_bytes"] > 0
    assert configured_launch_input_roots(env={}) == roots_before


def _substitute_gc_paths(line, replacements):
    expression = '|'.join(re.escape(path) for path in sorted(replacements, key=len, reverse=True))
    return re.sub(expression, lambda match: replacements[match.group(0)], line)


def test_shipped_gc_path_substitution_does_not_rewrite_inserted_fixture_root():
    root = Path('/var/lib/blueprint-adp-contained-static-fixture')
    replacements = {'/mnt/blueprint-work/lanes/g1': str(root / 'work/lanes/g1'),
                    '/var/lib/blueprint': str(root / 'sandbox-blueprint')}
    line = 'ReadWritePaths=-/mnt/blueprint-work/lanes/g1 /var/lib/blueprint/storage-gc'
    assert _substitute_gc_paths(line, replacements) == (
        'ReadWritePaths=-' + str(root / 'work/lanes/g1') + ' '
        + str(root / 'sandbox-blueprint/storage-gc'))


def test_native_gc_result_uses_shipped_writable_report_directory():
    root = Path('/var/lib/blueprint-adp-contained-static-fixture')
    report_root = _gc_sandbox_report_root(root, 'a' * 32)
    assert report_root == root / 'sandbox-control-plane/storage-gc' / ('a' * 32)
    unit = (Path(__file__).parents[1] / 'deploy/systemd/blueprint-control-plane-storage-gc.service').read_text()
    assert '/var/lib/blueprint/pipeline-control-plane/storage-gc' in next(
        line for line in unit.splitlines() if line.startswith('ReadWritePaths='))


def install_protected_feature(root):
    """Exact real protected installation reusable by the experiment roundtrip."""
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from tests.test_registered_checkpoint_cache import tiny_inventory

    source = Path(__file__).parents[1]
    package = root / "installed/operator_door"
    package.mkdir(parents=True, mode=0o700)
    for name in ("__init__.py", "config.py"):
        target = package / name
        target.write_bytes((source / "deploy/operator-door/operator_door" / name).read_bytes())
        target.chmod(0o600)
    owners.INSTALLED_PACKAGE_ROOT = package.parent
    state = root / "state"
    state.mkdir(mode=0o755)
    (state / "requests").mkdir(mode=0o755)
    private = state / "requests/needed-checkpoint-cache-records"
    public = state / "needed-checkpoint-cache-registration"
    authority = public / "authority"
    gid = grp.getgrnam("blueprint").gr_gid
    work = root / "work/lanes"
    work.mkdir(parents=True, mode=0o750)
    work.parent.chmod(0o755)
    os.chown(work, 0, gid)
    lock = work / ".lane-scratch.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    lane = work / "g1-checkpoint"
    lane.mkdir(mode=0o750)
    os.chown(lane, 0, gid)
    policy = root / "policy.json"
    policy.write_bytes(
        encoded(
            dict(
                schema_version=owners.POLICY_SCHEMA,
                enabled=True,
                principals=[
                    dict(
                        principal="operator",
                        owners=["owner"],
                        allowed_actions=["register"],
                        max_consent_seconds=3600,
                    )
                ],
            )
        )
    )
    policy.chmod(0o600)
    inventory, payloads = tiny_inventory()
    inventory_path = root / "installed-inventory.json"
    inventory_path.write_bytes(encoded(inventory))
    inventory_path.chmod(0o644)
    config = root / "door.json"
    settings = dict(
        state_root=str(state),
        needed_checkpoint_cache_creation_enabled=True,
        needed_checkpoint_cache_inventory_file=str(inventory_path),
        lane_owner_policy_file=str(policy),
        lane_scratch_work_root=str(work),
        lane_scratch_inputs_root=str(root / "inputs/lanes"),
    )
    config.write_bytes(encoded(settings))
    config.chmod(0o600)
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import control_plane_lane_experiment_retirement as retirement

    # This is a disposable compiled installation repin, before configuration is
    # admitted. The real provisioner creates metadata only, with no enrollment,
    # current HEAD, owner grant or flag mutation.
    consumer.LANE_ROOTS = (work, Path(settings["lane_scratch_inputs_root"]))
    prior_config = config.read_bytes()
    prepared = retirement.prepare_registered_experiment_state(installed_config_path=config)
    assert prepared == dict(decision="prepared", creation_enabled=False,
                            retirement_enabled=False, cache_creation_enabled=True)
    assert config.read_bytes() == prior_config
    expected = (
        (state, 0o755, 0), (state / "requests", 0o755, 0),
        (private, 0o700, 0), (public, 0o755, 0), (authority, 0o750, gid),
        (state / "requests/experiment-records", 0o700, 0),
        (state / "experiment-authority", 0o750, gid),
    )
    for path, mode, group in expected:
        info = path.stat()
        assert info.st_uid == 0 and info.st_gid == group and info.st_mode & 0o777 == mode
    for parent, name, mode, group in (
        (private, ".cache-store.lock", 0o600, 0),
        (authority, ".authority.lock", 0o640, gid),
        (state / "requests/experiment-records", ".experiment-authority.lock", 0o600, 0),
        (state / "experiment-authority", ".authority.lock", 0o640, gid),
    ):
        path = parent / name
        info = path.stat()
        assert info.st_uid == 0 and info.st_gid == group and info.st_mode & 0o777 == mode
        assert info.st_nlink == 1 and path.read_bytes() == b""
    assert not (public / "HEAD.json").exists()
    assert not (state / "experiment-authority/HEAD.json").exists()
    return dict(
        config=config,
        settings=settings,
        private=private,
        public=public,
        authority=authority,
        inventory_path=inventory_path,
        payloads=payloads,
        policy=policy,
        work=work,
    )


def _linux_cache_roundtrip():
    """Executed as actual root, with an actual different-UID forked reader."""
    import fcntl
    import socket
    from urllib.parse import unquote
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import native_g1_checkpoint_cache as native

    assert sys.platform == "linux" and os.geteuid() == 0
    created_account, child = False, None
    try:
        account = pwd.getpwnam("blueprint")
    except KeyError:
        subprocess.run(
            [
                "useradd",
                "--system",
                "--user-group",
                "--no-create-home",
                "--shell",
                "/usr/sbin/nologin",
                "blueprint",
            ],
            check=True,
        )
        created_account = True
        account = pwd.getpwnam("blueprint")
    root = Path(tempfile.mkdtemp(prefix="blueprint-adp-disk-test-", dir="/var/lib"))
    root.chmod(0o755)
    parent_sock = child_sock = None
    try:
        assert account.pw_uid != 0 and account.pw_gid == grp.getgrnam("blueprint").gr_gid
        value = install_protected_feature(root)
        from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer

        consumer.LANE_ROOTS = (value["work"], Path(value["settings"]["lane_scratch_inputs_root"]))
        # Disposable fixed installation repin precedes actual issuance and fill.
        cache._PUBLIC_REGISTRATION = value["public"]
        cache._PUBLIC_INVENTORY = value["inventory_path"]
        cache._REGISTERED_ROOTS = (value["work"],)
        cache.reserve_control_plane_disk = _cache_fixture_disk_reservation(root)
        inventory_raw = value["inventory_path"].read_bytes()
        grant = cache.issue_needed_checkpoint_cache_intent(
            principal="operator",
            owner="owner",
            name="needed-models",
            reference_kind="run_ref",
            reference_value="run1",
            lease_ttl_seconds=1800,
            size_budget_bytes=8 * 1024 * 1024,
            inventory_raw_sha256="sha256:" + hashlib.sha256(inventory_raw).hexdigest(),
            inventory_raw_size_bytes=len(inventory_raw),
            installed_config_path=value["config"],
            now=lambda: 1000,
        )
        fetcher = native._fetcher()

        class Response(io.BytesIO):
            def __init__(self, url):
                super().__init__(value["payloads"][unquote(url.removeprefix(fetcher.MODEL_BASE))])
                self.url = url

            def geturl(self):
                return self.url

        fetcher._open_https = lambda url, **kw: Response(url)
        native._fetcher = lambda: fetcher
        filled = cache.fill_needed_checkpoint_cache(
            grant["intent_id"],
            expected_sha256=grant["intent"]["sha256"],
            expected_size_bytes=grant["intent"]["size_bytes"],
            installed_config_path=value["config"],
            now=lambda: 1100,
        )
        target = Path(filled["path"])
        parent_sock, child_sock = socket.socketpair()
        parent_sock.settimeout(60)
        child_sock.settimeout(60)
        child = os.fork()
        if child == 0:
            parent_sock.close()
            try:
                os.setgroups([])
                os.setgid(account.pw_gid)
                os.setuid(account.pw_uid)
                assert os.geteuid() == account.pw_uid
                denied = 0
                for path in (
                    value["config"],
                    value["policy"],
                    value["private"] / (grant["intent_id"] + ".json"),
                ):
                    try:
                        fd = os.open(path, os.O_RDONLY)
                    except PermissionError:
                        denied += 1
                    else:
                        os.close(fd)
                        raise AssertionError("private authority readable by blueprint")
                assert denied == 3
                # Default API must never open its private door config at this UID.
                with cache.NeededCheckpointCacheUse.open_registered(
                    target, now=lambda: 1200
                ) as use:
                    rows = native.verify_local_g1_checkpoint_cache(target, _cache_use=use)
                    assert len(rows) == 24
                    child_sock.sendall(b"held")
                    assert child_sock.recv(32) == b"revoked"
                    try:
                        use.hash_file(target / next(iter(value["payloads"])), role="wam_hash")
                    except cache.NeededCheckpointCacheError:
                        assert use.failure is not None
                    else:
                        raise AssertionError("revoked reader performed payload work")
                child_sock.sendall(b"closed")
                os._exit(0)
            except BaseException as exc:
                child_sock.sendall(
                    ("failure:" + type(exc).__name__ + ":" + str(exc)).encode()[:512]
                )
                os._exit(1)
        child_sock.close()
        assert parent_sock.recv(512) == b"held"
        independent = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(independent)
        cache.update_needed_checkpoint_cache_authority(
            operation="revoke",
            intent_id=grant["intent_id"],
            installed_config_path=value["config"],
            now=lambda: 1201,
        )
        parent_sock.sendall(b"revoked")
        assert parent_sock.recv(512) == b"closed"
        _, status = os.waitpid(child, 0)
        child = None
        assert os.waitstatus_to_exitcode(status) == 0
        assert all((target / path).read_bytes() == data for path, data in value["payloads"].items())
        return dict(
            status="passed",
            actual_uid=account.pw_uid,
            actual_gid=account.pw_gid,
            private_denials=3,
            verified_files=24,
            retained_target_lock=True,
            root_revoke_observed=True,
            payload_retained=True,
        )
    finally:
        if child:
            os.kill(child, 9)
            os.waitpid(child, 0)
        if parent_sock:
            parent_sock.close()
        if child_sock:
            child_sock.close()
        assert root.parent == Path("/var/lib") and root.name.startswith("blueprint-adp-disk-test-")
        shutil.rmtree(root)
        if created_account:
            subprocess.run(["userdel", "blueprint"], check=True)
            subprocess.run(["groupdel", "blueprint"], check=False)


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") != "1",
    reason="requires explicitly authorized disposable Linux root/UID fixture; Mac skip is unmet",
)
def test_actual_root_blueprint_uid_cache_lifetime_and_revoke(tmp_path):
    """A required successful Linux result, not a skipped-test acceptance claim."""
    command = [sys.executable, str(Path(__file__).resolve()), "--cache-root-fixture"]
    if os.geteuid() != 0:
        command = [
            "sudo",
            "-n",
            "env",
            "BLUEPRINT_DISPOSABLE_LINUX_TEST=1",
            "PYTHONDONTWRITEBYTECODE=1",
            *command,
        ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert (
        receipt["status"] == "passed"
        and receipt["actual_uid"] != 0
        and receipt["verified_files"] == 24
    )


def _fixture_task_spec():
    from tests.test_adp_task_scoring import _rigid_v2_spec
    from tests.test_native_rigid_episode_telemetry import _spec
    return _rigid_v2_spec() | _spec() | {
        "task_kind": "rigid_pick_place", "prompt": "pick the box",
        "start_pose_world": [1.1, 2.1, .8, 0., 0., 0., 1.],
    }


def test_toy_native_episode_has_complete_real_scoring_result(tmp_path, monkeypatch):
    """Validate toy receipts through the real scoring/recording before Linux."""
    from types import SimpleNamespace
    from blueprint_pipeline import native_g1_joint_episode_environment as environment
    from blueprint_pipeline import native_g1_run_preflight as preflight
    from blueprint_pipeline import native_task_arena_readback as readback
    from blueprint_pipeline.native_g1_shared_scene_episode import run_g1_built_scene_policy_episode
    from tests.test_native_g1_shared_scene_episode import _Scene, _Policy, _Bridge
    from tests.test_native_task_episode_environment import _RigidNativeReadback
    scene = _Scene()
    scene.plan = {**scene.plan, 'task_kind': 'rigid_pick_place', 'task_spec': _fixture_task_spec()}
    scene.read_state = lambda: {'step_index': scene.step}
    candidate = 'humanoidarena_dp_g1_dex3_sonic'
    monkeypatch.setattr(environment, 'NativeG1JointEpisodeEnvironment', lambda **kw: scene)
    monkeypatch.setattr(readback, 'NativeRigidTaskArenaReadback', lambda built: _RigidNativeReadback(
        finger_separation_m=.08, grasp_frame_position_world_m=[1.1, 2.1, .9],
        destination_scene_forbidden_contact_peak_force_n=0.))
    monkeypatch.setattr(preflight, 'preflight_g1_shared_scene_run', lambda **kw: {
        'status': 'staged_inputs_verified', 'robot_id': 'unitree_g1', 'candidate_id': candidate,
        'scene_plan_digest': scene.plan['plan_digest'], 'policy_role': 'manipulation'})
    result = run_g1_built_scene_policy_episode(
        built=SimpleNamespace(plan=scene.plan), policy_client=_Policy(), sonic_bridge=_Bridge(),
        candidate_id=candidate, max_steps=3, output_dir=tmp_path / 'episode', preflight_inputs={},
        to_tensor=lambda value: value, make_action_tensor=lambda value, **kw: value)
    assert result['status'] == 'development_only_scored_episode'
    assert result['score']['status'] == 'scored'
    assert result['ranking_eligible'] is False and result['physical_outcome_claimed'] is False
    assert json.loads((tmp_path / 'episode/native_g1_score_attempt.v1.json').read_bytes())['status'] == 'scored'


def _fixture_producer(arguments):
    """Fake CPU hardware/model, with the real native producer and child APIs."""
    from types import SimpleNamespace
    from blueprint_pipeline import native_g1_registered_containment as contained

    if arguments[0] == "--policy-child":
        contained._policy_child(arguments[1:])
        return
    assert (
        len(arguments) == 4
        and arguments[0] == "--producer-bootstrap"
        and arguments[2] == "--target"
    )
    from blueprint_pipeline import native_g1_development_worker as worker
    from blueprint_pipeline import native_g1_runtime_assembly as assembly
    from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor
    from blueprint_pipeline import native_g1_joint_episode_environment as environment
    from blueprint_pipeline import native_task_arena_readback as readback
    from blueprint_pipeline import native_g1_run_preflight as preflight_module
    from tests.test_native_g1_shared_scene_episode import _Scene, _Bridge
    from tests.test_native_task_episode_environment import _RigidNativeReadback

    target = Path(arguments[3])
    root = target.parents[3]

    def preflight(**values):
        plan = json.loads(values["scene_plan_path"].read_bytes())
        # The CPU-only toy model still has the real preflight receipt shape.
        # Bind its candidate inventory to the actual tiny config bytes, using
        # the native inventory's canonical file-list digest convention.
        model = (root / "approved-inputs/policy/config.json").read_bytes()
        files = [{"path": "config.json", "size_bytes": len(model),
                  "sha256": "sha256:" + hashlib.sha256(model).hexdigest()}]
        inventory_digest = "sha256:" + hashlib.sha256(json.dumps(
            files, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        return {
            "status": "staged_inputs_verified",
            "scene_plan_digest": plan["plan_digest"],
            "candidate_id": values["candidate_id"],
            "robot_id": "unitree_g1",
            "policy_role": "manipulation",
            "inventory_file_sha256": "sha256:" + "b" * 64,
            "candidate_inventory_digest": inventory_digest,
        }

    worker.preflight_g1_shared_scene_run = preflight
    supervisor.preflight_g1_shared_scene_run = preflight
    preflight_module.preflight_g1_shared_scene_run = preflight
    worker._verify_packet = lambda path: {
        "arena_scene_plan_digest": json.loads(
            (path / "native_task_arena_scene_plan.v1.json").read_bytes()
        )["plan_digest"]
    }
    worker._launch_scene = lambda **kw: (
        SimpleNamespace(close=lambda: None),
        {"status": "tiny_cpu_launched"},
    )

    class Scene(_Scene):
        def __init__(self, plan):
            super().__init__()
            self.plan = plan

        def read_state(self):
            return {"step_index": self.step}

        def close(self):
            return None

    def build(**kw):
        scene = Scene(kw["plan"])
        return SimpleNamespace(plan=kw["plan"], env=scene), {"status": "tiny_cpu_bound"}

    worker._build_scene = build
    environment.NativeG1JointEpisodeEnvironment = lambda built, **kw: built.env
    readback.NativeRigidTaskArenaReadback = lambda built: _RigidNativeReadback(
        finger_separation_m=0.08,
        grasp_frame_position_world_m=[1.1, 2.1, 0.9],
        destination_scene_forbidden_contact_peak_force_n=0.0,
    )
    assembly.build_pinned_g1_sonic_bridge = lambda **kw: _Bridge()
    supervisor._source_revision = lambda *a, **kw: supervisor.PINNED_SOURCE_REVISION
    supervisor._candidate_policy_dir = lambda *a: root / "approved-inputs/policy"
    supervisor.require_policy_tokenizer_reference = lambda *a: None
    # Scoring, recording, pair, worker, runtime, supervisor, real HTTP client,
    # child bootstrap and the real OS child are deliberately NOT replaced.
    contained._producer_main(Path(arguments[1]), target)


def _linux_contained_phase(root):
    """Real root issuer, blueprint systemd service, joined child and GC reader."""
    import fcntl
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuance
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import native_g1_registered_containment as contained
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_native_g1_development_pair import _paired_requests, _seal_request

    # Disposable-runner diagnostic only: this fixed property allowlist contains
    # no Environment, credentials or request/private record bodies. Preserve the
    # actual native return and every production complete-key/security guard.
    native_control = contained._native_control
    fixed_query = "--property=" + ",".join(contained._UNIT_PROPERTIES)

    def observed_native_control(arguments):
        raw = native_control(arguments)
        if (
            len(arguments) == 6
            and arguments[:2] == [contained._SYSTEMCTL, "show"]
            and arguments[3:] == ["--no-pager", "--all", fixed_query]
        ):
            sys.stderr.write("fixed_systemd_unit_observation=" + json.dumps(raw) + "\n")
            values = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            if values.get("Result") == "exit-code" and values.get("ExecMainStatus") != "0":
                # Only the exact disposable unit selected by the fixed native
                # observer. Its fake CPU/request fixture has no provider or
                # credential payload. Preserve the genuine child's traceback
                # before the root cleanup removes the transient unit.
                unit = arguments[2]
                assert unit.startswith("blueprint-experiment-") and unit.endswith(".service")
                journal = contained._native_control([
                    "/usr/bin/journalctl", "--unit=" + unit, "--no-pager", "--lines=40",
                    "--output=cat", "--quiet",
                ])
                assert len(journal.encode()) <= 65536
                sys.stderr.write("fixed_disposable_child_journal=" + json.dumps(journal) + "\n")
        return raw

    contained._native_control = observed_native_control

    assert (
        os.geteuid() == 0
        and root.parent == Path("/var/lib")
        and root.name.startswith("blueprint-adp-contained-")
    )
    gid = grp.getgrnam("blueprint").gr_gid
    account = pwd.getpwnam("blueprint")
    value = install_protected_feature(root)
    owners.INSTALLED_PACKAGE_ROOT = root / "installed"
    policy = json.loads(value["policy"].read_bytes())
    policy["principals"][0]["allowed_actions"] = ["register", "offload", "keep"]
    value["policy"].write_bytes(encoded(policy))
    pins = root / "pins"
    pins.mkdir(mode=0o700)
    gc_env = root / "gc.env"
    gc_env.write_text("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=" + str(pins) + "\n")
    gc_env.chmod(0o600)
    private = root / "state/requests/experiment-records"
    roots = (value["work"], root / "inputs/lanes")
    for lane_root in roots:
        if not lane_root.exists():
            lane_root.mkdir(parents=True, mode=0o750)
            lane_root.parent.chmod(0o755)
            (lane_root / ".lane-scratch.lock").write_bytes(b"")
            (lane_root / ".lane-scratch.lock").chmod(0o600)
        os.chown(lane_root, 0, gid)
        (lane_root / "g1").mkdir(mode=0o750)
        os.chown(lane_root / "g1", 0, gid)
    settings = value["settings"] | {
        "experiment_creation_enabled": True,
        "experiment_retirement_enabled": True,
        "experiment_gc_environment_file": str(gc_env),
    }
    value["config"].write_bytes(encoded(settings))
    inputs = root / "approved-inputs"
    inputs.mkdir(mode=0o755)
    paths, plan = _paired_requests(inputs)
    plan.update(
        task_spec=_fixture_task_spec(),
        scenario={"seed": 19},
    )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    (inputs / "packet/native_task_arena_scene_plan.v1.json").write_bytes(encoded(plan))
    (inputs / "policy").mkdir(mode=0o755)
    (inputs / "policy/config.json").write_bytes(b'{"type":"fixture"}')
    script = inputs / "tiny-policy.py"
    script.write_text("""import argparse,json,os,signal,time
from http.server import BaseHTTPRequestHandler,HTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int);p.add_argument('--host');p.add_argument('--device');p.add_argument('--policy-path');a=p.parse_args()
child=os.fork()
if child==0:
 while True:time.sleep(.1)
def stop(*args):
 os.kill(child,signal.SIGTERM);os.waitpid(child,0);raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  n=int(self.headers['Content-Length']);assert n<=4000000;json.loads(self.rfile.read(n))
  action=[0.0]*40;action[3:9]=[1,0,0,1,0,0]
  value={'ok':True} if self.path=='/reset' else {'action_chunk':[action]*2}
  payload=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 def log_message(self,*args):pass
HTTPServer.allow_reuse_address=True
HTTPServer((a.host,a.port),Handler).serve_forever()
""")
    script.chmod(0o644)
    for path in paths:
        request = json.loads(path.read_bytes())
        request.update(
            policy_server_source=str(script),
            python_executable=str(root / "fixture-python"),
            max_steps=2,
        )
        path.write_bytes(
            encoded(_seal_request(request, candidate_id=request["candidate_id"], plan=plan))
        )
        path.chmod(0o640)
        os.chown(path, 0, gid)
    now = time.time()
    selectors = tuple(
        (
            path,
            {
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            },
        )
        for path in paths
    )
    grant = issuance.issue_experiment_creation_intent(
        installed_config_path=value["config"],
        principal="operator",
        owner="owner",
        root="work",
        reference_value="tiny-native-pair",
        lease_ttl_seconds=1800,
        participant_profile="g1_local_contained_completed.v1",
        request_records=selectors,
        now=lambda: now,
    )
    born = birth.create_registered_experiment(
        grant["intent_id"],
        expected_intent=grant["intent"],
        installed_config_path=value["config"],
        now=time.time,
    )
    issuance.issue_experiment_producer_bootstrap(
        grant["intent_id"],
        expected_intent_sha256=grant["intent"]["sha256"],
        expected_intent_size_bytes=grant["intent"]["size_bytes"],
        request_paths=paths,
        installed_config_path=value["config"],
        now=time.time,
    )
    result = contained.run_registered_experiment(
        grant["intent_id"], expected_intent=grant["intent"], installed_config_path=value["config"]
    )
    assert result["status"] == "completed"
    target = Path(born["path"])
    receipt = json.loads((target / (pair.SCHEMA + ".json")).read_bytes())
    assert receipt["status"] == "completed_development_only" and len(receipt["attempts"]) == 2
    for row in receipt["attempts"]:
        assert row["score"] is not None and row["review_media"] is not None
        worker = json.loads(Path(row["worker_result_path"]).read_bytes())
        assert worker["teardown"] == {"environment": "closed", "simulator": "closed"}
        assert (
            worker["supervised_episode"]["server_teardown"]["registered_child_lifetime"][
                "direct_child_exited"
            ]
            is True
        )
    with consumer.RegisteredExperimentUse.admit(target) as use:
        use.check()
        first = json.loads(paths[0].read_bytes())
        assert pair._read_result(
            Path(receipt["attempts"][0]["worker_result_path"]),
            candidate_id=first["candidate_id"],
            scene_plan_digest=plan["plan_digest"],
            request_digest=first["request_digest"],
        )
        independent = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError("post-pair reader did not exclude cleanup")
        finally:
            os.close(independent)
    completion = json.loads(
        (private / (grant["intent_id"] + ".producer-completion.json")).read_bytes()
    )
    assert completion["participant_profile"] == "g1_local_contained_completed.v1"
    assert completion["kernel_unit"]["kernel"]["tasks"] == 0
    assert completion["kernel_unit"]["finished"]["ActiveState"] == "inactive"
    roundtrip = _linux_completed_gc_restore(value, grant, born, receipt, paths, now, account, pins)
    arena_result = _linux_arena_owner_review(value, account, pins, now)
    return dict(
        status="passed",
        actual_uid=account.pw_uid,
        candidates=2,
        **arena_result,
        **roundtrip,
        actual_systemd=True,
        actual_native_child=True,
        actual_descendant_join=True,
        root_kernel_completion=True,
        postpair_reader_sh=True,
        exec_start=completion["kernel_unit"]["started"]["ExecStart"],
    )


def _linux_arena_owner_review(value, account, pins, origin):
    """Actual ordinary UID, current SH writer, then shipped GC KEEP only."""
    import fcntl
    import select
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth
    from blueprint_pipeline import control_plane_arena_scratch as arena

    lane = Path(value["settings"]["lane_scratch_inputs_root"]) / "arena"
    lane.mkdir(mode=0o750)
    os.chown(lane, 0, account.pw_gid)
    issued = origin + 1806
    grant = issuer.issue_experiment_creation_intent(
        installed_config_path=value["config"], principal="operator", owner="owner",
        root="inputs", reference_value="arena-launch-r33", lease_ttl_seconds=30,
        participant_profile="arena_owner_review.v1", request_records=(), now=lambda: issued)
    born = birth.create_registered_experiment(grant["intent_id"], expected_intent=grant["intent"],
        installed_config_path=value["config"], now=lambda: issued + 1)
    target = Path(born["path"])
    ready_read, ready_write = os.pipe()
    finish_read, finish_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.close(finish_write)
        try:
            os.initgroups("blueprint", account.pw_gid)
            os.setgid(account.pw_gid)
            os.setuid(account.pw_uid)
            for private in (value["config"], value["policy"],
                            value["config"].parent / "state/requests/experiment-records"):
                try:
                    fd = os.open(private, os.O_RDONLY)
                except PermissionError:
                    pass
                else:
                    os.close(fd)
                    raise AssertionError("Arena ordinary UID opened private owner authority")
            with arena.admit_registered_arena_attempt("r33", now=lambda: issued + 2) as use:
                assert use.path == target and use.entry["intent_id"] == grant["intent_id"]
                assert arena.prepare_arena_attempt("r33", _registered_use=use) == target
                payload = arena.mkdir_arena_payload("r33", "arena_packet", _registered_use=use)
                assert payload == target / "arena_packet"
                use.check()
                (payload / "evidence").write_bytes(b"tiny genuine Arena evidence")
                os.link(payload / "evidence", payload / "copied-evidence")
                use.check()
                os.write(ready_write, encoded({"status": "held", "uid": os.geteuid()}))
                assert os.read(finish_read, 1) == b"x"
            os.write(ready_write, encoded({"status": "closed", "uid": os.geteuid()}))
        except BaseException as error:
            os.write(ready_write, encoded({"status": "failed", "error": repr(error)}))
        finally:
            os.close(ready_write)
            os.close(finish_read)
            os._exit(0)
    os.close(ready_write)
    os.close(finish_read)
    try:
        assert select.select([ready_read], [], [], 30)[0], "Arena actual UID writer timed out"
        response = json.loads(os.read(ready_read, 4097))
        assert response == {"status": "held", "uid": account.pw_uid}, response
        fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        os.write(finish_write, b"x")
        assert select.select([ready_read], [], [], 30)[0]
        response = json.loads(os.read(ready_read, 4097))
        assert response == {"status": "closed", "uid": account.pw_uid}, response
    finally:
        os.close(finish_write)
        os.close(ready_read)
        waited, _ = os.waitpid(child, os.WNOHANG)
        if not waited:
            os.kill(child, 15)
            os.waitpid(child, 0)
    before = {str(path.relative_to(target)): (path.stat().st_ino, path.stat().st_nlink, path.read_bytes())
              for path in target.rglob("*") if path.is_file()}
    action = issuer.issue_experiment_action_intent(grant["intent_id"], principal="operator", owner="owner",
        action="owner_review", expires_at_epoch=issued + 600, installed_config_path=value["config"],
        now=lambda: issued + 31)
    raw = json.loads((value["config"].parent / "state/requests/experiment-records"
                     / (action["action_id"] + ".action.json")).read_bytes())
    assert raw["manifest"] is None
    sandbox = _run_shipped_gc_sandbox(value, action, issued + 32, pins)
    row = next(row for row in sandbox["report"]["registered_experiments"]["outcomes"]
               if row["action_id"] == action["action_id"])
    assert row["decision"] == "kept" and row["reason"] == "owner_review"
    assert row["removed_logical_bytes"] == row["removed_allocated_bytes"] == 0
    after = {str(path.relative_to(target)): (path.stat().st_ino, path.stat().st_nlink, path.read_bytes())
             for path in target.rglob("*") if path.is_file()}
    assert after == before and target.is_dir()
    return {"arena_actual_uid": account.pw_uid, "arena_current_writer_sh": True,
            "arena_expired_shipped_gc_owner_review_kept": True}


def _linux_completed_gc_restore(
    value, grant, born, pair_receipt, request_paths, origin, account, pins
):
    """Successful actual producer proof is consumed by the real GC/restore."""
    import functools
    import select
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_restore as restoration
    from blueprint_pipeline import control_plane_disk_budget as disk
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import control_plane_lane_scratch as scratch
    from tests.test_registered_experiment_offload import Cloud

    target = Path(born["path"])
    metadata = {scratch.LEASE_FILE, ".registered-experiment.v1.json"}
    before = {
        str(p.relative_to(target)): p.read_bytes()
        for p in target.rglob("*")
        if p.is_file() and p.name not in metadata
    }
    assert before and pair_receipt["status"] == "completed_development_only"
    original_inode = target.stat().st_ino
    clock = origin + 1802
    action = issuer.issue_experiment_action_intent(
        grant["intent_id"],
        principal="operator",
        owner="owner",
        action="offload",
        expires_at_epoch=clock + 600,
        installed_config_path=value["config"],
        now=lambda: clock,
    )
    sandbox = _run_shipped_gc_sandbox(value, action, clock + 1, pins)
    report = sandbox['report']
    cloud = Cloud()
    cloud.objects = {item['key']: base64.b64decode(item['payload'], validate=True)
                     for item in sandbox['objects']}
    cloud.metadata = sandbox['object_metadata']
    # Restore consumes the exact bytes uploaded and freshly read back by the
    # actual shipped GC sandbox. The fake object client performs no network.
    archive._client = lambda *args: (cloud, "development-only")
    rows = report["registered_experiments"]["outcomes"]
    row = next(row for row in rows if row["action_id"] == action["action_id"])
    assert row["decision"] == "retired", row
    assert row["receipt"] and row["removed_logical_bytes"] > 0
    assert target.stat().st_ino == original_inode and {p.name for p in target.iterdir()} == metadata
    events = [
        json.loads(p.read_bytes())
        for p in (
            value["config"].parent
            / "state/requests/experiment-records/operations"
            / action["action_id"]
        ).glob("e-*.json")
    ]
    ready = next(e for e in events if e["event_kind"] == "preservation_ready")
    assert ready["body"]["archive"]["full_byte_service_account_readback_passed"] is True
    assert ready["sequence"] < min(
        e["sequence"] for e in events if e["event_kind"] == "member_removed"
    )
    assert cloud.objects and sandbox['all_native_responses_closed']
    restore = issuer.issue_experiment_restore_intent(
        grant["intent_id"],
        principal="operator",
        owner="owner",
        lease_ttl_seconds=600,
        expires_at_epoch=clock + 600,
        installed_config_path=value["config"],
        now=lambda: clock + 2,
    )
    ledger = value["config"].parent / "actual-restore-reservation-ledger"
    ledger.mkdir(mode=0o700)
    # Redirect only the protected installation's ledger path; admission,
    # filesystem measurement, atomic reservation and release are actual code.
    restoration.reserve_control_plane_disk = functools.partial(
        disk.reserve_control_plane_disk, reservation_root=ledger
    )
    outcome = issuer.restore_registered_experiment(
        restore["action_id"],
        expected_restore_intent=restore["restore_intent"],
        installed_config_path=value["config"],
        now=lambda: clock + 3,
        _pins_root=pins,
    )
    assert outcome["decision"] == "restored", outcome
    assert target.stat().st_ino == original_inode
    after = {
        str(p.relative_to(target)): p.read_bytes()
        for p in target.rglob("*")
        if p.is_file() and p.name not in metadata
    }
    assert after == before
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read)
        try:
            os.initgroups("blueprint", account.pw_gid)
            os.setgid(account.pw_gid)
            os.setuid(account.pw_uid)
            for private in (
                value["config"],
                value["policy"],
                value["config"].parent / "state/requests/experiment-records",
            ):
                try:
                    fd = os.open(private, os.O_RDONLY)
                except PermissionError:
                    pass
                else:
                    os.close(fd)
                    raise AssertionError("ordinary reader accessed private authority")
            with consumer.RegisteredExperimentUse.admit(target, now=lambda: clock + 4) as use:
                use.check()
                from blueprint_pipeline import native_g1_development_pair as pair

                request = json.loads(request_paths[0].read_bytes())
                assert pair._read_result(
                    Path(pair_receipt["attempts"][0]["worker_result_path"]),
                    candidate_id=request["candidate_id"],
                    scene_plan_digest=request["rights_review"]["scene_plan_digest"],
                    request_digest=request["request_digest"],
                )
            os.write(write, encoded({"status": "passed", "uid": os.geteuid()}))
        except BaseException as error:
            os.write(write, encoded({"status": "failed", "error": repr(error)}))
        finally:
            os.close(write)
            os._exit(0)
    os.close(write)
    try:
        assert select.select([read], [], [], 60)[0], "ordinary restored reader timed out"
        result = json.loads(os.read(read, 4097))
        assert result == {"status": "passed", "uid": account.pw_uid}, result
    finally:
        os.close(read)
        os.waitpid(child, 0)
    return dict(
        actual_expired_gc_offload=True,
        actual_shipped_gc_sandbox=True,
        native_archive_full_readback=True,
        actual_root_restore=True,
        ordinary_uid_restored_reader=True,
    )


def _gc_sandbox_report_root(root, action_id):
    """Keep per-action native evidence inside the shipped GC write allowlist."""
    assert isinstance(action_id, str) and len(action_id) == 32
    assert all(char in '0123456789abcdef' for char in action_id)
    return root / 'sandbox-control-plane/storage-gc' / action_id


def _run_shipped_gc_sandbox(value, action, clock, pins, *, realtime=False, invocation=0):
    """Run actual GC under the shipped unit's protections and finite RW roots."""
    root = value['config'].parent
    installed = root / 'installed'
    report_root = _gc_sandbox_report_root(root, action['action_id'])
    if invocation:
        assert type(invocation) is int and 0 < invocation <= 3
        report_root = report_root / ('attempt-' + str(invocation))
    report_root.mkdir(parents=True, mode=0o700)
    selected = report_root / 'selected.json'
    selected.write_bytes(encoded(dict(config=str(value['config']), pins=str(pins), now=clock,
                                     action_id=action['action_id'], realtime=realtime)))
    selected.chmod(0o600)
    wrapper = root / 'fixture-gc-python'
    dependencies = installed / 'dependencies'
    wrapper.write_text('#!' + str(Path(sys._base_executable).resolve()) + '\nimport sys\n'
        + 'sys.path[:0]=' + repr([str(installed), str(dependencies)]) + '\n'
        + 'from tests.test_registered_feature_linux import _gc_sandbox_main\n'
        + 'assert len(sys.argv)==2\n_gc_sandbox_main(sys.argv[1])\n')
    wrapper.chmod(0o755)
    unit = 'blueprint-experiment-gc-' + action['action_id'] + '.service'
    unit_path = Path('/etc/systemd/system') / unit
    assert not unit_path.exists() and not unit_path.is_symlink()
    original = (installed / 'deploy/systemd/blueprint-control-plane-storage-gc.service').read_text()
    replacements = {
        '/var/lib/blueprint-operator-door/requests/experiment-records': str(root / 'state/requests/experiment-records'),
        '/var/lib/blueprint-operator-door/experiment-authority': str(root / 'state/experiment-authority'),
        '/mnt/blueprint-work/lanes/g1': str(value['work'] / 'g1'),
        '/mnt/blueprint-work/lanes/diagnostics': str(value['work'] / 'diagnostics'),
        '/var/lib/blueprint/task-evaluation-inputs/lanes/diagnostics': str(root / 'inputs/lanes/diagnostics'),
        '/var/lib/blueprint/task-evaluation-inputs/lanes/g1': str(root / 'inputs/lanes/g1'),
        '/var/lib/blueprint/pipeline-control-plane/storage-pins': str(pins),
        '/var/lib/blueprint/pipeline-control-plane': str(root / 'sandbox-control-plane'),
        '/var/lib/blueprint/task-evaluation-inputs': str(root / 'sandbox-inputs'),
        '/var/lib/blueprint/pubsub-handoffs': str(root / 'sandbox-pubsub'),
        '/var/lib/blueprint': str(root / 'sandbox-blueprint'),
        '/etc/blueprint': str(root / 'sandbox-etc'),
    }
    lines = []
    for line in original.splitlines():
        line = _substitute_gc_paths(line, replacements)
        if line.startswith('ExecStart='):
            line = 'ExecStart=' + str(wrapper) + ' ' + str(selected)
        if line.startswith(('ReadWritePaths=', 'ReadOnlyPaths=')):
            for path in line.split('=', 1)[1].split():
                Path(path.removeprefix('-')).mkdir(parents=True, exist_ok=True)
        lines.append(line)
    # Preserve all shipped restrictions. This disposable job has no provider
    # network, and its only executable/data substitutions are root protected.
    lines.extend(['Environment=PYTHONDONTWRITEBYTECODE=1', 'PrivateNetwork=yes',
        'BindReadOnlyPaths=' + str(Path(__import__('site').getsitepackages()[0])) + ':' + str(dependencies)])
    unit_path.write_text('\n'.join(lines) + '\n')
    unit_path.chmod(0o644)
    try:
        subprocess.run(['/usr/bin/systemctl', 'daemon-reload'], check=True)
        started = subprocess.run(['/usr/bin/systemctl', 'start', unit], capture_output=True,
                                 text=True, timeout=90)
        if started.returncode:
            log = subprocess.run(['/usr/bin/journalctl', '--unit=' + unit, '--no-pager',
                '--lines=40', '--output=cat', '--quiet'], capture_output=True, text=True)
            assert len(log.stdout.encode()) <= 65536
            raise AssertionError(started.stderr + log.stdout)
        output = report_root / 'native-result.json'
        raw = output.read_bytes()
        assert len(raw) <= 10 * 1024**2 and output.stat().st_uid == 0
        result = json.loads(raw)
        assert result['actual_shipped_sandbox'] and result['outside_rw_refused']
        assert result['no_new_privileges'] == '1'
        assert result['effective_caps'] == (1 | 2 | (1 << 19))
        return result
    finally:
        subprocess.run(['/usr/bin/systemctl', 'stop', unit], check=False)
        unit_path.unlink()
        subprocess.run(['/usr/bin/systemctl', 'daemon-reload'], check=True)


def _retain_native_limit(method, failures):
    """Observe an actual failed native method without changing its refusal."""
    def observed(files, *args, **kwargs):
        try:
            return method(files, *args, **kwargs)
        except ValueError as error:
            if str(error) == 'owner_target_resource_exhausted' and len(failures) < 8:
                trace, frames = error.__traceback__, []
                while trace is not None and len(frames) < 16:
                    frames.append(dict(function=trace.tb_frame.f_code.co_name, line=trace.tb_lineno))
                    trace = trace.tb_next
                failures.append(dict(owner_class=type(files).__name__, method=method.__name__,
                    code=str(error), frames=frames, owned=len(files.owned), probes=len(files.probe_owned),
                    raw_cap=files.raw_cap, metadata_bytes=getattr(files, 'metadata_bytes', None),
                    budget_counts=dict(files.budget.counts), budget_limits=dict(files.budget.limits),
                    budget_last=files.budget.last, budget_deadline=files.budget.deadline))
            raise
    return observed


def _gc_sandbox_main(selected):
    """Root fixture entry: real GC, fake archive service, actual kernel sandbox."""
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline.control_plane_storage_gc import run_storage_gc, RUN_ACK
    from tests.test_registered_experiment_offload import Cloud
    from blueprint_pipeline.control_plane_lane_owner_target_io import _TargetFiles
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import _ReferenceFiles

    selected = Path(selected)
    assert os.geteuid() == 0 and selected.name == 'selected.json'
    roots = [path for path in selected.parents if path.parent == Path('/var/lib')
             and path.name.startswith('blueprint-adp-contained-')]
    assert len(roots) == 1
    root = roots[0]
    selection = json.loads(selected.read_bytes())
    assert set(selection) == {'config', 'pins', 'now', 'action_id', 'realtime'}
    assert type(selection['realtime']) is bool
    assert selection['config'] == str(root / 'door.json')
    owners.INSTALLED_PACKAGE_ROOT = root / 'installed'
    settings = json.loads((root / 'door.json').read_bytes())
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    effective = int(status['CapEff'].strip(), 16)
    assert effective == (1 | 2 | (1 << 19)) and status['NoNewPrivs'].strip() == '1'
    probe = root / 'outside-gc-rw-probe'
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        assert error.errno == errno.EROFS
    else:
        os.close(fd)
        raise AssertionError('shipped ProtectSystem=strict failed to fence unlisted writes')
    cloud = Cloud()
    archive._client = lambda *args: (cloud, 'development-only')
    native_limit_failures = []
    # Metadata-only observers execute and rethrow the actual original methods.
    # No process view, descriptor, clock or allowance is replaced here.
    _TargetFiles.slot = _retain_native_limit(_TargetFiles.slot, native_limit_failures)
    _TargetFiles.read_bytes = _retain_native_limit(_TargetFiles.read_bytes, native_limit_failures)
    _ReferenceFiles.read_bytes = _retain_native_limit(_ReferenceFiles.read_bytes, native_limit_failures)
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import DiagnosticReferences
    from tests.registered_disk_diagnostic_native_acceptance import (
        PidOpenEvidence, observe_pid_opens, observe_reference_failures,
    )
    native_reference_failures = {'records': [], 'truncated': False}
    observe_reference_failures(DiagnosticReferences, native_reference_failures)
    pid_open_evidence = PidOpenEvidence()
    try:
        with observe_pid_opens(pid_open_evidence):
            report = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(),
                pins_root=Path(selection['pins']), apply=True, ack=RUN_ACK,
                lane_scratch_roots=(settings['lane_scratch_work_root'], settings['lane_scratch_inputs_root']),
                lane_scratch_enabled=True, _experiment_config_path=root / 'door.json',
                now=time.time if selection['realtime'] else lambda: selection['now'])
    finally:
        try:
            native_reference_failures['pid_open_evidence'] = pid_open_evidence.packet()
        except Exception:
            pass  # Diagnostic projection never replaces the original GC refusal.
    outcomes = report['registered_experiments']['outcomes']
    chosen = next((row for row in outcomes if row['action_id'] == selection['action_id']), None)
    if chosen is None:
        assert not outcomes and not cloud.objects and not cloud.bodies
    elif chosen['decision'] == 'retired':
        action = json.loads((root / 'state/requests/experiment-records' / (selection['action_id'] + '.action.json')).read_bytes())
        assert bool(cloud.objects) == (action['action'] == 'offload')
        assert all(body.closed for body in cloud.bodies)
    else:
        assert chosen['decision'] == 'kept'
        assert chosen['removed_logical_bytes'] == chosen['removed_allocated_bytes'] == 0
        assert not cloud.objects and not cloud.bodies
    result = dict(report={'registered_experiments': report['registered_experiments']},
        native_limit_failures=native_limit_failures,
        native_reference_failures=native_reference_failures,
        objects=[dict(key=key, payload=base64.b64encode(raw).decode('ascii'))
                 for key, raw in cloud.objects.items()], object_metadata=cloud.metadata,
        all_native_responses_closed=True, actual_shipped_sandbox=True,
        outside_rw_refused=True, no_new_privileges=status['NoNewPrivs'].strip(), effective_caps=effective)
    raw = encoded(result)
    assert len(raw) <= 10 * 1024**2
    output = selected.parent / 'native-result.json'
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _linux_contained_roundtrip(*, phase="--contained-root-phase"):
    """Install only disposable protected code/input paths, never a live host."""
    assert sys.platform == "linux" and os.geteuid() == 0
    created_account = False
    try:
        pwd.getpwnam("blueprint")
    except KeyError:
        subprocess.run(
            [
                "useradd",
                "--system",
                "--user-group",
                "--no-create-home",
                "--shell",
                "/usr/sbin/nologin",
                "blueprint",
            ],
            check=True,
        )
        created_account = True
    root = Path(tempfile.mkdtemp(prefix="blueprint-adp-contained-", dir="/var/lib"))
    root.chmod(0o755)
    try:
        source = Path(__file__).parents[1]
        installed = root / "installed"
        installed.mkdir(mode=0o755)
        for folder in ("blueprint_pipeline", "tests", "scripts"):
            origin = (
                source / "src/blueprint_pipeline"
                if folder == "blueprint_pipeline"
                else source / folder
            )
            for path in origin.rglob("*.py"):
                target = installed / folder / path.relative_to(origin)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                target.write_bytes(path.read_bytes())
                target.chmod(0o644)
        operator = installed / "deploy/operator-door/operator_door"
        operator.mkdir(parents=True, mode=0o755)
        for name in ("__init__.py", "config.py"):
            (operator / name).write_bytes(
                (source / "deploy/operator-door/operator_door" / name).read_bytes()
            )
            (operator / name).chmod(0o644)
        shipped_gc = installed / 'deploy/systemd/blueprint-control-plane-storage-gc.service'
        shipped_gc.parent.mkdir(parents=True, mode=0o755)
        shipped_gc.write_bytes((source / 'deploy/systemd/blueprint-control-plane-storage-gc.service').read_bytes())
        shipped_gc.chmod(0o644)
        substitutions = {
            '/etc/systemd/system/blueprint-control-plane-storage-gc.service': str(shipped_gc),
            "/mnt/blueprint-work/lanes": str(root / "work/lanes"),
            "/var/lib/blueprint/task-evaluation-inputs/lanes": str(root / "inputs/lanes"),
            "/var/lib/blueprint-operator-door/experiment-authority": str(
                root / "state/experiment-authority"
            ),
            "/opt/blueprint/task-evaluation-control-plane/.venv/bin/python": str(
                root / "fixture-python"
            ),
        }
        for name in (
            "control_plane_scratch_lifetime",
            "control_plane_lane_experiment_consumer",
            "control_plane_lane_scratch",
            "control_plane_lane_scratch_retention",
            "native_g1_development_pair",
            "native_g1_registered_containment",
            'control_plane_lane_legacy_owner',
        ):
            path = installed / "blueprint_pipeline" / (name + ".py")
            text = path.read_text()
            for old, new in substitutions.items():
                text = text.replace(old, new)
            path.write_text(text)
        site = Path(__import__("site").getsitepackages()[0])
        dependencies = installed / "dependencies"
        dependencies.mkdir(mode=0o755)
        contained = installed / "blueprint_pipeline/native_g1_registered_containment.py"
        text = contained.read_text().replace(
            '"ReadOnlyPaths=/"]',
            '"ReadOnlyPaths=/", "BindReadOnlyPaths=' + str(site) + ":" + str(dependencies) + '"]',
        )
        contained.write_text(text)
        wrapper = root / "fixture-python"
        wrapper.write_text(
            "#!"
            + str(Path(sys._base_executable).resolve())
            + "\nimport sys\n"
            + "sys.path[:0]="
            + repr([str(installed), str(dependencies)])
            + "\n"
            + "from tests.test_registered_feature_linux import _fixture_producer\n"
            + 'assert sys.argv[1:3]==["-m","blueprint_pipeline.native_g1_registered_containment"]\n'
            + "_fixture_producer(sys.argv[3:])\n"
        )
        wrapper.chmod(0o755)
        code = installed / "tests/test_registered_feature_linux.py"
        env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(installed)}
        value = subprocess.run(
            [sys.executable, str(code), phase, str(root)],
            capture_output=True,
            text=True,
            timeout=240,
            env=env,
            cwd=installed,
        )
        assert value.returncode == 0, value.stdout + value.stderr
        return json.loads(value.stdout.strip().splitlines()[-1])
    finally:
        units = subprocess.run(
            [
                "/usr/bin/systemctl",
                "list-units",
                "--all",
                "--plain",
                "--no-legend",
                "blueprint-experiment-*.service",
            ],
            capture_output=True,
            text=True,
        )
        # Stop only units whose fixed command actually selects this disposable root.
        for row in units.stdout.splitlines():
            unit = row.split()[0]
            observed = subprocess.run(
                ["/usr/bin/systemctl", "show", unit, "--property=ExecStart"],
                capture_output=True,
                text=True,
            )
            if str(root / "fixture-python") in observed.stdout:
                subprocess.run(["/usr/bin/systemctl", "stop", unit], check=False)
        shutil.rmtree(root)
        if created_account:
            subprocess.run(["userdel", "blueprint"], check=True)
            subprocess.run(["groupdel", "blueprint"], check=False)


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") != "1",
    reason="mandatory disposable root/systemd/blueprint UID native pair; Mac skip is unmet",
)
def test_actual_contained_blueprint_native_pair_child_and_kernel_completion():
    command = [sys.executable, str(Path(__file__).resolve()), "--contained-root-fixture"]
    if os.geteuid() != 0:
        command = [
            "sudo",
            "-n",
            "env",
            "BLUEPRINT_DISPOSABLE_LINUX_TEST=1",
            "PYTHONDONTWRITEBYTECODE=1",
            *command,
        ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert (
        receipt["status"] == "passed" and receipt["actual_uid"] != 0 and receipt["candidates"] == 2
    )
    assert (
        receipt["actual_systemd"]
        and receipt["actual_native_child"]
        and receipt["root_kernel_completion"]
    )
    assert receipt["postpair_reader_sh"]
    assert receipt['actual_shipped_gc_sandbox']
    assert receipt["actual_expired_gc_offload"] and receipt["native_archive_full_readback"]
    assert receipt["actual_root_restore"] and receipt["ordinary_uid_restored_reader"]
    assert receipt["arena_actual_uid"] != 0 and receipt["arena_current_writer_sh"]
    assert receipt["arena_expired_shipped_gc_owner_review_kept"]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') != '1',
    reason='mandatory installed root diagnostic, real expiry, shipped GC and distinct UID; Mac skip unmet')
def test_actual_registered_disk_diagnostic_delete_offload_and_restore():
    command = [sys.executable, str(Path(__file__).resolve()), '--diagnostic-root-fixture']
    if os.geteuid() != 0:
        command = ['sudo', '-n', 'env', 'BLUEPRINT_DISPOSABLE_LINUX_TEST=1',
                   'PYTHONDONTWRITEBYTECODE=1', *command]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True,
        timeout=300, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt['status'] == 'passed' and receipt['actual_ordinary_uid'] != 0
    assert receipt['real_lease_expiry'] and receipt['default_off_kept']
    assert receipt['actual_delete'] and receipt['actual_offload'] and receipt['full_readback_before_remove']
    assert receipt['actual_restore'] and receipt['restore_no_overwrite'] and receipt['sealed_writer_denied']
    assert receipt['actual_open_fd_kept'] and receipt['zero_repeat_credit'] and receipt['shipped_gc_sandbox']
    assert receipt['actual_encoded_queue_kept'] and receipt['actual_directory_alias_kept']


if __name__ == "__main__":
    assert os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") == "1", (
        "disposable fixture opt-in required"
    )
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    sys.path.insert(0, str(Path(__file__).parents[1]))
    if sys.argv[1:] == ["--cache-root-fixture"]:
        value = _linux_cache_roundtrip()
    elif sys.argv[1:] == ["--contained-root-fixture"]:
        value = _linux_contained_roundtrip()
    elif sys.argv[1:] == ["--diagnostic-root-fixture"]:
        value = _linux_contained_roundtrip(phase="--diagnostic-root-phase")
    else:
        assert len(sys.argv) == 3
        if sys.argv[1] == "--diagnostic-root-phase":
            from tests.registered_disk_diagnostic_native_acceptance import run_observed
            value = run_observed(Path(sys.argv[2]))
        else:
            assert sys.argv[1] == "--contained-root-phase"
            value = _linux_contained_phase(Path(sys.argv[2]))
    print(json.dumps(value, sort_keys=True))
