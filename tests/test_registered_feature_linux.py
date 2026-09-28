"""ADP-009D real root/blueprint UID acceptance on a disposable hosted Linux runner.

Requires explicit BLUEPRINT_DISPOSABLE_LINUX_TEST=1. A Mac skip is NOT proof.
No production host, remote provider, cloud SDK, or actual network is contacted.
Root-owned installation lives under /var/lib, with actual blueprint group modes.
"""

from __future__ import annotations

import grp
import hashlib
import io
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


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
    private = state / "requests/needed-checkpoint-cache-records"
    private.mkdir(parents=True, mode=0o700)
    private.parent.chmod(0o700)
    public = state / "needed-checkpoint-cache-registration"
    public.mkdir(mode=0o755)
    authority = public / "authority"
    authority.mkdir(mode=0o750)
    gid = grp.getgrnam("blueprint").gr_gid
    os.chown(authority, 0, gid)
    for parent, name, mode, selected_gid in (
        (private, ".cache-store.lock", 0o600, 0),
        (authority, ".authority.lock", 0o640, gid),
    ):
        path = parent / name
        path.write_bytes(b"")
        os.chown(path, 0, selected_gid)
        path.chmod(mode)
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
        # Disposable fixed installation repin precedes actual issuance and fill.
        cache._PUBLIC_REGISTRATION = value["public"]
        cache._PUBLIC_INVENTORY = value["inventory_path"]
        cache._REGISTERED_ROOTS = (value["work"],)
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
        return {
            "status": "staged_inputs_verified",
            "scene_plan_digest": plan["plan_digest"],
            "candidate_id": values["candidate_id"],
            "robot_id": "unitree_g1",
            "policy_role": "manipulation",
            "inventory_file_sha256": "sha256:" + "b" * 64,
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
    from tests.test_adp_task_scoring import _rigid_v2_spec

    assert (
        os.geteuid() == 0
        and root.parent == Path("/var/lib")
        and root.name.startswith("blueprint-adp-contained-")
    )
    gid = grp.getgrnam("blueprint").gr_gid
    account = pwd.getpwnam("blueprint")
    value = install_protected_feature(root)
    owners.INSTALLED_PACKAGE_ROOT = root / "installed"
    private = root / "state/requests/experiment-records"
    private.mkdir(mode=0o700)
    (private / ".experiment-authority.lock").write_bytes(b"")
    (private / ".experiment-authority.lock").chmod(0o600)
    public = root / "state/experiment-authority"
    public.mkdir(mode=0o750)
    os.chown(public, 0, gid)
    (public / ".authority.lock").write_bytes(b"")
    (public / ".authority.lock").chmod(0o640)
    os.chown(public / ".authority.lock", 0, gid)
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
    }
    value["config"].write_bytes(encoded(settings))
    inputs = root / "approved-inputs"
    inputs.mkdir(mode=0o755)
    paths, plan = _paired_requests(inputs)
    plan.update(
        task_spec=_rigid_v2_spec() | {"task_kind": "rigid_pick_place", "prompt": "pick the box"},
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
        now=lambda: now + 1,
    )
    issuance.issue_experiment_producer_bootstrap(
        grant["intent_id"],
        expected_intent_sha256=grant["intent"]["sha256"],
        expected_intent_size_bytes=grant["intent"]["size_bytes"],
        request_paths=paths,
        installed_config_path=value["config"],
        now=lambda: now + 2,
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
    return dict(
        status="passed",
        actual_uid=account.pw_uid,
        candidates=2,
        actual_systemd=True,
        actual_native_child=True,
        actual_descendant_join=True,
        root_kernel_completion=True,
        postpair_reader_sh=True,
        exec_start=completion["kernel_unit"]["started"]["ExecStart"],
    )


def _linux_contained_roundtrip():
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
        for folder in ("blueprint_pipeline", "tests"):
            origin = (
                source / "src/blueprint_pipeline"
                if folder == "blueprint_pipeline"
                else source / "tests"
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
        substitutions = {
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
            "native_g1_registered_containment",
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
            [sys.executable, str(code), "--contained-root-phase", str(root)],
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
    else:
        assert len(sys.argv) == 3 and sys.argv[1] == "--contained-root-phase"
        value = _linux_contained_phase(Path(sys.argv[2]))
    print(json.dumps(value, sort_keys=True))
