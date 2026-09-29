"""Actual G1 pair/worker payload is protected by authenticated current target SH."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_consumer.py
#   src/blueprint_pipeline/native_g1_development_pair.py
#   src/blueprint_pipeline/native_g1_development_worker.py

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth, prepare
from tests.test_native_g1_development_pair import _paired_requests


def test_contained_producer_forwards_admitted_paths_to_actual_native_validator(
    installation, tmp_path, monkeypatch  # noqa: F811
):
    from blueprint_pipeline import native_g1_registered_containment as contained
    from blueprint_pipeline import native_g1_development_pair as pair

    consumer, target, _, paths, bootstrap = _contained_bootstrap(
        installation, tmp_path, monkeypatch
    )
    original_admit = consumer.RegisteredExperimentUse.admit
    held = []

    def admit(path, *, _producer_bootstrap_path):
        use = original_admit(path, _producer_bootstrap_path=_producer_bootstrap_path,
                             now=lambda: 1200)
        held.append(use)
        return use

    monkeypatch.setattr(consumer.RegisteredExperimentUse, 'admit', staticmethod(admit))
    # This portable case isolates the real input API. Actual kernel membership
    # and successful child completion remain mandatory in the Linux test.
    monkeypatch.setattr(contained, '_unit_membership', lambda *args: None)

    class NativeInputVerified(Exception):
        pass

    def native_pair(*, request_paths, output_dir, mode, _registered_use):
        assert [str(path) for path in request_paths] == [str(path) for path in paths]
        assert output_dir == target and mode == 'local' and _registered_use is held[0]
        _registered_use.check()
        for path in request_paths:
            # Exercise actual filesystem Path operations and raw native seals,
            # without running hardware, providers or fabricating completion.
            assert pair._sealed_json(path)['request_digest']
        raise NativeInputVerified

    monkeypatch.setattr(pair, 'run_g1_development_pair', native_pair)
    with pytest.raises(NativeInputVerified):
        contained._producer_main(bootstrap, target)
    assert len(held) == 1 and held[0]._closed and not held[0].files.owned


@pytest.mark.parametrize("caller", ["pair", "worker"])
def test_registered_missing_use_refuses_before_request_or_payload_read(
    tmp_path, monkeypatch, caller
):
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker

    root = tmp_path / "lanes"
    target = root / "g1" / ("registered-" + "a" * 32)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime

    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("unadmitted payload read"))
    with pytest.raises(ValueError, match="experiment_consumer_authority_required"):
        if caller == "pair":
            pair.run_g1_development_pair(request_paths=[tmp_path / "absent"], output_dir=target)
        else:
            worker.run_g1_development_worker(request={}, output_dir=target / "candidate")


def test_actual_registered_pair_and_worker_preflight_output_holds_target_sh(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as issuer
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker

    prepare(installation)
    request_root = installation[0].parent / 'producer-requests'
    request_root.mkdir(mode=0o700)
    paths, _ = _paired_requests(request_root)
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
    grant = issue(
        installation, participant_profile="g1_local_prelaunch_block.v1", request_records=selectors  # noqa: F811
    )
    monkeypatch.setattr(issuer, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *a: None)
    born = birth(installation, grant)
    root = Path(installation[1]["lane_scratch_work_root"])
    monkeypatch.setattr(
        consumer, "LANE_ROOTS", (root, Path(installation[1]["lane_scratch_inputs_root"]))
    )
    monkeypatch.setattr(
        consumer, "AUTHORITY_ROOT", installation[2].parents[1] / "experiment-authority"
    )
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime

    monkeypatch.setattr(lifetime, "LANE_ROOTS", consumer.LANE_ROOTS)
    use = consumer.RegisteredExperimentUse.admit(
        Path(born["path"]),
        expected_birth=born["birth"],
        expected_generation=born["generation"],
        now=lambda: 1200,
        _producer_request_paths=paths,
        _producer_config_path=installation[0],
    )
    entered = []

    def preflight(request):
        entered.append(True)
        fd = os.open(born["path"], os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        raise ValueError("tiny_fixture_preflight_refused")

    monkeypatch.setattr(worker, "_preflight_inputs", preflight)
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=Path(born["path"]), mode="local", _registered_use=use
    )
    assert result["status"] == "blocked" and entered == [True]
    attempt = result["attempts"][0]
    receipt = json.loads(Path(attempt["worker_result_path"]).read_bytes())
    assert receipt["phase_reached"] == "preflight" and receipt["teardown"] == {
        "environment": "not_started",
        "simulator": "not_started",
    }
    assert use._closed
    fd = os.open(born["path"], os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def _registered_fixture(
    installation, tmp_path, monkeypatch, *, profile="g1_local_prelaunch_block.v1"  # noqa: F811
):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as issuer
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer

    prepare(installation)
    request_root = installation[0].parent / 'producer-requests'
    request_root.mkdir(mode=0o700)
    paths, _ = _paired_requests(request_root)
    if profile == "g1_local_contained_completed.v1":
        for path in paths:
            path.chmod(0o640)
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
    grant = issue(installation, participant_profile=profile, request_records=selectors)
    monkeypatch.setattr(issuer, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *a: None)
    born = birth(installation, grant)
    monkeypatch.setattr(
        consumer,
        "LANE_ROOTS",
        tuple(
            Path(installation[1][key])
            for key in ("lane_scratch_work_root", "lane_scratch_inputs_root")
        ),
    )
    monkeypatch.setattr(
        consumer, "AUTHORITY_ROOT", installation[2].parents[1] / "experiment-authority"
    )
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    monkeypatch.setattr(consumer, "PRODUCER_CONFIG_PATH", installation[0])
    return consumer, Path(born["path"]), born, paths


@pytest.mark.parametrize("fault", ["expired", "wrong_generation", "changed_request"])
def test_failed_real_admission_releases_target_sh(installation, tmp_path, monkeypatch, fault):  # noqa: F811
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    if fault == "changed_request":
        paths[0].write_bytes(b"{}")
    with pytest.raises(ValueError):
        consumer.RegisteredExperimentUse.admit(
            target,
            expected_birth=born["birth"],
            expected_generation="f" * 32 if fault == "wrong_generation" else born["generation"],
            now=lambda: 2900 if fault == "expired" else 1200,
            _producer_request_paths=paths,
        )
    fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_plain_read_admission_is_not_root_producer_permission(installation, tmp_path, monkeypatch):  # noqa: F811
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    with consumer.RegisteredExperimentUse.admit(
        target, expected_birth=born["birth"], now=lambda: 1200
    ) as use:
        with pytest.raises(ValueError, match="experiment_producer_authority_required"):
            use.authorize_g1_pair(paths)


def test_admitted_producer_rechecks_original_request_before_payload(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    with consumer.RegisteredExperimentUse.admit(
        target, expected_birth=born["birth"], now=lambda: 1200, _producer_request_paths=paths
    ) as use:
        paths[0].write_bytes(b"{}")
        with pytest.raises(ValueError, match="experiment_producer_request_changed"):
            use.authorize_g1_pair(paths)


def test_actual_prelaunch_pair_closure_publishes_authenticated_completion_for_offload(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker

    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", consumer.LANE_ROOTS)
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime

    monkeypatch.setattr(lifetime, "LANE_ROOTS", consumer.LANE_ROOTS)

    def preflight(_request):
        raise ValueError("development_only_prelaunch_refusal")

    monkeypatch.setattr(worker, "_preflight_inputs", preflight)
    use = consumer.RegisteredExperimentUse.admit(
        target,
        expected_birth=born["birth"],
        expected_generation=born["generation"],
        now=lambda: 1200,
        _producer_request_paths=paths,
        _producer_config_path=installation[0],
    )
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=target, _registered_use=use
    )
    assert result["status"] == "blocked" and use._closed and not use.files.owned
    public = consumer.AUTHORITY_ROOT
    head = json.loads((public / "HEAD.json").read_bytes())
    current = json.loads((public / head["record_name"]).read_bytes())
    entry = current["enrollments"][0]
    assert entry["completion"] is not None and entry["state"] == "active"
    completion = installation[2] / (entry["intent_id"] + ".producer-completion.json")
    raw = completion.read_bytes()
    assert entry["completion"] == dict(
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    value = json.loads(raw)
    assert value["participant_profile"] == "g1_local_prelaunch_block.v1"
    assert (
        value["pair"]["sha256"]
        == "sha256:" + hashlib.sha256((target / (pair.SCHEMA + ".json")).read_bytes()).hexdigest()
    )
    assert value["lifetime_closed"] is True and value["child_execution_started"] is False


@pytest.mark.parametrize("tampered", [False, True])
def test_real_completed_prelaunch_offload_intent_requires_exact_closure_record(
    installation, tmp_path, monkeypatch, tampered  # noqa: F811
):  # noqa: F811
    from tests.test_registered_experiment_issuer import encoded
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    policy = json.loads(installation[3].read_bytes())
    policy["principals"][0]["allowed_actions"].append("offload")
    installation[3].write_bytes(encoded(policy))
    config, settings, _, _ = installation
    settings["experiment_retirement_enabled"] = True
    config.write_bytes(encoded(settings))
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", consumer.LANE_ROOTS)
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime

    monkeypatch.setattr(lifetime, "LANE_ROOTS", consumer.LANE_ROOTS)

    def preflight(_request):
        raise ValueError("development_only_prelaunch_refusal")

    monkeypatch.setattr(worker, "_preflight_inputs", preflight)
    use = consumer.RegisteredExperimentUse.admit(
        target, now=lambda: 1200, _producer_request_paths=paths
    )
    pair.run_g1_development_pair(request_paths=paths, output_dir=target, _registered_use=use)
    intent_id = use.entry["intent_id"]
    if tampered:
        record = installation[2] / (intent_id + ".producer-completion.json")
        record.write_bytes(record.read_bytes() + b" ")
        with pytest.raises(ValueError, match="experiment_completion_changed"):
            root.issue_experiment_action_intent(
                intent_id,
                principal="operator",
                owner="owner",
                action="offload",
                expires_at_epoch=3500,
                installed_config_path=config,
                now=lambda: 2900,
            )
    else:
        action = root.issue_experiment_action_intent(
            intent_id,
            principal="operator",
            owner="owner",
            action="offload",
            expires_at_epoch=3500,
            installed_config_path=config,
            now=lambda: 2900,
        )
        assert action["action_intent"]["size_bytes"] > 0
        assert (
            json.loads((installation[2] / (action["action_id"] + ".action.json")).read_bytes())[
                "completion"
            ]
            is not None
        )


def _contained_bootstrap(installation, tmp_path, monkeypatch):  # noqa: F811
    """Genuine root-issued intent/birth and its fixed protected public publication."""
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    consumer, target, born, paths = _registered_fixture(
        installation, tmp_path, monkeypatch, profile="g1_local_contained_completed.v1"  # noqa: F811
    )
    intent_id = target.name.removeprefix("registered-")
    intent = installation[2] / (intent_id + ".json")
    raw = intent.read_bytes()
    root.issue_experiment_producer_bootstrap(
        intent_id,
        expected_intent_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        expected_intent_size_bytes=len(raw),
        request_paths=paths,
        installed_config_path=installation[0],
        now=lambda: 1200,
    )
    selected = consumer.AUTHORITY_ROOT / (intent_id + ".producer-bootstrap.json")
    assert selected.is_file() and selected.stat().st_mode & 0o777 == 0o640
    return consumer, target, born, paths, selected


def test_actual_contained_public_bootstrap_admits_producer_without_private_records(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    monkeypatch.setattr(os, "geteuid", lambda: 1001)
    original_read = consumer._BirthFiles.read
    private = installation[2]

    def public_only(files, path, *args, **kwargs):
        assert not Path(path).is_relative_to(private), "ordinary producer opened private intent"
        assert Path(path) != installation[0], "ordinary producer opened private door config"
        return original_read(files, path, *args, **kwargs)

    monkeypatch.setattr(consumer._BirthFiles, "read", public_only)
    monkeypatch.setattr(
        root, "_configuration", lambda *a: pytest.fail("private config used by ordinary producer")
    )
    with consumer.RegisteredExperimentUse.admit(
        target,
        expected_birth=born["birth"],
        expected_generation=born["generation"],
        now=lambda: 1200,
        _producer_bootstrap_path=selected,
    ) as use:
        use.authorize_g1_pair(paths)
        assert len(use._producer_requests) == 2
        fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    assert use._closed and not use.files.owned and not use.files.unresolved


@pytest.mark.parametrize("change", ["request", "bootstrap", "wrong_path"])
def test_real_contained_bootstrap_cannot_refresh_or_replace_selected_producer_inputs(
    installation, tmp_path, monkeypatch, change  # noqa: F811
):  # noqa: F811
    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    if change == "wrong_path":
        supplied = selected.with_name("arbitrary-bootstrap.json")
        supplied.write_bytes(selected.read_bytes())
    else:
        supplied = selected
    use = None
    try:
        if change != "wrong_path":
            use = consumer.RegisteredExperimentUse.admit(
                target, now=lambda: 1200, _producer_bootstrap_path=supplied
            )
            (paths[0] if change == "request" else selected).write_bytes(b"{}")
            with pytest.raises(ValueError, match="experiment_"):
                use.authorize_g1_pair(paths)
        else:
            with pytest.raises(ValueError, match="experiment_"):
                consumer.RegisteredExperimentUse.admit(
                    target, now=lambda: 1200, _producer_bootstrap_path=supplied
                )
    finally:
        if use is not None:
            use.close()


@pytest.mark.parametrize("entrypoint", ["runtime", "supervisor"])
def test_registered_native_child_entrypoints_refuse_missing_use_before_payload(
    tmp_path, monkeypatch, entrypoint
):
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import native_g1_runtime_assembly as assembly
    from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor

    root = tmp_path / "lanes"
    target = root / "g1" / ("registered-" + "a" * 32)
    monkeypatch.setattr(consumer, "LANE_ROOTS", (root,))
    monkeypatch.setattr(
        supervisor,
        "preflight_g1_shared_scene_run",
        lambda **kw: pytest.fail("unadmitted child inputs read"),
    )
    with pytest.raises(ValueError, match="experiment_consumer_authority_required"):
        if entrypoint == "runtime":
            assembly.run_g1_supervised_built_scene_episode(
                built=None,
                candidate_id="candidate",
                preflight_inputs={},
                python_executable=tmp_path / "python",
                port=8443,
                device="cuda:0",
                max_steps=1,
                output_dir=target / "candidate/episode",
                to_tensor=lambda x: x,
                make_action_tensor=lambda x: x,
            )
        else:
            supervisor.start_g1_policy_server(
                preflight_inputs={},
                python_executable=tmp_path / "python",
                port=8443,
                device="cuda:0",
                log_path=target / "candidate/episode/server.log",
            )
    assert not target.exists()


def test_actual_contained_ordinary_pair_does_not_attempt_root_completion_publication(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime

    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", consumer.LANE_ROOTS)
    monkeypatch.setattr(lifetime, "LANE_ROOTS", consumer.LANE_ROOTS)
    monkeypatch.setattr(os, "geteuid", lambda: 1001)

    def blocked(_request):
        raise ValueError("free_fixture_preflight_refused")

    monkeypatch.setattr(worker, "_preflight_inputs", blocked)
    use = consumer.RegisteredExperimentUse.admit(
        target, expected_birth=born["birth"], now=lambda: 1200, _producer_bootstrap_path=selected
    )
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=target, _registered_use=use
    )
    assert result["status"] == "blocked" and use._closed and not use.files.owned
    assert not (installation[2] / (use.entry["intent_id"] + ".producer-completion.json")).exists()


def test_actual_worker_forwards_same_public_producer_use_into_runtime(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from types import SimpleNamespace
    from blueprint_pipeline import native_g1_development_worker as worker
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime

    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    monkeypatch.setattr(lifetime, "LANE_ROOTS", consumer.LANE_ROOTS)
    request = json.loads(paths[0].read_bytes())
    plan = json.loads(
        (Path(request["bundle_root"]) / "native_task_arena_scene_plan.v1.json").read_bytes()
    )
    monkeypatch.setattr(
        worker, "_verify_packet", lambda p: {"arena_scene_plan_digest": plan["plan_digest"]}
    )
    monkeypatch.setattr(
        worker,
        "preflight_g1_shared_scene_run",
        lambda **kw: {
            "status": "staged_inputs_verified",
            "scene_plan_digest": plan["plan_digest"],
            "candidate_id": request["candidate_id"],
            "inventory_file_sha256": request["rights_review"]["inventory_file_sha256"],
            "robot_id": "unitree_g1",
            "policy_role": "manipulation",
        },
    )
    closed = []
    built = SimpleNamespace(
        plan=plan, env=SimpleNamespace(close=lambda: closed.append("environment"))
    )
    monkeypatch.setattr(
        worker,
        "_launch_scene",
        lambda **kw: (SimpleNamespace(close=lambda: closed.append("simulator")), {}),
    )
    monkeypatch.setattr(worker, "_build_scene", lambda **kw: (built, {}))
    seen = []
    with consumer.RegisteredExperimentUse.admit(
        target, expected_birth=born["birth"], now=lambda: 1200, _producer_bootstrap_path=selected
    ) as use:

        def runtime(**kw):
            assert kw["_registered_use"] is use
            use.check()
            seen.append(True)
            raise ValueError("free_fixture_runtime_stop")

        monkeypatch.setattr(worker, "run_g1_supervised_built_scene_episode", runtime)
        receipt = worker.run_g1_development_worker(
            request=request,
            output_dir=target / request["candidate_id"],
            scratch_lifetime=use,
            _registered_use=use,
        )
        assert receipt["status"] == "blocked" and receipt["phase_reached"] == "episode"
        assert receipt["blocker"]["message"] == "free_fixture_runtime_stop"
        assert seen == [True] and closed == ["environment", "simulator"]


def test_actual_runtime_forwards_same_public_use_into_native_supervisor(
    installation, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    from types import SimpleNamespace
    from blueprint_pipeline import native_g1_runtime_assembly as assembly

    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    request = json.loads(paths[0].read_bytes())
    plan = json.loads(
        (Path(request["bundle_root"]) / "native_task_arena_scene_plan.v1.json").read_bytes()
    )
    seen = []
    with consumer.RegisteredExperimentUse.admit(
        target, expected_birth=born["birth"], now=lambda: 1200, _producer_bootstrap_path=selected
    ) as use:

        def server(**kw):
            assert kw["_registered_use"] is use
            use.check()
            seen.append(True)
            raise ValueError("free_fixture_native_child_stop")

        monkeypatch.setattr(assembly, "start_g1_policy_server", server)
        with pytest.raises(ValueError, match="free_fixture_native_child_stop"):
            assembly.run_g1_supervised_built_scene_episode(
                built=SimpleNamespace(plan=plan),
                candidate_id=request["candidate_id"],
                preflight_inputs={"candidate_id": request["candidate_id"]},
                python_executable=Path(request["python_executable"]),
                port=8443,
                device="cuda:0",
                max_steps=1,
                output_dir=target / request["candidate_id"] / "episode",
                to_tensor=lambda x: x,
                make_action_tensor=lambda x: x,
                _registered_use=use,
            )
        assert seen == [True]


@pytest.mark.parametrize("fault", ["expired", "changed_intent", "disabled"])
def test_actual_root_contained_runner_refuses_before_any_unit_launch(
    installation, tmp_path, monkeypatch, fault  # noqa: F811
):  # noqa: F811
    from blueprint_pipeline import native_g1_registered_containment as contained

    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    intent_id = target.name.removeprefix("registered-")
    raw = (installation[2] / (intent_id + ".json")).read_bytes()
    expected = {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    if fault == "changed_intent":
        expected["sha256"] = "sha256:" + "f" * 64
    elif fault == "disabled":
        value = json.loads(installation[0].read_bytes())
        value["experiment_creation_enabled"] = False
        installation[0].write_bytes(json.dumps(value).encode())
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("refused authority launched unit")
    )
    with pytest.raises(ValueError, match="experiment_|owner_"):
        contained.run_registered_experiment(
            intent_id,
            expected_intent=expected,
            installed_config_path=installation[0],
            now=lambda: 4000 if fault == "expired" else 1200,
        )


def test_actual_root_unit_arguments_are_fixed_to_native_ordinary_uid_scope():
    from blueprint_pipeline import native_g1_registered_containment as contained

    intent_id = "a" * 32
    target = Path("/mnt/blueprint-work/lanes/g1") / ("registered-" + intent_id)
    bootstrap = Path("/var/lib/blueprint-operator-door/experiment-authority") / (
        intent_id + ".producer-bootstrap.json"
    )
    command = contained._unit_arguments(intent_id, target, bootstrap)
    assert command[:3] == ["/usr/bin/systemd-run", "--no-block", "--quiet"]
    assert "--unit=blueprint-experiment-" + intent_id in command
    for prop in (
        "User=blueprint",
        "Group=blueprint",
        "UMask=0077",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "ProtectControlGroups=yes",
        "KillMode=control-group",
        "Delegate=no",
        "TasksMax=64",
        "TimeoutStopSec=30",
    ):
        assert "--property=" + prop in command
    assert not any(arg.startswith("--collect") for arg in command)
    assert command[-6:] == [
        "-m",
        contained.__name__,
        "--producer-bootstrap",
        str(bootstrap),
        "--target",
        str(target),
    ]


@pytest.mark.parametrize("change", ["log_mode", "parent_name", "foreign_descriptor"])
def test_actual_child_log_refusal_still_finalizes_every_other_known_original(
    installation, tmp_path, monkeypatch, change  # noqa: F811
):  # noqa: F811
    from types import SimpleNamespace
    from blueprint_pipeline import native_g1_registered_containment as contained

    consumer, target, born, paths, selected = _contained_bootstrap(
        installation, tmp_path, monkeypatch  # noqa: F811
    )
    directory = target / "child-log"
    directory.mkdir(mode=0o700)
    log_path = directory / "g1_policy_server.log"
    log_path.write_bytes(b"tiny child output")
    log_path.chmod(0o600)
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    log = os.open(log_path, os.O_WRONLY)
    parent_proof = contained._typed(os.fstat(parent))
    log_proof = contained._typed(os.fstat(log))
    foreign = None
    use = consumer.RegisteredExperimentUse.admit(
        target, now=lambda: 1200, _producer_bootstrap_path=selected
    )
    lifetime = contained._ChildLifetime(
        use,
        process=SimpleNamespace(pid=321, poll=lambda: 0),
        parent=parent,
        log=log,
        parent_proof=parent_proof,
        log_proof=log_proof,
        parent_path=directory,
        log_name=log_path.name,
        handshake={},
    )
    try:
        if change == "log_mode":
            log_path.chmod(0o644)
        elif change == "parent_name":
            directory.rename(target / "moved-child-log")
        else:
            other = target / "foreign-owned.txt"
            other.write_bytes(b"FOREIGN")
            foreign = os.open(other, os.O_RDWR)
            os.dup2(foreign, log)
        with pytest.raises(ValueError, match="experiment_child_"):
            lifetime.finish()
        assert lifetime.closed is False
        with pytest.raises(OSError):
            os.fstat(parent)
        if change == "foreign_descriptor":
            assert os.pread(log, 7, 0) == b"FOREIGN"
        else:
            with pytest.raises(OSError):
                os.fstat(log)
    finally:
        use.close()
        for fd in (parent, log, foreign):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def test_native_unit_observer_requests_complete_empty_systemd_properties(monkeypatch):
    """Actual systemctl omits empty properties unless show receives --all."""
    from blueprint_pipeline import native_g1_registered_containment as contained
    calls = []
    empty = {'AmbientCapabilities', 'CapabilityBoundingSet'}
    def show(arguments):
        calls.append(arguments)
        return ''.join(key + '=' + ('' if key in empty else 'observed') + '\n'
                       for key in contained._UNIT_PROPERTIES if '--all' in arguments or key not in empty)
    monkeypatch.setattr(contained, '_native_control', show)
    observed = contained._show_unit('a' * 32)
    assert set(observed) == set(contained._UNIT_PROPERTIES)
    assert observed['AmbientCapabilities'] == observed['CapabilityBoundingSet'] == ''
    assert calls == [[contained._SYSTEMCTL, 'show', 'blueprint-experiment-' + 'a' * 32 + '.service',
                      '--no-pager', '--all', '--property=' + ','.join(contained._UNIT_PROPERTIES)]]


# Exact NONSECRET systemctl show bytes from Ubuntu24 native job109119465042.
_ABSENT_UNIT_SHOW = 'TimeoutStopUSec=1min 30s\nRemainAfterExit=no\nMainPID=0\nResult=success\nExecMainStatus=0\nControlGroup=\nDelegate=no\nTasksMax=19151\nUMask=0022\nCapabilityBoundingSet=cap_chown cap_dac_override cap_dac_read_search cap_fowner cap_fsetid cap_kill cap_setgid cap_setuid cap_setpcap cap_linux_immutable cap_net_bind_service cap_net_broadcast cap_net_admin cap_net_raw cap_ipc_lock cap_ipc_owner cap_sys_module cap_sys_rawio cap_sys_chroot cap_sys_ptrace cap_sys_pacct cap_sys_admin cap_sys_boot cap_sys_nice cap_sys_resource cap_sys_time cap_sys_tty_config cap_mknod cap_lease cap_audit_write cap_audit_control cap_setfcap cap_mac_override cap_mac_admin cap_syslog cap_wake_alarm cap_block_suspend cap_audit_read cap_perfmon cap_bpf cap_checkpoint_restore\nAmbientCapabilities=\nUser=\nGroup=\nReadWritePaths=\nReadOnlyPaths=\nPrivateTmp=no\nProtectControlGroups=no\nPrivateNetwork=no\nProtectHome=no\nProtectSystem=no\nNoNewPrivileges=no\nRestrictSUIDSGID=no\nRestrictNamespaces=no\nKillMode=control-group\nId=blueprint-experiment-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.service\nLoadState=not-found\nActiveState=inactive\nSubState=dead\nInvocationID=\n'

def test_actual_systemd_absent_exec_array_is_typed_no_command(monkeypatch):
    from blueprint_pipeline import native_g1_registered_containment as contained
    monkeypatch.setattr(contained, '_native_control', lambda arguments: _ABSENT_UNIT_SHOW)
    value = contained._show_unit('a' * 32)
    assert set(value) == set(contained._UNIT_PROPERTIES)
    assert value['ExecStart'] == '' and value['LoadState'] == 'not-found'


@pytest.mark.parametrize('change', ['loaded','active','substate','pid','invocation','group',
                                  'wrong_id','missing_other','duplicate','result','status'])
def test_missing_native_exec_never_repairs_loaded_or_ambiguous_unit(monkeypatch, change):
    from blueprint_pipeline import native_g1_registered_containment as contained
    lines = _ABSENT_UNIT_SHOW.splitlines()
    replacements = {'loaded': ('LoadState','loaded'), 'active': ('ActiveState','active'),
        'substate': ('SubState','running'), 'pid': ('MainPID','12'),
        'invocation': ('InvocationID','b'*32), 'group': ('ControlGroup','/system.slice/other.service'),
        'wrong_id': ('Id','blueprint-experiment-'+ 'b'*32+'.service'),
        'result': ('Result','failed'), 'status': ('ExecMainStatus','1')}
    if change == 'missing_other':
        lines = [line for line in lines if not line.startswith('User=')]
    elif change == 'duplicate':
        lines.append('Id=blueprint-experiment-'+ 'a'*32+'.service')
    else:
        key,value = replacements[change]
        lines = [key+'='+value if line.startswith(key+'=') else line for line in lines]
    monkeypatch.setattr(contained, '_native_control', lambda arguments: '\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='experiment_unit_observation_failed'):
        contained._show_unit('a'*32)


def _successful_unit_transition():
    from blueprint_pipeline import native_g1_registered_containment as contained
    intent = 'a' * 32
    target = Path('/mnt/blueprint-work/lanes/g1/registered-' + intent)
    bootstrap = Path('/var/lib/blueprint/authority/' + intent + '.producer-bootstrap.json')
    command = contained._unit_arguments(intent, target, bootstrap)
    argv = command[command.index('--') + 1:]
    started = dict(Id='blueprint-experiment-' + intent + '.service', LoadState='loaded',
        ActiveState='active', SubState='running', InvocationID='b' * 32, MainPID='42',
        Result='success', ExecMainStatus='0', ControlGroup='/system.slice/blueprint-experiment-' + intent + '.service',
        ExecStart='{ path=' + argv[0] + ' ; argv[]=' + ' '.join(argv) + ' ; ignore_errors=no ; start_time=[now] ; stop_time=[n/a] ; pid=42 ; code=(null) ; status=0/0 }',
        User='blueprint',Group='blueprint',UMask='0077',NoNewPrivileges='yes',CapabilityBoundingSet='',
        AmbientCapabilities='',ProtectControlGroups='yes',KillMode='control-group',Delegate='no',TasksMax='64',
        TimeoutStopUSec='30s',RemainAfterExit='yes',ProtectSystem='strict',ProtectHome='yes',PrivateTmp='yes',
        PrivateNetwork='yes',RestrictNamespaces='yes',RestrictSUIDSGID='yes',ReadWritePaths=str(target),ReadOnlyPaths='/')
    return intent, command, started


def test_successful_same_invocation_terminal_empty_controlgroup_requires_original_start():
    from blueprint_pipeline import native_g1_registered_containment as contained
    intent, command, started = _successful_unit_transition()
    contained._check_unit(started, command, intent)
    terminal = started | {'ActiveState': 'active', 'SubState': 'exited', 'MainPID': '0', 'ControlGroup': ''}
    contained._check_unit(terminal, command, intent, started=started)
    with pytest.raises(ValueError, match='experiment_unit_identity_changed'):
        contained._check_unit(terminal, command, intent)


@pytest.mark.parametrize('change', ['running-empty', 'different-invocation', 'different-group', 'failed', 'live-pid'])
def test_terminal_cgroup_transition_never_accepts_changed_or_live_unit(change):
    from blueprint_pipeline import native_g1_registered_containment as contained
    intent, command, started = _successful_unit_transition()
    terminal = started | {'ActiveState': 'active', 'SubState': 'exited', 'MainPID': '0', 'ControlGroup': ''}
    if change == 'running-empty':
        terminal['SubState'] = 'running'
    elif change == 'different-invocation':
        terminal['InvocationID'] = 'c' * 32
    elif change == 'different-group':
        terminal['ControlGroup'] = '/system.slice/foreign.service'
    elif change == 'failed':
        terminal['Result'] = 'exit-code'
    else:
        terminal['MainPID'] = '43'
    with pytest.raises(ValueError, match='experiment_unit_'):
        contained._check_unit(terminal, command, intent, started=started)


def test_collected_after_exact_successful_stop_preserves_raw_absent_snapshot():
    from blueprint_pipeline import native_g1_registered_containment as contained
    intent, command, started = _successful_unit_transition()
    stopping = started | {'ActiveState': 'active', 'SubState': 'exited', 'MainPID': '0', 'ControlGroup': ''}
    finished = dict(line.split('=', 1) for line in _ABSENT_UNIT_SHOW.splitlines()) | {'ExecStart': ''}
    kernel = {'tasks': 0, 'groups': 0, 'absent_after_started': True}
    stop_command = [contained._SYSTEMCTL, 'stop', started['Id']]
    contained._check_finished_unit(finished, command, intent, started=started, stopping=stopping,
                                   kernel=kernel, stop_command=stop_command)
    assert finished['InvocationID'] == '' and finished['LoadState'] == 'not-found'


@pytest.mark.parametrize('change', ['no-kernel-closure', 'foreign-stop', 'foreign-invocation', 'active-after-stop'])
def test_poststop_absence_never_clears_unproved_or_foreign_lifetime(change):
    from blueprint_pipeline import native_g1_registered_containment as contained
    intent, command, started = _successful_unit_transition()
    stopping = started | {'ActiveState': 'active', 'SubState': 'exited', 'MainPID': '0', 'ControlGroup': ''}
    finished = dict(line.split('=', 1) for line in _ABSENT_UNIT_SHOW.splitlines()) | {'ExecStart': ''}
    kernel = {'tasks': 0, 'groups': 0, 'absent_after_started': True}
    stop_command = [contained._SYSTEMCTL, 'stop', started['Id']]
    if change == 'no-kernel-closure':
        kernel['tasks'] = 1
    elif change == 'foreign-stop':
        stop_command[-1] = 'foreign.service'
    elif change == 'foreign-invocation':
        stopping['InvocationID'] = 'c' * 32
    else:
        finished['ActiveState'] = 'active'
    with pytest.raises(ValueError, match='experiment_unit_'):
        contained._check_finished_unit(finished, command, intent, started=started, stopping=stopping,
                                       kernel=kernel, stop_command=stop_command)
