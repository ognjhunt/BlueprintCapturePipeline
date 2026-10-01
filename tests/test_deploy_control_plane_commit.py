"""A deploy must not move the ground under a running paid attempt.

Activating the release symlink swaps the tree a running allocator was started
from, while that allocator is holding a rented GPU. That happened on
2026-08-13: a deploy repointed the link 20 minutes into another lane's paid
Content Agents run, which had passed admission under the previous commit and
was mid-heartbeat on a live instance.

It did no visible harm -- the process had already imported its modules -- but
"probably fine" is not a property worth relying on with an instance billing by
the second, and a lane that reads any file from that path afterwards reads bytes
from a commit it was never admitted under.

The lock already existed. `vast_provider_adapter` writes it before the launch
API call and records the holding pid; the deploy just never looked.
"""

from __future__ import annotations

import contextlib
import errno
import importlib.util
import io
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import active_deployed_release_admission as admission
from blueprint_pipeline import control_plane_break_glass as break_glass

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "deploy_control_plane_commit", REPO_ROOT / "scripts" / "deploy_control_plane_commit.py"
)
deploy = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(deploy)


def test_terminal_controls_are_prepared_as_service_user_before_workers_resume(monkeypatch, tmp_path):
    calls = []
    commit = 'a'*40
    unit = tmp_path/'deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service'
    unit.parent.mkdir(parents=True)
    unit.write_bytes((REPO_ROOT/'deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service').read_bytes())
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ['systemctl', 'show']:
            return SimpleNamespace(stdout='LoadState=loaded\nActiveState=failed\nMainPID=0\n')
        if argv[:2] == ['systemctl', 'reset-failed']:
            return SimpleNamespace(stdout='')
        assert argv[0] == 'systemd-run' and '--wait' in argv
        assert '--property=User=blueprint' in argv
        assert sum(arg.startswith('--setenv=BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_') for arg in argv) == 6
        assert f'PYTHONPATH={tmp_path / "src"}' in argv
        assert 'blueprint_pipeline.task_evaluation_terminal_controls_deploy' in argv
        assert kwargs['timeout'] >= 1800  # retained validation exceeded ten minutes on the host
        return SimpleNamespace(stdout=json.dumps({'status':'prepared','source_commit':commit,
            'provider_mutation_performed':False,'model_called':False,'placement_materialized':False}))
    monkeypatch.setattr(deploy.subprocess, 'run', run)
    result = deploy._prepare_terminal_controls_adoptions(release_path=tmp_path, commit=commit, config_path=tmp_path/'config.json')
    assert result['status'] == 'prepared'
    assert sum(argv[:2] == ['systemctl','reset-failed'] for argv in calls) == 3


def test_terminal_controls_child_failure_reports_typed_code_without_stderr(monkeypatch, tmp_path):
    unit = tmp_path/'deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service'
    unit.parent.mkdir(parents=True)
    unit.write_bytes((REPO_ROOT/'deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service').read_bytes())

    def run(argv, **_kwargs):
        if argv[:2] == ['systemctl', 'show']:
            return SimpleNamespace(stdout='LoadState=loaded\nActiveState=inactive\nMainPID=0\n')
        raise subprocess.CalledProcessError(
            1, argv, output='',
            stderr='secret-token-value\nConfiguredControlsProvisioningError: configured_controls_provisioning_registry_conflict:permission denied\n',
        )

    monkeypatch.setattr(deploy.subprocess, 'run', run)
    with pytest.raises(deploy.ControlPlaneDeployError) as error:
        deploy._prepare_terminal_controls_adoptions(
            release_path=tmp_path, commit='a'*40, config_path=tmp_path/'config.json'
        )
    assert 'configured_controls_provisioning_registry_conflict' in str(error.value)
    assert 'secret-token-value' not in str(error.value)
    assert 'stderr_sha256_' in str(error.value)


def test_terminal_controls_deploy_refuses_missing_scoped_artifact_store(tmp_path):
    unit = tmp_path/'deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service'
    unit.parent.mkdir(parents=True)
    unit.write_text('[Service]\nEnvironment=BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE=/tmp/legacy\n')
    with pytest.raises(deploy.ControlPlaneDeployError, match='artifact_store_env_missing'):
        deploy._terminal_controls_artifact_store_env(tmp_path)


@pytest.mark.parametrize('active,pid', [('active','0'), ('inactive','12'), ('activating','0')])
def test_terminal_controls_deploy_refuses_a_worker_race(monkeypatch, tmp_path, active, pid):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=f'LoadState=loaded\nActiveState={active}\nMainPID={pid}\n')
    monkeypatch.setattr(deploy.subprocess, 'run', run)
    with pytest.raises(deploy.ControlPlaneDeployError, match='worker_not_quiescent'):
        deploy._prepare_terminal_controls_adoptions(release_path=tmp_path, commit='a'*40, config_path=tmp_path/'config.json')
    assert len(calls) == 1


@pytest.mark.parametrize("finishes", [True, False])
def test_terminal_controls_drain_is_bounded_and_never_kills_worker(monkeypatch, finishes):
    elapsed = [0.0]
    calls = []
    monkeypatch.setattr(deploy.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(deploy.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] == ["systemctl", "show"]
        busy = argv[2].endswith(".service") and not (finishes and elapsed[0] >= 2)
        return SimpleNamespace(stdout="LoadState=loaded\nActiveState=" +
            ("activating\nMainPID=123\n" if busy else "inactive\nMainPID=0\n"))

    monkeypatch.setattr(deploy.subprocess, "run", run)
    if finishes:
        deploy._require_terminal_controls_quiescence(wait_seconds=4)
        assert elapsed[0] == 2
    else:
        with pytest.raises(deploy.ControlPlaneDeployError, match="worker_not_quiescent"):
            deploy._require_terminal_controls_quiescence(wait_seconds=4)
        assert elapsed[0] == 4
    assert calls[0][2].endswith(".path") and calls[1][2].endswith(".timer")


def test_terminal_controls_drain_refuses_armed_trigger_without_waiting(monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **kw:
        SimpleNamespace(stdout="LoadState=loaded\nActiveState=active\nMainPID=0\n"))
    monkeypatch.setattr(deploy.time, "sleep", lambda _: pytest.fail("must not wait with trigger armed"))
    with pytest.raises(deploy.ControlPlaneDeployError, match="worker_not_quiescent"):
        deploy._require_terminal_controls_quiescence(wait_seconds=600)


def test_retention_reader_repair_preserves_owner_and_private_file_modes(tmp_path):
    root = tmp_path / "release-retention"
    root.mkdir(mode=0o700)
    private = root / "plan.json"
    private.write_text("{}")
    private.chmod(0o600)
    owner = root.stat().st_uid
    receipt = deploy._install_retention_plan_reader_access(root, owner_gid=os.getgid())
    assert receipt["file_permissions_changed"] is False
    assert root.stat().st_uid == owner and root.stat().st_mode & 0o777 == 0o750
    assert private.stat().st_mode & 0o777 == 0o600
    linked = tmp_path / "linked"
    linked.symlink_to(root)
    with pytest.raises(deploy.ControlPlaneDeployError, match="root_unsafe"):
        deploy._install_retention_plan_reader_access(linked, owner_gid=os.getgid())


def test_script_bootstraps_repo_src_for_the_bare_host_interpreter() -> None:
    """The production host runs this script with python3, not an installed CLI."""

    probe = "\n".join(
        (
            "import importlib.util",
            "import json",
            "import sys",
            "from pathlib import Path",
            f"repo_root = Path({str(REPO_ROOT)!r})",
            "spec = importlib.util.spec_from_file_location(",
            "    'deploy_control_plane_commit_probe',",
            "    repo_root / 'scripts' / 'deploy_control_plane_commit.py',",
            ")",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "print(json.dumps(sys.path))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    isolated_path = json.loads(completed.stdout)

    assert isolated_path.index(str(REPO_ROOT / "src")) < isolated_path.index(
        str(REPO_ROOT / "scripts")
    )


def _lock(tmp_path: Path, **overrides) -> Path:
    record = {
        "acquired_at": "2026-08-13T12:49:36.475276+00:00",
        "job_dir": "/var/lib/blueprint/.../vast_provider_run",
        "pid": os.getpid(),
        "purpose": "vast_paid_instance_launch_single_flight_guard",
    }
    record.update(overrides)
    path = tmp_path / "vast_paid_launch.lock"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _provenance(tmp_path: Path, commit: str) -> Path:
    path = tmp_path / f"provenance-{commit[:8]}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "blueprint.deploy_release_provenance.v1",
                "status": "verified",
                "git_sha": commit,
                "run_id": 123,
                "run_url": "https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/123",
                "workflow_name": "Full Test Lane",
                "workflow_path": ".github/workflows/full-test-lane.yml",
                "job_name": "Full pytest lane on CPU runner",
                "collection": {"test_count": 100},
                "claim_boundary": {"canonical_full_lane_verified": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def _iteration_provenance(commit: str) -> tuple[bytes, dict[str, object]]:
    receipt: dict[str, object] = {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "iteration",
        "git_sha": commit,
        "promotion_eligible": False,
        "claim_boundary": {
            "canonical_full_lane_verified": False,
            "promotion_eligible": False,
            "evidence_grade": "development_only",
        },
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    return payload, receipt


def _verified_provenance(commit: str) -> tuple[bytes, dict[str, object]]:
    receipt: dict[str, object] = {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "verified",
        "git_sha": commit,
        "promotion_eligible": True,
        "run_id": 123,
        "claim_boundary": {"canonical_full_lane_verified": True},
    }
    return json.dumps(receipt).encode(), receipt


@pytest.mark.parametrize("status", ["iteration", "canary"])
def test_verified_provenance_supersedes_same_commit_iteration_once(
    tmp_path: Path, status: str,
) -> None:
    commit = "a" * 40
    state_root = tmp_path / "state"
    iteration_payload, iteration_receipt = _iteration_provenance(commit)
    iteration_receipt["status"] = status
    iteration_payload = json.dumps(iteration_receipt).encode()
    verified_payload, verified_receipt = _verified_provenance(commit)

    deploy._install_release_provenance(
        payload=iteration_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=iteration_receipt,
    )
    installed = deploy._install_release_provenance(
        payload=verified_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=verified_receipt,
    )

    canonical = state_root / commit / deploy.DEPLOY_RELEASE_PROVENANCE_NAME
    superseded = (
        state_root / commit / deploy.SUPERSEDED_ITERATION_PROVENANCE_NAME
    )
    assert canonical.read_bytes() == verified_payload
    assert superseded.read_bytes() == iteration_payload
    assert canonical.stat().st_mode & 0o777 == 0o440
    assert superseded.stat().st_mode & 0o777 == 0o440
    assert installed["superseded_iteration_provenance"]["path"] == str(
        superseded
    )
    assert installed["superseded_iteration_provenance"]["status"] == status

    # A repeated promotion is idempotent and does not rewrite history.
    repeated = deploy._install_release_provenance(
        payload=verified_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=verified_receipt,
    )
    assert canonical.read_bytes() == verified_payload
    assert superseded.read_bytes() == iteration_payload
    assert "superseded_iteration_provenance" not in repeated


def test_merged_canary_provenance_upgrades_to_iteration_once(
    tmp_path: Path,
) -> None:
    commit = "e" * 40
    state_root = tmp_path / "state"
    canary_payload, canary_receipt = _iteration_provenance(commit)
    canary_receipt["status"] = "canary"
    canary_payload = json.dumps(canary_receipt, sort_keys=True).encode()
    iteration_payload, iteration_receipt = _iteration_provenance(commit)
    deploy._install_release_provenance(
        payload=canary_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=canary_receipt,
    )

    installed = deploy._install_release_provenance(
        payload=iteration_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=iteration_receipt,
    )
    canonical = state_root / commit / deploy.DEPLOY_RELEASE_PROVENANCE_NAME
    superseded = state_root / commit / deploy.SUPERSEDED_ITERATION_PROVENANCE_NAME
    assert canonical.read_bytes() == iteration_payload
    assert superseded.read_bytes() == canary_payload
    assert installed["superseded_iteration_provenance"]["status"] == "canary"
    assert canonical.stat().st_mode & 0o777 == 0o440
    assert superseded.stat().st_mode & 0o777 == 0o440

    repeated = deploy._install_release_provenance(
        payload=iteration_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=iteration_receipt,
    )
    assert "superseded_iteration_provenance" not in repeated
    assert superseded.read_bytes() == canary_payload


def test_canary_provenance_cannot_upgrade_to_changed_development_claim(
    tmp_path: Path,
) -> None:
    commit = "f" * 40
    state_root = tmp_path / "state"
    canary_payload, canary_receipt = _iteration_provenance(commit)
    canary_receipt["status"] = "canary"
    canary_payload = json.dumps(canary_receipt, sort_keys=True).encode()
    deploy._install_release_provenance(
        payload=canary_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=canary_receipt,
    )
    iteration_payload, iteration_receipt = _iteration_provenance(commit)
    iteration_receipt["claim_boundary"]["evidence_grade"] = "production_promoted"
    iteration_payload = json.dumps(iteration_receipt).encode()
    with pytest.raises(
        deploy.ControlPlaneDeployError, match="deploy_release_provenance_conflict"
    ):
        deploy._install_release_provenance(
            payload=iteration_payload,
            state_root=state_root,
            source_commit=commit,
            receipt=iteration_receipt,
        )
    canonical = state_root / commit / deploy.DEPLOY_RELEASE_PROVENANCE_NAME
    assert canonical.read_bytes() == canary_payload
    assert not (state_root / commit / deploy.SUPERSEDED_ITERATION_PROVENANCE_NAME).exists()


def test_release_provenance_never_downgrades_verified_to_iteration(
    tmp_path: Path,
) -> None:
    commit = "b" * 40
    state_root = tmp_path / "state"
    iteration_payload, iteration_receipt = _iteration_provenance(commit)
    verified_payload, verified_receipt = _verified_provenance(commit)
    deploy._install_release_provenance(
        payload=verified_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=verified_receipt,
    )

    with pytest.raises(
        deploy.ControlPlaneDeployError, match="deploy_release_provenance_conflict"
    ):
        deploy._install_release_provenance(
            payload=iteration_payload,
            state_root=state_root,
            source_commit=commit,
            receipt=iteration_receipt,
        )

    canonical = state_root / commit / deploy.DEPLOY_RELEASE_PROVENANCE_NAME
    assert canonical.read_bytes() == verified_payload
    assert not (
        state_root / commit / deploy.SUPERSEDED_ITERATION_PROVENANCE_NAME
    ).exists()


def test_release_provenance_does_not_upgrade_an_iteration_from_another_commit(
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    state_root = tmp_path / "state"
    other_payload, _ = _iteration_provenance("d" * 40)
    verified_payload, verified_receipt = _verified_provenance(commit)
    destination = state_root / commit / deploy.DEPLOY_RELEASE_PROVENANCE_NAME
    destination.parent.mkdir(parents=True)
    destination.write_bytes(other_payload)

    with pytest.raises(
        deploy.ControlPlaneDeployError, match="deploy_release_provenance_conflict"
    ):
        deploy._install_release_provenance(
            payload=verified_payload,
            state_root=state_root,
            source_commit=commit,
            receipt=verified_receipt,
        )

    assert destination.read_bytes() == other_payload


def test_the_canonical_lock_is_checked_by_default() -> None:
    """An operator who forgets the flag still gets the guard."""

    assert deploy.DEFAULT_PAID_LAUNCH_LOCKS == (
        "/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
    )


def test_the_real_intake_restart_cannot_be_omitted() -> None:
    assert deploy._required_restart_units(()) == (
        "blueprint-pipeline-intake.service",
    )
    assert deploy._required_restart_units(("another.service",)) == (
        "blueprint-pipeline-intake.service",
        "another.service",
    )


def test_restart_reloads_drop_ins_before_restarting_the_intake(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        stdout = "active\n" if argv[:2] == ["systemctl", "is-active"] else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    restarted = deploy._restart_units(("blueprint-pipeline-intake.service",))

    assert calls == [
        ("systemctl", "daemon-reload"),
        ("systemctl", "restart", "blueprint-pipeline-intake.service"),
        ("systemctl", "is-active", "blueprint-pipeline-intake.service"),
    ]
    assert restarted == [
        {"unit": "blueprint-pipeline-intake.service", "state": "active"}
    ]


def test_deploy_installs_exact_queue_unit_bytes_atomically(tmp_path: Path) -> None:
    release = tmp_path / "release"
    unit_dir = release / "deploy/systemd"
    unit_dir.mkdir(parents=True)
    service = unit_dir / "blueprint-task-evaluation-launch-dispatcher.service"
    service.write_text("[Service]\nKillMode=process\n", encoding="utf-8")
    path_unit = unit_dir / "blueprint-task-evaluation-launch-dispatcher.path"
    path_unit.write_text(
        "[Path]\nPathChanged=/queue/pending\n"
        "PathExistsGlob=/queue/pending/*.json\n",
        encoding="utf-8",
    )
    preparation_service = (
        unit_dir / "blueprint-task-evaluation-launch-preparation.service"
    )
    preparation_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-prepare\n", encoding="utf-8"
    )
    preparation_path = unit_dir / "blueprint-task-evaluation-launch-preparation.path"
    preparation_path.write_text(
        "[Path]\nPathChanged=/preparations/pending\n"
        "PathExistsGlob=/preparations/pending/*.json\n",
        encoding="utf-8",
    )
    preparation_timer = unit_dir / "blueprint-task-evaluation-launch-preparation.timer"
    preparation_timer.write_text("[Timer]\nOnUnitInactiveSec=2min\n", encoding="utf-8")
    sam31_service = unit_dir / "blueprint-task-evaluation-sam31-preparation-execution.service"
    sam31_service.write_text("[Service]\nExecStart=/usr/bin/blueprint-sam31-phase\n", encoding="utf-8")
    sam31_path = unit_dir / "blueprint-task-evaluation-sam31-preparation-execution.path"
    sam31_path.write_text("[Path]\nPathExistsGlob=/sam31/pending/*.json\n", encoding="utf-8")
    sam31_timer = unit_dir / "blueprint-task-evaluation-sam31-preparation-execution.timer"
    sam31_timer.write_text("[Timer]\nOnUnitInactiveSec=30s\n", encoding="utf-8")
    compilation_service = (
        unit_dir / "blueprint-task-evaluation-episode-compilation.service"
    )
    compilation_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-compile-episode\n",
        encoding="utf-8",
    )
    compilation_path = (
        unit_dir / "blueprint-task-evaluation-episode-compilation.path"
    )
    compilation_path.write_text(
        "[Path]\nPathChanged=/episode-compilations/pending\n"
        "PathExistsGlob=/episode-compilations/pending/*.json\n",
        encoding="utf-8",
    )
    compilation_timer = unit_dir / "blueprint-task-evaluation-episode-compilation.timer"
    compilation_timer.write_text("[Timer]\nOnUnitInactiveSec=5min\n", encoding="utf-8")
    remote_units = []
    for suffix, body in (
        (".service", "[Service]\nExecStart=/usr/bin/blueprint-collect-remote-episodes\n"),
        (".timer", "[Timer]\nOnUnitInactiveSec=60s\n"),
        (".path", "[Path]\nPathChanged=/remote-cpu-jobs/handoffs/episode_compilation\n"),
    ):
        remote_unit = unit_dir / f"blueprint-task-evaluation-episode-compilation-remote{suffix}"
        remote_unit.write_text(body, encoding="utf-8")
        remote_units.append(remote_unit)
    activation_service = (
        unit_dir / "blueprint-task-evaluation-launch-activation.service"
    )
    activation_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-activate\n", encoding="utf-8"
    )
    activation_path = unit_dir / "blueprint-task-evaluation-launch-activation.path"
    activation_path.write_text(
        "[Path]\nPathChanged=/activations/pending\n"
        "PathExistsGlob=/activations/pending/*.json\n",
        encoding="utf-8",
    )
    canary_service = (
        unit_dir / "blueprint-task-evaluation-policy-canary-dispatcher.service"
    )
    canary_service.write_text(
        "[Service]\nKillMode=process\nExecStart=/usr/bin/blueprint-policy-canary\n",
        encoding="utf-8",
    )
    canary_path = (
        unit_dir / "blueprint-task-evaluation-policy-canary-dispatcher.path"
    )
    canary_path.write_text(
        "[Path]\nPathChanged=/policy-canaries/pending\n"
        "PathExistsGlob=/policy-canaries/pending/*.json\n",
        encoding="utf-8",
    )
    g1_service = unit_dir / "blueprint-native-g1-team-campaign-dispatcher.service"
    g1_service.write_text(
        "[Service]\nKillMode=process\nExecStart=/usr/bin/blueprint-g1-team\n",
        encoding="utf-8",
    )
    g1_timer = unit_dir / "blueprint-native-g1-team-campaign-dispatcher.timer"
    g1_timer.write_text("[Timer]\nOnUnitInactiveSec=2min\n", encoding="utf-8")
    g1_settlement_service = unit_dir / "blueprint-native-g1-team-campaign-settlement.service"
    g1_settlement_service.write_text("[Service]\nExecStart=/usr/bin/blueprint-g1-settlement\n", encoding="utf-8")
    g1_settlement_timer = unit_dir / "blueprint-native-g1-team-campaign-settlement.timer"
    g1_settlement_timer.write_text("[Timer]\nOnUnitInactiveSec=5min\n", encoding="utf-8")
    discovery_service = unit_dir / "blueprint-scene-object-discovery.service"
    discovery_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-discover-scene-objects\n",
        encoding="utf-8",
    )
    discovery_path = unit_dir / "blueprint-scene-object-discovery.path"
    discovery_path.write_text(
        "[Path]\nPathChanged=/scene-object-discoveries/pending\n"
        "PathExistsGlob=/scene-object-discoveries/pending/*.json\n",
        encoding="utf-8",
    )
    progression_service = (
        unit_dir
        / "blueprint-task-evaluation-configured-controls-progression.service"
    )
    progression_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-progress-configured-controls\n",
        encoding="utf-8",
    )
    progression_timer = (
        unit_dir
        / "blueprint-task-evaluation-configured-controls-progression.timer"
    )
    progression_timer.write_text(
        "[Timer]\nOnUnitInactiveSec=2min\n",
        encoding="utf-8",
    )
    progression_path = (
        unit_dir
        / "blueprint-task-evaluation-configured-controls-progression.path"
    )
    progression_path.write_text(
        "[Path]\nPathChanged=/task-evaluation-episode-compilations/results\n",
        encoding="utf-8",
    )
    storage_gc_service = unit_dir / "blueprint-control-plane-storage-gc.service"
    storage_gc_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-reclaim-storage\n",
        encoding="utf-8",
    )
    storage_gc_timer = unit_dir / "blueprint-control-plane-storage-gc.timer"
    storage_gc_timer.write_text(
        "[Timer]\nOnUnitInactiveSec=6h\n",
        encoding="utf-8",
    )
    capacity_service = unit_dir / "blueprint-control-plane-capacity.service"
    capacity_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-measure-capacity\n",
        encoding="utf-8",
    )
    capacity_timer = unit_dir / "blueprint-control-plane-capacity.timer"
    capacity_timer.write_text(
        "[Timer]\nOnUnitActiveSec=10min\n",
        encoding="utf-8",
    )
    preflight_service = unit_dir / "blueprint-control-plane-preflight.service"
    preflight_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-chain-preflight\n",
        encoding="utf-8",
    )
    preflight_timer = unit_dir / "blueprint-control-plane-preflight.timer"
    preflight_timer.write_text(
        "[Timer]\nOnUnitActiveSec=15min\n",
        encoding="utf-8",
    )
    intake_service = unit_dir / "blueprint-pipeline-intake.service"
    intake_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-live-pipeline-intake\n",
        encoding="utf-8",
    )
    control_plane_service = unit_dir / "blueprint-pipeline-control-plane.service"
    control_plane_service.write_text(
        "[Service]\nExecStart=/usr/bin/blueprint-live-pipeline-control-plane\n",
        encoding="utf-8",
    )
    additional_sources = []
    for name in (
        "blueprint-pubsub-handoff-listener.service", "blueprint-pubsub-handoff-listener.timer",
        "blueprint-agent-execution.service", "blueprint-agent-run-dispatcher.service",
        "blueprint-agent-run-dispatcher.timer", "blueprint-agent-stage-replay.service", "blueprint-agent-stage-replay.timer",
        "blueprint-task-evaluation-scene-progression.service", "blueprint-task-evaluation-scene-progression.timer",
        "blueprint-task-evaluation-launch-supervisor.service", "blueprint-task-evaluation-launch-supervisor.timer",
        "blueprint-task-evaluation-launch-reconciler.service", "blueprint-task-evaluation-launch-reconciler.timer",
        "blueprint-provider-billing-reconciler.service", "blueprint-provider-billing-reconciler.timer",
        "blueprint-task-evaluation-terminal-resource-release.service", "blueprint-task-evaluation-terminal-resource-release.path",
        "blueprint-gpu-spend-guard.service", "blueprint-gpu-spend-guard.timer",
        "blueprint-completed-replay-cache-gc.service", "blueprint-completed-replay-cache-gc.timer",
        "blueprint-scene-project-spend-refresh.service",
    ):
        extra = unit_dir / name
        extra.write_text("[Unit]\nDescription=Exact fixture " + name + "\n")
        additional_sources.append(extra)
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    (systemd / service.name).write_text(
        "[Service]\nKillMode=control-group\n", encoding="utf-8"
    )
    # The hand-copied watcher this deploy must replace byte-for-byte.
    (systemd / path_unit.name).write_text(
        "[Path]\nPathExistsGlob=/queue/pending/*.json\n", encoding="utf-8"
    )

    receipts = deploy._install_release_systemd_units(
        release_path=release,
        systemd_dir=systemd,
    )

    expected = []
    for source in (
        *additional_sources[:7],
        service,
        path_unit,
        preparation_service,
        preparation_path,
        preparation_timer,
        sam31_service,
        sam31_path,
        sam31_timer,
        compilation_service,
        compilation_path,
        compilation_timer,
        *remote_units,
        activation_service,
        activation_path,
        canary_service,
        canary_path,
        g1_service,
        g1_timer,
        g1_settlement_service,
        g1_settlement_timer,
        discovery_service,
        discovery_path,
        progression_service,
        progression_timer,
        progression_path,
        *additional_sources[7:],
        storage_gc_service,
        storage_gc_timer,
        capacity_service,
        capacity_timer,
        preflight_service,
        preflight_timer,
        control_plane_service,
        intake_service,
    ):
        destination = systemd / source.name
        assert destination.read_bytes() == source.read_bytes()
        assert destination.stat().st_mode & 0o777 == 0o644
        expected.append(
            {
                "unit": source.name,
                "source_path": str(source),
                "installed_path": str(destination),
                "sha256": deploy._sha256_bytes(source.read_bytes()),
                "size_bytes": len(source.read_bytes()),
                "mode": "0644",
            }
        )
    assert receipts == expected


def test_deployed_unit_set_contains_paid_and_no_spend_queue_pairs() -> None:
    """Deploying one half of the pair is how the watcher went stale.

    PR #1057 changed how the queue wakes the dispatcher (``PathChanged=``),
    and the canonical deploy would have installed only the ``.service`` --
    leaving the watcher on whatever bytes an operator once copied by hand.
    """

    assert deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS == (
        "blueprint-pubsub-handoff-listener.service",
        "blueprint-pubsub-handoff-listener.timer",
        "blueprint-agent-execution.service",
        "blueprint-agent-run-dispatcher.service",
        "blueprint-agent-run-dispatcher.timer",
        "blueprint-agent-stage-replay.service",
        "blueprint-agent-stage-replay.timer",
        "blueprint-task-evaluation-launch-dispatcher.service",
        "blueprint-task-evaluation-launch-dispatcher.path",
        "blueprint-task-evaluation-launch-preparation.service",
        "blueprint-task-evaluation-launch-preparation.path",
        "blueprint-task-evaluation-launch-preparation.timer",
        "blueprint-task-evaluation-sam31-preparation-execution.service",
        "blueprint-task-evaluation-sam31-preparation-execution.path",
        "blueprint-task-evaluation-sam31-preparation-execution.timer",
        "blueprint-task-evaluation-episode-compilation.service",
        "blueprint-task-evaluation-episode-compilation.path",
        "blueprint-task-evaluation-episode-compilation.timer",
        # Plan 14 §1: the paid remote-compilation unit, woken by its timer and by hand-offs.
        "blueprint-task-evaluation-episode-compilation-remote.service",
        "blueprint-task-evaluation-episode-compilation-remote.timer",
        "blueprint-task-evaluation-episode-compilation-remote.path",
        "blueprint-task-evaluation-launch-activation.service",
        "blueprint-task-evaluation-launch-activation.path",
        "blueprint-task-evaluation-policy-canary-dispatcher.service",
        "blueprint-task-evaluation-policy-canary-dispatcher.path",
        "blueprint-native-g1-team-campaign-dispatcher.service",
        "blueprint-native-g1-team-campaign-dispatcher.timer",
        "blueprint-native-g1-team-campaign-settlement.service",
        "blueprint-native-g1-team-campaign-settlement.timer",
        "blueprint-scene-object-discovery.service",
        "blueprint-scene-object-discovery.path",
        "blueprint-task-evaluation-configured-controls-progression.service",
        "blueprint-task-evaluation-configured-controls-progression.timer",
        "blueprint-task-evaluation-configured-controls-progression.path",
        "blueprint-task-evaluation-scene-progression.service",
        "blueprint-task-evaluation-scene-progression.timer",
        "blueprint-task-evaluation-launch-supervisor.service",
        "blueprint-task-evaluation-launch-supervisor.timer",
        "blueprint-task-evaluation-launch-reconciler.service",
        "blueprint-task-evaluation-launch-reconciler.timer",
        "blueprint-provider-billing-reconciler.service",
        "blueprint-provider-billing-reconciler.timer",
        "blueprint-task-evaluation-terminal-resource-release.service",
        "blueprint-task-evaluation-terminal-resource-release.path",
        "blueprint-gpu-spend-guard.service",
        "blueprint-gpu-spend-guard.timer",
        "blueprint-completed-replay-cache-gc.service",
        "blueprint-completed-replay-cache-gc.timer",
        "blueprint-scene-project-spend-refresh.service",
        "blueprint-control-plane-storage-gc.service",
        "blueprint-control-plane-storage-gc.timer",
        "blueprint-control-plane-capacity.service",
        "blueprint-control-plane-capacity.timer",
        "blueprint-control-plane-preflight.service",
        "blueprint-control-plane-preflight.timer",
        "blueprint-pipeline-control-plane.service",
        "blueprint-pipeline-intake.service",
    )
    assert deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS == (
        "blueprint-task-evaluation-launch-preparation.path",
        "blueprint-task-evaluation-episode-compilation.path",
        "blueprint-task-evaluation-launch-activation.path",
        "blueprint-scene-object-discovery.path",
    )
    assert deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS == (
        "blueprint-task-evaluation-launch-preparation.timer",
        "blueprint-native-g1-team-campaign-dispatcher.timer",
        "blueprint-native-g1-team-campaign-settlement.timer",
        "blueprint-agent-run-dispatcher.timer",
        "blueprint-agent-stage-replay.timer",
        "blueprint-task-evaluation-scene-progression.timer",
        "blueprint-task-evaluation-sam31-preparation-execution.timer",
        "blueprint-task-evaluation-episode-compilation.timer",
        "blueprint-task-evaluation-episode-compilation-remote.timer",
        "blueprint-task-evaluation-configured-controls-progression.timer",
        "blueprint-task-evaluation-configured-controls-progression.path",
        "blueprint-control-plane-storage-gc.timer",
        "blueprint-completed-replay-cache-gc.timer",
        "blueprint-control-plane-capacity.timer",
        "blueprint-control-plane-preflight.timer",
    )
    assert deploy.DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS == (
        "blueprint-task-evaluation-sam31-preparation-execution.path",
        "blueprint-task-evaluation-policy-canary-dispatcher.path",
        "blueprint-task-evaluation-episode-compilation-remote.path",
    )


@pytest.mark.parametrize(
    "unit",
    [
        "blueprint-task-evaluation-launch-dispatcher.socket",
        "blueprint-task-evaluation-launch-dispatcher.mount",
        "../blueprint-task-evaluation-launch-dispatcher.service",
        "dispatcher",
    ],
)
def test_only_service_path_and_timer_unit_suffixes_may_be_installed(
    tmp_path: Path, unit: str
) -> None:
    with pytest.raises(
        deploy.ControlPlaneDeployError, match="deploy_systemd_unit_name_invalid"
    ):
        deploy._install_release_systemd_units(
            release_path=tmp_path / "release",
            systemd_dir=tmp_path / "systemd",
            units=(unit,),
        )


def test_configured_controls_timer_is_installed_and_armed_by_default(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    unit = "blueprint-task-evaluation-configured-controls-progression.timer"
    enabled = "disabled"
    active = "inactive"

    def completed(argv, **kwargs):
        nonlocal enabled, active
        calls.append(tuple(argv))
        if argv[:2] == ["systemctl", "enable"]:
            enabled = "enabled"
        elif argv[:2] == ["systemctl", "restart"]:
            active = "active"
        stdout = ""
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = enabled + "\n"
        elif argv[:2] == ["systemctl", "is-active"]:
            stdout = active + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    observed = deploy._installed_path_unit_states([{"unit": unit}])
    restored = deploy._restore_installed_path_units(
        [{"unit": unit}],
        before=observed,
        arm_path_units=False,
        always_arm_timer_units=deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS,
    )

    assert observed == {unit: {"enabled": "disabled", "state": "inactive"}}
    assert calls == [
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
        ("systemctl", "enable", unit),
        ("systemctl", "restart", unit),
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
    ]
    assert restored == [
        {
            "unit": unit,
            "before": {"enabled": "disabled", "state": "inactive"},
            "requested_intent": "arm_configured_controls_progression",
            "after": {"enabled": "enabled", "state": "active"},
            "operator_freeze_preserved": False,
        }
    ]


def test_deploy_never_rearms_a_held_timer(tmp_path, monkeypatch) -> None:
    unit = "blueprint-task-evaluation-scene-progression.timer"
    holds = tmp_path / "holds"
    holds.mkdir()
    (holds / f"{unit}.json").write_text(json.dumps({
        "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice",
        "reason": "inspect capture", "request_id": "20260927T000000Z-hold-0000abcd",
        "status": "active", "expires_at": "2099-01-01T00:00:00+00:00",
        "expires_at_epoch": 4070908800,
    }), encoding="utf-8")
    active, warning = deploy._active_door_holds(holds)
    assert warning is None and active[unit]["owner"] == "alice"
    calls = []
    state = {"enabled": "enabled", "state": "inactive"}

    def completed(argv, **_kwargs):
        calls.append(tuple(argv))
        if argv[:2] == ["systemctl", "stop"]:
            state["state"] = "inactive"
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda _unit: dict(state))
    restored = deploy._restore_installed_path_units(
        [{"unit": unit}], before={unit: {"enabled": "enabled", "state": "active"}},
        arm_path_units=False, always_arm_timer_units=deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS,
        held_units=active,
    )
    assert calls == [("systemctl", "stop", unit)]
    assert restored[0]["held"] is True
    assert {key: restored[0][key] for key in ("owner", "reason", "expires_at")} == {
        "owner": "alice", "reason": "inspect capture", "expires_at": "2099-01-01T00:00:00+00:00"}
    assert restored[0]["after"] == {"enabled": "enabled", "state": "inactive"}


def test_expired_door_hold_is_ignored_and_unreadable_holds_warn(tmp_path) -> None:
    unit = "blueprint-task-evaluation-scene-progression.timer"
    holds = tmp_path / "holds"
    holds.mkdir()
    (holds / f"{unit}.json").write_text(json.dumps({
        "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice", "reason": "inspect",
        "request_id": "20260927T000000Z-hold-0000abcd", "status": "active",
        "expires_at": "2026-01-01T00:00:00+00:00", "expires_at_epoch": 1767225600,
    }), encoding="utf-8")
    assert deploy._active_door_holds(holds) == ({}, None)
    bad = tmp_path / "not-a-directory"
    bad.write_text("", encoding="utf-8")
    assert deploy._active_door_holds(bad) == ({}, "door_holds_unreadable")


def test_failed_deploy_rechecks_a_hold_created_after_unit_snapshot(tmp_path, monkeypatch) -> None:
    unit = "blueprint-task-evaluation-scene-progression.timer"
    holds = tmp_path / "holds"
    holds.mkdir()
    restored = []
    monkeypatch.setattr(deploy, "_installed_path_unit_states", lambda _units: {
        unit: {"enabled": "enabled", "state": "active"}
    })
    monkeypatch.setattr(deploy, "_quiesce_active_path_units", lambda _before: [])
    monkeypatch.setattr(deploy, "_restore_installed_path_units",
                        lambda _units, **kwargs: restored.append(kwargs) or [])

    with pytest.raises(ValueError, match="deploy_failed"):
        with deploy._restore_path_unit_states_on_deploy_failure(
            [{"unit": unit}], door_holds_dir=holds,
        ):
            (holds / f"{unit}.json").write_text(json.dumps({
                "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice",
                "reason": "inspect", "request_id": "20260927T000000Z-hold-0000abcd",
                "status": "active", "expires_at": "2099-01-01T00:00:00+00:00",
                "expires_at_epoch": 4070908800,
            }), encoding="utf-8")
            raise ValueError("deploy_failed")

    assert restored[0]["held_units"][unit]["owner"] == "alice"


@pytest.mark.parametrize("unit", deploy.CONFIGURED_CONTROLS_AUTOMATION_UNITS)
@pytest.mark.parametrize("enabled", ["disabled", "enabled"])
def test_explicit_controls_pause_survives_default_progression_arming(
    monkeypatch, unit: str, enabled: str,
) -> None:
    calls = []
    before = {"enabled": enabled, "state": "inactive"}

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda _unit: dict(before))
    restored = deploy._restore_installed_path_units(
        [{"unit": unit}], before={unit: before}, arm_path_units=False,
        always_arm_timer_units=deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS,
        preserve_configured_controls_state=True,
    )
    assert calls == [
        ("systemctl", "enable" if enabled == "enabled" else "disable", unit),
        ("systemctl", "stop", unit),
    ]
    assert restored[0]["after"] == before
    assert restored[0]["requested_intent"] == "preserve"
    assert restored[0]["operator_freeze_preserved"] is True


def test_conflicting_controls_intent_refuses_before_any_host_action(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("host action"))
    with pytest.raises(deploy.ControlPlaneDeployError, match="conflicting_configured_controls_intent"):
        deploy.deploy_control_plane_commit(
            source_repo=tmp_path, source_commit="a" * 40,
            release_root=tmp_path / "releases", state_root=tmp_path / "state",
            active_link=tmp_path / "active", arm_path_units=True,
            preserve_configured_controls_state=True,
        )


def test_cli_forwards_explicit_controls_state_preservation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda path: True)
    monkeypatch.setattr(deploy, "deploy_control_plane_commit", lambda **kwargs: calls.append(kwargs) or {"status": "deployed"})
    assert deploy.main([
        "--source-repo", str(tmp_path), "--source-commit", "a" * 40,
        "--release-root", str(tmp_path / "releases"), "--state-root", str(tmp_path / "state"),
        "--active-link", str(tmp_path / "active"), "--preserve-configured-controls-state",
    ]) == 0
    assert calls[0]["preserve_configured_controls_state"] is True
    assert calls[0]["arm_path_units"] is False


def test_cli_sigterm_unwinds_deploy_and_restores_signal_handler(tmp_path, monkeypatch, capsys):
    original = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda _path: True)

    def interrupted(**_kwargs):
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(deploy, "deploy_control_plane_commit", interrupted)
    assert deploy.main([
        "--source-repo", str(tmp_path), "--source-commit", "a" * 40,
        "--release-root", str(tmp_path / "releases"), "--state-root", str(tmp_path / "state"),
        "--active-link", str(tmp_path / "active"),
    ]) == 2
    assert "deploy_interrupted:SIGTERM" in capsys.readouterr().out
    assert signal.getsignal(signal.SIGTERM) is original


def test_cli_defers_sigterm_during_active_release_transition(tmp_path, monkeypatch):
    original = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda _path: True)

    def transition(**_kwargs):
        deploy._DEPLOY_ACTIVE_TRANSITION = True
        signal.raise_signal(signal.SIGTERM)
        deploy._DEPLOY_ACTIVE_TRANSITION = False
        return {"status": "deployed"}

    monkeypatch.setattr(deploy, "deploy_control_plane_commit", transition)
    assert deploy.main([
        "--source-repo", str(tmp_path), "--source-commit", "a" * 40,
        "--release-root", str(tmp_path / "releases"), "--state-root", str(tmp_path / "state"),
        "--active-link", str(tmp_path / "active"),
    ]) == 0
    assert signal.getsignal(signal.SIGTERM) is original


def test_cli_defers_sigterm_until_success_receipt_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda _path: True)
    def transition(**_kwargs):
        deploy._DEPLOY_ACTIVE_TRANSITION = True
        return {"status": "deployed"}

    monkeypatch.setattr(deploy, "deploy_control_plane_commit", transition)
    real_write = deploy._write_receipt_and_return

    def interrupted_write(receipt, path):
        signal.raise_signal(signal.SIGTERM)
        return real_write(receipt, path)

    monkeypatch.setattr(deploy, "_write_receipt_and_return", interrupted_write)
    output = tmp_path / "receipt.json"
    assert deploy.main([
        "--source-repo", str(tmp_path), "--source-commit", "a" * 40,
        "--release-root", str(tmp_path / "releases"), "--state-root", str(tmp_path / "state"),
        "--active-link", str(tmp_path / "active"), "--receipt-out", str(output),
    ]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "deployed"


def _cli_args(tmp_path: Path, source: Path, *extra: str) -> list[str]:
    return [
        "--source-repo", str(source), "--source-commit", "a" * 40,
        "--release-root", str(tmp_path / "releases"), "--state-root", str(tmp_path / "state"),
        "--active-link", str(tmp_path / "active"), "--iteration",
        "--receipt-out", str(tmp_path / "receipt.json"), *extra,
    ]


def _deploy_note(
    root: Path,
    *,
    age_seconds: int = 60,
    actions: tuple[str, ...] = (break_glass.DEPLOY_FROM_UNTRUSTED_SOURCE,),
) -> Path:
    created = time.time() - age_seconds
    return break_glass.record_note(
        root=root,
        operator="alice",
        reason="the door is down; deploying the fix from a scratch checkout",
        actions=list(actions),
        now=lambda: created,
        environ={},
    )


@pytest.mark.parametrize(
    ("note", "code"),
    [
        (None, "break_glass_note_missing"),
        ("stale", "break_glass_note_expired"),
        ("other_action", "break_glass_note_action_missing"),
        ("edited", "break_glass_note_digest_mismatch"),
        ("absent", "break_glass_note_unreadable"),
        ("outside", "break_glass_note_outside_notes_root"),
        ("outside_link", "break_glass_note_outside_notes_root"),
        ("inside_link", "break_glass_note_unsafe"),
    ],
)
def test_main_refuses_an_untrusted_source_without_a_note(
    tmp_path, monkeypatch, capsys, note, code
):
    """On 2026-09-26 a scratch-checkout deploy left GPU admission refusing for hours.

    The deploy itself succeeded; the release it produced was one GPU admission
    cannot verify, so every sponsored step refused until a later deploy
    replaced it. The CLI now refuses such a source before anything moves.
    """

    source = tmp_path / "control-plane-deploy-sources" / "scratch"
    source.mkdir(parents=True)
    asked = []
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda path: asked.append(path) or False)
    monkeypatch.setattr(deploy, "deploy_control_plane_commit", lambda **_kwargs: pytest.fail("deploy ran"))
    monkeypatch.setattr(deploy.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("host action"))
    notes = tmp_path / "cleanup-receipts"
    monkeypatch.setattr(deploy, "DEFAULT_BREAK_GLASS_NOTES_ROOT", notes)
    extra: list[str] = []
    if note == "stale":
        extra = ["--break-glass-note", str(_deploy_note(notes, age_seconds=24 * 3600 + 60))]
    elif note == "other_action":
        extra = ["--break-glass-note", str(_deploy_note(notes, actions=("unit-restart",)))]
    elif note == "edited":
        path = _deploy_note(notes)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["reason"] = "a reason nobody sealed"
        path.write_text(json.dumps(document), encoding="utf-8")
        extra = ["--break-glass-note", str(path)]
    elif note == "absent":
        extra = ["--break-glass-note", str(notes / "20260926T120000Z-0123456789ab.json")]
    elif note == "outside":
        extra = ["--break-glass-note", str(_deploy_note(tmp_path / "other-notes"))]
    elif note == "outside_link":
        original = _deploy_note(tmp_path / "other-notes")
        notes.mkdir()
        linked = notes / original.name
        linked.symlink_to(original)
        extra = ["--break-glass-note", str(linked)]
    elif note == "inside_link":
        original = _deploy_note(notes)
        linked = notes / "alias.json"
        linked.symlink_to(original.name)
        extra = ["--break-glass-note", str(linked)]

    assert deploy.main(_cli_args(tmp_path, source, *extra)) == 2

    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    [blocker] = blocked["blockers"]
    assert blocker == f"deploy_source_repo_untrusted:{code}"
    assert "GPU admission will still refuse" in blocked["remedy"]
    assert "deploy_control_plane_canary.sh" in blocked["remedy"]
    assert "deploy_control_plane_iteration.sh" in blocked["remedy"]
    assert "/" not in blocker, "a refusal names no host path"
    # The question asked is the one admission asks of the receipt's source path.
    assert asked == [source.resolve()]
    assert not (tmp_path / "receipt.json").exists()


def test_main_accepts_an_untrusted_source_with_a_fresh_deploy_note(tmp_path, monkeypatch, capsys):
    source = tmp_path / "scratch"
    source.mkdir()
    note = _deploy_note(tmp_path / "cleanup-receipts")
    monkeypatch.setattr(deploy, "DEFAULT_BREAK_GLASS_NOTES_ROOT", note.parent)
    calls = []
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda path: False)
    monkeypatch.setattr(
        deploy, "deploy_control_plane_commit", lambda **kwargs: calls.append(kwargs) or {"status": "deployed"}
    )

    assert deploy.main(_cli_args(tmp_path, source, "--break-glass-note", str(note))) == 0

    sealed = break_glass.verify_note(note)
    expected = {
        "name": note.name,
        "path": str(note.resolve()),
        "digest": sealed["note_digest"],
        "operator": "alice",
        "reason": "the door is down; deploying the fix from a scratch checkout",
        "created_at": sealed["created_at"],
        "actions": [break_glass.DEPLOY_FROM_UNTRUSTED_SOURCE],
    }
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))["break_glass_note"] == expected
    output = json.loads(capsys.readouterr().out)
    assert output["break_glass_note"] == expected
    assert "break_glass_deploy_from_untrusted_source" in output["alerts"]
    assert calls[0]["source_repo"] == str(source)
    # Every CLI deploy reports the notes the last one did not, this one included.
    assert calls[0]["break_glass_notes_root"] == note.parent


def test_main_needs_no_note_for_a_trusted_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda path: True)
    monkeypatch.setattr(deploy, "deploy_control_plane_commit", lambda **_kwargs: {"status": "deployed"})

    assert deploy.main(_cli_args(tmp_path, tmp_path)) == 0

    assert json.loads(capsys.readouterr().out)["break_glass_note"] is None


def test_trusted_source_warns_when_a_break_glass_note_is_ignored(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda _path: True)
    monkeypatch.setattr(deploy, "deploy_control_plane_commit", lambda **_kwargs: {"status": "deployed"})
    note = _deploy_note(tmp_path / "cleanup-receipts")
    assert deploy.main(_cli_args(tmp_path, tmp_path, "--break-glass-note", str(note))) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["break_glass_note"] is None
    assert "break_glass_note_ignored_trusted_source" in output["alerts"]


def test_untrusted_door_source_refusal_names_the_ownership_remedy(monkeypatch):
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda _path: False)
    source = Path("/opt/blueprint/control-plane-config-tools/operator-door-source")
    with pytest.raises(deploy.UntrustedDeploySourceError) as error:
        deploy._require_trusted_deploy_source(source, None)
    assert "root:root 0755" in error.value.remedy


def test_door_and_iteration_wrappers_use_trusted_sources() -> None:
    """Every supported deploy path passes a source GPU admission trusts."""

    sys.path.insert(0, str(REPO_ROOT / "deploy" / "operator-door"))
    from operator_door.config import DoorConfig

    door = (REPO_ROOT / "deploy" / "operator-door" / "door-deploy.sh").read_text(encoding="utf-8")
    assert door.count("--source-repo") == 1
    assert '--source-repo "$DOOR_SOURCE_CLONE" ' in door
    assert Path(DoorConfig().source_clone).parent == admission.CONFIG_TOOLS_ROOT
    for wrapper in ("deploy_control_plane_iteration.sh", "deploy_control_plane_canary.sh"):
        text = (REPO_ROOT / "scripts" / wrapper).read_text(encoding="utf-8")
        assert f"\nCP={admission.SOURCE_CHECKOUT}\n" in text, wrapper
        assert text.count("--source-repo") == 1, wrapper
        assert "--source-repo $CP " in text, wrapper


def _stub_host_deploy(monkeypatch, tmp_path: Path, commit: str) -> dict[str, object]:
    """Replace every host-touching deploy step with a no-op; return deploy arguments."""

    release = tmp_path / "release"
    release.mkdir()
    active = tmp_path / "active"
    active.symlink_to(release, target_is_directory=True)
    source = tmp_path / "source"
    source.mkdir()
    staged = {"source_commit": commit, "release_path": str(release), "created_release_checkout": True}
    runtime_sources = {"sources": [
        {"id": "text-to-cad", "path": "/runtime/text-to-cad"},
        {"id": "multi-agent-cad", "path": "/runtime/Multi-Agent-CAD"},
    ]}
    entrypoints = dict.fromkeys(("node", "browser_root", "browser", "node_modules"), "/runtime")
    stubs = {
        "_holding_paid_launch_gate": lambda _locks: contextlib.nullcontext([]),
        "_installed_path_unit_states": lambda _installed: {},
        "_quiesce_active_path_units": lambda _before: [],
        "stage_task_evaluation_control_plane_release": lambda **_kwargs: staged,
        "_install_unit_sandbox_paths": lambda **_kwargs: [],
        "provision_production_cad_skill_sources": lambda _root: runtime_sources,
        "validate_splat_render_prerequisites": lambda **_kwargs: {"entrypoints": entrypoints},
        "_provision_scene_configuration_from_release": lambda **_kwargs: {"environment": {}},
        "_install_scene_configuration_environment": lambda *_args, **_kwargs: {},
        "_drain_agent_execution_before_release_switch": lambda **_kwargs: {},
        "_move_source_checkout": lambda *_args: None,
        "_surface_commit": lambda *_args, **_kwargs: commit,
        "_install_release_systemd_units": lambda **_kwargs: [],
        "_install_scene_object_discovery_runtime_directories": lambda: [],
        "_install_episode_compilation_runtime_directories": lambda: [],
        "_install_storage_pins_runtime_root": lambda: {},
        "_install_configured_controls_runtime_prerequisites": lambda: {},
        "_install_configured_controls_autostart_registry": lambda **_kwargs: {},
        "_install_intake_runtime_identity_drop_in": lambda *_args, **_kwargs: {},
        "_service_account_ids": lambda _account: None,
        "_restart_units": lambda _units: [],
        "_verify_intake_runtime": lambda *_args, **_kwargs: {"commit_proven": True},
        "_activate_agent_execution": lambda **_kwargs: {},
        "_restore_installed_path_units": lambda _installed, **_kwargs: [],
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(deploy, name, stub)
    monkeypatch.setattr(deploy.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("host command"))
    return {
        "source_repo": source,
        "source_commit": commit,
        "release_root": tmp_path / "releases",
        "state_root": tmp_path / "state",
        "active_link": active,
        "release_provenance": _provenance(tmp_path, commit),
        "paid_launch_locks": (str(tmp_path / "vast_paid_launch.lock"),),
        "scene_configuration_runtime_root": tmp_path / "system-runtimes",
        "scene_preparation_bootstrap_file": tmp_path / "absent-bootstrap.json",
        "controls_autoprovision_bootstrap_file": tmp_path / "absent-controls-bootstrap.json",
    }


@pytest.mark.parametrize("failure", [None, "disk_admission", "paid_gate", "intake_identity"])
def test_deploy_preserves_superseded_and_interrupted_retirement_trees(
    tmp_path, monkeypatch, failure,
) -> None:
    """Deployment preserves old bytes even when deletion would free admission space."""

    commit = "d" * 40
    arguments = _stub_host_deploy(monkeypatch, tmp_path, commit)
    releases = arguments["release_root"]
    runtimes = arguments["scene_configuration_runtime_root"]
    current = releases / commit
    current.mkdir(parents=True)
    arguments["active_link"].unlink()
    arguments["active_link"].symlink_to(current, target_is_directory=True)
    monkeypatch.setattr(deploy, "stage_task_evaluation_control_plane_release", lambda **_kwargs: {
        "source_commit": commit, "release_path": str(current), "created_release_checkout": False,
    })
    retained = []
    for root in (releases, *(runtimes / name for name in deploy.RELEASE_RUNTIME_COMPONENTS)):
        for index, old in enumerate(("a" * 40, "b" * 40, "c" * 40, "e" * 40)):
            tree = root / old
            (tree / "generated").mkdir(parents=True)
            (tree / "generated/receipt.json").write_bytes(b'{"generated":"retained"}\n')
            (tree / "untracked.bin").write_bytes(b"untracked bytes\x00\xff")
            (tree / "untracked.bin").chmod(0o600)
            os.link(tree / "untracked.bin", tree / "hardlinked.bin")
            stamp = time.time() - (10 + index) * 86_400
            os.utime(tree, (stamp, stamp))
            retained.append(tree)
            if root != releases:
                publication = root / f"{old}.publication.v1.json"
                publication.write_bytes(b'{"publication":"retained"}\n')
                os.utime(publication, (stamp, stamp))
                retained.append(publication)
        leftover = root / ".retiring" / f"{'f' * 40}-0123456789ab"
        leftover.mkdir(parents=True)
        (leftover / "generated.bin").write_bytes(b"interrupted retirement bytes")
        retained.append(leftover.parent)

    sources = _protection_sources(tmp_path / "protection")
    arguments["release_protection_sources"] = sources
    plan = deploy.build_release_retirement_plan(
        release_root=releases, runtime_root=runtimes, active_link=arguments["active_link"],
        current_commit=commit,
        protections=deploy.collect_release_protections(sources, now=time.time(), migrate=False),
    )
    assert plan["status"] == "dry_run" and plan["candidates"]
    for candidate in plan["candidates"]:
        old = candidate["commit"]
        expected = {str(releases / old)}
        for component in deploy.RELEASE_RUNTIME_COMPONENTS:
            expected.update({str(runtimes / component / old),
                             str(runtimes / component / f"{old}.publication.v1.json")})
        assert set(candidate["paths"]) == expected
    summary = arguments["state_root"] / "release-retention/latest-deploy-retirement.json"
    summary.parent.mkdir(parents=True)
    summary.write_bytes(b'{"status":"blocked","alerts":["previous_retirement_failure"]}\n')
    retained.append(summary)

    def snapshot():
        result = {}
        for root in retained:
            for path in (root, *root.rglob("*")):
                info = path.lstat()
                result[str(path)] = (
                    info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
                    info.st_mtime_ns, path.read_bytes() if path.is_file() else None,
                )
        return result

    before = snapshot()
    for name in ("_sweep_retiring_trees", "_retire_superseded_release_trees", "_finish_release_retirement"):
        monkeypatch.setattr(deploy, name, lambda *args, **kwargs: pytest.fail("deployment requested retirement"))

    if failure == "disk_admission":
        arguments["disk_reservation_root"] = tmp_path / "disk-reservations"
        monkeypatch.setattr(deploy, "_install_disk_reservation_runtime_prerequisites", lambda _root: {})
        monkeypatch.setattr(deploy, "_release_footprint_estimate", lambda *_args, **_kwargs: {
            "bytes": 4096, "basis": "test_estimate",
        })

        def refuse_disk(*_args, **_kwargs):
            raise deploy.ControlPlaneDiskBudgetError("insufficient_headroom")

        monkeypatch.setattr(deploy, "reserve_control_plane_disk", refuse_disk)
    elif failure:
        def refuse(*_args, **_kwargs):
            raise deploy.ControlPlaneDeployError(f"test_{failure}_refused")

        monkeypatch.setattr(deploy, "_holding_paid_launch_gate" if failure == "paid_gate"
                            else "_verify_intake_runtime", refuse)

    if failure:
        code = "deploy_disk_budget_exceeded:insufficient_headroom" if failure == "disk_admission" else f"test_{failure}_refused"
        with pytest.raises(deploy.ControlPlaneDeployError, match=code):
            deploy.deploy_control_plane_commit(**arguments)
    else:
        receipt = deploy.deploy_control_plane_commit(**arguments)
        retirement = receipt["release_retirement"]
        assert receipt["status"] == "deployed"
        assert retirement["status"] == "not_requested"
        assert retirement["reason"] == "requires_separate_action"
        assert retirement["retired_bytes"] == 0
        for key in ("retired_commits", "renamed", "deleted", "direct_delete_fallback", "swept", "startup_swept"):
            assert retirement[key] == []
        assert retirement["worktree_prune"]["status"] == "not_requested"
    assert snapshot() == before
    assert arguments["active_link"].resolve() == current


def test_deploy_reports_and_marks_break_glass_notes(tmp_path, monkeypatch) -> None:
    """The deploy function reports notes but never marks them before receipt output."""

    commit = "d" * 40
    notes = tmp_path / "cleanup-receipts"
    earlier = _deploy_note(notes, age_seconds=7200, actions=("unit-restart",))
    break_glass.mark_reported(notes, break_glass.unreported_notes(notes), deploy_commit="e" * 40)
    first = _deploy_note(notes, age_seconds=3600, actions=("unit-stop",))
    second = _deploy_note(notes, age_seconds=60)
    arguments = _stub_host_deploy(monkeypatch, tmp_path, commit)

    receipt = deploy.deploy_control_plane_commit(**arguments, break_glass_notes_root=notes)

    assert receipt["status"] == "deployed"
    sealed = [break_glass.verify_note(path) for path in (first, second)]
    assert receipt["break_glass_notes"] == [
        {
            "name": path.name,
            "digest": note["note_digest"],
            "operator": "alice",
            "reason": note["reason"],
            "created_at": note["created_at"],
        }
        for path, note in zip((first, second), sealed)
    ]
    assert receipt["alerts"] == ["break_glass_notes_reported:2"]
    assert "break_glass_notes_error" not in receipt
    ledger = (notes / break_glass.REPORTED_LEDGER).read_text(encoding="utf-8").splitlines()
    assert [(row["name"], row["deploy_commit"]) for row in map(json.loads, ledger)] == [
        (earlier.name, "e" * 40),
    ]

    # Without a saved CLI receipt, the same notes are still unreported.
    again = deploy.deploy_control_plane_commit(**arguments, break_glass_notes_root=notes)
    assert len(again["break_glass_notes"]) == 2
    # A direct caller that names no notes root reports nothing and reads nothing.
    assert "break_glass_notes" not in deploy.deploy_control_plane_commit(**arguments)


def test_deploy_reads_held_units_at_restore_and_warns_on_unreadable_records(tmp_path, monkeypatch) -> None:
    commit = "d" * 40
    args = _stub_host_deploy(monkeypatch, tmp_path, commit)
    unit = "blueprint-task-evaluation-scene-progression.timer"
    holds = tmp_path / "holds"
    holds.mkdir()
    (holds / f"{unit}.json").write_text(json.dumps({
        "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice",
        "reason": "inspect", "request_id": "20260927T000000Z-hold-0000abcd", "status": "active",
        "expires_at": "2099-01-01T00:00:00+00:00", "expires_at_epoch": 4070908800,
    }), encoding="utf-8")
    passed = []
    monkeypatch.setattr(deploy, "_restore_installed_path_units",
                        lambda _installed, **kwargs: passed.append(kwargs["held_units"]) or [])
    receipt = deploy.deploy_control_plane_commit(**args, door_holds_dir=holds)
    assert receipt["status"] == "deployed" and passed[0][unit]["owner"] == "alice"
    bad = tmp_path / "bad-holds"
    bad.write_text("", encoding="utf-8")
    receipt = deploy.deploy_control_plane_commit(**args, door_holds_dir=bad)
    assert "door_holds_unreadable" in receipt["alerts"]


@pytest.mark.parametrize("rollback", [False, True])
def test_timer_restore_allows_hold_sweep_to_take_its_lock(tmp_path, monkeypatch, rollback):
    """A real flock must not be held while systemd waits for the sweep dependency."""
    import fcntl

    sys.path.insert(0, str(REPO_ROOT / "deploy" / "operator-door"))
    from operator_door import holds as door_holds

    restore = deploy._restore_installed_path_units
    arguments = _stub_host_deploy(monkeypatch, tmp_path, "d" * 40)
    timer = "blueprint-pubsub-handoff-listener.timer"
    held = "blueprint-agent-run-dispatcher.timer"
    root = tmp_path / "holds"
    root.mkdir()
    record = {
        "schema": door_holds.SCHEMA, "unit": held, "owner": "release-owner",
        "reason": "compatibility review", "request_id": "20260927T000000Z-hold-0000abcd",
        "requested_by": "cloud", "status": "active", "enabled_before": True,
        "expires_at": "2099-01-01T00:00:00+00:00", "expires_at_epoch": 4070908800,
    }
    record_path = root / f"{held}.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    original_record = record_path.read_bytes()
    states = {
        timer: {"enabled": "enabled", "state": "inactive"},
        held: {"enabled": "disabled", "state": "inactive"},
    }
    queued = []
    sweep_calls = []
    commands = []
    paid_lock = tmp_path / "vast_paid_launch.lock"
    paid_lock.touch()

    @contextlib.contextmanager
    def paid_gate(_locks):
        with deploy._holding_paid_launch_locks([str(paid_lock)]):
            yield []

    def sweep_dependency():
        with paid_lock.open("r") as paid_contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(paid_contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Independent open descriptions exercise the kernel lock, even in one PID.
        with (root / ".lock").open("a") as contender:
            try:
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise subprocess.TimeoutExpired("hold-sweep dependency waiting for deploy lock", 0.1)
        assert door_holds.sweep(root) == 0
        sweep_calls.append(True)

    def systemctl(argv, **_kwargs):
        commands.append(tuple(argv))
        assert argv[0] == "systemctl", "No real host command is allowed"
        verb, unit = [item for item in argv[1:] if not item.startswith("--")]
        if verb == "restart":
            if "--no-block" in argv:
                queued.append(unit)
            else:
                sweep_dependency()
                states[unit]["state"] = "active"
        elif verb in {"enable", "disable"}:
            states[unit]["enabled"] = "enabled" if verb == "enable" else "disabled"
        elif verb == "stop":
            states[unit]["state"] = "inactive"
        else:
            raise AssertionError(f"Unexpected host operation: {verb}")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def complete_jobs(_seconds):
        if queued:
            sweep_dependency()
            for unit in queued:
                states[unit]["state"] = "active"
            queued.clear()

    monkeypatch.setattr(deploy, "DEFAULT_DEPLOYED_SYSTEMD_UNITS", (timer, held))
    monkeypatch.setattr(deploy, "_holding_paid_launch_gate", paid_gate)
    monkeypatch.setattr(deploy, "_restore_installed_path_units", restore)
    monkeypatch.setattr(deploy, "_install_release_systemd_units",
                        lambda **_kwargs: [{"unit": timer}, {"unit": held}])
    monkeypatch.setattr(deploy, "_installed_path_unit_states", lambda _units: {
        timer: {"enabled": "enabled", "state": "active"},
        held: {"enabled": "disabled", "state": "inactive"},
    })
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda unit, **_kwargs: dict(states[unit]))
    monkeypatch.setattr(deploy.subprocess, "run", systemctl)
    monkeypatch.setattr(deploy.time, "sleep", complete_jobs)
    if rollback:
        def fail_runtime(*_args, **_kwargs):
            raise ValueError("runtime_identity_refused")
        monkeypatch.setattr(deploy, "_verify_intake_runtime", fail_runtime)
        with pytest.raises(ValueError, match="^runtime_identity_refused$"):
            deploy.deploy_control_plane_commit(**arguments, door_holds_dir=root)
    else:
        receipt = deploy.deploy_control_plane_commit(**arguments, door_holds_dir=root)
        assert receipt["status"] == "deployed"
        restored = next(row for row in receipt["timer_unit_states"] if row["unit"] == held)
        assert restored["held"] and restored["owner"] == record["owner"]
    assert sweep_calls == [True]
    assert states[timer] == {"enabled": "enabled", "state": "active"}
    assert states[held] == {"enabled": "disabled", "state": "inactive"}
    assert record_path.read_bytes() == original_record
    assert not any(command[-1] == held and "restart" in command for command in commands)
    with paid_lock.open("r") as paid_contender:
        fcntl.flock(paid_contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_owner_hold_cancels_a_queued_timer_restore(tmp_path, monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"
    root = tmp_path / "holds"
    root.mkdir()
    state = {"enabled": "enabled", "state": "inactive"}
    commands = []

    def systemctl(argv, **_kwargs):
        commands.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", systemctl)
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda _unit, **_kwargs: dict(state))
    with deploy._locked_door_holds(root) as (held_units, warning):
        assert warning is None
        restored = deploy._restore_installed_path_units(
            [{"unit": unit}], before={unit: {"enabled": "enabled", "state": "active"}},
            arm_path_units=False, held_units=held_units, defer_start_verification=True,
        )
    assert commands == [("systemctl", "enable", unit),
                        ("systemctl", "--no-block", "restart", unit)]
    # An owner can obtain the lock between enqueue and completion and cancel the job.
    record = {
        "schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "new-owner",
        "reason": "inspect", "request_id": "20260927T000000Z-hold-0000abcd",
        "status": "active", "expires_at": "2099-01-01T00:00:00+00:00",
        "expires_at_epoch": 4070908800,
    }
    path = root / f"{unit}.json"
    with deploy._locked_door_holds(root):
        state.update(enabled="disabled", state="inactive")
        path.write_text(json.dumps(record), encoding="utf-8")
    original = path.read_bytes()
    assert deploy._verify_deferred_path_unit_starts(restored, door_holds_dir=root) is None
    assert restored[0]["held"] and restored[0]["owner"] == "new-owner"
    assert restored[0]["after"] == state
    assert "_start_pending" not in restored[0]
    assert len(commands) == 2 and path.read_bytes() == original


def test_queued_timer_start_has_a_bounded_verification_timeout(tmp_path, monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"
    state = {"enabled": "enabled", "state": "inactive"}
    restored = [{"unit": unit, "after": state, "_start_pending": True}]
    clock = iter([0.0, *([1.0] * 10)])
    monkeypatch.setattr(deploy.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda _unit, **_kwargs: dict(state))
    monkeypatch.setattr(deploy.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("host mutation"))
    with pytest.raises(deploy.ControlPlaneDeployError, match="^deploy_path_unit_start_timeout:"):
        deploy._verify_deferred_path_unit_starts(
            restored, door_holds_dir=tmp_path / "absent", timeout_seconds=0.5,
        )


def test_timer_verification_preserves_boot_policy_check(tmp_path, monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"
    restored = [{"unit": unit, "after": {"enabled": "enabled"}, "_start_pending": True}]
    monkeypatch.setattr(deploy, "_systemd_unit_state",
                        lambda _unit, **_kwargs: {"enabled": "disabled", "state": "active"})
    with pytest.raises(deploy.ControlPlaneDeployError, match="enabled_state_mismatch"):
        deploy._verify_deferred_path_unit_starts(restored, door_holds_dir=tmp_path / "absent")


def test_timer_verification_state_probe_is_bounded(monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"

    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 15
        raise subprocess.TimeoutExpired(argv, 15)

    monkeypatch.setattr(deploy.subprocess, "run", timeout)
    with pytest.raises(deploy.ControlPlaneDeployError, match="^deploy_systemd_state_probe_failed:"):
        deploy._systemd_unit_state(unit)


def test_timer_verification_bounds_contended_hold_lock(tmp_path, monkeypatch):
    import fcntl

    root = tmp_path / "holds"
    root.mkdir()
    unit = "blueprint-pubsub-handoff-listener.timer"
    restored = [{"unit": unit, "after": {"enabled": "enabled"}, "_start_pending": True}]
    clock = [0.0]
    monkeypatch.setattr(deploy.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(deploy.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda *_args, **_kwargs: pytest.fail("probe under contended lock"))
    with (root / ".lock").open("a") as contender:
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX)
        with pytest.raises(deploy.ControlPlaneDeployError, match="^deploy_door_holds_lock_timeout$"):
            deploy._verify_deferred_path_unit_starts(restored, door_holds_dir=root, timeout_seconds=0.2)
    # The timed-out reader closes its descriptor rather than retaining the lock.
    with deploy._locked_door_holds(root):
        pass
    assert restored[0]["_start_pending"]


def test_timer_verification_rejects_late_final_active_probe(tmp_path, monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"
    restored = [{"unit": unit, "after": {"enabled": "enabled"}, "_start_pending": True}]
    clock = [0.0]
    monkeypatch.setattr(deploy.time, "monotonic", lambda: clock[0])

    def late_probe(_unit, *, deadline):
        clock[0] = deadline
        return {"enabled": "enabled", "state": "active"}

    monkeypatch.setattr(deploy, "_systemd_unit_state", late_probe)
    with pytest.raises(deploy.ControlPlaneDeployError, match="^deploy_path_unit_start_timeout:"):
        deploy._verify_deferred_path_unit_starts(restored, door_holds_dir=tmp_path / "absent", timeout_seconds=0.2)
    assert restored[0]["_start_pending"]


def test_timer_verification_probes_share_the_remaining_budget(monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"
    clock = [0.0]
    timeouts = []
    monkeypatch.setattr(deploy.time, "monotonic", lambda: clock[0])

    def probe(argv, *, timeout, **_kwargs):
        timeouts.append(timeout)
        clock[0] += 0.1
        return subprocess.CompletedProcess(argv, 0, stdout="enabled" if argv[1] == "is-enabled" else "active")

    monkeypatch.setattr(deploy.subprocess, "run", probe)
    assert deploy._systemd_unit_state(unit, deadline=0.3) == {"enabled": "enabled", "state": "active"}
    assert timeouts == pytest.approx([0.3, 0.2])


def test_timer_verification_probe_deadline_prevents_late_success(monkeypatch):
    unit = "blueprint-pubsub-handoff-listener.timer"
    clock = [0.0]
    monkeypatch.setattr(deploy.time, "monotonic", lambda: clock[0])

    def late_probe(argv, **_kwargs):
        clock[0] = 0.3
        return subprocess.CompletedProcess(argv, 0, stdout="enabled")

    monkeypatch.setattr(deploy.subprocess, "run", late_probe)
    with pytest.raises(deploy.ControlPlaneDeployError, match="^deploy_systemd_state_probe_failed:"):
        deploy._systemd_unit_state(unit, deadline=0.3)


def test_unreadable_break_glass_notes_never_fail_a_finished_deploy(tmp_path, monkeypatch) -> None:
    notes = tmp_path / "cleanup-receipts"
    _deploy_note(notes)
    receipt: dict[str, object] = {"status": "deployed", "alerts": ["earlier_alert"]}
    notes_root = tmp_path / "not-a-directory"
    notes_root.write_text("", encoding="utf-8")

    deploy._report_break_glass_notes(receipt, root=notes_root, deploy_commit="d" * 40)

    assert receipt["break_glass_notes"] is None
    assert receipt["break_glass_notes_error"] == "break_glass_notes_root_unsafe"
    assert receipt["alerts"] == [
        "earlier_alert",
        "break_glass_notes_unreadable:break_glass_notes_root_unsafe",
    ]
    assert str(tmp_path) not in json.dumps(receipt)


def _stub_main_break_glass_notes(monkeypatch, tmp_path):
    notes = tmp_path / "cleanup-receipts"
    note = _deploy_note(notes)
    monkeypatch.setattr(deploy, "DEFAULT_BREAK_GLASS_NOTES_ROOT", notes)
    monkeypatch.setattr(deploy, "trusted_deploy_source", lambda _path: True)

    def deployed(**_kwargs):
        receipt = {"status": "deployed", "source_commit": "a" * 40}
        deploy._report_break_glass_notes(receipt, root=notes, deploy_commit="a" * 40)
        return receipt

    monkeypatch.setattr(deploy, "deploy_control_plane_commit", deployed)
    return notes, note


def test_main_marks_break_glass_notes_only_after_atomic_receipt_write(tmp_path, monkeypatch, capsys):
    notes, note = _stub_main_break_glass_notes(monkeypatch, tmp_path)
    real_mark = deploy.mark_break_glass_notes_reported

    def mark(root, rows, **kwargs):
        written = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert written["break_glass_notes"][0]["name"] == note.name
        real_mark(root, rows, **kwargs)

    monkeypatch.setattr(deploy, "mark_break_glass_notes_reported", mark)
    assert deploy.main(_cli_args(tmp_path, tmp_path)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "deployed"
    assert break_glass.unreported_notes(notes) == []


def test_receipt_replacement_keeps_prior_file_if_rename_fails(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError(errno.EIO, "rename failed")

    monkeypatch.setattr(deploy.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failed"):
        deploy._write_receipt_and_return({"status": "deployed"}, str(output))
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []


def test_atomic_deploy_receipt_is_readable_by_the_operator_door(tmp_path):
    output = tmp_path / "receipt.json"
    deploy._write_receipt_and_return({"status": "deployed"}, str(output))
    assert output.stat().st_mode & 0o777 == 0o644


def test_receipt_rename_is_synced_before_notes_can_be_marked(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync(fd):
        events.append("directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync")
        real_fsync(fd)

    def replace(source, target):
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(deploy.os, "fsync", fsync)
    monkeypatch.setattr(deploy.os, "replace", replace)
    deploy._write_receipt_and_return({"status": "deployed"}, str(output))
    assert events == ["file_fsync", "replace", "directory_fsync"]


def test_failed_receipt_write_prints_deployed_receipt_and_keeps_notes_unreported(tmp_path, monkeypatch, capsys):
    notes, note = _stub_main_break_glass_notes(monkeypatch, tmp_path)

    def fail_write(_receipt, _path):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(deploy, "_write_receipt_and_return", fail_write)
    assert deploy.main(_cli_args(tmp_path, tmp_path)) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "deployed"
    assert printed["blockers"] == ["deploy_receipt_write_failed:ENOSPC"]
    assert printed["break_glass_notes"][0]["name"] == note.name
    assert [row["name"] for row in break_glass.unreported_notes(notes)] == [note.name]


def test_failed_note_mark_rewrites_receipt_with_alert(tmp_path, monkeypatch, capsys):
    notes, note = _stub_main_break_glass_notes(monkeypatch, tmp_path)

    def fail_mark(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(deploy, "mark_break_glass_notes_reported", fail_mark)
    assert deploy.main(_cli_args(tmp_path, tmp_path)) == 0
    printed = json.loads(capsys.readouterr().out)
    assert "break_glass_notes_not_marked:break_glass_io_error:ENOSPC" in printed["alerts"]
    assert json.loads((tmp_path / "receipt.json").read_text()) == printed
    assert [row["name"] for row in break_glass.unreported_notes(notes)] == [note.name]


def test_no_receipt_path_never_marks_break_glass_notes(tmp_path, monkeypatch, capsys):
    notes, note = _stub_main_break_glass_notes(monkeypatch, tmp_path)
    args = _cli_args(tmp_path, tmp_path)
    args = args[:args.index("--receipt-out")]
    assert deploy.main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert "break_glass_notes_not_marked:no_receipt_out" in printed["alerts"]
    assert [row["name"] for row in break_glass.unreported_notes(notes)] == [note.name]


def test_authority_gated_paid_dispatch_watcher_is_armed_by_default(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    unit = "blueprint-task-evaluation-policy-canary-dispatcher.path"
    enabled = "enabled"
    active = "inactive"

    def completed(argv, **kwargs):
        nonlocal enabled, active
        calls.append(tuple(argv))
        if argv[:2] == ["systemctl", "enable"]:
            enabled = "enabled"
        elif argv[:2] == ["systemctl", "restart"]:
            active = "active"
        stdout = ""
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = enabled + "\n"
        elif argv[:2] == ["systemctl", "is-active"]:
            stdout = active + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    observed = deploy._installed_path_unit_states([{"unit": unit}])
    restored = deploy._restore_installed_path_units(
        [{"unit": unit}],
        before=observed,
        arm_path_units=False,
        always_arm_authority_gated_units=(
            deploy.DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS
        ),
    )

    assert observed == {unit: {"enabled": "enabled", "state": "inactive"}}
    assert calls == [
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
        ("systemctl", "enable", unit),
        ("systemctl", "restart", unit),
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
    ]
    assert restored == [
        {
            "unit": unit,
            "before": {"enabled": "enabled", "state": "inactive"},
            "requested_intent": "arm_authority_gated_paid_dispatch",
            "after": {"enabled": "enabled", "state": "active"},
            "operator_freeze_preserved": False,
        }
    ]
def test_path_unit_state_restore_preserves_an_active_enabled_watcher(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        stdout = ""
        if argv[:2] == ["systemctl", "is-active"]:
            stdout = "active\n"
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = "enabled\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    restored = deploy._restore_installed_path_units(
        [
            {"unit": "blueprint-task-evaluation-launch-dispatcher.service"},
            {"unit": "blueprint-task-evaluation-launch-dispatcher.path"},
        ],
        before={
            "blueprint-task-evaluation-launch-dispatcher.path": {
                "enabled": "enabled",
                "state": "active",
            }
        },
        arm_path_units=False,
    )

    path_unit = "blueprint-task-evaluation-launch-dispatcher.path"
    assert calls == [
        ("systemctl", "enable", path_unit),
        ("systemctl", "restart", path_unit),
        ("systemctl", "is-enabled", path_unit),
        ("systemctl", "is-active", path_unit),
    ], "the oneshot service must never be started by the deploy itself"
    assert restored == [
        {
            "unit": path_unit,
            "before": {"enabled": "enabled", "state": "active"},
            "requested_intent": "preserve",
            "after": {"enabled": "enabled", "state": "active"},
            "operator_freeze_preserved": False,
        }
    ]


def test_path_unit_state_restore_failure_names_the_unit_and_verb(monkeypatch) -> None:
    def completed(argv, **kwargs):
        code = 1 if argv[:2] == ["systemctl", "restart"] else 0
        return subprocess.CompletedProcess(argv, code, stdout="", stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match="deploy_path_unit_state_restore_failed:"
        "blueprint-task-evaluation-launch-dispatcher.path:restart",
    ):
        deploy._restore_installed_path_units(
            [{"unit": "blueprint-task-evaluation-launch-dispatcher.path"}],
            before={
                "blueprint-task-evaluation-launch-dispatcher.path": {
                    "enabled": "enabled",
                    "state": "active",
                }
            },
            arm_path_units=False,
        )


def test_missing_fresh_path_is_already_restored_when_disable_and_stop_fail(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[:2] in (["systemctl", "disable"], ["systemctl", "stop"]):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="not found")
        stdout = "not-found\n" if argv[:2] == ["systemctl", "is-enabled"] else "inactive\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    unit = "blueprint-task-evaluation-policy-canary-dispatcher.path"

    restored = deploy._restore_installed_path_units(
        [{"unit": unit}], before={}, arm_path_units=False
    )

    assert restored[0]["after"] == {"enabled": "disabled", "state": "inactive"}
    assert restored[0]["operator_freeze_preserved"] is True
    assert calls[:4] == [
        ("systemctl", "disable", unit),
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
        ("systemctl", "stop", unit),
    ]


def test_a_watcher_that_is_not_waiting_after_requested_arm_blocks_the_deploy(
    monkeypatch,
) -> None:
    def completed(argv, **kwargs):
        stdout = ""
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = "enabled\n"
        if argv[:2] == ["systemctl", "is-active"]:
            stdout = "failed\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match="deploy_path_unit_active_state_mismatch:"
        "blueprint-task-evaluation-launch-dispatcher.path:failed:active",
    ):
        deploy._restore_installed_path_units(
            [{"unit": "blueprint-task-evaluation-launch-dispatcher.path"}],
            before={},
            arm_path_units=True,
        )


def test_path_unit_state_restore_preserves_an_enabled_operator_freeze(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        stdout = ""
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = "enabled\n"
        if argv[:2] == ["systemctl", "is-active"]:
            stdout = "inactive\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    unit = "blueprint-task-evaluation-launch-dispatcher.path"
    restored = deploy._restore_installed_path_units(
        [{"unit": unit}],
        before={unit: {"enabled": "enabled", "state": "inactive"}},
        arm_path_units=False,
    )

    assert calls == [
        ("systemctl", "enable", unit),
        ("systemctl", "stop", unit),
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
    ]
    assert restored[0]["after"] == {"enabled": "enabled", "state": "inactive"}
    assert restored[0]["operator_freeze_preserved"] is True


def test_fresh_path_unit_stays_disabled_until_explicit_arm(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        stdout = ""
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = "disabled\n"
        if argv[:2] == ["systemctl", "is-active"]:
            stdout = "inactive\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    unit = "blueprint-task-evaluation-launch-dispatcher.path"
    restored = deploy._restore_installed_path_units(
        [{"unit": unit}], before={}, arm_path_units=False
    )

    assert calls[:2] == [
        ("systemctl", "disable", unit),
        ("systemctl", "stop", unit),
    ]
    assert restored[0]["before"] == {"enabled": "disabled", "state": "inactive"}
    assert restored[0]["operator_freeze_preserved"] is True


def test_no_spend_preparation_watcher_arms_without_paid_dispatcher(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    enabled: dict[str, str] = {
        "blueprint-task-evaluation-launch-dispatcher.path": "disabled",
        "blueprint-task-evaluation-launch-preparation.path": "enabled",
    }
    active: dict[str, str] = {
        "blueprint-task-evaluation-launch-dispatcher.path": "inactive",
        "blueprint-task-evaluation-launch-preparation.path": "active",
    }

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        unit = argv[-1]
        stdout = ""
        if argv[:2] == ["systemctl", "is-enabled"]:
            stdout = enabled[unit] + "\n"
        elif argv[:2] == ["systemctl", "is-active"]:
            stdout = active[unit] + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    paid = "blueprint-task-evaluation-launch-dispatcher.path"
    preparation = "blueprint-task-evaluation-launch-preparation.path"
    restored = deploy._restore_installed_path_units(
        [{"unit": paid}, {"unit": preparation}],
        before={},
        arm_path_units=False,
        always_arm_units=(preparation,),
    )

    assert calls[:2] == [
        ("systemctl", "disable", paid),
        ("systemctl", "stop", paid),
    ]
    assert calls[4:6] == [
        ("systemctl", "enable", preparation),
        ("systemctl", "restart", preparation),
    ]
    assert restored == [
        {
            "unit": paid,
            "before": {"enabled": "disabled", "state": "inactive"},
            "requested_intent": "preserve",
            "after": {"enabled": "disabled", "state": "inactive"},
            "operator_freeze_preserved": True,
        },
        {
            "unit": preparation,
            "before": {"enabled": "disabled", "state": "inactive"},
            "requested_intent": "arm_no_spend",
            "after": {"enabled": "enabled", "state": "active"},
            "operator_freeze_preserved": False,
        },
    ]


def test_active_watcher_is_quiesced_before_release_surfaces_move(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        stdout = "inactive\n" if argv[1] == "is-active" else "enabled\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)
    unit = "blueprint-task-evaluation-launch-dispatcher.path"
    result = deploy._quiesce_active_path_units(
        {unit: {"enabled": "enabled", "state": "active"}}
    )

    assert calls == [
        ("systemctl", "stop", unit),
        ("systemctl", "is-enabled", unit),
        ("systemctl", "is-active", unit),
    ]
    assert result == [{"unit": unit, "state": "inactive"}]


def test_the_deploy_holds_the_lock_for_its_whole_duration(tmp_path: Path) -> None:
    """Not a check-then-deploy: a launch can start between the two.

    That is not hypothetical. On 2026-08-13 the check passed and the parallel
    lane acquired the lock 20 seconds later, mid-deploy.
    """

    import fcntl

    lock = _lock(tmp_path)
    observed: list[bool] = []

    with deploy._holding_paid_launch_locks([str(lock)]):
        # A launch trying to start now must be refused, which is what the
        # adapter's own non-blocking flock does.
        with lock.open("r", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed.append(True)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                observed.append(False)

    assert observed == [False], "a launch could start while the deploy held the lock"

    # And released afterwards, or the next launch could never start.
    with lock.open("r", encoding="utf-8") as probe:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)


def test_a_lock_held_by_a_launch_refuses_the_deploy_by_name(tmp_path: Path) -> None:
    import fcntl

    lock = _lock(tmp_path)
    with lock.open("r", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(deploy.ControlPlaneDeployError) as excinfo:
            with deploy._holding_paid_launch_locks([str(lock)]):
                pass
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    # Names the run, not the file.
    assert str(excinfo.value).startswith("deploy_refused_paid_launch_in_flight:")
    assert "vast_provider_run" in str(excinfo.value)


def test_an_absent_lock_is_not_created_by_the_deploy(tmp_path: Path) -> None:
    """The adapter creates it as the service account at 0600.

    A deploy running as root that created it first would leave a file the
    service can never open again, taking every paid lane down.
    """

    absent = tmp_path / "never-launched" / "vast_paid_launch.lock"

    with deploy._holding_paid_launch_locks([str(absent)]):
        pass

    assert not absent.exists()
    assert not absent.parent.exists()


def test_the_deploy_does_not_move_a_surface_while_refusing(tmp_path: Path, monkeypatch) -> None:
    """A launch caught taking its slot holds the gate; the deploy refuses, moving nothing."""
    import fcntl

    from blueprint_pipeline.vast_provider_adapter import vast_launch_gate_path

    moved: list[str] = []
    monkeypatch.setattr(
        deploy, "_move_source_checkout", lambda repo, commit: moved.append(commit)
    )
    monkeypatch.setattr(deploy, "PAID_LAUNCH_GATE_WAIT_SECONDS", 0)
    source = tmp_path / "source"
    source.mkdir()
    lock = _lock(tmp_path)
    gate = vast_launch_gate_path(lock)
    gate.touch()

    with gate.open("r", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(deploy.ControlPlaneDeployError):
            deploy.deploy_control_plane_commit(
                source_repo=source,
                source_commit="a" * 40,
                release_root=tmp_path / "releases",
                state_root=tmp_path / "state",
                active_link=tmp_path / "active",
                release_provenance=_provenance(tmp_path, "a" * 40),
                paid_launch_locks=(str(lock),),
            )
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    assert moved == []


def test_runtime_identity_drop_in_is_atomic_and_contains_no_credentials(
    tmp_path: Path,
) -> None:
    drop_in = tmp_path / "intake.service.d" / "90-deploy-identity.conf"

    receipt = deploy._install_intake_runtime_identity_drop_in(
        drop_in,
        source_repo=tmp_path / "repo",
        source_commit="b" * 40,
    )

    content = drop_in.read_text(encoding="utf-8")
    identity_env = drop_in.with_suffix(".env")
    env_content = identity_env.read_text(encoding="utf-8")
    assert content == (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Loaded after the base unit credential EnvironmentFile.\n"
        "[Service]\n"
        f"EnvironmentFile={identity_env}\n"
        "TimeoutStartSec=300s\n"
    )
    assert f"BLUEPRINT_SOURCE_COMMIT={'b' * 40}" in env_content
    assert f"BLUEPRINT_PIPELINE_REPO={tmp_path / 'repo'}" in env_content
    assert f"BLUEPRINT_PIPELINE_PYTHON={Path(sys.executable).absolute()}" in env_content
    assert f"PYTHONPATH={tmp_path / 'repo' / 'src'}" in env_content
    # Environment= loses to the base unit's EnvironmentFile= regardless of
    # drop-in order.  The regression is specifically that this must be a later
    # EnvironmentFile, not merely a later Environment directive.
    assert "\nEnvironment=" not in content
    assert "TOKEN" not in content + env_content
    assert "SECRET" not in content + env_content
    assert drop_in.stat().st_mode & 0o777 == 0o644
    assert identity_env.stat().st_mode & 0o777 == 0o644
    assert receipt["identity_environment_file"] == str(identity_env)
    assert receipt["pythonpath"] == str(tmp_path / "repo" / "src")
    assert receipt["timeout_start_seconds"] == 300
    assert receipt["credential_environment_file_opened"] is False
    assert receipt["credential_values_recorded"] is False


def test_intake_version_probe_rejects_a_stale_running_process(monkeypatch) -> None:
    monkeypatch.setattr(
        deploy.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            json.dumps(
                {"commit_proven": True, "source_commit": "stale-runtime"}
            ).encode("utf-8")
        ),
    )

    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match="deploy_intake_runtime_commit_mismatch:stale-runtime",
    ):
        deploy._verify_intake_runtime(
            "http://127.0.0.1:8765/api/live-pipeline/version",
            expected_commit="c" * 40,
        )


def test_intake_version_probe_is_loopback_only() -> None:
    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match="deploy_intake_version_url_not_loopback_http",
    ):
        deploy._verify_intake_runtime(
            "https://production.example/api/live-pipeline/version",
            expected_commit="c" * 40,
        )


def test_intake_version_probe_retries_while_the_restarted_server_binds(
    monkeypatch,
) -> None:
    calls = 0

    def delayed_server(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("not listening yet")
        return io.BytesIO(
            json.dumps(
                {"commit_proven": True, "source_commit": "c" * 40}
            ).encode("utf-8")
        )

    monkeypatch.setattr(deploy.urllib.request, "urlopen", delayed_server)

    result = deploy._verify_intake_runtime(
        "http://127.0.0.1:8765/api/live-pipeline/version",
        expected_commit="c" * 40,
        attempts=2,
        retry_delay_seconds=0,
    )

    assert calls == 2
    assert result["source_commit"] == "c" * 40


def test_deploy_holds_paid_slot_through_restart_and_runtime_probe(
    tmp_path: Path, monkeypatch
) -> None:
    """A second launch cannot enter during the newly added runtime checks."""

    import fcntl

    commit = "d" * 40
    source = tmp_path / "source"
    source.mkdir()
    active = tmp_path / "active"
    release = tmp_path / "release"
    release.mkdir()
    active.symlink_to(release, target_is_directory=True)
    lock = _lock(tmp_path)
    observed: list[str] = []

    monkeypatch.setattr(deploy, "_move_source_checkout", lambda *args: None)
    monkeypatch.setattr(
        deploy,
        "stage_task_evaluation_control_plane_release",
        lambda **kwargs: {
            "source_commit": commit,
            "release_path": str(release),
            "created_release_checkout": True,
        },
    )
    monkeypatch.setattr(deploy, "_surface_commit", lambda *args, **kwargs: commit)
    monkeypatch.setattr(
        deploy,
        "_install_intake_runtime_identity_drop_in",
        lambda *args, **kwargs: {"source_commit": commit},
    )
    monkeypatch.setattr(
        deploy,
        "_install_release_systemd_units",
        lambda **kwargs: [
            {"unit": "blueprint-task-evaluation-launch-dispatcher.service"},
            {"unit": "blueprint-task-evaluation-launch-dispatcher.path"},
        ],
    )
    monkeypatch.setattr(
        deploy,
        "_install_scene_object_discovery_runtime_directories",
        lambda: [{"path": "/runtime/scene-object-discoveries"}],
    )
    monkeypatch.setattr(
        deploy,
        "_install_episode_compilation_runtime_directories",
        lambda: [{"path": "/runtime/episode-compilations/pending"}],
    )
    monkeypatch.setattr(
        deploy,
        "_install_configured_controls_runtime_prerequisites",
        lambda: {"plan_root": "/etc/blueprint/configured-controls"},
    )
    monkeypatch.setattr(
        deploy,
        "_install_configured_controls_autostart_registry",
        lambda **kwargs: {
            "root": "/etc/blueprint/configured-controls-intents",
            "entry_count": 1,
        },
    )
    monkeypatch.setattr(
        deploy,
        "validate_splat_render_prerequisites",
        lambda **kwargs: {
            "entrypoints": {
                "node": "/runtime/node",
                "browser_root": "/runtime/browser",
                "browser": "/runtime/browser/chrome",
                "node_modules": "/runtime/node_modules",
            }
        },
    )
    monkeypatch.setattr(
        deploy,
        "_provision_scene_configuration_from_release",
        lambda **kwargs: {
            "status": "ready",
            "environment": {
                "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT": "/runtime/splat",
                "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT": "/runtime/toolchain",
                "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX": "s3://blueprint-production-inputs/coordinator-release-windows/",
                "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX": "s3://blueprint-production-inputs/task-evaluation-activations",
            },
        },
    )
    monkeypatch.setattr(
        deploy,
        "provision_production_cad_skill_sources",
        lambda _root: {
            "status": "ready",
            "sources": [
                {"id": "text-to-cad", "path": "/runtime/text-to-cad"},
                {"id": "multi-agent-cad", "path": "/runtime/Multi-Agent-CAD"},
            ],
        },
    )
    disk_runtime_receipt = {
        "status": "ready",
        "account": "blueprint",
        "repaired_paths": [str(tmp_path / "disk-reservations/.lock")],
    }

    reserve_calls: list[dict] = []
    observed_bytes: list[int] = []

    class Reservation:
        def receipt(self):
            return {"reservation_token": "deploy-test"}

        def observe(self, _bytes):
            observed_bytes.append(_bytes)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    monkeypatch.setattr(
        deploy,
        "_install_disk_reservation_runtime_prerequisites",
        lambda root: disk_runtime_receipt,
    )
    monkeypatch.setattr(
        deploy,
        "reserve_control_plane_disk",
        lambda *args, **kwargs: reserve_calls.append(kwargs) or Reservation(),
    )
    storage_pins_receipt = {
        "status": "ready",
        "path": "/var/lib/blueprint/pipeline-control-plane/storage-pins",
    }
    monkeypatch.setattr(
        deploy,
        "_install_storage_pins_runtime_root",
        lambda: storage_pins_receipt,
    )

    def assert_lock_held(stage: str):
        with lock.open("r", encoding="utf-8") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed.append(stage)

    monkeypatch.setattr(
        deploy,
        "_restart_units",
        lambda units: (assert_lock_held("restart") or [{"unit": units[0]}]),
    )
    monkeypatch.setattr(
        deploy,
        "_verify_intake_runtime",
        lambda *args, **kwargs: (
            assert_lock_held("runtime_probe")
            or {"commit_proven": True, "source_commit": commit}
        ),
    )
    monkeypatch.setattr(
        deploy,
        "_installed_path_unit_states",
        lambda installed: {
            "blueprint-task-evaluation-launch-dispatcher.path": {
                "enabled": "enabled",
                "state": "active",
            }
        },
    )
    monkeypatch.setattr(
        deploy,
        "_quiesce_active_path_units",
        lambda before: (
            assert_lock_held("path_quiesce")
            or [
                {
                    "unit": "blueprint-task-evaluation-launch-dispatcher.path",
                    "state": "inactive",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        deploy,
        "_restore_installed_path_units",
        lambda installed, **kwargs: (
            assert_lock_held("path_activation")
            or [
                {
                    "unit": entry["unit"],
                    "before": {"enabled": "enabled", "state": "active"},
                    "requested_intent": "preserve",
                    "after": {"enabled": "enabled", "state": "active"},
                    "operator_freeze_preserved": False,
                }
                for entry in installed
                if str(entry["unit"]).endswith(".path")
            ]
        ),
    )

    receipt = deploy.deploy_control_plane_commit(
        source_repo=source,
        source_commit=commit,
        release_root=tmp_path / "releases",
        state_root=tmp_path / "state",
        active_link=active,
        release_provenance=_provenance(tmp_path, commit),
        paid_launch_locks=(str(lock),),
        intake_runtime_drop_in=tmp_path / "drop-in",
        scene_configuration_environment_file=tmp_path / "scene-runtime.env",
        scene_configuration_runtime_root=tmp_path / "system-runtimes",
        scene_preparation_bootstrap_file=tmp_path / "absent-bootstrap.json",
        controls_autoprovision_bootstrap_file=tmp_path / "absent-controls-bootstrap.json",
        disk_reservation_root=tmp_path / "disk-reservations",
    )

    assert observed == [
        "path_quiesce",
        "restart",
        "runtime_probe",
        "path_activation",
    ]
    assert receipt["intake_runtime"]["source_commit"] == commit
    assert receipt["disk_reservation_runtime"] == disk_runtime_receipt
    assert receipt["disk_reservation"] == {"reservation_token": "deploy-test"}
    # The source is not a git checkout here, so the estimate falls back to the
    # role's footprint; the staged release checkout is what the deploy observed.
    assert receipt["disk_reservation_estimate"]["basis"] == "declared_default"
    assert reserve_calls[0]["expected_bytes"] == receipt["disk_reservation_estimate"]["bytes"]
    assert reserve_calls[0]["workload"] == "control_plane_release"
    assert len(observed_bytes) == 1 and observed_bytes[0] > 0
    assert receipt["storage_pins_runtime"] == storage_pins_receipt
    assert receipt["restarted_units"][0]["unit"] == deploy.DEFAULT_RESTART_UNITS[0]
    assert receipt["installed_systemd_units"][0]["unit"] == (
        "blueprint-task-evaluation-launch-dispatcher.service"
    )
    assert receipt["episode_compilation_runtime_directories"] == [
        {"path": "/runtime/episode-compilations/pending"}
    ]
    assert receipt["configured_controls_runtime"] == {
        "plan_root": "/etc/blueprint/configured-controls"
    }
    assert receipt["configured_controls_autostart_registry"] == {
        "root": "/etc/blueprint/configured-controls-intents",
        "entry_count": 1,
    }
    assert receipt["activated_path_units"] == [
        {
            "unit": "blueprint-task-evaluation-launch-dispatcher.path",
            "enabled": "enabled",
            "state": "active",
        }
    ]
    assert receipt["release_provenance"]["git_sha"] == commit
    assert Path(receipt["release_provenance"]["path"]).stat().st_mode & 0o777 == 0o440
    assert receipt["release_retirement"]["status"] == "not_requested"
    assert receipt["release_retirement"]["retired_bytes"] == 0
    assert not (tmp_path / "state/release-retention/latest-deploy-retirement.json").exists()


def test_disk_reservation_runtime_repairs_root_owned_ledger_and_reports_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "disk-reservations"
    root.mkdir(mode=0o755)
    lock = root / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o644)
    blueprint_gid = 2401
    ownership = {
        str(root): (0, 0),
        str(lock): (0, 0),
    }
    chowns: list[tuple[str, int, int]] = []

    def chown(path: Path, uid: int, gid: int) -> None:
        chowns.append((str(path), uid, gid))
        ownership[str(path)] = (uid, gid)

    def stat_reader(path: Path) -> SimpleNamespace:
        metadata = path.stat()
        # Inodes the installer creates itself start root:root, as on the host.
        uid, gid = ownership.get(str(path), (0, 0))
        return SimpleNamespace(st_uid=uid, st_gid=gid, st_mode=metadata.st_mode)

    monkeypatch.setattr(
        deploy,
        "_service_account_ids",
        lambda account: (3101, blueprint_gid) if account == "blueprint" else None,
    )

    receipt = deploy._install_disk_reservation_runtime_prerequisites(
        root,
        chown=chown,
        stat_reader=stat_reader,
    )

    history = root / "history"
    assert chowns == [
        (str(root), 0, blueprint_gid),
        (str(lock), 0, blueprint_gid),
        (str(history), 0, blueprint_gid),
    ]
    assert root.stat().st_mode & 0o7777 == 0o2770
    assert lock.stat().st_mode & 0o777 == 0o660
    assert receipt == {
        "status": "ready",
        "account": "blueprint",
        "repaired_paths": [str(root), str(lock), str(history)],
        "installed": [
            {
                "kind": "directory",
                "path": str(root),
                "owner": "root",
                "group": "blueprint",
                "owner_uid": 0,
                "owner_gid": blueprint_gid,
                "mode": "2770",
            },
            {
                "kind": "lock",
                "path": str(lock),
                "owner": "root",
                "group": "blueprint",
                "owner_uid": 0,
                "owner_gid": blueprint_gid,
                "mode": "0660",
            },
            {
                "kind": "history_directory",
                "path": str(history),
                "owner": "root",
                "group": "blueprint",
                "owner_uid": 0,
                "owner_gid": blueprint_gid,
                "mode": "2770",
            },
        ],
    }

    repeated = deploy._install_disk_reservation_runtime_prerequisites(
        root,
        chown=chown,
        stat_reader=stat_reader,
    )
    assert repeated["repaired_paths"] == []
    assert len(chowns) == 3


def test_deploy_installs_the_footprint_history_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root (deploy) and the runtime account both append footprint samples, so the
    history directory is installed and verified like the ledger itself."""

    root = tmp_path / "disk-reservations"
    history = root / "history"
    history.mkdir(parents=True, mode=0o700)
    ownership: dict[str, tuple[int, int]] = {}
    chowns: list[tuple[str, int, int]] = []

    def chown(path: Path, uid: int, gid: int) -> None:
        chowns.append((str(path), uid, gid))
        ownership[str(path)] = (uid, gid)

    def stat_reader(path: Path) -> SimpleNamespace:
        uid, gid = ownership.get(str(path), (0, 0))
        return SimpleNamespace(st_uid=uid, st_gid=gid, st_mode=path.stat().st_mode)

    monkeypatch.setattr(deploy, "_service_account_ids", lambda account: (3101, 2401))

    receipt = deploy._install_disk_reservation_runtime_prerequisites(
        root, chown=chown, stat_reader=stat_reader
    )

    assert history.stat().st_mode & 0o7777 == 0o2770
    assert (str(history), 0, 2401) in chowns and str(history) in receipt["repaired_paths"]
    assert receipt["installed"][-1] == {
        "kind": "history_directory", "path": str(history), "owner": "root", "group": "blueprint",
        "owner_uid": 0, "owner_gid": 2401, "mode": "2770",
    }
    history.rmdir()
    history.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(deploy.ControlPlaneDeployError) as refused:
        deploy._install_disk_reservation_runtime_prerequisites(root, chown=chown, stat_reader=stat_reader)
    assert str(refused.value) == "deploy_disk_reservation_runtime_symlink:history"


def test_disk_ledger_refusals_name_the_item_never_its_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is a typed code: it names the ledger item, not where it lives."""

    monkeypatch.setattr(deploy, "_service_account_ids", lambda account: (3101, 2401))

    def installed_as(chown, stat_reader, *, root: Path) -> str:
        with pytest.raises(deploy.ControlPlaneDeployError) as refused:
            deploy._install_disk_reservation_runtime_prerequisites(
                root, chown=chown, stat_reader=stat_reader
            )
        assert str(tmp_path) not in str(refused.value)
        return str(refused.value)

    def root_owned(path: Path) -> SimpleNamespace:
        return SimpleNamespace(st_uid=0, st_gid=0, st_mode=path.stat().st_mode)

    symlinked = tmp_path / "symlinked-lock"
    symlinked.mkdir()
    (symlinked / ".lock").symlink_to(tmp_path / "elsewhere")
    assert installed_as(lambda *_a: None, root_owned, root=symlinked) == (
        "deploy_disk_reservation_runtime_symlink:lock"
    )
    # The group repair does not take: the readback names the first item it checks.
    assert installed_as(lambda *_a: None, root_owned, root=tmp_path / "unrepaired") == (
        "deploy_disk_reservation_runtime_readback_mismatch:directory"
    )

    def refuse_history(path: Path, _uid: int, _gid: int) -> None:
        if path.name == "history":
            raise PermissionError(1, "Operation not permitted", str(path))

    assert installed_as(refuse_history, root_owned, root=tmp_path / "unwritable") == (
        "deploy_disk_reservation_runtime_install_failed:history"
    )


def test_a_redeploy_that_creates_nothing_records_no_release_footprint(tmp_path: Path) -> None:
    commit = "a" * 40
    runtime_root = tmp_path / "system-runtimes"
    (runtime_root / "splat-render" / commit).mkdir(parents=True)  # an earlier deploy's tree
    release = tmp_path / "release"
    release.mkdir()
    (release / "module.py").write_bytes(b"m" * 4096)
    before = deploy._release_runtime_trees(runtime_root, commit)

    def created(checkout: bool):
        return deploy._created_release_usage(
            created_release_checkout=checkout, release_path=release, runtime_root=runtime_root,
            commit=commit, runtime_trees_before=before)

    # Nothing new on disk: no sample, rather than a zero that drags the p95 down.
    assert created(False) is None
    tree = runtime_root / "scene-configuration" / commit
    tree.mkdir(parents=True)
    (tree / "toolchain.bin").write_bytes(b"t" * 4096)
    assert created(False).allocated_bytes >= 4096
    assert created(True).allocated_bytes >= created(False).allocated_bytes + 4096


def test_unreadable_created_release_is_an_incomplete_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline.control_plane_disk_usage import TreeUsage

    release = tmp_path / "release"
    release.mkdir()
    monkeypatch.setattr(
        deploy, "tree_usage",
        lambda _path: TreeUsage(allocated_bytes=4096, unreadable=1),
    )
    usage = deploy._created_release_usage(
        created_release_checkout=True, release_path=release,
        runtime_root=tmp_path / "runtime", commit="a" * 40,
        runtime_trees_before=set(),
    )
    assert usage is not None and usage.unreadable == 1
    observed = []
    reservation = SimpleNamespace(observe=observed.append, measurement_incomplete=False)
    deploy._observe_created_release_usage(reservation, usage)
    assert observed == [4096]
    assert reservation.measurement_incomplete is True


def _git_repo_with_commit(tmp_path: Path) -> Path:
    source = tmp_path / "estimate-source"
    (source / "nested").mkdir(parents=True)
    (source / "module.py").write_bytes(b"x" * 10_000)
    (source / "nested" / "data.json").write_bytes(b"{}\n")
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture"],
    ):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    return source


def _head(source: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_deploy_reserves_a_git_tree_estimate_not_a_flat_two_gib(tmp_path: Path) -> None:
    source = _git_repo_with_commit(tmp_path)
    estimate = deploy._release_footprint_estimate(source, _head(source))
    assert estimate["basis"] == "git_tree_estimate"
    assert estimate["bytes"] >= estimate["tree_bytes"]
    assert estimate["bytes"] < 2 * 1024**3
    assert (estimate["tree_bytes"], estimate["file_count"]) == (10_003, 2)
    # Blob bytes plus a quarter for block rounding, a block per file, and 256 MiB
    # for the index and the runtime trees staged beside the release.
    assert estimate["bytes"] == -(-10_003 * 5 // 4) + 2 * 4096 + 256 * 1024**2


def test_release_estimate_falls_back_to_the_role_footprint_without_a_git_tree(
    tmp_path: Path,
) -> None:
    estimate = deploy._release_footprint_estimate(
        tmp_path / "not-a-checkout", "a" * 40, reservation_root=tmp_path / "ledger"
    )
    assert estimate["basis"] == "declared_default"
    assert estimate["bytes"] == 2 * 1024**3
    assert estimate["tree_bytes"] is None and estimate["file_count"] is None


def test_storage_pins_runtime_repairs_root_owned_directory_and_reports_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "storage-pins"
    root.mkdir(mode=0o700)
    blueprint_uid = 3101
    blueprint_gid = 3102
    ownership = {str(root): (0, 0)}
    chowns: list[tuple[str, int, int]] = []

    def chown(path: Path, uid: int, gid: int) -> None:
        chowns.append((str(path), uid, gid))
        ownership[str(path)] = (uid, gid)

    def stat_reader(path: Path) -> SimpleNamespace:
        metadata = path.stat()
        uid, gid = ownership[str(path)]
        return SimpleNamespace(st_uid=uid, st_gid=gid, st_mode=metadata.st_mode)

    monkeypatch.setattr(
        deploy,
        "_service_account_ids",
        lambda account: (
            (blueprint_uid, blueprint_gid) if account == "blueprint" else None
        ),
    )

    receipt = deploy._install_storage_pins_runtime_root(
        pins_root=root,
        chown=chown,
        stat_reader=stat_reader,
    )

    assert chowns == [(str(root), blueprint_uid, blueprint_gid)]
    assert root.stat().st_mode & 0o777 == 0o750
    assert receipt == {
        "status": "ready",
        "path": str(root),
        "account": "blueprint",
        "owner_uid": blueprint_uid,
        "owner_gid": blueprint_gid,
        "mode": "0750",
        "repaired": True,
    }

    repeated = deploy._install_storage_pins_runtime_root(
        pins_root=root,
        chown=chown,
        stat_reader=stat_reader,
    )
    assert repeated["repaired"] is False
    assert len(chowns) == 1


def test_episode_compilation_directory_retry_skips_correct_privileged_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "episode-compilations"
    directories = (root, *(root / state for state in ("pending", "processing", "completed", "blocked")))
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o750)
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (os.getuid(), os.getgid())
    )

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("already-correct directory must not be mutated")

    monkeypatch.setattr(deploy.os, "chown", unexpected_mutation)
    monkeypatch.setattr(Path, "chmod", unexpected_mutation)

    receipts = deploy._install_episode_compilation_runtime_directories(
        directories=tuple(str(path) for path in directories), account="test-service"
    )

    assert [row["path"] for row in receipts] == [str(path) for path in directories]
    assert all(row["mode"] == "0750" for row in receipts)


def test_deploy_restores_dispatcher_access_to_old_gc_stranded_queue(tmp_path, monkeypatch):
    queue = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canary-dispatches"
    expected = {queue, *(f"{queue}/{state}" for state in
        ("pending", "processing", "completed", "blocked", "stranded"))}
    assert expected <= set(deploy.DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES)
    stranded = tmp_path / "stranded"
    stranded.mkdir(mode=0o700)
    envelope = stranded / "retained.json"
    envelope.write_bytes(b"immutable queue envelope")
    envelope.chmod(0o440)
    monkeypatch.setattr(deploy, "_service_account_ids", lambda _: (os.getuid(), os.getgid()))
    deploy._install_episode_compilation_runtime_directories(directories=(str(stranded),))
    assert stranded.stat().st_mode & 0o777 == 0o750
    assert envelope.read_bytes() == b"immutable queue envelope"
    assert envelope.stat().st_mode & 0o777 == 0o440


def test_configured_controls_prerequisites_skip_correct_cross_owner_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    secret = tmp_path / "submit-secret"
    secret.write_bytes(b"not-read-by-installer")
    service_uid = 1234
    service_gid = 2345
    root_uid = 0
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (service_uid, service_gid)
    )

    class Metadata:
        def __init__(self, uid: int, gid: int, mode: int) -> None:
            self.st_uid = uid
            self.st_gid = gid
            self.st_mode = mode

    def metadata(path: Path) -> Metadata:
        return (
            Metadata(service_uid, service_gid, 0o40750)
            if path == root
            else Metadata(root_uid, service_gid, 0o100440)
        )

    def unexpected(*_args: object) -> None:
        raise AssertionError("already-correct cross-owner state must not mutate")

    receipt = deploy._install_configured_controls_runtime_prerequisites(
        plan_root=str(root),
        webapp_secret=str(secret),
        account="blueprint",
        root_uid=root_uid,
        chown=unexpected,
        stat_reader=metadata,
    )
    assert receipt["secret_bytes_read"] is False


def test_configured_controls_prerequisites_repair_wrong_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    secret = tmp_path / "submit-secret"
    secret.write_bytes(b"not-read-by-installer")
    service_uid, service_gid, root_uid = 1234, 2345, 0
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (service_uid, service_gid)
    )
    calls: list[tuple[str, int, int]] = []
    reads = {root: 0, secret: 0}

    class Metadata:
        def __init__(self, uid: int, gid: int, mode: int) -> None:
            self.st_uid = uid
            self.st_gid = gid
            self.st_mode = mode

    def metadata(path: Path) -> Metadata:
        reads[path] += 1
        if reads[path] == 1:
            return Metadata(999, 999, 0o40777 if path == root else 0o100400)
        return (
            Metadata(service_uid, service_gid, 0o40750)
            if path == root
            else Metadata(root_uid, service_gid, 0o100440)
        )

    monkeypatch.setattr(Path, "chmod", lambda path, mode: calls.append((str(path), mode, -1)))
    receipt = deploy._install_configured_controls_runtime_prerequisites(
        plan_root=str(root),
        webapp_secret=str(secret),
        account="blueprint",
        root_uid=root_uid,
        chown=lambda path, uid, gid: calls.append((str(path), uid, gid)),
        stat_reader=metadata,
    )
    assert (str(root), service_uid, service_gid) in calls
    assert (str(secret), root_uid, service_gid) in calls
    assert (str(root), 0o750, -1) in calls
    assert (str(secret), 0o440, -1) in calls
    assert receipt["secret_bytes_read"] is False


def test_autostart_registry_atomically_refreshes_on_consecutive_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "configured-controls-intents"
    first_source = tmp_path / "first.json"
    second_source = tmp_path / "second.json"
    adoption_source = tmp_path / "adoption.json"
    identity = {
        "team_namespace": "blueprint-adp",
        "scene_id": "interiorgs-839873",
        "task_id": "scene-839873-mug-planar-push",
    }
    first = {
        **identity,
        "expected_production_commit": "a" * 40,
        "configuration_adoption": {"mode": "same_commit_automatic"},
        "intent_digest": "sha256:" + "1" * 64,
    }
    second = {
        **identity,
        "expected_production_commit": "b" * 40,
        "configuration_adoption": {"mode": "same_commit_automatic"},
        "intent_digest": "sha256:" + "2" * 64,
    }
    adoption = {
        **identity,
        "expected_production_commit": "b" * 40,
        "configuration_adoption": {
            "mode": "explicit_terminal_adoption",
            "source_launch_id": "scene-839873-2deff449-r1",
        },
        "intent_digest": "sha256:" + "3" * 64,
    }
    first_source.write_text(json.dumps(first), encoding="utf-8")
    second_source.write_text(json.dumps(second), encoding="utf-8")
    adoption_source.write_text(json.dumps(adoption), encoding="utf-8")
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (os.getuid(), os.getgid())
    )
    monkeypatch.setattr(
        deploy,
        "validate_configured_controls_autostart_intent",
        lambda value: dict(value),
    )

    first_receipt = deploy._install_configured_controls_autostart_registry(
        intent_root=str(root),
        intent_sources=(str(first_source.resolve()),),
        source_commit="a" * 40,
        account="test-service",
        root_uid=os.getuid(),
    )
    second_receipt = deploy._install_configured_controls_autostart_registry(
        intent_root=str(root),
        intent_sources=(
            str(second_source.resolve()),
            str(adoption_source.resolve()),
        ),
        source_commit="b" * 40,
        account="test-service",
        root_uid=os.getuid(),
    )

    automatic_entry = next(
        row
        for row in second_receipt["entries"]
        if row["configuration_adoption_mode"] == "same_commit_automatic"
    )
    adoption_entry = next(
        row
        for row in second_receipt["entries"]
        if row["configuration_adoption_mode"] == "explicit_terminal_adoption"
    )
    destination = Path(automatic_entry["path"])
    assert first_receipt["entry_count"] == 1
    assert second_receipt["entry_count"] == 2
    assert automatic_entry["replaced_previous_sha256"] == (
        first_receipt["entries"][0]["sha256"]
    )
    assert automatic_entry["path"] != adoption_entry["path"]
    assert json.loads(destination.read_text(encoding="utf-8")) == second


def test_deploy_refuses_mismatched_promotion_before_moving_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved: list[str] = []
    monkeypatch.setattr(
        deploy, "_move_source_checkout", lambda repo, commit: moved.append(commit)
    )
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(
        deploy.ControlPlaneDeployError, match="deploy_release_provenance_mismatch"
    ):
        deploy.deploy_control_plane_commit(
            source_repo=source,
            source_commit="a" * 40,
            release_root=tmp_path / "releases",
            state_root=tmp_path / "state",
            active_link=tmp_path / "active",
            release_provenance=_provenance(tmp_path, "b" * 40),
            paid_launch_locks=(),
        )

    assert moved == []


def test_scene_runtime_failure_blocks_before_source_or_active_release_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "d" * 40
    source = tmp_path / "source"
    source.mkdir()
    original = tmp_path / "original-release"
    original.mkdir()
    active = tmp_path / "active"
    active.symlink_to(original)
    staged = tmp_path / "staged-release"
    staged.mkdir()
    moved: list[str] = []
    watcher = "blueprint-scene-object-discovery.path"
    restored: list[dict[str, object]] = []
    monkeypatch.setattr(
        deploy,
        "_install_release_provenance",
        lambda **kwargs: {"git_sha": commit},
    )
    monkeypatch.setattr(
        deploy,
        "provision_production_cad_skill_sources",
        lambda *_args, **_kwargs: {
            "sources": [
                {"id": "text-to-cad", "path": str(tmp_path / "text-to-cad")},
                {
                    "id": "multi-agent-cad",
                    "path": str(tmp_path / "Multi-Agent-CAD"),
                },
            ]
        },
    )
    monkeypatch.setattr(
        deploy,
        "_installed_path_unit_states",
        lambda _units: {watcher: {"enabled": "enabled", "state": "active"}},
    )
    monkeypatch.setattr(
        deploy,
        "_quiesce_active_path_units",
        lambda _before: [{"unit": watcher, "state": "inactive"}],
    )
    monkeypatch.setattr(
        deploy,
        "_restore_installed_path_units",
        lambda installed, **kwargs: restored.append(
            {"installed": installed, **kwargs}
        ) or [],
    )
    monkeypatch.setattr(
        deploy,
        "stage_task_evaluation_control_plane_release",
        lambda **kwargs: {
            "source_commit": commit,
            "release_path": str(staged),
            "created_release_checkout": True,
        },
    )
    monkeypatch.setattr(
        deploy,
        "validate_splat_render_prerequisites",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("splat_render_prerequisite_manifest_invalid")
        ),
    )
    monkeypatch.setattr(
        deploy,
        "_move_source_checkout",
        lambda *_args, **_kwargs: moved.append("source"),
    )

    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match=(
            "deploy_scene_configuration_runtime_invalid:"
            "splat_render_prerequisite_manifest_invalid"
        ),
    ):
        deploy.deploy_control_plane_commit(
            source_repo=source,
            source_commit=commit,
            release_root=tmp_path / "releases",
            state_root=tmp_path / "state",
            active_link=active,
            release_provenance=_provenance(tmp_path, commit),
            paid_launch_locks=(),
        )

    assert moved == []
    assert active.resolve() == original
    assert restored == [
        {
            "installed": [
                {"unit": unit}
                for unit in deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
                if unit.endswith((".path", ".timer"))
            ],
            "before": {watcher: {"enabled": "enabled", "state": "active"}},
            "arm_path_units": False,
            "always_arm_units": (),
            "held_units": {},
            "defer_start_verification": True,
        }
    ]


def test_the_receipt_records_every_slot_it_was_exclusive_with(tmp_path: Path) -> None:
    """A receipt that under-reports its own guarantee misleads its reader.

    The lock is an N-slot semaphore; recording the single base path the caller
    named would say "1 lock checked" for a deploy that actually held three.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    held = deploy._expanded_slots([str(base)])

    assert held == vast_launch_lock_paths(base)
    assert len(held) > 1, "the semaphore should expand to more than the base path"
    assert held[0] == base


def test_a_deploy_receipt_never_claims_a_lane_that_did_not_run(tmp_path) -> None:
    """The receipt summary must report the claim it actually installed.

    An iteration release is deployed without the canonical Full Test Lane and
    its provenance file says so. The deploy receipt that summarises that file
    hardcoded canonical_full_lane_verified=True, so every reader of a receipt
    -- including anyone deciding whether a release may carry a promotion-grade
    claim -- was told the lane had verified a release it never saw.
    """

    commit = "d" * 40
    state_root = tmp_path / "state"
    iteration_payload, iteration_receipt = _iteration_provenance(commit)

    installed = deploy._install_release_provenance(
        payload=iteration_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=iteration_receipt,
    )

    assert installed["canonical_full_lane_verified"] is False
    assert installed["promotion_eligible"] is False
    assert installed["provenance_status"] == "iteration"
    # The summary agrees with the bytes it summarises.
    written = json.loads(
        (state_root / commit / deploy.DEPLOY_RELEASE_PROVENANCE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert (
        installed["canonical_full_lane_verified"]
        is written["claim_boundary"]["canonical_full_lane_verified"]
    )

    verified_payload, verified_receipt = _verified_provenance(commit)
    promoted = deploy._install_release_provenance(
        payload=verified_payload,
        state_root=state_root,
        source_commit=commit,
        receipt=verified_receipt,
    )
    assert promoted["canonical_full_lane_verified"] is True
    assert promoted["promotion_eligible"] is True
    assert promoted["provenance_status"] == "verified"


# --- promotion proof the service account can actually read -------------------
#
# Deploy runs as root; every service that consumes the promotion proof runs as
# `blueprint`. The installer set mode 0440 but never set ownership, so on the
# live control plane on 2026-08-29 every `deploy-release-provenance.json` was
# `root:root 0440` and `sudo -u blueprint cat` returned Permission denied --
# alongside 304 root-owned directories with no `o+x`, which hide readable files
# beneath them. Nothing failed: the deploy passed, and the reader's side was
# simply never asserted.


def _provenance_tree(tmp_path: Path) -> tuple[Path, Path]:
    commit_dir = tmp_path / "state" / ("c" * 40)
    commit_dir.mkdir(parents=True)
    destination = commit_dir / deploy.DEPLOY_RELEASE_PROVENANCE_NAME
    destination.write_bytes(b"{}")
    destination.chmod(0o440)
    return commit_dir, destination


def test_provenance_access_gate_passes_when_the_service_account_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _commit_dir, destination = _provenance_tree(tmp_path)
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (os.getuid(), os.getgid())
    )

    receipt = deploy._install_release_provenance_access(destination, None)

    assert receipt["status"] == "readable"
    assert str(destination) in receipt["verified_paths"]


def test_provenance_access_gate_fails_closed_when_the_grant_does_not_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant that silently no-ops must fail the deploy, not pass it.

    This is the regression that matters: the gate re-derives readability from
    the installed inode instead of trusting that the chown happened, so the
    installer cannot quietly return to writing proof nobody can open.
    """

    _commit_dir, destination = _provenance_tree(tmp_path)
    foreign_uid, foreign_gid = os.getuid() + 4242, os.getgid() + 4242
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (foreign_uid, foreign_gid)
    )

    with pytest.raises(deploy.ControlPlaneDeployError) as excinfo:
        deploy._install_release_provenance_access(
            destination, None, chown=lambda *_args, **_kwargs: None
        )

    assert str(excinfo.value).startswith(
        "deploy_release_provenance_unreadable_by_service_account:"
    )


def test_untraversable_parent_directory_is_reported_as_a_blocker(
    tmp_path: Path,
) -> None:
    """The 304-directory shape: a readable file nobody can reach."""

    commit_dir, destination = _provenance_tree(tmp_path)
    commit_dir.chmod(0o600)  # readable, but not traversable
    try:
        blocker = deploy._service_account_read_blocker(
            destination, owner_uid=os.getuid(), owner_gid=os.getgid()
        )
    finally:
        commit_dir.chmod(0o750)

    assert blocker == f"untraversable_directory:{commit_dir}"


def test_unreadable_provenance_file_is_reported_as_a_blocker(
    tmp_path: Path,
) -> None:
    _commit_dir, destination = _provenance_tree(tmp_path)
    destination.chmod(0o000)
    try:
        blocker = deploy._service_account_read_blocker(
            destination, owner_uid=os.getuid(), owner_gid=os.getgid()
        )
    finally:
        destination.chmod(0o440)

    assert blocker == f"unreadable_file:{destination}"


def test_readable_provenance_reports_no_blocker(tmp_path: Path) -> None:
    _commit_dir, destination = _provenance_tree(tmp_path)

    assert (
        deploy._service_account_read_blocker(
            destination, owner_uid=os.getuid(), owner_gid=os.getgid()
        )
        is None
    )


def test_grant_moves_the_group_and_never_the_owning_uid(tmp_path: Path) -> None:
    """0440 root:blueprint keeps the reader unable to rewrite its own proof.

    Chowning the receipt to the service account would let the consumer chmod
    the file that authorises it, so the grant must only ever move the group.
    """

    _commit_dir, destination = _provenance_tree(tmp_path)
    calls: list[tuple[str, int, int]] = []

    def _record(path: object, uid: int, gid: int) -> None:
        calls.append((str(path), uid, gid))

    deploy._grant_service_account_read(
        [destination], owner_gid=os.getgid() + 4242, chown=_record
    )

    assert calls, "expected the group to be moved"
    assert all(uid == -1 for _path, uid, _gid in calls)


def test_missing_service_account_is_not_applicable_rather_than_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Developer and CI hosts have no `blueprint` user; deploy must still run."""

    _commit_dir, destination = _provenance_tree(tmp_path)
    monkeypatch.setattr(deploy, "_service_account_ids", lambda _account: None)

    receipt = deploy._install_release_provenance_access(destination, None)

    assert receipt["status"] == "not_applicable_no_service_account"


def test_installer_records_the_service_account_access_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate runs inside the installer, so no call site can skip it."""

    commit = "d" * 40
    payload, receipt = _verified_provenance(commit)
    monkeypatch.setattr(
        deploy, "_service_account_ids", lambda _account: (os.getuid(), os.getgid())
    )

    installed = deploy._install_release_provenance(
        payload=payload,
        state_root=tmp_path / "state",
        source_commit=commit,
        receipt=receipt,
    )

    assert installed["service_account_access"]["status"] == "readable"



def _protection_sources(tmp_path: Path, **overrides: object):
    """Typed protection sources laid out like the host, under ``tmp_path``."""

    from blueprint_pipeline.control_plane_release_leases import LIVE_QUEUE_STATES

    control_plane = tmp_path / "var/lib/blueprint/pipeline-control-plane"
    for queue, states in LIVE_QUEUE_STATES.items():
        for state in states:
            (control_plane / queue / state).mkdir(parents=True, exist_ok=True)
    for name in ("standing-authorizations", "task-evaluation-release-retention-bindings"):
        (control_plane / name).mkdir(parents=True, exist_ok=True)
    profiles = tmp_path / "etc/blueprint/task-evaluation-launch-profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "control_plane_root": control_plane,
        "profile_dir": profiles,
        "standing_authorization_dir": control_plane / "standing-authorizations",
        "binding_root": control_plane / "task-evaluation-release-retention-bindings",
        "lease_root": control_plane / "release-leases",
        "config_files": (),
        "intent_root": control_plane / "task-evaluation-scene-intents",
        "launch_run_root": control_plane / "task-evaluation-launch-runs",
    }
    values.update(overrides)
    return deploy.ProtectionSources(**values)


def _release_trees(releases: Path, ages: dict[str, float], *, now: float) -> None:
    for commit, age in ages.items():
        directory = releases / commit
        directory.mkdir(parents=True)
        (directory / "renderer").write_bytes(b"retained renderer")
        stamp = now - age
        os.utime(directory / "renderer", (stamp, stamp))
        os.utime(directory, (stamp, stamp))


def test_release_retirement_is_skipped_without_protection_sources_and_applied_with_them(
    tmp_path: Path,
) -> None:
    """The explicit compatibility helper retires only provably unused trees."""

    import time as _time

    releases = tmp_path / "releases"
    runtimes = tmp_path / "runtimes"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    current, superseded = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400}, now=_time.time())
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)

    def retire(sources):
        return deploy._retire_superseded_release_trees(
            release_root=releases,
            runtime_root=runtimes,
            active_link=active,
            current_commit=current,
            protection_sources=sources,
            keep_last=1,
            proc_root=no_processes,
        )

    # A live launch names a profile this host cannot read: nothing is provable.
    unreadable = _protection_sources(tmp_path / "unreadable")
    (unreadable.control_plane_root / "task-evaluation-launches/pending/launch.json").write_text(
        json.dumps({"launch_profile_id": "never-published"}), encoding="utf-8"
    )
    skipped = retire(unreadable)
    assert skipped["status"] == "skipped"
    assert skipped["blockers"] == ["release_protection_profile_missing:never-published"]
    assert skipped["alerts"] == [
        "release_retirement_blocked:release_protection_profile_missing:never-published"
    ]
    assert (releases / superseded).is_dir()

    # Without the publishers' lock root the deploy cannot exclude a publisher.
    unlocked = retire(
        _protection_sources(tmp_path / "unlocked", control_plane_root=tmp_path / "absent")
    )
    assert unlocked["status"] == "blocked"
    assert unlocked["blockers"] == ["release_reference_lock_root_unavailable"]
    assert (releases / superseded).is_dir()

    applied = retire(_protection_sources(tmp_path / "readable"))
    assert applied["status"] == "applied"
    assert applied["retired_commits"] == [superseded]
    assert applied["skipped"] == []
    assert applied["alerts"] == []
    assert not (releases / superseded).exists()
    assert (releases / current).is_dir()


def test_deploy_retirement_honors_required_historical_evidence_binding(tmp_path: Path) -> None:
    """A legacy binding gets a lease on the first deploy and protects until it lapses."""
    import time

    releases, runtimes = tmp_path / "releases", tmp_path / "runtimes"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    now = time.time()
    current, retained = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, retained: 10 * 86_400}, now=now)
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    sources = _protection_sources(tmp_path)
    binding = sources.binding_root / "sam-prefix.json"
    binding.write_text(json.dumps({
        "schema_version": "task_evaluation_release_retention_binding.v1",
        "status": "required", "source_commit": retained,
        "reason": "Completed prefix replay reopens the original renderer release.",
    }))
    before = binding.read_bytes()

    def deploy_at(moment: float) -> dict:
        return deploy._retire_superseded_release_trees(
            release_root=releases, runtime_root=runtimes, active_link=active,
            current_commit=current, protection_sources=sources, keep_last=1,
            now=lambda: moment, proc_root=no_processes,
        )

    first = deploy_at(now)
    assert first["status"] == "applied"
    assert first["retired_commits"] == []
    assert first["migrated_binding_count"] == 1
    assert first["protected_by_kind"] == {
        "active_release": 1, "current_deploy": 1, "keep_last": 1, "retention_binding": 1,
    }
    assert first["lease_protected_tree_count"] == 1
    assert (releases / retained / "renderer").read_bytes() == b"retained renderer"
    lease_root = sources.lease_root / "bindings"
    assert (lease_root / "sam-prefix.json.lease.v1.json").is_file()
    assert stat.S_IMODE(lease_root.stat().st_mode) == 0o750

    later = deploy_at(now + 15 * 86_400)
    assert later["status"] == "applied"
    assert later["retired_commits"] == [retained]
    assert later["lapsed_count"] == 1 and later["migrated_binding_count"] == 0
    assert not (releases / retained).exists()
    assert binding.read_bytes() == before


def test_deploy_retirement_creates_missing_protection_roots_on_a_fresh_host(tmp_path: Path) -> None:
    """A fresh host has no authorizations or bindings yet; that must not read as unreadable."""
    import time

    releases, runtimes = tmp_path / "releases", tmp_path / "runtimes"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    current, superseded = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400}, now=time.time())
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    sources = _protection_sources(tmp_path)
    sources.standing_authorization_dir.rmdir()
    sources.binding_root.rmdir()

    def retire() -> dict:
        return deploy._retire_superseded_release_trees(
            release_root=releases, runtime_root=runtimes, active_link=active,
            current_commit=current, protection_sources=sources, keep_last=1,
            proc_root=no_processes,
        )

    # A root that had to be created proves nothing about what used to be in
    # it, so the deploy that creates one retires nothing and says so.
    created = retire()
    assert created["status"] == "skipped" and created["reason"] == "protection_root_created"
    assert created["alerts"] == [
        "release_protection_root_created:standing-authorizations",
        "release_protection_root_created:task-evaluation-release-retention-bindings",
    ]
    assert created["created_protection_roots"] == [
        str(sources.standing_authorization_dir), str(sources.binding_root),
    ]
    assert (releases / superseded).is_dir()
    for root in (sources.standing_authorization_dir, sources.binding_root):
        assert root.is_dir() and list(root.iterdir()) == []
        assert stat.S_IMODE(root.stat().st_mode) == 0o750

    # The next deploy finds them present (and empty) and retires normally.
    applied = retire()
    assert applied["status"] == "applied" and applied["retired_commits"] == [superseded]
    assert applied["created_protection_roots"] == []


def test_deploy_retirement_holds_publisher_locks_and_writes_its_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishers take the reference lock shared; retirement takes every root exclusively."""
    import contextlib
    import time

    releases, runtimes = tmp_path / "releases", tmp_path / "runtimes"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    now = time.time()
    current, superseded = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400}, now=now)
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    sources = _protection_sources(tmp_path)
    state = tmp_path / "deploy-state"
    state.mkdir()
    events: list[tuple] = []

    @contextlib.contextmanager
    def recorder(root, *, exclusive, **bounded):
        assert bounded["timeout_seconds"] == deploy.DEFAULT_RELEASE_LOCK_TIMEOUT_SECONDS
        events.append(("lock", Path(root), exclusive))
        try:
            yield
        finally:
            events.append(("unlock", Path(root)))

    collect, apply = deploy.collect_release_protections, deploy.apply_release_retirement_plan
    delete = deploy.delete_retiring_trees
    monkeypatch.setattr(deploy, "release_reference_lock", recorder)
    monkeypatch.setattr(
        deploy, "delete_retiring_trees",
        lambda roots: events.append(("delete",)) or delete(roots),
    )
    monkeypatch.setattr(
        deploy, "collect_release_protections",
        lambda *args, **kwargs: events.append(("collect", kwargs["migrate"])) or collect(*args, **kwargs),
    )
    monkeypatch.setattr(
        deploy, "apply_release_retirement_plan",
        lambda plan, **kwargs: events.append(("apply",)) or apply(plan, **kwargs),
    )
    summary_path = state / "release-retention" / "latest-deploy-retirement.json"

    result = deploy._retire_superseded_release_trees(
        release_root=releases, runtime_root=runtimes, active_link=active,
        current_commit=current, protection_sources=sources, keep_last=1, state_root=state,
        now=lambda: now, proc_root=no_processes, summary_path=summary_path,
    )

    roots = sorted([state.resolve(), sources.control_plane_root.resolve()])
    # Leftovers are swept and this run's trees deleted only while no lock is held;
    # under the lock the deploy only collects, plans and renames.
    assert events == [
        ("delete",),
        ("lock", roots[0], True),
        ("lock", roots[1], True),
        ("collect", True),
        ("apply",),
        ("unlock", roots[1]),
        ("unlock", roots[0]),
        ("delete",),
    ]
    assert result["status"] == "applied" and result["retired_commits"] == [superseded]
    assert result["lock_roots"] == [str(root) for root in roots]
    assert result["summary"] == {"status": "written", "path": str(summary_path), "mode": "0644"}
    assert stat.S_IMODE(summary_path.stat().st_mode) == 0o644
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "control_plane_release_retirement_summary.v1"
    assert summary["generated_at_epoch"] == now
    assert summary["source_commit"] == current
    assert summary["retired_commits"] == [superseded]
    assert summary["alerts"] == [] and summary["lease_protected_tree_count"] == 0


def test_deploy_retirement_reports_what_it_moved_deleted_and_swept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipts list every rename and deletion, even when apply stops halfway."""
    import time

    releases, runtimes = tmp_path / "releases", tmp_path / "runtimes"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    now = time.time()
    current, superseded, older = "a" * 40, "b" * 40, "c" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400, older: 11 * 86_400}, now=now)
    for component in ("splat-render", "scene-configuration"):
        _release_trees(runtimes / component, {superseded: 10 * 86_400}, now=now)
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    # A previous deploy stopped after moving a tree aside and before deleting it.
    leftover = releases / ".retiring" / f"{'d' * 40}-0123456789ab"
    leftover.mkdir(parents=True)
    (leftover / "payload").write_bytes(b"x" * 64)
    sources = _protection_sources(tmp_path)

    import blueprint_pipeline.control_plane_release_retirement as retirement

    stage = retirement._stage_aside
    moved: list[Path] = []

    def stop_after_two(path: Path, token: str) -> Path:
        if len(moved) == 2:
            raise RuntimeError("interrupted")
        moved.append(stage(path, token))
        return moved[-1]

    monkeypatch.setattr(retirement, "_stage_aside", stop_after_two)
    result = deploy._retire_superseded_release_trees(
        release_root=releases, runtime_root=runtimes, active_link=active,
        current_commit=current, protection_sources=sources, keep_last=1, proc_root=no_processes,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["deploy_release_retirement_failed:RuntimeError"]
    assert result["swept"] == [{"path": str(leftover), "bytes": 64, "shared_bytes": 0}]
    assert [row["staged_path"] for row in result["renamed"]] == [str(path) for path in moved]
    assert sorted(row["path"] for row in result["deleted"]) == sorted(str(path) for path in moved)
    assert result["retired_bytes"] == sum(row["bytes"] for row in result["deleted"]) > 0
    for root in (releases, runtimes / "splat-render", runtimes / "scene-configuration"):
        assert not (root / ".retiring").exists()
    # What was not moved stays; the next deploy retires it normally.
    monkeypatch.setattr(retirement, "_stage_aside", stage)
    again = deploy._retire_superseded_release_trees(
        release_root=releases, runtime_root=runtimes, active_link=active,
        current_commit=current, protection_sources=sources, keep_last=1, proc_root=no_processes,
    )
    assert again["status"] == "applied" and again["swept"] == [] and again["deletion_failures"] == []
    assert sorted(path.name for path in releases.iterdir()) == [current]


def test_deploy_retirement_locks_the_control_plane_root_once_when_it_is_the_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the host the deploy state root is the control-plane root; one descriptor only."""
    import contextlib
    import time

    releases = tmp_path / "releases"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    current, superseded = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400}, now=time.time())
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    sources = _protection_sources(tmp_path)
    locked: list[Path] = []

    @contextlib.contextmanager
    def recorder(root, *, exclusive, **_bounded):
        locked.append(Path(root))
        yield

    monkeypatch.setattr(deploy, "release_reference_lock", recorder)
    result = deploy._retire_superseded_release_trees(
        release_root=releases, runtime_root=tmp_path / "runtimes", active_link=active,
        current_commit=current, protection_sources=sources, keep_last=1,
        state_root=sources.control_plane_root, proc_root=no_processes,
    )

    assert result["status"] == "applied" and result["retired_commits"] == [superseded]
    assert locked == [sources.control_plane_root.resolve()]
    assert result["lock_roots"] == [str(sources.control_plane_root.resolve())]


def test_deploy_retirement_alerts_when_a_rename_or_a_deletion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import time

    import blueprint_pipeline.control_plane_release_retirement as retirement

    releases, runtimes = tmp_path / "releases", tmp_path / "runtimes"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    now = time.time()
    current, superseded = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400}, now=now)
    for component in ("splat-render", "scene-configuration"):
        _release_trees(runtimes / component, {superseded: 10 * 86_400}, now=now)
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    stage, remove_tree = retirement._stage_aside, retirement._remove_tree

    def refuse_first_rename(path: Path, token: str) -> Path:
        if path == releases / superseded:
            raise OSError(errno.EACCES, "Permission denied")
        return stage(path, token)

    def refuse_splat_render_removal(path: Path) -> None:
        if path.parent.parent.name == "splat-render":
            raise OSError(errno.EBUSY, "Device or resource busy")
        remove_tree(path)

    monkeypatch.setattr(retirement, "_stage_aside", refuse_first_rename)
    monkeypatch.setattr(retirement, "_remove_tree", refuse_splat_render_removal)
    result = deploy._retire_superseded_release_trees(
        release_root=releases, runtime_root=runtimes, active_link=active,
        current_commit=current, protection_sources=_protection_sources(tmp_path), keep_last=1,
        proc_root=no_processes,
    )

    assert result["status"] == "applied"
    assert result["skipped"] == [{"commit": superseded, "reason": "rename_failed:PermissionError"}]
    assert [row["reason"] for row in result["deletion_failures"]] == ["removal_failed:OSError"]
    assert result["alerts"] == [
        "release_retirement_rename_failed:1",
        "release_retirement_deletion_failed:1",
    ]


def test_a_retired_release_can_be_deployed_again_for_a_rollback(tmp_path: Path) -> None:
    """Retirement deletes a real worktree; Git must forget it so the commit can return."""
    import time

    from tests.test_task_evaluation_control_plane_release import _git, _source_repo

    repo, first, second = _source_repo(tmp_path)
    releases, state, active = tmp_path / "releases", tmp_path / "state", tmp_path / "active"
    for commit in (first, second):
        deploy.stage_task_evaluation_control_plane_release(
            source_repo=repo, source_commit=commit, release_root=releases, state_root=state,
            active_link=active, activate=commit == second,
        )
    old = time.time() - 10 * 86_400
    os.utime(releases / first, (old, old))
    no_processes = tmp_path / "proc"
    no_processes.mkdir()

    result = deploy._retire_superseded_release_trees(
        release_root=releases, runtime_root=tmp_path / "runtimes", active_link=active,
        current_commit=second, protection_sources=_protection_sources(tmp_path), keep_last=1,
        proc_root=no_processes, source_repo=repo,
    )

    assert result["status"] == "applied" and result["retired_commits"] == [first]
    assert result["worktree_prune"] == {"status": "pruned"}
    assert not (releases / first).exists()
    assert str((releases / first).resolve()) not in _git(repo, "worktree", "list", "--porcelain")
    # Rolling back to the retired commit stages and activates it again.
    rollback = deploy.stage_task_evaluation_control_plane_release(
        source_repo=repo, source_commit=first, release_root=releases, state_root=state,
        active_link=active, activate=True,
    )
    assert rollback["created_release_checkout"] is True and rollback["activated"] is True
    assert _git(releases / first, "rev-parse", "HEAD") == first


def test_deploy_retirement_gives_up_when_a_publisher_holds_the_lock(tmp_path: Path) -> None:
    """A stuck publisher costs this deploy its retirement, never the deploy itself."""
    import time

    from blueprint_pipeline.task_evaluation_release_reference_lock import release_reference_lock

    releases = tmp_path / "releases"
    no_processes = tmp_path / "proc"
    no_processes.mkdir()
    current, superseded = "a" * 40, "b" * 40
    _release_trees(releases, {current: 3_600, superseded: 10 * 86_400}, now=time.time())
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)
    sources = _protection_sources(tmp_path)

    with release_reference_lock(sources.control_plane_root, exclusive=False):
        result = deploy._retire_superseded_release_trees(
            release_root=releases, runtime_root=tmp_path / "runtimes", active_link=active,
            current_commit=current, protection_sources=sources, keep_last=1,
            proc_root=no_processes, lock_timeout_seconds=0.2,
        )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["release_reference_lock_busy"]
    assert result["alerts"] == ["release_retirement_blocked:release_reference_lock_busy"]
    assert (releases / superseded).is_dir()


def _stage_real_units(tmp_path: Path) -> Path:
    """A fake release carrying the repository's real unit files."""

    release = tmp_path / "release"
    unit_dir = release / "deploy" / "systemd"
    unit_dir.mkdir(parents=True)
    for source in (REPO_ROOT / "deploy" / "systemd").glob("blueprint-*"):
        (unit_dir / source.name).write_bytes(source.read_bytes())
    return release


def test_unit_sandbox_paths_are_provisioned_from_the_staged_units(tmp_path: Path) -> None:
    """The class that killed the preparation worker twice: a sandbox path no deploy created."""

    release = _stage_real_units(tmp_path)
    host = tmp_path / "host"
    ids = (os.getuid(), os.getgid())

    # Nothing exists yet: every path the deploy may not create is a blocker,
    # and every one of them lives outside the service state tree.
    with pytest.raises(deploy.ControlPlaneDeployError) as refused:
        deploy._install_unit_sandbox_paths(release_path=release, root_prefix=host, owner_ids=ids)
    blockers = str(refused.value).split(",")
    assert blockers and all(row.startswith("deploy_unit_sandbox_path_missing:") for row in blockers)
    missing_paths = {row.split(":", 2)[2] for row in blockers}
    assert missing_paths and not any(
        path.startswith("/var/lib/blueprint/") and not path.endswith((".json", ".lock", ".env", ".sqlite", ".log"))
        for path in missing_paths
    )
    assert not (host / "var/lib/blueprint").exists()

    for path in missing_paths:
        target = host / path.lstrip("/")
        if Path(path).suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
    # A directory that already exists keeps its owner and mode.
    preexisting = host / "var/lib/blueprint/pipeline-control-plane/gpu_spend_guard"
    preexisting.mkdir(parents=True)
    preexisting.chmod(0o755)

    receipt = deploy._install_unit_sandbox_paths(release_path=release, root_prefix=host, owner_ids=ids)

    assert receipt["status"] == "ready"
    assert receipt["created_count"] == len(receipt["created"]) > 0
    created = {row["path"] for row in receipt["created"]}
    for expected in (
        "/var/lib/blueprint/pipeline-control-plane/disk-reservations",
        "/var/lib/blueprint/pipeline-control-plane/storage-pins",
        "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes",
        "/var/lib/blueprint/task-evaluation-inputs/launch-activations",
    ):
        assert expected in created, expected
        assert (host / expected.lstrip("/")).is_dir()
        assert (host / expected.lstrip("/")).stat().st_mode & 0o777 == 0o750
    assert all(row["mode"] == "0750" and row["owner_uid"] == ids[0] for row in receipt["created"])
    assert preexisting.stat().st_mode & 0o777 == 0o755
    assert "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard" not in created
    # The optional catalog file is never created and never a blocker.
    assert not (host / "var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-profile-catalog.json").exists()

    # Idempotent: a second deploy verifies and creates nothing.
    again = deploy._install_unit_sandbox_paths(release_path=release, root_prefix=host, owner_ids=ids)
    assert again["created"] == [] and again["verified_count"] >= receipt["created_count"]


def test_unit_sandbox_paths_are_classified_and_provisioned_before_the_release_moves() -> None:
    from blueprint_pipeline.control_plane_storage_roots import classify_path

    seen: set[str] = set()
    for unit in sorted((REPO_ROOT / "deploy" / "systemd").glob("blueprint-*.service")):
        for path, _optional, _directive in deploy._unit_sandbox_entries(
            unit.read_text(encoding="utf-8")
        ):
            if path.startswith("/var/lib/blueprint/") or path.startswith("/opt/blueprint"):
                root = classify_path(path)
                assert root is not None, (unit.name, path)
                seen.add(path)
    assert seen

    source = Path(deploy.__file__).read_text(encoding="utf-8")
    assert source.index("unit_sandbox_paths = _install_unit_sandbox_paths(") < source.index(
        "_move_source_checkout(source, source_commit)"
    )
    assert '"unit_sandbox_paths": unit_sandbox_paths,' in source
    assert '"stage_timings_seconds": stage_timings,' in source


def test_unit_sandbox_entries_parse_optional_and_multi_path_directives() -> None:
    text = (
        "[Service]\n"
        "ReadWritePaths=/var/lib/blueprint /var/lib/blueprint/pipeline-control-plane/x/\n"
        "ReadOnlyPaths=-/var/lib/blueprint/pipeline-control-plane/catalog.json /etc/blueprint/profiles\n"
        "ExecStart=/bin/true\n"
    )
    assert deploy._unit_sandbox_entries(text) == [
        ("/var/lib/blueprint", False, "ReadWritePaths"),
        ("/var/lib/blueprint/pipeline-control-plane/x", False, "ReadWritePaths"),
        ("/var/lib/blueprint/pipeline-control-plane/catalog.json", True, "ReadOnlyPaths"),
        ("/etc/blueprint/profiles", False, "ReadOnlyPaths"),
    ]


def test_unit_sandbox_provisioning_skips_absent_release_units_and_needs_an_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "release"
    (empty / "deploy" / "systemd").mkdir(parents=True)
    receipt = deploy._install_unit_sandbox_paths(release_path=empty, root_prefix=tmp_path / "host")
    assert receipt == {
        "status": "ready",
        "unit_count": len(deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS),
        "verified_count": 0,
        "created_count": 0,
        "created": [],
    }

    release = tmp_path / "one-unit"
    unit_dir = release / "deploy" / "systemd"
    unit_dir.mkdir(parents=True)
    (unit_dir / deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS[0]).write_text(
        "[Service]\nReadWritePaths=/var/lib/blueprint/pipeline-control-plane/new-ledger\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deploy, "_service_account_ids", lambda _account: None)
    with pytest.raises(
        deploy.ControlPlaneDeployError, match="deploy_unit_sandbox_account_missing:blueprint"
    ):
        deploy._install_unit_sandbox_paths(release_path=release, root_prefix=tmp_path / "host2")


def test_admitted_agent_worker_is_enabled_started_and_proven(tmp_path, monkeypatch):
    from tests.test_agent_production_service import fixture
    _, _, _, config_path = fixture(tmp_path)
    calls = []
    def systemctl(argv, **kwargs):
        calls.append(argv)
        state = 'enabled' if argv[1] == 'is-enabled' else 'active'
        return SimpleNamespace(returncode=0, stdout=state + '\n', stderr='')
    monkeypatch.setattr(deploy.subprocess, 'run', systemctl)
    result = deploy._activate_agent_execution(expected_commit='a' * 40, config_path=config_path)
    assert result['activated'] and result['enabled'] == 'enabled' and result['state'] == 'active'
    unit = 'blueprint-agent-execution.service'
    assert ['systemctl', 'enable', unit] in calls
    assert ['systemctl', 'restart', unit] in calls
    assert ['systemctl', 'is-enabled', unit] in calls
    assert ['systemctl', 'is-active', unit] in calls


def test_unconfigured_or_undrained_release_never_activates_agent(tmp_path, monkeypatch):
    from tests.test_agent_production_service import fixture
    from blueprint_pipeline.agent_execution import release
    service, task, _, config_path = fixture(tmp_path)
    service.enqueue(task.task_id, "fixture-client")
    calls = []
    monkeypatch.setattr(release, 'drain', lambda *args, **kwargs: {'status': 'reconciliation_pending'})
    monkeypatch.setattr(deploy, '_systemd_unit_state', lambda unit: {'state': 'active'})
    monkeypatch.setattr(deploy.subprocess, 'run', lambda *args, **kwargs: calls.append(args))
    assert deploy._activate_agent_execution(expected_commit='a' * 40, config_path=tmp_path / 'absent')['activated'] is False
    with pytest.raises(deploy.ControlPlaneDeployError, match='agent_configuration_requires_clean_drain'):
        deploy._activate_agent_execution(expected_commit='b' * 40, config_path=config_path)
    assert calls == []


@pytest.mark.parametrize('terminal', [False, True])
def test_deploy_drains_prior_agent_tasks_before_adopting_release(tmp_path, monkeypatch, terminal):
    from tests.test_agent_production_service import fixture
    from blueprint_pipeline.agent_execution import release
    service, task, _, config_path = fixture(tmp_path)
    service.enqueue(task.task_id, "fixture-client")
    if terminal:
        service.journal.set_state(task.task_id, 'completed', result={'output': {'summary': 'Retained observation'}})
    calls = []
    def systemctl(argv, **kwargs):
        calls.append(argv)
        state = 'enabled' if argv[1] == 'is-enabled' else 'active'
        return SimpleNamespace(returncode=0, stdout=state + '\n', stderr='')
    monkeypatch.setattr(deploy.subprocess, 'run', systemctl)
    # The installed worker runs independently under its own service account.
    monkeypatch.setattr(release.time, 'sleep', lambda seconds: service.service.tick())
    result = deploy._activate_agent_execution(expected_commit='b' * 40, config_path=config_path)
    assert result['release_cleanup']['status'] == 'drained'
    assert result['configuration_adoption']['status'] == 'adopted'
    state = service.journal.task(task.task_id)
    assert state['state'] == ('completed' if terminal else 'cancelled') and state['cleanup_state'] == 'deleted'
    if terminal:
        assert state['result']['output']['summary'] == 'Retained observation'
    assert json.loads(config_path.read_text())['source_commit'] == 'b' * 40
    assert ['systemctl', 'restart', 'blueprint-agent-execution.service'] in calls


def test_deploy_drains_agent_before_switching_away_from_its_worker(tmp_path, monkeypatch):
    from tests.test_agent_production_service import fixture
    from blueprint_pipeline.agent_execution import release

    service, task, _, config_path = fixture(tmp_path)
    service.enqueue(task.task_id, "fixture-client")
    service.journal.set_state(task.task_id, "completed", result={"output": {"summary": "retained"}})
    monkeypatch.setattr(deploy, "_systemd_unit_state", lambda unit: {"state": "active"})
    monkeypatch.setattr(release.time, "sleep", lambda seconds: service.service.tick())
    result = deploy._drain_agent_execution_before_release_switch(
        expected_commit="b" * 40, config_path=config_path,
    )
    assert result["status"] == "drained"
    assert service.journal.task(task.task_id)["cleanup_state"] == "deleted"
    assert json.loads(config_path.read_text())["source_commit"] == "a" * 40
    assert (service.journal.root / release.MARKER).is_file()


@pytest.mark.slow
def test_runtime_provision_uses_target_release_imports(tmp_path, monkeypatch):
    """An old deployer's imported module must not define the new bundle inventory."""
    import types
    target = tmp_path / 'new-release'
    (target / 'scripts').mkdir(parents=True)
    package = target / 'src/blueprint_pipeline'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'runtime_inventory.py').write_text("REQUIRED = ['artifixer_metric_state.py']\n")
    stale = types.ModuleType('blueprint_pipeline.runtime_inventory')
    stale.REQUIRED = []
    monkeypatch.setitem(sys.modules, 'blueprint_pipeline.runtime_inventory', stale)
    monkeypatch.setenv('PYTHONPATH', '/stale-release/src')
    script = target / 'scripts/provision_task_evaluation_scene_configuration_release.py'
    script.write_text("""import json,sys
from blueprint_pipeline.runtime_inventory import REQUIRED
assert REQUIRED == ['artifixer_metric_state.py']
assert sys.argv[sys.argv.index('--readback-user')+1] == 'blueprint'
assert sys.argv[sys.argv.index('--astra-blender-archive')+1] == '/runtime/blender.tar.xz'
print(json.dumps({'status':'ready','source_commit':sys.argv[sys.argv.index('--source-commit')+1],
                  'environment':{'required_module':REQUIRED[0]}}))
""")
    result = deploy._provision_scene_configuration_from_release(
        repository_root=target, source_commit='a'*40, readback_user='blueprint',
        runtime_root=tmp_path/'runtimes', astra_blender_archive_path='/runtime/blender.tar.xz')
    assert result['environment']['required_module'] == 'artifixer_metric_state.py'
    assert stale.REQUIRED == []


@pytest.mark.parametrize('stdout', ['not-json', '', '[]',
    '{"status":"ready","source_commit":"wrong","environment":{}}'])
def test_runtime_provision_refuses_invalid_target_receipt(tmp_path, monkeypatch, stdout):
    monkeypatch.setattr(deploy.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=stdout))
    with pytest.raises(ValueError, match='scene_configuration_target_release_provision'):
        deploy._provision_scene_configuration_from_release(
            repository_root=tmp_path, source_commit='a'*40, readback_user='blueprint')


def test_runtime_provision_failure_does_not_fall_back_to_old_builder(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, a[0], stderr='target runtime invalid')
    monkeypatch.setattr(deploy.subprocess, 'run', fail)
    with pytest.raises(ValueError, match='scene_configuration_target_release_provision_failed'):
        deploy._provision_scene_configuration_from_release(
            repository_root=tmp_path, source_commit='a'*40, readback_user='blueprint')
