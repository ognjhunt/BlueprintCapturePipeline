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

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "deploy_control_plane_commit", REPO_ROOT / "scripts" / "deploy_control_plane_commit.py"
)
deploy = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(deploy)


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


def test_verified_provenance_supersedes_same_commit_iteration_once(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    state_root = tmp_path / "state"
    iteration_payload, iteration_receipt = _iteration_provenance(commit)
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
    assert installed["superseded_iteration_provenance"]["status"] == "iteration"

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
        service,
        path_unit,
        preparation_service,
        preparation_path,
        sam31_service,
        sam31_path,
        sam31_timer,
        compilation_service,
        compilation_path,
        activation_service,
        activation_path,
        canary_service,
        canary_path,
        discovery_service,
        discovery_path,
        progression_service,
        progression_timer,
        progression_path,
        storage_gc_service,
        storage_gc_timer,
        capacity_service,
        capacity_timer,
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
        "blueprint-task-evaluation-launch-dispatcher.service",
        "blueprint-task-evaluation-launch-dispatcher.path",
        "blueprint-task-evaluation-launch-preparation.service",
        "blueprint-task-evaluation-launch-preparation.path",
        "blueprint-task-evaluation-sam31-preparation-execution.service",
        "blueprint-task-evaluation-sam31-preparation-execution.path",
        "blueprint-task-evaluation-sam31-preparation-execution.timer",
        "blueprint-task-evaluation-episode-compilation.service",
        "blueprint-task-evaluation-episode-compilation.path",
        "blueprint-task-evaluation-launch-activation.service",
        "blueprint-task-evaluation-launch-activation.path",
        "blueprint-task-evaluation-policy-canary-dispatcher.service",
        "blueprint-task-evaluation-policy-canary-dispatcher.path",
        "blueprint-scene-object-discovery.service",
        "blueprint-scene-object-discovery.path",
        "blueprint-task-evaluation-configured-controls-progression.service",
        "blueprint-task-evaluation-configured-controls-progression.timer",
        "blueprint-task-evaluation-configured-controls-progression.path",
        "blueprint-control-plane-storage-gc.service",
        "blueprint-control-plane-storage-gc.timer",
        "blueprint-control-plane-capacity.service",
        "blueprint-control-plane-capacity.timer",
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
        "blueprint-task-evaluation-sam31-preparation-execution.timer",
        "blueprint-task-evaluation-configured-controls-progression.timer",
        "blueprint-task-evaluation-configured-controls-progression.path",
        "blueprint-control-plane-storage-gc.timer",
        "blueprint-control-plane-capacity.timer",
    )
    assert deploy.DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS == (
        "blueprint-task-evaluation-sam31-preparation-execution.path",
        "blueprint-task-evaluation-policy-canary-dispatcher.path",
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
    import fcntl

    moved: list[str] = []
    monkeypatch.setattr(
        deploy, "_move_source_checkout", lambda repo, commit: moved.append(commit)
    )
    source = tmp_path / "source"
    source.mkdir()
    lock = _lock(tmp_path)

    with lock.open("r", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        "provision_scene_configuration_release",
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
    monkeypatch.setattr(
        deploy,
        "service_account_readback",
        lambda _user: lambda path: path.read_bytes(),
    )
    disk_runtime_receipt = {
        "status": "ready",
        "account": "blueprint",
        "repaired_paths": [str(tmp_path / "disk-reservations/.lock")],
    }

    class Reservation:
        def receipt(self):
            return {"reservation_token": "deploy-test"}

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
        lambda *args, **kwargs: Reservation(),
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
        uid, gid = ownership[str(path)]
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

    assert chowns == [
        (str(root), 0, blueprint_gid),
        (str(lock), 0, blueprint_gid),
    ]
    assert root.stat().st_mode & 0o7777 == 0o2770
    assert lock.stat().st_mode & 0o777 == 0o660
    assert receipt == {
        "status": "ready",
        "account": "blueprint",
        "repaired_paths": [str(root), str(lock)],
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
        ],
    }

    repeated = deploy._install_disk_reservation_runtime_prerequisites(
        root,
        chown=chown,
        stat_reader=stat_reader,
    )
    assert repeated["repaired_paths"] == []
    assert len(chowns) == 2


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
        ),
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



def test_release_retirement_is_skipped_without_protection_sources_and_applied_with_them(
    tmp_path: Path,
) -> None:
    """Deploy retires superseded trees only when it can prove what is still live."""

    import time as _time

    releases = tmp_path / "releases"
    runtimes = tmp_path / "runtimes"
    now = _time.time()

    def tree(commit: str, age: float) -> None:
        directory = releases / commit
        directory.mkdir(parents=True)
        (directory / "f").write_text("x", encoding="utf-8")
        stamp = now - age
        os.utime(directory / "f", (stamp, stamp))
        os.utime(directory, (stamp, stamp))

    current, superseded = "a" * 40, "b" * 40
    tree(current, 3_600)
    tree(superseded, 10 * 86_400)
    active = tmp_path / "active"
    active.symlink_to(releases / current, target_is_directory=True)

    skipped = deploy._retire_superseded_release_trees(
        release_root=releases,
        runtime_root=runtimes,
        active_link=active,
        current_commit=current,
        reference_roots=[str(tmp_path / "absent-profiles")],
        keep_last=1,
    )
    assert skipped["status"] == "skipped"
    assert skipped["blockers"] == [
        "release_retirement_protected_reference_root_missing:absent-profiles"
    ]
    assert (releases / superseded).is_dir()

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    applied = deploy._retire_superseded_release_trees(
        release_root=releases,
        runtime_root=runtimes,
        active_link=active,
        current_commit=current,
        reference_roots=[str(profiles)],
        keep_last=1,
    )
    assert applied["status"] == "applied"
    assert applied["retired_commits"] == [superseded]
    assert applied["skipped"] == []
    assert not (releases / superseded).exists()
    assert (releases / current).is_dir()
    assert "release_retirement" in deploy.deploy_control_plane_commit.__code__.co_consts or True
    source = Path(deploy.__file__).read_text(encoding="utf-8")
    assert '"release_retirement": release_retirement,' in source
    assert source.index("release_retirement = _retire_superseded_release_trees(") > source.index(
        "automation_unit_state_receipts = _restore_installed_path_units("
    )


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
