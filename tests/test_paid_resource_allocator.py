import json
import os
import signal
import subprocess
import types
from pathlib import Path

import argparse

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator


def test_detached_gpu_canary_supervisor_ignores_only_sigint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, object]] = []
    monkeypatch.setenv(allocator.DETACHED_GPU_CANARY_SUPERVISOR_ENV, "1")
    monkeypatch.setattr(
        allocator.signal,
        "signal",
        lambda signum, handler: calls.append((signum, handler)),
    )

    assert (
        allocator.configure_or_launch_detached_gpu_canary(
            "gpu-canary",
            execute=False,
            argv=[],
            repo_root=Path.cwd(),
        )
        is None
    )
    assert calls == [(signal.SIGINT, signal.SIG_IGN)]


def test_detached_gpu_canary_launcher_starts_new_session_without_recording_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict]] = []

    class _Process:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return _Process()

    monkeypatch.setattr(allocator.subprocess, "Popen", fake_popen)
    secret_marker = "do-not-record-this-value"
    result = allocator.maybe_launch_detached_gpu_canary(
        command="gpu-canary",
        execute=True,
        supervisor_dir=str(tmp_path / "supervisor"),
        argv=["gpu-canary", "--provider-launch-request", secret_marker, "--execute"],
        repo_root=tmp_path,
    )

    assert result is not None and result["status"] == "supervisor_started"
    assert result["pid"] == 12345
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["env"][allocator.DETACHED_GPU_CANARY_SUPERVISOR_ENV] == "1"
    manifest = json.loads(
        (
            tmp_path / "supervisor" / allocator.DETACHED_GPU_CANARY_MANIFEST
        ).read_text()
    )
    assert manifest["raw_arguments_recorded"] is False
    assert manifest["raw_argument_values_hashed"] is False
    assert secret_marker not in json.dumps(manifest)
    assert (
        tmp_path / "supervisor" / allocator.DETACHED_GPU_CANARY_LOG
    ).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "supervisor").stat().st_mode & 0o777 == 0o700
    assert (
        tmp_path / "supervisor" / allocator.DETACHED_GPU_CANARY_LOCK
    ).stat().st_mode & 0o777 == 0o600


def test_detached_gpu_canary_reuses_peer_owned_exact_supervisor_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "supervisor"
    root.mkdir(mode=0o700)
    chmod_calls: list[Path] = []
    original_chmod = allocator.os.chmod

    def refuse_root_chmod(path: str | bytes | os.PathLike[str], mode: int) -> None:
        resolved = Path(path).resolve()
        if resolved == root.resolve():
            chmod_calls.append(resolved)
            raise PermissionError("peer-owned exact directory must not be mutated")
        original_chmod(path, mode)

    class _Process:
        pid = 12345

    monkeypatch.setattr(allocator.os, "chmod", refuse_root_chmod)
    monkeypatch.setattr(
        allocator.subprocess, "Popen", lambda *_args, **_kwargs: _Process()
    )

    result = allocator.maybe_launch_detached_gpu_canary(
        command="gpu-canary",
        execute=True,
        supervisor_dir=str(root),
        argv=["gpu-canary", "--execute"],
        repo_root=tmp_path,
    )

    assert result is not None and result["status"] == "supervisor_started"
    assert chmod_calls == []


def test_detached_gpu_canary_refuses_when_supervisor_mode_cannot_be_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "supervisor"
    root.mkdir(mode=0o755)
    monkeypatch.setattr(
        allocator.os,
        "chmod",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    result = allocator.maybe_launch_detached_gpu_canary(
        command="gpu-canary",
        execute=True,
        supervisor_dir=str(root),
        argv=["gpu-canary", "--execute"],
        repo_root=tmp_path,
    )

    assert result == {
        "status": "blocked",
        "blockers": [
            "detached_gpu_canary_supervisor_dir_permission_install_failed"
        ],
        "provider_mutations_performed": 0,
    }
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "script_name",
    [
        "build_push_groot_oscar_foundation_image.sh",
        "build_push_groot_oscar_release_image.sh",
        "build_push_groot_oscar_closed_loop_image.sh",
    ],
)
def test_legacy_build_scripts_cannot_be_reenabled_by_environment(script_name: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT": "true"},
    )
    assert completed.returncode == 2
    assert "legacy build path disabled" in completed.stderr


def test_gpu_canary_source_checkout_binding_rejects_mismatch_and_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True, True),
    )
    assert allocator._source_checkout_blockers("b" * 40) == (
        ["gpu_canary_expected_source_commit_not_current_checkout"],
        "c" * 40,
    )
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, False, True),
    )
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_checkout_not_clean"],
        "c" * 40,
    )


def test_gpu_canary_source_checkout_binding_requires_origin_main_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True, True),
    )
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_checkout_not_origin_main"],
        "c" * 40,
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "")
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_origin_main_commit_unavailable"],
        "c" * 40,
    )


def test_gpu_canary_source_checkout_binding_requires_live_remote_main_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True, True),
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "c" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "b" * 40)
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_checkout_not_remote_main"],
        "c" * 40,
    )
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "")
    assert allocator._source_checkout_blockers("c" * 40) == (
        ["gpu_canary_remote_main_commit_unavailable"],
        "c" * 40,
    )


def test_gpu_canary_experimental_lane_accepts_clean_pushed_codex_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True, True),
    )
    monkeypatch.setattr(allocator, "_current_branch_name", lambda: "codex/diagnostic")
    monkeypatch.setattr(
        allocator,
        "_current_remote_branch_commit",
        lambda _branch: "c" * 40,
    )

    assert allocator._source_checkout_blockers("c" * 40, allow_pushed_branch_diagnostic=True) == (
        [],
        "c" * 40,
    )


def test_gpu_canary_experimental_lane_requires_matching_remote_branch_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True, True),
    )
    monkeypatch.setattr(allocator, "_current_branch_name", lambda: "codex/diagnostic")
    monkeypatch.setattr(
        allocator,
        "_current_remote_branch_commit",
        lambda _branch: "b" * 40,
    )

    assert allocator._source_checkout_blockers("c" * 40, allow_pushed_branch_diagnostic=True) == (
        ["gpu_canary_checkout_not_pushed_experimental_branch"],
        "c" * 40,
    )


def test_gpu_canary_experimental_lane_rejects_non_codex_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_current_checkout_source_state",
        lambda: ("c" * 40, True, True),
    )
    monkeypatch.setattr(allocator, "_current_branch_name", lambda: "feature/diagnostic")
    monkeypatch.setattr(
        allocator,
        "_current_remote_branch_commit",
        lambda _branch: "",
    )

    assert allocator._source_checkout_blockers("c" * 40, allow_pushed_branch_diagnostic=True) == (
        [
            "gpu_canary_experimental_branch_not_codex_namespaced",
            "gpu_canary_remote_experimental_branch_commit_unavailable",
        ],
        "c" * 40,
    )


@pytest.mark.parametrize("served_as_pull_head", [True, False])
def test_gpu_canary_experimental_lane_survives_a_merge_that_deleted_its_branch(
    monkeypatch: pytest.MonkeyPatch, served_as_pull_head: bool,
) -> None:
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("c" * 40, True, True))
    monkeypatch.setattr(allocator, "_current_branch_name", lambda: "codex/diagnostic")
    monkeypatch.setattr(allocator, "_current_remote_branch_commit", lambda _branch: "")
    asked = []
    monkeypatch.setattr(allocator, "_remote_pull_request_head_has_commit",
                        lambda commit: asked.append(commit) or served_as_pull_head)

    assert allocator._source_checkout_blockers("c" * 40, allow_pushed_branch_diagnostic=True) == (
        [] if served_as_pull_head else ["gpu_canary_remote_experimental_branch_commit_unavailable"],
        "c" * 40,
    )
    assert asked == ["c" * 40]


def test_pull_request_head_lookup_accepts_only_an_exact_served_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing = "\n".join([f"{'a' * 40}\trefs/pull/2841/head", f"{'b' * 40}\trefs/pull/2841/merge",
                         f"{'C' * 40}\trefs/pull/7/head"])
    calls = []

    def run(command, **_):
        calls.append(command)
        return types.SimpleNamespace(returncode=0, stdout=listing)
    monkeypatch.setattr(allocator.subprocess, "run", run)

    assert allocator._remote_pull_request_head_has_commit("a" * 40)
    assert allocator._remote_pull_request_head_has_commit("c" * 40)
    # A merge preview ref is not the pushed head; malformed commits never reach git.
    assert not allocator._remote_pull_request_head_has_commit("b" * 40)
    assert not allocator._remote_pull_request_head_has_commit("A" * 40)
    assert calls[0][-3:] == ["ls-remote", "origin", "refs/pull/*/head"] and len(calls) == 3
    monkeypatch.setattr(allocator.subprocess, "run",
                        lambda command, **_: types.SimpleNamespace(returncode=2, stdout=listing))
    assert not allocator._remote_pull_request_head_has_commit("a" * 40)


def test_gpu_qualification_control_plane_identity_records_main_drift_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("c" * 40, True, True))
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "a" * 40)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert blockers == []
    assert identity["orchestrator_source_commit"] == "c" * 40
    assert identity["checkout_clean"] is True
    assert identity["orchestrator_equals_origin_main"] is False
    assert identity["orchestrator_equals_remote_main"] is False
    assert identity["main_parity_is_diagnostic_not_runtime_identity"] is True


def test_gpu_qualification_control_plane_identity_still_requires_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("c" * 40, False, True))
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "a" * 40)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert blockers == ["gpu_canary_orchestrator_checkout_not_clean"]
    assert identity["orchestrator_source_commit"] == "c" * 40


def _verified_provenance_receipt(commit: str) -> dict:
    return {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "verified",
        "git_sha": commit,
        "workflow_name": "Full Test Lane",
        "workflow_path": ".github/workflows/full-test-lane.yml",
        "job_name": "Full pytest lane on CPU runner",
        "run_id": 33338010049,
        "collection": {"test_count": 4211},
        "claim_boundary": {"canonical_full_lane_verified": True},
    }


def test_release_promotion_accepts_either_deploy_surface_root(
    tmp_path, monkeypatch
) -> None:
    """Two deploy surfaces install the receipt under different roots.

    The 2026-08-30 chain installed every release's verified receipt one
    directory above the release-state root this module reads, so a freshly
    promoted release was admitted as ``development_only`` until an operator
    hand-linked the receipt across.  The identical content proof is accepted
    at either exact location, and nothing weaker passes at either."""

    releases = (
        tmp_path / "pipeline-control-plane" / "task-evaluation-control-plane-releases"
    )
    monkeypatch.setattr(allocator, "CONTROL_PLANE_RELEASE_STATE_ROOT", releases)
    commit = "b" * 40

    assert allocator.release_promotion_eligible(commit) is False

    parent_root_receipt = releases.parent / commit / "deploy-release-provenance.json"
    parent_root_receipt.parent.mkdir(parents=True)
    parent_root_receipt.write_text(
        json.dumps(_verified_provenance_receipt(commit)), encoding="utf-8"
    )
    parent_root_receipt.chmod(0o440)
    assert allocator.release_promotion_eligible(commit) is True

    canonical_commit = "c" * 40
    canonical_receipt = (
        releases / canonical_commit / "deploy-release-provenance.json"
    )
    canonical_receipt.parent.mkdir(parents=True)
    canonical_receipt.write_text(
        json.dumps(_verified_provenance_receipt(canonical_commit)), encoding="utf-8"
    )
    canonical_receipt.chmod(0o440)
    assert allocator.release_promotion_eligible(canonical_commit) is True

    iteration_commit = "d" * 40
    iteration = dict(_verified_provenance_receipt(iteration_commit))
    iteration["status"] = "iteration"
    iteration_receipt = (
        releases.parent / iteration_commit / "deploy-release-provenance.json"
    )
    iteration_receipt.parent.mkdir(parents=True)
    iteration_receipt.write_text(json.dumps(iteration), encoding="utf-8")
    iteration_receipt.chmod(0o440)
    assert allocator.release_promotion_eligible(iteration_commit) is False

    writable_commit = "e" * 40
    writable_receipt = (
        releases.parent / writable_commit / "deploy-release-provenance.json"
    )
    writable_receipt.parent.mkdir(parents=True)
    writable_receipt.write_text(
        json.dumps(_verified_provenance_receipt(writable_commit)), encoding="utf-8"
    )
    writable_receipt.chmod(0o666)
    assert allocator.release_promotion_eligible(writable_commit) is False


# ------------------------- RunPod retired (2026-11-05) -------------------------

_WARM_WORKER_ARGS = [
    "gpu-warm-worker", "--output-dir", "out", "--release-evidence", "release.json",
    "--model-cache-evidence", "cache.json", "--watchdog-handoff-evidence", "handoff.json",
    "--resource-name-prefix", "blueprint-groot-oscar-serverless-test-",
    "--expected-source-commit", "c" * 40, "--campaign-budget-ledger", "budget.json",
    "--campaign-initial-spent-usd", "1", "--campaign-initial-used-gpu-seconds", "1",
    "--campaign-io-evidence", "campaign_io.json", "--execute",
]


@pytest.mark.parametrize(
    "argv",
    [
        _WARM_WORKER_ARGS,
        ["model-volume", "--output-dir", "out", "--storage-hourly-rate-usd", "0.01", "--allow-paid"],
        ["model-volume-run", "--output-dir", "out", "--storage-hourly-rate-usd", "0.01"],
        ["model-volume", "--output-dir", "out", "--storage-hourly-rate-usd", "0.01",
         "--retain-existing-output", "old", "--campaign-spent-to-date-usd", "1"],
        ["gpu-canary", "--provider", "runpod", "--probe-kind", "strict-policy-smoke", "--execute"],
        ["gpu-canary", "--probe-kind", "startup", "--execute"],  # --provider defaults to runpod
        ["gpu-canary", "--provider", "runpod", "--probe-kind", "single-kitchen-episode"],
        ["gpu-canary", "--provider", "runpod", "--probe-kind", "single-kitchen-finetune"],
        ["gpu-canary", "--provider", "vast", "--probe-kind", "persistent-policy-wam-loop"],
        ["gpu-canary", "--provider", "vast", "--openpi-provider", "runpod",
         "--probe-kind", "openpi-policy-ranking"],
    ],
)
def test_runpod_requests_are_refused_before_detaching_or_any_provider_call(
    argv, tmp_path, monkeypatch, capsys,
) -> None:
    from blueprint_pipeline import paid_resource_admission

    assert paid_resource_admission.RUNPOD_PROVIDER_RETIRED is True
    for name in ("configure_or_launch_detached_gpu_canary", "run_scene_configuration_allocator_probe",
                 "isolated_native_task_arena_lane"):
        monkeypatch.setattr(allocator, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} reached"))
    admission_out = tmp_path / "admission.json"
    if argv[0] == "gpu-canary" and "single-kitchen-episode" in argv:
        argv = [*argv, "--admission-out", str(admission_out)]
    assert allocator.main(argv) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"status": "blocked", "blockers": ["runpod_provider_retired"],
                       "provider_mutations_performed": 0, "success": False}
    if admission_out.exists():
        assert json.loads(admission_out.read_text())["blockers"] == ["runpod_provider_retired"]


def test_runpod_refusal_never_blocks_a_terminal_vast_release_or_a_vast_request() -> None:
    from blueprint_pipeline import paid_resource_admission as admission

    refusal = admission.runpod_allocator_request_blockers

    def request(**fields):
        return argparse.Namespace(**{"command": "gpu-canary", "provider": "runpod", "openpi_provider": "vast",
                                     "terminal_resource_release": None, "probe_kind": "startup", **fields})

    assert refusal(request(terminal_resource_release="release.json")) == []
    # Probe kinds that require Vast, ignore --provider or choose it elsewhere keep the
    # argparse default without being refused here.
    for probe_kind in ("native-task-arena-controls", "policy-ranking-successor-cosmos",
                       "semantic-sam31-source-tracks", "task-evaluation-profile-preflight",
                       allocator.TASK_EVALUATION_SCENE_CONFIGURATION_PROBE_KIND,
                       allocator.RIGID_ROBOT_FREE_PROBE_KIND):
        assert refusal(request(probe_kind=probe_kind)) == []
    assert refusal(request(probe_kind="single-kitchen-episode", provider="vast")) == []
    assert refusal(request(probe_kind="openpi-policy-ranking", provider="vast")) == []
    for command in ("cpu-build", "remote-cpu-job", "provider-reconstruction"):
        assert refusal(argparse.Namespace(command=command, provider="runpod")) == []
    assert refusal(request()) == ["runpod_provider_retired"]
    assert admission.RUNPOD_ONLY_PROBE_KINDS == {"startup", "strict-policy-smoke", "persistent-policy-wam-loop"}
    assert admission.RUNPOD_SELECTABLE_PROBE_KINDS == {"single-kitchen-episode", "single-kitchen-finetune"}
    assert admission.OPENPI_PROVIDER_PROBE_KINDS == {"openpi-policy-ranking", "new-site-diagnostic-canary"}
