from __future__ import annotations

import json
import stat
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_task_evaluation_scene_configuration_diagnostic_iteration as iteration


SOURCE_COMMIT = "a" * 40


def _args(tmp_path: Path, *, execute: bool = False) -> Namespace:
    source = tmp_path / "source"
    source.mkdir()
    release_root = tmp_path / "releases"
    state_root = tmp_path / "state"
    python = tmp_path / "venv-python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    envelope = tmp_path / "envelope.json"
    envelope.write_text("{}\n", encoding="utf-8")
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    runtime = tmp_path / "splat-runtime"
    runtime.mkdir()
    (runtime / "browser-sentinel").write_bytes(b"do-not-copy")
    checkpoint = tmp_path / "checkpoint-reference.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    project_spend = tmp_path / "project-spend.json"
    project_spend.write_text("{}\n", encoding="utf-8")
    provider_zero = tmp_path / "provider-zero.json"
    provider_zero.write_text("{}\n", encoding="utf-8")
    return Namespace(
        source_repo=str(source.resolve()),
        source_commit=SOURCE_COMMIT,
        remote_branch="codex/scene-fix",
        release_root=str(release_root.resolve()),
        state_root=str(state_root.resolve()),
        python_executable=str(python.resolve()),
        construction_envelope=str(envelope.resolve()),
        toolchain_root=str(toolchain.resolve()),
        splat_render_runtime_root=str(runtime.resolve()),
        diagnostic_checkpoint_reference=str(checkpoint.resolve()),
        fresh_diagnostic_bootstrap=False,
        bundle_output_root=str((tmp_path / "bundle-output").resolve()),
        project_spend_reconciliation=str(project_spend.resolve()),
        initial_provider_zero=str(provider_zero.resolve()),
        authorization_reference="explicit-user-direction-fixture",
        authorized_by="fixture-owner",
        authorized_on="2026-08-27T12:00:00Z",
        pod_name="blueprint-scene-diagnostic-fixture",
        max_hourly_rate_usd=0.8,
        hard_cap_usd=0.5,
        hard_ttl_seconds=1800,
        provider_compute_spend_cap_usd=0.4,
        openai_max_cost_usd=0.1,
        openai_max_requests=2,
        openai_artifixer_semantic_teacher_max_cost_usd=0.0,
        openai_artifixer_visual_review_max_cost_usd=0.0,
        openai_content_agents_max_cost_usd=0.1,
        scene_configuration_attempt_authority=str(
            (tmp_path / "authority.json").resolve()
        ),
        scene_configuration_job_dir=str((tmp_path / "job").resolve()),
        admission_out=str((tmp_path / "admission.json").resolve()),
        adapter_output=str((tmp_path / "adapter.json").resolve()),
        iteration_preparation_receipt=str(
            (tmp_path / "preparation.json").resolve()
        ),
        retain_warm_session=False,
        allowed_vast_machine_id=[],
        warm_session_authority=None,
        warm_session_output_root=None,
        maximum_warm_iterations=8,
        execute=execute,
    )


def test_one_command_stages_source_builds_fixed_chain_and_revalidates_before_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    release_path = Path(args.release_root) / SOURCE_COMMIT
    release_path.mkdir(parents=True)
    (release_path / "src").mkdir()
    release_receipt = Path(args.state_root) / SOURCE_COMMIT / "diagnostic-release.json"
    release_receipt.parent.mkdir(parents=True)
    release_receipt.write_text("{}\n", encoding="utf-8")
    events: list[str] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(
        iteration,
        "stage_scene_configuration_diagnostic_release",
        lambda **_kwargs: {
            "release_path": str(release_path),
            "receipt_path": str(release_receipt),
            "receipt_digest": "sha256:" + "1" * 64,
            "remote_ref": "refs/heads/codex/scene-fix",
            "reused_existing_checkout": True,
        },
    )

    def validate(*_args, **_kwargs):
        events.append("validate")
        return {"status": "staged"}

    monkeypatch.setattr(
        iteration,
        "validate_scene_configuration_diagnostic_release_receipt",
        validate,
    )

    def runner(argv, **_kwargs):
        command = list(argv)
        commands.append(command)
        module = command[command.index("-m") + 1]
        if module.endswith("task_evaluation_scene_configuration_bundle"):
            events.append("bundle")
            output = Path(command[command.index("--output-root") + 1])
            output.mkdir(parents=True)
            staging = output / "stage"
            staging.mkdir()
            (staging / "expanded-runtime.bin").write_bytes(b"regenerable")
            (output / f"{iteration.BUNDLE_SCHEMA_VERSION}.receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": iteration.BUNDLE_SCHEMA_VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                        "bundle_sha256": "sha256:" + "2" * 64,
                        "receipt_digest": "sha256:" + "3" * 64,
                        "diagnostic_bootstrap_mode": "checkpoint_resume",
                        "source_diagnostic_checkpoint_digest": (
                            "sha256:" + "5" * 64
                        ),
                        "carried_completed_stage_count": 3,
                    }
                ),
                encoding="utf-8",
            )
        elif module.endswith("task_evaluation_scene_configuration_paid_authority"):
            assert not (Path(args.bundle_output_root) / "stage").exists()
            events.append("authority")
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "schema_version": iteration.AUTHORITY_SCHEMA_VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "bundle_sha256": "sha256:" + "2" * 64,
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                        "authority_digest": "sha256:" + "4" * 64,
                        "diagnostic_bootstrap_mode": "checkpoint_resume",
                    }
                ),
                encoding="utf-8",
            )
        elif module.endswith("paid_resource_allocator"):
            events.append("allocator")
        else:  # pragma: no cover - fixed command regression diagnostic
            pytest.fail(f"unexpected fixed module: {module}")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    times = iter((0.0, 0.11, 0.2, 0.41, 0.5, 0.62))
    result = iteration.run_scene_configuration_diagnostic_iteration(
        args, runner=runner, clock=lambda: next(times)
    )

    assert events == ["bundle", "authority", "validate", "allocator"]
    assert [command[command.index("-m") + 1] for command in commands] == [
        "blueprint_pipeline.task_evaluation_scene_configuration_bundle",
        "blueprint_pipeline.task_evaluation_scene_configuration_paid_authority",
        "blueprint_pipeline.paid_resource_allocator",
    ]
    flattened = "\n".join(" ".join(command) for command in commands)
    assert "deploy_control_plane" not in flattened
    assert "systemctl" not in flattened
    assert "--active-link" not in flattened
    assert "--experimental-branch-diagnostic" in commands[-1]
    assert "--scene-configuration-diagnostic-only" in commands[-1]
    assert "--release-evidence" in commands[-1]
    assert "--execute" not in commands[-1]
    assert result["source_materialization_elapsed_ms"] == 110
    assert result["source_materialization_target_met"] is True
    assert result["total_preparation_elapsed_ms"] == 620
    assert result["total_preparation_seconds_claimed"] is False
    assert result["bundle_staging_tree_removed_after_seal"] is True
    assert result["splat_runtime_reused_by_reference"] is True
    assert result["splat_runtime_copied"] is False
    assert result["remaining_preparation_bottleneck"][
        "toolchain_tree_copied_and_provider_zip_rebuilt"
    ] is True
    assert result["active_release_link_updated"] is False
    assert result["systemd_services_restarted"] is False
    assert result["diagnostic_only"] is True
    assert result["qualification_eligible"] is False
    assert (Path(args.splat_render_runtime_root) / "browser-sentinel").read_bytes() == (
        b"do-not-copy"
    )
    preparation = json.loads(
        Path(args.iteration_preparation_receipt).read_text(encoding="utf-8")
    )
    assert preparation == result
    assert stat.S_IMODE(
        Path(args.iteration_preparation_receipt).stat().st_mode
    ) == 0o440


def test_execute_only_adds_canonical_allocator_execute_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    _install_paid_runtime_environment(tmp_path, monkeypatch)
    release_path = Path(args.release_root) / SOURCE_COMMIT
    release_path.mkdir(parents=True)
    (release_path / "src").mkdir()
    receipt = Path(args.state_root) / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        iteration,
        "stage_scene_configuration_diagnostic_release",
        lambda **_kwargs: {
            "release_path": str(release_path),
            "receipt_path": str(receipt),
            "receipt_digest": "sha256:" + "1" * 64,
            "remote_ref": "refs/heads/codex/scene-fix",
            "reused_existing_checkout": False,
        },
    )
    monkeypatch.setattr(
        iteration,
        "validate_scene_configuration_diagnostic_release_receipt",
        lambda *_args, **_kwargs: {},
    )
    allocator_commands: list[list[str]] = []
    bundle_commands: list[list[str]] = []

    def runner(argv, **_kwargs):
        command = list(argv)
        module = command[command.index("-m") + 1]
        if module.endswith("bundle"):
            bundle_commands.append(command)
            output = Path(command[command.index("--output-root") + 1])
            output.mkdir(parents=True)
            (output / f"{iteration.BUNDLE_SCHEMA_VERSION}.receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": iteration.BUNDLE_SCHEMA_VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                        "bundle_sha256": "sha256:" + "2" * 64,
                        "diagnostic_bootstrap_mode": "checkpoint_resume",
                        "source_diagnostic_checkpoint_digest": (
                            "sha256:" + "3" * 64
                        ),
                        "carried_completed_stage_count": 3,
                    }
                ),
                encoding="utf-8",
            )
        elif module.endswith("paid_authority"):
            Path(command[command.index("--output") + 1]).write_text(
                json.dumps(
                    {
                        "schema_version": iteration.AUTHORITY_SCHEMA_VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "bundle_sha256": "sha256:" + "2" * 64,
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                        "diagnostic_bootstrap_mode": "checkpoint_resume",
                    }
                ),
                encoding="utf-8",
            )
        else:
            allocator_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    times = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    iteration.run_scene_configuration_diagnostic_iteration(
        args, runner=runner, clock=lambda: next(times)
    )
    assert allocator_commands and allocator_commands[0][-1] == "--execute"


@pytest.mark.parametrize("fresh_bootstrap", [False, True])
def test_execute_can_retain_one_warm_session_through_canonical_allocator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bootstrap: bool,
) -> None:
    args = _args(tmp_path, execute=True)
    _install_paid_runtime_environment(tmp_path, monkeypatch)
    args.retain_warm_session = True
    args.allowed_vast_machine_id = [44762, 21899, 44762]
    args.warm_session_authority = str((tmp_path / "warm-authority.json").resolve())
    args.warm_session_output_root = str((tmp_path / "warm-session").resolve())
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    if fresh_bootstrap:
        args.fresh_diagnostic_bootstrap = True
        args.diagnostic_checkpoint_reference = None
    else:
        Path(args.diagnostic_checkpoint_reference).write_text(
            json.dumps({"checkpoint_root": str(checkpoint_root.resolve())}),
            encoding="utf-8",
        )
    release_path = Path(args.release_root) / SOURCE_COMMIT
    release_path.mkdir(parents=True)
    (release_path / "src").mkdir()
    receipt = Path(args.state_root) / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        iteration,
        "stage_scene_configuration_diagnostic_release",
        lambda **_kwargs: {
            "release_path": str(release_path),
            "receipt_path": str(receipt),
            "receipt_digest": "sha256:" + "1" * 64,
            "remote_ref": "refs/heads/codex/scene-fix",
            "reused_existing_checkout": False,
        },
    )
    monkeypatch.setattr(
        iteration,
        "validate_scene_configuration_diagnostic_release_receipt",
        lambda *_args, **_kwargs: {},
    )
    warm_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        iteration,
        "materialize_scene_configuration_warm_session_authority",
        lambda **kwargs: warm_calls.append(kwargs) or {},
    )
    allocator_commands: list[list[str]] = []
    bundle_commands: list[list[str]] = []

    def runner(argv, **_kwargs):
        command = list(argv)
        module = command[command.index("-m") + 1]
        if module.endswith("bundle"):
            bundle_commands.append(command)
            output = Path(command[command.index("--output-root") + 1])
            output.mkdir(parents=True)
            (output / f"{iteration.BUNDLE_SCHEMA_VERSION}.receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": iteration.BUNDLE_SCHEMA_VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                        "bundle_sha256": "sha256:" + "2" * 64,
                        "diagnostic_bootstrap_mode": (
                            "fresh" if fresh_bootstrap else "checkpoint_resume"
                        ),
                        "source_diagnostic_checkpoint_digest": (
                            None if fresh_bootstrap else "sha256:" + "3" * 64
                        ),
                        "carried_completed_stage_count": (
                            0 if fresh_bootstrap else 3
                        ),
                    }
                ),
                encoding="utf-8",
            )
        elif module.endswith("paid_authority"):
            Path(command[command.index("--output") + 1]).write_text(
                json.dumps(
                    {
                        "schema_version": iteration.AUTHORITY_SCHEMA_VERSION,
                        "source_commit": SOURCE_COMMIT,
                        "bundle_sha256": "sha256:" + "2" * 64,
                        "diagnostic_only": True,
                        "qualification_eligible": False,
                        "configured_revision_publication_permitted": False,
                        "offering_publication_permitted": False,
                        "terminal_e2e_completion_permitted": False,
                        "diagnostic_bootstrap_mode": (
                            "fresh" if fresh_bootstrap else "checkpoint_resume"
                        ),
                    }
                ),
                encoding="utf-8",
            )
        else:
            allocator_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    times = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    result = iteration.run_scene_configuration_diagnostic_iteration(
        args, runner=runner, clock=lambda: next(times)
    )

    assert len(warm_calls) == 1
    assert warm_calls[0]["checkpoint_root"] == (
        None if fresh_bootstrap else checkpoint_root.resolve()
    )
    assert ("--fresh-diagnostic-bootstrap" in bundle_commands[0]) is (
        fresh_bootstrap
    )
    assert ("--diagnostic-checkpoint-reference" in bundle_commands[0]) is (
        not fresh_bootstrap
    )
    command = allocator_commands[0]
    assert "--scene-configuration-retain-warm-session" in command
    assert command[command.index("--scene-configuration-warm-session-authority") + 1] == args.warm_session_authority
    assert command[command.index("--scene-configuration-warm-session-output-root") + 1] == args.warm_session_output_root
    machine_flag = "--scene-configuration-allowed-vast-machine-id"
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == machine_flag
    ] == ["21899", "44762"]
    assert result["warm_session_retention_requested"] is True
    assert result["allowed_vast_machine_ids"] == [21899, 44762]
    assert result["diagnostic_bootstrap_mode"] == (
        "fresh" if fresh_bootstrap else "checkpoint_resume"
    )


def test_child_failure_is_redacted_and_does_not_print_child_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    release_path = Path(args.release_root) / SOURCE_COMMIT
    release_path.mkdir(parents=True)
    (release_path / "src").mkdir()
    receipt = Path(args.state_root) / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        iteration,
        "stage_scene_configuration_diagnostic_release",
        lambda **_kwargs: {
            "release_path": str(release_path),
            "receipt_path": str(receipt),
            "receipt_digest": "sha256:" + "1" * 64,
            "remote_ref": "refs/heads/codex/scene-fix",
            "reused_existing_checkout": True,
        },
    )
    secret_text = "sk-test-secret-must-not-escape"

    def failed(argv, **_kwargs):
        output = Path(args.bundle_output_root)
        (output / "stage" / "provider_runtime").mkdir(parents=True)
        (output / "stage" / "provider_runtime" / "partial.bin").write_bytes(
            b"unsealed-regenerable-partial"
        )
        return subprocess.CompletedProcess(
            list(argv), 2, stdout=secret_text, stderr=secret_text
        )

    with pytest.raises(
        iteration.SceneConfigurationDiagnosticIterationError,
        match="diagnostic_iteration_bundle_failed",
    ) as exc:
        iteration.run_scene_configuration_diagnostic_iteration(args, runner=failed)
    assert secret_text not in str(exc.value)
    assert not (Path(args.bundle_output_root) / "stage").exists()


def _install_paid_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in iteration._OPENAI_RUNTIME_FILE_ENV_NAMES:
        value = tmp_path / name.lower()
        value.write_text("fixture\n", encoding="utf-8")
        monkeypatch.setenv(name, str(value.resolve()))
    for name in iteration._OPENAI_RUNTIME_VALUE_ENV_NAMES:
        monkeypatch.setenv(name, "fixture-identity")
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "spend"))


def test_execute_preflights_openai_environment_before_staging_or_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    stage_called = False

    def stage(**_kwargs):
        nonlocal stage_called
        stage_called = True
        pytest.fail("release staging must not run after environment preflight refusal")

    monkeypatch.setattr(
        iteration, "stage_scene_configuration_diagnostic_release", stage
    )
    for name in (
        *iteration._OPENAI_RUNTIME_FILE_ENV_NAMES,
        *iteration._OPENAI_RUNTIME_VALUE_ENV_NAMES,
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        iteration.SceneConfigurationDiagnosticIterationError,
        match=(
            "scene_configuration_diagnostic_iteration_openai_runtime_environment_missing:"
            "OPENAI_ADMIN_API_KEY_FILE"
        ),
    ):
        iteration.run_scene_configuration_diagnostic_iteration(args)

    assert stage_called is False


def test_execute_preflights_spend_identity_before_bundle_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, execute=True)
    _install_paid_runtime_environment(tmp_path, monkeypatch)
    stage_called = False

    def stage(**_kwargs):
        nonlocal stage_called
        stage_called = True
        pytest.fail("release staging must not run after spend-identity refusal")

    monkeypatch.setattr(
        iteration, "stage_scene_configuration_diagnostic_release", stage
    )
    monkeypatch.setattr(
        iteration,
        "prepare_consumption_root",
        lambda: (_ for _ in ()).throw(
            iteration.SpendAuthorityRootError(
                "spend_authority_consumption_root_not_owned"
            )
        ),
    )

    with pytest.raises(
        iteration.SceneConfigurationDiagnosticIterationError,
        match=(
            "scene_configuration_diagnostic_iteration_spend_identity_invalid:"
            "spend_authority_consumption_root_not_owned"
        ),
    ):
        iteration.run_scene_configuration_diagnostic_iteration(args)

    assert stage_called is False


def test_child_failure_surfaces_typed_detail_but_redacts_credentials() -> None:
    def failed(argv, **_kwargs):
        return subprocess.CompletedProcess(
            list(argv),
            2,
            stdout="",
            stderr=(
                "scene_configuration_openai_runtime_secret_configuration_missing "
                "sk-fixturesecret12345678"
            ),
        )

    with pytest.raises(
        iteration.SceneConfigurationDiagnosticIterationError,
        match=(
            "diagnostic_iteration_allocator_failed:"
            "scene_configuration_openai_runtime_secret_configuration_missing"
        ),
    ) as exc:
        iteration._run_fixed(
            ["python", "-m", "blueprint_pipeline.paid_resource_allocator"],
            cwd=Path("/"),
            environment={},
            runner=failed,
            code="scene_configuration_diagnostic_iteration_allocator_failed",
        )
    assert "fixturesecret" not in str(exc.value)
    assert "<redacted>" in str(exc.value)


def test_paths_are_explicit_absolute_and_no_arbitrary_command_option_exists(
    tmp_path: Path
) -> None:
    args = _args(tmp_path)
    args.bundle_output_root = "relative-output"
    with pytest.raises(
        iteration.SceneConfigurationDiagnosticIterationError,
        match="bundle_output_root_must_be_absolute",
    ):
        iteration.run_scene_configuration_diagnostic_iteration(args)

    option_names = {
        option
        for action in iteration._parser()._actions
        for option in action.option_strings
    }
    assert "--command" not in option_names
    assert "--argv" not in option_names
    assert "--shell" not in option_names


def test_normal_venv_python_symlink_preserves_venv_entrypoint(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "python-target"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(executable)

    assert iteration._python_executable(venv_python) == venv_python
