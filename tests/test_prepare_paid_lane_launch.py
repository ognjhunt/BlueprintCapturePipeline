"""Hermetic tests for ``scripts/prepare_paid_lane_launch.py``.

No subprocess, no network, no provider: every step is a recorded fake, so the
ordering and fail-closed contract is exercised without running a real command.
"""

from __future__ import annotations

import json
import os
import pwd
import grp
import stat
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from scripts import prepare_paid_lane_launch as prep


def _lane(tmp_path: Path) -> tuple[prep.LaneStep, ...]:
    return (
        prep.LaneStep(
            step_id="first",
            argv=("cmd-first", "{value}"),
            produces=str(tmp_path / "first.json"),
            exports=(("carried", "published_uri"),),
        ),
        prep.LaneStep(
            step_id="second",
            argv=("cmd-second", "{carried}"),
            produces=str(tmp_path / "second.json"),
        ),
    )


def test_scene_configuration_graph_stages_autostart_before_live_profile() -> None:
    """The Website activation graph must carry the downstream CPU intent."""

    steps = prep.LANES["task_evaluation_scene_configuration"]
    step_ids = [step.step_id for step in steps]
    assert step_ids.index("configured_controls_autostart_intent") < step_ids.index(
        "live_profile"
    )
    autostart = steps[step_ids.index("configured_controls_autostart_intent")]
    live_profile = steps[step_ids.index("live_profile")]
    assert "{configured_controls_autostart_intent_source}" in autostart.argv
    assert autostart.produces == (
        "{set_root}/task_evaluation_configured_controls_autostart_intent.v1.json"
    )
    # The profile binds the intent through the context rather than a fixed
    # argv, so a scene with no continuation yet publishes a profile without
    # one instead of naming a path the skipped step never wrote.
    assert (
        "--configured-controls-autostart-intent",
        "configured_controls_autostart_intent_artifacts",
    ) in live_profile.repeated_argv


def _writing_runner(
    calls: list[list[str]],
    artifacts: dict[str, str],
) -> object:
    def runner(argv: Sequence[str]) -> int:
        calls.append(list(argv))
        name = str(argv[0])
        content = artifacts.get(name)
        if content is not None:
            path, _, body = content.partition("::")
            Path(path).write_text(body, encoding="utf-8")
        return 0

    return runner


def test_steps_run_in_order_and_exported_values_reach_later_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    calls: list[list[str]] = []
    runner = _writing_runner(
        calls,
        {
            "cmd-first": f"{tmp_path / 'first.json'}::"
            + json.dumps({"published_uri": "r2://bucket/key.json"}),
            "cmd-second": f"{tmp_path / 'second.json'}::" + json.dumps({"ok": True}),
        },
    )

    receipt = prep.prepare_paid_lane_launch(
        "fake",
        {
            "value": "v",
            "reference_bindings": {
                "prior_webapp_lineage": {"receipt_digest": "sha256:" + "a" * 64}
            },
        },
        runner=runner,
    )

    assert [call[0] for call in calls] == ["cmd-first", "cmd-second"]
    assert calls[1][1] == "r2://bucket/key.json"
    assert receipt["status"] == "prepared"
    assert [s["step_id"] for s in receipt["completed_steps"]] == ["first", "second"]
    assert [row["step_id"] for row in receipt["step_logs"]] == ["first", "second"]
    assert all(row["credential_redaction_applied"] for row in receipt["step_logs"])
    assert receipt["completed_steps"][0]["exports"] == {
        "carried": "r2://bucket/key.json"
    }
    assert receipt["reference_bindings"] == {
        "prior_webapp_lineage": {"receipt_digest": "sha256:" + "a" * 64}
    }


def test_a_failing_step_stops_the_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    calls: list[list[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(list(argv))
        return 3

    receipt = prep.prepare_paid_lane_launch("fake", {"value": "v"}, runner=runner)

    assert [call[0] for call in calls] == ["cmd-first"]
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["first:exit_3"]
    assert receipt["completed_steps"] == []


def test_blocked_receipt_resumes_after_verified_completed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    first_calls: list[list[str]] = []

    def first_runner(argv: Sequence[str]) -> int:
        first_calls.append(list(argv))
        if argv[0] == "cmd-first":
            (tmp_path / "first.json").write_text(
                json.dumps({"published_uri": "r2://bucket/key.json"}),
                encoding="utf-8",
            )
            return 0
        return 3

    blocked = prep.prepare_paid_lane_launch(
        "fake", {"value": "v"}, runner=first_runner
    )
    assert blocked["status"] == "blocked"
    assert [row["step_id"] for row in blocked["completed_steps"]] == ["first"]

    resumed_calls: list[list[str]] = []

    def resumed_runner(argv: Sequence[str]) -> int:
        resumed_calls.append(list(argv))
        (tmp_path / "second.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )
        return 0

    resumed = prep.prepare_paid_lane_launch(
        "fake",
        {"value": "v"},
        resume_receipt=blocked,
        runner=resumed_runner,
    )

    assert [call[0] for call in resumed_calls] == ["cmd-second"]
    assert resumed_calls[0][1] == "r2://bucket/key.json"
    assert resumed["status"] == "prepared"
    assert [row["step_id"] for row in resumed["completed_steps"]] == [
        "first",
        "second",
    ]


def test_reserved_receipt_is_installed_for_service_identity(tmp_path: Path) -> None:
    account = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    output, descriptor = prep._reserve_receipt_output(tmp_path / "receipt.json")

    prep._write_reserved_receipt(
        output,
        descriptor,
        {"status": "prepared"},
        service_account=account,
        service_group=group,
    )

    metadata = output.stat()
    assert metadata.st_uid == os.getuid()
    assert metadata.st_gid == os.getgid()
    assert stat.S_IMODE(metadata.st_mode) == 0o440
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "prepared"
    }


def test_a_failing_step_retains_redacted_digest_bound_stdout_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph refusal must be diagnosable without rerunning its command."""

    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    secret = "sk-example-secret-material"
    long_detail = "diagnostic-" + "x" * 600

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            3,
            stdout=f"publication started {secret}\n{long_detail}\n",
            stderr=(
                "upload refused "
                "https://objects.example/item?X-Amz-Signature=signed-value\n"
            ),
        )

    receipt = prep.prepare_paid_lane_launch(
        "fake",
        {"value": "v", "api_key": secret},
        runner=runner,
    )

    assert receipt["blockers"] == ["first:exit_3"]
    assert len(receipt["step_logs"]) == 1
    log_record = receipt["step_logs"][0]
    assert log_record["step_id"] == "first"
    stdout_path = Path(log_record["stdout_path"])
    stderr_path = Path(log_record["stderr_path"])
    assert stdout_path.parent == tmp_path
    assert stderr_path.parent == tmp_path
    assert stdout_path.name == "first.json.first.stdout.log"
    assert stderr_path.name == "first.json.first.stderr.log"
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert secret not in stdout_text
    assert long_detail in stdout_text
    assert "signed-value" not in stderr_text
    assert "<redacted>" in stdout_text
    assert "<redacted>" in stderr_text
    assert log_record["stdout_sha256"] == prep._sha256_file(stdout_path)
    assert log_record["stderr_sha256"] == prep._sha256_file(stderr_path)
    assert stat.S_IMODE(stdout_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(stderr_path.stat().st_mode) == 0o600


def test_step_log_symlink_blocks_the_graph_after_command_without_following_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    outside = tmp_path / "outside.log"
    outside.write_text("owner bytes", encoding="utf-8")
    log_path = tmp_path / "first.json.first.stdout.log"
    log_path.symlink_to(outside)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        (tmp_path / "first.json").write_text(
            json.dumps({"published_uri": "r2://bucket/key.json"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(list(argv), 0, stdout="safe\n", stderr="")

    receipt = prep.prepare_paid_lane_launch(
        "fake", {"value": "v"}, runner=runner
    )

    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        "first:paid_lane_step_log_path_invalid"
    ]
    assert receipt["completed_steps"] == []
    assert receipt["step_logs"] == []
    assert receipt["provider_allocation_performed"] is False
    assert outside.read_text(encoding="utf-8") == "owner bytes"


def test_validate_only_renders_the_complete_plan_without_running_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))

    receipt = prep.validate_paid_lane_launch(
        "fake",
        {"value": "v"},
    )

    assert receipt["status"] == "validated_no_commands_run"
    assert receipt["subprocesses_executed"] == 0
    assert receipt["provider_mutation_performed"] is False
    assert receipt["planned_steps"][0]["argv"] == ["cmd-first", "v"]
    assert receipt["planned_steps"][1]["argv"] == [
        "cmd-second",
        "<export:first:published_uri>",
    ]
    assert not (tmp_path / "first.json").exists()
    assert not (tmp_path / "second.json").exists()


def test_validate_only_cli_requires_one_immutable_receipt_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = tmp_path / "must-not-run.json"
    monkeypatch.setitem(
        prep.LANES,
        "semantic_teacher_image_edit",
        (
            prep.LaneStep(
                step_id="must-not-run",
                argv=("{python}", "forbidden-subprocess"),
                produces=str(never),
            ),
        ),
    )
    output = tmp_path / "validation.json"

    assert (
        prep.main(
            [
                "--lane",
                "semantic_teacher_image_edit",
                "--validate-only",
                "--receipt-out",
                str(output),
            ]
        )
        == 0
    )
    first_bytes = output.read_bytes()
    assert json.loads(first_bytes)["status"] == "validated_no_commands_run"
    assert not never.exists()

    assert (
        prep.main(
            [
                "--lane",
                "semantic_teacher_image_edit",
                "--validate-only",
                "--receipt-out",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == first_bytes


def test_a_step_that_exits_zero_without_its_artifact_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit status alone is not evidence the artifact exists."""

    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    calls: list[list[str]] = []
    runner = _writing_runner(calls, {})

    receipt = prep.prepare_paid_lane_launch("fake", {"value": "v"}, runner=runner)

    assert [call[0] for call in calls] == ["cmd-first"]
    assert receipt["blockers"] == ["first:declared_artifact_missing"]


def test_an_incomplete_context_prepares_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production regression: a half-prepared lane hides which step is stale.

    An unsupplied placeholder must fail before the first command runs, not
    resolve to an empty path partway through.
    """

    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    calls: list[list[str]] = []

    def runner(argv: Sequence[str]) -> int:  # pragma: no cover - must not run
        calls.append(list(argv))
        return 0

    with pytest.raises(prep.PaidLaneLaunchPreparationError) as excinfo:
        prep.prepare_paid_lane_launch("fake", {}, runner=runner)

    assert "first:value" in str(excinfo.value)
    assert calls == []


def test_an_empty_context_value_counts_as_unsupplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))

    with pytest.raises(prep.PaidLaneLaunchPreparationError):
        prep.validate_lane_context("fake", {"value": ""})


def test_an_unresolvable_export_blocks_before_the_dependent_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(prep.LANES, "fake", _lane(tmp_path))
    calls: list[list[str]] = []
    runner = _writing_runner(
        calls,
        {"cmd-first": f"{tmp_path / 'first.json'}::" + json.dumps({"status": "blocked"})},
    )

    receipt = prep.prepare_paid_lane_launch("fake", {"value": "v"}, runner=runner)

    assert [call[0] for call in calls] == ["cmd-first"]
    assert receipt["blockers"] == ["first:export_unavailable:carried"]


def test_unknown_lane_is_rejected() -> None:
    with pytest.raises(prep.PaidLaneLaunchPreparationError):
        prep.validate_lane_context("not_a_lane", {})


def test_every_shipped_lane_is_satisfiable_by_its_own_command_line() -> None:
    """A shipped lane must not name a value the CLI cannot supply.

    Placeholders are resolved from operator arguments plus values exported by
    earlier steps, so a lane naming anything else could never be prepared.
    """

    supplied = {
        "python",
        "repository_root",
        "set_root",
        "packet",
        "source_commit",
        "destination_prefix",
        "token_file",
        "runtime_image_identity",
        "profile_dir",
        "webapp_catalog_out",
        "service_account",
        "service_group",
        "pod_name",
        "revision",
        "authorization_reference",
        "authorized_by",
        "authorized_on",
        "backend_entry_digest",
        "task_count",
        "camera_count",
        "maximum_hourly_rate_usd",
        "hard_total_spend_cap_usd",
        "provider_compute_spend_cap_usd",
        "openai_max_cost_usd",
        "openai_max_requests",
        "openai_artifixer_semantic_teacher_max_cost_usd",
        "openai_artifixer_visual_review_max_cost_usd",
        "openai_content_agents_max_cost_usd",
        "anthropic_max_cost_usd",
        "anthropic_max_requests",
        "hard_ttl_seconds",
        "provider",
        "aggregate_goal_spend_before_usd",
        "aggregate_goal_spend_cap_usd",
        # Native Task Arena contexts derive these from the versioned context
        # file rather than accepting independent command-line strings.
        "packet_dir",
        "runtime_source_packet",
        "container_image",
        "construction_envelope",
        "toolchain_root",
        "team_namespace",
        "scene_id",
        "task_id",
        "project_spend_reconciliation",
        "initial_provider_zero",
        "configured_controls_autostart_intent_source",
        "destination_probe_request",
        "configured_scene_support_plane",
        "destination_static_qualification",
        "destination_native_import_qualification",
        "destination_geometry",
        "prior_authority",
        "prior_result",
        "prior_provider_zero",
        "prior_spend_reconciliation",
        "construction_result",
        "zero_action_result",
        "standing_authorization_dir",
        "standing_authorization_expires_at",
    }
    for lane, steps in prep.LANES.items():
        available = set(supplied)
        for step in steps:
            unmet = sorted(prep.step_placeholders(step) - available)
            assert unmet == [], f"{lane}:{step.step_id} cannot resolve {unmet}"
            available |= {name for name, _ in step.exports}


def test_scene_configuration_prepares_rehearses_then_publishes_once() -> None:
    lane = "task_evaluation_scene_configuration"
    steps = prep.LANES[lane]
    order = [step.step_id for step in steps]
    assert order == [
        "provider_bundle",
        "immutable_manifest",
        "paid_authority",
        "allocator_dry_run",
        "configured_controls_autostart_intent",
        "live_profile",
        "terminal_rehearsal",
        "profile_publication",
        "standing_authorization",
    ]
    by_id = {step.step_id: step for step in steps}
    assert (
        "--production-semantic-reuse-checkpoint-root",
        "production_semantic_reuse_checkpoint_root",
    ) in by_id["provider_bundle"].repeated_argv
    assert (
        "--production-semantic-reuse-queue-root",
        "production_semantic_reuse_queue_root",
    ) in by_id["provider_bundle"].repeated_argv
    assert (
        "--production-semantic-reuse-revision-id",
        "production_semantic_reuse_revision_id",
    ) in by_id["provider_bundle"].repeated_argv
    assert (
        "--scene-configuration-queue-root",
        "production_semantic_reuse_queue_root",
    ) in by_id["allocator_dry_run"].repeated_argv
    assert "--execute" not in by_id["allocator_dry_run"].argv
    assert (
        by_id["allocator_dry_run"].argv[
            by_id["allocator_dry_run"].argv.index("--probe-kind") + 1
        ]
        == "task-evaluation-scene-configuration"
    )
    assert (
        by_id["terminal_rehearsal"].argv[
            by_id["terminal_rehearsal"].argv.index("--lane-module") + 1
        ]
        == "task_evaluation_scene_configuration_vast.py"
    )
    assert (
        by_id["standing_authorization"].argv[
            by_id["standing_authorization"].argv.index("--max-launches") + 1
        ]
        == "1"
    )
    # The allocator refuses a pod name that differs from the paid attempt
    # authority's resource_name, so both must derive from the same {pod_name}
    # placeholder: the authority binds it, and the live profile embeds it.
    assert (
        by_id["paid_authority"].argv[
            by_id["paid_authority"].argv.index("--resource-name") + 1
        ]
        == "{pod_name}"
    )
    assert (
        by_id["live_profile"].argv[
            by_id["live_profile"].argv.index("--pod-name") + 1
        ]
        == "{pod_name}"
    )


def test_scene_configuration_context_reopens_server_owned_inputs(
    tmp_path: Path,
) -> None:
    envelope = tmp_path / "construction-envelope.json"
    envelope.write_text('{"schema_version":"fixture.v1"}\n', encoding="utf-8")
    envelope.chmod(0o440)
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir(mode=0o550)
    semantic_reuse = tmp_path / "semantic-reuse"
    semantic_reuse.mkdir(mode=0o550)
    construction_queue = tmp_path / "construction-queue"
    construction_queue.mkdir(mode=0o750)
    context_path = tmp_path / "context.json"
    source_commit = "a" * 40
    context_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    prep.SCENE_CONFIGURATION_CONTEXT_SCHEMA_VERSION
                ),
                "lane": "task_evaluation_scene_configuration",
                "team_namespace": "team-a",
                "reference_bindings": {
                    "construction_envelope_sha256": prep._sha256_file(envelope),
                    "source_commit": source_commit,
                },
                "operations": {
                    "construction_envelope": str(envelope),
                    "toolchain_root": str(toolchain),
                    "source_commit": source_commit,
                    "production_semantic_reuse_checkpoint_root": str(
                        semantic_reuse
                    ),
                    "production_semantic_reuse_queue_root": str(
                        construction_queue
                    ),
                    "production_semantic_reuse_revision_id": "corrective-r2",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = prep._load_scene_configuration_context(
        context_path, expected_lane="task_evaluation_scene_configuration"
    )

    assert loaded["construction_envelope"] == str(envelope.resolve())
    assert loaded["toolchain_root"] == str(toolchain.resolve())
    assert loaded["team_namespace"] == "team-a"
    assert loaded["production_semantic_reuse_checkpoint_root"] == str(
        semantic_reuse
    )
    assert loaded["production_semantic_reuse_queue_root"] == str(
        construction_queue
    )
    assert loaded["production_semantic_reuse_revision_id"] == "corrective-r2"
    assert loaded["reference_bindings"]["source_commit"] == source_commit


def test_scene_claim_reference_refuses_tampering_and_symlinks(tmp_path: Path) -> None:
    value = {
        "schema_version": "task_evaluation_scene_rights_admission.v1",
        "status": "admitted",
        "scene_id": "scene-1",
        "private_provider_processing_allowed": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
        "rights_admission_digest": "",
    }
    value["rights_admission_digest"] = prep._canonical_artifact_digest(
        value, digest_field="rights_admission_digest"
    )
    source = tmp_path / "rights.json"
    source.write_text(json.dumps(value), encoding="utf-8")

    prep._load_scene_claim_reference(
        path=source,
        expected_digest=value["rights_admission_digest"],
        expected_schema="task_evaluation_scene_rights_admission.v1",
        expected_status="admitted",
        digest_field="rights_admission_digest",
        scene_id="scene-1",
    )

    value["provider_training_allowed"] = True
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        prep.PaidLaneLaunchPreparationError,
        match="native_task_arena_scene_claim_reference_invalid",
    ):
        prep._load_scene_claim_reference(
            path=source,
            expected_digest=value["rights_admission_digest"],
            expected_schema="task_evaluation_scene_rights_admission.v1",
            expected_status="admitted",
            digest_field="rights_admission_digest",
            scene_id="scene-1",
        )

    value["rights_admission_digest"] = prep._canonical_artifact_digest(
        value, digest_field="rights_admission_digest"
    )
    source.write_text(json.dumps(value), encoding="utf-8")
    symlink = tmp_path / "rights-link.json"
    symlink.symlink_to(source)
    with pytest.raises(
        prep.PaidLaneLaunchPreparationError,
        match="native_task_arena_scene_claim_reference_invalid",
    ):
        prep._load_scene_claim_reference(
            path=symlink,
            expected_digest=value["rights_admission_digest"],
            expected_schema="task_evaluation_scene_rights_admission.v1",
            expected_status="admitted",
            digest_field="rights_admission_digest",
            scene_id="scene-1",
        )


def test_provider_packet_rejects_source_not_admitted_for_upload() -> None:
    digest = "sha256:" + "a" * 64
    packet_receipt = {
        "source_bindings": [
            {
                "semantic_role": "scene_appearance",
                "source": {"sha256": digest, "size_bytes": 17},
                "staged_sha256": digest,
                "staged_size_bytes": 17,
            }
        ]
    }
    source_manifest = {
        "artifacts": [
            {
                "role": "raw_source_splat",
                "sha256": digest,
                "size_bytes": 17,
                "provider_upload_allowed": False,
            }
        ]
    }

    with pytest.raises(
        prep.PaidLaneLaunchPreparationError,
        match="native_task_arena_provider_source_rights_invalid",
    ):
        prep._validate_provider_packet_source_rights(
            packet_receipt=packet_receipt,
            source_manifest=source_manifest,
        )

    source_manifest["artifacts"][0]["provider_upload_allowed"] = True
    prep._validate_provider_packet_source_rights(
        packet_receipt=packet_receipt,
        source_manifest=source_manifest,
    )


def test_optional_retry_inputs_emit_no_empty_arguments() -> None:
    assert prep._repeated_values([]) == ()
    assert prep._repeated_values(None) == ()


def test_root_style_set_root_is_handed_to_service_group_without_touching_token(
    tmp_path: Path,
) -> None:
    """Exact production regression: root's set root blocked later traversal.

    Only the named preparation root is handed off.  A credential below it must
    remain owner-private, and an unrelated sibling must not be re-permissioned.
    """

    account = pwd.getpwuid(os.geteuid()).pw_name
    group = grp.getgrgid(pwd.getpwnam(account).pw_gid).gr_name
    set_root = tmp_path / "task-evaluation-inputs" / "semantic-r6"
    set_root.mkdir(parents=True)
    set_root.chmod(0o700)
    token = set_root / "openai_api_key"
    token.write_text("secret", encoding="utf-8")
    token.chmod(0o600)
    sibling = set_root.parent / "unrelated-retained-run"
    sibling.mkdir()
    sibling.chmod(0o700)

    observed = prep._prepare_set_root_for_service(
        set_root,
        service_account=account,
        service_group=group,
    )

    assert observed == set_root.resolve()
    assert stat.S_IMODE(set_root.stat().st_mode) == 0o750
    assert set_root.stat().st_gid == grp.getgrnam(group).gr_gid
    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    assert token.read_text(encoding="utf-8") == "secret"
    assert stat.S_IMODE(sibling.stat().st_mode) == 0o700


def test_already_correct_set_root_requires_no_privileged_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = pwd.getpwuid(os.geteuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    set_root = tmp_path / "already-correct"
    set_root.mkdir()
    set_root.chmod(0o750)

    def forbidden_chown(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("already-correct set root must not be re-owned")

    original_chmod = Path.chmod

    def forbidden_target_chmod(path: Path, mode: int, **kwargs: object) -> None:
        if path == set_root:
            raise PermissionError("already-correct set root must not be rewritten")
        original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(prep.os, "chown", forbidden_chown)
    monkeypatch.setattr(Path, "chmod", forbidden_target_chmod)

    assert prep._prepare_set_root_for_service(
        set_root,
        service_account=account,
        service_group=group,
    ) == set_root.resolve()


def test_set_root_symlink_is_refused_before_any_lane_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = pwd.getpwuid(os.geteuid()).pw_name
    group = grp.getgrgid(pwd.getpwnam(account).pw_gid).gr_name
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    lane = (
        prep.LaneStep(
            step_id="must-not-run",
            argv=("command",),
            produces=str(real_root / "receipt.json"),
        ),
    )
    monkeypatch.setitem(prep.LANES, "hostile", lane)
    calls: list[list[str]] = []

    with pytest.raises(
        prep.PaidLaneLaunchPreparationError, match="paid_lane_set_root_symlink"
    ):
        prep.prepare_paid_lane_launch(
            "hostile",
            {
                "set_root": str(linked_root),
                "service_account": account,
                "service_group": group,
            },
            runner=lambda argv: calls.append(list(argv)) or 0,
        )

    assert calls == []


def test_configured_runtime_rights_use_derived_scope_and_exact_destination(tmp_path):
    digest = "sha256:" + "a" * 64
    tray_digest = "sha256:" + "b" * 64
    request = {"request_digest": ""}
    request["request_digest"] = prep._canonical_artifact_digest(request, digest_field="request_digest")
    binding = {"semantic_role": "scene_collision", "source": {"sha256": digest, "size_bytes": 17}, "staged_sha256": digest, "staged_size_bytes": 17}
    receipt = {"source_bindings": [binding], "request_digest": request["request_digest"]}
    revision = {"source": {"provider_disclosure_decision": {"rights_admission_permits_upload": False, "human_authority_accepts_provider_terms": True}}, "geometry": {"configured_collision": {"digest": digest, "size_bytes": 17}}}
    rights = {"private_provider_processing_allowed": True, "provider_disclosure": {"sage_collision_runtime_bytes_may_be_privately_processed": True, "provider_training_allowed": False, "public_redistribution_allowed": False, "provider_retention_rule": "exact run then teardown"}}
    kwargs = dict(packet_receipt=receipt, packet_request=request, source_manifest={"artifacts": []}, configured_scene_revision=revision, rights_admission=rights)
    prep._validate_provider_packet_source_rights(**kwargs)
    rights["provider_disclosure"]["sage_collision_runtime_bytes_may_be_privately_processed"] = False
    with pytest.raises(prep.PaidLaneLaunchPreparationError, match="provider_source_rights_invalid"):
        prep._validate_provider_packet_source_rights(**kwargs)
    rights["provider_disclosure"]["sage_collision_runtime_bytes_may_be_privately_processed"] = True
    binding["source"]["sha256"] = "sha256:" + "c" * 64
    with pytest.raises(prep.PaidLaneLaunchPreparationError, match="provider_source_rights_invalid"):
        prep._validate_provider_packet_source_rights(**kwargs)
    binding["source"]["sha256"] = digest
    identity = {"id": "tray", "version": "v1"}
    tray_rights = {"schema_version": "task_evaluation_rigid_destination_rights_admission.v1", "status": "admitted", "destination_identity": identity, "private_provider_processing_allowed": True, "provider_training_allowed": False, "public_redistribution_allowed": False, "rights_admission_digest": ""}
    tray_rights["rights_admission_digest"] = prep._canonical_artifact_digest(tray_rights, digest_field="rights_admission_digest")
    path = tmp_path / "tray-rights.json"
    path.write_text(json.dumps(tray_rights))
    destination = {"identity": identity, "asset": {"digest": tray_digest, "size_bytes": 12}, "rights_admission": str(path), "rights_admission_digest": prep._sha256_file(path)}
    receipt["source_bindings"].append({"semantic_role": "task_support", "asset_id": "tray", "source": {"sha256": tray_digest, "size_bytes": 12}, "staged_sha256": tray_digest, "staged_size_bytes": 12})
    with pytest.raises(prep.PaidLaneLaunchPreparationError, match="provider_source_rights_invalid"):
        prep._validate_provider_packet_source_rights(**kwargs)
    prep._validate_provider_packet_source_rights(**kwargs, destination=destination)
    path.write_text('{}')
    with pytest.raises(prep.PaidLaneLaunchPreparationError, match="destination_rights_invalid"):
        prep._validate_provider_packet_source_rights(**kwargs, destination=destination)


@pytest.mark.parametrize("kind", ["source_manifest", "rights_admission"])
def test_completed_scene_claim_reference_keeps_exact_bytes_and_identity(tmp_path, kind):
    expected = "task_evaluation_scene_" + kind + ".v1"
    value = {"schema_version": "task_evaluation_completed_scene_" + kind + ".v1",
        "scene_id": "completed-scene-1", "source": "owner_provided_completed_asset",
        "status": "candidate_source_bytes_retained" if kind == "source_manifest" else "admitted_for_internal_development"}
    source = tmp_path / "claim.json"
    source.write_text(json.dumps(value))
    digest = prep._sha256_file(source)
    kwargs = dict(path=source, expected_digest=digest, expected_schema=expected,
        expected_status="retained" if kind == "source_manifest" else "admitted",
        digest_field=kind + "_digest", scene_id="completed-scene-1")
    prep._load_scene_claim_reference(**kwargs)
    with pytest.raises(prep.PaidLaneLaunchPreparationError):
        prep._load_scene_claim_reference(**{**kwargs, "scene_id": "another-scene"})
    value["schema_version"] = value["schema_version"].replace(".v1", ".v2")
    source.write_text(json.dumps(value))
    with pytest.raises(prep.PaidLaneLaunchPreparationError):
        prep._load_scene_claim_reference(**{**kwargs, "expected_digest": prep._sha256_file(source)})


@pytest.mark.parametrize("mutation", [None, "unknown_schema", "foreign_scene", "raw_upload",
                                      "training", "public", "missing_derived", "unknown_role",
                                      "foreign_bytes"])
def test_completed_scene_derived_provider_rights_stay_bounded(mutation):
    digest = "sha256:" + "a" * 64
    source = {"schema_version": "task_evaluation_completed_scene_source_manifest.v1",
              "source": "owner_provided_completed_asset", "scene_id": "completed-1",
              "captured_observation_supplied": False,
              "coordinate_system": {"physical_metrology_claimed": False}, "artifacts": []}
    rights = {"schema_version": "task_evaluation_completed_scene_rights_admission.v1",
              "source": "owner_provided_completed_asset", "scene_id": "completed-1",
              "status": "admitted_for_internal_development", "source_bytes_redistributable": False,
              "physical_metrology_claimed": False,
              "private_provider_processing_allowed": True, "provider_training_allowed": False,
              "public_redistribution_allowed": False,
              "provider_disclosure": {"raw_owner_source_bytes_may_be_uploaded": False,
                  "minimum_digest_bound_derived_runtime_bytes_may_be_privately_processed": True,
                  "provider_training_allowed": False, "public_redistribution_allowed": False,
                  "provider_retention_rule": "exact run then teardown"}}
    revision = {"scene_identity": {"id": "completed-1"}, "source": {
        "provider_disclosure_decision": {"rights_admission_permits_upload": False,
                                        "human_authority_accepts_provider_terms": True}},
        "geometry": {"configured_collision": {"digest": digest, "size_bytes": 17}}}
    binding = {"semantic_role": "scene_collision", "source": {"sha256": digest, "size_bytes": 17},
               "staged_sha256": digest, "staged_size_bytes": 17}
    if mutation == "unknown_schema":
        rights["schema_version"] = rights["schema_version"].replace(".v1", ".v2")
    if mutation == "foreign_scene":
        rights["scene_id"] = "foreign"
    if mutation == "raw_upload":
        rights["provider_disclosure"]["raw_owner_source_bytes_may_be_uploaded"] = True
    if mutation == "training":
        rights["provider_training_allowed"] = True
    if mutation == "public":
        rights["public_redistribution_allowed"] = True
    if mutation == "missing_derived":
        rights["provider_disclosure"].pop("minimum_digest_bound_derived_runtime_bytes_may_be_privately_processed")
    if mutation == "unknown_role":
        binding["semantic_role"] = "arbitrary-raw-source"
    if mutation == "foreign_bytes":
        binding["source"]["sha256"] = "sha256:" + "b" * 64
    kwargs = dict(packet_receipt={"source_bindings": [binding]}, source_manifest=source,
                  configured_scene_revision=revision, rights_admission=rights)
    if mutation is None:
        prep._validate_provider_packet_source_rights(**kwargs)
    else:
        with pytest.raises(prep.PaidLaneLaunchPreparationError, match="provider_source_rights_invalid"):
            prep._validate_provider_packet_source_rights(**kwargs)
