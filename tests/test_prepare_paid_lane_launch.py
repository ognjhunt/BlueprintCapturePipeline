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

    receipt = prep.prepare_paid_lane_launch("fake", {"value": "v"}, runner=runner)

    assert [call[0] for call in calls] == ["cmd-first", "cmd-second"]
    assert calls[1][1] == "r2://bucket/key.json"
    assert receipt["status"] == "prepared"
    assert [s["step_id"] for s in receipt["completed_steps"]] == ["first", "second"]
    assert receipt["completed_steps"][0]["exports"] == {
        "carried": "r2://bucket/key.json"
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
        "hard_ttl_seconds",
        "aggregate_goal_spend_before_usd",
        "aggregate_goal_spend_cap_usd",
        # Native Task Arena contexts derive these from the versioned context
        # file rather than accepting independent command-line strings.
        "packet_dir",
        "runtime_source_packet",
        "container_image",
        "scene_id",
        "task_id",
        "project_spend_reconciliation",
        "initial_provider_zero",
        "prior_authority",
        "prior_result",
        "prior_provider_zero",
        "prior_spend_reconciliation",
        "construction_result",
        "standing_authorization_dir",
        "standing_authorization_expires_at",
    }
    for lane, steps in prep.LANES.items():
        available = set(supplied)
        for step in steps:
            unmet = sorted(prep.step_placeholders(step) - available)
            assert unmet == [], f"{lane}:{step.step_id} cannot resolve {unmet}"
            available |= {name for name, _ in step.exports}


def test_shipped_semantic_teacher_lane_keeps_its_required_order() -> None:
    """The bundle must precede its manifest, authority, dry run, and profile."""

    order = [step.step_id for step in prep.LANES["semantic_teacher_image_edit"]]
    assert order == [
        "provider_bundle",
        "immutable_manifest",
        "paid_authority",
        "allocator_dry_run",
        "live_profile",
        "profile_publication",
        "terminal_rehearsal",
    ]


@pytest.mark.parametrize(
    "lane",
    ["native_task_arena_construction", "native_task_arena_controls"],
)
def test_native_lane_prepares_rehearses_then_publishes_once(lane: str) -> None:
    order = [step.step_id for step in prep.LANES[lane]]
    assert order == [
        "provider_bundle",
        "immutable_manifest",
        "paid_authority",
        "allocator_dry_run",
        "live_profile",
        "terminal_rehearsal",
        "profile_publication",
        "standing_authorization",
    ]
    dry_run = prep.LANES[lane][order.index("allocator_dry_run")]
    assert "--execute" not in dry_run.argv
    assert dry_run.argv[dry_run.argv.index("--provider") + 1] == "vast"
    assert dry_run.argv[dry_run.argv.index("--probe-kind") + 1] == lane.replace(
        "_", "-"
    )
    standing = prep.LANES[lane][order.index("standing_authorization")]
    assert standing.argv[standing.argv.index("--max-launches") + 1] == "1"
    bundle = prep.LANES[lane][order.index("provider_bundle")]
    assert bundle.argv[bundle.argv.index("--container-image") + 1] == (
        "{container_image}"
    )


def test_native_context_reopens_independent_versioned_references(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    robot = {"robot_id": "customer_arm_v3", "adapter": "fixed_arm_adapter_v1"}
    (packet / "native_task_arena_packet_receipt.v1.json").write_text(
        json.dumps(
            {
                "scene_id": "public-scene-17",
                "task_id": "move-can-v2",
                "receipt_digest": "sha256:" + "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    (packet / "native_task_runtime_contract.v1.json").write_text(
        json.dumps(
            {
                "task_spec_digest": "sha256:" + "2" * 64,
                "robot": robot,
            }
        ),
        encoding="utf-8",
    )
    runtime_source = tmp_path / "runtime-source.json"
    runtime_source.write_text(
        json.dumps({"receipt_digest": "sha256:" + "3" * 64}),
        encoding="utf-8",
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest_value = {
        "schema_version": "task_evaluation_scene_source_manifest.v1",
        "status": "retained",
        "scene_id": "public-scene-17",
        "artifacts": [],
        "source_manifest_digest": "",
    }
    source_manifest_value["source_manifest_digest"] = (
        prep._canonical_artifact_digest(
            source_manifest_value, digest_field="source_manifest_digest"
        )
    )
    source_manifest.write_text(json.dumps(source_manifest_value), encoding="utf-8")
    rights_admission = tmp_path / "rights-admission.json"
    rights_admission_value = {
        "schema_version": "task_evaluation_scene_rights_admission.v1",
        "status": "admitted",
        "scene_id": "public-scene-17",
        "private_provider_processing_allowed": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": True,
        "rights_admission_digest": "",
    }
    rights_admission_value["rights_admission_digest"] = (
        prep._canonical_artifact_digest(
            rights_admission_value, digest_field="rights_admission_digest"
        )
    )
    rights_admission.write_text(json.dumps(rights_admission_value), encoding="utf-8")
    context_file = tmp_path / "context.json"
    context_file.write_text(
        json.dumps(
            {
                "schema_version": prep.NATIVE_CONTEXT_SCHEMA_VERSION,
                "lane": "native_task_arena_construction",
                "team_namespace": "robot-team-a",
                "references": {
                    "scene": {
                        "scene_id": "public-scene-17",
                        "packet_dir": str(packet),
                        "packet_receipt_digest": "sha256:" + "1" * 64,
                        "source_manifest": str(source_manifest),
                        "source_manifest_digest": source_manifest_value[
                            "source_manifest_digest"
                        ],
                        "rights_admission": str(rights_admission),
                        "rights_admission_digest": rights_admission_value[
                            "rights_admission_digest"
                        ],
                    },
                    "task": {
                        "task_id": "move-can-v2",
                        "task_spec_digest": "sha256:" + "2" * 64,
                    },
                    "robot": {
                        "robot_id": "customer_arm_v3",
                        "configuration_digest": prep._canonical_mapping_digest(robot),
                    },
                    "runtime": {
                        "source_packet": str(runtime_source),
                        "source_packet_receipt_digest": "sha256:" + "3" * 64,
                        "container_image": (
                            "registry.example/robot-runtime@sha256:" + "4" * 64
                        ),
                    },
                },
                "operations": {
                    "set_root": str(tmp_path / "set"),
                    "repository_root": str(tmp_path / "repo"),
                    "source_commit": "a" * 40,
                    "destination_prefix": "r2://bucket/manifests",
                    "profile_dir": str(tmp_path / "profiles"),
                    "webapp_catalog_out": str(tmp_path / "catalog.json"),
                    "standing_authorization_dir": str(tmp_path / "authorities"),
                    "standing_authorization_expires_at": (
                        "2026-08-26T14:30:00+00:00"
                    ),
                    "pod_name": "new-scene-canary",
                    "revision": "r1",
                    "authorization_reference": "user-authorized-new-lane",
                    "authorized_by": "user",
                    "authorized_on": "2026-08-25T14:30:00+00:00",
                    "maximum_hourly_rate_usd": 0.8,
                    "hard_total_spend_cap_usd": 0.75,
                    "hard_ttl_seconds": 3300,
                    "project_spend_reconciliation": str(
                        tmp_path / "project-spend.json"
                    ),
                    "initial_provider_zero": str(tmp_path / "provider-zero.json"),
                },
            }
        ),
        encoding="utf-8",
    )

    context = prep._load_native_context(
        context_file, expected_lane="native_task_arena_construction"
    )

    assert context["scene_id"] == "public-scene-17"
    assert context["task_id"] == "move-can-v2"
    assert context["packet_dir"] == str(packet.resolve())
    assert context["runtime_source_packet"] == str(runtime_source.resolve())
    assert context["reference_bindings"]["robot"]["robot_id"] == "customer_arm_v3"
    assert context["reference_bindings"]["source_manifest_path"] == str(
        source_manifest.resolve()
    )
    assert context["reference_bindings"]["rights_admission_path"] == str(
        rights_admission.resolve()
    )

    rights_admission_value["provider_training_allowed"] = True
    rights_admission_value["rights_admission_digest"] = (
        prep._canonical_artifact_digest(
            rights_admission_value, digest_field="rights_admission_digest"
        )
    )
    rights_admission.write_text(json.dumps(rights_admission_value), encoding="utf-8")
    context_value = json.loads(context_file.read_text(encoding="utf-8"))
    context_value["references"]["scene"]["rights_admission_digest"] = (
        rights_admission_value["rights_admission_digest"]
    )
    context_file.write_text(json.dumps(context_value), encoding="utf-8")
    with pytest.raises(
        prep.PaidLaneLaunchPreparationError,
        match="native_task_arena_reference_binding_invalid",
    ):
        prep._load_native_context(
            context_file, expected_lane="native_task_arena_construction"
        )


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


def test_native_context_refuses_operational_reference_override(tmp_path: Path) -> None:
    context = {
        "schema_version": prep.NATIVE_CONTEXT_SCHEMA_VERSION,
        "lane": "native_task_arena_construction",
        "team_namespace": "robot-team-a",
        "references": {},
        "operations": {"packet_dir": "/tmp/unbound"},
    }
    source = tmp_path / "context.json"
    source.write_text(json.dumps(context), encoding="utf-8")
    with pytest.raises(prep.PaidLaneLaunchPreparationError):
        prep._load_native_context(
            source, expected_lane="native_task_arena_construction"
        )


def test_semantic_teacher_lane_passes_each_tool_the_argument_shape_it_wants() -> None:
    """Two arguments look interchangeable with a neighbour and are not.

    Both were caught only by running the sequence by hand. The profile builder
    takes the *local publication receipt path* and resolves whatever it is given
    as a filesystem path, so handing it the `r2://` URI the receipt contains
    silently became a relative path under the caller's working directory. The
    rehearsal resolves its lane module against ``src/blueprint_pipeline``, so it
    wants a bare filename and rejects a dotted module path as missing.
    """

    steps = {step.step_id: step for step in prep.LANES["semantic_teacher_image_edit"]}

    profile_argv = list(steps["live_profile"].argv)
    manifest_argument = profile_argv[profile_argv.index("--raw-manifest-uri") + 1]
    assert manifest_argument.endswith("manifest_publication_receipt.v1.json")
    assert "://" not in manifest_argument

    rehearsal_argv = list(steps["terminal_rehearsal"].argv)
    lane_module = rehearsal_argv[rehearsal_argv.index("--lane-module") + 1]
    assert lane_module == "semantic_teacher_image_edit_vast.py"
    assert "." in lane_module and "/" not in lane_module
    assert not lane_module.startswith("blueprint_pipeline")

    publication_argv = list(steps["profile_publication"].argv)
    assert publication_argv[publication_argv.index("--service-account") + 1] == (
        "{service_account}"
    )
    assert publication_argv[publication_argv.index("--service-group") + 1] == (
        "{service_group}"
    )


def test_semantic_retry_inputs_reach_authority_and_profile() -> None:
    steps = {step.step_id: step for step in prep.LANES["semantic_teacher_image_edit"]}
    resolved = {
        "prior_spend_reconciliations": ["/evidence/official-spend.json"],
        "excluded_machine_ids": [76546, 76547],
    }

    authority_argv: list[str] = []
    for flag, context_name in steps["paid_authority"].repeated_argv:
        for value in prep._repeated_values(resolved[context_name]):
            authority_argv.extend((flag, value))
    assert authority_argv == [
        "--prior-spend-reconciliation",
        "/evidence/official-spend.json",
    ]

    dry_run_argv: list[str] = []
    for flag, context_name in steps["allocator_dry_run"].repeated_argv:
        for value in prep._repeated_values(resolved[context_name]):
            dry_run_argv.extend((flag, value))
    assert dry_run_argv == [
        "--semantic-teacher-excluded-machine-id",
        "76546",
        "--semantic-teacher-excluded-machine-id",
        "76547",
    ]

    profile_argv: list[str] = []
    for flag, context_name in steps["live_profile"].repeated_argv:
        for value in prep._repeated_values(resolved[context_name]):
            profile_argv.extend((flag, value))
    assert profile_argv == [
        "--excluded-machine-id",
        "76546",
        "--excluded-machine-id",
        "76547",
    ]


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
