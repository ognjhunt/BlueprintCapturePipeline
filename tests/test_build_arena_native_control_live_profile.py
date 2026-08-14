"""The Arena native-control lane must be launchable from the website.

The allocator has dispatched `adp-isaac-lab-arena-native-control` for some
time, the transport is written, and its bundle got a command line so it could
be rebuilt after a deploy. What was missing was the one artifact that carries a
lane across the website boundary: a live launch profile. Until this builder
existed `tests/test_website_reachable_probe_kinds.py` recorded the lane as
`awaiting_builder`, which is an admission that working code could not be
reached from the product path.

Two things about this lane make its profile different from its siblings, and
both are pinned below.

The allocator does not consume a pre-built bundle here -- it calls
`build_arena_native_control_bundle` itself, from the approval named in argv. So
the bundle digest this profile publishes is only true if the approval it pins
is the one that bundle was built from. Nothing downstream re-checks that, which
makes it this builder's job.

And the whole lane exists to run a *zero-action negative control*. A receipt
that has lost `control_id`, or that records a candidate policy as queried, is
not this control any more, and a profile built from one would launch a paid run
that cannot answer the question it was authorized for.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.adp_founder_sim_protocol import APPROVAL_SCHEMA_VERSION, PROTOCOL_ID
from blueprint_pipeline.adp_isaac_lab_arena_vast import PROBE_KIND
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSPORT = REPO_ROOT / "src" / "blueprint_pipeline" / "adp_isaac_lab_arena_vast.py"
_SPEC = importlib.util.spec_from_file_location(
    "build_arena_native_control_live_profile",
    REPO_ROOT / "scripts" / "build_arena_native_control_live_profile.py",
)
builder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
# `dataclasses` resolves a class's annotations through `sys.modules`, so the
# module has to be registered before it is executed.
sys.modules["build_arena_native_control_live_profile"] = builder
_SPEC.loader.exec_module(builder)

COMMIT = "0" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/protocol.json"
PROTOCOL_DIGEST = "sha256:" + "1" * 64


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    approval_digest: str = "sha256:" + "2" * 64,
    receipt_approval_digest: str | None = None,
    status: str = "ready",
    control_id: str = "arena_zero_action_negative",
    retry_cap: int = 0,
    candidate_policy_queried: bool = False,
    approved: bool = True,
    approval_overrides: dict[str, object] | None = None,
    with_avoidlist: bool = False,
) -> dict[str, Path]:
    """A bundle receipt and the approval it was built from, as the lane makes them."""

    bundle = tmp_path / "adp_arena_provider_runtime_bundle.zip"
    bundle.write_bytes(b"arena-native-control-bundle")

    approval = tmp_path / "founder_sim_approval_receipt.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": APPROVAL_SCHEMA_VERSION,
                "approved": approved,
                "approver_role": "blueprint_founder_sim_owner",
                "protocol_id": PROTOCOL_ID,
                "protocol_digest": PROTOCOL_DIGEST,
                "approval_receipt_digest": approval_digest,
                **(approval_overrides or {}),
            }
        ),
        encoding="utf-8",
    )

    receipt = tmp_path / "adp_arena_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "adp_arena_provider_bundle.v1",
                "status": status,
                "control_id": control_id,
                "retry_cap": retry_cap,
                "candidate_policy_queried": candidate_policy_queried,
                "candidate_outcomes_accessed": False,
                "protocol_digest": PROTOCOL_DIGEST,
                "approval_receipt_digest": (
                    approval_digest
                    if receipt_approval_digest is None
                    else receipt_approval_digest
                ),
                "bundle_path": str(bundle),
                "bundle_sha256": _digest(bundle.read_bytes()),
            }
        ),
        encoding="utf-8",
    )

    paths = {"receipt": receipt, "approval": approval, "bundle": bundle}
    if with_avoidlist:
        avoidlist = tmp_path / "machine_avoidlist.json"
        avoidlist.write_text(json.dumps({"machine_ids": [1234]}), encoding="utf-8")
        paths["avoidlist"] = avoidlist
    return paths


def _build(paths: dict[str, Path], **overrides):
    return builder.build_arena_native_control_live_profile(
        bundle_receipt_path=paths["receipt"],
        approval_path=overrides.pop("approval_path", paths["approval"]),
        source_commit=overrides.pop("source_commit", COMMIT),
        raw_manifest_uri=URI,
        **overrides,
    )


def _argument(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_builds_a_profile_that_passes_every_shared_validator(tmp_path: Path) -> None:
    """Residency, immutable-input, and launch-profile validation all run inside
    the skeleton, so a profile coming back at all is the assertion."""

    profile = _build(_fixture(tmp_path))

    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    assert profile["allocator"]["retry_cap"] == 0
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["watchdog_required"] is True


def test_emits_the_probe_kind_the_allocator_dispatches(tmp_path: Path) -> None:
    """This is the whole point: the allocator branches on this string."""

    argv = _build(_fixture(tmp_path))["allocator"]["argv"]

    assert builder.SPEC.probe_kind == PROBE_KIND
    assert _argument(argv, "--probe-kind") == "adp-isaac-lab-arena-native-control"


def test_carries_the_two_arguments_the_allocator_refuses_without(tmp_path: Path) -> None:
    """`adp_arena_approval` and `adp_job_dir` are named in the branch's own
    missing-argument list, so omitting either is a refusal after a provider has
    been handed over."""

    paths = _fixture(tmp_path)
    argv = _build(paths)["allocator"]["argv"]

    assert _argument(argv, "--adp-arena-approval") == str(paths["approval"].resolve())
    job_dir = _argument(argv, "--adp-job-dir")
    assert job_dir.startswith("{launch_run_root}/"), (
        "a job directory naming a real path names one machine's directory"
    )


def test_the_declared_spend_is_the_worst_case_this_profile_can_reach(
    tmp_path: Path,
) -> None:
    """Rate times TTL. The allocator is handed the same number it published, so
    an operator cannot declare a ceiling the profile is able to exceed."""

    profile = _build(_fixture(tmp_path), max_hourly_rate_usd=1.0, hard_ttl_seconds=14_400)

    assert profile["allocator"]["max_spend_usd"] == 4.0
    argv = profile["allocator"]["argv"]
    assert _argument(argv, "--adp-max-spend-usd") == "4.0"
    assert _argument(argv, "--adp-max-hourly-rate-usd") == "1.0"
    assert _argument(argv, "--adp-hard-ttl-seconds") == "14400"


def test_refuses_a_ttl_outside_the_band_the_allocator_accepts(tmp_path: Path) -> None:
    """`adp_arena_hard_ttl_invalid` is raised by the allocator, not here, and
    only after the launch has been admitted."""

    for seconds in (1_799, 14_401):
        # One test rather than a parametrize: an empty parameter set reports as
        # a skip, and the CPU full lane blocks on any skipped test.
        with pytest.raises(TaskEvaluationLaunchError, match="hard_ttl_out_of_band"):
            _build(_fixture(tmp_path), hard_ttl_seconds=seconds)


def test_refuses_a_budget_the_allocator_would_call_invalid(tmp_path: Path) -> None:
    """The allocator requires `0 < rate <= spend`. Below an hour of TTL the
    worst case falls under the hourly rate and that stops being true."""

    with pytest.raises(TaskEvaluationLaunchError, match="arena_budget_invalid"):
        _build(_fixture(tmp_path), max_hourly_rate_usd=4.0, hard_ttl_seconds=1_800)


def test_refuses_an_approval_the_pinned_bundle_was_not_built_from(
    tmp_path: Path,
) -> None:
    """The allocator rebuilds the bundle from this approval at launch time, so a
    mismatched pair publishes a bundle digest describing something the run will
    never produce -- and nothing downstream compares them."""

    paths = _fixture(tmp_path, receipt_approval_digest="sha256:" + "9" * 64)

    with pytest.raises(TaskEvaluationLaunchError, match="arena_approval_not_the_bundle_approval"):
        _build(paths)


def test_refuses_an_approval_that_does_not_approve(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="arena_approval_not_approved"):
        _build(_fixture(tmp_path, approved=False))


def test_refuses_a_missing_approval(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(TaskEvaluationLaunchError, match="arena_approval_missing"):
        _build(paths, approval_path=tmp_path / "absent.json")


def test_refuses_an_approval_for_some_other_protocol(tmp_path: Path) -> None:
    """The allocator's own admission raises on each of these, but only after the
    profile has been published and a launch has been started against it."""

    cases = {
        "arena_approval_schema_invalid": {"schema_version": "some.other.v1"},
        "arena_approval_protocol_id_mismatch": {"protocol_id": "adp-some-other-protocol"},
        "arena_approval_protocol_digest_mismatch": {
            "protocol_digest": "sha256:" + "8" * 64
        },
    }
    # Iterated inside one test rather than parametrized: an empty parameter set
    # reports as a skip, and the CPU full lane blocks on any skipped test.
    for index, (blocker, override) in enumerate(cases.items()):
        root = tmp_path / f"case{index}"
        root.mkdir()
        with pytest.raises(TaskEvaluationLaunchError, match=blocker):
            _build(_fixture(root, approval_overrides=override))


def test_refuses_a_receipt_that_is_no_longer_the_zero_action_control(
    tmp_path: Path,
) -> None:
    """The lane's entire claim is a zero-action negative control. A receipt that
    lost the control id, or that records a policy query, would spend real money
    on a run that cannot answer what it was authorized for."""

    with pytest.raises(TaskEvaluationLaunchError, match="arena_bundle_control_id_mismatch"):
        _build(_fixture(tmp_path, control_id="arena_scored_rollout"))

    other = tmp_path / "queried"
    other.mkdir()
    with pytest.raises(
        TaskEvaluationLaunchError, match="arena_bundle_candidate_policy_queried"
    ):
        _build(_fixture(other, candidate_policy_queried=True))


def test_refuses_a_receipt_that_permits_a_retry(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="arena_bundle_retry_cap_not_zero"):
        _build(_fixture(tmp_path, retry_cap=1))


def test_refuses_an_unready_receipt(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="bundle_receipt_not_ready"):
        _build(_fixture(tmp_path, status="blocked"))


def test_an_avoidlist_is_optional_and_pinned_when_supplied(tmp_path: Path) -> None:
    """The allocator digests the avoidlist into its allocation binding, so which
    machines are excluded is part of what was authorized."""

    without = _build(_fixture(tmp_path))["allocator"]["argv"]
    assert "--adp-machine-avoidlist" not in without

    other = tmp_path / "with"
    other.mkdir()
    paths = _fixture(other, with_avoidlist=True)
    profile = _build(paths, machine_avoidlist_path=paths["avoidlist"])

    argv = profile["allocator"]["argv"]
    assert _argument(argv, "--adp-machine-avoidlist") == str(paths["avoidlist"].resolve())
    pinned = {row["name"]: row for row in profile["immutable_inputs"]}
    assert pinned["arena_machine_avoidlist"]["digest"] == _digest(
        paths["avoidlist"].read_bytes()
    )


def test_refuses_an_avoidlist_that_is_not_there(tmp_path: Path) -> None:
    """`adp_arena_machine_avoidlist_missing` is otherwise raised at the paid boundary."""

    paths = _fixture(tmp_path)

    with pytest.raises(TaskEvaluationLaunchError, match="arena_machine_avoidlist_missing"):
        _build(paths, machine_avoidlist_path=tmp_path / "absent-avoidlist.json")


def test_pins_the_approval_and_the_bundle_by_digest(tmp_path: Path) -> None:
    """The approval is the only input that decides what the allocator builds."""

    paths = _fixture(tmp_path)
    pinned = {row["name"]: row for row in _build(paths)["immutable_inputs"]}

    assert set(pinned) >= {"source_bundle_manifest", "evaluation_run_spec"}
    assert pinned["founder_sim_approval_receipt"]["path"] == str(paths["approval"].resolve())
    assert pinned["founder_sim_approval_receipt"]["digest"] == _digest(
        paths["approval"].read_bytes()
    )
    assert pinned["arena_native_control_bundle"]["path"] == str(paths["bundle"].resolve())
    assert pinned["arena_native_control_bundle"]["digest"] == _digest(
        paths["bundle"].read_bytes()
    )


def test_revision_yields_a_distinct_profile_id(tmp_path: Path) -> None:
    """Published profiles are immutable, so a rebuild needs an id of its own."""

    paths = _fixture(tmp_path)

    base = _build(paths)["profile_id"]
    revised = _build(paths, revision="b2")["profile_id"]

    assert revised == f"{base}-b2"


def test_the_flag_table_is_the_builder_call_signature() -> None:
    """A flag that is missing from the parser does not fail -- it silently fixes
    a paid decision at its default. Deriving the check from the callee is what
    makes that impossible to do by accident."""

    signature = inspect.signature(builder.build_arena_native_control_live_profile)
    parameters = {
        name: value
        for name, value in signature.parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert {flag.kwarg for flag in builder.FLAGS.values()} == set(parameters), (
        "the parser and the builder disagree about what this lane decides"
    )
    assert {flag.kwarg for flag in builder.FLAGS.values() if flag.required} == {
        name
        for name, value in parameters.items()
        if value.default is inspect.Parameter.empty
    }


def test_every_flag_reaches_the_profile_it_decides(tmp_path: Path) -> None:
    """End to end through the real parser: each value an operator can pass has
    to be observable in the profile, or the flag was accepted and dropped."""

    paths = _fixture(tmp_path, with_avoidlist=True)
    output = tmp_path / "profile.json"

    assert (
        builder.main(
            [
                "--bundle-receipt", str(paths["receipt"]),
                "--approval", str(paths["approval"]),
                "--source-commit", COMMIT,
                "--raw-manifest-uri", URI,
                "--machine-avoidlist", str(paths["avoidlist"]),
                "--revision", "c3",
                "--max-hourly-rate-usd", "0.5",
                "--hard-ttl-seconds", "7200",
                "--output", str(output),
            ]
        )
        == 0
    )

    profile = json.loads(output.read_text(encoding="utf-8"))
    argv = profile["allocator"]["argv"]
    assert profile["profile_id"].endswith("-c3")
    assert _argument(argv, "--expected-source-commit") == COMMIT
    assert _argument(argv, "--adp-arena-approval") == str(paths["approval"].resolve())
    assert _argument(argv, "--adp-machine-avoidlist") == str(paths["avoidlist"].resolve())
    assert _argument(argv, "--adp-max-hourly-rate-usd") == "0.5"
    assert _argument(argv, "--adp-hard-ttl-seconds") == "7200"
    assert profile["allocator"]["max_spend_usd"] == 1.0
    assert profile["evaluation_run_spec"]["uri"] == URI


def test_a_refused_profile_is_reported_rather_than_written(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"

    assert (
        builder.main(
            [
                "--bundle-receipt", str(_fixture(tmp_path, status="blocked")["receipt"]),
                "--approval", str(tmp_path / "founder_sim_approval_receipt.json"),
                "--source-commit", COMMIT,
                "--raw-manifest-uri", URI,
                "--output", str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def _terminal_result_fields() -> set[str]:
    """The fields the transport's terminal result actually carries.

    Read from the transport's own source by field name rather than by looking
    for the helper that fills them in: a helper can be renamed, moved, or
    called and have its output dropped, and none of that would change what the
    allocator writes to `result.json` -- which is the only thing the terminal
    contract reads.
    """

    tree = ast.parse(TRANSPORT.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_arena_native_control_vast"
    )
    terminal = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "status"
            and isinstance(value, ast.IfExp)
            for key, value in zip(node.keys, node.values)
        )
    ]
    assert len(terminal) == 1, (
        "expected exactly one result whose status can be `completed`; the "
        "transport's shape changed and this check no longer reads it"
    )
    return {
        key.value
        for key in terminal[0].keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def test_the_terminal_contract_asks_only_for_evidence_this_lane_emits(
    tmp_path: Path,
) -> None:
    """A required path field the lane never writes turns every completed run
    into a failed one, after the money is spent."""

    required = set(_build(_fixture(tmp_path))["terminal_contract"]["required_path_fields"])

    assert required, "a terminal contract that requires nothing proves nothing"
    assert required <= _terminal_result_fields(), (
        f"the terminal contract requires {sorted(required - _terminal_result_fields())}, "
        "which the transport's completed result does not carry"
    )
