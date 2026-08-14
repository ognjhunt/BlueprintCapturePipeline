"""The new-site diagnostic canary is reachable from the website, or it is not.

The allocator has executed this probe kind for a long time and no live profile
builder emitted it, so `tests/test_website_reachable_probe_kinds.py` carried it
as `awaiting_builder`: work that exists, runs, and cannot be triggered from the
product path.

The lane is unusual in one way that decides most of what is pinned here. Its
allocator branch is shared with the frozen `openpi-policy-ranking` campaign and
`--probe-kind` is *not* forwarded to the transport -- which lane actually runs
is decided by the schema of the input bundle receipt
(`build_openpi_policy_ranking_gpu_admission` reads `execution_mode` off it).
A profile labelled `new-site-diagnostic-canary` that pinned a full-campaign
receipt would therefore launch the frozen campaign under the canary's name, and
nothing downstream would notice. That is the first thing below.

The rest are the paid decisions this branch takes from an argparse default when
a flag is omitted: TTL (14,400 s), campaign spend ($3.00), the four
execute-only campaign arguments, and the Vast hourly ceiling the execute path
re-derives for itself.

Reads and writes only temporary bytes; rents nothing and mutates no provider.
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

from blueprint_pipeline.host_resident_launch_inputs import LAUNCH_INPUT_ROOTS_ENV
from blueprint_pipeline.openpi_policy_ranking_gpu_admission import (
    MAX_TTL_SECONDS,
    NEW_SITE_CANARY_PROBE_KIND,
    VAST_DEFAULT_MAX_HOURLY_RATE_USD,
)
from blueprint_pipeline.openpi_policy_ranking_runpod import CANARY_NAME_PREFIX
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256
from blueprint_pipeline.production_gpu_campaign_budget import (
    AUTHORIZED_GPU_WALL_CAP_SECONDS,
    AUTHORIZED_SPEND_CAP_USD,
    SCHEMA_VERSION as BUDGET_LEDGER_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts" / "build_new_site_diagnostic_canary_live_profile.py"
ALLOCATOR = REPO_ROOT / "src" / "blueprint_pipeline" / "paid_resource_allocator.py"

_SPEC = importlib.util.spec_from_file_location(
    "build_new_site_diagnostic_canary_live_profile", BUILDER_PATH
)
builder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
# Registered before execution because the module declares dataclasses, and
# `dataclasses` resolves a field's annotation through `sys.modules[__module__]`.
# The reachability gate loads every builder the same way for the same reason.
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)

COMMIT = "0" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/canary_protocol.json"
CANARY_RECEIPT_SCHEMA = "new_site_diagnostic_canary_input_receipt.v2"
CANARY_MANIFEST_SCHEMA = "new_site_diagnostic_canary_input.v2"


@pytest.fixture(autouse=True)
def _control_plane_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge residency against the fixture, not against whatever this host has.

    A profile is refused when it binds a path outside the roots the control
    plane owns. Pointing the roots at `tmp_path` makes that check real here
    instead of silently disabled, which is what it is on a workstation with no
    `/var/lib/blueprint`.
    """

    monkeypatch.setenv(LAUNCH_INPUT_ROOTS_ENV, str(tmp_path))


def _identity(manifest: dict) -> dict:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _private_url_file(path: Path, url: str, *, mode: int = 0o600) -> Path:
    path.write_text(url + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _fixture(
    root: Path,
    *,
    receipt_schema: str = CANARY_RECEIPT_SCHEMA,
    manifest_schema: str = CANARY_MANIFEST_SCHEMA,
    receipt_status: str = "completed",
    manifest_overrides: dict | None = None,
    url_file_mode: int = 0o600,
    ledger_state: dict | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "new_site_canary_input_bundle.zip"
    bundle.write_bytes(b"new-site-diagnostic-canary-input-bundle")
    manifest = _identity(
        {
            "schema_version": manifest_schema,
            "experiment_id": "policy_ranking_new_site_smoke_interiorgs_0787_20260729_v1",
            "arm_id": "skeleton_only",
            "scene_id": "interiorgs_0787",
            "task_instruction": "pick up the can",
            "policy_id": "pi05_droid_joint_position",
            "variant": "center",
            "protocol_sha256": "a" * 64,
            "background_sha256": "b" * 64,
            "raw_3dgs_included": False,
            "redistribution_authorized": False,
            "label_free": True,
            "purpose": "private_internal_noncommercial_new_site_diagnostic_canary",
            **(manifest_overrides or {}),
        }
    )
    receipt = root / "canary_input_bundle_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": receipt_schema,
                "status": receipt_status,
                "bundle_path": str(bundle),
                # The lane writes a bare hex digest; the launch contract wants
                # the `sha256:` form. Recorded here exactly as the lane records
                # it, so the projection is what is under test.
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "bundle_size_bytes": bundle.stat().st_size,
                "manifest": manifest,
            }
        ),
        encoding="utf-8",
    )
    release = root / "openpi_gpu_release.json"
    release.write_text(
        json.dumps(
            {
                "schema_version": "openpi_policy_ranking_gpu_release.v1",
                "status": "passed",
                "source_commit": COMMIT,
                "resolved_digest_ref": "example.invalid/openpi@sha256:" + "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    preflight = root / "openpi_provider_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "openpi_policy_ranking_provider_preflight.v2",
                "status": "verified",
                "provider": "vast",
                "container_disk_bytes": 100 * 1024**3,
            }
        ),
        encoding="utf-8",
    )
    ledger = root / "campaign_budget.json"
    if ledger_state is not None:
        ledger.write_text(json.dumps(ledger_state), encoding="utf-8")
    return {
        "receipt": receipt,
        "bundle": bundle,
        "release": release,
        "preflight": preflight,
        "ledger": ledger,
        "input_url": _private_url_file(
            root / "input_secret_url", "https://example.invalid/in", mode=url_file_mode
        ),
        "put_url": _private_url_file(
            root / "output_put_secret_url", "https://example.invalid/put", mode=url_file_mode
        ),
        "get_url": _private_url_file(
            root / "output_get_secret_url", "https://example.invalid/get", mode=url_file_mode
        ),
    }


def _build(paths: dict[str, Path], **overrides) -> dict:
    keywords = {
        "bundle_receipt_path": paths["receipt"],
        "release_evidence_path": paths["release"],
        "provider_preflight_path": paths["preflight"],
        "input_secret_url_file": paths["input_url"],
        "output_secret_put_url_file": paths["put_url"],
        "output_secret_get_url_file": paths["get_url"],
        "campaign_budget_ledger_path": paths["ledger"],
        "campaign_initial_spent_usd": 1.5,
        "campaign_initial_used_gpu_seconds": 3_600,
        "source_commit": COMMIT,
        "raw_manifest_uri": URI,
    }
    keywords.update(overrides)
    return builder.build_new_site_diagnostic_canary_live_profile(**keywords)


def _argument(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --------------------------------------------------------------------------
# The lane runs at all
# --------------------------------------------------------------------------


def test_builds_a_live_profile_that_passes_every_validator(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path))

    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    assert profile["allocator"]["retry_cap"] == 0
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["terminal_contract"]["required_path_fields"] == [
        "teardown_manifest_path",
        "artifact_manifest_path",
    ]


def test_emits_the_canary_probe_kind(tmp_path: Path) -> None:
    """This is what makes the lane reachable from the website at all."""

    argv = _build(_fixture(tmp_path))["allocator"]["argv"]

    assert _argument(argv, "--probe-kind") == NEW_SITE_CANARY_PROBE_KIND
    assert builder.SPEC.probe_kind == NEW_SITE_CANARY_PROBE_KIND


# --------------------------------------------------------------------------
# The lane that runs is the lane the profile names
# --------------------------------------------------------------------------


def test_refuses_an_input_receipt_belonging_to_the_frozen_campaign(tmp_path: Path) -> None:
    """`--probe-kind` is never forwarded; the receipt schema picks the lane.

    `build_openpi_policy_ranking_gpu_admission` reads `execution_mode` off the
    input bundle receipt, so a full-campaign receipt under this probe kind runs
    the frozen policy-ranking campaign and reports it as a canary.
    """

    paths = _fixture(
        tmp_path,
        receipt_schema="openpi_policy_ranking_gpu_input_bundle_receipt.v1",
        manifest_schema="openpi_policy_ranking_gpu_input_bundle.v2",
    )

    with pytest.raises(TaskEvaluationLaunchError, match="canary_input_receipt_schema_invalid"):
        _build(paths)


def test_refuses_an_input_bundle_whose_rights_freeze_is_broken(tmp_path: Path) -> None:
    """The allocator names this after a provider has been handed over."""

    paths = _fixture(tmp_path, manifest_overrides={"redistribution_authorized": True})

    with pytest.raises(TaskEvaluationLaunchError, match="canary_input_freeze_invalid"):
        _build(paths)


def test_refuses_an_input_bundle_whose_manifest_identity_does_not_hold(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["manifest"]["task_instruction"] = "some other task"
    paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        TaskEvaluationLaunchError, match="canary_input_manifest_sha256_invalid"
    ):
        _build(paths)


def test_refuses_a_receipt_the_lane_never_finished(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, receipt_status="blocked")

    with pytest.raises(TaskEvaluationLaunchError, match="bundle_receipt_not_ready"):
        _build(paths)


# --------------------------------------------------------------------------
# Paid decisions that an omitted flag would take from an argparse default
# --------------------------------------------------------------------------


def test_the_allocators_own_required_arguments_are_all_supplied(tmp_path: Path) -> None:
    """Read from the allocator branch rather than restated here.

    The branch builds its `missing` list from two literal tuples of argparse
    destinations -- one always, one only under `--execute`. An omission in the
    second is invisible until a live launch, which is when this lane's
    execute-only campaign arguments would have been discovered.
    """

    required = _required_allocator_destinations()
    assert len(required) >= 8, "the allocator branch extraction stopped working"

    argv = _build(_fixture(tmp_path))["allocator"]["argv"]
    flags = {item for item in argv if item.startswith("--")}
    missing = sorted(
        destination
        for destination in required
        if "--" + destination.replace("_", "-") not in flags
    )

    assert not missing, f"the allocator would refuse this profile for: {missing}"


def test_carries_this_profiles_ttl_and_spend_rather_than_the_lane_defaults(
    tmp_path: Path,
) -> None:
    """`--openpi-hard-ttl-seconds` and `--openpi-max-spend-usd` are what the
    campaign actually reserves and bills against. They default to 14,400 s and
    $3.00, so omitting them silently detaches the run from the profile that
    authorized it."""

    profile = _build(_fixture(tmp_path), max_hourly_rate_usd=1.0, hard_ttl_seconds=3_600)
    argv = profile["allocator"]["argv"]

    assert int(_argument(argv, "--openpi-hard-ttl-seconds")) == 3_600
    assert int(_argument(argv, "--openpi-hard-ttl-seconds")) == (
        profile["allocator"]["hard_ttl_seconds"]
    )
    assert float(_argument(argv, "--openpi-max-spend-usd")) == (
        profile["allocator"]["max_spend_usd"]
    )
    assert profile["allocator"]["max_spend_usd"] == 1.0


def test_declared_spend_is_rate_times_ttl(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path), max_hourly_rate_usd=1.0, hard_ttl_seconds=7_200)

    assert profile["allocator"]["max_spend_usd"] == 2.0


def test_the_defaults_reproduce_the_lanes_own_frozen_ceiling(tmp_path: Path) -> None:
    """0.75/hr x 4h is the $3.00 the allocator's own default declares, which is
    where that number came from. Reaching it by derivation rather than by
    restating it is what keeps the two from drifting apart."""

    profile = _build(_fixture(tmp_path))["allocator"]

    assert profile["hard_ttl_seconds"] == MAX_TTL_SECONDS
    assert profile["max_spend_usd"] == 3.0


def test_refuses_an_hourly_rate_below_the_frozen_vast_ceiling(tmp_path: Path) -> None:
    """The execute path re-collects the Vast preflight without a rate argument,
    so the admission compares the declared spend against
    `VAST_DEFAULT_MAX_HOURLY_RATE_USD` regardless of what the profile says. A
    lower rate here yields `openpi_gpu_ttl_cost_exceeds_max_spend` after the
    launch has already started."""

    with pytest.raises(TaskEvaluationLaunchError, match="hourly_rate_below_vast_lane_ceiling"):
        _build(
            _fixture(tmp_path),
            max_hourly_rate_usd=VAST_DEFAULT_MAX_HOURLY_RATE_USD / 2,
        )


def test_refuses_a_ttl_above_the_admissions_ceiling(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="hard_ttl_out_of_band"):
        _build(_fixture(tmp_path), hard_ttl_seconds=MAX_TTL_SECONDS + 1)


def test_refuses_a_ttl_shorter_than_the_workers_own_transport_windows(
    tmp_path: Path,
) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="hard_ttl_out_of_band"):
        _build(_fixture(tmp_path), hard_ttl_seconds=300)


# --------------------------------------------------------------------------
# Refusals that would otherwise be raised mid-launch, uncaught
# --------------------------------------------------------------------------


def test_pod_name_stays_inside_the_watchdog_scope(tmp_path: Path) -> None:
    """The teardown watchdog only reaps instances under this name prefix, and
    the campaign refuses any other pod name outright."""

    profile = _build(_fixture(tmp_path))

    assert profile["profile_id"].startswith(CANARY_NAME_PREFIX)
    assert _argument(profile["allocator"]["argv"], "--pod-name") == profile["profile_id"]


def test_refuses_a_secret_url_file_the_worker_would_reject(tmp_path: Path) -> None:
    """`_read_private_https_url` demands mode 0600 and raises, uncaught, at
    launch. A world-readable ledger has already cost this program one run."""

    with pytest.raises(TaskEvaluationLaunchError, match="secret_url_file_not_private"):
        _build(_fixture(tmp_path, url_file_mode=0o644))


def test_refuses_a_campaign_cap_above_the_standing_authorization(tmp_path: Path) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="campaign_cap_exceeds_authorization"):
        _build(
            _fixture(tmp_path),
            campaign_total_spend_cap_usd=AUTHORIZED_SPEND_CAP_USD + 1,
        )


def test_refuses_a_campaign_identity_the_existing_ledger_disagrees_with(
    tmp_path: Path,
) -> None:
    """`ProductionGpuCampaignBudget` re-validates the identity of a ledger it
    finds on disk and raises before any reservation is taken."""

    paths = _fixture(
        tmp_path,
        ledger_state={
            "schema_version": BUDGET_LEDGER_SCHEMA_VERSION,
            "total_spend_cap_usd": AUTHORIZED_SPEND_CAP_USD,
            "combined_gpu_wall_cap_seconds": AUTHORIZED_GPU_WALL_CAP_SECONDS,
            "initial_spent_usd": 9.25,
            "initial_used_gpu_seconds": 3_600,
            "reservations": [],
        },
    )

    with pytest.raises(
        TaskEvaluationLaunchError, match="campaign_budget_ledger_identity_mismatch"
    ):
        _build(paths)


def test_accepts_a_ledger_whose_identity_matches(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        ledger_state={
            "schema_version": BUDGET_LEDGER_SCHEMA_VERSION,
            "total_spend_cap_usd": AUTHORIZED_SPEND_CAP_USD,
            "combined_gpu_wall_cap_seconds": AUTHORIZED_GPU_WALL_CAP_SECONDS,
            "initial_spent_usd": 1.5,
            "initial_used_gpu_seconds": 3_600,
            "reservations": [],
        },
    )

    assert _build(paths)["execution_admission"]["live_enabled"] is True


# --------------------------------------------------------------------------
# Profile shape
# --------------------------------------------------------------------------


def test_immutable_inputs_pin_the_receipt_the_bundle_and_the_image_release(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    rows = {row["name"]: row for row in _build(paths)["immutable_inputs"]}

    assert set(rows) >= {"source_bundle_manifest", "evaluation_run_spec"}
    assert rows["source_bundle_manifest"]["path"] == str(paths["receipt"].resolve())
    assert rows["new_site_canary_input_bundle"]["path"] == str(paths["bundle"].resolve())
    assert rows["new_site_canary_input_bundle"]["digest"] == (
        "sha256:" + hashlib.sha256(paths["bundle"].read_bytes()).hexdigest()
    )
    assert rows["evaluation_run_spec"]["path"] == str(paths["release"].resolve())


def test_the_secret_url_files_are_never_digest_bound(tmp_path: Path) -> None:
    """A presigned URL rotates. Pinning its digest would make every rotation a
    profile rebuild, and would put a secret-bearing path under an immutable
    contract it cannot keep."""

    paths = _fixture(tmp_path)
    profile = _build(paths)
    bound = {row["path"] for row in profile["immutable_inputs"]}

    for name in ("input_url", "put_url", "get_url"):
        assert str(paths[name].resolve()) not in bound


def test_revision_yields_a_distinct_profile_id(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    base = _build(paths)["profile_id"]
    revised = _build(paths, revision="b2")["profile_id"]

    assert revised == f"{base}-b2"


def test_the_profile_binds_no_path_outside_the_control_plane_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The residency half of the skeleton is on for this lane too."""

    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    paths = _fixture(outside)
    monkeypatch.setenv(LAUNCH_INPUT_ROOTS_ENV, str(tmp_path))

    with pytest.raises(TaskEvaluationLaunchError, match="host_resident_input"):
        _build(paths)


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_the_residency_layers_copy_of_the_receipt_schema_cannot_drift() -> None:
    """`host_resident_launch_inputs` spells the schema rather than importing it,
    because importing the lane would drag NumPy, Pillow, and MuJoCo into every
    builder. This is what makes the copy safe."""

    from blueprint_pipeline import host_resident_launch_inputs
    from blueprint_pipeline.new_site_diagnostic_canary_gpu import (
        INPUT_RECEIPT_SCHEMA_VERSION,
    )

    assert host_resident_launch_inputs.NEW_SITE_CANARY_RECEIPT_SCHEMA == (
        INPUT_RECEIPT_SCHEMA_VERSION
    )


def test_the_flag_table_is_the_call_signature() -> None:
    """A keyword with no flag is a paid decision fixed at its default."""

    keywords = {
        name
        for name, value in inspect.signature(
            builder.build_new_site_diagnostic_canary_live_profile
        ).parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert keywords, "the builder takes no keyword arguments; the check is vacuous"
    assert keywords == set(builder.PARAMS), (
        "flag table and call signature disagree: "
        f"unsupplied={sorted(keywords - set(builder.PARAMS))} "
        f"dead={sorted(set(builder.PARAMS) - keywords)}"
    )


def test_every_flag_the_table_declares_is_actually_parsed() -> None:
    parser = builder.build_parser()
    declared = {param.flag for param in builder.PARAMS.values()}
    parsed = {
        option
        for action in parser._actions  # noqa: SLF001 - argparse exposes no public view
        for option in action.option_strings
    }

    assert declared <= parsed


def test_the_command_line_writes_a_profile(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "profile.json"

    code = builder.main(
        [
            "--bundle-receipt", str(paths["receipt"]),
            "--release-evidence", str(paths["release"]),
            "--provider-preflight", str(paths["preflight"]),
            "--input-secret-url-file", str(paths["input_url"]),
            "--output-secret-put-url-file", str(paths["put_url"]),
            "--output-secret-get-url-file", str(paths["get_url"]),
            "--campaign-budget-ledger", str(paths["ledger"]),
            "--campaign-initial-spent-usd", "1.5",
            "--campaign-initial-used-gpu-seconds", "3600",
            "--source-commit", COMMIT,
            "--raw-manifest-uri", URI,
            "--output", str(output),
        ]
    )

    assert code == 0
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["profile_id"].startswith(CANARY_NAME_PREFIX)
    assert _argument(profile["allocator"]["argv"], "--probe-kind") == (
        NEW_SITE_CANARY_PROBE_KIND
    )


def test_the_command_line_fails_closed_without_writing(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, receipt_status="blocked")
    output = tmp_path / "profile.json"

    code = builder.main(
        [
            "--bundle-receipt", str(paths["receipt"]),
            "--release-evidence", str(paths["release"]),
            "--provider-preflight", str(paths["preflight"]),
            "--input-secret-url-file", str(paths["input_url"]),
            "--output-secret-put-url-file", str(paths["put_url"]),
            "--output-secret-get-url-file", str(paths["get_url"]),
            "--campaign-budget-ledger", str(paths["ledger"]),
            "--campaign-initial-spent-usd", "1.5",
            "--campaign-initial-used-gpu-seconds", "3600",
            "--source-commit", COMMIT,
            "--raw-manifest-uri", URI,
            "--output", str(output),
        ]
    )

    assert code == 2
    assert not output.exists()


# --------------------------------------------------------------------------
# Helpers that read the allocator rather than restating it
# --------------------------------------------------------------------------


def _required_allocator_destinations() -> set[str]:
    """The argparse destinations this probe kind's branch demands.

    Extracted from the allocator's own source, the way
    `test_website_reachable_probe_kinds` extracts its dispatch table, so a
    newly required argument fails here instead of at a live launch.
    """

    tree = ast.parse(ALLOCATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if not isinstance(left, ast.Attribute) or left.attr != "probe_kind":
            continue
        names: set[str] = set()
        for comparator in node.test.comparators:
            elements = (
                comparator.elts
                if isinstance(comparator, (ast.Set, ast.Tuple, ast.List))
                else [comparator]
            )
            names.update(e.id for e in elements if isinstance(e, ast.Name))
        if "NEW_SITE_CANARY_PROBE_KIND" not in names:
            continue
        found: set[str] = set()
        for inner in ast.walk(node):
            if not isinstance(inner, (ast.GeneratorExp, ast.ListComp)):
                continue
            source = inner.generators[0].iter
            if not isinstance(source, (ast.Tuple, ast.List)):
                continue
            found.update(
                element.value
                for element in source.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
        return found
    return set()
