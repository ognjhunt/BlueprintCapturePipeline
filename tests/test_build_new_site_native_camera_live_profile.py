"""The fresh-site native camera lane must be launchable from the website.

`new-site-native-camera` had a bundle, a release contract, a transport, and an
allocator branch, and no launch profile -- the one thing that carries a lane
across the website boundary. It read `awaiting_builder` in
`tests/test_website_reachable_probe_kinds.py`.

Two of the checks here are derived rather than written down, because a
hand-written list is how this lane class keeps failing:

* the flag table is read back out of the builder's own signature, so a
  parameter without a flag cannot be added; and
* the allocator arguments are read out of the allocator's own call, so an
  argument that only matters under `--execute` cannot be silently left at its
  default -- which is how a paid decision gets fixed by omission.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.host_resident_launch_inputs import (
    LAUNCH_INPUT_ROOTS_ENV,
    resolve_host_resident_bundle_receipt,
)
from blueprint_pipeline.isaac_worker_image_manifest import (
    SCHEMA_VERSION as ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_admission import (
    CANARY_NAME_PREFIX,
    PROBE_KIND,
    build_native_camera_gpu_release_evidence,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle import (
    BUNDLE_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts" / "build_new_site_native_camera_live_profile.py"
ALLOCATOR_PATH = REPO_ROOT / "src" / "blueprint_pipeline" / "paid_resource_allocator.py"
COMMIT = "c" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/native-camera-request.json"

_SPEC = importlib.util.spec_from_file_location(
    "build_new_site_native_camera_live_profile", BUILDER_PATH
)
builder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
# Registered before execution because the builder defines dataclasses, and
# `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
sys.modules[_SPEC.name] = builder
_SPEC.loader.exec_module(builder)


def _private(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def _fixture(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """A staged native camera job: bundle, receipt, release, preflight, secrets."""

    job = tmp_path / "native-camera-job"
    job.mkdir()

    bundle = job / "nvidia_warehouse_native_camera_gpu_bundle.zip"
    bundle.write_bytes(b"native-camera-bundle-archive")
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_commit": overrides.get("manifest_commit", COMMIT),
        "dataset_revision": "d" * 40,
        "asset_count": 4,
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "purpose": "private_internal_nvidia_warehouse_native_camera_canary",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": overrides.get("receipt_status", "completed"),
        "bundle_path": str(bundle),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "bundle_size_bytes": bundle.stat().st_size,
        "manifest": manifest,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = job / "nvidia_warehouse_native_camera_gpu_bundle_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    release = build_native_camera_gpu_release_evidence(
        image_manifest={
            "schema_version": ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION,
            "status": "completed",
            "resolved_digest_ref": "registry.example/isaac-eval-worker@sha256:" + "e" * 64,
            "runnable_platform": "linux/amd64",
            "raw_secret_values_recorded": False,
            "worker_build_identity": {
                "status": "verified",
                "source_commit": overrides.get("release_commit", COMMIT),
                "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
                "worker_image_family": "isaac-eval-worker",
                "isaac_sim_major_version": 6,
            },
        },
        expected_source_commit=overrides.get("release_commit", COMMIT),
    )
    release_path = job / "native_camera_gpu_release_evidence.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")

    preflight = {
        "schema_version": "openpi_policy_ranking_provider_preflight.v2",
        "status": "verified",
        "provider": "vast",
        "provider_api_verified": True,
        "provider_inventory_verified_zero": True,
        "single_gpu_available": True,
        "gpu_type_id": "RTX_4090",
        "gpu_memory_bytes": 24 * 1024**3,
        "container_disk_bytes": overrides.get("container_disk_bytes", 96 * 1024**3),
        "on_demand_price_usd_per_hour": overrides.get("hourly_price", 0.4),
        "observed_at_epoch": 1_760_000_000.0,
    }
    preflight_path = job / "native_camera_provider_preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    secrets = job / "secrets"
    secrets.mkdir()
    bundle_url = _private(secrets / "bundle-url.txt", "https://example.invalid/in?sig=1")
    put_url = _private(secrets / "output-put-url.txt", "https://example.invalid/put?sig=1")
    get_url = _private(secrets / "output-get-url.txt", "https://example.invalid/get?sig=1")

    ledger = job / "campaign" / "native_camera_campaign_budget.json"
    ledger.parent.mkdir()

    return {
        "bundle": bundle,
        "bundle_receipt_path": receipt_path,
        "release_evidence_path": release_path,
        "preflight_bundle_path": preflight_path,
        "provider_bundle_url_file": bundle_url,
        "provider_output_put_url_file": put_url,
        "provider_output_get_url_file": get_url,
        "campaign_budget_ledger": ledger,
    }


def _build(paths: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "bundle_receipt_path": paths["bundle_receipt_path"],
        "release_evidence_path": paths["release_evidence_path"],
        "preflight_bundle_path": paths["preflight_bundle_path"],
        "provider_bundle_url_file": paths["provider_bundle_url_file"],
        "provider_output_put_url_file": paths["provider_output_put_url_file"],
        "provider_output_get_url_file": paths["provider_output_get_url_file"],
        "campaign_budget_ledger": paths["campaign_budget_ledger"],
        "campaign_initial_spent_usd": 0.0,
        "campaign_initial_used_gpu_seconds": 0,
        "source_commit": COMMIT,
        "raw_manifest_uri": URI,
    }
    kwargs.update(overrides)
    return builder.build_new_site_native_camera_live_profile(**kwargs)


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_builds_a_publishable_zero_retry_native_camera_profile(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path))
    argv = profile["allocator"]["argv"]

    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["allocator"]["retry_cap"] == 0
    assert profile["allocator"]["subcommand"] == "gpu-canary"
    assert _argv_value(argv, "--probe-kind") == PROBE_KIND
    assert _argv_value(argv, "--provider") == "vast"
    assert profile["terminal_contract"]["required_values"] == {
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
    }
    assert profile["required_controls"]["provider_zero_required"] is True
    assert "--execute" not in argv


def test_the_pod_name_stays_inside_the_watchdog_scope(tmp_path: Path) -> None:
    """`run_native_camera_gpu_lane` refuses any other name, after admission."""

    profile = _build(_fixture(tmp_path))

    pod_name = _argv_value(profile["allocator"]["argv"], "--pod-name")
    assert pod_name.startswith(CANARY_NAME_PREFIX)
    assert profile["profile_id"] == pod_name


def _allocator_arguments_for_the_lane() -> set[str]:
    """Every `args.*` the allocator hands this lane, read from its own call."""

    tree = ast.parse(ALLOCATOR_PATH.read_text(encoding="utf-8"))
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
        if name != "run_native_camera_gpu_lane":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "args"
            ):
                attributes.add(child.attr)
    return attributes


def test_the_profile_carries_every_allocator_argument_the_lane_reads(
    tmp_path: Path,
) -> None:
    """An omitted flag fixes a paid decision at a default nobody chose.

    Derived from the allocator's own call rather than written down here, so a
    new argument on that call fails this instead of quietly defaulting on the
    next live run.
    """

    argv = _build(_fixture(tmp_path))["allocator"]["argv"]
    # `--execute` is appended by the dispatcher and forbidden in a profile.
    expected = sorted(_allocator_arguments_for_the_lane() - {"execute"})

    assert expected, "the allocator call for this lane was not found"
    missing = [
        name for name in expected if "--" + name.replace("_", "-") not in argv
    ]
    assert not missing, f"profile argv omits allocator arguments: {missing}"


def test_every_builder_parameter_is_reachable_from_the_command_line() -> None:
    """The flag table is the call signature, so neither can drift from it."""

    signature = inspect.signature(builder.build_new_site_native_camera_live_profile)
    parameters = dict(signature.parameters)
    flagged = {spec.parameter: flag for flag, spec in builder.FLAGS.items()}

    assert set(flagged) == set(parameters), (
        "the flag table and the builder signature disagree: "
        f"{sorted(set(flagged) ^ set(parameters))}"
    )
    mandatory = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    optional_flags = {
        spec.parameter for spec in builder.FLAGS.values() if not spec.required
    }
    assert not (mandatory & optional_flags), (
        "these parameters have no default and no required flag, so the CLI "
        f"cannot supply them: {sorted(mandatory & optional_flags)}"
    )


def test_the_command_line_writes_the_profile_it_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "profile.json"
    values = {
        "--bundle-receipt": paths["bundle_receipt_path"],
        "--release-evidence": paths["release_evidence_path"],
        "--preflight-bundle": paths["preflight_bundle_path"],
        "--provider-bundle-url-file": paths["provider_bundle_url_file"],
        "--provider-output-put-url-file": paths["provider_output_put_url_file"],
        "--provider-output-get-url-file": paths["provider_output_get_url_file"],
        "--campaign-budget-ledger": paths["campaign_budget_ledger"],
        "--campaign-initial-spent-usd": "0.0",
        "--campaign-initial-used-gpu-seconds": "0",
        "--source-commit": COMMIT,
        "--raw-manifest-uri": URI,
    }
    argv = [str(item) for pair in values.items() for item in pair]

    assert builder.main([*argv, "--output", str(output)]) == 0

    reported = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert reported["status"] == "built"
    assert reported["provider_mutation_performed"] is False
    assert reported["profile_id"] == written["profile_id"]
    assert reported["profile_digest"] == written["profile_digest"]
    assert written["profile_id"].startswith(CANARY_NAME_PREFIX)
    assert _argv_value(written["allocator"]["argv"], "--expected-source-commit") == COMMIT
    assert written == _build(paths), "the CLI and the callee disagree"


def test_the_command_line_reports_a_refusal_without_writing_a_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    paths["provider_bundle_url_file"].chmod(0o644)
    output = tmp_path / "profile.json"

    code = builder.main(
        [
            "--bundle-receipt", str(paths["bundle_receipt_path"]),
            "--release-evidence", str(paths["release_evidence_path"]),
            "--preflight-bundle", str(paths["preflight_bundle_path"]),
            "--provider-bundle-url-file", str(paths["provider_bundle_url_file"]),
            "--provider-output-put-url-file", str(paths["provider_output_put_url_file"]),
            "--provider-output-get-url-file", str(paths["provider_output_get_url_file"]),
            "--campaign-budget-ledger", str(paths["campaign_budget_ledger"]),
            "--campaign-initial-spent-usd", "0.0",
            "--campaign-initial-used-gpu-seconds", "0",
            "--source-commit", COMMIT,
            "--raw-manifest-uri", URI,
            "--output", str(output),
        ]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not output.exists()


def test_the_secret_url_files_are_never_pinned_as_immutable_inputs(
    tmp_path: Path,
) -> None:
    """A presigned URL is a rotating secret, not a digest-bound input."""

    paths = _fixture(tmp_path)
    profile = _build(paths)

    pinned = {row["path"] for row in profile["immutable_inputs"]}
    for name in (
        "provider_bundle_url_file",
        "provider_output_put_url_file",
        "provider_output_get_url_file",
    ):
        assert str(paths[name]) not in pinned
    assert {"source_bundle_manifest", "evaluation_run_spec"} <= {
        row["name"] for row in profile["immutable_inputs"]
    }


def test_the_bundle_and_release_are_pinned_by_digest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = {row["name"]: row for row in _build(paths)["immutable_inputs"]}

    bundle = inputs["native_camera_input_bundle"]
    assert bundle["path"] == str(paths["bundle"])
    assert bundle["digest"] == "sha256:" + hashlib.sha256(
        paths["bundle"].read_bytes()
    ).hexdigest()
    assert inputs["native_camera_release_evidence"]["digest"].startswith("sha256:")


def test_every_bound_path_stays_under_the_control_plane_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authoring-path defect that reached a provider on 2026-08-12."""

    paths = _fixture(tmp_path)
    monkeypatch.setenv(LAUNCH_INPUT_ROOTS_ENV, str(tmp_path))

    profile = _build(paths)

    bound = [
        item
        for item in profile["allocator"]["argv"]
        if item.startswith("/") and "{" not in item
    ]
    assert bound, "no absolute allocator argument was bound"
    assert all(item.startswith(str(tmp_path.resolve()) + os.sep) for item in bound)


def test_a_receipt_from_another_commit_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, manifest_commit="f" * 40)

    with pytest.raises(TaskEvaluationLaunchError, match="bundle_commit_not_source_commit"):
        _build(paths)


def test_a_release_from_another_commit_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, release_commit="0" * 40)

    with pytest.raises(TaskEvaluationLaunchError, match="release_"):
        _build(paths)


def test_a_ttl_outside_the_lane_band_is_refused(tmp_path: Path) -> None:
    """The lane's admission refuses anything outside 60..3600 seconds."""

    paths = _fixture(tmp_path)

    with pytest.raises(TaskEvaluationLaunchError, match="hard_ttl_out_of_band"):
        _build(paths, hard_ttl_seconds=7_200)


def test_a_world_readable_secret_url_file_is_refused(tmp_path: Path) -> None:
    """`_read_private_https_url` insists on 0600, after a provider is rented."""

    paths = _fixture(tmp_path)
    paths["provider_output_put_url_file"].chmod(0o644)

    with pytest.raises(TaskEvaluationLaunchError, match="not_private_0600"):
        _build(paths)


def test_a_non_https_secret_url_file_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _private(paths["provider_bundle_url_file"], "http://example.invalid/in")

    with pytest.raises(TaskEvaluationLaunchError, match="not_https"):
        _build(paths)


def test_a_container_disk_below_the_lane_floor_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, container_disk_bytes=8 * 1024**3)

    with pytest.raises(TaskEvaluationLaunchError, match="container_disk"):
        _build(paths)


def test_a_worst_case_above_the_declared_spend_is_refused(tmp_path: Path) -> None:
    """The lane compares the preflight's own price against the spend cap."""

    paths = _fixture(tmp_path, hourly_price=9.0)

    with pytest.raises(TaskEvaluationLaunchError, match="spend"):
        _build(paths, max_hourly_rate_usd=0.5, max_spend_usd=0.5, hard_ttl_seconds=3_600)


def test_a_campaign_start_above_its_own_cap_is_refused(tmp_path: Path) -> None:
    """`ProductionGpuCampaignBudget` raises on this, after the watchdog is armed."""

    paths = _fixture(tmp_path)

    with pytest.raises(TaskEvaluationLaunchError, match="campaign"):
        _build(paths, campaign_initial_spent_usd=25.0)


def test_the_probe_kind_is_the_one_the_allocator_dispatches() -> None:
    assert builder.SPEC.probe_kind == PROBE_KIND


def test_the_receipt_projection_normalises_the_digest_and_the_status(
    tmp_path: Path,
) -> None:
    """This receipt says `completed` and states a bare-hex digest.

    Both are correct for the lane and unreadable to the launch layer, which
    admits only `ready` and only prefixed digests. The projection translates;
    it does not rewrite the retained bytes the allocator opens.
    """

    paths = _fixture(tmp_path)
    receipt_path = paths["bundle_receipt_path"]
    original = json.loads(receipt_path.read_text(encoding="utf-8"))

    resolution = resolve_host_resident_bundle_receipt(receipt_path)

    assert resolution["status"] == "ready"
    assert resolution["receipt"]["status"] == "ready"
    assert resolution["receipt"]["native_receipt_status"] == "completed"
    assert resolution["receipt"]["bundle_sha256"] == "sha256:" + original["bundle_sha256"]
    assert resolution["resolutions"]["bundle"]["path"] == str(paths["bundle"])
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == original


def test_a_receipt_that_lost_its_label_free_freeze_is_not_promoted(
    tmp_path: Path,
) -> None:
    """A translation layer must not be a way to launder a broken boundary."""

    paths = _fixture(tmp_path)
    receipt_path = paths["bundle_receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest"]["rankings_or_policy_outcomes_accessed"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    resolution = resolve_host_resident_bundle_receipt(receipt_path)

    assert resolution["receipt"]["status"] != "ready"
    with pytest.raises(TaskEvaluationLaunchError, match="bundle_receipt_not_ready"):
        _build(paths)
