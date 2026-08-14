"""The worker-smoke lane's launch profile, and the refusals it exists to make.

`reconstruction-worker-smoke` has had an allocator branch and an execute adapter
for a long time and no launch profile, which is the one thing that carries a
lane across the website boundary. It was the fourth entry in
`NOT_WEBSITE_REACHABLE` under `awaiting_builder`.

Every refusal below is one the allocator already makes *after* a provider has
been handed over, or one a live run would only discover once a GPU was running.
Making them at authoring time is the whole point of a profile builder.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.reconstruction_gpu_admission import (
    MIN_CONTAINER_DISK_BYTES,
    PREFLIGHT_SCHEMA_VERSION,
    PROBE_KIND,
    REQUEST_SCHEMA_VERSION,
)
from blueprint_pipeline.reconstruction_vast_worker_smoke import (
    NAME_PREFIX,
    RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    canonical_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_reconstruction_worker_smoke_live_profile",
    REPO_ROOT / "scripts" / "build_reconstruction_worker_smoke_live_profile.py",
)
builder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(builder)

COMMIT = "a" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/worker-smoke-request.json"
HOURLY_RATE_USD = 0.60
TTL_SECONDS = 1_800
MAX_SPEND_USD = 0.50


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _request(**overrides: Any) -> dict[str, Any]:
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "worker_smoke",
        "capture_profile": "trainer_smoke_fixture",
        "source_commit_sha": COMMIT,
        "worker_image_digest": "registry.example/reconstruction-worker@sha256:" + "b" * 64,
        "worker_stack_manifest_digest": _digest("stack"),
        "deterministic_configuration_digest": _digest("configuration"),
        "operation_request_digest": _digest("operation-request"),
        "operation_input_bundle_digest": _digest("operation-input-bundle"),
        "reconstruction_dataset_digest": _digest("dataset"),
        "frozen_split_digest": _digest("split"),
        "calibration_digest": _digest("calibration"),
        "expected_runtime_result_schema": RESULT_SCHEMA_VERSION,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": MAX_SPEND_USD,
        "hard_ttl_seconds": TTL_SECONDS,
        "retry_cap": 0,
        "authority_id": "reconstruction-worker-smoke-1",
        "proof_effect": "none",
    }
    request.update(overrides)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def _preflight_seed(**overrides: Any) -> dict[str, Any]:
    watchdog = {
        "status": "armed",
        "independent_process": True,
        "name_prefix": NAME_PREFIX,
    }
    seed = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "verified",
        "provider": "vast",
        "conflicting_owner_present": False,
        "watchdog": watchdog,
        "container_disk_bytes": MIN_CONTAINER_DISK_BYTES,
        "capacity_request": {"max_hourly_rate_usd": HOURLY_RATE_USD},
    }
    seed.update(overrides)
    return seed


def _fixture(
    tmp_path: Path,
    *,
    request: dict[str, Any] | None = None,
    seed: dict[str, Any] | None = None,
) -> dict[str, Path]:
    request_path = tmp_path / "reconstruction-worker-smoke-request.json"
    request_path.write_text(json.dumps(request or _request()), encoding="utf-8")
    seed_path = tmp_path / "reconstruction-gpu-provider-preflight.json"
    seed_path.write_text(json.dumps(seed or _preflight_seed()), encoding="utf-8")
    put_url = tmp_path / "output-put-url"
    get_url = tmp_path / "output-get-url"
    for path in (put_url, get_url):
        path.write_text("https://object.example/worker-smoke?signature=fixture\n", encoding="utf-8")
        path.chmod(0o600)
    return {
        "request": request_path,
        "seed": seed_path,
        "put_url": put_url,
        "get_url": get_url,
    }


def _build(paths: dict[str, Path], **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "request_path": paths["request"],
        "preflight_seed_path": paths["seed"],
        "output_put_url_file": paths["put_url"],
        "output_get_url_file": paths["get_url"],
        "source_commit": COMMIT,
        "raw_manifest_uri": URI,
        "max_hourly_rate_usd": HOURLY_RATE_USD,
        "container_disk_bytes": MIN_CONTAINER_DISK_BYTES,
    }
    arguments.update(overrides)
    return builder.build_reconstruction_worker_smoke_live_profile(**arguments)


def _argument(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_the_builder_emits_the_probe_kind_the_allocator_dispatches() -> None:
    """This is what removes the lane from `NOT_WEBSITE_REACHABLE`."""

    assert builder.PROBE_KIND == PROBE_KIND == "reconstruction-worker-smoke"


def test_builds_a_publishable_zero_retry_worker_smoke_profile(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path))
    argv = profile["allocator"]["argv"]

    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["execution_admission"]["blockers"] == []
    assert profile["allocator"]["retry_cap"] == 0
    assert _argument(argv, "--probe-kind") == PROBE_KIND
    assert _argument(argv, "--provider") == "vast"
    assert _argument(argv, "--provider-launch-request") == str(tmp_path / (
        "reconstruction-worker-smoke-request.json"
    ))
    assert profile["profile_digest"].startswith("sha256:")


def test_the_control_surface_is_the_shared_one_rather_than_a_copy(tmp_path: Path) -> None:
    """A lane that quietly dropped provider-zero keeps passing until it matters."""

    from blueprint_pipeline.task_evaluation_live_profile import shared_control_surface

    profile = _build(_fixture(tmp_path))
    shared = shared_control_surface()

    for block in ("required_controls", "terminal_contract", "webapp_sync", "reconciliation"):
        assert profile[block] == shared[block]


def test_the_launch_refreshes_the_preflight_because_a_seed_goes_stale(tmp_path: Path) -> None:
    """Admission refuses a preflight older than five minutes.

    A profile is published once and launched days later, so a snapshot taken at
    authoring time is stale by construction. The refresh flag is the only
    producer of a fresh one inside the launch.
    """

    argv = _build(_fixture(tmp_path))["allocator"]["argv"]

    assert "--reconstruction-refresh-preflight" in argv
    assert _argument(argv, "--reconstruction-name-prefix") == NAME_PREFIX


def test_spend_ttl_retry_and_authority_come_from_the_request(tmp_path: Path) -> None:
    """The allocator compares each of these to the request for exact equality.

    Taking them from a flag instead means a paid decision can be fixed at an
    argparse default that disagrees with the sealed request, and the disagreement
    surfaces as `..._binding_mismatch` after admission.
    """

    profile = _build(_fixture(tmp_path))
    argv = profile["allocator"]["argv"]

    assert _argument(argv, "--reconstruction-max-spend-usd") == str(MAX_SPEND_USD)
    assert _argument(argv, "--reconstruction-hard-ttl-seconds") == str(TTL_SECONDS)
    assert _argument(argv, "--reconstruction-retry-cap") == "0"
    assert _argument(argv, "--reconstruction-authority-id") == "reconstruction-worker-smoke-1"
    # The declared spend is the request's cap, not a builder-chosen worst case.
    assert profile["allocator"]["max_spend_usd"] == MAX_SPEND_USD
    assert profile["allocator"]["hard_ttl_seconds"] == TTL_SECONDS


def test_the_signed_url_files_are_bound_but_never_digest_pinned(tmp_path: Path) -> None:
    """Signed URLs rotate, so pinning one would fail the next launch -- and the
    profile is exactly the artifact that gets copied around later."""

    paths = _fixture(tmp_path)
    profile = _build(paths)
    argv = profile["allocator"]["argv"]
    names = {row["name"] for row in profile["immutable_inputs"]}
    pinned = {row["path"] for row in profile["immutable_inputs"]}

    assert _argument(argv, "--provider-output-put-url-file") == str(paths["put_url"])
    assert _argument(argv, "--provider-output-get-url-file") == str(paths["get_url"])
    assert {"source_bundle_manifest", "evaluation_run_spec"}.issubset(names)
    assert not any("url" in name for name in names)
    assert str(paths["put_url"]) not in pinned and str(paths["get_url"]) not in pinned


def test_the_mutated_preflight_seed_is_not_digest_pinned(tmp_path: Path) -> None:
    """The refresh writes the refreshed snapshot back over the seed, so pinning
    it would make every launch after the first fail on a digest mismatch."""

    paths = _fixture(tmp_path)
    profile = _build(paths)

    assert _argument(profile["allocator"]["argv"], "--preflight-bundle") == str(paths["seed"])
    assert str(paths["seed"]) not in {row["path"] for row in profile["immutable_inputs"]}


def test_a_tampered_request_is_refused(tmp_path: Path) -> None:
    request = _request()
    request["max_spend_usd"] = 99.0
    paths = _fixture(tmp_path, request=request)

    with pytest.raises(TaskEvaluationLaunchError, match="request_digest_mismatch"):
        _build(paths)


def test_an_operation_this_profile_cannot_supply_inputs_for_is_refused(
    tmp_path: Path,
) -> None:
    """The probe kind also dispatches pose, trainer, Isaac and measurement
    operations, and each needs an input bundle receipt and signed bundle URLs
    this profile does not bind."""

    request = _request(
        operation="trainer_canary",
        expected_runtime_result_schema="reconstruction_training_result.v1",
    )
    paths = _fixture(tmp_path, request=request)

    with pytest.raises(TaskEvaluationLaunchError, match="operation_not_worker_smoke"):
        _build(paths)


def test_a_nonzero_retry_cap_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, request=_request(retry_cap=1))

    with pytest.raises(TaskEvaluationLaunchError, match="retry_cap"):
        _build(paths)


def test_a_budget_below_the_worst_case_ttl_cost_is_refused(tmp_path: Path) -> None:
    """Admission computes rate x TTL and refuses a cap under it, which is a
    refusal after the provider snapshot rather than before."""

    paths = _fixture(tmp_path, request=_request(max_spend_usd=0.05))

    with pytest.raises(TaskEvaluationLaunchError, match="budget_below_worst_case_cost"):
        _build(paths)


def test_a_ttl_outside_the_admitted_band_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, request=_request(hard_ttl_seconds=20_000, max_spend_usd=4.0))

    with pytest.raises(TaskEvaluationLaunchError, match="hard_ttl"):
        _build(paths)


def test_a_commit_the_request_does_not_name_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(TaskEvaluationLaunchError, match="commit_mismatch"):
        _build(paths, source_commit="c" * 40)


def test_a_url_file_readable_by_anyone_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["get_url"].chmod(0o644)

    with pytest.raises(TaskEvaluationLaunchError, match="permissions_not_0600"):
        _build(paths)


def test_a_watchdog_scoped_to_another_prefix_is_refused(tmp_path: Path) -> None:
    """The smoke verifies provider zero under its own name prefix and validates
    the watchdog against it, so a watchdog armed for another prefix is a live
    run with no independent kill switch."""

    seed = _preflight_seed(
        watchdog={
            "status": "armed",
            "independent_process": True,
            "name_prefix": "blueprint-something-else-",
        }
    )
    paths = _fixture(tmp_path, seed=seed)

    with pytest.raises(TaskEvaluationLaunchError, match="watchdog"):
        _build(paths)


def test_a_seed_that_admits_a_conflicting_owner_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, seed=_preflight_seed(conflicting_owner_present=True))

    with pytest.raises(TaskEvaluationLaunchError, match="conflicting_owner"):
        _build(paths)


def test_a_container_disk_below_the_provider_floor_is_refused(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(TaskEvaluationLaunchError, match="container_disk"):
        _build(paths, container_disk_bytes=MIN_CONTAINER_DISK_BYTES - 1)


def test_the_flag_table_is_the_call_signature() -> None:
    """A parameter with no flag is a paid decision fixed at its default."""

    parameters = inspect.signature(
        builder.build_reconstruction_worker_smoke_live_profile
    ).parameters
    keyword_only = {
        name
        for name, value in parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    }
    required = {
        name
        for name, value in parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
        and value.default is inspect.Parameter.empty
    }

    assert set(builder.PARAMETERS) == keyword_only, (
        "the flag table and the call signature have drifted"
    )
    assert required <= set(builder.PARAMETERS)
    for name in required:
        assert builder.PARAMETERS[name].get("required") is True, (
            f"{name} has no value unless a flag supplies one"
        )


def test_the_command_line_writes_a_profile_it_can_publish(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "profiles" / "worker-smoke.json"

    code = builder.main(
        [
            "--canary-request", str(paths["request"]),
            "--preflight-seed", str(paths["seed"]),
            "--output-put-url-file", str(paths["put_url"]),
            "--output-get-url-file", str(paths["get_url"]),
            "--source-commit", COMMIT,
            "--raw-manifest-uri", URI,
            "--max-hourly-rate-usd", str(HOURLY_RATE_USD),
            "--container-disk-bytes", str(MIN_CONTAINER_DISK_BYTES),
            "--revision", "r2",
            "--output", str(output),
        ]
    )

    assert code == 0
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["profile_id"].endswith(f"{COMMIT}-r2")
    assert published["allocator"]["argv"] == _build(paths, revision="r2")["allocator"]["argv"]


def test_the_command_line_reports_a_refusal_instead_of_raising(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, request=_request(retry_cap=2))

    code = builder.main(
        [
            "--canary-request", str(paths["request"]),
            "--preflight-seed", str(paths["seed"]),
            "--output-put-url-file", str(paths["put_url"]),
            "--output-get-url-file", str(paths["get_url"]),
            "--source-commit", COMMIT,
            "--raw-manifest-uri", URI,
            "--max-hourly-rate-usd", str(HOURLY_RATE_USD),
            "--container-disk-bytes", str(MIN_CONTAINER_DISK_BYTES),
            "--output", str(tmp_path / "blocked.json"),
        ]
    )

    assert code == 2
    assert not (tmp_path / "blocked.json").exists()
