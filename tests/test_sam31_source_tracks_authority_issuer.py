"""The SAM 3.1 source-track lane has to be authorizable from a production path.

`scripts/build_sam31_source_tracks_live_profile.py` refuses to build a profile
without `--attempt-authority`, and the only function that mints one --
`materialize_sam31_paid_attempt_authority` -- could be called from no script.
The lane therefore had a live profile builder whose required input no production
path could produce: authorizing a SAM run meant opening a Python session.

That is the same defect as #512 (lanes), #520 (bundle modules) and #523
(authority materializers), in the one authority module #523's scan reached but
its fix did not.

The flag table in the issuer *is* the call: the parser and the keyword arguments
are both built from it, and the contract below derives the left column from the
materializer's own signature. A hand-listed table is how #523's first cut
dropped four of six predecessor parameters -- and on this lane a dropped
parameter is spend that goes uncounted against a shared cap.

Spend and TTL are required rather than defaulted, which departs from the
appearance-chain issuer deliberately: upstream refuses unless `hard_cap_usd` and
`hard_ttl_seconds` equal the request's own `max_spend_usd` and `hard_ttl_seconds`
exactly, so a default here is not a convenience but a second number that has to
agree with a file the operator did not read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_gpu_admission import REQUEST_SCHEMA_VERSION
from blueprint_pipeline.sam31_paid_attempt_authority import (
    AUTHORITY_SCHEMA_VERSION,
    validate_sam31_paid_attempt_authority,
)
from blueprint_pipeline.sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "issue_sam31_source_tracks_paid_attempt_authority.py"

COMMIT = "a" * 40


def _load():
    name = "issue_sam31_source_tracks"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so a file-loaded module has to be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


issuer = _load()


def _inputs(tmp_path: Path, *, cap_usd: float = 1.0, ttl_seconds: int = 600) -> dict[str, Path]:
    """A request, bundle and receipt that agree with each other, as upstream demands."""

    bundle = tmp_path / "input.zip"
    bundle.write_bytes(b"deterministic-sam31-bundle")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_commit_sha": COMMIT,
        "worker_image_digest": "registry.example/sam31@sha256:" + "b" * 64,
        "input_bundle_digest": digest,
        "input_bundle_size_bytes": bundle.stat().st_size,
        "max_spend_usd": cap_usd,
        "hard_ttl_seconds": ttl_seconds,
        "retry_cap": 0,
        "authority_id": "goal-authority-1",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    receipt = {
        "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle": {
            "filename": bundle.name,
            "sha256": digest,
            "size_bytes": bundle.stat().st_size,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    return {"request": request_path, "bundle": bundle, "receipt": receipt_path}


def _argv(tmp_path: Path, files: dict[str, Path], **overrides: str) -> list[str]:
    arguments = {
        "--request": str(files["request"]),
        "--bundle": str(files["bundle"]),
        "--bundle-receipt": str(files["receipt"]),
        "--authorized-by": "fixture-user",
        "--authority-reference": "User directed one bounded SAM source-track run",
        "--blueprint-commit": COMMIT,
        "--max-hourly-rate-usd": "0.5",
        "--hard-cap-usd": "1.0",
        "--hard-ttl-seconds": "600",
        "--aggregate-spend-before-usd": "0.0",
        "--aggregate-spend-cap-usd": "12.0",
        "--output": str(tmp_path / "authority.json"),
    }
    arguments.update(overrides)
    return [value for flag, argument in arguments.items() for value in (flag, argument)]


def test_every_materializer_keyword_has_a_flag() -> None:
    """Derives the requirement from the signature, so upstream cannot outrun it.

    A new keyword on the materializer -- especially another spend anchor --
    fails here until a flag supplies it, rather than being silently defaulted
    away at the point where a paid attempt is authorized.
    """

    upstream = {
        name
        for name, parameter in inspect.signature(issuer.MATERIALIZE).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    missing = upstream - set(issuer.PARAMS)
    assert not missing, (
        f"the issuer cannot supply {sorted(missing)} from a command line. Add a "
        "flag to the table; do not let a paid authority default it away."
    )


def test_no_flag_invents_a_keyword_the_materializer_will_not_take() -> None:
    upstream = set(inspect.signature(issuer.MATERIALIZE).parameters)

    assert not set(issuer.PARAMS) - upstream


def test_flags_are_distinct() -> None:
    """Two keywords sharing a flag would let one silently overwrite the other."""

    flags = [param.flag for param in issuer.PARAMS.values()]

    assert len(flags) == len(set(flags))


@pytest.mark.parametrize(
    "keyword", ["authorized_by", "authorization_reference", "output_path"]
)
def test_the_unattributable_arguments_are_required(keyword: str) -> None:
    """Neither is derivable, and an unattributed paid attempt is not authorized."""

    assert issuer.PARAMS[keyword].required


@pytest.mark.parametrize(
    "keyword",
    [
        "max_hourly_rate_usd",
        "hard_cap_usd",
        "hard_ttl_seconds",
        "aggregate_goal_spend_before_attempt_usd",
        "aggregate_goal_spend_cap_usd",
    ],
)
def test_no_spend_or_ttl_bound_is_defaulted_away(keyword: str) -> None:
    """A defaulted bound is a number nobody chose, at the point money is committed.

    `hard_cap_usd` and `hard_ttl_seconds` additionally have to equal the
    request's own, and `aggregate_goal_spend_before_attempt_usd` is what keeps a
    second attempt accountable for the first attempt's spend.
    """

    assert issuer.PARAMS[keyword].required


def _stub_files() -> dict[str, Path]:
    """Paths that only have to parse; these cases never reach the materializer."""

    return {name: Path(f"/nonexistent/{name}.json") for name in ("request", "bundle", "receipt")}


def test_an_omitted_repeatable_flag_stays_empty_not_none() -> None:
    """`None` would reach a `Sequence[int]` parameter that iterates it."""

    args = issuer.build_parser().parse_args(_argv(Path("/nonexistent"), _stub_files()))

    assert issuer.call_arguments(args)["allowed_active_instance_ids"] == ()


def test_an_omitted_optional_path_stays_none_not_empty() -> None:
    """`prior_spend_reconciliation_path` is checked with `is not None` upstream."""

    args = issuer.build_parser().parse_args(_argv(Path("/nonexistent"), _stub_files()))

    assert issuer.call_arguments(args)["prior_spend_reconciliation_path"] is None


def test_the_authorization_date_defaults_to_today_not_to_empty() -> None:
    args = issuer.build_parser().parse_args(_argv(Path("/nonexistent"), _stub_files()))

    authorized_on = issuer.call_arguments(args)["authorized_on"]

    assert authorized_on and authorized_on.count("-") == 2


def test_the_issued_authority_validates_against_its_own_request_and_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end the lane actually needs: a file the live profile builder accepts."""

    files = _inputs(tmp_path)
    output = tmp_path / "authority.json"

    code = issuer.main(_argv(tmp_path, files))

    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["status"] == "issued"
    assert summary["provider_mutation_performed"] is False

    authority = json.loads(output.read_text(encoding="utf-8"))
    assert authority["schema_version"] == AUTHORITY_SCHEMA_VERSION
    assert authority["authorization_digest"] == summary["authorization_digest"]
    assert authority["maximum_paid_attempts"] == 1
    assert authority["maximum_automatic_retries"] == 0

    validate_sam31_paid_attempt_authority(
        authority,
        request=json.loads(files["request"].read_text(encoding="utf-8")),
        bundle_path=files["bundle"],
        bundle_receipt=json.loads(files["receipt"].read_text(encoding="utf-8")),
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=1.0,
        hard_ttl_seconds=600,
    )


def test_repeatable_instance_ids_reach_the_allowlist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anything active and unlisted fails the attempt closed, so the list is load-bearing."""

    files = _inputs(tmp_path)
    argv = _argv(tmp_path, files)
    argv += ["--allow-active-instance", "42", "--allow-active-instance", "7"]

    code = issuer.main(argv)
    capsys.readouterr()

    assert code == 0
    authority = json.loads((tmp_path / "authority.json").read_text(encoding="utf-8"))
    assert authority["active_instance_allowlist"]["external_provider_owned"] == [7, 42]


def test_a_missing_request_is_refused_without_touching_a_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of this script is refusing before anything is rented."""

    code = issuer.main(_argv(tmp_path, _stub_files()))

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["provider_mutation_performed"] is False
    assert payload["blockers"], "a refusal has to name its cause"
    assert not (tmp_path / "authority.json").exists()


def test_a_cap_that_disagrees_with_the_request_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Why the money flags are required: they have to match a file, not a default."""

    files = _inputs(tmp_path, cap_usd=1.0)

    code = issuer.main(_argv(tmp_path, files, **{"--hard-cap-usd": "2.0"}))

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "blocked"
    assert not (tmp_path / "authority.json").exists()


def test_issuing_twice_over_the_same_output_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The authority is single-use; silently reissuing one would erase that."""

    files = _inputs(tmp_path)

    assert issuer.main(_argv(tmp_path, files)) == 0
    first = json.loads((tmp_path / "authority.json").read_text(encoding="utf-8"))
    capsys.readouterr()

    code = issuer.main(_argv(tmp_path, files))

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "blocked"
    assert json.loads((tmp_path / "authority.json").read_text(encoding="utf-8")) == first


def test_the_issuer_produces_what_the_live_profile_builder_demands() -> None:
    """The reason this entry point is debt rather than an unused convenience.

    The lane has a real live profile builder, and its `--attempt-authority` is
    required. An authority nothing can mint makes that builder unreachable too.
    """

    builder = (REPO_ROOT / "scripts" / "build_sam31_source_tracks_live_profile.py").read_text(
        encoding="utf-8"
    )

    assert '"--attempt-authority", required=True' in builder
    assert issuer.MATERIALIZE.__name__ == "materialize_sam31_paid_attempt_authority"
