#!/usr/bin/env python3
"""Build a live launch profile for the pinned reconstruction worker smoke.

The allocator has dispatched `reconstruction-worker-smoke` for a long time and
the execute adapter is written and focus-tested. What was missing was a launch
profile, which is the one thing that carries a lane across the website boundary,
so the lane sat in `NOT_WEBSITE_REACHABLE` as `awaiting_builder`.

This lane is request-driven rather than receipt-driven. Its allocator branch
opens a `reconstruction_gpu_canary_request.v1` and a provider preflight; the
`worker_smoke` operation binds no input archive at all, because the run is the
pinned worker image's own healthcheck. There is therefore no bundle receipt to
resolve, so this builder shares the half of the skeleton that decides whether a
run is provable -- `shared_control_surface` -- rather than the receipt-residency
half, exactly as the 840313 lane does.

Three things here are refusals the allocator only makes after a provider has
been handed over, or that a live run only discovers with a GPU already running:

* Spend, TTL, retry cap, and authority are read out of the sealed request rather
  than taken from a flag. Admission compares each for exact equality, so a flag
  whose default disagreed with the request would spend an admission to learn it.
* The preflight is refreshed inside the launch. Admission refuses a snapshot
  older than five minutes, and a profile is published once and launched days
  later, so an authoring-time snapshot is stale by construction.
* The watchdog in the seed must be armed for this lane's own name prefix. The
  smoke validates it against `NAME_PREFIX` and verifies provider zero under the
  same scope, so a watchdog armed elsewhere is a paid run with no kill switch.

Known gap, recorded rather than worked around: this lane does not seal terminal
artifacts. `reconstruction_vast_worker_smoke` writes its teardown and
provider-zero receipts under its own names and never emits
`artifact_manifest_path`, `teardown_manifest_path`,
`continuing_spend_from_this_run`, or `retry_cap` on the allocator result, so the
shared terminal contract cannot be met by a run that otherwise succeeds. The
module escapes `tests/test_paid_lane_terminal_artifact_contract.py` only because
its filename does not end in `_vast.py`. That is a lane-side fix, not a profile
one, and inventing a narrower terminal contract here would hide it.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.host_resident_launch_inputs import launch_profile_residency_blockers
from blueprint_pipeline.reconstruction_gpu_admission import (
    MAX_TTL_SECONDS,
    MIN_CONTAINER_DISK_BYTES,
    PREFLIGHT_SCHEMA_VERSION,
    PROBE_KIND,
    REQUEST_SCHEMA_VERSION,
    build_reconstruction_gpu_canary_request,
)
from blueprint_pipeline.reconstruction_vast_worker_smoke import (
    NAME_PREFIX,
    RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    TaskEvaluationLaunchError,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)
from blueprint_pipeline import task_evaluation_live_profile as live_profile_contract
from blueprint_pipeline.task_evaluation_live_profile import (
    PROFILE_SCHEMA_VERSION,
    PROGRAM_ID,
    RUN_ROOT,
    shared_control_surface,
)

PROFILE_ID_PREFIX = "adp-reconstruction-worker-smoke-live"
#: The only operation this profile can supply every input for. The probe kind
#: also dispatches the pose, trainer, Isaac, and measurement operations, each of
#: which additionally needs an input bundle receipt and a signed bundle URL that
#: nothing here binds.
OPERATION = "worker_smoke"

#: One table, read twice: once to build the parser and once to build the call.
#: A parameter with no flag is a paid decision silently fixed at its default,
#: and a flag with no parameter is a value the caller thinks it is supplying.
PARAMETERS: dict[str, dict[str, Any]] = {
    "request_path": {
        "flag": "--canary-request",
        "required": True,
        "help": "Sealed reconstruction_gpu_canary_request.v1 for one worker smoke.",
    },
    "preflight_seed_path": {
        "flag": "--preflight-seed",
        "required": True,
        "help": (
            "Provider preflight the launch refreshes in place. Carries the armed "
            "watchdog and the conflicting-owner declaration."
        ),
    },
    "output_put_url_file": {
        "flag": "--output-put-url-file",
        "required": True,
        "help": "Mode-0600 file holding the signed result PUT URL.",
    },
    "output_get_url_file": {
        "flag": "--output-get-url-file",
        "required": True,
        "help": "Mode-0600 file holding the signed result GET URL.",
    },
    "source_commit": {"flag": "--source-commit", "required": True},
    "raw_manifest_uri": {
        "flag": "--raw-manifest-uri",
        "required": True,
        "help": "Local digest-bound content-addressed publication receipt for this run spec.",
    },
    "max_hourly_rate_usd": {
        "flag": "--max-hourly-rate-usd",
        "required": True,
        "type": float,
        "help": "Ceiling the refreshed capacity probe selects an offer under.",
    },
    "container_disk_bytes": {
        "flag": "--container-disk-bytes",
        "required": True,
        "type": int,
        "help": "Container disk the refreshed preflight declares; floored at 100 GiB.",
    },
    "revision": {
        "flag": "--revision",
        "help": "Distinguish a rebuilt profile whose inputs changed at the same commit.",
    },
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"worker_smoke_profile_input_not_object:{path.name}")
    return dict(value)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _request_blockers(request: Mapping[str, Any], *, source_commit: str) -> list[str]:
    """Re-derive the request through the lane's own contract, then compare.

    Reimplementing the field checks here would be a second opinion that drifts.
    Rebuilding the request recomputes its digest from the bytes on disk, so a
    request edited after it was sealed is refused by the same code the allocator
    will run.
    """

    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("worker_smoke_request_schema_invalid")
        return blockers
    try:
        rebuilt = build_reconstruction_gpu_canary_request(request)
    except ValueError as exc:
        return [f"worker_smoke_request_invalid:{code}" for code in str(exc).split(";")]
    if rebuilt["request_digest"] != request.get("request_digest"):
        blockers.append("worker_smoke_request_digest_mismatch")
    if request.get("operation") != OPERATION:
        blockers.append(f"worker_smoke_operation_not_worker_smoke:{request.get('operation')}")
    if request.get("expected_runtime_result_schema") != RESULT_SCHEMA_VERSION:
        blockers.append("worker_smoke_expected_result_schema_invalid")
    if request.get("source_commit_sha") != source_commit:
        blockers.append("worker_smoke_request_commit_mismatch")
    if request.get("retry_cap") != 0:
        blockers.append(f"worker_smoke_retry_cap_not_zero:{request.get('retry_cap')}")
    return blockers


def _preflight_blockers(seed: Mapping[str, Any]) -> list[str]:
    """Refuse a seed the refreshed preflight is guaranteed to block on.

    The refresh keeps the seed's watchdog and conflicting-owner declaration and
    replaces only the capacity and inventory snapshot, so these two travel from
    the seed into the admission unchanged.
    """

    blockers: list[str] = []
    if seed.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        blockers.append("worker_smoke_preflight_seed_schema_invalid")
    if seed.get("conflicting_owner_present") is not False:
        blockers.append("worker_smoke_preflight_seed_conflicting_owner_present")
    watchdog = seed.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    if watchdog.get("status") != "armed" or watchdog.get("independent_process") is not True:
        blockers.append("worker_smoke_independent_watchdog_not_armed")
    scope = str(watchdog.get("name_prefix") or watchdog.get("pod_name_prefix") or "")
    if scope != NAME_PREFIX:
        # The smoke validates the watchdog against its own prefix and verifies
        # provider zero under the same scope, so a mismatch is a paid run whose
        # kill switch is watching a different set of instances.
        blockers.append(f"worker_smoke_watchdog_name_prefix_mismatch:{scope}")
    return blockers


def _private_url_file_blockers(label: str, path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        return [f"worker_smoke_{label}_missing_or_unsafe"]
    if path.stat().st_mode & 0o077:
        return [f"worker_smoke_{label}_permissions_not_0600"]
    return []


def build_reconstruction_worker_smoke_live_profile(
    *,
    request_path: str | Path,
    preflight_seed_path: str | Path,
    output_put_url_file: str | Path,
    output_get_url_file: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    max_hourly_rate_usd: float,
    container_disk_bytes: int,
    revision: str | None = None,
) -> dict[str, Any]:
    """Derive one publishable worker-smoke profile, or refuse with every reason."""

    request_file = Path(request_path).expanduser().resolve()
    seed_file = Path(preflight_seed_path).expanduser().resolve()
    put_url_file = Path(output_put_url_file).expanduser().resolve()
    get_url_file = Path(output_get_url_file).expanduser().resolve()

    blockers: list[str] = []
    for label, path in (("request", request_file), ("preflight_seed", seed_file)):
        if path.is_symlink() or not path.is_file():
            blockers.append(f"worker_smoke_{label}_missing_or_unsafe")
    blockers.extend(_private_url_file_blockers("output_put_url_file", put_url_file))
    blockers.extend(_private_url_file_blockers("output_get_url_file", get_url_file))
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    request = _read(request_file)
    seed = _read(seed_file)
    blockers.extend(_request_blockers(request, source_commit=source_commit))
    blockers.extend(_preflight_blockers(seed))

    if not 0 < float(max_hourly_rate_usd):
        blockers.append(f"worker_smoke_max_hourly_rate_invalid:{max_hourly_rate_usd}")
    if int(container_disk_bytes) < MIN_CONTAINER_DISK_BYTES:
        blockers.append(f"worker_smoke_container_disk_below_floor:{container_disk_bytes}")

    hard_ttl_seconds = request.get("hard_ttl_seconds")
    max_spend_usd = request.get("max_spend_usd")
    if (
        not isinstance(hard_ttl_seconds, int)
        or isinstance(hard_ttl_seconds, bool)
        or not 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
    ):
        blockers.append(f"worker_smoke_hard_ttl_out_of_band:{hard_ttl_seconds}")
    elif isinstance(max_spend_usd, (int, float)) and not isinstance(max_spend_usd, bool):
        # Admission recomputes this against the offer the refreshed preflight
        # selected. Checking it against the rate ceiling here is the same
        # refusal, made before an admission is spent to learn it.
        worst_case = float(max_hourly_rate_usd) * hard_ttl_seconds / 3600.0
        if worst_case > float(max_spend_usd):
            blockers.append(
                f"worker_smoke_budget_below_worst_case_cost:{round(worst_case, 6)}"
            )
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    profile_id = f"{PROFILE_ID_PREFIX}-{source_commit}"
    if revision:
        # Published profiles are immutable, so a profile whose inputs changed
        # needs its own id rather than a conflicting rewrite of an existing one.
        profile_id = f"{profile_id}-{revision}"
    request_digest = _file_digest(request_file)
    immutable_inputs = [
        {
            "name": "source_bundle_manifest",
            "path": str(request_file),
            "digest": _file_digest(request_file),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(request_file),
            "digest": _file_digest(request_file),
        },
    ]
    manifest_uri, manifest_publication, immutable_inputs = (
        live_profile_contract.bind_live_profile_manifest_publication(
            reference=raw_manifest_uri,
            source_commit=source_commit,
            run_spec_digest=request_digest,
            profile_builder="build_reconstruction_worker_smoke_live_profile.py",
            immutable_inputs=immutable_inputs,
        )
    )

    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "program_id": PROGRAM_ID,
        "claim_ceiling": "development_only",
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--admission-out", f"{RUN_ROOT}/allocator/admission.json",
                "--bound-request-out", f"{RUN_ROOT}/allocator/bound-request.json",
                "--adapter-output", f"{RUN_ROOT}/allocator/result.json",
                "--pod-name", profile_id,
                "--expected-source-commit", source_commit,
                "--provider", "vast",
                "--probe-kind", PROBE_KIND,
                "--provider-launch-request", str(request_file),
                # Input and output of the same launch: the refresh writes the
                # refreshed snapshot back here, which is why it is bound by
                # path and never by digest.
                "--preflight-bundle", str(seed_file),
                "--reconstruction-refresh-preflight",
                "--reconstruction-name-prefix", NAME_PREFIX,
                "--reconstruction-container-disk-bytes", str(int(container_disk_bytes)),
                "--reconstruction-max-hourly-rate-usd", str(float(max_hourly_rate_usd)),
                "--reconstruction-max-spend-usd", str(max_spend_usd),
                "--reconstruction-hard-ttl-seconds", str(hard_ttl_seconds),
                "--reconstruction-retry-cap", "0",
                "--reconstruction-authority-id", str(request["authority_id"]),
                "--provider-output-put-url-file", str(put_url_file),
                "--provider-output-get-url-file", str(get_url_file),
            ],
            "max_spend_usd": float(max_spend_usd),
            "hard_ttl_seconds": int(hard_ttl_seconds),
            "retry_cap": 0,
        },
        "execution_admission": {
            "live_enabled": True,
            "blockers": [],
            "readiness_receipt": {"uri": manifest_uri, "digest": request_digest},
        },
        "evaluation_run_spec": {"uri": manifest_uri, "digest": request_digest},
        "source_bundle": {
            "bundle_id": f"reconstruction-worker-smoke-{source_commit[:12]}",
            "source_kind": "interiorgs_sage",
            "uri": manifest_uri,
            "digest": request_digest,
        },
        # The sealed request is the only input of this launch that is both
        # digest-stable and not a secret. The signed URL files rotate and the
        # preflight seed is rewritten by the launch itself, so pinning either
        # would fail the launch after the one it was authored for.
        "immutable_inputs": immutable_inputs,
        "runtime_environment": {},
        **shared_control_surface(),
    }
    if manifest_publication is not None:
        profile["manifest_publication"] = manifest_publication
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")

    validation = [
        *validate_launch_profile(profile),
        *verify_profile_immutable_inputs(profile),
        # A profile is published once and read on every later launch, so this is
        # the last chance to catch an authoring path.
        *launch_profile_residency_blockers(profile),
    ]
    if validation:
        raise TaskEvaluationLaunchError(",".join(sorted(set(validation))))
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, options in PARAMETERS.items():
        settings = {key: value for key, value in options.items() if key != "flag"}
        parser.add_argument(options["flag"], dest=name, **settings)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        profile = build_reconstruction_worker_smoke_live_profile(
            **{name: getattr(args, name) for name in PARAMETERS}
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "built",
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "max_spend_usd": profile["allocator"]["max_spend_usd"],
                "output": str(output),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
