#!/usr/bin/env python3
"""Build a live launch profile for the Arena native-control probe.

The allocator has dispatched `adp-isaac-lab-arena-native-control` throughout,
the transport is written and tested, and the bundle got a command line of its
own so it could be rebuilt after a deploy moved the control plane's commit.
None of that makes the lane reachable: a launch profile is the one artifact
that carries a lane across the website boundary, and this lane had none.
`tests/test_website_reachable_probe_kinds.py` recorded it as
`awaiting_builder` -- an admission that working, funded code could not be
started from the product path. This is what removes that row.

Two things make this lane's profile different from its siblings, and both are
refusals rather than fields.

*The allocator builds the bundle itself.* Every other lane hands it a sealed
archive; this branch calls `build_arena_native_control_bundle` from the
approval named in argv. So the bundle digest published here describes something
that will be *rebuilt* at launch, and it is only true if the approval pinned
here is the approval that bundle came from. Nothing downstream compares the
two, which makes it this builder's job.

*The lane is a zero-action negative control.* A receipt that has lost its
control id, or that records a candidate policy as queried, is no longer that
control -- and a profile built from one would spend real money on a run that
cannot answer the question it was authorized for.

The rest -- residency, spend binding, terminal contract, validation -- is the
shared skeleton in `task_evaluation_live_profile`, deliberately not restated
here. Those are exactly the checks a per-lane copy drops.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from blueprint_pipeline.adp_founder_sim_protocol import (
    APPROVAL_SCHEMA_VERSION,
    PROTOCOL_ID,
)
from blueprint_pipeline.adp_isaac_lab_arena_vast import PROBE_KIND
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

# The allocator refuses anything outside this band for this probe
# (`adp_arena_hard_ttl_invalid`).
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 14_400
DEFAULT_MAX_HOURLY_RATE_USD = 1.0

#: The only control this lane runs. The transport seals it into the receipt and
#: the allocator writes it into its allocation binding.
CONTROL_ID = "arena_zero_action_negative"


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _approval_blockers(context: LaneLiveProfileContext) -> list[str]:
    """Refuse an approval that will not rebuild the bundle this profile pins."""

    path = context.extra_paths.get("approval")
    if path is None or path.is_symlink() or not path.is_file():
        return ["arena_approval_missing"]
    approval = _read_json(path)
    if approval is None:
        return ["arena_approval_unreadable"]

    receipt = context.receipt
    blockers: list[str] = []
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        blockers.append(f"arena_approval_schema_invalid:{approval.get('schema_version')}")
    if approval.get("approved") is not True:
        blockers.append("arena_approval_not_approved")
    if approval.get("protocol_id") != PROTOCOL_ID:
        blockers.append("arena_approval_protocol_id_mismatch")
    # These two are what bind the rebuild to the pinned digests. The allocator
    # runs `build_arena_native_control_bundle` from this file at launch, so a
    # different approval -- or the same approval against a different protocol --
    # produces a different archive than the one named in `source_bundle`.
    if approval.get("approval_receipt_digest") != receipt.get("approval_receipt_digest"):
        blockers.append("arena_approval_not_the_bundle_approval")
    if approval.get("protocol_digest") != receipt.get("protocol_digest"):
        blockers.append("arena_approval_protocol_digest_mismatch")
    return blockers


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    receipt = context.receipt
    blockers: list[str] = []
    # The allocator's own budget rule (`adp_arena_budget_invalid`). Declared
    # spend is rate x TTL, so any TTL under an hour puts the ceiling below the
    # hourly rate and the launch is refused after admission instead of here.
    if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
        blockers.append("arena_budget_invalid")
    if receipt.get("retry_cap") != 0:
        blockers.append(f"arena_bundle_retry_cap_not_zero:{receipt.get('retry_cap')}")
    if receipt.get("control_id") != CONTROL_ID:
        blockers.append(f"arena_bundle_control_id_mismatch:{receipt.get('control_id')}")
    if receipt.get("candidate_policy_queried") is not False:
        blockers.append("arena_bundle_candidate_policy_queried")
    if receipt.get("candidate_outcomes_accessed") is not False:
        blockers.append("arena_bundle_candidate_outcomes_accessed")
    blockers.extend(_approval_blockers(context))

    avoidlist = context.extra_paths.get("machine_avoidlist")
    if avoidlist is not None and not avoidlist.is_file():
        blockers.append("arena_machine_avoidlist_missing")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    argv = [
        "--adp-arena-approval", str(context.extra_paths["approval"]),
        "--adp-job-dir", context.job_dir("arena-native-control-job"),
        "--adp-max-hourly-rate-usd", str(context.max_hourly_rate_usd),
        "--adp-max-spend-usd", str(context.max_spend_usd),
        "--adp-hard-ttl-seconds", str(context.hard_ttl_seconds),
    ]
    avoidlist = context.extra_paths.get("machine_avoidlist")
    if avoidlist is not None:
        argv += ["--adp-machine-avoidlist", str(avoidlist)]
    return argv


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    approval = context.extra_paths["approval"]
    rows = [
        {
            "name": "source_bundle_manifest",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        # The approval is the only input that decides what the allocator builds,
        # so it is pinned as tightly as the archive it produces.
        {
            "name": "founder_sim_approval_receipt",
            "path": str(approval),
            "digest": file_digest(approval),
        },
        {
            "name": "arena_native_control_bundle",
            # Where the receipt's archive resolved *here*, not where it was built.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
    ]
    avoidlist = context.extra_paths.get("machine_avoidlist")
    if avoidlist is not None:
        # Which machines are excluded is digested into the allocator's own
        # allocation binding, so it is part of what was authorized.
        rows.append(
            {
                "name": "arena_machine_avoidlist",
                "path": str(avoidlist),
                "digest": file_digest(avoidlist),
            }
        )
    return rows


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp-arena-native-control-live",
    profile_builder="build_arena_native_control_live_profile.py",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: f"arena-native-control-{context.source_commit[:12]}",
    # The dispatcher admits three source kinds and this is not free text. All
    # three name a capture provenance, and this lane's stage is an Arena-native
    # asset (`isaac_lab_arena_pick_and_place_maple_table_v1`) rather than any
    # Blueprint capture, so none of them describes it. Declared as the sibling
    # Arena family declares it (`build_native_task_arena_live_profile`) so the
    # two Arena lanes at least agree; widening the enum is a dispatcher change,
    # not something to work around here.
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    extra_path_names=("approval",),
)


def build_arena_native_control_live_profile(
    *,
    bundle_receipt_path: str | Path,
    approval_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    machine_avoidlist_path: str | Path | None = None,
    revision: str | None = None,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
    hard_ttl_seconds: int = MAX_TTL_SECONDS,
) -> dict[str, Any]:
    """Derive this lane's live profile from the receipt and approval it will run.

    The declared spend is deliberately not an argument. It is rate x TTL -- the
    worst case this profile can actually reach -- so a standing authorization
    reserves what one launch can spend rather than a lane's lifetime ceiling.
    """

    extra_paths: dict[str, str | Path] = {"approval": approval_path}
    spec = SPEC
    if machine_avoidlist_path is not None:
        # The skeleton requires every path a spec declares, so the optional
        # avoidlist is only declared on the calls that actually supply one.
        extra_paths["machine_avoidlist"] = machine_avoidlist_path
        spec = replace(spec, extra_path_names=(*SPEC.extra_path_names, "machine_avoidlist"))
    return build_lane_live_profile(
        spec,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        extra_paths=extra_paths,
    )


@dataclass(frozen=True)
class Flag:
    """One command-line flag, named for the builder argument it supplies.

    The parser and the call below are both generated from `FLAGS`, so a flag
    cannot go missing. A missing flag does not fail: it silently fixes a paid
    decision -- a spend ceiling, a TTL, an avoidlist -- at whatever default the
    builder happens to carry, which is how the appearance-chain issuer lost four
    of its six predecessor parameters.
    """

    kwarg: str
    help: str
    required: bool = False
    type: Callable[[str], Any] | None = None
    default: Any = None


FLAGS: dict[str, Flag] = {
    "--bundle-receipt": Flag(
        "bundle_receipt_path",
        "adp_arena_bundle_receipt.json, from `python -m "
        "blueprint_pipeline.adp_isaac_lab_arena_vast`.",
        required=True,
    ),
    "--approval": Flag(
        "approval_path",
        "The founder approval the allocator rebuilds the bundle from.",
        required=True,
    ),
    "--source-commit": Flag(
        "source_commit", "The commit the control plane is running.", required=True
    ),
    "--raw-manifest-uri": Flag(
        "raw_manifest_uri",
        "Local digest-bound GCS publication receipt for this run spec.",
        required=True,
    ),
    "--machine-avoidlist": Flag(
        "machine_avoidlist_path", "Optional Vast machines this launch must not take."
    ),
    "--revision": Flag(
        "revision",
        "Distinguish a rebuilt profile whose inputs changed at the same commit.",
    ),
    "--max-hourly-rate-usd": Flag(
        "max_hourly_rate_usd",
        "Price ceiling per hour; with the TTL this fixes the declared spend.",
        type=float,
        default=DEFAULT_MAX_HOURLY_RATE_USD,
    ),
    "--hard-ttl-seconds": Flag(
        "hard_ttl_seconds",
        f"Between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}; outside that the "
        "allocator refuses after admission.",
        type=int,
        default=MAX_TTL_SECONDS,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, entry in FLAGS.items():
        options: dict[str, Any] = (
            {"required": True} if entry.required else {"default": entry.default}
        )
        if entry.type is not None:
            options["type"] = entry.type
        parser.add_argument(flag, help=entry.help, **options)
    parser.add_argument("--output", required=True, help="Where to write the profile.")
    args = parser.parse_args(argv)

    try:
        profile = build_arena_native_control_live_profile(
            **{
                entry.kwarg: getattr(args, flag.removeprefix("--").replace("-", "_"))
                for flag, entry in FLAGS.items()
            }
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "task_evaluation_launch_profile.v1",
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
