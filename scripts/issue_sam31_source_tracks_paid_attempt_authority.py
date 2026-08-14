#!/usr/bin/env python3
"""Issue the one-attempt paid authority for the SAM 3.1 source-track lane.

`scripts/build_sam31_source_tracks_live_profile.py` will not build a profile
without `--attempt-authority`, and the only function that mints one --
`materialize_sam31_paid_attempt_authority` -- could be called from no script.
So the lane had a live profile builder whose required input no production path
could produce, and authorizing a SAM run meant opening a Python session.

That is the same defect as #512 (lanes), #520 (bundle modules) and #523
(authority materializers), in the one authority module #523's scan reached but
its fix did not. The other three authority modules already delegate to a script
here; this one did not.

The flag table below *is* the call: the parser and the keyword arguments are
both built from it, and `tests/test_sam31_source_tracks_authority_issuer.py`
derives the left column from the materializer's own signature. #523's first cut
hand-listed its flags and silently dropped four of six predecessor parameters,
which on a spend-accumulating lane is spend that goes uncounted.

Every money and TTL bound is required rather than defaulted, which departs from
`issue_appearance_chain_paid_attempt_authority.py` deliberately. Upstream
refuses unless `hard_cap_usd` and `hard_ttl_seconds` equal the request's own
`max_spend_usd` and `hard_ttl_seconds` exactly, so a default here would not be a
convenience -- it would be a second number that has to agree with a file the
operator never opened. `aggregate-spend-before-usd` is required for the same
reason: defaulting it to zero is how a second attempt stops accounting for the
first attempt's spend against the shared cap.

What cannot be derived at all is the authorization: who is approving one paid
attempt, and what they are approving it against. Both are required and recorded
verbatim.

Reads retained bytes only; performs no provider mutation and rents nothing. The
authority it writes is single-use, and reissuing over an existing file is
refused upstream rather than overwritten here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from blueprint_pipeline.sam31_paid_attempt_authority import (
    materialize_sam31_paid_attempt_authority,
)

MATERIALIZE = materialize_sam31_paid_attempt_authority


@dataclass(frozen=True)
class Param:
    """One materializer keyword and the flag that supplies it."""

    flag: str
    help: str
    required: bool = False
    type: Callable[[str], Any] | None = None
    default: Any = None
    accumulate: bool = False


PARAMS: dict[str, Param] = {
    "request_path": Param(
        "--request", "The sealed SAM 3.1 GPU admission request this attempt runs.", required=True
    ),
    "bundle_path": Param("--bundle", "The input bundle the request is bound to.", required=True),
    "bundle_receipt_path": Param(
        "--bundle-receipt", "The receipt that sealed that bundle.", required=True
    ),
    "authorized_by": Param(
        "--authorized-by", "Who is approving one paid attempt.", required=True
    ),
    "authorization_reference": Param(
        "--authority-reference",
        "What they approved: the instruction this attempt rests on.",
        required=True,
    ),
    "authorized_on": Param("--authorized-on", "Defaults to today, UTC."),
    "blueprint_commit": Param(
        "--blueprint-commit",
        "Must equal the request's own `source_commit_sha`, which is the commit "
        "the control plane has to be running for the allocator to admit this.",
        required=True,
    ),
    "max_hourly_rate_usd": Param(
        "--max-hourly-rate-usd",
        "Refused unless rate x TTL stays inside the hard cap.",
        required=True,
        type=float,
    ),
    "hard_cap_usd": Param(
        "--hard-cap-usd",
        "Must equal the request's `max_spend_usd`; ceiling is $2.00.",
        required=True,
        type=float,
    ),
    "hard_ttl_seconds": Param(
        "--hard-ttl-seconds",
        "Must equal the request's `hard_ttl_seconds`; ceiling is 3600.",
        required=True,
        type=int,
    ),
    "aggregate_goal_spend_before_attempt_usd": Param(
        "--aggregate-spend-before-usd",
        "Campaign spend already committed. Anything above zero also requires "
        "--prior-spend-reconciliation.",
        required=True,
        type=float,
    ),
    "aggregate_goal_spend_cap_usd": Param(
        "--aggregate-spend-cap-usd",
        "The shared campaign cap this attempt has to fit under.",
        required=True,
        type=float,
    ),
    "prior_spend_reconciliation_path": Param(
        "--prior-spend-reconciliation",
        "Digest-bound proof of the prior spend figure, terminal and provider-zero.",
    ),
    "allowed_active_instance_ids": Param(
        "--allow-active-instance",
        "Repeatable. Instances that may already be running when this attempt "
        "starts. Anything else active fails the attempt closed, which is what "
        "keeps a concurrent lane's instances from being treated as ours.",
        type=int,
        default=(),
        accumulate=True,
    ),
    "output_path": Param("--output", "Where to write the sealed authority.", required=True),
}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for keyword, param in PARAMS.items():
        options: dict[str, Any] = {"dest": keyword, "help": param.help or None}
        if param.accumulate:
            # `action="append"` appends to whatever default it is given, so a
            # tuple default raises `AttributeError` the first time the flag is
            # actually passed. Start from `None` and let `call_arguments`
            # restore the declared default.
            options["action"] = "append"
            options["default"] = None
        elif param.required:
            options["required"] = True
        else:
            options["default"] = param.default
        if param.type is not None:
            options["type"] = param.type
        parser.add_argument(param.flag, **options)
    return parser


def call_arguments(namespace: argparse.Namespace) -> dict[str, Any]:
    """The materializer keywords this invocation supplies, and nothing else."""

    supplied = vars(namespace)
    arguments: dict[str, Any] = {}
    for keyword, param in PARAMS.items():
        value = supplied.get(keyword, param.default)
        if param.accumulate:
            value = tuple(value or ())
        arguments[keyword] = value
    if not str(arguments.get("authorized_on") or "").strip():
        arguments["authorized_on"] = _today()
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        authority = MATERIALIZE(**call_arguments(args))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    summary: dict[str, Any] = {
        "status": "issued",
        "output": str(Path(args.output_path).expanduser().resolve()),
        "provider_mutation_performed": False,
    }
    if isinstance(authority, Mapping):
        for key in (
            "authorization_digest",
            "request_authority_id",
            "blueprint_commit",
            "hard_attempt_spend_cap_usd",
            "maximum_hourly_rate_usd",
            "maximum_single_resource_ttl_seconds",
            "aggregate_goal_spend_before_attempt_usd",
            "aggregate_goal_spend_cap_usd",
            "active_instance_allowlist",
        ):
            if key in authority:
                summary[key] = authority[key]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
