#!/usr/bin/env python3
"""Issue a one-attempt paid authority for either link of the appearance chain.

The appearance path is a chained, spend-accumulating campaign rather than a set
of independent lanes:

    prior Aura terminal  -->  ArtiFixer3D  -->  paired-target native import

Each link's authority validates its predecessor's terminal evidence and carries
the campaign's running spend forward against a shared cap, so an authority
cannot be minted out of order and cannot be minted twice.

Both `materialize_*_paid_attempt_authority` functions already existed and
neither could be called from a command line. That is the same defect #512 and
#520 fixed for lanes and bundle modules, in a third scope: modules that mint an
authority rather than seal a bundle. Without an entry point the campaign was
authorizable only from a Python session, which is not a production path.

One script with two subcommands, because the two links differ only in which
predecessor evidence they demand -- and a per-link copy would be a per-link
opportunity to drop one of those demands.

The flag table below *is* the call: the parser and the keyword arguments are
both built from it, and a contract test derives the left column from each
materializer's own signature. That is deliberate. The first cut of this script
hand-listed its flags and silently dropped four of the six predecessor
parameters, which would have left a second ArtiFixer3D attempt unable to
account for the first attempt's spend.

What cannot be derived is the authorization itself: who is approving one paid
attempt, and what they are approving it against. Both are required and recorded
verbatim.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from blueprint_pipeline.paired_target_native_import_vast import (
    materialize_paired_target_native_import_paid_attempt_authority,
)
from blueprint_pipeline.public_scene_artifixer3d_vast import (
    materialize_artifixer3d_paid_attempt_authority,
)


@dataclass(frozen=True)
class Param:
    """One materializer keyword and the flag that supplies it."""

    flag: str
    help: str
    required: bool = False
    type: Callable[[str], Any] | None = None
    default: Any = None
    accumulate: bool = False


#: Supplied the same way by both links.
SHARED: dict[str, Param] = {
    "bundle_receipt_path": Param(
        "--bundle-receipt", "The sealed bundle this attempt will run.", required=True
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
        "Must equal the commit the bundle was built at.",
        required=True,
    ),
    "max_hourly_rate_usd": Param("--max-hourly-rate-usd", "", type=float, default=1.0),
    "hard_cap_usd": Param("--hard-cap-usd", "", type=float, default=10.0),
    "hard_ttl_seconds": Param("--hard-ttl-seconds", "", type=int, default=10_800),
    "output_path": Param("--output", "Where to write the sealed authority.", required=True),
}

#: The four predecessor paths are an all-or-nothing group upstream: passing
#: some but not all raises `artifixer3d_predecessor_attempt_incomplete`. They
#: are what lets a *second* attempt account for the first attempt's spend.
ARTIFIXER_ONLY: dict[str, Param] = {
    "prior_aura_authority_path": Param(
        "--prior-aura-authority",
        "A retired lane's historical authority is still this campaign's spend "
        "anchor. Retiring AuraFusion360 did not delete its receipts.",
        required=True,
    ),
    "prior_terminal_result_path": Param(
        "--prior-terminal-result", "The Aura terminal result that anchor closed on.", required=True
    ),
    "prior_artifixer_authority_path": Param(
        "--prior-artifixer-authority", "Second attempt onward: all four or none."
    ),
    "prior_artifixer_result_path": Param(
        "--prior-artifixer-result", "Second attempt onward: all four or none."
    ),
    "prior_artifixer_cleanup_path": Param(
        "--prior-artifixer-cleanup", "Second attempt onward: all four or none."
    ),
    "prior_artifixer_provider_zero_path": Param(
        "--prior-artifixer-provider-zero", "Second attempt onward: all four or none."
    ),
    "supplemental_prior_spend_reconciliation_path": Param(
        "--supplemental-prior-spend-reconciliation",
        "Campaign spend the receipt chain does not itself carry. Omitting this "
        "under-counts the total against the shared cap.",
    ),
}

PAIRED_ONLY: dict[str, Param] = {
    "prior_artifixer_authority_path": Param(
        "--prior-artifixer-authority", "The completed ArtiFixer3D attempt.", required=True
    ),
    "prior_artifixer_result_path": Param(
        "--prior-artifixer-result", "Its terminal result.", required=True
    ),
    "prior_artifixer_cleanup_path": Param(
        "--prior-artifixer-cleanup", "Its object store cleanup.", required=True
    ),
    "prior_artifixer_provider_zero_path": Param(
        "--prior-artifixer-provider-zero", "Its provider-zero receipt.", required=True
    ),
    "supplemental_prior_spend_reconciliation_path": Param(
        "--supplemental-prior-spend-reconciliation",
        "Campaign spend the receipt chain does not itself carry.",
    ),
    "prior_native_preallocation_provider_zero_path": Param(
        "--prior-native-preallocation-provider-zero",
        "Provider zero observed before this lane allocates anything.",
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
}


@dataclass(frozen=True)
class Link:
    summary: str
    materialize: Callable[..., Any]
    params: Mapping[str, Param] = field(default_factory=dict)


LINKS: dict[str, Link] = {
    "artifixer3d": Link(
        "Head of the chain; anchors on a prior Aura terminal.",
        materialize_artifixer3d_paid_attempt_authority,
        {**SHARED, **ARTIFIXER_ONLY},
    ),
    "paired-target": Link(
        "Import gate; anchors on a completed ArtiFixer3D run.",
        materialize_paired_target_native_import_paid_attempt_authority,
        {
            **SHARED,
            "hard_cap_usd": Param("--hard-cap-usd", "", type=float, default=2.0),
            "hard_ttl_seconds": Param("--hard-ttl-seconds", "", type=int, default=7_200),
            **PAIRED_ONLY,
        },
    ),
}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="link", required=True)
    for name, link in LINKS.items():
        target = sub.add_parser(name, help=link.summary)
        for keyword, param in link.params.items():
            options: dict[str, Any] = {"dest": keyword, "help": param.help or None}
            if param.accumulate:
                options["action"] = "append"
            if param.type is not None:
                options["type"] = param.type
            if param.required:
                options["required"] = True
            else:
                options["default"] = param.default
            target.add_argument(param.flag, **options)
    return parser


def call_arguments(link: Link, namespace: argparse.Namespace) -> dict[str, Any]:
    """The materializer keywords this invocation supplies, and nothing else."""

    supplied = vars(namespace)
    arguments: dict[str, Any] = {}
    for keyword, param in link.params.items():
        value = supplied.get(keyword, param.default)
        if param.accumulate:
            value = tuple(value or ())
        arguments[keyword] = value
    if not str(arguments.get("authorized_on") or "").strip():
        arguments["authorized_on"] = _today()
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    link = LINKS[args.link]

    try:
        authority = link.materialize(**call_arguments(link, args))
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
        "link": args.link,
        "output": str(Path(args.output_path).expanduser().resolve()),
        "provider_mutation_performed": False,
    }
    if isinstance(authority, Mapping):
        for key in (
            "authorization_digest",
            "bundle_sha256",
            "hard_attempt_spend_cap_usd",
            "prior_goal_spend_usd",
            "aggregate_goal_spend_cap_usd",
            "active_instance_allowlist",
        ):
            if key in authority:
                summary[key] = authority[key]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
