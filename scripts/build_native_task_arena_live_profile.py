#!/usr/bin/env python3
"""Build a live launch profile for any link of the native Arena chain.

The Arena family is three probe kinds over one transport, and they are ordered:

    construction  -->  controls  -->  policy

Controls consumes the construction result; policy consumes both. The allocator
enforces that itself -- it appends
`native_task_arena_construction_result` / `native_task_arena_control_result` to
its own missing list -- so a profile that omits a predecessor is refused after
a provider has already been handed over. Refusing here costs nothing.

All three had bundles, a transport, and an allocator branch, and no launch
profile, which is the one thing that carries a lane across the website
boundary. `tests/test_website_reachable_probe_kinds.py` recorded them as
`awaiting_builder`; this is what removes them from that row.

One script with three subcommands rather than three near-copies. The links
differ only in which predecessor results they carry, and a per-link copy would
be a per-link opportunity to drop one -- which is exactly how the appearance
chain issuer lost four of its six predecessor parameters.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.native_task_arena_construction_bundle import (
    PROBE_KIND as CONSTRUCTION_PROBE_KIND,
)
from blueprint_pipeline.native_task_arena_controls_bundle import (
    PROBE_KIND as CONTROLS_PROBE_KIND,
)
from blueprint_pipeline.native_task_arena_policy_bundle import PROBE_KIND as POLICY_PROBE_KIND
from blueprint_pipeline.native_task_arena_paid_authority import (
    AUTHORITY_SCHEMA_VERSION,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

# The allocator refuses anything outside this band for all three probe kinds.
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 14_400

#: The packet directory holds this; it is distinct from the provider-bundle
#: receipt consumed by the shared live-profile skeleton.
PACKET_RECEIPT_NAME = "native_task_arena_packet_receipt.v1.json"


@dataclass(frozen=True)
class ArenaLink:
    """One link, and the predecessor evidence the allocator will demand."""

    probe_kind: str
    profile_id_prefix: str
    job_dirname: str
    #: extra_paths key -> allocator flag, for results this link must carry.
    predecessors: Mapping[str, str]


LINKS: dict[str, ArenaLink] = {
    "construction": ArenaLink(
        CONSTRUCTION_PROBE_KIND, "arena-construction-live", "arena-construction-job", {}
    ),
    "controls": ArenaLink(
        CONTROLS_PROBE_KIND,
        "arena-controls-live",
        "arena-controls-job",
        {"construction_result": "--native-task-arena-construction-result"},
    ),
    "policy": ArenaLink(
        POLICY_PROBE_KIND,
        "arena-policy-live",
        "arena-policy-job",
        {
            "construction_result": "--native-task-arena-construction-result",
            "control_result": "--native-task-arena-control-result",
            "policy_execution_spec": "--native-task-arena-policy-execution-spec",
        },
    ),
}


def _lane_blockers(link: ArenaLink):
    def blockers(context: LaneLiveProfileContext) -> list[str]:
        found: list[str] = []
        if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
            found.append("native_task_arena_budget_invalid")
        receipt = context.receipt
        if receipt.get("implementation_commit") not in (None, context.source_commit):
            found.append(
                f"bundle_commit_not_source_commit:{receipt.get('implementation_commit')}"
            )
        for name in (
            "packet_dir",
            "runtime_source_packet",
            "attempt_authority",
            *link.predecessors,
        ):
            path = context.extra_paths.get(name)
            if path is None or not path.exists():
                # The allocator names the same absence, but only after it has
                # been handed a provider.
                found.append(f"native_task_arena_{name}_missing")
        packet_receipt = context.extra_paths["packet_dir"] / PACKET_RECEIPT_NAME
        if not packet_receipt.is_file():
            found.append("native_task_arena_packet_receipt_missing")
        authority_path = context.extra_paths.get("attempt_authority")
        if authority_path is not None and authority_path.is_file():
            try:
                authority = json.loads(authority_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                authority = None
            bundle_record = (
                authority.get("bundle_receipt")
                if isinstance(authority, Mapping)
                else None
            )
            if (
                not isinstance(authority, Mapping)
                or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
                or authority.get("authorization_digest")
                != canonical_digest(authority, digest_field="authorization_digest")
                or authority.get("blueprint_commit") != context.source_commit
                or authority.get("bundle_sha256")
                != context.receipt.get("bundle_sha256")
                or authority.get("maximum_hourly_rate_usd")
                != context.max_hourly_rate_usd
                or authority.get("hard_attempt_spend_cap_usd")
                != context.max_spend_usd
                or authority.get("maximum_single_resource_ttl_seconds")
                != context.hard_ttl_seconds
                or not isinstance(bundle_record, Mapping)
                or Path(str(bundle_record.get("path") or "")).expanduser().resolve()
                != context.receipt_path
                or bundle_record.get("sha256") != file_digest(context.receipt_path)
                or bundle_record.get("size_bytes")
                != context.receipt_path.stat().st_size
            ):
                found.append("native_task_arena_attempt_authority_invalid")
        for value in context.extra_paths.get("allowed_active_instance_ids", ()) or ():
            if int(value) <= 0:
                found.append("native_task_arena_allowed_active_instance_id_invalid")
        return found

    return blockers


def _lane_argv(link: ArenaLink):
    def argv(context: LaneLiveProfileContext) -> list[str]:
        built = [
            "--native-task-arena-packet", str(context.extra_paths["packet_dir"]),
            "--native-task-arena-runtime-source-packet",
            str(context.extra_paths["runtime_source_packet"]),
            "--native-task-arena-bundle-receipt",
            str(context.receipt_path),
            "--native-task-arena-attempt-authority",
            str(context.extra_paths["attempt_authority"]),
            "--adp-job-dir", context.job_dir(link.job_dirname),
            "--adp-max-hourly-rate-usd", str(context.max_hourly_rate_usd),
            "--adp-max-spend-usd", str(context.max_spend_usd),
            "--adp-hard-ttl-seconds", str(context.hard_ttl_seconds),
        ]
        for name, flag in link.predecessors.items():
            built += [flag, str(context.extra_paths[name])]
        avoidlist = context.extra_paths.get("machine_avoidlist")
        if avoidlist is not None:
            built += ["--adp-machine-avoidlist", str(avoidlist)]
        return built

    return argv


def _immutable_inputs(link: ArenaLink):
    def inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
        packet_receipt = context.extra_paths["packet_dir"] / PACKET_RECEIPT_NAME
        rows = [
            {
                "name": "source_bundle_manifest",
                "path": str(context.receipt_path),
                "digest": file_digest(context.receipt_path),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(packet_receipt),
                "digest": file_digest(packet_receipt),
            },
            {
                "name": "native_task_arena_runtime_source_packet",
                "path": str(context.extra_paths["runtime_source_packet"]),
                "digest": file_digest(context.extra_paths["runtime_source_packet"]),
            },
            {
                "name": "native_task_arena_attempt_authority",
                "path": str(context.extra_paths["attempt_authority"]),
                "digest": file_digest(context.extra_paths["attempt_authority"]),
            },
        ]
        # Each predecessor result is pinned by digest: this link's verdict is
        # only about the packet it actually consumed.
        for name in link.predecessors:
            path = context.extra_paths[name]
            rows.append(
                {"name": f"native_task_arena_{name}", "path": str(path), "digest": file_digest(path)}
            )
        return rows

    return inputs


def _spec(
    link: ArenaLink | str, *, with_avoidlist: bool = False
) -> LaneLiveProfileSpec:
    if isinstance(link, str):
        # Shared builder-contract probes call candidate factories with a
        # placeholder string. Real launches pass the exact ArenaLink below.
        link = LINKS.get(link, LINKS["construction"])
    return LaneLiveProfileSpec(
        profile_id_prefix=link.profile_id_prefix,
        profile_builder="build_native_task_arena_live_profile.py",
        probe_kind=link.probe_kind,
        min_ttl_seconds=MIN_TTL_SECONDS,
        max_ttl_seconds=MAX_TTL_SECONDS,
        source_bundle_id=lambda context: (
            f"{link.profile_id_prefix}-{context.source_commit[:12]}"
        ),
        # The dispatcher admits three source kinds and this is not free text.
        # The packet is built over the public scene substrate rather than a new
        # capture, which is what the closest sibling in this family
        # (`build_adp009d_840313_launch_profile`) declares for the same scene.
        source_kind="interiorgs_sage",
        lane_argv=_lane_argv(link),
        immutable_inputs=_immutable_inputs(link),
        lane_blockers=_lane_blockers(link),
        # The skeleton requires every declared path, so the optional
        # avoidlist is only declared on the calls that actually supply one.
        extra_path_names=(
            "packet_dir",
            "runtime_source_packet",
            "attempt_authority",
            *(("machine_avoidlist",) if with_avoidlist else ()),
            *link.predecessors,
        ),
    )


def build_native_task_arena_live_profile(
    *,
    link: str,
    packet_dir: str | Path,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    runtime_source_packet_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    construction_result_path: str | Path | None = None,
    control_result_path: str | Path | None = None,
    policy_execution_spec_path: str | Path | None = None,
    machine_avoidlist_path: str | Path | None = None,
    revision: str | None = None,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = 2.0,
    hard_ttl_seconds: int = 7_200,
) -> dict[str, Any]:
    """Derive a live profile from the packet receipt the link will run."""

    entry = LINKS[link]
    packet = Path(packet_dir).expanduser().resolve()
    supplied: dict[str, Any] = {
        "packet_dir": packet,
        "runtime_source_packet": runtime_source_packet_path,
        "attempt_authority": attempt_authority_path,
        "construction_result": construction_result_path,
        "control_result": control_result_path,
        "policy_execution_spec": policy_execution_spec_path,
        "machine_avoidlist": machine_avoidlist_path,
    }
    missing = [name for name in entry.predecessors if supplied.get(name) is None]
    if missing:
        raise TaskEvaluationLaunchError(
            f"native_task_arena_predecessor_required:{','.join(sorted(missing))}"
        )
    extra = {
        name: value
        for name, value in supplied.items()
        if value is not None
        and (
            name in entry.predecessors
            or name
            in {
                "packet_dir",
                "runtime_source_packet",
                "attempt_authority",
                "machine_avoidlist",
            }
        )
    }
    return build_lane_live_profile(
        _spec(entry, with_avoidlist=machine_avoidlist_path is not None),
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        max_spend_usd=max_spend_usd,
        extra_paths=extra,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="link", required=True)
    for name, entry in LINKS.items():
        target = sub.add_parser(name, help=f"Probe kind {entry.probe_kind}.")
        target.add_argument("--packet-dir", required=True)
        target.add_argument("--bundle-receipt", required=True)
        target.add_argument("--attempt-authority", required=True)
        target.add_argument("--runtime-source-packet", required=True)
        target.add_argument("--source-commit", required=True)
        target.add_argument(
            "--raw-manifest-uri",
            required=True,
            help="Local digest-bound content-addressed publication receipt for this run spec.",
        )
        target.add_argument("--machine-avoidlist")
        target.add_argument(
            "--revision",
            help="Distinguish a rebuilt profile whose inputs changed at the same commit.",
        )
        target.add_argument("--max-hourly-rate-usd", type=float, default=1.0)
        target.add_argument("--max-spend-usd", type=float, default=2.0)
        target.add_argument("--hard-ttl-seconds", type=int, default=7_200)
        target.add_argument("--output", required=True)
        if "construction_result" in entry.predecessors:
            target.add_argument("--construction-result", required=True)
        if "control_result" in entry.predecessors:
            target.add_argument("--control-result", required=True)
        if "policy_execution_spec" in entry.predecessors:
            target.add_argument("--policy-execution-spec", required=True)
    args = parser.parse_args(argv)

    try:
        profile = build_native_task_arena_live_profile(
            link=args.link,
            packet_dir=args.packet_dir,
            bundle_receipt_path=args.bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            runtime_source_packet_path=args.runtime_source_packet,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            construction_result_path=getattr(args, "construction_result", None),
            control_result_path=getattr(args, "control_result", None),
            policy_execution_spec_path=getattr(args, "policy_execution_spec", None),
            machine_avoidlist_path=args.machine_avoidlist,
            revision=args.revision,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            max_spend_usd=args.max_spend_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
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
                "link": args.link,
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
