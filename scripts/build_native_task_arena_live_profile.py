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
    load_verified_native_task_arena_construction_bundle,
)
from blueprint_pipeline.native_task_arena_controls_bundle import (
    PROBE_KIND as CONTROLS_PROBE_KIND,
    load_verified_native_task_arena_controls_bundle,
)
from blueprint_pipeline.native_task_arena_policy_bundle import (
    PROBE_KIND as POLICY_PROBE_KIND,
    load_verified_native_task_arena_policy_bundle,
)
from blueprint_pipeline.native_task_arena_paid_authority import (
    validate_native_task_arena_paid_attempt_authority,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_runtime import (
    NativeTaskArenaRuntimeError,
    validate_native_task_arena_runtime_plan,
)
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
PACKET_REQUEST_NAME = "native_task_arena_packet_request.v1.json"
SCENE_PLAN_NAME = "native_task_arena_scene_plan.v1.json"


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

BUNDLE_LOADERS = {
    CONSTRUCTION_PROBE_KIND: load_verified_native_task_arena_construction_bundle,
    CONTROLS_PROBE_KIND: load_verified_native_task_arena_controls_bundle,
    POLICY_PROBE_KIND: load_verified_native_task_arena_policy_bundle,
}


def _identifier(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if (
        not text
        or not text.replace("_", "a").replace("-", "a").isalnum()
        or text in {".", ".."}
    ):
        raise TaskEvaluationLaunchError(
            f"native_task_arena_expected_{field}_invalid"
        )
    return text


def _read_mapping(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return dict(value)


def _bound_runtime_input_digests(bundle: Mapping[str, Any]) -> dict[str, str]:
    return {
        Path(str(row.get("relative_path") or "")).name: str(
            row.get("sha256") or ""
        )
        for row in bundle.get("bound_runtime_inputs") or []
        if isinstance(row, Mapping)
    }


def _lane_blockers(
    link: ArenaLink, *, expected_scene_id: str, expected_task_id: str
):
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
        packet_request = context.extra_paths["packet_dir"] / PACKET_REQUEST_NAME
        scene_plan_path = context.extra_paths["packet_dir"] / SCENE_PLAN_NAME
        if not packet_receipt.is_file():
            found.append("native_task_arena_packet_receipt_missing")
        prepared_bundle: Mapping[str, Any] | None = None
        packet: Mapping[str, Any] = {}
        request: Mapping[str, Any] = {}
        scene_plan: Mapping[str, Any] = {}
        try:
            packet = _read_mapping(
                packet_receipt,
                error="native_task_arena_packet_receipt_invalid",
            )
            request = _read_mapping(
                packet_request,
                error="native_task_arena_packet_request_invalid",
            )
            scene_plan = _read_mapping(
                scene_plan_path,
                error="native_task_arena_scene_plan_invalid",
            )
            runtime_source = _read_mapping(
                context.extra_paths["runtime_source_packet"],
                error="native_task_arena_runtime_source_packet_invalid",
            )
            observed_scenes = {
                str(packet.get("scene_id") or ""),
                str(request.get("scene_id") or ""),
                str(scene_plan.get("scene_id") or ""),
            }
            observed_tasks = {
                str(packet.get("task_id") or ""),
                str(request.get("task_id") or ""),
                str(scene_plan.get("task_id") or ""),
            }
            if observed_scenes != {expected_scene_id}:
                found.append("native_task_arena_scene_identity_mismatch")
            if observed_tasks != {expected_task_id}:
                found.append("native_task_arena_task_identity_mismatch")
            if (
                packet.get("schema_version")
                != "native_task_arena_packet_receipt.v1"
                or packet.get("status") != "construction_packet_completed"
                or packet.get("receipt_digest")
                != canonical_digest(packet, digest_field="receipt_digest")
                or request.get("schema_version")
                != "native_task_arena_packet_request.v1"
                or request.get("request_digest")
                != canonical_digest(request, digest_field="request_digest")
                or scene_plan.get("schema_version")
                != "native_task_arena_scene_plan.v1"
                or scene_plan.get("plan_digest")
                != canonical_digest(scene_plan, digest_field="plan_digest")
                or packet.get("request_digest") != request.get("request_digest")
                or packet.get("arena_scene_plan_digest")
                != scene_plan.get("plan_digest")
                or runtime_source.get("receipt_digest")
                != canonical_digest(runtime_source, digest_field="receipt_digest")
            ):
                raise ValueError("native_task_arena_predecessor_receipt_invalid")
            loader = BUNDLE_LOADERS[link.probe_kind]
            prepared_bundle = loader(
                context.receipt_path,
                expected_implementation_commit=context.source_commit,
                expected_packet_receipt_digest=str(packet["receipt_digest"]),
                expected_runtime_source_packet_digest=str(
                    runtime_source["receipt_digest"]
                ),
            )
            if (
                prepared_bundle.get("scene_id") != expected_scene_id
                or prepared_bundle.get("task_id") != expected_task_id
                or prepared_bundle.get("request_digest")
                != request.get("request_digest")
                or prepared_bundle.get("arena_scene_plan_digest")
                != scene_plan.get("plan_digest")
            ):
                found.append("native_task_arena_bundle_identity_mismatch")
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            found.append("native_task_arena_provider_bundle_invalid")

        bound_inputs = _bound_runtime_input_digests(prepared_bundle or {})
        construction: Mapping[str, Any] = {}
        if "construction_result" in link.predecessors:
            try:
                construction_path = context.extra_paths["construction_result"]
                construction = _read_mapping(
                    construction_path,
                    error="native_task_arena_construction_result_invalid",
                )
                if (
                    construction.get("schema_version")
                    != "native_task_arena_construction_result.v1"
                    or construction.get("status") != "completed"
                    or construction.get("construction_gate_qualified") is not True
                    or construction.get("scene_plan_digest")
                    != scene_plan.get("plan_digest")
                    or construction.get("result_digest")
                    != canonical_digest(
                        construction, digest_field="result_digest"
                    )
                    or bound_inputs.get(
                        "native_task_arena_construction_result.v1.json"
                    )
                    != file_digest(construction_path)
                ):
                    raise ValueError(
                        "native_task_arena_construction_result_invalid"
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                found.append("native_task_arena_construction_result_invalid")

        controls: Mapping[str, Any] = {}
        if "control_result" in link.predecessors:
            try:
                control_path = context.extra_paths["control_result"]
                controls = _read_mapping(
                    control_path,
                    error="native_task_arena_control_result_invalid",
                )
                if (
                    controls.get("schema_version")
                    != "native_task_arena_control_result.v1"
                    or controls.get("status") != "completed"
                    or controls.get("controls_qualified") is not True
                    or controls.get("scene_plan_digest")
                    != scene_plan.get("plan_digest")
                    or controls.get("construction_result_digest")
                    != construction.get("result_digest")
                    or controls.get("result_digest")
                    != canonical_digest(controls, digest_field="result_digest")
                    or bound_inputs.get(
                        "native_task_arena_control_result.v1.json"
                    )
                    != file_digest(control_path)
                ):
                    raise ValueError("native_task_arena_control_result_invalid")
            except (OSError, ValueError, json.JSONDecodeError):
                found.append("native_task_arena_control_result_invalid")

        if "policy_execution_spec" in link.predecessors:
            try:
                spec_path = context.extra_paths["policy_execution_spec"]
                policy_spec = _read_mapping(
                    spec_path,
                    error="native_task_arena_policy_execution_spec_invalid",
                )
                pair = controls.get("control_pair") or {}
                if (
                    policy_spec.get("schema_version")
                    != "native_task_arena_policy_execution_spec.v1"
                    or policy_spec.get("execution_spec_digest")
                    != canonical_digest(
                        policy_spec, digest_field="execution_spec_digest"
                    )
                    or policy_spec.get("task_id") != expected_task_id
                    or policy_spec.get("cell_id")
                    != (scene_plan.get("scenario") or {}).get("cell_id")
                    or policy_spec.get("scene_plan_digest")
                    != scene_plan.get("plan_digest")
                    or policy_spec.get("construction_result_digest")
                    != construction.get("result_digest")
                    or policy_spec.get("control_result_digest")
                    != controls.get("result_digest")
                    or policy_spec.get("control_pair_digest")
                    != pair.get("pair_digest")
                    or bound_inputs.get(
                        "native_task_arena_policy_execution_spec.v1.json"
                    )
                    != file_digest(spec_path)
                ):
                    raise ValueError(
                        "native_task_arena_policy_execution_spec_invalid"
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                found.append("native_task_arena_policy_execution_spec_invalid")
        authority_path = context.extra_paths.get("attempt_authority")
        if authority_path is not None and authority_path.is_file():
            try:
                authority = json.loads(authority_path.read_text(encoding="utf-8"))
                if not isinstance(authority, Mapping) or prepared_bundle is None:
                    raise ValueError("native_task_arena_attempt_authority_invalid")
                validate_native_task_arena_paid_attempt_authority(
                    authority,
                    prepared_bundle=prepared_bundle,
                    max_hourly_rate_usd=context.max_hourly_rate_usd,
                    hard_cap_usd=context.max_spend_usd,
                    hard_ttl_seconds=context.hard_ttl_seconds,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                found.append("native_task_arena_attempt_authority_invalid")
        for value in context.extra_paths.get("allowed_active_instance_ids", ()) or ():
            if int(value) <= 0:
                found.append("native_task_arena_allowed_active_instance_id_invalid")
        # Ask the adapter itself whether it would accept this packet. Every
        # refusal it raises before construction is a check on the plan, the
        # staged bytes, or the camera rows -- none of which needs Isaac. Two
        # paid attempts were spent discovering answers this call returns for
        # free, so a profile is not built for a packet the runtime refuses:
        # no profile, no authority consumed, no provider.
        packet_dir = context.extra_paths.get("packet_dir")
        if packet_dir is not None and (packet_dir / SCENE_PLAN_NAME).is_file():
            try:
                validate_native_task_arena_runtime_plan(
                    json.loads(
                        (packet_dir / SCENE_PLAN_NAME).read_text(encoding="utf-8")
                    ),
                    bundle_root=packet_dir,
                )
            except NativeTaskArenaRuntimeError as exc:
                found.extend(
                    f"native_task_arena_runtime_would_refuse:{code}"
                    for code in exc.errors
                )
            except (OSError, ValueError) as exc:
                found.append(f"native_task_arena_scene_plan_unreadable:{exc}")
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
        packet_request = context.extra_paths["packet_dir"] / PACKET_REQUEST_NAME
        scene_plan = context.extra_paths["packet_dir"] / SCENE_PLAN_NAME
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
                "name": "native_task_arena_packet_request",
                "path": str(packet_request),
                "digest": file_digest(packet_request),
            },
            {
                "name": "native_task_arena_scene_plan",
                "path": str(scene_plan),
                "digest": file_digest(scene_plan),
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
    link: ArenaLink | str,
    *,
    expected_scene_id: str = "contract_probe_scene",
    expected_task_id: str = "contract_probe_task",
    with_avoidlist: bool = False,
) -> LaneLiveProfileSpec:
    if isinstance(link, str):
        # Shared builder-contract probes call candidate factories with a
        # placeholder string. Real launches pass the exact ArenaLink below.
        link = LINKS.get(link, LINKS["construction"])
    return LaneLiveProfileSpec(
        profile_id_prefix=(
            f"{link.profile_id_prefix}-{expected_scene_id}-{expected_task_id}"
        ),
        profile_builder="build_native_task_arena_live_profile.py",
        probe_kind=link.probe_kind,
        min_ttl_seconds=MIN_TTL_SECONDS,
        max_ttl_seconds=MAX_TTL_SECONDS,
        source_bundle_id=lambda context: (
            f"{link.profile_id_prefix}-{expected_scene_id}-{expected_task_id}-"
            f"{context.source_commit[:12]}"
        ),
        # The dispatcher admits three source kinds and this is not free text.
        # The packet is built over the public scene substrate rather than a new
        # capture, which is what the closest sibling in this family
        # (`build_adp009d_840313_launch_profile`) declares for the same scene.
        source_kind="interiorgs_sage",
        lane_argv=_lane_argv(link),
        immutable_inputs=_immutable_inputs(link),
        lane_blockers=_lane_blockers(
            link,
            expected_scene_id=expected_scene_id,
            expected_task_id=expected_task_id,
        ),
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
    expected_scene_id: str,
    expected_task_id: str,
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
    scene_id = _identifier(expected_scene_id, field="scene_id")
    task_id = _identifier(expected_task_id, field="task_id")
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
        _spec(
            entry,
            expected_scene_id=scene_id,
            expected_task_id=task_id,
            with_avoidlist=machine_avoidlist_path is not None,
        ),
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
        target.add_argument("--scene-id", required=True)
        target.add_argument("--task-id", required=True)
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
            expected_scene_id=args.scene_id,
            expected_task_id=args.task_id,
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
