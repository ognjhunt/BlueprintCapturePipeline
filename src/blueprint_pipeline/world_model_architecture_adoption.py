"""Separate adopting a published architecture from admitting its release.

Blueprint's RoboWorld admission checklist conflated two genuinely different
decisions behind one ``awaiting_upstream_release`` status:

1. **Admitting the upstream backend** -- running RoboWorld's released code and
   weights as an evaluator engine, and reproducing its published results. That
   is legitimately blocked: the code is not released, so nothing can be pinned,
   digested, or reproduced.
2. **Adopting the architectural recipe** -- building an action-conditioned
   world model using published design ideas (frame-causal attention, action
   cross-attention injected per frame, a few-step denoising schedule matched
   between training and inference) on components Blueprint already holds under
   permissive licences.

The second is not blocked by the first. It is blocked only by contract text
that treated them as one thing. The distinction matters commercially: the
published result was obtained on a 1.3B backbone at batch size 8, and Blueprint
already pins that backbone family under Apache-2.0 in its OSCAR runtime asset
contract -- currently using only its VAE.

Nothing here weakens a reproduction claim. Separating the tracks makes the
claims *stricter*, because a model built on track 2 is Blueprint-authored and
may not describe itself using the paper's name, metrics, or reported numbers.
It must earn its own evaluator qualification through the ordinary evidence
gates, exactly like any other backend.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .oscar_runtime_asset_contract import OSCAR_RUNTIME_LICENSE_COMPONENTS


ADOPTION_SCHEMA_VERSION = "world_model_architecture_adoption.v1"

UPSTREAM_ADMISSION_TRACK = "upstream_backend_admission"
ARCHITECTURE_ADOPTION_TRACK = "architecture_adoption_on_licensed_components"

# Licences under which Blueprint may build a derived model without a separate
# negotiated grant. Anything else routes to the upstream-admission track.
PERMISSIVE_LICENCES = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause"})

# What the architecture-adoption track is allowed to take from published work:
# design ideas, not artifacts and not results.
ADOPTABLE_DESIGN_ELEMENTS = (
    "frame_causal_attention",
    "per_frame_action_cross_attention",
    "few_step_denoising_schedule_matched_between_training_and_inference",
    "sliding_window_kv_cache",
    "graded_task_progress_scoring_rubric",
)

# What it may never take.
NON_ADOPTABLE_WITHOUT_UPSTREAM_RELEASE = (
    "upstream_model_weights",
    "upstream_training_code",
    "upstream_reported_metrics",
    "upstream_benchmark_result_attribution",
    "upstream_project_name_as_blueprint_capability",
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def licensed_component_inventory() -> list[dict[str, Any]]:
    """Components Blueprint already pins, with their build eligibility."""

    inventory: list[dict[str, Any]] = []
    for component in OSCAR_RUNTIME_LICENSE_COMPONENTS:
        licence = _string(component.get("declared_license"))
        inventory.append(
            {
                "component_id": component.get("component_id"),
                "source_id": component.get("source_id"),
                "revision": component.get("revision"),
                "declared_license": licence,
                "permissively_licensed": licence in PERMISSIVE_LICENCES,
                "revision_pinned": len(_string(component.get("revision"))) == 40,
            }
        )
    return inventory


def build_architecture_adoption_plan(
    *,
    proposal_id: str,
    design_elements: Sequence[str],
    base_component_ids: Sequence[str],
    upstream_admission_status: str = "awaiting_upstream_release",
    declared_as_blueprint_authored: bool = False,
    inherits_upstream_metrics: bool = False,
    uses_upstream_weights_or_code: bool = False,
) -> dict[str, Any]:
    """Decide what an architecture-adoption proposal is authorised to do.

    Authorisation here is narrow: it permits *building*. It confers no evaluator
    standing whatsoever -- the resulting model still has to pass the ordinary
    evidence profiles, calibration anchors and qualification workflow before any
    of its output can back a claim.
    """

    blockers: list[str] = []
    if not _string(proposal_id):
        blockers.append("architecture_adoption_proposal_id_missing")

    inventory = {row["component_id"]: row for row in licensed_component_inventory()}
    selected: list[dict[str, Any]] = []
    for component_id in base_component_ids:
        row = inventory.get(_string(component_id))
        if row is None:
            blockers.append(f"architecture_adoption_component_not_pinned:{component_id}")
            continue
        if not row["permissively_licensed"]:
            blockers.append(
                f"architecture_adoption_component_not_permissively_licensed:{component_id}"
            )
        if not row["revision_pinned"]:
            blockers.append(f"architecture_adoption_component_revision_unpinned:{component_id}")
        selected.append(row)
    if not selected:
        blockers.append("architecture_adoption_requires_at_least_one_pinned_component")

    requested = [_string(item) for item in design_elements if _string(item)]
    unadoptable = [item for item in requested if item not in ADOPTABLE_DESIGN_ELEMENTS]
    for item in unadoptable:
        blockers.append(f"architecture_adoption_element_not_adoptable:{item}")
    if not requested:
        blockers.append("architecture_adoption_requires_at_least_one_design_element")

    # The three ways this track could quietly become the other one.
    if uses_upstream_weights_or_code:
        blockers.append("architecture_adoption_may_not_use_upstream_weights_or_code")
    if inherits_upstream_metrics:
        blockers.append("architecture_adoption_may_not_inherit_upstream_metrics")
    if not declared_as_blueprint_authored:
        blockers.append("architecture_adoption_must_be_declared_blueprint_authored")

    blockers = sorted(set(blockers))
    authorized = not blockers
    return {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "proposal_id": _string(proposal_id),
        "track": ARCHITECTURE_ADOPTION_TRACK,
        "status": "authorized_to_build" if authorized else "blocked",
        "design_elements": requested,
        "base_components": selected,
        "tracks": {
            UPSTREAM_ADMISSION_TRACK: {
                "status": _string(upstream_admission_status),
                "blocked_by": "third_party_code_and_weight_release",
                "unblocks": [
                    "running the upstream backend as an evaluator engine",
                    "reproducing the upstream published result",
                    "attributing the upstream benchmark numbers",
                ],
                "independent_of_architecture_adoption": True,
            },
            ARCHITECTURE_ADOPTION_TRACK: {
                "status": "authorized_to_build" if authorized else "blocked",
                "blocked_by": None if authorized else "proposal_contract",
                "unblocks": [
                    "building a Blueprint-authored action-conditioned world model "
                    "on already-licensed components"
                ],
                "does_not_unblock": [
                    "any evaluator claim",
                    "any rank-fidelity claim",
                    "any reproduction claim",
                    "use of the upstream project's name or reported metrics",
                ],
            },
        },
        "non_adoptable_without_upstream_release": list(NON_ADOPTABLE_WITHOUT_UPSTREAM_RELEASE),
        "blockers": blockers,
        "claim_boundary": {
            "authorization_is_to_build_not_to_claim": True,
            "resulting_model_is_blueprint_authored_not_a_reproduction": True,
            "resulting_model_must_pass_ordinary_evaluator_qualification": True,
            "upstream_metrics_are_not_inherited": True,
            "public_claim_upgrade_allowed": False,
        },
    }


def backend_selection_principle() -> dict[str, Any]:
    """Record what backend preference ordering should actually be based on.

    The backend catalogue orders candidates by parameter scale. The evidence the
    repository itself records runs the other way: the published correlations it
    tracks are inversely ordered against model size, and what varies across them
    is the training objective and the scoring rubric, neither of which is a
    capacity property. Scale is therefore a cost input, not a quality ranking.
    """

    return {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "principle": "order_backends_by_measured_fidelity_not_parameter_scale",
        "ranking_inputs": [
            "measured rank fidelity under a frozen Blueprint protocol",
            "evaluator test-retest reliability",
            "rollout throughput per dollar",
            "licence and commercial eligibility",
            "abstention behaviour outside the qualified domain",
        ],
        "explicitly_not_a_ranking_input": [
            "parameter count",
            "paper reputation",
            "vendor identity",
        ],
        "rationale": (
            "the correlations this repository records are inversely ordered "
            "against parameter scale, so scale cannot serve as a quality proxy"
        ),
        "claim_boundary": {
            "principle_is_not_a_measurement": True,
            "no_backend_is_ranked_until_measured_under_one_frozen_protocol": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan architecture adoption on already-licensed components"
    )
    parser.add_argument("--input", required=True, help="adoption proposal JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = _mapping(json.loads(Path(args.input).read_text(encoding="utf-8")))
    plan = build_architecture_adoption_plan(
        proposal_id=_string(payload.get("proposal_id")),
        design_elements=payload.get("design_elements", []) or [],
        base_component_ids=payload.get("base_component_ids", []) or [],
        upstream_admission_status=_string(payload.get("upstream_admission_status"))
        or "awaiting_upstream_release",
        declared_as_blueprint_authored=bool(payload.get("declared_as_blueprint_authored")),
        inherits_upstream_metrics=bool(payload.get("inherits_upstream_metrics")),
        uses_upstream_weights_or_code=bool(payload.get("uses_upstream_weights_or_code")),
    )
    write_json(Path(args.output), plan)
    print(json.dumps({"path": args.output, "status": plan["status"]}, sort_keys=True))
    return 0 if plan["status"] == "authorized_to_build" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
