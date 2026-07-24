"""Audit which claim gates can actually be satisfied, and which cannot.

A fail-closed platform accumulates blockers, and a blocker list is only
actionable if a reader can tell what kind of blocker each one is. "38 blockers"
mixes at least four different situations:

``satisfiable_now``
    The evidence could be supplied today; nobody has supplied it.
``awaiting_execution``
    Blocked on a run that has not happened -- a GPU job, a trial campaign.
``awaiting_upstream_release``
    Blocked on a third party publishing code, weights, or results.
``unreachable_by_construction``
    No input can clear it, because the code that would set the required state
    never sets it. This is a defect, not a status.
``divergent_registry``
    Two parts of the system define incompatible vocabularies for the same
    concept, so satisfying one makes the other harder to satisfy coherently.

Only the first three are progress. The fourth is a bug wearing a status's
clothes: a checklist that reports it alongside genuine pending work invites the
team to keep waiting for something that will never arrive.

The audit **probes** rather than asserts wherever it can. It calls the real
validators and scans the real source, so a gate that someone later repairs stops
being reported as dead without anybody editing this file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


AUDIT_SCHEMA_VERSION = "gate_reachability_audit.v1"

SATISFIABLE_NOW = "satisfiable_now"
AWAITING_EXECUTION = "awaiting_execution"
AWAITING_UPSTREAM_RELEASE = "awaiting_upstream_release"
UNREACHABLE_BY_CONSTRUCTION = "unreachable_by_construction"
DIVERGENT_REGISTRY = "divergent_registry"

CLASSIFICATIONS = (
    SATISFIABLE_NOW,
    AWAITING_EXECUTION,
    AWAITING_UPSTREAM_RELEASE,
    UNREACHABLE_BY_CONSTRUCTION,
    DIVERGENT_REGISTRY,
)

# Claim fields that are emitted as literal ``False`` in source. A literal is not
# automatically wrong -- several are deliberate, permanent claim boundaries --
# but a caller cannot flip one by supplying better evidence, so each needs a
# recorded intent rather than being left to look like pending work.
HARDCODED_CLAIM_FIELD_TARGETS: tuple[tuple[str, str], ...] = (
    ("benchmark_uncertainty.py", "public_rank_fidelity_claim_eligible"),
    ("oscar_cosmos_wam_evaluator.py", "full_closed_loop_episode_proven"),
)

_LITERAL_FALSE_RE_TEMPLATE = r'"{field}"\s*:\s*False\b'


def _repo_source_root() -> Path:
    return Path(__file__).resolve().parent


def _probe_external_study_gate() -> dict[str, Any]:
    """Can ``validate_external_study`` ever report a validated study?

    ``sc3_eval_protocol`` makes public rank-fidelity eligibility conditional on
    that validator returning ``"validated"``. This calls it with a
    maximally-cooperative payload and reports what comes back.
    """

    from .external_study_protocols import validate_external_study

    payload = {
        "status": "accepted_frozen_study",
        "independent_reproduction": {"status": "passed"},
        "human_protocol": {"status": "accepted"},
    }
    try:
        result = validate_external_study(payload)
    except Exception as error:  # noqa: BLE001 - a raising validator is also a finding
        return {
            "probe": "validate_external_study",
            "observed_status": None,
            "error": f"{type(error).__name__}: {error}",
            "validated_status_reachable": False,
        }
    observed = str(result.get("status") or "")
    source = (_repo_source_root() / "external_study_protocols.py").read_text(encoding="utf-8")
    return {
        "probe": "validate_external_study",
        "observed_status": observed,
        "validated_literal_present_in_source": '"validated"' in source,
        "validated_status_reachable": observed == "validated",
    }


def _probe_ood_axis_registries() -> dict[str, Any]:
    """Do the two OOD axis vocabularies agree?"""

    from .sc3_fidelity_contracts import SC3_OOD_AXES

    # decision_grade_ranking pins its required set inline.
    decision_axes = {"site", "task", "embodiment", "viewpoint", "appearance"}
    sc3_axes = set(SC3_OOD_AXES)
    return {
        "probe": "ood_axis_registries",
        "sc3_frozen_axes": sorted(sc3_axes),
        "decision_grade_required_axes": sorted(decision_axes),
        "identical": sc3_axes == decision_axes,
        "decision_axes_absent_from_sc3": sorted(decision_axes - sc3_axes),
        "sc3_axes_absent_from_decision": sorted(sc3_axes - decision_axes),
        # They validate different artifacts, so this is a vocabulary divergence
        # rather than a contradiction that makes either one unsatisfiable.
        "binds_the_same_artifact": False,
    }


def _scan_hardcoded_claim_fields() -> list[dict[str, Any]]:
    """Find claim fields emitted as literal ``False`` in source."""

    root = _repo_source_root()
    findings: list[dict[str, Any]] = []
    for filename, field in HARDCODED_CLAIM_FIELD_TARGETS:
        path = root / filename
        if not path.is_file():
            findings.append(
                {
                    "file": filename,
                    "field": field,
                    "occurrences": [],
                    "scanned": False,
                }
            )
            continue
        pattern = re.compile(_LITERAL_FALSE_RE_TEMPLATE.format(field=re.escape(field)))
        occurrences = [
            index
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if pattern.search(line)
        ]
        findings.append(
            {
                "file": filename,
                "field": field,
                "occurrences": occurrences,
                "occurrence_count": len(occurrences),
                "scanned": True,
            }
        )
    return findings


def audit_gate_reachability() -> dict[str, Any]:
    """Classify known claim gates by whether any input can satisfy them."""

    external = _probe_external_study_gate()
    ood = _probe_ood_axis_registries()
    hardcoded = _scan_hardcoded_claim_fields()

    gates: list[dict[str, Any]] = []

    external_dead = not external["validated_status_reachable"]
    gates.append(
        {
            "gate_id": "external_study_validated",
            "classification": (
                UNREACHABLE_BY_CONSTRUCTION if external_dead else SATISFIABLE_NOW
            ),
            "evidence": (
                "external_study_protocols.validate_external_study returns "
                f"{external['observed_status']!r} for every input"
            ),
            "probe": external,
            "what_would_change_it": (
                "give validate_external_study a path that returns 'validated' when a "
                "frozen, independently reproduced study is supplied"
            ),
        }
    )
    for dependent in (
        "sc3_eval_protocol.public_rank_fidelity_claim_eligible",
        "sc3_eval_protocol.claim_ready",
        "sc3_eval_protocol.eligible_preregistered_external_rank_fidelity",
    ):
        gates.append(
            {
                "gate_id": dependent,
                "classification": (
                    UNREACHABLE_BY_CONSTRUCTION if external_dead else AWAITING_EXECUTION
                ),
                "evidence": (
                    "conjunction includes external_study_validation.status == 'validated', "
                    "which the validator never returns"
                ),
                "what_would_change_it": "repair external_study_validated",
            }
        )

    gates.append(
        {
            "gate_id": "ood_axis_vocabulary_agreement",
            "classification": SATISFIABLE_NOW if ood["identical"] else DIVERGENT_REGISTRY,
            "evidence": (
                f"sc3 frozen axes {ood['sc3_frozen_axes']} versus decision-grade "
                f"required axes {ood['decision_grade_required_axes']}"
            ),
            "probe": ood,
            "what_would_change_it": (
                "reconcile the two axis vocabularies, or state explicitly that they "
                "describe different generalization structures"
            ),
        }
    )

    for finding in hardcoded:
        occurrences = finding.get("occurrences") or []
        gates.append(
            {
                "gate_id": f"{finding['file']}::{finding['field']}",
                "classification": (
                    UNREACHABLE_BY_CONSTRUCTION if occurrences else SATISFIABLE_NOW
                ),
                "evidence": (
                    f"emitted as a literal False at line(s) {occurrences}"
                    if occurrences
                    else "no literal False emission found"
                ),
                "probe": finding,
                "what_would_change_it": (
                    "derive the field from supplied evidence, or record it as a "
                    "permanent claim boundary so it is not read as pending work"
                ),
            }
        )

    counts = {name: 0 for name in CLASSIFICATIONS}
    for gate in gates:
        counts[gate["classification"]] = counts.get(gate["classification"], 0) + 1

    dead = [
        gate["gate_id"]
        for gate in gates
        if gate["classification"] == UNREACHABLE_BY_CONSTRUCTION
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "gates": gates,
        "counts_by_classification": counts,
        "unreachable_gate_ids": sorted(dead),
        "unreachable_gate_count": len(dead),
        "status": "clean" if not dead else "unreachable_gates_present",
        "claim_boundary": {
            "audit_reports_gate_reachability_not_evidence_quality": True,
            "a_reachable_gate_is_not_a_satisfied_gate": True,
            "permanent_claim_boundaries_may_be_intentionally_unreachable": True,
        },
    }


def classify_blockers(
    blockers: Sequence[str], *, audit: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Split a blocker list by whether waiting could ever clear it."""

    report = dict(audit or audit_gate_reachability())
    dead_tokens = {
        str(gate_id).split("::")[-1].split(".")[-1]
        for gate_id in report.get("unreachable_gate_ids", []) or []
    }
    classified: list[dict[str, Any]] = []
    for blocker in blockers:
        text = str(blocker)
        matched = next((token for token in dead_tokens if token and token in text), None)
        classified.append(
            {
                "blocker": text,
                "classification": (
                    UNREACHABLE_BY_CONSTRUCTION if matched else SATISFIABLE_NOW
                ),
                "matched_gate_token": matched,
            }
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "classified_blockers": classified,
        "unreachable_blocker_count": sum(
            1
            for row in classified
            if row["classification"] == UNREACHABLE_BY_CONSTRUCTION
        ),
        "total_blocker_count": len(classified),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit which Blueprint claim gates can actually be satisfied"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fail-on-unreachable",
        action="store_true",
        help="exit non-zero when an unreachable-by-construction gate is present",
    )
    args = parser.parse_args(argv)

    report = audit_gate_reachability()
    if args.output:
        from .common import write_json

        write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_unreachable and report["unreachable_gate_count"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
