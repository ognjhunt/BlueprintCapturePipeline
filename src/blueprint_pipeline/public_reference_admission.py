"""Fail-closed admission for an external real-to-sim reference substrate.

The source manifest is descriptive evidence assembled from public primary
sources.  This module, not the author of the manifest, decides whether the
packet is complete enough for ADP-002.  A structurally valid but incomplete
packet produces a digest-bound blocked receipt with the smallest known gaps.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


MANIFEST_SCHEMA_VERSION = "public_reference_admission.v1"
RECEIPT_SCHEMA_VERSION = "public_reference_admission_receipt.v1"
PHASE_LABEL = "retrospective_external_reference"
CLAIM_CEILING = "development_only"

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FORBIDDEN_TRUE_CLAIMS = {
    "customer_value",
    "deployment_readiness",
    "digital_twin",
    "general_policy_ranking",
    "general_sim_to_real_fidelity",
    "physical_safety",
    "prospective_validation",
    "universal_robot_or_simulator_support",
}


class PublicReferenceAdmissionError(ValueError):
    """A malformed or internally contradictory source manifest."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256.fullmatch(_string(value)))


def _is_sha1(value: Any) -> bool:
    return bool(_SHA1.fullmatch(_string(value)))


def _is_identifier(value: Any) -> bool:
    return bool(_IDENTIFIER.fullmatch(_string(value)))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_checkpoint_digest(candidate: Mapping[str, Any]) -> str:
    identity = {
        "candidate_id": candidate.get("candidate_id"),
        "checkpoint_prefix": candidate.get("checkpoint_prefix"),
        "checkpoint_objects": candidate.get("checkpoint_objects"),
    }
    return canonical_digest(identity)


def _validate_license(errors: list[str], value: Any, *, path: str) -> None:
    license_value = _mapping(value)
    if not _string(license_value.get("spdx")):
        errors.append(f"{path}.spdx:missing")
    if not _string(license_value.get("source_url")):
        errors.append(f"{path}.source_url:missing")
    if not _is_sha256(license_value.get("text_sha256")):
        errors.append(f"{path}.text_sha256:invalid")


def _validate_manifest(value: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    blockers: list[str] = []

    if _string(value.get("schema_version")) != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version:must_be:{MANIFEST_SCHEMA_VERSION}")
    if _string(value.get("program_id")) != "arm-decision-proof-v1":
        errors.append("program_id:must_be:arm-decision-proof-v1")
    if not _is_identifier(value.get("reference_id")):
        errors.append("reference_id:invalid")
    source_identity = {
        key: value.get(key)
        for key in (
            "reference_id",
            "source",
            "rights",
            "task",
            "candidates",
            "conditions",
            "physical_reference",
            "claim_boundaries",
        )
    }
    if value.get("source_identity_digest") != canonical_digest(source_identity):
        errors.append("source_identity_digest:mismatch")
    if _string(value.get("phase_label")) != PHASE_LABEL:
        errors.append(f"phase_label:must_be:{PHASE_LABEL}")
    if _string(value.get("claim_ceiling")) != CLAIM_CEILING:
        errors.append(f"claim_ceiling:must_be:{CLAIM_CEILING}")

    source = _mapping(value.get("source"))
    repository = _mapping(source.get("repository"))
    if not _string(repository.get("url")):
        errors.append("source.repository.url:missing")
    if not _is_sha1(repository.get("commit")):
        errors.append("source.repository.commit:invalid")
    if not _is_sha1(repository.get("tree")):
        errors.append("source.repository.tree:invalid")
    _validate_license(errors, repository.get("license"), path="source.repository.license")
    submodules = _rows(repository.get("submodules"))
    if not submodules:
        errors.append("source.repository.submodules:missing")
    for index, submodule in enumerate(submodules):
        prefix = f"source.repository.submodules[{index}]"
        if not _string(submodule.get("path")) or not _string(submodule.get("url")):
            errors.append(f"{prefix}:path_or_url_missing")
        if not _is_sha1(submodule.get("commit")):
            errors.append(f"{prefix}.commit:invalid")
        _validate_license(errors, submodule.get("license"), path=f"{prefix}.license")

    assets = _rows(source.get("asset_bindings"))
    required_asset_roles = {
        "environment_code",
        "object_assets",
        "observation_background",
        "robot_assets",
        "scene_asset",
    }
    asset_roles = {_string(row.get("role")) for row in assets}
    for role in sorted(required_asset_roles - asset_roles):
        errors.append(f"source.asset_bindings:missing_role:{role}")
    for index, asset in enumerate(assets):
        if not _string(asset.get("path")) or not _is_sha1(asset.get("git_object_sha1")):
            errors.append(f"source.asset_bindings[{index}]:path_or_git_object_invalid")
        if _string(asset.get("git_object_type")) not in {"blob", "tree"}:
            errors.append(f"source.asset_bindings[{index}].git_object_type:invalid")

    rights = _mapping(value.get("rights"))
    if _string(rights.get("review_status")) != "terms_recorded_not_accepted_by_agent":
        errors.append("rights.review_status:invalid")
    terms = _rows(rights.get("terms"))
    if not terms:
        errors.append("rights.terms:missing")
    for index, term in enumerate(terms):
        _validate_license(errors, term, path=f"rights.terms[{index}]")

    task = _mapping(value.get("task"))
    for key in ("task_id", "environment_id", "robot_id"):
        if not _is_identifier(task.get(key)):
            errors.append(f"task.{key}:invalid")
    if _string(task.get("task_family")) != "bounded_rigid_object_pick_place":
        errors.append("task.task_family:invalid")
    for key in ("observation_schema", "action_schema", "controller", "reset", "evaluator"):
        if not _mapping(task.get(key)):
            errors.append(f"task.{key}:missing")
    evaluator = _mapping(task.get("evaluator"))
    if evaluator.get("policy_self_report_used") is not False:
        errors.append("task.evaluator.policy_self_report_used:must_be_false")
    if not _is_sha1(evaluator.get("source_git_blob_sha1")):
        errors.append("task.evaluator.source_git_blob_sha1:invalid")

    candidates = _rows(value.get("candidates"))
    if len(candidates) != 2:
        errors.append("candidates:must_contain_exactly_two")
    candidate_ids: list[str] = []
    checkpoint_digests: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_id = _string(candidate.get("candidate_id"))
        if not _is_identifier(candidate_id):
            errors.append(f"candidates[{index}].candidate_id:invalid")
        candidate_ids.append(candidate_id)
        if candidate.get("genuine_public_checkpoint") is not True:
            errors.append(f"candidates[{index}].genuine_public_checkpoint:must_be_true")
        prefix = _string(candidate.get("checkpoint_prefix"))
        objects = _rows(candidate.get("checkpoint_objects"))
        if not prefix or not objects:
            errors.append(f"candidates[{index}].checkpoint_identity:missing")
        declared_total = candidate.get("checkpoint_total_size_bytes")
        computed_total = 0
        for object_index, checkpoint_object in enumerate(objects):
            object_path = f"candidates[{index}].checkpoint_objects[{object_index}]"
            name = _string(checkpoint_object.get("name"))
            size = checkpoint_object.get("size_bytes")
            generation = _string(checkpoint_object.get("generation"))
            if not name.startswith(prefix + "/"):
                errors.append(f"{object_path}.name:outside_prefix")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                errors.append(f"{object_path}.size_bytes:invalid")
            else:
                computed_total += size
            if not generation.isdigit():
                errors.append(f"{object_path}.generation:invalid")
            for key in ("md5_base64", "crc32c_base64"):
                if not _string(checkpoint_object.get(key)):
                    errors.append(f"{object_path}.{key}:missing")
        if declared_total != computed_total:
            errors.append(f"candidates[{index}].checkpoint_total_size_bytes:mismatch")
        checkpoint_digests.append(_candidate_checkpoint_digest(candidate))
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("candidates:candidate_id_duplicate")
    if len(set(checkpoint_digests)) != len(checkpoint_digests):
        errors.append("candidates:checkpoint_identity_duplicate")

    conditions = _rows(value.get("conditions"))
    condition_ids: list[str] = []
    trial_counts: dict[str, int] = {}
    for index, condition in enumerate(conditions):
        condition_id = _string(condition.get("condition_id"))
        if not _is_identifier(condition_id):
            errors.append(f"conditions[{index}].condition_id:invalid")
        condition_ids.append(condition_id)
        trial_count = condition.get("published_physical_trial_count_per_candidate")
        if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count <= 0:
            errors.append(f"conditions[{index}].published_physical_trial_count_per_candidate:invalid")
        else:
            trial_counts[condition_id] = trial_count
        if not _mapping(condition.get("reset_binding")):
            errors.append(f"conditions[{index}].reset_binding:missing")
    if not conditions:
        errors.append("conditions:missing")
    if len(set(condition_ids)) != len(condition_ids):
        errors.append("conditions:condition_id_duplicate")

    physical = _mapping(value.get("physical_reference"))
    source_artifact = _mapping(physical.get("source_artifact"))
    if not _string(source_artifact.get("url")) or not _is_sha256(
        source_artifact.get("sha256")
    ):
        errors.append("physical_reference.source_artifact:invalid")
    if _string(physical.get("outcome_granularity")) != "candidate_by_condition_aggregate":
        errors.append("physical_reference.outcome_granularity:unsupported")
    outcomes_artifact = _mapping(physical.get("outcomes_artifact"))
    if not _string(outcomes_artifact.get("relative_path")):
        errors.append("physical_reference.outcomes_artifact.relative_path:missing")
    if not _is_sha256(outcomes_artifact.get("digest")):
        errors.append("physical_reference.outcomes_artifact.digest:invalid")
    if outcomes_artifact.get("programmatic_release_required") is not True:
        errors.append(
            "physical_reference.outcomes_artifact.programmatic_release_required:must_be_true"
        )
    cells = _rows(physical.get("cell_bindings"))
    observed_pairs: set[tuple[str, str]] = set()
    for index, cell in enumerate(cells):
        candidate_id = _string(cell.get("candidate_id"))
        condition_id = _string(cell.get("condition_id"))
        pair = (candidate_id, condition_id)
        if pair in observed_pairs:
            errors.append(f"physical_reference.cells[{index}]:duplicate_pair")
        observed_pairs.add(pair)
        if candidate_id not in candidate_ids:
            errors.append(f"physical_reference.cells[{index}].candidate_id:unknown")
        if condition_id not in condition_ids:
            errors.append(f"physical_reference.cells[{index}].condition_id:unknown")
        if cell.get("trial_count") != trial_counts.get(condition_id):
            errors.append(f"physical_reference.cell_bindings[{index}].trial_count:mismatch")
    expected_pairs = {(candidate_id, condition_id) for candidate_id in candidate_ids for condition_id in condition_ids}
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        if missing:
            errors.append("physical_reference.cell_bindings:missing_pairs:" + ",".join(f"{a}/{b}" for a, b in missing))
        if extra:
            errors.append("physical_reference.cell_bindings:extra_pairs:" + ",".join(f"{a}/{b}" for a, b in extra))
    if outcomes_artifact.get("cell_count") != len(cells):
        errors.append("physical_reference.outcomes_artifact.cell_count:mismatch")

    runtime = _mapping(value.get("runtime"))
    lock = _mapping(runtime.get("environment_lock"))
    if _string(lock.get("status")) != "exact_immutable" or not _is_sha256(lock.get("digest")):
        blockers.append("runtime_environment_lock_incomplete")
    feasibility = _mapping(runtime.get("zero_spend_feasibility"))
    feasibility_status = _string(feasibility.get("status"))
    if feasibility_status != "passed":
        blockers.append(
            "zero_spend_feasibility_not_passed:"
            + (feasibility_status or "missing")
        )
    if _number(feasibility.get("cost_usd")) != 0.0:
        errors.append("runtime.zero_spend_feasibility.cost_usd:must_be_zero")
    for key in ("cpu", "gpu", "storage", "expected_duration"):
        if not _mapping(runtime.get(key)):
            errors.append(f"runtime.{key}:missing")
    language_encoder = _mapping(runtime.get("language_encoder"))
    if not _is_sha256(language_encoder.get("archive_sha256")):
        errors.append("runtime.language_encoder.archive_sha256:invalid")
    if not isinstance(language_encoder.get("archive_size_bytes"), int):
        errors.append("runtime.language_encoder.archive_size_bytes:invalid")
    if _string(language_encoder.get("license")) != "Apache-2.0":
        errors.append("runtime.language_encoder.license:invalid")

    boundaries = _mapping(value.get("claim_boundaries"))
    for claim in sorted(_FORBIDDEN_TRUE_CLAIMS):
        if boundaries.get(claim) is not False:
            errors.append(f"claim_boundaries.{claim}:must_be_false")
    if boundaries.get("retrospective_harness_qualification") is not True:
        errors.append("claim_boundaries.retrospective_harness_qualification:must_be_true")

    supplied_digest = _string(value.get("manifest_digest"))
    expected_digest = canonical_digest(value, digest_field="manifest_digest")
    if supplied_digest and supplied_digest != expected_digest:
        errors.append("manifest_digest:mismatch")
    return errors, blockers


def build_public_reference_admission_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate *value* and return an admitted or blocked immutable receipt."""

    if not isinstance(value, Mapping):
        raise PublicReferenceAdmissionError(["manifest:not_mapping"])
    normalized = json.loads(json.dumps(value))
    errors, blockers = _validate_manifest(normalized)
    if errors:
        raise PublicReferenceAdmissionError(errors)

    manifest_digest = canonical_digest(normalized, digest_field="manifest_digest")
    candidates = _rows(normalized.get("candidates"))
    conditions = _rows(normalized.get("conditions"))
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-002",
        "gate_id": "public_reference_pinned",
        "reference_id": normalized["reference_id"],
        "source_identity_digest": normalized["source_identity_digest"],
        "manifest_digest": manifest_digest,
        "status": "blocked" if blockers else "admitted",
        "blockers": sorted(blockers),
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
        "candidate_bindings": [
            {
                "candidate_id": row["candidate_id"],
                "checkpoint_identity_digest": _candidate_checkpoint_digest(row),
            }
            for row in candidates
        ],
        "condition_ids": sorted(row["condition_id"] for row in conditions),
        "physical_reference_cell_count": len(
            _rows(_mapping(normalized.get("physical_reference")).get("cell_bindings"))
        ),
        "physical_outcomes_artifact_digest": _mapping(
            normalized.get("physical_reference")
        ).get("outcomes_artifact", {}).get("digest"),
        "physical_outcome_values_read": False,
        "exact_candidate_condition_join_available": True,
        "qualified_execution_ready": not blockers,
        "no_capture_or_reconstruction_feature_required": True,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-admitted", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = build_public_reference_admission_receipt(value)
    rendered = canonical_json(receipt) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if args.require_admitted and receipt["status"] != "admitted" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLAIM_CEILING",
    "MANIFEST_SCHEMA_VERSION",
    "PHASE_LABEL",
    "PublicReferenceAdmissionError",
    "build_public_reference_admission_receipt",
]
