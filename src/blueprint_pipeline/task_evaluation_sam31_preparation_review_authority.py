"""Task-bound SAM review permission derived from retained owner authorization.

ADP-009D/day-28: prepare before overlays exist without extending a historical
candidate's disclosure grant. The standing receipt is not review acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from . import public_scene_sam31_track_selection_review as review
from .task_evaluation_scene_configuration_submission_inputs import checked_file, sha

SCHEMA = "task_evaluation_sam31_standing_review_authority.v1"
TERMS = {
    "provider_id": "openai", "runtime": "openai_agents_sdk", "model": review.AI_REVIEW_MODEL,
    "program_id": "arm-decision-proof-v1", "declared_use_scope": review.AI_REVIEW_DECLARED_USE,
    "api_data_training_policy": "not_used_for_training_by_default_unless_opted_in",
    "training_opt_in": False, "default_abuse_monitoring_retention_max_days": 30,
    "responses_application_state_retention_max_days": 30,
    "image_manual_csam_review_retention_possible": True, "zero_data_retention_claimed": False,
    "response_store": False, "tracing_disabled": True, "trace_sensitive_data_included": False,
    "openai_api_data_use_terms_accepted": True, "openai_image_safety_review_terms_accepted": True,
    "frame_redistribution_authorized": False, "frame_publication_authorized": False,
    "derived_overlay_pngs_only": True, "raw_source_splat_or_dataset_bytes_included": False,
    "max_inference_spend_usd": review.AI_REVIEW_MAX_COST_USD,
    "private_derived_frame_disclosure_authorized": True,
    "agent_accepted_terms": False, "issued_by_agent": False,
}


class Sam31ReviewAuthorityError(ValueError):
    """Retained task or provider authority cannot admit this review."""


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise Sam31ReviewAuthorityError("sam31_review_authority_" + code)


def _path(value: str | Path, *, output: bool = False) -> Path:
    p = Path(value).expanduser()
    _require(p.is_absolute() and not any(x.is_symlink() for x in (p, *p.parents)), "path_invalid")
    _require(not p.exists() if output else p.is_file(), "output_exists" if output else "path_invalid")
    return p


def _read(value: str | Path) -> tuple[Path, dict[str, Any]]:
    p = _path(value)
    try:
        v = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise Sam31ReviewAuthorityError("sam31_review_authority_json_invalid") from exc
    _require(isinstance(v, dict), "json_invalid")
    return p, v


def _record(p: Path) -> dict[str, Any]:
    return {"path": str(p), "sha256": sha(p), "size_bytes": p.stat().st_size}


def _reopen(row: Any) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(row, dict), "reference_invalid")
    p = _path(row.get("path", ""))
    checked_file(p, {k: row.get(k) for k in ("path", "sha256", "size_bytes")})
    return _read(p)


def _derive(task_path: Path, terms_path: Path) -> dict[str, Any]:
    _, task = _read(task_path)
    _, terms = _read(terms_path)
    owner = task.get("human_authority")
    _require(isinstance(owner, dict), "task_owner_missing")
    _require(owner.get("accepted_by") == review.AI_REVIEW_ACCEPTED_BY
             and all(isinstance(owner.get(k), str) and owner[k].strip()
                     for k in ("accepted_on", "authority_reference")), "task_owner_invalid")
    _require(all(owner.get(k) is True for k in (
        "sam31_visual_review_authorized", "private_derived_frame_disclosure_authorized",
        "provider_retention_terms_accepted", "provider_training_terms_accepted"))
        and owner.get("provider_training_authorized") is False
        and not isinstance(owner.get("sam31_visual_review_maximum_cost_usd"), bool)
        and owner.get("sam31_visual_review_maximum_cost_usd") == review.AI_REVIEW_MAX_COST_USD,
        "task_scope_invalid")
    _require(terms.get("schema_version") == review.AI_RIGHTS_SCHEMA_VERSION
             and terms.get("attestation_digest") == canonical_digest(terms, digest_field="attestation_digest")
             and terms.get("status") == "accepted_for_private_derived_visual_review"
             and terms.get("accepted_by") == owner["accepted_by"]
             and all(terms.get(k) == v and type(terms.get(k)) is type(v)
                     for k, v in TERMS.items() if k != "max_inference_spend_usd")
             and not isinstance(terms.get("max_inference_spend_usd"), bool)
             and terms.get("max_inference_spend_usd") == review.AI_REVIEW_MAX_COST_USD
             and all(isinstance(terms.get(k), str) and terms[k].strip()
                     for k in ("accepted_on", "human_authority_reference")), "provider_terms_invalid")
    receipt = {
        "schema_version": SCHEMA, "status": "authorized_for_exact_task_pending_candidate",
        "task_request": _record(task_path), "provider_terms_evidence": _record(terms_path),
        "source_commit": task.get("expected_production_commit"),
        "task_identity": task.get("task_identity"), "scene_identity": task.get("scene_identity"),
        "publisher_scene_id": task.get("publisher_scene_id"),
        "accepted_by": owner["accepted_by"], "accepted_on": owner["accepted_on"],
        "human_authority_reference": owner["authority_reference"],
        "authority_source": "exact_task_request.human_authority",
        "historical_candidate_authority_reused": False,
        "provider_terms_evidence_use": "retained_provider_terms_only",
        "review_frame_count": review.AI_REVIEW_FRAME_COUNT, **TERMS,
        "human_review_required": False, "track_selection_accepted": False,
        "candidate_policy_queried": False, "evaluation_authorized": False,
        "authority_digest": "",
    }
    receipt["authority_digest"] = canonical_digest(receipt, digest_field="authority_digest")
    return receipt


def _write(path: str | Path, value: dict[str, Any]) -> Path:
    p = _path(path, output=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return p


def materialize_sam31_review_authority(*, task_request_path: str | Path,
                                      provider_terms_evidence_path: str | Path,
                                      output_path: str | Path) -> dict[str, Any]:
    """Record current task permission; never infer it from old candidate rights."""
    result = _derive(_path(task_request_path), _path(provider_terms_evidence_path))
    _write(output_path, result)
    return result


def validate_sam31_review_authority(authority_path: str | Path, *,
                                    task_request_path: str | Path | None = None) -> dict[str, Any]:
    _, value = _read(authority_path)
    _require(value.get("schema_version") == SCHEMA, "schema_invalid")
    task_path, _ = _reopen(value.get("task_request"))
    terms_path, _ = _reopen(value.get("provider_terms_evidence"))
    if task_request_path is not None:
        _require(_record(_path(task_request_path)) == value["task_request"], "task_mismatch")
    _require(value == _derive(task_path, terms_path), "receipt_invalid")
    return value


def resolve_sam31_review_rights(*, authority_path: str | Path, task_request_path: str | Path,
                                candidate_path: str | Path, output_path: str | Path,
                                completed_prefix_adoption_path: str | Path | None = None) -> Path:
    """Bind new overlays to the existing exact task permission before disclosure."""
    source, raw = _read(authority_path)
    if raw.get("schema_version") == review.AI_RIGHTS_SCHEMA_VERSION:
        # Compatibility is exact-candidate validation, never reinterpretation.
        return review.validate_sam31_ai_visual_review_rights(
            candidate_path=candidate_path, rights_attestation_path=source)[0]
    authority = validate_sam31_review_authority(source, task_request_path=task_request_path)
    candidate_file, candidate = review.load_validated_sam31_track_selection_review_candidate(candidate_path)
    bindings = candidate.get("selection_bindings", [])
    _require(len(bindings) == 1, "candidate_task_mismatch")
    freeze_path, freeze = _reopen(bindings[0].get("task_freeze"))
    from .public_scene_removal_selection import validate_removal_task_selection
    validate_removal_task_selection(freeze)
    _, scene = _reopen(freeze.get("scene_selection"))
    _require(freeze.get("scene_freeze_digest") == scene.get("scene_freeze_digest")
             and scene.get("scene_freeze_digest") == canonical_digest(scene, digest_field="scene_freeze_digest"),
             "candidate_task_mismatch")
    source_task = (scene.get("source_evidence") or {}).get("task_request")
    adoption_record = None
    if source_task != authority["task_request"]:
        _require(completed_prefix_adoption_path is not None, "candidate_task_mismatch")
        from .task_evaluation_sam31_prefix_adoption import validate_completed_prefix_adoption
        adopted = validate_completed_prefix_adoption(
            _path(completed_prefix_adoption_path), expected_source_commit=authority["source_commit"],
            approved_roots=(Path("/var/lib/blueprint"), Path("/opt/blueprint"), Path("/etc/blueprint")),
        )
        _, original_plan = _reopen(adopted["record"]["source_plan"])
        _require(adopted["record"]["current_host_inputs"]["task_request"] == authority["task_request"]
                 and adopted["artifacts"]["task_selection"] == _record(freeze_path)
                 and original_plan["host_inputs"]["task_request"] == source_task,
                 "candidate_task_mismatch")
        adoption_record = _record(_path(completed_prefix_adoption_path))
    _path(output_path, output=True)
    _path(Path(output_path).with_suffix(".derivation.json"), output=True)
    result = review.materialize_sam31_ai_visual_review_rights(
        candidate_path=candidate_file, accepted_by=authority["accepted_by"],
        accepted_on=authority["accepted_on"],
        human_authority_reference=authority["human_authority_reference"], output_path=output_path)
    # Separate immutable derivation receipt preserves compatibility with exact
    # candidate rights consumers without rewriting their freshly sealed bytes.
    derivation = {
        "schema_version": "task_evaluation_sam31_review_rights_derivation.v1",
        "standing_authority": _record(source), "authority_digest": authority["authority_digest"],
        "task_request": authority["task_request"], "provider_terms_evidence": authority["provider_terms_evidence"],
        "candidate": _record(candidate_file), "candidate_digest": candidate["candidate_digest"],
        "derived_rights": _record(Path(output_path)), "attestation_digest": result["attestation_digest"],
        "new_terms_acceptance": False, "historical_candidate_authority_reused": False,
        "receipt_digest": "",
    }
    if adoption_record is not None:
        derivation["completed_prefix_adoption"] = adoption_record
    derivation["receipt_digest"] = canonical_digest(derivation, digest_field="receipt_digest")
    _write(Path(output_path).with_suffix(".derivation.json"), derivation)
    return Path(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-request", required=True)
    parser.add_argument("--provider-terms-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_sam31_review_authority(task_request_path=args.task_request,
            provider_terms_evidence_path=args.provider_terms_evidence, output_path=args.output)
    except (OSError, ValueError) as exc:
        print(canonical_json({"status": "blocked", "blockers": [str(exc)]}))
        return 2
    print(canonical_json({"status": receipt["status"], "authority_digest": receipt["authority_digest"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
