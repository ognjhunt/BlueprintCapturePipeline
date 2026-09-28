"""Candidate-specific, runtime-verifiable rights authority for ADP-009D policies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest
try:  # flat provider-bundle layout
    from adp009d_groot_worker_identity import (
        expected_checkpoint_content_binding,
        expected_checkpoint_interface_binding,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_groot_worker_identity import (
        expected_checkpoint_content_binding,
        expected_checkpoint_interface_binding,
    )


SCHEMA_VERSION = "adp009d_candidate_policy_rights.v1"
RESULT_FILENAME = "adp009d_candidate_policy_rights.v1.json"
FROZEN_CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")
EXPECTED_SOURCE_READINESS_DIGEST = (
    "sha256:c3f76892f80514ef81ddbb48b14ebf3c9e39cbbfaf82d516dbcb70f4d8989ffc"
)
EXPECTED_SCENARIO_SUITE_DIGEST = (
    "sha256:5adcfd1b9c96da80aff16d49c004591db0347442c4298a763e7d916a10b40e34"
)
EXPECTED_TASK_FREEZE_DIGEST = (
    "sha256:fa3f422fb595badb3bedd11c56e4732a088bfff112b80b1f65d0c53d1927baa3"
)


class CandidatePolicyRightsError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def validate_candidate_policy_rights(
    value: Mapping[str, Any],
    *,
    candidate_id: str,
    policy_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate rights without importing a candidate package or contacting it."""

    payload = json.loads(json.dumps(value, allow_nan=False))
    errors: list[str] = []
    source = _mapping(payload.get("source_identity"))
    checkpoint = _mapping(payload.get("checkpoint_identity"))
    rights = _mapping(payload.get("rights"))
    interface = _mapping(payload.get("interface_identity"))
    robot = _mapping(payload.get("robot_binding"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("candidate_policy_rights_schema_invalid")
    if candidate_id not in FROZEN_CANDIDATE_IDS or payload.get(
        "candidate_id"
    ) != candidate_id:
        errors.append("candidate_policy_rights_candidate_invalid")
    if (
        payload.get("program_id") != "arm-decision-proof-v1"
        or payload.get("scene_id") != "840920"
        or payload.get("task_id") != "task_a_washer_door_open"
    ):
        errors.append("candidate_policy_rights_task_binding_invalid")
    if (
        robot.get("embodiment_id") != "franka"
        or robot.get("runtime_robot_id") != "franka_panda"
        or robot.get("alias_authority") != "native_task_arena_scene_plan.v1"
        or not _digest(robot.get("scene_plan_digest"))
    ):
        errors.append("candidate_policy_rights_robot_binding_invalid")
    if payload.get("readiness_verdict") != "READY_WAITING_ONLY_FOR_CONTROLS":
        errors.append("candidate_policy_rights_readiness_verdict_invalid")
    if (
        payload.get("source_readiness_digest")
        != EXPECTED_SOURCE_READINESS_DIGEST
        or payload.get("scenario_suite_digest")
        != EXPECTED_SCENARIO_SUITE_DIGEST
        or payload.get("task_freeze_digest") != EXPECTED_TASK_FREEZE_DIGEST
    ):
        errors.append("candidate_policy_rights_upstream_authority_invalid")
    if (
        payload.get("claim_ceiling") != "development_only"
        or payload.get("outcome_blind") is not True
        or payload.get("learned_policy_outcomes_observed") is not False
        or payload.get("provider_execution_performed") is not False
        or payload.get("raw_secret_values_recorded") is not False
    ):
        errors.append("candidate_policy_rights_claim_boundary_invalid")
    for field in ("policy_input_schema", "policy_output_schema", "runtime_dependencies"):
        if not isinstance(interface.get(field), Mapping) or not interface.get(field):
            errors.append("candidate_policy_rights_interface_invalid")
    if not str(interface.get("action_adapter") or "").strip():
        errors.append("candidate_policy_rights_interface_invalid")
    if (
        rights.get("rights_ready") is not True
        or rights.get("checkpoint_ready") is not True
        or rights.get("provider_retrieval_path_ready") is not True
        or rights.get("missing_secrets_or_gated_access") != []
        or not str(rights.get("provider_use_status") or "").strip()
        or not str(rights.get("redistribution_status") or "").strip()
    ):
        errors.append("candidate_policy_rights_not_ready")

    if candidate_id == "pi05_droid":
        provenance = _mapping(rights.get("rights_provenance"))
        expected_provenance = {
            "publisher_model_release",
            "apache_license",
            "gemma_terms",
            "exact_checkpoint_config",
        }
        if (
            source.get("repository")
            != "https://github.com/Physical-Intelligence/openpi"
            or source.get("revision") != policy_spec.get("openpi_revision")
            or checkpoint.get("repository") != policy_spec.get("checkpoint_uri")
            or checkpoint.get("inventory_digest")
            != "sha256:" + str(policy_spec.get("checkpoint_inventory_sha256") or "")
            or checkpoint.get("object_count")
            != policy_spec.get("checkpoint_object_count")
            or checkpoint.get("total_bytes")
            != policy_spec.get("checkpoint_size_bytes")
        ):
            errors.append("candidate_policy_rights_pi05_identity_mismatch")
        if (
            rights.get("checkpoint_specific_terms_bound") is not True
            or set(provenance) != expected_provenance
            or any(
                str(source.get("revision") or "") not in str(uri)
                for uri in provenance.values()
            )
        ):
            errors.append("candidate_policy_rights_pi05_terms_invalid")
    elif candidate_id == "groot_n17_droid":
        gated = _mapping(rights.get("gated_backbone"))
        expected_content = expected_checkpoint_content_binding()
        expected_interface = expected_checkpoint_interface_binding()
        policy_input = _mapping(interface.get("policy_input_schema"))
        if (
            source.get("repository") != "https://github.com/NVIDIA/Isaac-GR00T"
            or source.get("revision") != policy_spec.get("groot_source_revision")
            or checkpoint.get("revision") != policy_spec.get("checkpoint_revision")
            or checkpoint.get("repository")
            != "https://huggingface.co/nvidia/GR00T-N1.7-DROID"
            or checkpoint.get("license") != "NVIDIA Open Model License"
            or checkpoint.get("inventory_digest")
            != expected_content["inventory_digest"]
            or checkpoint.get("file_count") != expected_content["file_count"]
            or checkpoint.get("total_bytes") != expected_content["total_bytes"]
            or checkpoint.get("content_manifest")
            != expected_content["file_manifest"]
            or checkpoint.get("content_manifest_digest")
            != expected_content["file_manifest_digest"]
        ):
            errors.append("candidate_policy_rights_groot_identity_mismatch")
        if policy_input.get("frame_history") != expected_interface[
            "video_delta_indices"
        ]:
            errors.append("candidate_policy_rights_groot_interface_mismatch")
        if (
            gated.get("access_probe_status") != "authorized"
            or not _digest(gated.get("access_probe_receipt_digest"))
            or gated.get("secret_material_recorded") is not False
        ):
            errors.append("candidate_policy_rights_groot_gated_access_invalid")
    if payload.get("rights_receipt_digest") != canonical_digest(
        payload, digest_field="rights_receipt_digest"
    ):
        errors.append("candidate_policy_rights_digest_invalid")
    if errors:
        raise CandidatePolicyRightsError(errors)
    return payload


def build_candidate_policy_rights(
    readiness: Mapping[str, Any],
    *,
    candidate_id: str,
    policy_spec: Mapping[str, Any],
    runtime_robot_id: str,
    scene_plan_digest: str,
) -> dict[str, Any]:
    """Project one validated readiness seal into a runtime-safe binding."""

    matches = [
        row
        for row in readiness.get("candidates", [])
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise CandidatePolicyRightsError(
            ["candidate_policy_rights_source_candidate_missing"]
        )
    candidate = dict(matches[0])
    checkpoint = _mapping(candidate.get("checkpoint"))
    rights = {
        "rights_ready": candidate.get("rights_ready"),
        "checkpoint_ready": checkpoint.get("checkpoint_ready"),
        "provider_retrieval_path_ready": checkpoint.get(
            "provider_retrieval_path_ready"
        ),
        "missing_secrets_or_gated_access": checkpoint.get(
            "missing_secrets_or_gated_access"
        ),
        "provider_use_status": checkpoint.get("provider_use_status"),
        "redistribution_status": checkpoint.get("redistribution_status"),
    }
    for field in (
        "checkpoint_specific_terms_bound",
        "rights_provenance",
        "gated_backbone",
    ):
        if field in checkpoint:
            rights[field] = checkpoint[field]
    checkpoint_identity = {
        key: checkpoint[key]
        for key in (
            "repository",
            "revision",
            "inventory_digest",
            "object_count",
            "file_count",
            "total_bytes",
            "license",
        )
        if key in checkpoint
    }
    if candidate_id == "groot_n17_droid":
        expected_content = expected_checkpoint_content_binding()
        expected_weight_digests = [
            "sha256:" + str(row["digest"])
            for row in expected_content["file_manifest"]
            if row["digest_algorithm"] == "sha256"
            and str(row["path"]).endswith(".safetensors")
        ]
        if checkpoint.get("weight_sha256") != expected_weight_digests:
            raise CandidatePolicyRightsError(
                ["candidate_policy_rights_groot_source_weight_identity_invalid"]
            )
        checkpoint_identity.update(
            {
                "content_manifest": expected_content["file_manifest"],
                "content_manifest_digest": expected_content[
                    "file_manifest_digest"
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": readiness.get("program_id"),
        "scene_id": readiness.get("scene_id"),
        "task_id": readiness.get("task_id"),
        "robot_binding": {
            "embodiment_id": readiness.get("robot_id"),
            "runtime_robot_id": runtime_robot_id,
            "alias_authority": "native_task_arena_scene_plan.v1",
            "scene_plan_digest": scene_plan_digest,
        },
        "candidate_id": candidate_id,
        "claim_ceiling": readiness.get("claim_ceiling"),
        "outcome_blind": readiness.get("outcome_blind"),
        "learned_policy_outcomes_observed": readiness.get(
            "learned_policy_outcomes_observed"
        ),
        "provider_execution_performed": readiness.get(
            "provider_execution_performed"
        ),
        "readiness_verdict": readiness.get("verdict"),
        "source_readiness_digest": readiness.get("readiness_digest"),
        "scenario_suite_digest": readiness.get("scenario_suite_digest"),
        "task_freeze_digest": readiness.get("task_freeze_digest"),
        "source_identity": dict(candidate.get("source") or {}),
        "checkpoint_identity": checkpoint_identity,
        "interface_identity": {
            "action_adapter": candidate.get("action_adapter"),
            "policy_input_schema": dict(candidate.get("policy_input_schema") or {}),
            "policy_output_schema": dict(candidate.get("policy_output_schema") or {}),
            "runtime_dependencies": dict(candidate.get("runtime_dependencies") or {}),
        },
        "rights": rights,
        "raw_secret_values_recorded": False,
    }
    payload["rights_receipt_digest"] = canonical_digest(
        payload, digest_field="rights_receipt_digest"
    )
    return validate_candidate_policy_rights(
        payload, candidate_id=candidate_id, policy_spec=policy_spec
    )


def validate_candidate_policy_rights_authorities(
    value: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    scenario_suite: Mapping[str, Any],
    candidate_id: str,
    policy_spec: Mapping[str, Any],
    runtime_robot_id: str,
    scene_plan_digest: str,
) -> dict[str, Any]:
    """Bind a candidate receipt to the committed full authority records.

    The narrow rights receipt is not an authority by itself.  Both the bundle
    builder and provider worker call this function against separately
    digest-bound readiness and scenario-suite bytes.  Re-digesting a forged
    execution spec therefore cannot manufacture checkpoint or license rights.
    """

    errors: list[str] = []
    if (
        readiness.get("schema_version")
        != "adp009d_scene_policy_readiness.v1"
        or readiness.get("readiness_digest")
        != canonical_digest(readiness, digest_field="readiness_digest")
        or readiness.get("readiness_digest")
        != EXPECTED_SOURCE_READINESS_DIGEST
    ):
        errors.append("candidate_policy_rights_readiness_authority_invalid")
    if (
        scenario_suite.get("schema_version")
        != "third_scene_task_scenario_suite.v1"
        or scenario_suite.get("suite_digest")
        != canonical_digest(scenario_suite, digest_field="suite_digest")
        or scenario_suite.get("suite_digest")
        != EXPECTED_SCENARIO_SUITE_DIGEST
        or scenario_suite.get("task_freeze_digest")
        != EXPECTED_TASK_FREEZE_DIGEST
        or readiness.get("scenario_suite_digest")
        != scenario_suite.get("suite_digest")
        or readiness.get("task_freeze_digest")
        != scenario_suite.get("task_freeze_digest")
    ):
        errors.append("candidate_policy_rights_scenario_authority_invalid")
    if errors:
        raise CandidatePolicyRightsError(errors)
    expected = build_candidate_policy_rights(
        readiness,
        candidate_id=candidate_id,
        policy_spec=policy_spec,
        runtime_robot_id=runtime_robot_id,
        scene_plan_digest=scene_plan_digest,
    )
    validated = validate_candidate_policy_rights(
        value, candidate_id=candidate_id, policy_spec=policy_spec
    )
    if validated != expected:
        raise CandidatePolicyRightsError(
            ["candidate_policy_rights_authoritative_projection_mismatch"]
        )
    return validated


def materialize_candidate_policy_rights(
    *,
    readiness_path: str | Path,
    scenario_suite_path: str | Path,
    candidate_id: str,
    policy_spec: Mapping[str, Any],
    runtime_robot_id: str,
    scene_plan_digest: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive a narrow paid-runtime authority from the full readiness seal."""

    from .adp009d_scene_policy_readiness import load_scene_policy_readiness

    readiness = load_scene_policy_readiness(
        readiness_path, scenario_suite_path=scenario_suite_path
    )
    validated = build_candidate_policy_rights(
        readiness,
        candidate_id=candidate_id,
        policy_spec=policy_spec,
        runtime_robot_id=runtime_robot_id,
        scene_plan_digest=scene_plan_digest,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise CandidatePolicyRightsError(
            ["candidate_policy_rights_output_exists"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validated


__all__ = [
    "CandidatePolicyRightsError",
    "EXPECTED_SCENARIO_SUITE_DIGEST",
    "EXPECTED_SOURCE_READINESS_DIGEST",
    "EXPECTED_TASK_FREEZE_DIGEST",
    "FROZEN_CANDIDATE_IDS",
    "RESULT_FILENAME",
    "SCHEMA_VERSION",
    "build_candidate_policy_rights",
    "materialize_candidate_policy_rights",
    "validate_candidate_policy_rights",
    "validate_candidate_policy_rights_authorities",
]


def main(argv: list[str] | None = None) -> int:
    """Materialize candidate rights from explicit authoritative inputs."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-path", required=True)
    parser.add_argument("--scenario-suite-path", required=True)
    parser.add_argument("--candidate-id", required=True, choices=FROZEN_CANDIDATE_IDS)
    parser.add_argument("--policy-spec", required=True)
    parser.add_argument("--runtime-robot-id", required=True)
    parser.add_argument("--scene-plan-digest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        policy_spec = json.loads(
            Path(args.policy_spec).expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(policy_spec, Mapping):
            raise ValueError("candidate_policy_rights_policy_spec_invalid")
        result = materialize_candidate_policy_rights(
            readiness_path=args.readiness_path,
            scenario_suite_path=args.scenario_suite_path,
            candidate_id=args.candidate_id,
            policy_spec=policy_spec,
            runtime_robot_id=args.runtime_robot_id,
            scene_plan_digest=args.scene_plan_digest,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "sealed",
                "candidate_id": result["candidate_id"],
                "rights_receipt_digest": result["rights_receipt_digest"],
                "output": args.output,
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
