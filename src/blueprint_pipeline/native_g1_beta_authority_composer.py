"""Controller-only composition of G1 authority after live task/consent checks.

Provider and historical runtimes consume the sealed pure authority validator;
they must not acquire the beta controller's allocator/provider import closure.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import (
    DP_AUXILIARY, INVENTORY_SHA256, PI_AUXILIARY, ROBOT_USD_SHA256, SCHEMA,
    SCOPE, SONIC, SOURCE_FIELDS, SOURCE_REVISION, reviewed_model_terms,
    validate_beta_authority,
)


def compose_beta_authority(*, candidate_id: str, scene_plan_digest: str,
                           source_packet_receipt_digest: str, task_success_contract_digest: str,
                           checkpoint_digest: str, current_source: Mapping[str, Any],
                           site_id: str, registration_digest: str) -> dict[str, Any]:
    """Compose only after reopening current trusted task and capture consent."""
    from .beta_policy_screening_worker import require_current_source
    require_current_source(current_source, site_id=site_id)
    row = {"schema_version": SCHEMA, **SCOPE, "candidate_id": candidate_id,
           "scene_plan_digest": scene_plan_digest, "task_success_contract_digest": task_success_contract_digest,
           "source_packet_receipt_digest": source_packet_receipt_digest, "checkpoint_digest": checkpoint_digest,
           "inventory_file_sha256": INVENTORY_SHA256, "source_revision": SOURCE_REVISION,
           "sonic": dict(SONIC), "robot_usd_sha256": ROBOT_USD_SHA256,
           "auxiliary_assets": [dict(PI_AUXILIARY if "_pi05_" in candidate_id else DP_AUXILIARY)],
           "model_review": reviewed_model_terms(candidate_id),
           "current_task_source": {key: current_source[key] for key in SOURCE_FIELDS},
           "site_id": site_id, "registration_digest": registration_digest}
    row["authority_digest"] = canonical_digest(row, digest_field="authority_digest")
    return validate_beta_authority(row, candidate_id=candidate_id, scene_plan_digest=scene_plan_digest,
                                   current_source=current_source, task_success_contract_digest=task_success_contract_digest)
