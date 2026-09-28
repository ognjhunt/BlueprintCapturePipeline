"""Operator-owned frozen configuration for controlled native execution."""
from __future__ import annotations
import json
import hashlib
from typing import Any, Mapping
from .decision_evidence_contracts import canonical_digest
from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2

def validate_native_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(value, allow_nan=False))
    validate_company_policy_container_contract_v2(row["contract"])
    if (row.get("schema_version") != "blueprint.controlled_native_configuration.v1"
            or row.get("evidence_scope") != "development_only"
            or row.get("configuration_digest") != canonical_digest(row, digest_field="configuration_digest")
            or not row.get("scene_plan_digest") or not row.get("scenario_id") or not row.get("native_cell_id")
            or not isinstance(row.get("allowed_origins"), list)
            or not isinstance(row.get("camera_roles"), dict)
            or set(row["camera_roles"]) != {camera["name"] for camera in row["contract"]["observation_schema"]["cameras"]}
            or set(row["camera_roles"].values()) != {"external", "wrist"}
            or type(row.get("max_queries")) is not int or not 1 <= row["max_queries"] <= 128
            or not 0 < row.get("deadline_seconds", 0) <= 3600
            or not 0 < row.get("max_joint_delta_rad", 0) <= 0.1
            or not 0 < row.get("max_joint_setpoint_lead_rad", 0) <= 0.1):
        raise ValueError("controlled_native_configuration_invalid")
    return row



def canonical_request_digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
