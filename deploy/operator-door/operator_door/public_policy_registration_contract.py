"""Stdlib-only fixed public-policy installation contract.

The operator-door installer copies this module into its root-owned installation.
Privileged code never imports the active pipeline checkout; the unprivileged
proposal builder shares this fixed source contract in the release checkout.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA = "openwam_piper_policy_install_request.v1"
CANDIDATE_ID = "openwam_alpha_piper_bimanual"
TEAM_ID = "blueprint-public-models"
CHECKPOINT_ID = "openwam-alpha-piper-12bfca6f"
PROFILE_ID = "openwam_alpha_real_piper.v1"
EMBODIMENT_ID = "agilex_piper_bimanual"
YAM_SCHEMA = "molmoact2_yam_policy_install_request.v1"
YAM_CANDIDATE_ID = "molmoact2_bimanual_yam"
YAM_CHECKPOINT_ID = "molmoact2-bimanual-yam-8dcbed66"
YAM_PROFILE_ID = "molmoact2_bimanual_yam.v1"
YAM_EMBODIMENT_ID = "i2rt_yam_bimanual"
FRANKA_SCHEMA = "openwam_franka_policy_install_request.v1"
FRANKA_CANDIDATE_ID = "openwam_alpha_franka"
FRANKA_CHECKPOINT_ID = "openwam-alpha-franka-e75522eb"
FRANKA_PROFILE_ID = "openwam_alpha_real_franka.v1"
FRANKA_EMBODIMENT_ID = "franka_research_3_robotiq_2f85_sim"
IDENTITIES = {
    FRANKA_SCHEMA: {"candidate_id": FRANKA_CANDIDATE_ID, "team_id": TEAM_ID, "checkpoint_id": FRANKA_CHECKPOINT_ID,
        "policy_profile": FRANKA_PROFILE_ID, "embodiment_id": FRANKA_EMBODIMENT_ID,
        "operation": "install-public-franka-policy", "profile_file": "openwam_franka_registration_profile.json",
        "proposal_schema": "openwam_franka_policy_install_proposal.v1",
        "receipt_schema": "openwam_franka_policy_install_receipt.v1"},
    SCHEMA: {"candidate_id": CANDIDATE_ID, "team_id": TEAM_ID, "checkpoint_id": CHECKPOINT_ID,
        "policy_profile": PROFILE_ID, "embodiment_id": EMBODIMENT_ID,
        "operation": "install-public-piper-policy", "profile_file": "openwam_piper_registration_profile.json",
        "proposal_schema": "openwam_piper_policy_install_proposal.v1",
        "receipt_schema": "openwam_piper_policy_install_receipt.v1"},
    YAM_SCHEMA: {"candidate_id": YAM_CANDIDATE_ID, "team_id": TEAM_ID, "checkpoint_id": YAM_CHECKPOINT_ID,
        "policy_profile": YAM_PROFILE_ID, "embodiment_id": YAM_EMBODIMENT_ID,
        "operation": "install-public-yam-policy", "profile_file": "molmoact2_yam_registration_profile.json",
        "proposal_schema": "molmoact2_yam_policy_install_proposal.v1",
        "receipt_schema": "molmoact2_yam_policy_install_receipt.v1"},
}
STATE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/public-policy-registrations")
MODEL_ROOT = Path("/var/lib/blueprint/robot-models")
REGISTRY_PATH = Path("/etc/blueprint/team-evaluation-registry.json")
FIELDS = {"schema_version", "installation_id", "release_commit", "preparation_authorization_id",
          "preparation_publication_digest", "candidate_id", "team_id", "checkpoint_id", "policy_profile",
          "expected_registry_digest", "authorization_reference", "request_digest"}


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise ValueError("openwam_policy_install_" + reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any, field: str = "") -> str:
    return "sha256:" + hashlib.sha256(canonical({k: v for k, v in value.items() if k != field} if field else value)).hexdigest()


def decode(raw: bytes | str) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate_json_key")
            value[key] = item
        return value
    def nonfinite(_value):
        raise ValueError("openwam_policy_install_nonfinite_json")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    require(isinstance(value, dict), "record_invalid")
    return value


def request_identity(value: dict) -> dict:
    schema = value.get("schema_version")
    require(isinstance(schema, str) and schema in IDENTITIES, "request_schema_invalid")
    return dict(IDENTITIES[schema])


def validate_request(value: Any, *, operation: str | None = None) -> dict:
    require(isinstance(value, dict) and set(value) == FIELDS,
            "request_schema_invalid")
    identity = request_identity(value)
    require(operation is None or operation == identity["operation"], "request_operation_mismatch")
    for key in ("installation_id", "preparation_authorization_id"):
        require(isinstance(value[key], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value[key]), key + "_invalid")
    require(isinstance(value["release_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", value["release_commit"]),
            "release_commit_invalid")
    for key in ("preparation_publication_digest", "expected_registry_digest"):
        require(isinstance(value[key], str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value[key]), key + "_invalid")
    require(all(value[key] == identity[key] for key in ("candidate_id", "team_id", "checkpoint_id", "policy_profile")),
            "fixed_profile_mismatch")
    reference = value["authorization_reference"]
    require(isinstance(reference, str) and 1 <= len(reference) <= 300 and reference.isprintable(),
            "authorization_reference_invalid")
    require(value["request_digest"] == digest(value, "request_digest"), "request_digest_invalid")
    return dict(value)


def safe_path(path: Path) -> Path:
    require(path.is_absolute() and not any(part.is_symlink() for part in (path, *path.parents)), "symlink_path_refused")
    return path


def read(path: Path, *, limit: int = 8 * 1024**2) -> dict:
    safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit and not info.st_mode & 0o022,
                "record_file_invalid")
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "record_size_invalid")
    return decode(raw)


def file_record(path: Path) -> dict:
    safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 4 * 1024**3, "bound_file_invalid")
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(block)
        sha = "sha256:" + hasher.hexdigest()
    return {"path": str(path), "sha256": sha, "size_bytes": info.st_size}


def load_profile(request: dict, profile_root: Path) -> dict:
    identity = request_identity(request)
    profile = read(profile_root / identity["profile_file"])
    require(all(profile[key] == identity[key] for key in ("candidate_id", "team_id", "checkpoint_id", "policy_profile")),
            "fixed_profile_changed")
    if request["schema_version"] in {SCHEMA, FRANKA_SCHEMA}:
        card = "openwam_franka_model_card.md" if request["schema_version"] == FRANKA_SCHEMA else "openwam_piper_model_card.md"
        require(file_record(profile_root / card)["sha256"] == profile["model_card_sha256"],
                "model_license_evidence_changed")
    else:
        # These names are fixed installed evidence, not caller or proposal paths.
        for name in ("molmoact2_yam_publisher_readme.md", "molmoact2_yam_publisher_license.txt"):
            actual = file_record(profile_root / name)
            evidence = profile["license_evidence_files"][name]
            require(all(actual[key] == evidence[key] for key in ("sha256", "size_bytes")),
                    "model_license_evidence_changed")
    return profile


def expected_manifest(original: dict, profile: dict | None = None) -> dict:
    value = json.loads(canonical(original))
    body = profile["policy_spec_template"]["embodiment_id"] if profile else EMBODIMENT_ID
    require(value.get("embodiment_id") == body, "body_mismatch")
    if body == EMBODIMENT_ID:
        value.update(control_hz=60, action_chunk_rows=1)
    elif body == FRANKA_EMBODIMENT_ID:
        value.update(control_hz=60, action_chunk_rows=1)
        require(isinstance(value.get("cameras"), list)
                and [row["name"] for row in value["cameras"]] == ["overhead", "arm_wrist"], "source_cameras_changed")
        for camera in value["cameras"]:
            camera.update(width=640, height=480)
    else:
        require(body == YAM_EMBODIMENT_ID, "body_mismatch")
        value.update(control_hz=60, action_chunk_rows=30)
        cameras = profile["camera_overrides"]
        require(isinstance(value.get("cameras"), list)
                and [row["name"] for row in value["cameras"]] == ["overhead", "left_wrist", "right_wrist"],
                "source_cameras_changed")
        for camera in value["cameras"]:
            camera.update(cameras[camera["name"]])
    value["embodiment_digest"] = digest(value, "embodiment_digest")
    return value


def expected_policy(manifest: dict, profile: dict) -> dict:
    value = json.loads(canonical(profile["policy_spec_template"]))
    value["binding"]["configuration_digest"] = manifest["embodiment_digest"]
    return value


def rights_record(request: dict, profile: dict) -> dict:
    identity = request_identity(request)
    value = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
        **{key: identity[key] for key in ("candidate_id", "team_id", "checkpoint_id")},
        "license_to_run_granted": True, "policy_self_grading_permitted": False,
        "official_ranking_permitted": False, "secret_material_recorded": False,
        "adp_qualified_evidence_permitted": False, "license_id": "Apache-2.0",
        "model_identity_digest":
            profile["policy_spec_template"]["model_identity_digest"],
        "authorization_basis": "user_instruction_to_run_public_models",
        "authorization_reference": request["authorization_reference"],
        "private_vendor_authorization_claimed": False, "physical_validation_claimed": False}
    if request["schema_version"] in {SCHEMA, FRANKA_SCHEMA}:
        value.update(license_basis="publisher_pinned_public_model_card", model_card_url=profile["model_card_url"],
                     model_card_sha256=profile["model_card_sha256"])
    else:
        value.update(license_basis="publisher_model_license_statement_in_pinned_repository_readme",
            license_evidence_url=profile["license_evidence_url"], license_evidence_files=profile["license_evidence_files"],
            license_declaration=profile["license_declaration"], publisher_intended_use=profile["publisher_intended_use"],
            hf_model_card_license_field=None)
    return {**value, "rights_digest": digest(value)}


def retained_body(request: dict, *, model_root: Path = MODEL_ROOT) -> tuple[Path, dict, dict]:
    embodiment_id = request_identity(request)["embodiment_id"]
    publication = read(model_root / "batches" / (request["preparation_authorization_id"] + ".json"))
    require(publication.get("publication_digest") == request["preparation_publication_digest"]
            == digest(publication, "publication_digest") and publication.get("status") == "completed"
            and publication.get("claim_ceiling") == "robot_asset_import_only"
            and publication.get("candidate_policy_queried") is False and publication.get("task_executed") is False,
            "preparation_publication_changed")
    matches = [row for row in publication.get("results", []) if row.get("embodiment_id") == embodiment_id]
    require(len(matches) == 1 and matches[0].get("status") == "retained", "prepared_body_unavailable")
    row = matches[0]
    prepared = Path(row.get("directory", ""))
    require(prepared.parent.parent == model_root / embodiment_id and prepared.name == "prepared"
            and re.fullmatch(r"[0-9a-f]{64}", prepared.parent.name), "prepared_body_path_invalid")
    body = safe_path(prepared.parent)
    binding_path = body / "binding-inputs.json"
    require(row.get("binding_inputs_path") == str(binding_path)
            and row.get("binding_inputs_sha256") == file_record(binding_path)["sha256"], "prepared_binding_changed")
    binding = read(binding_path)
    require(binding.get("schema_version") == "embodiment_runtime_binding_inputs.v1"
            and binding["robot_embodiment"]["embodiment_id"] == embodiment_id, "prepared_binding_invalid")
    imported = read(prepared / "asset-import.json")
    model_relative = Path(imported["model"]["path"])
    require(not model_relative.is_absolute() and ".." not in model_relative.parts, "model_path_invalid")
    model_path = prepared / model_relative
    model = file_record(model_path)
    require(all(model[key] == imported["model"][key] == binding["model_record"][key]
                for key in ("sha256", "size_bytes")), "prepared_model_changed")
    return body, binding, model
