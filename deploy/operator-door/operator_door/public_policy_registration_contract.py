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
TRLC_SCHEMA = "trlc_pi05_policy_install_request.v1"
TRLC_CANDIDATE_ID = "pi05_trlc_rabc_10466a13"
TRLC_CHECKPOINT_ID = "pi05-trlc-rabc-10466a13"
TRLC_PROFILE_ID = "pi05_trlc_rabc_joint14.v1"
TRLC_EMBODIMENT_ID = "trlc_dk1_bimanual"
TRLC_TOKENIZER_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/g1-841757-development/pi-tokenizer-f1eaa13")
TRLC_NOTICE_FILES = ("APACHE-2.0.txt", "NOTICE.txt", "base-model-card.md", "community-model-card.md",
                     "gemma-prohibited-use.txt", "gemma-terms.txt", "provenance.json")
TRLC_TOKENIZER_FILES = {"added_tokens.json", "preprocessor_config.json", "special_tokens_map.json",
                        "tokenizer.json", "tokenizer.model", "tokenizer_config.json"}
SCRIPTED_SCHEMA = "scripted_robot_baseline_install_request.v1"
SCRIPTED_PROFILE_ID = "blueprint_scripted_pick_place.v1"
SCRIPTED_TEAM_ID = "blueprint-owned-baselines"
SCRIPTED_BODIES = ("i2rt_yam_bimanual", "agilex_piper_bimanual", "franka_research_3_bimanual",
    "universal_robots_ur5e_bimanual", "kinova_gen3", "kinova_gen3_bimanual", "ufactory_xarm7",
    "ufactory_xarm7_bimanual", "almond_axol", "trlc_dk1_bimanual",
    "universal_robots_ur5e_robotiq_2f85_sim", "universal_robots_ur10e_robotiq_2f85_sim",
    "franka_research_3_robotiq_2f85_sim",
    "yaskawa_gp12_robotiq_2f85_sim", "yaskawa_gp20_robotiq_2f85_sim",
    "fanuc_crx_10ia_robotiq_2f85_sim",
    "fanuc_crx_10ia_l_robotiq_2f85_sim",
    "fanuc_m710ic_45m_robotiq_2f85_sim",
    "abb_gofa_crb15000_5_95_robotiq_2f85_sim",
    "abb_gofa_crb15000_12_127_robotiq_2f85_sim")
IDENTITIES = {
    TRLC_SCHEMA: {"candidate_id": TRLC_CANDIDATE_ID, "team_id": TEAM_ID, "checkpoint_id": TRLC_CHECKPOINT_ID,
        "policy_profile": TRLC_PROFILE_ID, "embodiment_id": TRLC_EMBODIMENT_ID,
        "operation": "install-public-trlc-policy", "profile_file": "trlc_pi05_registration_profile.json",
        "proposal_schema": "trlc_pi05_policy_install_proposal.v1",
        "receipt_schema": "trlc_pi05_policy_install_receipt.v1"},
    FRANKA_SCHEMA: {"candidate_id": FRANKA_CANDIDATE_ID, "team_id": TEAM_ID, "checkpoint_id": FRANKA_CHECKPOINT_ID,
        "policy_profile": FRANKA_PROFILE_ID, "embodiment_id": FRANKA_EMBODIMENT_ID,
        "operation": "install-public-franka-policy", "profile_file": "openwam_franka_registration_profile.json",
        "proposal_schema": "openwam_franka_policy_install_proposal.v1",
        "receipt_schema": "openwam_franka_policy_install_receipt.v1"},
    SCRIPTED_SCHEMA: {"team_id": SCRIPTED_TEAM_ID, "policy_profile": SCRIPTED_PROFILE_ID,
        "operation": "install-scripted-robot-baselines",
        "profile_file": "scripted_robot_baseline_registration_profile.json",
        "proposal_schema": "scripted_robot_baseline_install_proposal.v1",
        "receipt_schema": "scripted_robot_baseline_install_receipt.v1"},
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
    require(isinstance(value, dict), "request_schema_invalid")
    fields = FIELDS - {"candidate_id", "checkpoint_id"} if value.get("schema_version") == SCRIPTED_SCHEMA else FIELDS
    require(set(value) == fields, "request_schema_invalid")
    identity = request_identity(value)
    require(operation is None or operation == identity["operation"], "request_operation_mismatch")
    for key in ("installation_id", "preparation_authorization_id"):
        require(isinstance(value[key], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value[key]), key + "_invalid")
    require(isinstance(value["release_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", value["release_commit"]),
            "release_commit_invalid")
    for key in ("preparation_publication_digest", "expected_registry_digest"):
        require(isinstance(value[key], str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value[key]), key + "_invalid")
    fixed = ("team_id", "policy_profile") if value["schema_version"] == SCRIPTED_SCHEMA else ("candidate_id", "team_id", "checkpoint_id", "policy_profile")
    require(all(value[key] == identity[key] for key in fixed),
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


def load_profile(request: dict, profile_root: Path, *, license_root: Path | None = None) -> dict:
    identity = request_identity(request)
    profile = read(profile_root / identity["profile_file"])
    require(all(profile[key] == identity[key] for key in ("candidate_id", "team_id", "checkpoint_id", "policy_profile")),
            "fixed_profile_changed")
    if request["schema_version"] == TRLC_SCHEMA:
        require(set(profile["license_evidence_files"]) == set(TRLC_NOTICE_FILES), "trlc_license_inventory_changed")
        for name in TRLC_NOTICE_FILES:
            actual = file_record((license_root or profile_root / "trlc_pi05_notices") / name)
            require(all(actual[key] == profile["license_evidence_files"][name][key]
                        for key in ("sha256", "size_bytes")), "model_license_evidence_changed")
        inventory_path = profile_root / "trlc_pi05_tokenizer_inventory.v1.json"
        require(file_record(inventory_path)["sha256"] == profile["tokenizer_inventory_sha256"]
                and read(inventory_path) == profile["tokenizer_inventory"], "trlc_tokenizer_inventory_changed")
    elif request["schema_version"] in {SCHEMA, FRANKA_SCHEMA}:
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


def trlc_tokenizer_record(profile: dict) -> dict:
    """Verify only fixed retained tokenizer bytes; no network, credentials or old task authority."""
    root = safe_path(TRLC_TOKENIZER_ROOT)
    require(root.is_dir(), "trlc_offline_tokenizer_unavailable")
    inventory = profile["tokenizer_inventory"]
    rows = inventory["files"]
    require(inventory.get("source_repository") == "https://huggingface.co/google/paligemma-3b-pt-224"
            and inventory.get("source_revision") == "35e4f46485b4d07967e7e9935bc3786aad50687c"
            and len(rows) == 6 and {row.get("path") for row in rows} == TRLC_TOKENIZER_FILES,
            "trlc_tokenizer_inventory_changed")
    files = []
    for row in rows:
        path = root / row["path"]
        require(path.is_file(), "trlc_offline_tokenizer_unavailable")
        actual = file_record(path)
        require(actual["size_bytes"] == row["size_bytes"], "trlc_offline_tokenizer_bytes_changed")
        if "sha256" in row:
            matches = actual["sha256"] == "sha256:" + row["sha256"]
        else:
            # SHA1 here identifies the publisher's Git blob; SHA256 binds the retained bytes.
            blob = hashlib.sha1(usedforsecurity=False)
            blob.update(f"blob {actual['size_bytes']}\0".encode("ascii"))
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                raw = stream.read(row["size_bytes"] + 1)
            blob.update(raw)
            matches = (len(raw) == row["size_bytes"] and blob.hexdigest() == row["git_blob_sha1"]
                       and "sha256:" + hashlib.sha256(raw).hexdigest() == actual["sha256"])
        require(matches, "trlc_offline_tokenizer_bytes_changed")
        files.append({"path": row["path"], **{key: actual[key] for key in ("sha256", "size_bytes")}})
    return {"status": "tokenizer_bytes_verified", "source_repository": inventory["source_repository"],
        "source_revision": inventory["source_revision"], "inventory_sha256": profile["tokenizer_inventory_sha256"],
        "files": files, "inference_executed": False}


def expected_manifest(original: dict, profile: dict | None = None) -> dict:
    value = json.loads(canonical(original))
    body = profile["policy_spec_template"]["embodiment_id"] if profile else EMBODIMENT_ID
    require(value.get("embodiment_id") == body, "body_mismatch")
    if body == EMBODIMENT_ID:
        value.update(control_hz=60, action_chunk_rows=1)
    elif body == TRLC_EMBODIMENT_ID:
        value.update(control_hz=60, action_chunk_rows=50)
        require(isinstance(value.get("cameras"), list)
                and [row["name"] for row in value["cameras"]] == ["overhead", "left_wrist", "right_wrist"],
                "source_cameras_changed")
        for camera in value["cameras"]:
            camera.update(profile["camera_overrides"][camera["name"]])
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
    if request["schema_version"] == TRLC_SCHEMA:
        value.update(license_id="Apache-2.0 AND Gemma-Terms-of-Use",
            license_basis="community_checkpoint_apache_with_inherited_gemma_public_terms",
            license_evidence_files=profile["license_evidence_files"],
            license_evidence_url=profile["license_evidence_url"],
            authorization_basis="current_user_instruction_for_internal_robot_simulation",
            usage_scope="internal_out_of_distribution_rigid_task_simulation_diagnostic",
            inherited_terms_apply=True, old_g1_task_authority_reused=False, new_human_review_claimed=False,
            model_redistribution_authorized=False, public_model_service_authorized=False,
            cloth_benchmark_claimed=False, vendor_policy_claimed=False,
            tokenizer_inventory_sha256=profile["tokenizer_inventory_sha256"])
    elif request["schema_version"] in {SCHEMA, FRANKA_SCHEMA}:
        value.update(license_basis="publisher_pinned_public_model_card", model_card_url=profile["model_card_url"],
                     model_card_sha256=profile["model_card_sha256"])
    else:
        value.update(license_basis="publisher_model_license_statement_in_pinned_repository_readme",
            license_evidence_url=profile["license_evidence_url"], license_evidence_files=profile["license_evidence_files"],
            license_declaration=profile["license_declaration"], publisher_intended_use=profile["publisher_intended_use"],
            hf_model_card_license_field=None)
    return {**value, "rights_digest": digest(value)}


def retained_body(request: dict, *, model_root: Path = MODEL_ROOT, embodiment_id: str | None = None) -> tuple[Path, dict, dict]:
    if request["schema_version"] == SCRIPTED_SCHEMA:
        require(embodiment_id in SCRIPTED_BODIES, "body_mismatch")
    else:
        require(embodiment_id is None, "body_mismatch")
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
