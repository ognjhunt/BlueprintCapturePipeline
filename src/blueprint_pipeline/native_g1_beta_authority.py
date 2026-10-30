"""Compose existing G1 model-term evidence with a current internal beta task.

This is a machine derivation, not a new human review. The old scene-specific
approvals remain unchanged. The current owner's instruction authorizes internal
simulator testing; current capture/task authority is rechecked by beta before
launch. Neither source permits public media, redistribution or physical claims.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA = "native_g1_beta_development_authority.v1"
CELL_SCHEMA = "native_g1_beta_cell_authority.v1"
SOURCE_REVISION = "68479287a784a69be9ce6ad739311d2f11f75ef9"
INVENTORY_SHA256 = "sha256:6be81cc07bd9c3a4c924c96c57bc23b1a8e518f7884cc30dcccfa1868aebe259"
SONIC_INVENTORY_SHA256 = "sha256:d4c6ca841d7ee573ecf5c52172f6b1ac4076cfc968795524d82c7599cbc21710"
ROBOT_USD_SHA256 = "sha256:a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"
SONIC = {
    "revision": "6733128a3d8a523b1418b06bca3cdf61c8b0987f",
    "encoder_sha256": "sha256:013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "decoder_sha256": "sha256:c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
    "inventory_file_sha256": SONIC_INVENTORY_SHA256,
}
DP_AUXILIARY = {"kind": "publisher_required_torchvision_initialization",
    "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "sha256": "sha256:f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "size_bytes": 46_830_571}
PI_AUXILIARY = {"kind": "reviewed_inherited_paligemma_tokenizer",
    "inventory_file_sha256": "sha256:5020414c0d782bb33cdf1f53b220569c83aff55356d2d920f7741657eb0fe830"}
MODEL_REVIEW_FILES = {
    "humanoidarena_dp_g1_dex3_sonic": "dd23f2e8b7f69394081408a19046c4e2545365b3e5786fc5b69f8b8b2e4a4836",
    "humanoidarena_pi05_g1_dex3_sonic": "e4885ba657fea4d869e3573696a250e86fade608a7f8cb58099660d731e22b6b",
    "humanoidarena_dp_g1_dex3_sonic_vision_navi": "4c8749bf5ecfbcc0279b9e52b585986e8a28fca62709b3d16d19d2d57da9b098",
    "humanoidarena_pi05_g1_dex3_sonic_vision_navi": "913e30fa3e4dd6a9829c8f591696c4845806d361bca6d28810f4aba63002e949",
}
REVIEW_DIGESTS = {
    "humanoidarena_dp_g1_dex3_sonic": "sha256:4b59132cacaec5a3fb684533b7e311c4c37a0ccaf3a6bac7b3c325c141de0dc7",
    "humanoidarena_pi05_g1_dex3_sonic": "sha256:ade1bbaded1bdfe49e0352d5ed1d2a9b6bb268f3b09f1cf02603975c0a2e8105",
    "humanoidarena_dp_g1_dex3_sonic_vision_navi": "sha256:2b1945a1704c3f589b73badf27482f784b6c20a7de2fab1ead73a008dbb2ef94",
    "humanoidarena_pi05_g1_dex3_sonic_vision_navi": "sha256:7c19f3aabf6313cf5fb5bc89604d2c492edf51bc248c49e4070175f8f6018b56",
}
CANDIDATE_DIGESTS = {
    "humanoidarena_dp_g1_dex3_sonic": "sha256:0cb71830d1ff4ac5e6db73c558656de50a0a8cae44ead94c741e44f6688d6da1",
    "humanoidarena_pi05_g1_dex3_sonic": "sha256:aa6ae0c8ab888c264a002bc6474b2b3a3f8b898d72cf427428aa0b374d9e706f",
    "humanoidarena_dp_g1_dex3_sonic_vision_navi": "sha256:a679eff7ddabf60ec80223beb51dfb613a9079dc539e4172864f108bf080708d",
    "humanoidarena_pi05_g1_dex3_sonic_vision_navi": "sha256:f305ac7cc70a01c1de7aa38a185ec749079ab721a1ec34e0f85a2cd3eee77398",
}
SOURCE_FIELDS = ("scene_intent_id", "scene_intent_digest", "capture_id", "task_id",
                 "task_context_digest", "scene_revision_digest")
SCOPE = {
    "authority_kind": "composed_internal_development",
    "authorization_basis": "owner_requested_compatible_policy_simulator_testing_2026_10_30",
    "claim_ceiling": "development_only",
    "model_terms_review_reused": True,
    "new_human_review_claimed": False,
    "private_task_summary_permitted": True,
    "public_media_publication_authorized": False,
    "model_redistribution_authorized": False,
    "ranking_eligible": False,
    "qualification_permitted": False,
    "physical_outcome_claimed": False,
}


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def reviewed_model_terms(candidate_id: str) -> dict[str, Any]:
    if candidate_id not in MODEL_REVIEW_FILES:
        raise ValueError("g1_beta_candidate_not_builtin")
    name = f"g1_841757_{candidate_id}_development_rights_review.v1.json"
    path = Path(__file__).resolve().parents[2] / "docs/arm_decision_proof_v1/manifests" / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_beta_model_review_missing")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MODEL_REVIEW_FILES[candidate_id]:
        raise ValueError("g1_beta_model_review_changed")
    return {"artifact_sha256": "sha256:" + MODEL_REVIEW_FILES[candidate_id],
            "receipt": json.loads(data)}


def validate_beta_authority(value: Mapping[str, Any], *, candidate_id: str,
                            scene_plan_digest: str, inventory_file_sha256: str = INVENTORY_SHA256,
                            current_source: Mapping[str, Any] | None = None,
                            task_success_contract_digest: str | None = None) -> dict[str, Any]:
    try:
        row = json.loads(json.dumps(value, allow_nan=False))
        if row.get("schema_version") == CELL_SCHEMA:
            if (set(row) != {"schema_version", "base_authority", "cell", "scene_plan_digest",
                            "packet_receipt_digest", "authority_digest"}
                    or row["base_authority"].get("schema_version") != SCHEMA
                    or row["scene_plan_digest"] != scene_plan_digest
                    or not _digest(row["packet_receipt_digest"])
                    or row["authority_digest"] != canonical_digest(row, digest_field="authority_digest")):
                raise ValueError("invalid cell authority")
            base = row["base_authority"]
            validate_beta_authority(base, candidate_id=candidate_id,
                scene_plan_digest=base["scene_plan_digest"], inventory_file_sha256=inventory_file_sha256,
                current_source=current_source, task_success_contract_digest=task_success_contract_digest)
            validate_frozen_cell(row["cell"])
            return row
        model = row["model_review"]
        review = model["receipt"]
        source = row["current_task_source"]
        fields = {"schema_version", "candidate_id", "scene_plan_digest", "task_success_contract_digest",
                  "source_packet_receipt_digest", "checkpoint_digest", "inventory_file_sha256", "source_revision",
                  "sonic", "auxiliary_assets", "robot_usd_sha256", "model_review", "current_task_source", "site_id",
                  "registration_digest", "authority_digest", *SCOPE}
        if (set(row) != fields or row["schema_version"] != SCHEMA
                or candidate_id not in MODEL_REVIEW_FILES or row["candidate_id"] != candidate_id
                or row["scene_plan_digest"] != scene_plan_digest or not _digest(scene_plan_digest)
                or row["inventory_file_sha256"] != inventory_file_sha256 or inventory_file_sha256 != INVENTORY_SHA256
                or row["source_revision"] != SOURCE_REVISION or row["sonic"] != SONIC
                or row["auxiliary_assets"] != [PI_AUXILIARY if "_pi05_" in candidate_id else DP_AUXILIARY]
                or row["robot_usd_sha256"] != ROBOT_USD_SHA256
                or row["checkpoint_digest"] != CANDIDATE_DIGESTS[candidate_id]
                or any(row[key] != expected for key, expected in SCOPE.items())
                or any(type(row[key]) is not bool for key, expected in SCOPE.items() if type(expected) is bool)
                or any(not _digest(row[key]) for key in ("task_success_contract_digest", "source_packet_receipt_digest",
                                                        "checkpoint_digest", "registration_digest"))
                or set(model) != {"artifact_sha256", "receipt"}
                or model["artifact_sha256"] != "sha256:" + MODEL_REVIEW_FILES[candidate_id]
                or review["rights_review_digest"] != REVIEW_DIGESTS[candidate_id]
                or review["rights_review_digest"] != canonical_digest(review, digest_field="rights_review_digest")
                or review["candidate_id"] != candidate_id or review["source_revision"] != SOURCE_REVISION
                or review["inventory_file_sha256"] != INVENTORY_SHA256
                or review["status"] != "approved_for_development_simulation"
                or review["checkpoint_terms_reviewed"] is not True or review["source_and_sonic_terms_reviewed"] is not True
                or set(source) != set(SOURCE_FIELDS)
                or any(not _digest(source[key]) if key.endswith("digest") else
                       not isinstance(source[key], str) or not source[key] for key in SOURCE_FIELDS)
                or not isinstance(row["site_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}", row["site_id"])
                or (current_source is not None and source != {key: current_source[key] for key in SOURCE_FIELDS})
                or (task_success_contract_digest is not None and row["task_success_contract_digest"] != task_success_contract_digest)
                or row["authority_digest"] != canonical_digest(row, digest_field="authority_digest")):
            raise ValueError("invalid")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("g1_beta_development_authority_invalid") from exc
    return row


def validate_frozen_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    if (set(cell) != {"cell_id", "seed", "family", "partition", "resolved_scenario",
                     "cell_spec_digest", "resolved_scenario_digest"}
            or not isinstance(cell["cell_id"], str)
            or not re.fullmatch(r"screening\.[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", cell["cell_id"])
            or ".." in cell["cell_id"]
            or type(cell["seed"]) is not int or not 0 <= cell["seed"] < 2**31
            or cell["family"] != "canonical_anchor" or cell["partition"] != "screening"
            or not isinstance(cell["resolved_scenario"], Mapping)
            or set(cell["resolved_scenario"]) != {"family", "ordinal", "parameters"}
            or cell["resolved_scenario"]["family"] != "canonical_anchor"
            or type(cell["resolved_scenario"]["ordinal"]) is not int or cell["resolved_scenario"]["ordinal"] < 0
            or cell["resolved_scenario"]["parameters"] != {}
            or cell["cell_spec_digest"] != canonical_digest(cell["resolved_scenario"])
            or cell["resolved_scenario_digest"] != cell["cell_spec_digest"]):
        raise ValueError("g1_beta_frozen_cell_invalid")
    return dict(cell)


def derive_cell_authority(*, base_authority: Mapping[str, Any], cell: Mapping[str, Any],
                          scene_plan_digest: str, packet_receipt_digest: str) -> dict[str, Any]:
    """Bind a fixed reset repetition without changing the model-review receipt."""
    validate_frozen_cell(cell)
    row = {"schema_version": CELL_SCHEMA, "base_authority": dict(base_authority), "cell": dict(cell),
           "scene_plan_digest": scene_plan_digest, "packet_receipt_digest": packet_receipt_digest}
    row["authority_digest"] = canonical_digest(row, digest_field="authority_digest")
    return validate_beta_authority(row, candidate_id=base_authority["candidate_id"],
                                   scene_plan_digest=scene_plan_digest)


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
