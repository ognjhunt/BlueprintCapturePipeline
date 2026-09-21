"""Website consent must remain bound to the exact derived native assets."""

import copy
import hashlib
import json

import pytest

from scripts import prepare_paid_lane_launch as prep
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_native_launch_claims import validate_join
from tests.test_task_evaluation_scene_intake import request


def seal(value):
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def claims():
    intake = request()
    source = seal(
        {
            "schema_version": "website_scene_preparation.v1",
            "status": "intake_ready",
            "claim_ceiling": "development_only",
            "blockers": [],
            "intake_request": intake,
            "binding": {"task_context_digest": "sha256:" + "a" * 64},
            "authoring_inputs": {"configuration": {"scene_id": "website-1"}},
        }
    )
    rights = seal(
        {
            "schema_version": "website_native_rights_admission.v1",
            "scene_id": "website-1",
            "status": "admitted_for_internal_development",
            "claim_ceiling": "development_only",
            "preparation_digest": source["digest"],
            "owner": copy.deepcopy(intake["owner"]),
            "consent": copy.deepcopy(intake["consent"]),
            "execution_authority": copy.deepcopy(intake["execution"]),
            "task_context_digest": source["binding"]["task_context_digest"],
            "private_provider_processing_allowed": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
            "physical_measurement_proven": False,
            "provider_disclosure": {
                "prepared_background_allowed": True,
                "captured_frame_derivatives_allowed": True,
                "raw_capture_video_allowed": False,
                "provider_training_allowed": False,
                "public_redistribution_allowed": False,
            },
        }
    )
    return source, rights


def test_original_website_records_are_read_by_transport_and_embedded_seals(tmp_path):
    source, rights = claims()
    for value, schema, field in [
        (source, "task_evaluation_scene_source_manifest.v1", "source_manifest_digest"),
        (rights, "task_evaluation_scene_rights_admission.v1", "rights_admission_digest"),
    ]:
        path = tmp_path / (field + ".json")
        path.write_text(json.dumps(value))
        kwargs = dict(
            path=path,
            expected_digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_schema=schema,
            expected_status="retained" if value is source else "admitted",
            digest_field=field,
            scene_id="website-1",
        )
        assert prep._load_scene_claim_reference(**kwargs)[1] == value
        path.write_text(json.dumps({**value, "claim_ceiling": "partner_proof"}))
        with pytest.raises(prep.PaidLaneLaunchPreparationError):
            prep._load_scene_claim_reference(**kwargs)
    assert validate_join(source, rights, scene_id="website-1")


@pytest.mark.parametrize(
    "field",
    [
        "owner",
        "consent",
        "execution_authority",
        "preparation_digest",
        "task_context_digest",
        "scene_id",
        "provider_disclosure",
    ],
)
def test_resealed_foreign_or_expanded_consent_is_refused(field):
    source, rights = claims()
    if field == "provider_disclosure":
        rights[field]["raw_capture_video_allowed"] = True
    elif field in ("owner", "consent", "execution_authority"):
        rights[field] = {}
    else:
        rights[field] = "foreign"
    seal(rights)
    with pytest.raises(ValueError, match="website_claim_binding_invalid"):
        validate_join(source, rights, scene_id="website-1")


def test_website_consent_admits_only_revision_bound_assets():
    source, rights = claims()
    asset = {"digest": "sha256:" + "b" * 64, "size_bytes": 40}
    revision = {
        "scene_identity": {"id": "website-1"},
        "appearance": {"configured_representation": asset},
        "geometry": {},
        "replacement": {},
    }
    row = {
        "semantic_role": "scene_appearance",
        "source": {"sha256": asset["digest"], "size_bytes": 40},
        "staged_sha256": asset["digest"],
        "staged_size_bytes": 40,
    }
    kwargs = dict(
        packet_receipt={"source_bindings": [row]},
        source_manifest=source,
        rights_admission=rights,
        configured_scene_revision=revision,
    )
    prep._validate_provider_packet_source_rights(**kwargs)
    row["source"]["sha256"] = row["staged_sha256"] = "sha256:" + "c" * 64
    with pytest.raises(prep.PaidLaneLaunchPreparationError, match="provider_source_rights_invalid"):
        prep._validate_provider_packet_source_rights(**kwargs)
