from __future__ import annotations

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_interpretation import (
    OpenAIMultimodalEpisodeInterpreter,
    validate_episode_interpretation_rights,
)
from blueprint_pipeline.episode_interpretation_batch_authority import (
    ALLOWED_ARTIFACT_ROLES,
    derive_episode_interpretation_rights,
    validate_episode_interpretation_batch_authority,
)
from tests.test_episode_interpretation import _episode_root, _request


class _NoInvoke:
    def invoke(self, *_args, **_kwargs):  # pragma: no cover - rights only
        raise AssertionError("inference is not part of rights derivation")


def test_run_authority_derives_exact_episode_rights_after_digest_exists(tmp_path):
    request = _request(_episode_root(tmp_path, no_drop=False, deterministic_success=True))
    interpreter = OpenAIMultimodalEpisodeInterpreter(
        invoker=_NoInvoke(),
        model="gpt-5.6-terra",
        model_version="gpt-5.6-terra",
    )
    profile_digest = "sha256:" + "b" * 64
    authority = {
        "schema_version": "policy_canary_episode_interpretation_batch_authority.v1",
        "status": "approved",
        "run_id": "quick10-run-1",
        "interpreter": interpreter.identity.__dict__,
        "interpreter_profile_digest": profile_digest,
        "allowed_artifact_roles": sorted(ALLOWED_ARTIFACT_ROLES),
        "external_disclosure_authorized": True,
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "maximum_cost_usd": 1.5,
        "source_rights_admission_digest": "sha256:" + "a" * 64,
        "accepted_by": "team:robot-owner",
        "accepted_on": "2026-09-03T20:45:00Z",
        "authority_reference": "website-confirmation-1",
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    validated = validate_episode_interpretation_batch_authority(
        authority,
        run_id=authority["run_id"],
        interpreter=interpreter,
        interpreter_profile_digest=profile_digest,
        maximum_cost_usd=1.5,
    )
    path = tmp_path / "rights" / (
        request.input_receipt["input_bundle_digest"].removeprefix("sha256:")
        + ".json"
    )

    derived = derive_episode_interpretation_rights(
        authority=validated,
        request=request,
        interpreter=interpreter,
        output_path=path,
    )

    assert derived["input_bundle_digest"] == request.input_receipt["input_bundle_digest"]
    assert derived["authority_reference"] == authority["authority_reference"]
    assert validate_episode_interpretation_rights(
        rights_path=path,
        request=request,
        interpreter=interpreter,
    )["rights_digest"] == derived["rights_digest"]
