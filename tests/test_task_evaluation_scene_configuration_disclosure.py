from __future__ import annotations

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    CONTROL_PLANE,
    PROVIDER_GPU,
    TaskEvaluationSceneConfigurationDisclosureError,
    renders_on_provider,
    resolve_scene_configuration_disclosure,
)


def _admission(**overrides):
    disclosure = {
        "raw_interiorgs_downloaded_bytes_may_be_uploaded": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
        "provider_retention_rule": "bounded_to_the_run_then_governed_teardown",
    }
    disclosure.update(overrides)
    return {"provider_disclosure": disclosure}


def _stage(**overrides):
    disclosure = {
        "raw_interiorgs_bytes": True,
        "derived_rendered_views": True,
        "provider_training": False,
        "public_redistribution": False,
    }
    disclosure.update(overrides.pop("provider_disclosure", {}))
    authority = {
        "authority_reference": "operator-direction-2026-08-26",
        "provider_retention_terms_accepted": True,
        "provider_training_authorized": False,
    }
    authority.update(overrides.pop("human_authority", {}))
    return {"provider_disclosure": disclosure, "human_authority": authority}


def test_provider_render_requires_rights_stage_and_human_authority_together() -> None:
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=_stage(), rights_admission=_admission()
    )
    assert decision["render_execution_site"] == PROVIDER_GPU
    assert decision["source_appearance_bytes_to_provider"] is True
    assert decision["refusals"] == []
    assert renders_on_provider(decision)


def test_a_scene_whose_rights_withhold_upload_still_renders_locally() -> None:
    """The historical default is a permitted outcome, not a failure.

    A gated dataset that forbids uploading publisher bytes must keep rendering
    on the control plane rather than fail the run -- but the refusal is recorded
    so the reason is inspectable afterwards.
    """

    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=_stage(),
        rights_admission=_admission(
            raw_interiorgs_downloaded_bytes_may_be_uploaded=False
        ),
    )
    assert decision["render_execution_site"] == CONTROL_PLANE
    assert decision["source_appearance_bytes_to_provider"] is False
    assert "scene_configuration_source_upload_not_rights_admitted" in decision["refusals"]
    assert not renders_on_provider(decision)


def test_upload_is_refused_without_an_accepted_human_authority() -> None:
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=_stage(
            human_authority={"provider_retention_terms_accepted": False}
        ),
        rights_admission=_admission(),
    )
    assert decision["render_execution_site"] == CONTROL_PLANE
    assert (
        "scene_configuration_source_upload_human_authority_missing"
        in decision["refusals"]
    )


def test_admission_permitting_training_or_redistribution_does_not_admit_upload() -> None:
    """An upload permission means nothing without its retention boundary."""

    for override in (
        {"provider_training_allowed": True},
        {"public_redistribution_allowed": True},
        {"provider_retention_rule": "   "},
    ):
        decision = resolve_scene_configuration_disclosure(
            stage_one_configuration=_stage(), rights_admission=_admission(**override)
        )
        assert decision["render_execution_site"] == CONTROL_PLANE, override
        assert decision["rights_admission_permits_upload"] is False, override


def test_silence_anywhere_resolves_to_the_control_plane() -> None:
    """A missing field must never read as permission."""

    assert (
        resolve_scene_configuration_disclosure(
            stage_one_configuration={}, rights_admission=_admission()
        )["render_execution_site"]
        == CONTROL_PLANE
    )
    assert (
        resolve_scene_configuration_disclosure(
            stage_one_configuration=_stage(), rights_admission={}
        )["render_execution_site"]
        == CONTROL_PLANE
    )


def test_a_blueprint_captured_scene_uses_the_publisher_neutral_key() -> None:
    """Owned captures carry no publisher, so the neutral key must work alone."""

    admission = {
        "provider_disclosure": {
            "source_appearance_downloaded_bytes_may_be_uploaded": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
            "provider_retention_rule": "bounded_to_the_run_then_governed_teardown",
        }
    }
    stage = _stage()
    stage["provider_disclosure"] = {
        "source_appearance_bytes": True,
        "derived_rendered_views": True,
        "provider_training": False,
        "public_redistribution": False,
    }
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=stage, rights_admission=admission
    )
    assert decision["render_execution_site"] == PROVIDER_GPU
    assert renders_on_provider(decision)


def test_a_tampered_decision_is_not_honoured() -> None:
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=_stage(), rights_admission=_admission()
    )
    forged = {**decision, "render_execution_site": PROVIDER_GPU}
    forged["source_appearance_bytes_to_provider"] = True
    forged["rights_admission_permits_upload"] = True
    forged["decision_digest"] = "sha256:" + "0" * 64
    assert not renders_on_provider(forged)

    downgraded = resolve_scene_configuration_disclosure(
        stage_one_configuration=_stage(),
        rights_admission=_admission(
            raw_interiorgs_downloaded_bytes_may_be_uploaded=False
        ),
    )
    assert not renders_on_provider(
        {**downgraded, "render_execution_site": PROVIDER_GPU}
    )


def test_malformed_inputs_fail_closed_loudly() -> None:
    with pytest.raises(TaskEvaluationSceneConfigurationDisclosureError):
        resolve_scene_configuration_disclosure(
            stage_one_configuration=None, rights_admission=_admission()
        )
