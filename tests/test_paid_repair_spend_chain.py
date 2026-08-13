from __future__ import annotations

from blueprint_pipeline import paid_repair_spend_chain as spend_chain
from blueprint_pipeline import public_scene_artifixer3d_vast as artifixer_vast


def test_artifixer_uses_backend_neutral_spend_lineage_contract() -> None:
    assert (
        artifixer_vast.validate_artifixer3d_terminal_spend_chain
        is spend_chain.validate_artifixer3d_terminal_spend_chain
    )
    assert (
        artifixer_vast._validate_prior_authority_chain
        is spend_chain._validate_prior_authority_chain
    )
    assert (
        artifixer_vast._validate_prior_terminal_result
        is spend_chain._validate_prior_terminal_result
    )
    assert (
        spend_chain.ARTIFIXER3D_PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        == "public_scene_artifixer3d_paid_attempt_authority.v1"
    )
    assert (
        spend_chain.ARTIFIXER3D_RESULT_SCHEMA_VERSION
        == "public_scene_artifixer3d_vast_run.v1"
    )
