from __future__ import annotations

from blueprint_pipeline import vast_runtime_environment_contract as contract


def test_public_openai_identity_names_are_exact_and_non_secret() -> None:
    for name in contract.PUBLIC_OPENAI_IDENTITY_NAMES:
        assert contract.is_public_openai_identity_name(name) is True
    assert contract.is_public_openai_identity_name("OPENAI_API_KEY") is False
    assert contract.is_public_openai_identity_name("OPENAI_UNKNOWN_API_KEY_ID") is False
