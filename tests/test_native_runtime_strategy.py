from __future__ import annotations

import pytest

from blueprint_pipeline.native_runtime_strategy import (
    LEGACY_SYNTHESIS_MODE_ENV,
    NATIVE_RUNTIME_BACKEND_ENV,
    native_runtime_strategy_catalog,
    native_runtime_strategy_for_mode,
    resolve_native_runtime_strategy,
)


def test_default_strategy_is_provider_neutral_truthful_preview() -> None:
    strategy = resolve_native_runtime_strategy({})

    assert strategy.backend_id == "site_splat"
    assert strategy.synthesis_mode == "splat_only"
    assert strategy.requires_model_runtime is False


def test_neutral_backend_setting_selects_legacy_cosmos_adapter_explicitly() -> None:
    strategy = resolve_native_runtime_strategy(
        {NATIVE_RUNTIME_BACKEND_ENV: "cosmos_wam"}
    )

    assert strategy.synthesis_mode == "cosmos_i2w"
    assert strategy.legacy_backend is True
    assert strategy.wam_backend_id == "cosmos_wam"


def test_legacy_synthesis_mode_is_a_compatibility_alias() -> None:
    strategy = resolve_native_runtime_strategy(
        {LEGACY_SYNTHESIS_MODE_ENV: "cosmos_i2w"}
    )

    assert strategy.backend_id == "cosmos_wam"


def test_conflicting_neutral_and_legacy_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="native_runtime_backend_conflict"):
        resolve_native_runtime_strategy(
            {
                NATIVE_RUNTIME_BACKEND_ENV: "site_splat",
                LEGACY_SYNTHESIS_MODE_ENV: "cosmos_i2w",
            }
        )


def test_unknown_backend_fails_at_the_configuration_boundary() -> None:
    with pytest.raises(ValueError, match="native_runtime_backend_unknown"):
        resolve_native_runtime_strategy({NATIVE_RUNTIME_BACKEND_ENV: "forced_mode"})


def test_catalog_and_mode_lookup_expose_claim_bounded_contract() -> None:
    catalog = native_runtime_strategy_catalog()

    assert set(catalog) == {"cosmos_wam", "site_splat"}
    assert catalog["cosmos_wam"]["claim_boundary"][
        "selection_does_not_prove_runtime_execution"
    ] is True
    assert native_runtime_strategy_for_mode("splat_only").backend_id == "site_splat"
