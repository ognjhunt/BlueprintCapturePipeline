"""Offline public CLI resource-limit compatibility; no allocation/import closure."""
import argparse

import pytest

from blueprint_pipeline.paid_resource_cli_arguments import add_adp_resource_arguments


def test_native_resource_defaults_and_explicit_selectors_are_preserved():
    parser = argparse.ArgumentParser()
    add_adp_resource_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.adp_max_hourly_rate_usd == 0.80
    assert defaults.adp_max_spend_usd == 2.00
    assert defaults.adp_hard_ttl_seconds == 7200
    assert defaults.adp_machine_avoidlist is None
    assert defaults.adp_allowed_vast_machine_id == []
    assert defaults.adp_excluded_vast_machine_id == []
    assert defaults.adp_allowed_active_vast_instance_id == []

    explicit = parser.parse_args([
        "--adp-max-hourly-rate-usd", "0.5", "--adp-max-spend-usd", "1.5",
        "--adp-hard-ttl-seconds", "900", "--adp-machine-avoidlist", "machines.json",
        "--adp-allowed-vast-machine-id", "21899", "--adp-allowed-vast-machine-id", "44762",
        "--adp-excluded-vast-machine-id", "123", "--adp-allowed-active-vast-instance-id", "456",
    ])
    assert explicit.adp_max_hourly_rate_usd == 0.5
    assert explicit.adp_max_spend_usd == 1.5
    assert explicit.adp_hard_ttl_seconds == 900
    assert explicit.adp_machine_avoidlist == "machines.json"
    assert explicit.adp_allowed_vast_machine_id == [21899, 44762]
    assert explicit.adp_excluded_vast_machine_id == [123]
    assert explicit.adp_allowed_active_vast_instance_id == [456]
    assert parser.parse_args([]).adp_allowed_vast_machine_id == []


@pytest.mark.parametrize("flag", [
    "--adp-hard-ttl-seconds", "--adp-allowed-vast-machine-id",
    "--adp-excluded-vast-machine-id", "--adp-allowed-active-vast-instance-id",
])
def test_native_resource_integer_flags_still_reject_noninteger_input(flag):
    parser = argparse.ArgumentParser()
    add_adp_resource_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args([flag, "not-an-integer"])
    assert error.value.code == 2
