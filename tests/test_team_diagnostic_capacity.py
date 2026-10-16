"""Beta L5: reject non-rendering rentals and scope the internal-only country opt-in."""

import hashlib
import json

import pytest

from blueprint_pipeline import team_candidate_quick10_vast as team_vast
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.vast_provider_adapter import _select_offer


@pytest.mark.parametrize("name", ["A800 PCIE", "A100 SXM4", "Unregistered accelerator"])
def test_isaac_offer_selection_never_falls_back_to_non_rt_gpu(name):
    offer = {"ask_contract_id": 1, "gpu_name": name, "gpu_ram": 81920,
             "dph_total": .4, "compute_cap": 800, "driver_version": "580.178.04"}
    assert _select_offer([offer], max_hourly_rate=.9) is None
    for policy in ("generation", "training", "open"):
        assert _select_offer([offer], max_hourly_rate=.9, gpu_selection_policy=policy) is not None


def _authority(tmp_path, **changes):
    inputs = {
        "run_id": "internal-team-diag-recovery-test-s04",
        "candidate_ids": ["blueprint-internal.hold-position-v1"],
        "scene_revision_digest": "sha256:e0fd71e52b0fcc8f16f42668e13bd9a9bd727d2db3cc0df95b785ee0191d2a81",
        "task_success_contract_digest": "sha256:c4faf026331e9bf3687b96f5b3c80a484f4a975a7e3812fec96f9524158bb962",
        "runtime_inputs_digest": "", **changes,
    }
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    path = tmp_path / "inputs.json"
    data = json.dumps(inputs).encode()
    path.write_bytes(data)
    return {"run_id": inputs["run_id"], "candidate_ids": inputs["candidate_ids"],
            "task_success_contract_digest": inputs["task_success_contract_digest"],
            "runtime_inputs_digest": inputs["runtime_inputs_digest"],
            "runtime_inputs": {"path": str(path), "size_bytes": len(data),
                               "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}}


def test_team_country_default_stays_us_without_reading_internal_fixture(monkeypatch):
    monkeypatch.delenv("BLUEPRINT_INTERNAL_TEAM_DIAGNOSTIC_VAST_COUNTRY", raising=False)
    assert team_vast.team_candidate_country_codes({}) == ("US",)


def test_singapore_opt_in_requires_exact_bound_authored_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_INTERNAL_TEAM_DIAGNOSTIC_VAST_COUNTRY", "SG")
    assert team_vast.team_candidate_country_codes(_authority(tmp_path)) == ("SG",)


@pytest.mark.parametrize("changes", [
    {"run_id": "customer-run"},
    {"candidate_ids": ["customer.policy"]},
    {"scene_revision_digest": "sha256:" + "a" * 64},
    {"task_success_contract_digest": "sha256:" + "b" * 64},
])
def test_singapore_cannot_change_customer_or_other_fixture_geography(tmp_path, monkeypatch, changes):
    monkeypatch.setenv("BLUEPRINT_INTERNAL_TEAM_DIAGNOSTIC_VAST_COUNTRY", "SG")
    with pytest.raises(ValueError, match="internal_diagnostic_country_scope_invalid"):
        team_vast.team_candidate_country_codes(_authority(tmp_path, **changes))


def test_singapore_rejects_changed_bound_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_INTERNAL_TEAM_DIAGNOSTIC_VAST_COUNTRY", "SG")
    authority = _authority(tmp_path)
    (tmp_path / "inputs.json").write_text("{}")
    with pytest.raises(ValueError, match="internal_diagnostic_country_input_binding_invalid"):
        team_vast.team_candidate_country_codes(authority)


@pytest.mark.parametrize("country", ["VN", "SG,US", "ALL", "*"])
def test_internal_override_never_expands_beyond_the_prepared_option(monkeypatch, country):
    monkeypatch.setenv("BLUEPRINT_INTERNAL_TEAM_DIAGNOSTIC_VAST_COUNTRY", country)
    with pytest.raises(ValueError, match="internal_diagnostic_country_invalid"):
        team_vast.team_candidate_country_codes({})
