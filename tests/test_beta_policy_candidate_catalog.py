"""Catalog configuration does not grant policy execution or alter the ADP pair."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from blueprint_pipeline.beta_policy_candidate_catalog import (
    CATALOG_SCHEMA, BetaPolicyCatalogError, catalog_candidate, load_catalog,
    load_default_catalog, main, seal_catalog, validate_catalog,
)
from blueprint_pipeline.task_evaluation_policy_candidate_registry import validate_selected_pair


def test_default_catalog_records_all_named_candidates_and_only_existing_initial_interfaces():
    value = load_default_catalog()
    assert len(value["candidates"]) == 120
    enabled = {row["candidate_id"] for row in value["candidates"] if row["eligibility"]["enabled"]}
    from blueprint_pipeline.beta_policy_screening_g1 import CANDIDATES
    assert enabled == {"pi05_droid", "groot_n17_droid", "cosmos3_edge_policy_droid", "molmoact2_droid", "cosmos3_nano_policy_droid", *CANDIDATES}
    company_rows = [row for row in value["candidates"] if row.get("deployment") and row["candidate_id"] not in CANDIDATES
                    and not (row["deployment"]["relationship"] == "public_controller_integration"
                             and not row["deployment"]["companies"])]
    # 74 packaged routes plus 5 public fixed-arm deployment associations without a route.
    assert len(company_rows) == 79
    assert all(row["eligibility"]["blockers"] for row in company_rows)
    assert {row["deployment"]["configuration_id"] for row in company_rows
            if row["deployment"]["relationship"] == "publisher_demonstrated"} == {
        "i2rt_yam_bimanual", "agilex_piper_bimanual", "universal_robots_ur5e_bimanual",
        "franka_research_3_bimanual", "kinova_gen3_bimanual", "ufactory_xarm7_bimanual"}
    assert catalog_candidate(value, "dreamlabs_dream1")["policy"]["digest"] is None
    assert validate_selected_pair(["groot_n17_droid", "pi05_droid"]) == ("pi05_droid", "groot_n17_droid")


def test_forty_configurations_need_data_only_and_duplicate_identity_is_refused():
    original = load_default_catalog()
    source = catalog_candidate(original, "pi05_droid")
    rows = [{**deepcopy(source), "candidate_id": f"configured_{number:02d}"} for number in range(40)]
    result = seal_catalog({**original, "candidates": rows})
    assert len(result["candidates"]) == 40
    assert len({row["configuration_digest"] for row in result["candidates"]}) == 40
    rows[-1]["candidate_id"] = rows[0]["candidate_id"]
    with pytest.raises(BetaPolicyCatalogError, match="duplicate_candidate_id"):
        seal_catalog({**original, "candidates": rows})


@pytest.mark.parametrize("mutation", ["digest", "command", "runnable", "credential", "horizon", "display_name", "candidate_id"])
def test_catalog_rejects_identity_drift_or_executable_overrides(mutation):
    value = load_default_catalog()
    row = next(row for row in value["candidates"] if row["candidate_id"] == "pi05_droid")
    if mutation == "digest":
        row["policy"]["digest"] = "sha256:" + "0" * 64
        with pytest.raises(BetaPolicyCatalogError, match="candidate_digest_mismatch"):
            validate_catalog(value)
        return
    if mutation in {"command", "runnable"}:
        row[mutation] = "untrusted" if mutation == "command" else True
    elif mutation == "credential":
        row["policy"]["source"] = "https://user:password@example.invalid/model"
    elif mutation == "horizon":
        row["interface"]["execution_horizon"] = 99
    elif mutation == "display_name":
        row["display_name"] = "x" * 201
    elif mutation == "candidate_id":
        row["candidate_id"] = "unsupported:native-id"
    with pytest.raises(BetaPolicyCatalogError):
        seal_catalog(value)


def test_catalog_file_is_operator_owned_and_not_a_symlink(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(load_default_catalog()))
    path.chmod(0o600)
    assert load_catalog(path)["catalog_id"] == "blueprint-beta-policies"
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(BetaPolicyCatalogError, match="unavailable"):
        load_catalog(link)
    path.chmod(0o666)
    with pytest.raises(BetaPolicyCatalogError, match="operator_owned"):
        load_catalog(path)


def test_duplicate_json_keys_and_unavailable_without_reason_fail(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text('{"schema_version":"first","schema_version":"second"}')
    with pytest.raises(BetaPolicyCatalogError, match="duplicate_json_key"):
        load_catalog(path)
    value = load_default_catalog()
    next(row for row in value["candidates"] if not row["eligibility"]["enabled"])["eligibility"]["blockers"] = []
    with pytest.raises(BetaPolicyCatalogError, match="disabled_reason_missing"):
        seal_catalog(value)


def test_published_catalog_schema_matches_runtime_validator():
    path = Path(__file__).parents[1] / "docs/schemas/beta_policy_candidate_catalog.v1.schema.json"
    assert json.loads(path.read_text()) == CATALOG_SCHEMA


def test_operator_cli_requires_reseal_after_revision_change_and_never_overwrites(tmp_path, capsys):
    original = load_default_catalog()
    source = tmp_path / "catalog.edit.json"
    target = tmp_path / "catalog.next.json"
    source.write_text(json.dumps(original))
    assert main(["validate", "--catalog", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    changed = {**original, "revision": original["revision"] + "-access-ready"}
    source.write_text(json.dumps(changed))
    assert main(["validate", "--catalog", str(source)]) == 2
    assert json.loads(capsys.readouterr().out)["blocker"] == "beta_catalog_digest_mismatch"
    assert main(["seal", "--input", str(source), "--output", str(target)]) == 0
    sealed = load_catalog(target)
    assert sealed["catalog_digest"] != original["catalog_digest"]
    assert sealed["revision"] == changed["revision"]
    assert json.loads(source.read_text()) == changed
    capsys.readouterr()
    assert main(["seal", "--input", str(source), "--output", str(target)]) == 2
    assert json.loads(capsys.readouterr().out)["blocker"] == "beta_catalog_output_exists"
    assert load_catalog(target) == sealed
    assert main(["inspect", "--catalog", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["candidate_count"] == len(original["candidates"])
    assert report["candidates"][0]["interface"]["policy_interface_id"] == "openpi_droid.v1"
