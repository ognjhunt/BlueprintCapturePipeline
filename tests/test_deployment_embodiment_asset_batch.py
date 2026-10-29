"""Many public robot sources progress independently, preserving verified retries."""
from copy import deepcopy
from http.client import IncompleteRead
import json

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
pytest_plugins = ["tests.test_deployment_embodiment_assets"]


def test_batch_reuses_verified_sources_and_keeps_unavailable_robots_visible(source_case):
    case = source_case
    root = case.destination.parent / "batch"
    result = assets.materialize_available_sources(root, get=case.get)
    rows = {row["embodiment_id"]: row for row in result["results"]}
    assert rows["kuka_lbr_iiwa_14"]["status"] == "source_prepared"
    receipt = assets.verify_source_bundle(root / "kuka_lbr_iiwa_14")
    assert rows["kuka_lbr_iiwa_14"]["bundle_digest"] == receipt["bundle_digest"]
    assert rows["kuka_lbr_iiwa_14"]["size_bytes"] == sum(len(value) for value in case.content.values())
    assert rows["standard_bots_ro1"]["status"] == "source_unavailable"
    assert rows["droid_franka_robotiq_2f85"]["status"] == "existing_runtime"
    assert result["execution_authorized"] is False
    case.calls.clear()
    assert assets.materialize_available_sources(root, get=case.get) == result
    assert case.calls == []
    assert not (root / "standard_bots_ro1").exists()


@pytest.mark.parametrize("failure", [OSError("fixture_download_unavailable"), IncompleteRead(b"", 10)])
def test_one_failed_download_does_not_stop_later_public_sources(source_case, failure):
    case = source_case
    other = deepcopy(case.recipe)
    other["embodiment_id"] = "second_fixture_robot"
    case.data["recipes"].append(other)
    case.data_path.write_text(json.dumps(case.data))
    failed = False

    def fail_once(url, *, maximum):
        nonlocal failed
        if not failed:
            failed = True
            raise failure
        return case.get(url, maximum=maximum)

    root = case.destination.parent / "batch"
    rows = {row["embodiment_id"]: row for row in
            assets.materialize_available_sources(root, get=fail_once)["results"]}
    assert rows["kuka_lbr_iiwa_14"] == {"embodiment_id": "kuka_lbr_iiwa_14",
        "status": "preparation_failed", "reason": str(failure)}
    assert rows["second_fixture_robot"]["status"] == "source_prepared"
    assert not (root / "kuka_lbr_iiwa_14").exists()
    assert assets.verify_source_bundle(root / "second_fixture_robot")


def test_changed_retained_asset_is_reported_without_overwriting(source_case):
    case = source_case
    root = case.destination.parent / "batch"
    assets.materialize_available_sources(root, get=case.get)
    entry = root / "kuka_lbr_iiwa_14" / "source" / case.recipe["source"]["entry_path"]
    entry.write_bytes(b"changed")
    row = assets.materialize_available_sources(root, get=case.get)["results"][0]
    assert row["status"] == "preparation_failed"
    assert row["reason"] == "asset_bundle_file_changed"
    assert entry.read_bytes() == b"changed"


@pytest.mark.parametrize("url", ["http://api.github.com/a", "https://api.github.com.evil.test/a",
    "https://raw.githubusercontent.com:444/a", "https://user:password@api.github.com/a"])
def test_public_download_rejects_unexpected_origins_before_network(monkeypatch, url):
    def unexpected(*args, **kwargs):
        pytest.fail("unexpected origin reached the transport")
    monkeypatch.setattr(assets, "request", unexpected)
    with pytest.raises(assets.AssetPreparationError, match="asset_source_origin_not_allowed"):
        assets._get(url, maximum=1024)


def test_cli_reports_partial_failure_after_processing_all_recipes(monkeypatch, tmp_path, capsys):
    report = {"results": [{"status": "preparation_failed"}, {"status": "source_prepared"}]}
    monkeypatch.setattr(assets, "materialize_available_sources", lambda destination: report)
    assert assets.main(["--all-available", "--destination", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out) == report
