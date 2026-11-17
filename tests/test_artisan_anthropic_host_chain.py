"""ARTISAN on Anthropic when every stage runs on the host: the same money and privacy rules, nothing rented.

The lane is B's all-host lane (``tests.test_scene_configuration_whole_chain_on_host``) with its
receipt declaring ARTISAN text on Anthropic. Its bundle entrypoint stands in for the chain; after
the chain it leaves stage 3's meter and stage input where the real ARTISAN driver does.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import artisan_provider_choice as choice
from blueprint_pipeline import paid_spend_meter
from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_vast
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests import test_scene_configuration_whole_chain_on_host as host_chain

PRODUCER = "stages/stage-3/producer/"
NO_FUNDS = {"id": "refused", "kind": "messages", "status": "priced", "cost_usd": 0.0,
            "cost_origin": "upstream_rejected_request_no_funds"}


@pytest.fixture(autouse=True)
def _no_host_folders(tmp_path, monkeypatch):
    """Nothing here may write to the host's own month journal or grant records."""
    monkeypatch.setattr(paid_spend_meter, "DEFAULT_METER_ROOT", tmp_path / "unexpected" / "meter")
    monkeypatch.setattr(choice, "DEFAULT_SUCCESSOR_ROOT", tmp_path / "unexpected" / "choices")
    yield
    assert not (tmp_path / "unexpected").exists()


def _anthropic_choice(key: Path) -> dict:
    grant = {"authoring_providers": ["openai", "anthropic"], "anthropic_provider_terms_reference": "sha256:" + "a" * 64,
             "privacy_policy_version": "2026-11-13.1"}
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    return choice.choose_authoring_provider(
        grant=grant, environ={choice.PREFER_ANTHROPIC_ENV: "1", choice.KEY_FILE_ENV: str(key)},
        read=lambda: {"status": choice.MODEL_READ_PASSED, "model_metadata_digest": "sha256:" + "b" * 64})


def _lane(tmp_path, monkeypatch, *, calls):
    """B's all-host lane declaring ARTISAN text on Anthropic; its stage 3 leaves ``calls`` in its meter."""
    real_staging = scene_vast._stage_owner_only_runtime_secrets
    key = tmp_path / "anthropic.key"
    key.write_text("synthetic-anthropic-key-never-sent")
    key.chmod(0o600)
    for name, value in (("ANTHROPIC_API_KEY_FILE", key), (paid_spend_meter.METER_ROOT_ENV, tmp_path / "meter"),
                        (choice.SUCCESSOR_ROOT_ENV, tmp_path / "choices")):
        monkeypatch.setenv(name, str(value))
    lane = host_chain._host_lane(tmp_path, monkeypatch)
    real_load = scene_vast.load_scene_configuration_provider_bundle_receipt
    monkeypatch.setattr(scene_vast, "load_scene_configuration_provider_bundle_receipt", lambda path, **kwargs: {
        **real_load(path, **kwargs), "replacement_authoring_text_provider": "anthropic"})
    scene_vast.validate_scene_configuration_paid_authority(None)["external_service_spend_caps"]["openai"][
        "stage_max_cost_usd"] = {"content_agents": 1.25}
    # The OpenAI half of the lane's secrets stays the fixture's; the Anthropic key and its staging are real.
    monkeypatch.setattr(scene_vast, "_provider_runtime_inputs",
                        lambda _authority, receipt=None: (scene_vast.anthropic_text_key(receipt), {}))
    monkeypatch.setattr(scene_vast, "_stage_owner_only_runtime_secrets", real_staging)
    lane.choice, lane.seen_at_authoring = _anthropic_choice(key), {}
    bound = host_chain.cpu_prestage.prepare_stage_prefix_before_gpu
    chain = bound.keywords["runner"]

    def entry(argv, *, cwd, env, stdout, stderr, check):
        # Authoring starts: what the month and the stage environment hold at this moment.
        staged = env.get("ANTHROPIC_API_KEY_FILE")
        lane.seen_at_authoring.update(journal=paid_spend_meter.read_journal(tmp_path / "meter")[0], key=staged,
                                      key_text=Path(staged).read_text() if staged else None,
                                      key_mode=Path(staged).stat().st_mode & 0o777 if staged else None)
        completed = chain(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=check)
        producer = Path(env["BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT"]) / PRODUCER
        (producer / "artisan_runtime").mkdir(parents=True, exist_ok=True)
        (producer / "artisan_runtime/blueprint_usage.json").write_text(json.dumps(
            {"schema_version": "artisan_upstream_usage.v1", "calls": calls}))
        (producer / "stage_production_input.v1.json").write_text(json.dumps(
            {"run_id": lane.receipt["run_id"], "configuration": {"artisan_provider_choice": lane.choice}}))
        return completed

    monkeypatch.setattr(host_chain.cpu_prestage, "prepare_stage_prefix_before_gpu",
                        functools.partial(bound.func, **{**bound.keywords, "runner": entry}))
    lane.key = key
    return lane


def _month(tmp_path):
    rows = paid_spend_meter.journal_rows(paid_spend_meter.read_journal(tmp_path / "meter")[0])
    return {row["meter_key"]: row for row in rows if row.get("provider") == "anthropic"}


def _forbid_network_and_vast(monkeypatch, *, resolved):
    import socket

    from blueprint_pipeline import vast_api_transport

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an all-host run contacted a provider or resolved a network name")

    def lookup(host, *args, **kwargs):
        resolved.append(host)
        if host != "api.anthropic.com":
            raise AssertionError(f"resolved {host}")
        return []
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    for module, name in ((socket, "create_connection"), (vast_api_transport, "_api_json"),
                         (scene_vast, "arm_independent_vast_watchdog"), (scene_vast, "close_independent_vast_watchdog"),
                         (scene_vast, "close_independent_vast_watchdog_without_allocation")):
        monkeypatch.setattr(module, name, forbidden)


def test_an_all_host_anthropic_run_holds_before_authoring_settles_after_and_rents_nothing(tmp_path, monkeypatch):
    calls = [{"id": "c1", "kind": "messages", "status": "priced", "cost_usd": 0.4, "cost_origin": "upstream_priced_usage"},
             {"id": "c2", "kind": "images", "status": "priced", "cost_usd": 0.2, "cost_origin": "upstream_priced_usage"}]
    lane = _lane(tmp_path, monkeypatch, calls=calls)
    resolved: list[str] = []
    _forbid_network_and_vast(monkeypatch, resolved=resolved)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert result["status"] == "completed", result["blockers"]
    # Nothing rented, staged, armed or called; no watchdog started.
    assert [name for name, _ in lane.events if name in {"adapter", "stage", "watchdog_close"}] == []
    assert result["independent_watchdog"]["status"] == host_chain.cpu_prestage.WATCHDOG_NOT_ARMED_STATUS
    assert result["independent_watchdog"]["provider_contacted"] is False
    assert result["object_store_cleanup"]["status"] == "nothing_staged_every_stage_ran_on_host"
    assert result["provider_mutations_performed"] == 0 and resolved in ([], ["api.anthropic.com"])
    # The key: this host's owner-only copy in the stage environment (not the operator's own file), removed after.
    staged = Path(lane.seen_at_authoring["key"])
    assert staged != lane.key and staged.is_relative_to(lane.job) and not staged.exists()
    assert lane.seen_at_authoring["key_text"] == lane.key.read_text() and lane.seen_at_authoring["key_mode"] == 0o600
    # The month: held at the allowance before authoring, settled to the Anthropic subtotal from the host archive.
    [held] = [row for row in lane.seen_at_authoring["journal"] if row.get("provider") == "anthropic"]
    assert (held["event"], held["held_usd"], held["basis"]) == ("admission", "1.25", "metered_list_price_not_officially_read")
    [row] = _month(tmp_path).values()
    assert (row["status"], float(row["settled_usd"])) == ("settled", 0.4)
    assert Path(result["provider_runtime_output_zip_path"]).name == "cpu_prestage_output.zip"
    assert choice.retained_successor(lane.choice["grant_authority_digest"]) is None


def test_an_all_host_run_that_ran_out_of_funds_turns_its_grant_to_openai(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, calls=[{"id": "c1", "kind": "messages", "status": "priced", "cost_usd": 0.1,
                                                "cost_origin": "upstream_priced_usage"}, NO_FUNDS])
    _forbid_network_and_vast(monkeypatch, resolved=[])
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert result["status"] == "completed", result["blockers"]
    successor = choice.retained_successor(lane.choice["grant_authority_digest"])
    assert successor["provider"] == "openai" and successor["reason"] == choice.NO_FUNDS_REASON
    assert successor["ended_attempt_run_id"] == lane.receipt["run_id"]
    assert successor["supersedes"]["digest"] == lane.choice["digest"]
    [row] = _month(tmp_path).values()
    assert float(row["settled_usd"]) == 0.1


def test_an_openai_all_host_run_writes_nothing_to_the_month_and_stages_no_anthropic_key(tmp_path, monkeypatch):
    real_staging = scene_vast._stage_owner_only_runtime_secrets
    monkeypatch.setenv(paid_spend_meter.METER_ROOT_ENV, str(tmp_path / "meter"))
    monkeypatch.delenv("ANTHROPIC_API_KEY_FILE", raising=False)
    lane = host_chain._host_lane(tmp_path, monkeypatch)
    monkeypatch.setattr(scene_vast, "_provider_runtime_inputs",
                        lambda _authority, receipt=None: (scene_vast.anthropic_text_key(receipt), {}))
    monkeypatch.setattr(scene_vast, "_stage_owner_only_runtime_secrets", real_staging)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert result["status"] == "completed", result["blockers"]
    assert "ANTHROPIC_API_KEY_FILE" not in lane.seen
    assert not (tmp_path / "meter").exists() or paid_spend_meter.read_journal(tmp_path / "meter")[0] == []
    assert os.environ.get("ANTHROPIC_API_KEY_FILE") is None


def test_a_month_hold_that_cannot_be_written_stops_the_attempt_before_authoring(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, calls=[])
    blocked = tmp_path / "meter-blocked"
    blocked.write_text("")
    monkeypatch.setenv(paid_spend_meter.METER_ROOT_ENV, str(blocked))
    with pytest.raises(scene_vast.TaskEvaluationSceneConfigurationVastError,
                       match="scene_configuration_anthropic_month_hold_unwritable"):
        lane.run(cpu_prestage_stage_limit="stage-4")
    assert lane.seen_at_authoring == {}  # no stage ran, no Anthropic call could be made
    assert [name for name, _ in lane.events if name in {"adapter", "stage", "watchdog_close"}] == []
