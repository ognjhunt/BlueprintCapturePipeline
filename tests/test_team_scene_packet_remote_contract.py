# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_job_contract.py
#   src/blueprint_pipeline/remote_cpu_job_allocator.py
"""The team-scene stage on plan 14's contract and paid seam: its descriptor, limits, receipts and admission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import remote_cpu_job_contract as contract
from tests.remote_cpu_allocator_fakes import (
    GIB,
    HOST_ENVIRONMENT,
    T0,
    RemoteCpuWorld,
    add_bootstrap_job,
    seal,
)
from tests.remote_team_scene_packet_support import (
    TEAM_JOB,
    TEAM_WORST_CASE_USD,
    team_authority,
    team_config,
    team_descriptor,
    team_stage,
)

EXECUTION = "blueprint-remote-cpu-team-scene-packet-x7k2p"


def _reasons(call) -> tuple[str, ...]:
    with pytest.raises(contract.RemoteCpuContractError) as caught:
        call()
    return caught.value.reasons


def test_the_team_stage_is_registered_with_its_own_contract_queue_and_retryable_blockers() -> None:
    stage = contract.STAGES["team_scene_packet"]
    assert (stage["stage_contract"], stage["queue"], stage["success_status"]) == (
        "team_scene_packet_remote.v1", "team-scene-packets", "completed")
    assert stage["retryable_blocker_prefixes"] == ("team_scene_packet_failed:OSError:",
                                                   "team_scene_packet_failed:MemoryError")
    abbreviations = [entry["abbreviation"] for entry in contract.STAGES.values()]
    assert len(set(abbreviations)) == len(abbreviations)
    descriptor = team_descriptor(team_config())
    assert descriptor["job_id"].startswith("rcj-tp-") and descriptor["stage_contract"] == "team_scene_packet_remote.v1"


def test_its_limits_come_from_the_config_entry_within_the_contract_maxima() -> None:
    config = team_config()
    limits = contract.stage_limits(config, "team_scene_packet")
    assert {name: limits[name] for name in ("vcpu", "memory_bytes", "task_timeout_seconds")} == {
        "vcpu": 4, "memory_bytes": 16 * GIB, "task_timeout_seconds": 3600}
    # The stage phase takes the configured timeout's remainder after fetch, seal/upload and the margin.
    assert limits["phase_seconds"] == {"fetch": 600, "stage": 3600 - 600 - 600 - 180, "seal_upload": 600}
    assert limits["max_output_bytes"] == 4 * GIB
    assert allocator.worst_case_usd(limits=limits, rate_table=config["rate_table"]) == TEAM_WORST_CASE_USD
    # A probe of this stage runs on its job under its own budget, so it fits the same $1 attempt cap.
    probe = allocator._probe_config(config, "team_scene_packet")
    probed = contract.stage_limits(probe, contract.PROBE_STAGE, profile="team_scene_packet")
    assert allocator.worst_case_usd(limits=probed, rate_table=config["rate_table"]) == TEAM_WORST_CASE_USD
    # Episode compilation keeps plan 14's own budget and bound, byte for byte.
    episode = contract.stage_limits(config, "episode_compilation")
    assert episode["phase_seconds"] == {"fetch": 300, "stage": 900, "seal_upload": 420}
    assert episode["max_output_bytes"] == 6 * GIB
    assert allocator.worst_case_usd(limits=episode, rate_table=config["rate_table"]) == 0.9072
    # A shorter configured timeout narrows only the stage phase; one too short for the budget is no config.
    shorter = team_config(stages={**config["stages"], "team_scene_packet": team_stage(task_timeout_seconds=1800)})
    assert contract.config_blockers(shorter) == []
    assert contract.stage_limits(shorter, "team_scene_packet")["phase_seconds"]["stage"] == 420
    minimum = contract.minimum_task_timeout_seconds("team_scene_packet")
    assert minimum == 600 + 600 + 180 + 300 and contract.minimum_task_timeout_seconds("episode_compilation") == 1
    for entry in (team_stage(task_timeout_seconds=minimum - 1), team_stage(task_timeout_seconds=3601),
                  team_stage(memory_bytes=33 * GIB)):
        assert "remote_cpu_config_invalid:stages" in contract.config_blockers(
            team_config(stages={**config["stages"], "team_scene_packet": entry}))


def test_its_output_root_is_the_attempts_scene_beside_the_documents() -> None:
    config = team_config()
    descriptor = team_descriptor(config)
    assert descriptor["outputs"]["output_root"].endswith("/0123456789abcdef-01234567/scene")
    assert contract.validate_descriptor(descriptor, config=config) == descriptor
    for root in ("/var/lib/blueprint/x/0123456789abcdef-01234567",
                 "/var/lib/blueprint/x/0123456789abcdef-76543210/scene",
                 "/var/lib/blueprint/x/0123456789abcdef-01234567/other"):
        assert "remote_cpu_descriptor_output_root_unbound" in _reasons(
            lambda root=root: team_descriptor(config, output_root=root))
    # A document beside the scene is an input; one inside it would be refused.
    inside = team_descriptor(config)
    inside["inputs"][2]["materialize_at"] = inside["outputs"]["output_root"] + "/definition.json"
    inside["descriptor_digest"] = contract.canonical_digest(inside, digest_field="descriptor_digest")
    assert "remote_cpu_descriptor_input_inside_output_root:inputs[2]" in _reasons(
        lambda: contract.validate_descriptor(inside, config=config))


def _receipt(descriptor: dict, *, status: str, blockers: list[str], output: dict | None = None) -> dict:
    result = {"schema_version": "team_scene_packet_remote_result.v1",
              "status": "blocked" if blockers else "completed", "blockers": blockers,
              "source_commit": descriptor["code"]["source_commit"], "result_digest": ""}
    receipt = {
        "schema_version": "remote_cpu_job_receipt.v1",
        **{name: descriptor[name] for name in ("job_id", "attempt", "attempt_id", "stage", "descriptor_digest")},
        "execution_name": EXECUTION, "status": status, "result": seal(result, "result_digest"), "output": output,
        "infrastructure_failures": [], "release_path_misses": [],
        "environment": {"environment_digest": descriptor["code"]["environment_digest"], "cpu_class": None},
        "phases": {"bootstrap": 1.0, "fetch": 2.0, "stage": 3.0, "seal_upload": 1.0}, "bytes_fetched": 1,
        "bytes_uploaded": 1, "private_url_recorded": False, "receipt_digest": ""}
    return seal(receipt, "receipt_digest")


@pytest.mark.parametrize("blocker, outcome", [
    ("team_scene_packet_failed:MemoryError", "infrastructure_failed"),
    ("team_scene_packet_failed:OSError:FileNotFoundError", "infrastructure_failed"),
    ("team_scene_owner_contract_invalid:owner_contract_required", "blocked"),
    ("team_scene_packet_failed:KeyError", "blocked"),
])
def test_resource_exhaustion_is_infrastructure_and_a_refusal_is_the_requests(blocker: str, outcome: str) -> None:
    descriptor = team_descriptor(team_config())
    verdict = contract.validate_receipt(_receipt(descriptor, status="blocked", blockers=[blocker]),
                                        descriptor=descriptor, execution_name=EXECUTION)
    assert verdict["outcome"] == outcome and verdict["terminal"] is (outcome == "blocked")
    if outcome == "infrastructure_failed":
        assert verdict["infrastructure_failures"] == [f"infrastructure_failed:retryable_stage_blocker:{blocker}"]


def test_a_receipt_over_the_stages_output_bound_is_refused() -> None:
    descriptor = team_descriptor(team_config())
    output = {"format": "remote_cpu_output.v1", "index": {"digest": "sha256:" + "6" * 64, "size_bytes": 900},
              "archive": {"digest": "sha256:" + "7" * 64, "size_bytes": 10240}, "paths_total": 12,
              "bytes_total": 4 * GIB + 1, "host_known": {"count": 0, "bytes": 0}}
    assert "remote_cpu_receipt_output_exceeds_limits" in _reasons(lambda: contract.validate_receipt(
        _receipt(descriptor, status="succeeded", blockers=[], output=output), descriptor=descriptor,
        execution_name=EXECUTION))


def _team_world(tmp_path: Path, monkeypatch, **changes) -> RemoteCpuWorld:
    world = RemoteCpuWorld(tmp_path, monkeypatch, config=team_config(), **changes)
    add_bootstrap_job(world.jobs, name=TEAM_JOB, timeout_seconds=3600)
    world.record_environment(stage="team_scene_packet")
    return world


def test_a_team_attempt_is_admitted_through_the_shared_seam_under_its_standing_authority(tmp_path: Path,
                                                                                       monkeypatch) -> None:
    world = _team_world(tmp_path, monkeypatch, authority=team_authority())
    descriptor = team_descriptor(world.config, environment_digest=HOST_ENVIRONMENT)
    dry = world.run("dispatch", descriptor=descriptor, stage="team_scene_packet", execute=False)
    assert (dry["status"], dry["admission"]["status"]) == ("dry_run_ready", "admitted")
    world.assert_untouched()
    result = world.run("dispatch", descriptor=descriptor, stage="team_scene_packet")
    assert (result["status"], result["success"]) == ("dispatched", True), result
    assert result["admission"]["resource_class"] == allocator.RESOURCE_CLASS
    [consumed] = world.consumed()
    record = json.loads(consumed.read_text(encoding="utf-8"))
    assert (record["stage"], record["worst_case_usd"]) == ("team_scene_packet", TEAM_WORST_CASE_USD)
    assert world.lease(descriptor)["state"] == "dispatched"
    [run] = world.rest.runs(validate_only=False)
    assert run.url.endswith(f"{TEAM_JOB}:run")
    assert contract.forbidden_record_content(result) == []


@pytest.mark.parametrize("authority, reason", [
    (None, "remote_cpu_standing_authority_missing"),
    (team_authority(stages=["episode_compilation"]), "remote_cpu_standing_authority_stage_not_covered"),
    (team_authority(max_attempt_usd=0.85), "remote_cpu_attempt_cap_exceeded"),
    (team_authority(max_daily_usd=0.5), "remote_cpu_daily_cap_exceeded"),
    (team_authority(expires_at_epoch=T0 - 1), "remote_cpu_standing_authority_expired"),
])
def test_no_team_attempt_starts_without_an_authority_that_covers_it(tmp_path: Path, monkeypatch, authority,
                                                                    reason: str) -> None:
    world = _team_world(tmp_path, monkeypatch, authority=authority, with_authority=authority is not None)
    result = world.run("dispatch", descriptor=team_descriptor(world.config), stage="team_scene_packet")
    assert result["status"] == "blocked" and reason in result["blockers"], result
    world.assert_untouched()


def test_a_team_attempt_without_its_probed_environment_or_job_is_refused(tmp_path: Path, monkeypatch) -> None:
    world = _team_world(tmp_path, monkeypatch, authority=team_authority())
    (world.root / "environment" / "team_scene_packet.json").unlink()
    result = world.run("dispatch", descriptor=team_descriptor(world.config), stage="team_scene_packet")
    assert "remote_cpu_environment_unrecorded" in result["blockers"]
    world.record_environment(stage="team_scene_packet")
    del world.jobs.jobs[TEAM_JOB]
    result = world.run("dispatch", descriptor=team_descriptor(world.config), stage="team_scene_packet")
    assert any(blocker.startswith("remote_cpu_job_unavailable:") for blocker in result["blockers"]), result
    world.assert_untouched()
    # The production config today names only episode compilation: the stage is refused by name.
    episode_only = team_config()
    episode_only["stages"].pop("team_scene_packet")
    world.write_config(seal(episode_only, "config_digest"))
    result = world.run("dispatch", descriptor=team_descriptor(team_config()), stage="team_scene_packet")
    assert result["blockers"] == ["remote_cpu_config_invalid:stage_missing:team_scene_packet"]
