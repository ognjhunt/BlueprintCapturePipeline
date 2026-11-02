"""A closed attempt's resumable ARTISAN run carries into the next attempt; it never grants spend.

capture-447ae4f4, 2026-11-02: ARTISAN built a candidate tray and stopped in its
review phase with $13.76 of a frozen $15 cap spent. These tests build the same
shape of source: a closed launch for the same owner, intent, scene and task,
whose authenticated output archive holds a resumable upstream run.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_artisan_continuation as continuation
from blueprint_pipeline import task_evaluation_partial_astra_transport as transport
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_owner_attempt_profiles import make_owner_attempt_record

SOURCE_RUN = "website-owner-0000aaaa-20261101t232659z"
SUCCESSOR_RUN = "website-owner-1111bbbb-20261102t030000z"
SHA = "sha256:" + "a" * 64
COMMIT = "a" * 40
PREFIX = continuation.PREFIX


def seal(value, field):
    digest = cross_runtime_canonical_digest if field in {"intent_digest", "attempt_digest"} else canonical_digest
    value[field] = digest(value, digest_field=field)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def configuration(cap):
    return {"schema_version": "rigid_replacement_authoring_configuration.v1", "authoring_backend": "artisan_v1",
            "authoring_budget": {"schema_version": "task_object_authoring_budget.v1", "mode": "finite_cap",
                                 "maximum_cost_usd": cap,
                                 "scope": "whole_authoring_stage_cumulative_including_resume"},
            "replacement_identity": {"id": "wooden_tray", "version": "v1"}, "metric_envelope": {"minimum_xyz_m": [0, 0, 0]}}


def request(run_id, cap, *, frame=SHA):
    value = {"schema_version": "task_object_construction_request.v1", "backend": "artisan_v1", "run_id": run_id,
             "source_commit": COMMIT, "source_envelope_digest": "sha256:" + run_id[-1] * 64,
             "configuration": configuration(cap), "configuration_sha256": canonical_digest(configuration(cap)),
             "authoring_budget": configuration(cap)["authoring_budget"],
             "source_frames": [{"path": "/workspace/source-0000.png", "sha256": frame, "size_bytes": 7}],
             "support_and_placement": {"recipe_id": run_id + "-recipe", "recipe_digest": "sha256:" + run_id[-2] * 64,
                                       "stage_sequence": [
                                           {"adapter": {"id": "website_prepared_appearance"},
                                            "configuration": {"digest": SHA, "uri": f"s3://inputs/{run_id}/stage-1"}},
                                           {"adapter": {"id": "content_agents_rigid_replacement"},
                                            "configuration": {"digest": canonical_digest(configuration(cap)),
                                                              "uri": f"s3://inputs/{run_id}/stage-3"}}]}}
    return seal(value, "request_digest")


def usage(costs):
    return {"schema_version": "artisan_upstream_usage.v1", "maximum_cost_usd": 15.0, "maximum_requests": 100,
            "deadline_epoch": 1.0, "calls": [{"id": str(i), "kind": "create", "model": "gpt-6-astra",
                                              "status": "priced", "cost_usd": cost} for i, cost in enumerate(costs)]}


CALLS = (2.5, 2.5, 2.5, 2.5, 2.5, 1.26)  # $13.76 in six calls; the largest is $2.50


def source_archive(path, *, source_request, terminal_phase="P5_review", calls=CALLS, extra=None, omit=()):
    upstream = "pl_" + source_request["request_digest"][7:39]
    runs = f"{PREFIX}workspace/data/cache/runs/{upstream}/"
    members = {
        PREFIX + "blueprint_request.v1.json": json.dumps(source_request),
        PREFIX + "terminal.v1.json": json.dumps({"schema_version": "blueprint_artisan_terminal.v1", "run_id": upstream,
                                                  "phase": terminal_phase, "status": "running", "registered": False,
                                                  "termination_reason": None}),
        PREFIX + "blueprint_usage.json": json.dumps(usage(list(calls))),
        PREFIX + "bridge_context.v1.json": json.dumps({"maximum_cost_usd": 15.0}),
        PREFIX + "prompt.txt": "build the tray",
        PREFIX + "last_failure.json": json.dumps({"exception_type": "AuthoringContractError"}),
        PREFIX + "official_openai_cost/openai_official_cost_run_reservation.v1.json": "{}",
        PREFIX + "blueprint_usage.lock": "",
        PREFIX + "controller-0.log": "artisan: artisan_shared_stage_authority_exhausted",
        runs + "pipeline_state.json": json.dumps({"phase": terminal_phase}),
        runs + "workspace/candidate/object.usd": "#usda 1.0",
        runs + "workspace/blueprint_sandbox_ipc/1.request.json": "{}",
        continuation.STAGE_INPUT: json.dumps({"configuration": source_request["configuration"]}),
        **(extra or {}),
    }
    members = {name: value for name, value in members.items() if not name.endswith(tuple(omit))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as packed:
        for name, value in sorted(members.items()):
            packed.writestr(name, value)
    return upstream


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "reserve_control_plane_disk", lambda *_a, **_k: nullcontext())
    intents, launches = tmp_path / "scene-intents", tmp_path / "task-evaluation-launch-runs"
    intent = seal({"intent_id": "scene-owner", "authenticated_issuer": "blueprint-webapp",
                   "request": {"owner": {"user_id": "owner", "organization_id": "org"},
                               "task": {"task_id": "pick-one", "reuse_completed_stages": False},
                               "execution": {"max_total_spend_usd": 1000.0}},
                   "accepted_at_epoch": 1789497600.0}, "intent_digest")
    write(intents / "scene-owner/intent.json", intent)
    preparation = {"scene_intent_digest": intent["intent_digest"], "scene": {"identity": {"id": "840938"}},
                   "task": {"identity": {"id": "pick-one"}}, "team_namespace": "new-namespace",
                   "run_id": SUCCESSOR_RUN, "expected_production_commit": COMMIT}
    attempts = []
    for attempt_id in ("old", "new"):
        attempt = seal({"intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
                        "attempt_id": attempt_id, "source_commit": COMMIT, "runtime_digest": SHA,
                        "input_digest": canonical_digest(preparation) if attempt_id == "new" else SHA,
                        "provider": "vast", "maximum_spend_usd": 20.0}, "attempt_digest")
        write(intents / "scene-owner/attempts" / (attempt_id + ".json"), attempt)
        attempts.append(attempt)
    current = make_owner_attempt_record(owner_fields=bind_scene_attempt(attempts[1]), phase="scene_configuration",
        team_namespace="new-namespace", scene_id="840938", task_id="pick-one", runtime_source_bundle_digest=SHA)
    owner_path = write(tmp_path / "current-owner.json", current)
    launch_root = launches / (SOURCE_RUN + "-activation-auto-launch")
    source_request = request(SOURCE_RUN, 15.0)
    archive = launch_root / transport.ARCHIVE_RELATIVE

    def build(**options):
        upstream = source_archive(archive, source_request=options.pop("source_request", source_request), **options)
        authority = seal({"bundle_sha256": SHA}, "authority_digest")
        authority_path = write(launch_root / "source-authority.json", authority)
        bundle_path = write(launch_root / "source-bundle.json", {"bundle_sha256": SHA})
        profile = {**bind_scene_attempt(attempts[0]), "source_commit": COMMIT,
                   "task_evaluation_run": {"run_mode": "scene_configuration", "team_namespace": "old-namespace",
                                           "scene_id": "840938", "task_id": "pick-one",
                                           "configuration_run_id": SOURCE_RUN},
                   "immutable_inputs": [
                       {"name": "source_bundle_manifest", "path": str(bundle_path), "digest": transport._sha(bundle_path)},
                       {"name": "scene_configuration_attempt_authority", "path": str(authority_path),
                        "digest": transport._sha(authority_path)}]}
        seal(profile, "profile_digest")
        write(launch_root / "launch_profile.json", profile)
        launch = seal({"status": "blocked", "launch_id": launch_root.name,
                       "launch_profile_digest": profile["profile_digest"]}, "receipt_digest")
        write(launch_root / "launch_receipt.json", launch)
        write(launch_root / "post_teardown_provider_zero_receipt.json", seal(
            {"launch_id": launch_root.name, "status": "provider_zero_confirmed", "provider_zero_verified": True,
             "continuing_spend_from_this_run": False, "blockers": [],
             "launch_profile_digest": profile["profile_digest"], "receipt_digest": launch["receipt_digest"]},
            "provider_zero_receipt_digest"))
        digest, size = transport._sha(archive), archive.stat().st_size
        write(launch_root / "allocator/scene-configuration-job/task_evaluation_scene_configuration_vast_result.v1.json",
              seal({"run_id": SOURCE_RUN, "source_commit": COMMIT, "bundle_sha256": SHA,
                    "authority_digest": authority["authority_digest"],
                    "provider_runtime_output_zip_path": str(archive), "provider_runtime_output_zip_sha256": digest,
                    "provider_runtime_output_remote_reference": {
                        "status": "remote_verified", "digest": digest, "readback_digest": digest, "size_bytes": size,
                        "readback_size_bytes": size, "full_byte_service_account_readback_passed": True}},
                   "result_digest"))
        return upstream

    def envelope(cap=25.0, config=None):
        path = write(tmp_path / f"successor-stage-3-{cap}.json", config or configuration(cap))
        return {"run_id": SUCCESSOR_RUN, "request": preparation, "team_namespace": "new-namespace",
                "expected_production_commit": COMMIT,
                "recipe": {"stage_sequence": [{"stage_id": "stage-1"}, {"stage_id": "stage-3"}]},
                "stage_configuration_references": [
                    {"contract_path": "construction.recipe.stage_sequence.0.configuration", "digest": SHA},
                    {"contract_path": "construction.recipe.stage_sequence.1.configuration",
                     "digest": transport._sha(path), "materialized_path": str(path)}]}

    def args(cap=25.0, config=None):
        return dict(owner_attempt_path=owner_path, envelope=envelope(cap, config), output_root=tmp_path / "selected",
                    intent_root=intents, launch_root=launches)

    return {"build": build, "args": args, "source_request": source_request, "tmp": tmp_path}


def _select(source, **kwargs):
    return continuation.select_artisan_continuation_source(**source["args"](**kwargs))


def test_a_resumable_source_is_selected_with_its_workspace_and_carried_spend(source):
    upstream = source["build"]()
    selection = _select(source)
    value = json.loads(selection.read_text())
    descriptor = json.loads(Path(value["descriptor"]["path"]).read_text())
    assert value["new_spend_authorized"] is False and descriptor["new_spend_authorized"] is False
    assert descriptor["source_upstream_run_id"] == upstream and descriptor["source_terminal_phase"] == "P5_review"
    assert descriptor["carried_spend"]["cost_usd"] == 13.76 and descriptor["carried_spend"]["call_count"] == 6
    assert descriptor["carried_spend"]["largest_call_usd"] == 2.5
    assert descriptor["original_runtime_root"] == continuation.ORIGINAL_ROOT
    with zipfile.ZipFile(value["runtime_archive"]["path"]) as archive:
        names = set(archive.namelist())
    runs = f"workspace/data/cache/runs/{upstream}/"
    assert {runs + "pipeline_state.json", runs + "workspace/candidate/object.usd"} <= names
    assert {continuation.RELOCATED_ROOT + name for name in continuation.RELOCATED} <= names
    # Per-run state, scratch and the source's own cost records never travel.
    assert not any("official_openai_cost" in n or "blueprint_sandbox_ipc" in n or n.endswith(".lock")
                   or "controller-" in n for n in names)
    assert not any(n in continuation.RELOCATED for n in names)  # nothing at the successor's own paths
    assert _select(source) == selection  # retained selection is reopened, not rebuilt


@pytest.mark.parametrize("cap", [16.0, 15.0, 13.76, 5.0])
def test_a_cap_without_room_for_one_source_sized_call_selects_nothing(source, cap):
    source["build"]()
    assert _select(source, cap=cap) is None
    assert not (source["tmp"] / "selected").exists()


@pytest.mark.parametrize("change", ["completed", "terminal_done", "no_checkpoint", "unpriced", "configuration",
                                    "articulated"])
def test_a_source_that_cannot_resume_safely_is_not_selected(source, change):
    options = {}
    if change == "completed":
        options["extra"] = {"stages/stage-3/producer/candidates/0000/replacement_asset.usdz": "x"}
    elif change == "terminal_done":
        options["terminal_phase"] = "DONE"
    elif change == "unpriced":
        options["calls"] = (12.0, None)  # an unknown charge must be reconciled first
    elif change == "articulated":
        changed = request(SOURCE_RUN, 15.0)
        changed["configuration"]["schema_version"] = "articulated_replacement_authoring_configuration.v1"
        options["source_request"] = seal({k: v for k, v in changed.items() if k != "request_digest"}, "request_digest")
    if change == "no_checkpoint":
        options["omit"] = ("pipeline_state.json",)
    source["build"](**options)
    config = None
    if change == "configuration":
        config = configuration(25.0)
        config["replacement_identity"] = {"id": "another_object", "version": "v1"}
    assert _select(source, config=config) is None
    assert not (source["tmp"] / "selected").exists()


def test_a_successor_that_does_not_run_artisan_selects_nothing(source):
    source["build"]()
    config = configuration(25.0)
    config["authoring_backend"] = "astra_cad_blender_v1"
    args = source["args"](config=config)
    assert continuation.successor_is_artisan(args["envelope"]) is False
    assert continuation.select_artisan_continuation_source(**args) is None


@pytest.fixture
def staged(source, tmp_path, monkeypatch):
    source["build"]()
    args = source["args"]()
    selection = continuation.select_artisan_continuation_source(**args)
    runtime = tmp_path / "provider_runtime"
    staged = continuation.stage_artisan_continuation(
        selection_path=selection, runtime=runtime, successor_run_id=SUCCESSOR_RUN,
        owner_attempt_path=args["owner_attempt_path"], envelope=args["envelope"])
    original = tmp_path / "workspace/runtime_output/stages/stage-3/producer/artisan_runtime"
    monkeypatch.setattr(continuation, "ORIGINAL_ROOT", str(original))
    descriptor = json.loads((runtime / staged["descriptor"]["path"]).read_text())
    # The provider sees the descriptor as sealed at selection; rebind only the test root.
    descriptor["original_runtime_root"] = str(original)
    descriptor = seal(descriptor, "adoption_digest")
    return {"staged": staged, "runtime": runtime, "root": original, "descriptor": descriptor,
            "archive": runtime / staged["runtime_archive"]["path"]}


def test_staging_copies_only_the_sealed_transport(staged):
    assert staged["staged"]["descriptor"]["path"] == "input/artisan_continuation/descriptor.json"
    assert staged["staged"]["runtime_archive"]["path"] == "input/artisan_continuation/retained_runtime.zip"
    assert staged["staged"]["verified_lineage"]["successor_run_id"] == SUCCESSOR_RUN


def _restore(staged):
    return continuation.restore_artisan_continuation(
        descriptor=staged["descriptor"], archive_path=staged["archive"], runtime_root=staged["root"],
        verified_lineage=staged["staged"]["verified_lineage"], successor_run_id=SUCCESSOR_RUN)


def test_restore_then_adopt_splits_the_cumulative_cap(staged):
    assert _restore(staged) is True
    upstream = staged["descriptor"]["source_upstream_run_id"]
    assert (staged["root"] / f"workspace/data/cache/runs/{upstream}/pipeline_state.json").is_file()
    assert not (staged["root"] / "blueprint_usage.json").exists()  # the successor's meter starts empty
    record = continuation.adopt_artisan_continuation(
        descriptor=staged["descriptor"], runtime_root=staged["root"], request=request(SUCCESSOR_RUN, 25.0),
        cumulative_cap=25.0, attempt_allocation=20.0)
    assert record["carried_cost_usd"] == 13.76 and record["increment_usd"] == 11.24
    assert record["upstream_cap_usd"] == 25.0 and record["new_spend_authorized"] is False
    # The attempt's own allocation bounds new spend even when the cumulative cap has more room.
    smaller = continuation.adopt_artisan_continuation
    with pytest.raises(continuation.ArtisanContinuationError, match="adoption_record_changed"):
        smaller(descriptor=staged["descriptor"], runtime_root=staged["root"], request=request(SUCCESSOR_RUN, 25.0),
                cumulative_cap=25.0, attempt_allocation=5.0)
    # A same-run retry of the successor finds its sealed record and does not restore again.
    assert _restore(staged) is False


def test_the_attempt_allocation_bounds_the_increment(staged):
    _restore(staged)
    record = continuation.adopt_artisan_continuation(
        descriptor=staged["descriptor"], runtime_root=staged["root"], request=request(SUCCESSOR_RUN, 25.0),
        cumulative_cap=25.0, attempt_allocation=5.0)
    assert record["increment_usd"] == 5.0 and record["upstream_cap_usd"] == 18.76


@pytest.mark.parametrize("cap", [13.76, 10.0])
def test_adoption_refuses_a_cap_the_carried_spend_reaches(staged, cap):
    _restore(staged)
    with pytest.raises(continuation.ArtisanContinuationError, match="cumulative_cap_exhausted"):
        continuation.adopt_artisan_continuation(
            descriptor=staged["descriptor"], runtime_root=staged["root"], request=request(SUCCESSOR_RUN, cap),
            cumulative_cap=cap, attempt_allocation=20.0)


def test_adoption_refuses_changed_scientific_inputs(staged):
    _restore(staged)
    with pytest.raises(continuation.ArtisanContinuationError, match="scientific_inputs_changed"):
        continuation.adopt_artisan_continuation(
            descriptor=staged["descriptor"], runtime_root=staged["root"],
            request=request(SUCCESSOR_RUN, 25.0, frame="sha256:" + "b" * 64),
            cumulative_cap=25.0, attempt_allocation=20.0)


def test_restore_refuses_an_occupied_root_and_a_changed_archive(staged):
    staged["root"].mkdir(parents=True)
    with pytest.raises(continuation.ArtisanContinuationError, match="original_root_occupied"):
        _restore(staged)
    staged["root"].rmdir()
    with zipfile.ZipFile(staged["archive"], "a") as archive:
        archive.writestr("workspace/injected.py", "print('x')")
    with pytest.raises(continuation.ArtisanContinuationError, match="archive_invalid"):
        _restore(staged)


def test_restore_refuses_a_descriptor_for_another_successor(staged):
    other = seal({**copy.deepcopy(staged["descriptor"]), "successor_run_id": "someone-else"}, "adoption_digest")
    with pytest.raises(continuation.ArtisanContinuationError, match="descriptor_invalid"):
        continuation.restore_artisan_continuation(
            descriptor=other, archive_path=staged["archive"], runtime_root=staged["root"],
            verified_lineage=staged["staged"]["verified_lineage"], successor_run_id=SUCCESSOR_RUN)


def test_semantic_request_drops_only_run_scope_and_the_cap():
    first, second = request(SOURCE_RUN, 15.0), request(SUCCESSOR_RUN, 25.0)
    assert first != second
    assert continuation.semantic_request(first) == continuation.semantic_request(second)
    changed = request(SUCCESSOR_RUN, 25.0)
    changed["configuration"]["replacement_identity"]["id"] = "other"
    assert continuation.semantic_request(first) != continuation.semantic_request(changed)


def test_validation_reuses_the_partial_astra_owner_chain_under_its_own_schema(staged):
    value = json.loads((staged["runtime"] / staged["staged"]["transport"]["path"]).read_text())
    descriptor = json.loads((staged["runtime"] / staged["staged"]["descriptor"]["path"]).read_text())
    continuation.validate_artisan_transport(value, descriptor)
    with pytest.raises(transport.PartialAstraTransportError, match="selection_invalid"):
        transport.validate_transport(value, descriptor)  # the Astra schema never accepts it


def test_an_artisan_successor_selects_its_continuation_before_any_astra_transport(source, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    source["build"]()
    args = source["args"]()
    monkeypatch.setenv(intake.ROOT_ENV, str(args["intent_root"]))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT", str(args["launch_root"]))
    monkeypatch.setattr(transport, "select_partial_astra_source",
                        lambda **_: pytest.fail("an ARTISAN continuation was found; Astra must not select"))
    operations = {"scene_owner_attempt": str(args["owner_attempt_path"])}
    continuation.select_stage_three_continuation(operations, envelope=args["envelope"],
        activation_root=source["tmp"] / "activation", activation_request=None, created_storage_pins=None)
    assert Path(operations["artisan_continuation_selection"]).name == "selection.json"
    assert "partial_astra_successor_selection" not in operations
