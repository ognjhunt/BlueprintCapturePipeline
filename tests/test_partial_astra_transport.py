"""Hermetic ownership/byte transport checks; real eae metadata is development-only."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_partial_astra_transport as transport
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_owner_attempt_profiles import make_owner_attempt_record

META = json.loads((Path(__file__).parent / "fixtures/partial_astra_eae_metadata.json").read_text())
SOURCE_RUN = META["request"]["run_id"]
SUCCESSOR_RUN = "same-scene-fresh-successor-scene-configuration"
SHA = "sha256:" + "a" * 64
COMMIT = "a" * 40


def seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "reserve_control_plane_disk", lambda *_a, **_k: nullcontext())
    intents, launches = tmp_path / "scene-intents", tmp_path / "task-evaluation-launch-runs"
    intent = seal({"intent_id": "scene-owner", "authenticated_issuer": "blueprint-webapp",
                   "request": {"owner": {"user_id": "owner", "organization_id": "org"},
                               "task": {"task_id": "pick-one", "reuse_completed_stages": True}}}, "intent_digest")
    write(intents / "scene-owner/intent.json", intent)
    preparation = {"scene_intent_digest": intent["intent_digest"], "scene": {"identity": {"id": "840938"}},
                   "task": {"identity": {"id": "pick-one"}}, "team_namespace": "new-namespace",
                   "run_id": SUCCESSOR_RUN, "expected_production_commit": COMMIT}
    attempts = []
    for attempt_id in ("old", "new"):
        attempt = seal({"intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
                        "attempt_id": attempt_id, "source_commit": COMMIT, "runtime_digest": SHA,
                        "input_digest": canonical_digest(preparation) if attempt_id == "new" else SHA, "provider": "vast"}, "attempt_digest")
        write(intents / "scene-owner/attempts" / (attempt_id + ".json"), attempt)
        attempts.append(attempt)
    current = make_owner_attempt_record(owner_fields=bind_scene_attempt(attempts[1]), phase="scene_configuration",
        team_namespace="new-namespace", scene_id="840938", task_id="pick-one", runtime_source_bundle_digest=SHA)
    owner_path = write(tmp_path / "current-owner.json", current)
    source = launches / (SOURCE_RUN + "-activation-auto-launch")
    profile = {**bind_scene_attempt(attempts[0]), "task_evaluation_run": {"run_mode": "scene_configuration",
               "team_namespace": "old-namespace", "scene_id": "840938", "task_id": "pick-one",
               "configuration_run_id": SOURCE_RUN}}
    write(source / "launch_profile.json", profile)
    write(source / "launch_receipt.json", {"status": "blocked", "launch_id": source.name})
    zero = seal({"launch_id": source.name, "status": "provider_zero_confirmed", "provider_zero_verified": True,
                 "continuing_spend_from_this_run": False, "blockers": []}, "provider_zero_receipt_digest")
    write(source / "post_teardown_provider_zero_receipt.json", zero)
    archive = source / transport.ARCHIVE_RELATIVE
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as packed:
        for name in sorted(transport.REQUIRED):
            value = META["request"] if name == "authoring/request.json" else (
                META["stage_source_binding"] if name == "stage_source_binding.json" else {"development_only": True})
            packed.writestr(transport.PREFIX + name, json.dumps(value))
        packed.writestr(transport.PREFIX + "authoring/cad/empty.stderr", b"")
        packed.writestr(transport.PREFIX + "official_openai_cost/receipt.json", '{"raw_keys_recorded":false}')
        packed.writestr(transport.PREFIX + "packaged_blender/reconstructible.bin", b"excluded")
    # Seal the same external download/result authority chain used by the live source.
    authority = seal({"bundle_sha256": SHA}, "authority_digest")
    authority_path = write(source / "source-authority.json", authority)
    bundle_path = write(source / "source-bundle.json", {"bundle_sha256": SHA})
    profile["source_commit"] = COMMIT
    profile["immutable_inputs"] = [
        {"name": "source_bundle_manifest", "path": str(bundle_path), "digest": transport._sha(bundle_path)},
        {"name": "scene_configuration_attempt_authority", "path": str(authority_path), "digest": transport._sha(authority_path)}]
    seal(profile, "profile_digest")
    write(source / "launch_profile.json", profile)
    launch = seal({"status": "blocked", "launch_id": source.name,
                   "launch_profile_digest": profile["profile_digest"]}, "receipt_digest")
    write(source / "launch_receipt.json", launch)
    zero.update(launch_profile_digest=profile["profile_digest"], receipt_digest=launch["receipt_digest"])
    write(source / "post_teardown_provider_zero_receipt.json", seal(zero, "provider_zero_receipt_digest"))
    digest, size = transport._sha(archive), archive.stat().st_size
    result = seal({"run_id": SOURCE_RUN, "source_commit": COMMIT, "bundle_sha256": SHA,
                   "authority_digest": authority["authority_digest"],
                   "provider_runtime_output_zip_path": str(archive), "provider_runtime_output_zip_sha256": digest,
                   "provider_runtime_output_remote_reference": {"status": "remote_verified", "digest": digest,
                       "readback_digest": digest, "size_bytes": size, "readback_size_bytes": size,
                       "full_byte_service_account_readback_passed": True}}, "result_digest")
    write(source / "allocator/scene-configuration-job/task_evaluation_scene_configuration_vast_result.v1.json", result)
    envelope = {"run_id": SUCCESSOR_RUN, "request": preparation, "team_namespace": "new-namespace",
                "expected_production_commit": COMMIT, "stage_configuration_references": [
        {"stage_id": "stage-3", "digest": META["stage_source_binding"]["configuration_sha256"]}]}
    return dict(owner_attempt_path=owner_path, envelope=envelope, output_root=tmp_path / "selected",
                intent_root=intents, launch_root=launches), source


def test_real_eae_metadata_selects_same_owner_successor_and_preserves_bytes(retained, tmp_path):
    args, source = retained
    assert META["source_archive_bytes"] == 676131591
    selection = transport.select_partial_astra_source(**args)
    value = json.loads(selection.read_text())
    descriptor = json.loads(Path(value["descriptor"]["path"]).read_text())
    assert descriptor["source_request_digest"] == META["request"]["request_digest"]
    assert descriptor["source_stage_binding_digest"] == META["stage_source_binding"]["binding_digest"]
    assert descriptor["original_runtime_root"] == transport.ORIGINAL_ROOT
    assert value["verified_lineage"]["stable_intent_id"] == "scene-owner"
    assert value["verified_lineage"]["source_run_id"] == SOURCE_RUN
    assert value["verified_lineage"]["successor_run_id"] == SUCCESSOR_RUN
    with zipfile.ZipFile(Path(value["runtime_archive"]["path"])) as archive:
        assert set(archive.namelist()) == transport.REQUIRED | {"official_openai_cost/receipt.json", "authoring/cad/empty.stderr"}
        assert json.loads(archive.read("authoring/request.json")) == META["request"]
    staged = transport.stage_partial_astra_transport(selection_path=selection, runtime=tmp_path / "runtime",
                                                     successor_run_id=SUCCESSOR_RUN, owner_attempt_path=args["owner_attempt_path"], envelope=args["envelope"])
    assert staged["descriptor"]["path"] == "input/partial_astra/descriptor.json"
    assert transport.select_partial_astra_source(**args) == selection
    assert source.exists()


@pytest.mark.parametrize("change", ["owner", "scene", "task", "configuration", "provider_zero", "canonical_attempt"])
def test_changed_source_identity_is_not_selected(retained, change):
    args, source = retained
    path = source / "launch_profile.json"
    value = json.loads(path.read_text())
    if change == "owner":
        value["scene_attempt_binding"]["intent_digest"] = "sha256:" + "f" * 64
    elif change in {"scene", "task"}:
        value["task_evaluation_run"][change + "_id"] = "foreign"
    elif change == "configuration":
        args["envelope"]["stage_configuration_references"][0]["digest"] = SHA
    elif change == "provider_zero":
        zero_path = source / "post_teardown_provider_zero_receipt.json"
        zero = json.loads(zero_path.read_text())
        zero["launch_id"] = "foreign"
        write(zero_path, seal(zero, "provider_zero_receipt_digest"))
    else:
        attempt_path = args["intent_root"] / "scene-owner/attempts/old.json"
        attempt = json.loads(attempt_path.read_text())
        attempt["runtime_digest"] = "sha256:" + "d" * 64
        write(attempt_path, seal(attempt, "attempt_digest"))
    write(path, value)
    assert transport.select_partial_astra_source(**args) is None
    assert not args["output_root"].exists()


@pytest.mark.parametrize("change", ["source_archive", "runtime_archive", "canonical_owner"])
def test_source_reopened_before_bundle_sealing(retained, tmp_path, change):
    args, source = retained
    selection = transport.select_partial_astra_source(**args)
    value = json.loads(selection.read_text())
    if change == "canonical_owner":
        path = args["intent_root"] / "scene-owner/intent.json"
        owner = json.loads(path.read_text())
        owner["request"]["owner"]["user_id"] = "other"
        write(path, seal(owner, "intent_digest"))
    else:
        path = Path(value[change]["path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        transport.stage_partial_astra_transport(selection_path=selection, runtime=tmp_path / "runtime",
                                                 successor_run_id=SUCCESSOR_RUN, owner_attempt_path=args["owner_attempt_path"], envelope=args["envelope"])


def test_descriptor_cannot_claim_its_own_authority(retained):
    args, _ = retained
    selection = transport.select_partial_astra_source(**args)
    value = json.loads(selection.read_text())
    descriptor = json.loads(Path(value["descriptor"]["path"]).read_text())
    descriptor["owner_intent_lineage"]["owner_id"] = "attacker"
    seal(descriptor, "adoption_digest")
    with pytest.raises(ValueError, match="descriptor_lineage_invalid"):
        transport.validate_transport(value, descriptor)


@pytest.mark.parametrize("member", ["../escape", transport.PREFIX + "authoring/.env"])
def test_bad_archive_member_refuses_without_leaving_partial_transport(retained, member):
    args, source = retained
    with zipfile.ZipFile(source / transport.ARCHIVE_RELATIVE, "a") as archive:
        archive.writestr(member, b"not admitted")
    with pytest.raises(ValueError, match="known_partial_source_unusable"):
        transport.select_partial_astra_source(**args)
    assert not args["output_root"].exists()
    assert (args["output_root"].parent / "partial_astra_successor_rejections.json").is_file()


def test_newer_foreign_scene_does_not_shadow_same_owner_source(retained):
    args, source = retained
    for index in range(20):
        foreign = args["launch_root"] / f"foreign-{index}"
        profile = copy.deepcopy(json.loads((source / "launch_profile.json").read_text()))
        profile["task_evaluation_run"]["scene_id"] = "other"
        write(foreign / "launch_profile.json", profile)
        write(foreign / "post_teardown_provider_zero_receipt.json", {})
    assert transport.select_partial_astra_source(**args) is not None


@pytest.mark.parametrize("changed", [None, "runtime_archive", "descriptor", "owner", "scene", "task", "intent"])
def test_provider_bundle_preflight_and_hydration_bind_partial_data(retained, tmp_path, changed):
    from tests.test_task_evaluation_scene_configuration_bundle import _build
    from blueprint_pipeline.task_evaluation_scene_configuration_provider_preflight import scene_configuration_bundle_contract
    import runpy

    args, _ = retained
    selection = transport.select_partial_astra_source(**args)
    runtime = tmp_path / "staged" / "provider_runtime"
    partial = transport.stage_partial_astra_transport(selection_path=selection, runtime=runtime,
        successor_run_id=SUCCESSOR_RUN, owner_attempt_path=args["owner_attempt_path"], envelope=args["envelope"])
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    receipt = _build(baseline, "bundle")
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        payloads = {i.filename: archive.read(i) for i in archive.infolist() if not i.is_dir()}
    for path in runtime.rglob("*"):
        if path.is_file():
            payloads["provider_runtime/" + path.relative_to(runtime).as_posix()] = path.read_bytes()
    envelope_name = "provider_runtime/input/portable_construction_envelope.v1.json"
    manifest_name = "provider_runtime/task_evaluation_scene_configuration_provider_bundle.v1.json"
    envelope = json.loads(payloads[envelope_name])
    envelope.update(run_id=SUCCESSOR_RUN, partial_astra_successor=partial,
                    request=copy.deepcopy(args["envelope"]["request"]), team_namespace="new-namespace",
                    expected_production_commit=COMMIT)
    if changed in {"scene", "task"}:
        envelope["request"][changed]["identity"]["id"] = "foreign"
    if changed == "intent":
        envelope["request"]["scene_intent_digest"] = SHA
    if changed == "owner":
        partial["verified_lineage"]["owner_id"] = "other"
    seal(envelope, "envelope_digest")
    manifest = json.loads(payloads[manifest_name])
    manifest.update(run_id=SUCCESSOR_RUN, portable_construction_envelope_digest=envelope["envelope_digest"],
                    partial_astra_successor_digest=canonical_digest(partial))
    seal(manifest, "manifest_digest")
    payloads[envelope_name] = json.dumps(envelope).encode()
    payloads[manifest_name] = json.dumps(manifest).encode()
    if changed in {"descriptor", "runtime_archive"}:
        payloads["provider_runtime/" + partial[changed]["path"]] += b"changed"
    bundle = tmp_path / "candidate.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(bundle) as archive:
        _, _, blockers = scene_configuration_bundle_contract(archive)
        if changed is not None:
            assert blockers
            return
        assert blockers == []
        hydrated_root = tmp_path / "hydrated"
        archive.extractall(hydrated_root)
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/task_evaluation_scene_configuration_provider_runner.py"))
    hydrated = namespace["_hydrate_envelope"](hydrated_root / "provider_runtime", envelope)
    reference = hydrated["partial_astra_successor"]["descriptor"]
    assert Path(reference["materialized_path"]).is_file()
    assert reference["digest"] == partial["descriptor"]["digest"]


def test_bundle_refuses_selection_for_another_actual_successor_owner(retained, tmp_path):
    args, _ = retained
    selection = transport.select_partial_astra_source(**args)
    owner = json.loads(args["owner_attempt_path"].read_text())
    owner["scene_id"] = "different"
    changed = write(tmp_path / "other-owner.json", seal(owner, "owner_attempt_digest"))
    with pytest.raises(ValueError, match="construction_owner_binding_changed"):
        transport.stage_partial_astra_transport(selection_path=selection, runtime=tmp_path / "runtime",
            successor_run_id=SUCCESSOR_RUN, owner_attempt_path=changed, envelope=args["envelope"])


def test_owner_context_carries_selection_through_real_bundle_argv(monkeypatch, tmp_path):
    from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
    from tests.test_task_evaluation_launch_activation_worker import test_activation_builds_robot_neutral_scene_configuration_context
    from scripts import prepare_paid_lane_launch as prep

    owner_path, selection = tmp_path / "owner.json", tmp_path / "selection.json"
    seen = []
    monkeypatch.setattr(worker, "_owner_attempt", lambda operations, *_a: operations.update(scene_owner_attempt=str(owner_path)))

    def select(**kwargs):
        assert kwargs["owner_attempt_path"] == str(owner_path)
        seen.append(kwargs)
        return selection

    monkeypatch.setattr(transport, "select_partial_astra_source", select)
    original = worker._build_scene_configuration_context

    def build(**kwargs):
        context = original(**kwargs)
        operations = context["operations"]
        assert operations["partial_astra_successor_selection"] == str(selection)
        step = prep.LANES["task_evaluation_scene_configuration"][0]
        argv = prep._step_argv(step, operations)
        assert argv[argv.index("--partial-astra-successor-selection") + 1] == str(selection)
        assert argv[argv.index("--scene-owner-attempt") + 1] == str(owner_path)
        return context

    monkeypatch.setattr(worker, "_build_scene_configuration_context", build)
    test_activation_builds_robot_neutral_scene_configuration_context(tmp_path)
    assert len(seen) == 2


@pytest.mark.parametrize("changed", ["truncated", "missing"])
def test_authenticated_source_archive_damage_never_falls_back_to_fresh_cad(retained, changed):
    args, source = retained
    archive = source / transport.ARCHIVE_RELATIVE
    if changed == "truncated":
        archive.write_bytes(archive.read_bytes()[:64])
    else:
        archive.unlink()
    with pytest.raises(ValueError, match="known_partial_source_unusable"):
        transport.select_partial_astra_source(**args)
    refusal = json.loads((args["output_root"].parent / "partial_astra_successor_rejections.json").read_text())
    assert refusal["status"] == "blocked"
    assert refusal["rejected_sources"]
    assert not args["output_root"].exists()


@pytest.mark.parametrize("changed", ["scene", "task", "intent", "run"])
def test_actual_construction_request_must_match_reserved_owner_input(retained, changed):
    args, _ = retained
    request = args["envelope"]["request"]
    if changed in {"scene", "task"}:
        request[changed]["identity"]["id"] = "another"
    elif changed == "intent":
        request["scene_intent_digest"] = SHA
    else:
        request["run_id"] = "another"
    with pytest.raises(ValueError, match="construction_owner_binding_changed"):
        transport.select_partial_astra_source(**args)
    assert not args["output_root"].exists()
