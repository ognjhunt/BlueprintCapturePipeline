"""Construction's real probe wiring and immutable publication join, without a GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_robot_free_construction_probe import (
    SOURCE_ALIASES, TRACE_FIELD, prepare_construction_probe, published_construction_probe,
)
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as readiness
from blueprint_pipeline import task_evaluation_scene_configuration_native_import_driver as driver
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH, STATIC_QUALIFICATION_CONTRACT_PATH,
)
from tests.test_rigid_robot_free_probe import FlatSupport, rigid_case
from tests.test_task_evaluation_rigid_relocation_native_adapter import _rewrite
from tests.test_task_evaluation_scene_configuration_native_import_driver import _environment, _observed


def _construction(tmp_path: Path):
    source = tmp_path / "sources"
    source.mkdir()
    revision, references, docs, _spec = rigid_case(source, strategy="pick_and_place")
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n")
    import hashlib
    native = copy.deepcopy(docs[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH])
    native["asset_digest"] = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    _rewrite(tmp_path=source, configured=revision, references=references,
        contract_path=NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH, document=native)
    envelope = {"run_id": revision["configuration_run_id"], "team_namespace": revision["team_namespace"],
        "request": {"scene": {"identity": revision["scene_identity"]}, "task": {"strategy": "pick_and_place"}},
        "materialized_references": [dict(references[key], contract_path=alias)
            for key, alias in SOURCE_ALIASES.items()]}
    prepared = prepare_construction_probe(envelope=envelope, asset_path=asset,
        static_path=Path(references[STATIC_QUALIFICATION_CONTRACT_PATH]["materialized_path"]),
        native_import=native)
    raw = record_rigid_robot_free_probe(scene=FlatSupport(prepared["plan"]["start_scoring_pose_world"]),
        plan=prepared["plan"], binding=prepared["binding"])
    native[TRACE_FIELD] = raw
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    _rewrite(tmp_path=source, configured=revision, references=references,
        contract_path=NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH, document=native)
    documents = {key: Path(row["materialized_path"]) for key, row in references.items()}
    task = readiness.rigid_task_for_revision(revision=revision, documents=documents)
    return revision, documents, task, raw, envelope


def test_published_native_receipt_supplies_the_automatic_rigid_probe(tmp_path: Path):
    revision, documents, task, raw, _envelope = _construction(tmp_path)
    projected = published_construction_probe(revision=revision, documents=documents, task=task)
    assert raw["binding"]["configured_scene_revision_digest"] is None
    assert projected["construction_trace_digest"] == raw["trace_digest"]
    assert projected["samples"] == raw["samples"]
    assert projected["binding"]["configured_scene_revision_digest"] == revision["revision_digest"]
    receipt = readiness.score_rigid_robot_free_readiness(task_spec=task["task_spec"], physics_probe=projected,
        configured_scene_revision_digest=revision["revision_digest"],
        replacement_asset_digest=task["replacement_asset_digest"])
    assert receipt["status"] == "passed"
    # Discovery needs only the published native receipt; no manual trace file.
    from blueprint_pipeline import task_evaluation_configured_controls_deferred_inputs as inputs
    refs = inputs._document_references(revision)
    store = {refs[key]["uri"]: path.read_bytes() for key, path in documents.items()}
    receipt = readiness.rigid_robot_free_readiness_for_revision(revision=revision,
        output_root=tmp_path / "retained", physics_probe=None, fetcher=lambda ref: store[ref["uri"]])
    assert receipt["status"] == "passed"


@pytest.mark.parametrize("changed", ["document", "asset", "run", "trace"])
def test_construction_trace_cannot_be_adopted_for_changed_source(tmp_path: Path, changed: str):
    revision, documents, task, _raw, _envelope = _construction(tmp_path)
    if changed == "document":
        path = documents[STATIC_QUALIFICATION_CONTRACT_PATH]
        path.write_text(path.read_text() + " ")
    elif changed == "asset":
        task["replacement_asset_digest"] = "sha256:" + "f" * 64
    elif changed == "run":
        revision["configuration_run_id"] += "-other"
    else:
        path = documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH]
        native = json.loads(path.read_text())
        native[TRACE_FIELD]["samples"][0]["pose"][0] += 1
        path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="rigid_construction_probe_"):
        published_construction_probe(revision=revision, documents=documents, task=task)


def test_native_import_automatically_requests_and_retains_the_rigid_probe(tmp_path: Path):
    environment = _environment(tmp_path)
    input_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    production = json.loads(input_path.read_text())
    production["construction_envelope"] = {"request": {"task": {"strategy": "pick_and_place"}}}
    input_path.write_text(json.dumps(production))
    calls = []

    def run(*, rigid_probe_factory, observation_consumer, **kwargs):
        # The production runner owns the live SimulationApp. Test that the
        # component supplies its factory and retains its returned evidence.
        calls.append(rigid_probe_factory)
        observed = _observed()
        observed[TRACE_FIELD] = {"status": "failed", "error": "simulator probe failure"}
        return observation_consumer(observed)

    result = driver.execute_native_import_component(environment=environment, native_runner=run)
    assert len(calls) == 1 and callable(calls[0])
    native = json.loads(Path(result["artifacts"][0]["path"]).read_text())
    assert native[TRACE_FIELD]["error"] == "simulator probe failure"
    assert result["provider_mutations_performed"] == 0


def test_native_component_factory_derives_the_real_source_bound_probe(tmp_path: Path):
    import hashlib
    revision, documents, _task, expected_trace, envelope = _construction(tmp_path)
    environment = _environment(tmp_path)
    input_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    production = json.loads(input_path.read_text())
    production["construction_envelope"] = envelope
    production["configuration"]["replacement_identity"] = revision["replacement"]["identity"]
    input_path.write_text(json.dumps(production))
    dependencies = []
    for role, path in (("statically_qualified_replacement_asset", tmp_path / "asset.usda"),
                       ("static_qualification_receipt", documents[STATIC_QUALIFICATION_CONTRACT_PATH])):
        dependencies.append({"role": role, "path": str(path), "size_bytes": path.stat().st_size,
            "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()})
    Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"]).write_text(
        json.dumps([{"output_artifacts": dependencies}]))

    def run(*, rigid_probe_factory, observation_consumer, **kwargs):
        observed = _observed()
        prepared = rigid_probe_factory(observed)
        assert prepared["plan"] == expected_trace["plan"]
        observed[TRACE_FIELD] = record_rigid_robot_free_probe(
            scene=FlatSupport(prepared["plan"]["start_scoring_pose_world"]),
            plan=prepared["plan"], binding=prepared["binding"])
        return observation_consumer(observed)

    result = driver.execute_native_import_component(environment=environment, native_runner=run)
    native = json.loads(Path(result["artifacts"][0]["path"]).read_text())
    assert native[TRACE_FIELD]["status"] == "recorded"
    assert native[TRACE_FIELD]["binding"]["construction_binding"]["configuration_run_id"] == envelope["run_id"]


@pytest.mark.parametrize("factory_fails", [False, True])
def test_native_gpu_session_records_probe_before_closing(monkeypatch, factory_fails):
    from blueprint_pipeline import rigid_robot_free_probe_isaac as isaac
    events = []

    class App:
        def close(self):
            events.append("close")

    monkeypatch.setattr(driver, "_bind_isaac_runtime_environment", lambda: None)
    monkeypatch.setattr(driver, "_import_simulation_app", lambda: lambda config: App())
    monkeypatch.setattr(driver, "_observe_isaac_runtime_identity", lambda app: {"engine_version": "6.0.1"})
    monkeypatch.setattr(driver, "_one_native_settle", lambda **kwargs: events.append("settle") or {})
    monkeypatch.setattr(isaac, "record_in_active_isaac",
        lambda prepared: events.append("probe") or {"status": "recorded"})

    def factory(observation):
        events.append("prepare")
        assert len(observation["repeats"]) == 3
        if factory_fails:
            raise ValueError("invalid task source")
        return {"prepared": True}

    observed = driver._run_native_import(asset_path=Path("unused"),
        required_checks={"state_digest_repeat_count": 3, "gravity_settle_seconds": 3},
        rigid_probe_factory=factory, observation_consumer=lambda observation: observation)
    assert events == ["settle"] * 3 + ["prepare"] + ([] if factory_fails else ["probe"]) + ["close"]
    assert observed[TRACE_FIELD]["status"] == ("failed" if factory_fails else "recorded")


@pytest.mark.parametrize("mutation", ["modified", "symlink"])
def test_shared_probe_preparation_rechecks_asset_bytes(tmp_path: Path, mutation):
    from blueprint_pipeline.rigid_robot_free_probe_isaac import prepare_probe_scene, RigidRobotFreeProbeIsaacError
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import SUPPORT_PLANE_CONTRACT_PATH
    revision, documents, task, _raw, _envelope = _construction(tmp_path)
    asset = tmp_path / "asset.usda"
    if mutation == "modified":
        asset.write_text("changed asset")
    else:
        linked = tmp_path / "linked.usda"
        linked.symlink_to(asset)
        asset = linked
    with pytest.raises(RigidRobotFreeProbeIsaacError, match="not_the_qualified_asset"):
        prepare_probe_scene(task=task, replacement_asset=asset,
            support=json.loads(documents[SUPPORT_PLANE_CONTRACT_PATH].read_text()), destination=None,
            revision_digest=revision["revision_digest"])


def test_destination_readback_requires_actual_destination_contact_after_pickup():
    from blueprint_pipeline.rigid_robot_free_probe_isaac import IsaacRigidProbeScene
    from tests.test_rigid_robot_free_probe_isaac import _Body

    class PhysX:
        def get_physx_interface(self):
            return self

        def get_rigidbody_transformation(self, path):
            return {"ret_val": True, "position": [1, 2, 3] if path == "subject" else [4, 5, 6],
                    "rotation": [0, 0, 0, 1]}

    contact = {"source_pairs": {("source", "subject")}, "destination_pairs": set()}
    scene = IsaacRigidProbeScene(simulation=None, omni_physx=PhysX(), body=_Body(), body_path="subject",
        offset=[0, 0, 0], contact=contact, destination_body_path="destination")
    assert scene.read_sample()["support_contact_active"] is True
    scene.set_scoring_pose([1, 2, 3, 0, 0, 0, 1])
    assert scene.read_sample()["support_contact_active"] is False
    contact["destination_pairs"].add(("destination", "subject"))
    assert scene.read_sample()["support_contact_active"] is True
    assert scene.read_sample()["destination_pose_world"] == [4, 5, 6, 0, 0, 0, 1]


def test_probe_shutdown_releases_subscription_after_simulator_failure(tmp_path: Path, monkeypatch):
    from blueprint_pipeline import rigid_robot_free_probe_isaac as isaac
    from tests.test_rigid_robot_free_probe_isaac import _Body
    revision, documents, task, _raw, _envelope = _construction(tmp_path)
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import SUPPORT_PLANE_CONTRACT_PATH
    prepared = isaac.prepare_probe_scene(task=task, replacement_asset=tmp_path / "asset.usda",
        support=json.loads(documents[SUPPORT_PLANE_CONTRACT_PATH].read_text()), destination=None,
        revision_digest=revision["revision_digest"])
    events = []

    class Simulation:
        def step(self, render=False):
            raise RuntimeError("readback failed")

        def stop(self):
            events.append("stop")
            raise RuntimeError("stop failed")

    class Subscription:
        def unsubscribe(self):
            events.append("unsubscribe")

    monkeypatch.setattr(isaac, "_build_stage", lambda value: {"simulation": Simulation(),
        "omni_physx": None, "body": _Body(), "body_path": "subject", "contact": {},
        "subscription": Subscription()})
    trace = isaac.record_in_active_isaac(prepared)
    assert trace["status"] == "failed"
    assert trace["shutdown_errors"] == ["RuntimeError"]
    assert events == ["stop", "unsubscribe"]


def test_destination_support_paths_follow_the_actual_usd_default_prim(tmp_path: Path):
    from pxr import Usd, UsdGeom, UsdPhysics
    from blueprint_pipeline.rigid_robot_free_probe_isaac import destination_probe_support_paths

    asset = tmp_path / "tray.usda"
    source = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(source, "/Tray").GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/DestinationPlacement/Destination", "Xform").GetReferences().AddReference(str(asset))
    paths = destination_probe_support_paths(source_root="/Tray", source_paths=["/Tray"])
    assert paths == {"/World/DestinationPlacement/Destination"}
    assert all(stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI) for path in paths)
    with pytest.raises(RuntimeError, match="destination_support_body_invalid"):
        destination_probe_support_paths(source_root="/Tray", source_paths=["/Asset"])
