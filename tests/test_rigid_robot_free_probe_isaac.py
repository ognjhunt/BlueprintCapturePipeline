"""The Isaac rigid robot-free probe is bound to the revision before any simulator starts.

Beta journey step: L5 team policy evaluation. Preparation is hermetic and
runs in the fast lane. The Isaac run itself (one SimulationApp per process,
so each scene runs through the module entrypoint in its own interpreter) is in
the slow/gpu lane and skips where Isaac Sim is not installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_robot_free_probe_isaac import (
    SCENE_CONTEXT,
    IsaacRigidProbeScene,
    RigidRobotFreeProbeIsaacError,
    prepare_isaac_probe,
)
from blueprint_pipeline.rigid_robot_free_probe_plan import physics_plan, probe_context
from blueprint_pipeline.robot_free_readiness_retention import TRACE_FILENAME
from blueprint_pipeline.task_evaluation_configured_controls_deferred_inputs import _document_name
from tests.test_rigid_robot_free_probe import rigid_case
from tests.test_task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT, SUPPORT, _rewrite

CUBE_EDGE_M = 0.127638  # twice the fixture's qualified centre-of-mass height above the support


def _bound_case(tmp_path: Path, asset: Path, *, support_max_x: float | None = None) -> dict[str, Any]:
    """The rigid fixture, with its native import qualifying ``asset``'s exact bytes."""

    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    configured, references, docs, _spec = rigid_case(sources)
    native = copy.deepcopy(docs[NATIVE_IMPORT])
    native["asset_digest"] = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    _rewrite(tmp_path=sources, configured=configured, references=references, contract_path=NATIVE_IMPORT,
             document=native)
    if support_max_x is not None:
        support = copy.deepcopy(docs[SUPPORT])
        support["bounds_max_xyz_m"][0] = support_max_x
        _rewrite(tmp_path=sources, configured=configured, references=references, contract_path=SUPPORT,
                 document=support)
    store = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()}
    return {"configured": configured, "references": references, "store": store,
            "fetcher": lambda reference: store[reference["uri"]], "asset_digest": native["asset_digest"]}


def test_preparation_binds_the_exact_qualified_asset_task_and_support(tmp_path: Path) -> None:
    asset = tmp_path / "replacement.usda"
    asset.write_text("#usda 1.0\n")
    case = _bound_case(tmp_path, asset)
    prepared = prepare_isaac_probe(revision=case["configured"], replacement_asset=asset,
                                   output_root=tmp_path / "probe", fetcher=case["fetcher"])
    documents = rigid.fetch_revision_documents(revision=case["configured"], output_root=tmp_path / "probe")
    spec = rigid.rigid_task_for_revision(revision=case["configured"], documents=documents)["task_spec"]
    assert prepared["plan"] == physics_plan(probe_context(spec))
    assert prepared["binding"] == {
        "configured_scene_revision_digest": case["configured"]["revision_digest"],
        "task_spec_digest": canonical_digest(spec), "replacement_asset_digest": case["asset_digest"],
        "scene_context": SCENE_CONTEXT, "scene_collisions_modeled": False}
    assert prepared["support_bounds_world_m"] == {"minimum": [2.5, -9.5, 0.0], "maximum": [4.5, -1.4, 0.7545]}
    assert prepared["scoring_offset_m"] == [0.0, 0.0, 0.063819]
    assert prepared["root_pose_world"][2] == pytest.approx(0.7545)
    other = tmp_path / "other.usda"
    other.write_text("#usda 1.0\n# not the qualified bytes\n")
    with pytest.raises(RigidRobotFreeProbeIsaacError, match="not_the_qualified_asset"):
        prepare_isaac_probe(revision=case["configured"], replacement_asset=other, output_root=tmp_path / "probe",
                            fetcher=case["fetcher"])


class _Body:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[float]]] = []

    def set_world_pose(self, *, position: Any, orientation: Any) -> None:
        self.calls.append(("pose", [*map(float, position), *map(float, orientation)]))

    def set_linear_velocity(self, value: Any) -> None:
        self.calls.append(("linear", list(map(float, value))))

    def set_angular_velocity(self, value: Any) -> None:
        self.calls.append(("angular", list(map(float, value))))


class _PhysX:
    def __init__(self, contact: dict[str, bool], events: list[str]) -> None:
        self.contact, self.events = contact, events

    def get_physx_interface(self) -> "_PhysX":
        return self

    def get_rigidbody_transformation(self, path: str) -> dict[str, Any]:
        assert path == "/World/Placement/Replacement"
        half = 0.5 ** 0.5  # body rotated a quarter turn about x
        return {"ret_val": True, "position": [1.0, 2.0, 3.0], "rotation": [half, 0.0, 0.0, half]}

    def step(self, render: bool = False) -> None:  # the SimulationContext stand-in
        event = self.events.pop(0)
        if event:
            self.contact[event] = True


def test_the_isaac_seam_maps_the_scoring_frame_and_tracks_support_contact() -> None:
    contact = {"found": False, "lost": False, "touching": False}
    physx, body = _PhysX(contact, ["found", "", "lost"]), _Body()
    scene = IsaacRigidProbeScene(simulation=physx, omni_physx=physx, body=body,
                                 body_path="/World/Placement/Replacement", offset=[0.0, 0.0, 0.1], contact=contact)
    half = 0.5 ** 0.5
    scene.set_scoring_pose([1.0, 2.0, 3.0, half, 0.0, 0.0, half])
    # The body root sits at the scoring point minus the rotated offset; Isaac core takes wxyz.
    position_and_wxyz = body.calls[0][1]
    assert position_and_wxyz == pytest.approx([1.0, 2.1, 3.0, half, half, 0.0, 0.0])
    assert body.calls[1:] == [("linear", [0.0, 0.0, 0.0]), ("angular", [0.0, 0.0, 0.0])]
    touching = []
    for _ in range(3):  # found, nothing reported (persisting), lost
        scene.step()
        touching.append(scene.read_sample()["support_contact_active"])
    assert touching == [True, True, False]
    assert scene.read_sample()["task_scoring_pose_world"] == pytest.approx([1.0, 1.9, 3.0, half, 0.0, 0.0, half])


def _cube_asset(path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, Sdf.Path("/Asset"))
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(0.5)
    cube = UsdGeom.Cube.Define(stage, Sdf.Path("/Asset/Collision"))
    cube.CreateSizeAttr(CUBE_EDGE_M)
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CUBE_EDGE_M / 2.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    return path


def _isaac_receipt(tmp_path: Path, **support: Any) -> dict[str, Any]:
    asset = _cube_asset(tmp_path / "replacement.usda")
    case = _bound_case(tmp_path, asset, **support)
    output = tmp_path / "probe"
    revision_path = tmp_path / "configured_scene_revision.v1.json"
    revision_path.write_text(json.dumps(case["configured"]))
    (output / "documents").mkdir(parents=True)
    for contract_path, reference in case["references"].items():
        (output / "documents" / _document_name(contract_path)).write_bytes(case["store"][reference["uri"]])
    subprocess.run([sys.executable, "-m", "blueprint_pipeline.rigid_robot_free_probe_isaac", "--revision",
                    str(revision_path), "--replacement-asset", str(asset), "--output-root", str(output)],
                   check=False, timeout=900, env={**os.environ, "PYTHONPATH": os.pathsep.join(
                       [str(Path(__file__).resolve().parents[1] / "src"), os.environ.get("PYTHONPATH", "")])})
    trace = json.loads((output / TRACE_FILENAME).read_text())
    assert trace["status"] == "recorded", trace.get("error")
    documents = rigid.fetch_revision_documents(revision=case["configured"], output_root=output)
    task = rigid.rigid_task_for_revision(revision=case["configured"], documents=documents)
    return rigid.score_rigid_robot_free_readiness(
        task_spec=task["task_spec"], physics_probe=trace,
        configured_scene_revision_digest=case["configured"]["revision_digest"],
        replacement_asset_digest=task["replacement_asset_digest"])


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.external_runtime
def test_isaac_holds_a_box_set_down_on_the_registered_support_and_the_scene_is_ready(tmp_path: Path) -> None:
    pytest.importorskip("isaacsim")
    receipt = _isaac_receipt(tmp_path)
    assert receipt["status"] == "passed", receipt["checks"]
    assert receipt["physics"]["direct_placement"]["outcome"] == "pushed_and_settled"
    assert receipt["proof_boundary"]["scene_context"] == SCENE_CONTEXT


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.external_runtime
def test_isaac_drops_a_box_set_down_beyond_the_support_edge_and_the_scene_is_not_ready(tmp_path: Path) -> None:
    pytest.importorskip("isaacsim")
    receipt = _isaac_receipt(tmp_path, support_max_x=3.05)
    assert receipt["status"] == "failed"
    assert receipt["checks"]["direct_placement_success"] is False
    assert receipt["checks"]["zero_action_not_success"] is True
