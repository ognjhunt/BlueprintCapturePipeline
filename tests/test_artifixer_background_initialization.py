import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.artifixer_background_initialization import (
    _raw_rows,
    local_foreground_mask,
    sample_registered_top_surface,
    write_initialization,
)
from blueprint_pipeline.artifixer_appearance_freeze import (
    freeze_source_appearance,
    verify_frozen_appearance,
)
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)


def splat(xyz, scales=None):
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    rng = np.random.default_rng(31)
    return SplatData(
        count=n,
        xyz=xyz,
        properties=(),
        scales=np.log(np.full((n, 3), 0.01) if scales is None else scales).astype(np.float32),
        quats=np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
        opacity=rng.uniform(-1, 1, n).astype(np.float32),
        f_dc=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        sh_rest=rng.uniform(-1, 1, (n, 45)).astype(np.float32),
    )


def test_local_foreground_keeps_broad_and_distant_background_candidates():
    candidate = splat(
        [[0.1, 0.1, 0.01], [0.1, 0.1, 0.01], [2.0, 0.1, 0.01]],
        [[0.01] * 3, [0.5, 0.5, 0.001], [0.01] * 3],
    )
    mask = local_foreground_mask(candidate, lower=[0, 0, 0], upper=[0.2, 0.2, 0.02])
    assert mask.tolist() == [True, False, False]


def test_initialization_preserves_reused_source_rows_exactly(tmp_path):
    retained, deleted, output = [
        tmp_path / name for name in ("retained.ply", "deleted.ply", "seed.ply")
    ]
    write_standard_3dgs_ply(splat([[1, 0, 0], [2, 0, 0]]), retained)
    write_standard_3dgs_ply(splat([[0.1, 0.1, 0.01], [3, 0, 0]]), deleted)
    original = retained.read_bytes(), deleted.read_bytes()
    partition = write_initialization(
        retained=retained,
        deleted=deleted,
        foreground_mask=np.array([True, False]),
        points=np.array([[0.1, 0.1, 0]], np.float32),
        colors=np.array([[0.3, 0.4, 0.5]], np.float32),
        output_path=output,
    )
    assert (retained.read_bytes(), deleted.read_bytes()) == original
    _, rows, _ = _raw_rows(output)
    assert rows[:2].tobytes() == _raw_rows(retained)[1].tobytes()
    assert rows[2:3].tobytes() == _raw_rows(deleted)[1][1:].tobytes()
    assert partition["frozen_source_count"] == 3 and partition["generated_support_count"] == 1
    assert read_standard_3dgs_ply(output).count == 4


def test_support_hole_is_not_replaced_with_an_aabb_plane(tmp_path):
    from pxr import Usd, UsdGeom

    path = tmp_path / "support.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/support")
    mesh.GetPointsAttr().Set([(0, 0, 0), (0.2, 0, 0), (0, 0.2, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    stage.GetRootLayer().Save()
    support = {
        "sage_prim_path": "/support",
        "bounds_min_xyz_m": [0, 0, 0],
        "bounds_max_xyz_m": [0.2, 0.2, 0],
        "top_z_m": 0,
    }
    with pytest.raises(ValueError, match="mesh_coverage_missing"):
        sample_registered_top_surface(
            mesh_path=path, support=support, lower=[0.02, 0.02, 0], upper=[0.18, 0.18, 0.02]
        )


def test_sample_uses_actual_mesh_and_records_bounded_publisher_box_difference(tmp_path):
    from pxr import Usd, UsdGeom

    path = tmp_path / "support.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/support")
    mesh.GetPointsAttr().Set(
        [(0, 0, 0.0015), (0.2, 0, 0.0015), (0.2, 0.2, 0.0015), (0, 0.2, 0.0015)]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    stage.GetRootLayer().Save()
    support = {
        "sage_prim_path": "/support",
        "bounds_min_xyz_m": [0.003, 0, 0],
        "bounds_max_xyz_m": [0.203, 0.2, 0],
        "top_z_m": 0,
    }
    points, _, _, registration = sample_registered_top_surface(
        mesh_path=path, support=support, lower=[0.02, 0.02, 0], upper=[0.18, 0.18, 0.02]
    )
    assert np.allclose(points[:, 2], 0.0015)
    assert registration["maximum_absolute_bounds_deviation_m"] == pytest.approx(0.003)
    assert registration["physical_alignment_qualified"] is False
    support["bounds_min_xyz_m"][0] = 0.02
    with pytest.raises(ValueError, match="registration_mismatch"):
        sample_registered_top_surface(
            mesh_path=path, support=support, lower=[0.02, 0.02, 0], upper=[0.18, 0.18, 0.02]
        )


def _model_fixture():
    torch = pytest.importorskip("torch")
    reference = splat([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    partition = {
        "frozen_source_count": 2,
        "generated_support_count": 1,
        "total_count": 3,
        "reused_source_vertex_rows_byte_exact": True,
    }

    class Model:
        def __init__(self):
            self.positions = torch.nn.Parameter(torch.tensor(reference.xyz), requires_grad=False)
            self.rotation = torch.nn.Parameter(torch.tensor(reference.quats), requires_grad=False)
            self.scale = torch.nn.Parameter(torch.tensor(reference.scales), requires_grad=False)
            self.density = torch.nn.Parameter(
                torch.tensor(reference.opacity[:, None]), requires_grad=False
            )
            self.features_albedo = torch.nn.Parameter(torch.tensor(reference.f_dc))
            sh = reference.sh_rest.reshape(3, 3, 15).transpose(0, 2, 1).reshape(3, 45)
            self.features_specular = torch.nn.Parameter(torch.tensor(sh))
            self.set_optimizable_parameters()

        def set_optimizable_parameters(self):
            pass

    return torch, reference, partition, Model


def test_released_training_hook_keeps_originals_fixed_through_adam_steps():
    torch, reference, partition, Model = _model_fixture()
    original = Model.set_optimizable_parameters
    with freeze_source_appearance(Model, reference=reference, partition=partition):
        model = Model()
        before = model.features_albedo.detach().clone()
        optimizer = torch.optim.Adam([model.features_albedo, model.features_specular], lr=0.1)
        for _ in range(4):
            optimizer.zero_grad()
            (model.features_albedo.sum() + model.features_specular.sum()).backward()
            optimizer.step()
        assert torch.equal(model.features_albedo[:2], before[:2])
        assert not torch.equal(model.features_albedo[2:], before[2:])
    assert Model.set_optimizable_parameters is original
    assert (
        verify_frozen_appearance(model=model, reference=reference, partition=partition)[
            "exact_source_appearance_prefix_match"
        ]
        is True
    )


def test_appearance_change_fails_closed_and_restores_scoped_hook():
    torch, reference, partition, Model = _model_fixture()
    original = Model.set_optimizable_parameters
    with pytest.raises(ValueError, match="frozen_appearance_changed:features_albedo"):
        with freeze_source_appearance(Model, reference=reference, partition=partition):
            model = Model()
            with torch.no_grad():
                model.features_albedo[0, 0] += 0.01
    assert Model.set_optimizable_parameters is original


def test_opacity_drift_is_rejected_even_on_generated_support():
    torch, reference, partition, Model = _model_fixture()
    model = Model()
    with torch.no_grad():
        model.density[-1] += 0.01
    with pytest.raises(ValueError, match="frozen_appearance_changed:density"):
        verify_frozen_appearance(model=model, reference=reference, partition=partition)


def test_declared_mode_requires_bound_initialization_receipt(tmp_path):
    script = Path(__file__).parents[1] / "scripts/public_scene_artifixer3d_runner.py"
    spec = importlib.util.spec_from_file_location("background_runner_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises(ValueError, match="appearance_initialization_missing"):
        runner._validated_appearance_initialization(
            tmp_path, {"artifixer3d": {"geometry_policy": runner.DECLARED_GEOMETRY_POLICY}}
        )
    overrides = runner._retained_geometry_training_overrides(
        steps=30_000, geometry_policy=runner.DECLARED_GEOMETRY_POLICY
    )
    assert "model.optimize_density=false" in overrides
    assert "model.progressive_training.init_n_features=3" in overrides


def test_dual_packet_carries_and_rechecks_initialization_identity(tmp_path):
    from blueprint_pipeline.artifixer_source_geometry_admission import _record
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
    from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
        materialize_artifixer3d_candidate_inputs,
    )
    from blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs import (
        materialize_dual_target_artifixer3d_inputs,
    )
    from tests.test_public_scene_artifixer3d_candidate_inputs import _preflight
    from tests.test_public_scene_artifixer3d_dual_target_inputs import _semantic_receipts

    preflight = _preflight(tmp_path / "fixture", count=1, cameras_per_task=2)
    source_root = tmp_path / "source"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight, output_root=source_root
    )
    n = candidate["shared_retained_scene"]["retained_gaussian_count"]
    partition = {
        "frozen_source_count": n - 1,
        "generated_support_count": 1,
        "total_count": n,
        "reused_source_vertex_rows_byte_exact": True,
    }
    initialization = {
        "schema_version": "artifixer_registered_background_initialization.v1",
        "geometry_mode": "freeze_declared_appearance_initialization",
        "parameter_partition": partition,
        "initialization": candidate["shared_retained_scene"],
        "policy": {
            "original_appearance_frozen": True,
            "generated_geometry_and_opacity_frozen": True,
        },
    }
    initialization["receipt_digest"] = canonical_digest(
        initialization, digest_field="receipt_digest"
    )
    receipt_path = tmp_path / "initialization.json"
    receipt_path.write_text(canonical_json(initialization) + "\n")
    candidate["appearance_initialization"] = {
        "receipt": _record(receipt_path),
        "receipt_digest": initialization["receipt_digest"],
        "geometry_mode": initialization["geometry_mode"],
        "parameter_partition": partition,
    }
    candidate["receipt_digest"] = canonical_digest(candidate, digest_field="receipt_digest")
    candidate_path = source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    candidate_path.write_text(canonical_json(candidate) + "\n")
    teachers = _semantic_receipts(tmp_path, source_root=source_root, source=candidate)
    dual_root = tmp_path / "dual"
    dual = materialize_dual_target_artifixer3d_inputs(
        source_candidate_inputs_receipt_path=candidate_path,
        semantic_teacher_receipt_paths=teachers,
        output_root=dual_root,
        transition_radius_pixels=1,
    )
    binding = dual["appearance_initialization"]
    assert binding["receipt"]["relative_path"] == "provenance/appearance_initialization.json"
    script = Path(__file__).parents[1] / "scripts/public_scene_artifixer3d_runner.py"
    spec = importlib.util.spec_from_file_location("background_binding_runner_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    request = {
        "artifixer3d": {
            "geometry_policy": runner.DECLARED_GEOMETRY_POLICY,
            "appearance_initialization": binding,
        }
    }
    assert runner._validated_appearance_initialization(dual_root, request) == binding
    altered = json.loads(json.dumps(request))
    altered["artifixer3d"]["appearance_initialization"]["parameter_partition"][
        "frozen_source_count"
    ] -= 1
    with pytest.raises(ValueError, match="initialization_receipt_invalid"):
        runner._validated_appearance_initialization(dual_root, altered)


def test_completed_checkpoint_survives_post_training_guard_refusal(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    script = Path(__file__).parents[1] / "scripts/public_scene_artifixer3d_runner.py"
    spec = importlib.util.spec_from_file_location("background_recovery_runner_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    input_root = tmp_path / "input"
    (input_root / "shared_initialization").mkdir(parents=True)
    reference = input_root / "shared_initialization" / "reference.ply"
    write_standard_3dgs_ply(splat([[0, 0, 1], [1, 0, 1]]), reference)
    output = tmp_path / "output" / "artifixer_output"
    checkpoint = tmp_path / "checkpoint.pt"

    def prepare(**kwargs):
        kwargs["log"].parent.mkdir(parents=True)
        return {
            "staged_task": input_root,
            "scene": SimpleNamespace(scene_id="fixture"),
            "steps": 10,
            "paths": SimpleNamespace(
                distillation_input_dir=tmp_path,
                run_root=tmp_path,
                distillation_selected_indices_path=tmp_path / "indices.json",
                override_image_dir=tmp_path / "overrides",
            ),
        }

    def train(*_args):
        print("optimization completed; checking source appearance")
        checkpoint.write_bytes(b"completed-optimization-weights")
        raise ValueError("artifixer_frozen_appearance_changed:features_albedo")

    package = ModuleType("data_processing")
    package.artifixer3d = SimpleNamespace(artifixer3d_checkpoint=lambda *_: checkpoint)
    package.threedgrut_training = SimpleNamespace(
        train_3dgrut=train, DEFAULT_THREEDGRUT_CONFIG_DIR=tmp_path
    )
    monkeypatch.setitem(sys.modules, "data_processing", package)
    monkeypatch.setattr(runner, "_prepare_dual_target_distillation_replay", prepare)
    request = {
        "artifixer3d": {
            "loss_overrides": {},
            "geometry_policy": runner.RETAINED_GEOMETRY_POLICY,
            "config_name": "fixture",
        }
    }
    with pytest.raises(ValueError, match="frozen_appearance_changed"):
        runner._dual_target_task_runtime(
            task={"task_id": "fixture"},
            input_root=input_root,
            source_root=tmp_path,
            output_root=output,
            request=request,
        )
    recovery = output.parent / "retained_training_evidence" / "fixture"
    assert (recovery / "checkpoint.pt").read_bytes() == checkpoint.read_bytes()
    outcome = json.loads((recovery / "export_outcome.json").read_text())
    assert outcome["optimization_complete"] is True
    assert outcome["blockers"] == ["artifixer_frozen_appearance_changed:features_albedo"]


def test_rebinding_keeps_accepted_teacher_bytes_and_original_receipts(tmp_path, monkeypatch):
    import blueprint_pipeline.artifixer_background_initialization as module
    from tests.test_public_scene_artifixer3d_dual_target_inputs import _dual_candidate

    _, candidate, teachers, _ = _dual_candidate(tmp_path / "fixture", cameras_per_task=2)
    before = teachers[0].read_bytes()
    old_teacher = json.loads(before)
    preflight = Path(candidate["calibrated_residual_preflight"]["path"])
    initialization = Path(candidate["shared_retained_scene"]["path"])
    receipt = {
        "receipt_digest": "sha256:" + "1" * 64,
        "parameter_partition": {
            "total_count": candidate["shared_retained_scene"]["retained_gaussian_count"]
        },
        "initialization": candidate["shared_retained_scene"],
    }
    initialization_receipt = tmp_path / "initialization.json"
    initialization_receipt.write_text(json.dumps(receipt))
    # Mesh construction is exercised separately on real triangles and run inputs;
    # this test isolates its handoff into the existing candidate/teacher compilers.
    monkeypatch.setattr(module, "materialize_background_initialization",
                        lambda **_: (initialization, initialization_receipt, receipt))
    updated, updated_path, teacher_path = module.prepare_background_supported_inputs(
        envelope={}, configuration={}, candidate=candidate, teacher_receipt_path=teachers[0],
        preflight_path=preflight, output_root=tmp_path / "rebound")
    assert teachers[0].read_bytes() == before
    teacher = json.loads(teacher_path.read_text())
    assert [f["whole_frame_semantic_teacher"]["sha256"] for f in teacher["frames"]] == [
        f["whole_frame_semantic_teacher"]["sha256"] for f in old_teacher["frames"]]
    assert teacher["source_candidate_inputs_receipt"]["receipt_digest"] == updated["receipt_digest"]
    assert teacher["source_candidate_inputs_receipt"]["path"] == str(updated_path)
    assert teacher["editor_identity"]["unchanged_admitted_teacher_receipt_digest"] == old_teacher["receipt_digest"]
