from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.articulated_isaac_bundle import (
    ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION,
    ArticulatedIsaacBundleError,
    build_articulated_isaac_bundle,
)


def _worker(tmp_path: Path) -> Path:
    return Path(__file__).resolve().parents[1] / (
        "scripts/run_adp009d_articulated_isaac_worker.py"
    )


def _digest_of(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_root(tmp_path: Path) -> Path:
    """A frozen native probe as materialize_articulated_native_probe writes it."""

    from blueprint_pipeline.articulated_native_probe import (
        materialize_articulated_native_probe,
    )
    from tests.test_articulated_native_probe import _candidate

    root = tmp_path / "probe"
    materialize_articulated_native_probe(
        candidate_usd_path=_candidate(tmp_path / "candidate.usda"),
        destination=root,
        task_joint_prim_path="/Asset/joints/upper_door_hinge",
        locked_joint_prim_paths=["/Asset/joints/lower_door_hinge"],
        commanded_sweep_degrees=[0.0, 25.0, 45.0, 55.0],
        reset_joint_positions_rad={
            "/Asset/joints/upper_door_hinge": 0.0,
            "/Asset/joints/lower_door_hinge": 0.0,
        },
        locked_joint_motion_tolerance_rad=0.001,
        settle_samples=40,
        control_frequency_hz=15.0,
    )
    return root


def _build(tmp_path: Path, **overrides):
    # only freeze a fresh probe when the caller did not supply one, so a test
    # that tampers with the probe is not silently handed a regenerated copy
    arguments = {
        "probe_root": overrides.pop("probe_root", None) or _probe_root(tmp_path),
        "job_dir": tmp_path / "job",
        "worker_source": Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_isaac_worker.py",
        "source_commit_sha": "a" * 40,
        "generated_at": "2026-08-09T00:00:00+00:00",
    }
    arguments.update(overrides)
    return build_articulated_isaac_bundle(**arguments)


def _scene_bind_probe(root: Path, *, scene_id: str = "840920") -> dict:
    spec_path = root / "articulated_native_probe_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    predecessor = {
        "schema_version": "paired_native_simready_predecessor_binding.v1",
        "scene_id": scene_id,
        "task_id": "task_a_washer_door_open",
        "asset_id": f"{scene_id}_simready_washer_candidate",
        "candidate_usd_sha256": spec["candidate_usd_sha256"],
        "binding_digest": "",
    }
    predecessor["binding_digest"] = canonical_digest(
        predecessor, digest_field="binding_digest"
    )
    spec["scene_id"] = scene_id
    spec["paired_native_predecessor"] = predecessor
    spec["receipt_digest"] = canonical_digest(spec, digest_field="receipt_digest")
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return predecessor


def test_bundle_ships_the_frozen_probe_and_the_articulated_worker(
    tmp_path: Path,
) -> None:
    receipt = _build(tmp_path)

    assert receipt["schema_version"] == ARTICULATED_ISAAC_BUNDLE_SCHEMA_VERSION
    assert receipt["status"] == "ready"
    assert receipt["retry_cap"] == 0
    assert receipt["blockers"] == []
    assert receipt["probe_spec_sha256"].startswith("sha256:")
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/run_isaac_realistic_runtime.sh" in names
    assert "provider_runtime/isaac_realistic_runtime_runner.py" in names
    assert "provider_runtime/native/articulated_native_probe_spec.json" in names
    assert "provider_runtime/native/blank_physics_stage.usda" in names
    assert "provider_runtime/native/articulation_stage.usda" in names


def test_bundle_carries_no_rigid_stimulus_stages(tmp_path: Path) -> None:
    """The rigid lane's drop/slide/tip probes are meaningless for a door."""

    receipt = _build(tmp_path)

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = "\n".join(archive.namelist())
    for stale in ("drop_stage", "isaac_slide_stage", "isaac_tip_stage", "gripper_stage"):
        assert stale not in names


def test_entrypoint_writes_a_typed_result_when_isaac_dies(tmp_path: Path) -> None:
    """A silent process exit must not read as 'no result, therefore fine'."""

    receipt = _build(tmp_path)

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_isaac_realistic_runtime.sh"
        ).decode("utf-8")
    # conform to the Isaac lane's crash-fallback contract, which the transport
    # preflight checks by marker before any paid mutation
    assert "write_missing_result" in entrypoint
    assert "isaac_runner_process_exited_without_runtime_result" in entrypoint
    assert "blocked_isaac_process_exited_without_result" in entrypoint
    assert "isaac_realistic_runtime_runner.py" in entrypoint
    assert "/isaac-sim/python.sh" in entrypoint


def test_bundle_binds_the_exact_probe_and_candidate_digests(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    spec = json.loads((root / "articulated_native_probe_spec.json").read_text())

    receipt = _build(tmp_path, probe_root=root)

    assert receipt["candidate_usd_sha256"] == spec["candidate_usd_sha256"]
    assert receipt["expected"]["assembly_joint_count"] == 2
    assert receipt["expected"]["maximum_commanded_degrees"] == 55.0
    assert receipt["required_readbacks"] == spec["required_readbacks"]
    assert receipt["probe_names"] == sorted(spec["required_readbacks"])


def test_scene_bound_bundle_exposes_profile_and_authority_bindings(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    predecessor = _scene_bind_probe(root)

    receipt = _build(tmp_path, probe_root=root)

    assert receipt["scene_id"] == "840920"
    assert receipt["native_probe_manifest_sha256"] == _digest_of(
        root / "articulated_native_probe_spec.json"
    )
    assert receipt["predecessor_binding_digest"] == predecessor["binding_digest"]
    assert receipt["paired_native_predecessor"] == predecessor


def test_scene_bound_bundle_refuses_a_wrong_scene_predecessor(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    _scene_bind_probe(root)
    spec_path = root / "articulated_native_probe_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["scene_id"] = "840313"
    spec["receipt_digest"] = canonical_digest(spec, digest_field="receipt_digest")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, probe_root=root)

    assert "predecessor_binding_invalid" in str(excinfo.value)


def test_a_probe_that_was_already_executed_is_refused(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    spec_path = root / "articulated_native_probe_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["status"] = "executed"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, probe_root=root)

    assert any("probe_not_frozen" in error for error in excinfo.value.errors)


def test_a_tampered_probe_spec_is_refused(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)
    stage = root / "blank_physics_stage.usda"
    stage.write_text(stage.read_text() + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, probe_root=root)

    assert any("stage_digest_mismatch" in error for error in excinfo.value.errors)


def test_a_short_commit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        _build(tmp_path, source_commit_sha="abc123")

    assert any("source_commit_invalid" in error for error in excinfo.value.errors)


def test_bundle_pins_the_same_image_the_allocator_checks(tmp_path: Path) -> None:
    """A separate image constant here would fail admission on every run."""

    from blueprint_pipeline.public_scene_simready_isaac_bundle import (
        DEFAULT_IMAGE as LANE_IMAGE,
    )

    receipt = _build(tmp_path)

    assert receipt["container_image"] == LANE_IMAGE
    assert "@sha256:" in receipt["container_image"]


def test_bundle_is_deterministic(tmp_path: Path) -> None:
    root = _probe_root(tmp_path)

    first = _build(tmp_path, probe_root=root, job_dir=tmp_path / "a")
    second = _build(tmp_path, probe_root=root, job_dir=tmp_path / "b")

    assert first["bundle_sha256"] == second["bundle_sha256"]


def test_the_lane_validates_the_probe_set_the_bundle_declares() -> None:
    """A rigid probe set must not be assumed for an articulated readback."""

    from blueprint_pipeline.public_scene_simready_isaac_vast import (
        RIGID_PROBE_NAMES,
        _execution_blockers,
    )

    articulated = {
        "status": "completed",
        "native_isaac_executed": True,
        "physical_success_established": False,
        "source_target_collider_active": False,
        "replacement_count": 1,
        "probe_results": [
            {"probe": "articulation_root_identity", "passed": True},
            {"probe": "commanded_sweep_reaches_maximum", "passed": True},
        ],
    }
    names = frozenset({"articulation_root_identity", "commanded_sweep_reaches_maximum"})

    assert _execution_blockers(articulated, names) == []
    # the rigid default still rejects it, so the check has not been weakened
    assert "simready_isaac_probe_set_invalid" in _execution_blockers(
        articulated, RIGID_PROBE_NAMES
    )


def test_bundle_fills_every_transport_slot_the_lane_requires(tmp_path: Path) -> None:
    """The lane's entry contract is its interface; the probe must satisfy it."""

    receipt = _build(tmp_path)

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        scene = archive.read("provider_runtime/generated_site_scene.usda").decode()
        manifest = json.loads(
            archive.read("provider_runtime/isaac_provider_eval_manifest.json")
        )
    for required in (
        "provider_runtime/isaac_provider_eval_manifest.json",
        "provider_runtime/generated_site_scene.usda",
        "provider_runtime/generated_site_scene.usd",
        "provider_runtime/scenario_eval_matrix.json",
        "provider_runtime/camera_manifest.json",
        "provider_runtime/episode_spec_manifest.json",
    ):
        assert required in names, required
    # the scene the runtime opens is the articulation stage, not a rigid drop
    assert "PhysicsScene" in scene
    assert manifest["proof_boundaries"]["physical_success_established"] is False
    assert manifest["relative_paths"]["probe_spec"].endswith(
        "articulated_native_probe_spec.json"
    )


def test_the_worker_result_attests_to_itself(tmp_path: Path) -> None:
    """The lane refuses a result that carries no self-consistent digest.

    Isaac v5 passed all eleven readbacks and was still recorded blocked,
    because the worker wrote no result_digest. The worker runs inside Isaac's
    interpreter without the package, so the digest definition is mirrored
    there; this pins the two to the same bytes.
    """

    import importlib.util

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    worker_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_isaac_worker.py"
    )
    spec = importlib.util.spec_from_file_location("articulated_worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {"schema_version": "x", "status": "completed", "readbacks": {"a": 1}}
    output = tmp_path / "result.json"
    module._persist(output, payload)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["result_digest"].startswith("sha256:")
    assert written["result_digest"] == canonical_digest(
        written, digest_field="result_digest"
    )


def test_a_different_probe_kind_bundles_through_the_same_builder(tmp_path: Path) -> None:
    """Probe kinds multiply; the transport does not.

    Forking this builder per probe would fork the slot layout, the image pin
    and the digest checks with it, and those are exactly the parts whose drift
    costs a launch to discover.
    """

    root = tmp_path / "controls_probe"
    root.mkdir()
    stage = root / "controls_stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    spec = {
        "schema_version": "articulated_controls_probe_spec.v1",
        "status": "frozen_not_executed",
        "stages": {
            "controls_stage": {
                "path": str(stage),
                "sha256": _digest_of(stage),
            }
        },
        "required_readbacks": ["zero_action_door_stays_shut"],
    }
    (root / "articulated_controls_probe_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )

    receipt = build_articulated_isaac_bundle(
        probe_root=root,
        job_dir=tmp_path / "job",
        worker_source=_worker(tmp_path),
        source_commit_sha="b" * 40,
        probe_spec_filename="articulated_controls_probe_spec.json",
        probe_spec_schema_version="articulated_controls_probe_spec.v1",
        primary_stage_name="controls_stage",
    )

    assert receipt["probe_names"] == ["zero_action_door_stays_shut"]


def test_a_spec_whose_schema_is_not_the_declared_one_fails_closed(
    tmp_path: Path,
) -> None:
    """Bundling the wrong probe kind would run the wrong worker against it."""

    root = tmp_path / "mismatched"
    root.mkdir()
    stage = root / "controls_stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    (root / "articulated_controls_probe_spec.json").write_text(
        json.dumps(
            {
                "schema_version": "articulated_native_probe_spec.v1",
                "status": "frozen_not_executed",
                "stages": {
                    "controls_stage": {
                        "path": str(stage),
                        "sha256": _digest_of(stage),
                    }
                },
                "required_readbacks": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArticulatedIsaacBundleError) as excinfo:
        build_articulated_isaac_bundle(
            probe_root=root,
            job_dir=tmp_path / "job2",
            worker_source=_worker(tmp_path),
            source_commit_sha="c" * 40,
            probe_spec_filename="articulated_controls_probe_spec.json",
            probe_spec_schema_version="articulated_controls_probe_spec.v1",
            primary_stage_name="controls_stage",
        )

    assert any("probe_schema_invalid" in error for error in excinfo.value.errors)


def test_the_entrypoint_points_at_the_spec_the_bundle_actually_carries(
    tmp_path: Path,
) -> None:
    """A hardcoded spec name boots Isaac and then cannot find its own input.

    The runner exits, the fallback writes "spec unreadable", and the launch is
    spent on a result that says nothing about the probe. Nothing earlier in the
    chain catches it: the bundle is well-formed, the digests match, and the dry
    run is clean.
    """

    root = tmp_path / "controls_probe"
    root.mkdir()
    stage = root / "controls_stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    (root / "articulated_controls_probe_spec.json").write_text(
        json.dumps(
            {
                "schema_version": "articulated_controls_probe_spec.v1",
                "status": "frozen_not_executed",
                "stages": {
                    "controls_stage": {
                        "path": str(stage),
                        "sha256": _digest_of(stage),
                    }
                },
                "required_readbacks": ["zero_action_door_stays_shut"],
            }
        ),
        encoding="utf-8",
    )

    receipt = build_articulated_isaac_bundle(
        probe_root=root,
        job_dir=tmp_path / "job",
        worker_source=_worker(tmp_path),
        source_commit_sha="d" * 40,
        probe_spec_filename="articulated_controls_probe_spec.json",
        probe_spec_schema_version="articulated_controls_probe_spec.v1",
        primary_stage_name="controls_stage",
    )

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_isaac_realistic_runtime.sh"
        ).decode("utf-8")
        names = set(archive.namelist())

    assert "articulated_controls_probe_spec.json" in entrypoint
    assert "articulated_native_probe_spec.json" not in entrypoint
    assert (
        "provider_runtime/native/articulated_controls_probe_spec.json" in names
    )
