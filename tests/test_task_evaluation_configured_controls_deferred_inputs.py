"""A first-run scene has no prior robot plan; the continuation derives it after publication.

The 839873 intents bound a native trajectory plan rebound from an earlier
diagnostic and an overview image from an earlier run, so a fresh scene could
not be provisioned before its first configuration run.  A deferred input names
what the autostart must derive from the completed run instead: the rigid
construction phase plan from the published revision's exact documents, and the
overview image from the published task thumbnail.  Nothing here executes a
provider; the plan remains a CPU input to placement screening.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart as autostart
from blueprint_pipeline import (
    task_evaluation_configured_controls_deferred_inputs as deferred,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
)
from tests.test_task_evaluation_configured_controls_autostart import (
    COMMIT,
    _release_template,
    _write,
)
from tests.test_task_evaluation_rigid_relocation_native_adapter import _case


def _paths(tmp_path: Path, *, trajectory: object, overview: object) -> dict[str, object]:
    names = (
        "robot_asset_usd_path",
        "robot_mount_interface_path",
        "scene_camera_calibration_path",
        "cameras_path",
        "runtime_binding_path",
    )
    paths: dict[str, object] = {
        name: str(_write(tmp_path / "inputs" / f"{name}.json")) for name in names
    }
    paths["native_trajectory_plan_path"] = trajectory
    paths["overview_image_paths"] = overview
    return paths


def _phases(tmp_path: Path) -> dict[str, dict[str, str]]:
    phases: dict[str, dict[str, str]] = {}
    for phase in ("construction", "controls"):
        names = ["release_window_template_path", "authorization_path", "launch_authority_path"]
        if phase == "construction":
            names.append("lineage_path")
        phases[phase] = {
            name: str(
                _write(
                    tmp_path / "inputs" / phase / f"{name}.json",
                    _release_template() if name == "release_window_template_path" else b"{}\n",
                )
            )
            for name in names
        }
    return phases


def _deferred_intent(tmp_path: Path) -> dict[str, object]:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(exist_ok=True)
    return autostart.materialize_configured_controls_autostart_intent(
        expected_production_commit=COMMIT,
        submitted_by="configured-controls-autostart",
        team_namespace="blueprint-adp",
        scene_id="interiorgs-841757",
        task_id="scene-841757-book-to-tray",
        target_position_world_m=[3.25, -6.76, 0.29],
        paths=_paths(
            tmp_path,
            trajectory={"deferred": deferred.TRAJECTORY_MODE},
            overview={"deferred": deferred.OVERVIEW_MODE},
        ),
        phases=_phases(tmp_path),
        profile_dir=profile_dir,
        output_path=tmp_path / "intent.json",
        openai_project_id="proj_test",
        openai_api_key_id="key_visual_review",
    )


def test_intent_accepts_deferred_trajectory_and_overview_and_binds_only_concrete_bytes(
    tmp_path: Path,
) -> None:
    intent = _deferred_intent(tmp_path)
    assert intent["paths"]["native_trajectory_plan_path"] == {"deferred": deferred.TRAJECTORY_MODE}
    assert intent["paths"]["overview_image_paths"] == {"deferred": deferred.OVERVIEW_MODE}
    inventory = set(intent["artifact_inventory"])
    assert "native_trajectory_plan_path" not in inventory
    assert not any(name.startswith("overview_image_paths") for name in inventory)
    assert {"robot_asset_usd_path", "cameras_path", "runtime_binding_path"} <= inventory
    assert autostart.validate_configured_controls_autostart_intent(intent) == intent
    assert deferred.deferred_declarations(intent["paths"]) == {
        "native_trajectory_plan_path": deferred.TRAJECTORY_MODE,
        "overview_image_paths": deferred.OVERVIEW_MODE,
    }


def test_intent_refuses_an_unknown_deferred_mode(tmp_path: Path) -> None:
    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_intent_paths_invalid",
    ):
        autostart.materialize_configured_controls_autostart_intent(
            expected_production_commit=COMMIT,
            submitted_by="configured-controls-autostart",
            team_namespace="blueprint-adp",
            scene_id="interiorgs-841757",
            task_id="scene-841757-book-to-tray",
            target_position_world_m=[3.25, -6.76, 0.29],
            paths=_paths(
                tmp_path,
                trajectory={"deferred": "guess_from_nothing"},
                overview={"deferred": deferred.OVERVIEW_MODE},
            ),
            phases=_phases(tmp_path),
            profile_dir=tmp_path / "profiles",
            output_path=tmp_path / "intent.json",
            openai_project_id="proj_test",
            openai_api_key_id="key_visual_review",
        )


def test_concrete_intents_are_unchanged(tmp_path: Path) -> None:
    from tests.test_task_evaluation_configured_controls_autostart import _intent

    _path, value = _intent(tmp_path)
    assert deferred.deferred_declarations(value["paths"]) == {}
    assert deferred.concrete_paths(value["paths"]) == value["paths"]


def test_trajectory_plan_is_derived_from_the_exact_published_documents(tmp_path: Path) -> None:
    _launch, revision, references, _docs = _case(tmp_path)
    documents = {
        contract_path: Path(row["materialized_path"]) for contract_path, row in references.items()
    }
    plan = deferred.derive_native_trajectory_plan(revision=revision, documents=documents)
    assert plan["schema_version"] == "native_rigid_construction_phase_plan.v1"
    assert plan["task_kind"] == "rigid_pick_place"
    assert plan["phase_count"] == len(plan["phases"]) >= 4
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    trajectory = placement_trajectory_from_native_plan(plan)
    assert trajectory["source_plan_digest"] == plan["plan_digest"]
    again = deferred.derive_native_trajectory_plan(revision=revision, documents=documents)
    assert again == plan


def test_adapter_binds_the_task_from_the_revision_alone_exactly_as_a_request_would(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
        adapt_rigid_relocation_task_template,
    )

    launch, revision, references, _docs = _case(tmp_path)
    with_request = adapt_rigid_relocation_task_template(
        request=launch, configured_revision=revision, materialized_references=references
    )
    without_request = adapt_rigid_relocation_task_template(
        configured_revision=revision, materialized_references=references
    )
    assert without_request["native_task_definition"] == with_request["native_task_definition"]
    assert without_request["native_success_criteria"] == with_request["native_success_criteria"]
    assert without_request["native_episode_execution"] == with_request["native_episode_execution"]


def test_trajectory_derivation_refuses_a_document_that_drifted_from_the_revision(
    tmp_path: Path,
) -> None:
    _launch, revision, references, _docs = _case(tmp_path)
    documents = {
        contract_path: Path(row["materialized_path"]) for contract_path, row in references.items()
    }
    drifted = documents["scene.configured_revision.task_template.definition"]
    value = json.loads(drifted.read_text())
    value["target_center_xyz_m"] = [9.0, 9.0, 9.0]
    drifted.write_text(json.dumps(value, sort_keys=True) + "\n")
    with pytest.raises(
        deferred.ConfiguredControlsDeferredInputError,
        match="configured_controls_deferred_document_mismatch",
    ):
        deferred.derive_native_trajectory_plan(revision=revision, documents=documents)


def test_resolution_writes_the_plan_and_thumbnail_once_from_the_completed_run(
    tmp_path: Path,
) -> None:
    _launch, revision, references, _docs = _case(tmp_path)
    thumbnail = b"\x89PNG\r\n\x1a\nfake"
    revision["presentation"]["task_thumbnail"] = {
        "uri": "s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/ns/task_thumbnail/sha256/"
        + hashlib.sha256(thumbnail).hexdigest()
        + "/configured_task_thumbnail.png",
        "digest": "sha256:" + hashlib.sha256(thumbnail).hexdigest(),
        "size_bytes": len(thumbnail),
    }
    revision["presentation"]["selection"]["frame_digest"] = "sha256:" + hashlib.sha256(thumbnail).hexdigest()
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    payloads = {
        row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()
    }
    payloads[revision["presentation"]["task_thumbnail"]["uri"]] = thumbnail
    fetched: list[str] = []

    def fetcher(reference: dict) -> bytes:
        fetched.append(reference["uri"])
        return payloads[reference["uri"]]

    intent = _deferred_intent(tmp_path)
    resolved = deferred.resolve_deferred_inputs(
        intent=intent,
        revision=revision,
        output_root=tmp_path / "progression" / "cpu-robot-binding",
        fetcher=fetcher,
    )
    plan_path = Path(resolved["native_trajectory_plan_path"])
    assert plan_path.is_file() and plan_path.stat().st_mode & 0o777 == 0o440
    plan = json.loads(plan_path.read_text())
    assert plan["schema_version"] == "native_rigid_construction_phase_plan.v1"
    overview = [Path(item) for item in resolved["overview_image_paths"]]
    assert len(overview) == 1 and overview[0].read_bytes() == thumbnail
    for name in ("robot_asset_usd_path", "cameras_path", "runtime_binding_path"):
        assert resolved[name] == intent["paths"][name]
    assert set(fetched) == set(payloads)

    again = deferred.resolve_deferred_inputs(
        intent=intent,
        revision=revision,
        output_root=tmp_path / "progression" / "cpu-robot-binding",
        fetcher=lambda reference: pytest.fail("resolved bytes are reused, never refetched"),
    )
    assert again == resolved


def _runtime_binding(mount_source: object) -> dict:
    return {
        "runtime": {
            "identity": {"id": "native-arena", "version": "isaac-2026-1"},
            "oci_image": "nvcr.io/nvidia/isaac-sim:6.0.1@sha256:" + "b" * 64,
            "entrypoint": ["/opt/blueprint/run-task-evaluation"],
            "health_protocol": {"uri": "s3://blueprint/h.json", "digest": "sha256:" + "1" * 64, "size_bytes": 518},
            "requirements": {"cpu_cores": 8, "memory_gib": 64, "gpu_count": 1, "disk_gib": 100},
            "network": {"default": "deny", "allowlist": []},
            "secret_refs": [],
            "mounts": [
                {"source": mount_source, "container_path": "/inputs", "mode": "read_only"},
                {"container_path": "/outputs", "mode": "output"},
            ],
            "output_limit_bytes": 20_000_000_000,
        },
        "execution_adapter": {
            "kind": "native_task_arena",
            "version": "v1",
            "runtime_source_bundle": {"uri": "s3://blueprint/r.zip", "digest": "sha256:" + "2" * 64, "size_bytes": 4096},
        },
        "spend": {
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 2.0,
            "hard_ttl_seconds": 9000,
            "retry_cap": 0,
            "selected_provider": "vast",
            "provider_allowlist": ["vast"],
        },
    }


def test_resolution_binds_the_runtime_mount_to_the_published_scene_bundle(tmp_path: Path) -> None:
    """The construction runtime mounts the exact published bundle, unknown before the run."""

    _launch, revision, _references, _docs = _case(tmp_path)
    intent = _deferred_intent(tmp_path)
    template_path = Path(intent["paths"]["runtime_binding_path"])
    template_path.write_text(
        json.dumps(_runtime_binding({"deferred": deferred.SCENE_BUNDLE_MODE}), sort_keys=True) + "\n"
    )
    resolved = deferred.resolve_runtime_binding(
        runtime_binding_path=template_path,
        revision=revision,
        output_root=tmp_path / "progression" / "cpu-robot-binding",
    )
    assert resolved != template_path
    binding = json.loads(resolved.read_text())
    assert binding["runtime"]["mounts"][0]["source"] == revision["configured_scene_bundle"]
    expected = _runtime_binding(revision["configured_scene_bundle"])
    assert binding == expected
    assert resolved.stat().st_mode & 0o777 == 0o440
    assert deferred.resolve_runtime_binding(
        runtime_binding_path=template_path,
        revision=revision,
        output_root=tmp_path / "progression" / "cpu-robot-binding",
    ) == resolved

    concrete = tmp_path / "concrete_binding.json"
    concrete.write_text(json.dumps(_runtime_binding(revision["configured_scene_bundle"])) + "\n")
    assert deferred.resolve_runtime_binding(
        runtime_binding_path=concrete, revision=revision, output_root=tmp_path / "x"
    ) == concrete


def test_runtime_binding_refuses_a_deferred_mount_that_is_not_the_scene_input(tmp_path: Path) -> None:
    _launch, revision, _references, _docs = _case(tmp_path)
    template = _runtime_binding({"uri": "s3://blueprint/other.zip", "digest": "sha256:" + "3" * 64, "size_bytes": 5})
    template["runtime"]["mounts"].append(
        {"source": {"deferred": deferred.SCENE_BUNDLE_MODE}, "container_path": "/extra", "mode": "read_only"}
    )
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(template) + "\n")
    with pytest.raises(
        deferred.ConfiguredControlsDeferredInputError,
        match="configured_controls_deferred_runtime_binding_invalid",
    ):
        deferred.resolve_runtime_binding(runtime_binding_path=path, revision=revision, output_root=tmp_path / "y")


def test_resolution_refuses_fetched_bytes_that_do_not_match_the_reference(tmp_path: Path) -> None:
    _launch, revision, references, _docs = _case(tmp_path)
    revision["presentation"]["task_thumbnail"] = {
        "uri": "s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/ns/x.png",
        "digest": "sha256:" + "0" * 64,
        "size_bytes": 4,
    }
    revision["presentation"]["selection"]["frame_digest"] = "sha256:" + "0" * 64
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    intent = _deferred_intent(tmp_path)
    with pytest.raises(
        deferred.ConfiguredControlsDeferredInputError,
        match="configured_controls_deferred_fetch_mismatch",
    ):
        deferred.resolve_deferred_inputs(
            intent=intent,
            revision=revision,
            output_root=tmp_path / "progression",
            fetcher=lambda reference: b"junk",
        )


def test_deferred_book_to_tray_phases_match_native_destination_binding(tmp_path, monkeypatch):
    import copy
    from tests.test_task_evaluation_rigid_relocation_native_adapter import (
        DEFINITION, EXECUTION, STATIC, _rewrite,
    )
    from tests.test_task_evaluation_native_arena_episode_compiler import _destination_case
    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import _stage_destination_asset
    from blueprint_pipeline.task_evaluation_rigid_destination_geometry import bind_destination_trajectory
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import adapt_rigid_relocation_task_template

    launch, revision, references, docs = _case(tmp_path)
    definition = copy.deepcopy(docs[DEFINITION])
    definition["strategy"] = "pick_and_place"
    definition["interaction_affordance"] = {
        "contact_point_scoring_frame_m": [-.06, 0., 0.], "approach_unit_scoring_frame": [-1., 0., 0.],
        "jaw_unit_scoring_frame": [0., 1., 0.], "lift_unit_world": [0., 0., 1.],
        "pregrasp_clearance_m": .12, "minimum_lift_m": .08,
    }
    execution = copy.deepcopy(docs[EXECUTION])
    execution["strategy"] = "pick_and_place"
    for contract, document in ((DEFINITION, definition), (EXECUTION, execution)):
        _rewrite(tmp_path=tmp_path, configured=revision, references=references,
                 contract_path=contract, document=document)
    docs[STATIC]["observed_structure"]["collision_bounds_body_frame_m"] = {
        "minimum": [-.04, -.05, 0.], "maximum": [.04, .05, .127638]}
    docs[STATIC]["result_digest"] = canonical_digest(docs[STATIC], digest_field="result_digest")
    _rewrite(tmp_path=tmp_path, configured=revision, references=references,
             contract_path=STATIC, document=docs[STATIC])
    adapted = adapt_rigid_relocation_task_template(configured_revision=revision, materialized_references=references)
    native_spec = deferred._runtime_subject_task_spec({
        **adapted["native_task_definition"]["task_spec"],
        "subject_asset_id": revision["replacement"]["identity"]["id"],
        "success_criteria": adapted["native_success_criteria"]["criteria"],
    })
    destination_root = tmp_path / "destination"
    destination_root.mkdir()
    native_request, destination_refs, context = _destination_case(destination_root,
        subject_identity=revision["replacement"]["identity"],
        subject_static_path=Path(references[STATIC]["materialized_path"]), subject_static=docs[STATIC],
        subject_scoring_transform=native_spec["interaction_affordance"]["asset_root_from_scoring_frame"],
        configured_scene_support_plane_path=Path(references[deferred.SUPPORT_PLANE_CONTRACT_PATH]["materialized_path"]),
    )
    destination = dict(native_request["task"]["destination"])
    for name in ("asset", "rights_admission", "static_qualification", "native_import_qualification", "geometry"):
        destination[name] = {key: destination_refs["task.destination." + name][key]
                             for key in ("uri", "digest", "size_bytes")}
    destination["native_probe"] = {}  # CPU planning does not execute or qualify the native probe.
    revision["task_template"]["destination"] = destination
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    geometry = Path(destination_refs["task.destination.geometry"]["materialized_path"])
    documents = {key: Path(row["materialized_path"]) for key, row in references.items()}
    documents["task.destination.geometry"] = geometry
    captured = {}
    materialize = deferred.materialize_native_task_construction_phase_plan

    def observe(plan):
        captured["scene_plan"] = copy.deepcopy(plan)
        return materialize(plan)

    monkeypatch.setattr(deferred, "materialize_native_task_construction_phase_plan", observe)
    cpu_plan = deferred.derive_native_trajectory_plan(revision=revision, documents=documents)
    # The native compiler still performs full rights, static/native import,
    # geometry, scene placement, and exact support admission before this binding.
    native_out = tmp_path / "native"
    native_out.mkdir()
    admitted = _stage_destination_asset(request=native_request, materialized_references=destination_refs,
        output_root=native_out, configured_collision_path=context["configured_collision_path"], task_spec=native_spec)
    native_scene_plan = captured["scene_plan"]
    native_scene_plan["task_spec"] = bind_destination_trajectory(native_spec, admitted)
    native_scene_plan["plan_digest"] = canonical_digest(native_scene_plan, digest_field="plan_digest")
    native_plan = materialize(native_scene_plan)
    assert cpu_plan["phases"] == native_plan["phases"]
    assert captured["scene_plan"]["task_spec"]["interaction_affordance"]["insertion_withdrawal_unit_world"] == [0., 0., 1.]
    assert native_scene_plan["task_spec"]["target_position_world_m"] != native_spec["target_position_world_m"]


def test_cached_plan_is_rederived_and_rejects_self_sealed_tampering(tmp_path):
    _, revision, references, _ = _case(tmp_path)
    payloads = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()}
    intent = {"expected_production_commit": COMMIT,
              "paths": {"native_trajectory_plan_path": {"deferred": deferred.TRAJECTORY_MODE}}}
    resolved = deferred.resolve_deferred_inputs(intent=intent, revision=revision, output_root=tmp_path / "out",
                                                fetcher=lambda ref: payloads[ref["uri"]])
    path = Path(resolved["native_trajectory_plan_path"])
    value = json.loads(path.read_text())
    value["tampered"] = True
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    path.chmod(0o600)
    path.write_text(json.dumps(value))
    with pytest.raises(deferred.ConfiguredControlsDeferredInputError, match="plan_conflict"):
        deferred.resolve_deferred_inputs(intent=intent, revision=revision, output_root=tmp_path / "out",
                                         fetcher=lambda ref: pytest.fail("retained documents should be rehashed"))
    new_intent = {**intent, "expected_production_commit": "f" * 40}
    new = deferred.resolve_deferred_inputs(intent=new_intent, revision=revision, output_root=tmp_path / "out",
                                           fetcher=lambda ref: payloads[ref["uri"]])
    assert new["native_trajectory_plan_path"] != str(path)


def test_deferred_publish_failure_leaves_no_partial_replay_file(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    original = deferred.os.link

    def interrupted(*args):
        raise OSError("injected interruption before atomic publication")

    monkeypatch.setattr(deferred.os, "link", interrupted)
    with pytest.raises(OSError, match="injected interruption"):
        deferred._write_immutable_bytes(path, b"complete bytes", conflict="conflict")
    assert not path.exists()
    assert not list(tmp_path.glob(".deferred-*"))
    monkeypatch.setattr(deferred.os, "link", original)
    assert deferred._write_immutable_bytes(path, b"complete bytes", conflict="conflict") == path
