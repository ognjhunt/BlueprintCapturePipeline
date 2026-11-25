"""The capture's Atlas job is the only website scene source.

No Marble world is bought or read (owner decision 2026-10-22). Preparation waits
while the Atlas job runs and reports why it cannot proceed otherwise.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_object_local_frame import REGISTRATION_REFUSALS
from blueprint_pipeline.website_recapture_required import REGISTRATION_CODES


def inputs(tmp_path):
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir(exist_ok=True)
    removal = pipeline / "removal.json"
    write_json(removal, {"entries": []})
    return dict(
        descriptor={"capture_id": "cap-1", "scene_id": "scene-1", "metadata": {
            "site_task_context": {"capture_id": "cap-1", "scene_id": "scene-1"}}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "source_geometry": None,
                     "task_masks": {}, "removal_manifest_path": str(removal)},
        capture_root=tmp_path, now=1800000000,
    )


ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"


def atlas_ready(tmp_path):
    collider = tmp_path / "pipeline/atlas/background.glb"
    collider.parent.mkdir(parents=True, exist_ok=True)
    collider.write_bytes(b"atlas-collider")
    base = {"splat_path": str(collider), "splat_digest": _sha256_file(collider), "collision_mesh_path": str(collider),
            "collision_mesh_digest": _sha256_file(collider), "up_axis": "Y", "meters_per_unit": 0.8,
            "provider": "world_labs_atlas", "scale_authority": "atlas_reference_estimate",
            "ground_plane_offset_m": None, "anchor": {"kind": "source_world_alignment"}, "operation_id": "atlas-op"}
    return {"status": "ready", "base_scene": base, "alignment": "anchored",
            "source_geometry": {"digest": "atlas-geometry", "geometry_backend": "world_labs_atlas"},
            "source_pose_digest": "sha256:" + "1" * 64, "background_pose_digest": "sha256:" + "2" * 64}


def room_recorder(monkeypatch) -> list:
    """Each call the handoff makes to ``write_room_package``, for tests whose fixture collider is not a real room
    (the real room package: ``tests/test_website_scene_handoff_navigation``)."""
    calls: list = []
    monkeypatch.setattr(handoff, "write_room_package", lambda **kw: calls.append(kw) or {
        "inputs_digest": "sha256:" + "9" * 64, "task_kind": "not_defined_by_blueprint",
        "task_profile": kw["task_profile"] or "none", "success_rule_defined": False, "success_rule_assumed": False})
    return calls


def install_family_runtimes(tmp_path, monkeypatch) -> None:
    """A host whose runtime catalog registers every task family: a family job compiles its own route as before."""
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    from blueprint_pipeline.task_family_scoring import FAMILY_ALIASES
    from tests.test_website_early_task_holds import _catalog
    root = tmp_path / "runtime-catalog"
    root.mkdir(exist_ok=True)
    monkeypatch.setenv(CATALOG_PATH_ENV, str(_catalog(root, sorted(FAMILY_ALIASES))))


def retained_hold(context, gap) -> dict:
    """A hold an earlier release wrote at the handoff (as ``retain_gap`` wrote it), which still reads as held."""
    value = {"schema_version": "website_scene_handoff.v1", "status": "needs_input", "blockers": [gap["code"]],
             "scene_id": context["scene_id"], "capture_id": context["capture_id"], "simulator_ready": False,
             "visual_reconstruction_ready": True, "native_task_preparation_gap": gap}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


@pytest.fixture
def ready_atlas(tmp_path, monkeypatch):
    ready = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: ready)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: {**kw["task_masks"], "source_geometry_digest": "atlas-geometry"})
    return ready


def test_novel_family_ships_the_room_package_without_masks_authoring_or_a_substitute_task(
        tmp_path, monkeypatch, ready_atlas):
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family="unimplemented_task_family", description="Fold the towels", context_digest="sha256:" + "a" * 64)
    kwargs["clean_plate"]["task_masks"] = {"deferred_target_ids": ["towel"]}

    def unexpected(**_):
        pytest.fail("Native masks, authoring and runtime preparation must wait for an appropriate adapter")

    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks", unexpected)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", unexpected)
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", unexpected)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", unexpected)
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", unexpected)
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    # Not held since 2026-11-16 (any job is accepted): the room package, with no rule of Blueprint's.
    assert (result["status"], result["blockers"], result["job_disposition"]) == ("room_package_written", [], "accepted")
    [call] = packaged
    assert call["task_profile"] is None and call["task_masks"] is None and call["task_context"] == context
    assert result["room_package"]["success_rule_defined"] is False and "native_task_preparation_gap" not in result
    assert result["visual_reconstruction_ready"] is True and result["simulator_ready"] is False
    assert result["provider_mutation_performed"] is False
    assert "runtime_inputs" not in result and "website_intake_outbox" not in result
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/base_scene.json").read_text()) == ready_atlas["base_scene"]
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


@pytest.mark.parametrize("atlas_origin,has_synthetic", [("synthetic_design", False), ("captured", True)])
def test_unknown_family_cannot_hide_a_subject_origin_conflict(tmp_path, monkeypatch, ready_atlas,
                                                             atlas_origin, has_synthetic):
    from tests.test_website_task_preparation import SYNTHETIC_BALL
    kwargs = inputs(tmp_path)
    kwargs["descriptor"]["metadata"]["site_task_context"].update(task_family="unimplemented_task_family", description="Original job")
    ready_atlas["subject_origin"] = atlas_origin
    write_json(tmp_path / "pipeline/removal.json", {"entries": [],
        **({"synthetic_subject": dict(SYNTHETIC_BALL)} if has_synthetic else {})})

    def unexpected(**_):
        pytest.fail("Source-origin conflict precedes mask and provider callbacks")

    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", unexpected)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", unexpected)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["website_subject_origin_conflict"]
    assert "native_task_preparation_gap" not in result and "runtime_inputs" not in result
    assert result["provider_mutation_performed"] is False


def test_a_subject_origin_conflict_ships_the_room_flagged_once_the_retry_cycle_is_spent(tmp_path, monkeypatch,
                                                                                          ready_atlas):
    """The conflict is raised once the room can be packaged, so the retry-cycle fallback reaches it."""
    kwargs = inputs(tmp_path)
    kwargs["descriptor"]["metadata"]["site_task_context"]["context_digest"] = "sha256:" + "c" * 64
    ready_atlas["subject_origin"] = "synthetic_design"
    write_json(tmp_path / "pipeline/removal.json", {"entries": []})
    packaged = room_recorder(monkeypatch)
    monkeypatch.setattr(handoff, "first_retry_cycle_spent", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **_: pytest.fail("a conflict precedes the compile"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written" and result["blockers"] == []
    assert [call["flags"] for call in packaged] == [["website_subject_origin_conflict"]]


@pytest.mark.parametrize("family", ["rigid_pick_and_place", "articulated_open_close"])
def test_synthetic_subject_profile_ignores_its_supports_mechanism(tmp_path, monkeypatch, ready_atlas, family):
    from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
    from tests.test_website_task_preparation import SYNTHETIC_BALL
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(schema_version="website_site_task_context.v1", task_family=family, description="Original job")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    ready_atlas["subject_origin"] = "synthetic_design"
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": "table-1", "target_role": "support",
        "task_effect": "static_contact", "disposition": "keep", "articulation_kind": "prismatic"}]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": [], "synthetic_subject": dict(SYNTHETIC_BALL)})
    calls = []

    def coverage(**_):
        calls.append("coverage")
        raise ValueError("coverage_pending")

    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", coverage)
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", lambda **_: pytest.fail("No native intake was prepared"))
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    if family == "articulated_open_close":
        # Not held since 2026-11-16: the room package, flagged with the mismatch.
        assert result["status"] == "room_package_written" and not calls
        assert [call["flags"] for call in packaged] == [[handoff.MISMATCH]]
        gap = handoff.native_task_preparation_gap(context, inferred_profile_id="rigid_pick_place")
        assert gap["inferred_profile_id"] == "rigid_pick_place"
        assert holds_native_task_preparation(retained_hold(context, gap), context)
    else:
        # The coverage step was reached; a step still pending keeps the job waiting (``keeps_holding``).
        assert result["status"] == "awaiting_inputs" and result["blockers"] == ["coverage_pending"]
        assert calls == ["coverage"] and packaged == []
        assert "native_task_preparation_gap" not in result
    assert "runtime_inputs" not in result


@pytest.mark.parametrize("family,mechanism", [("articulated_open_close", ""), ("rigid_pick_and_place", "prismatic")])
def test_mechanism_mismatch_ships_the_room_before_coverage_and_native_preparation(tmp_path, monkeypatch, ready_atlas,
                                                                              family, mechanism):
    from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(schema_version="website_site_task_context.v1", task_family=family, description="The original job")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": "subject", "articulation_kind": mechanism}]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": [{"target_id": "subject", "task_effect": "manipulated"}]})

    def unexpected(**_):
        pytest.fail("Contradictory task semantics must wait before coverage, specs or native preparation")

    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", unexpected)
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", unexpected)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", unexpected)
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", unexpected)
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert (result["status"], result["blockers"]) == ("room_package_written", [])
    assert [(call["task_profile"], call["flags"]) for call in packaged] == [(family, [handoff.MISMATCH])]
    assert "runtime_inputs" not in result and "website_intake_outbox" not in result
    assert "provider_mutation_performed" not in result
    inferred = "rigid_pick_place" if family == "articulated_open_close" else "articulated_open_close"
    gap = handoff.native_task_preparation_gap(context, inferred_profile_id=inferred)
    assert holds_native_task_preparation(retained_hold(context, gap), context)  # a retained hold still reads


@pytest.mark.parametrize("family,mechanism", [("rigid_pick_and_place", ""), ("articulated_open_close", "prismatic")])
def test_matching_native_profile_continues_existing_preparation(tmp_path, monkeypatch, ready_atlas, family, mechanism):
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family=family, description="The original job")
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": "subject", "articulation_kind": mechanism}]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": [{"target_id": "subject", "task_effect": "manipulated"}]})

    def coverage(**_):
        raise ValueError("coverage_pending")

    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", coverage)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["coverage_pending"]
    assert "native_task_preparation_gap" not in result


@pytest.mark.parametrize("family,initial_mechanism,revised_mechanism", [
    ("articulated_open_close", "prismatic", ""), ("rigid_pick_and_place", "", "prismatic")])
def test_a_later_real_compiler_mismatch_is_retained_without_a_native_intake(
        tmp_path, monkeypatch, ready_atlas, family, initial_mechanism, revised_mechanism):
    from tests.test_website_task_preparation import _arguments
    from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
    kwargs, arguments = inputs(tmp_path), _arguments(tmp_path)
    context, masks = arguments["task_context"], arguments["task_masks"]
    geometry = arguments["source_geometry"]
    geometry["geometry_backend"] = "world_labs_atlas"
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks["source_geometry_digest"] = geometry["digest"]
    context["task_family"] = family
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    masks["targets"][0]["articulation_kind"] = initial_mechanism
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    kwargs["descriptor"].update(scene_id=context["scene_id"], capture_id=context["capture_id"])
    kwargs["descriptor"]["metadata"]["site_task_context"] = context
    kwargs["clean_plate"]["task_masks"] = masks
    write_json(tmp_path / "pipeline/removal.json", arguments["removal_manifest"])
    ready_atlas.update(base_scene=arguments["base_scene"], source_geometry=arguments["source_geometry"])
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])

    def revised_coverage(**kw):
        revised = json.loads(json.dumps(kw["task_masks"]))
        revised["targets"][0]["articulation_kind"] = revised_mechanism
        revised["digest"] = canonical_digest(revised, digest_field="digest")
        return revised

    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", revised_coverage)
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", lambda **kw: kw["task_masks"])
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", lambda **_: pytest.fail("A held compiler result has no native intake"))
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written"
    assert [call["flags"] for call in packaged] == [[handoff.MISMATCH]]
    preparation = json.loads((tmp_path / "pipeline/website_scene_preparation/preparation.json").read_text())
    assert preparation["intake_request"] is None
    assert holds_native_task_preparation(retained_hold(context, preparation["native_task_preparation_gap"]), context)
    assert "runtime_inputs" not in result and "provider_mutation_performed" not in result


def test_a_ready_atlas_scene_drives_preparation_but_never_invents_execution(tmp_path, monkeypatch, ready_atlas):
    kwargs = inputs(tmp_path)
    calls = []
    def compile(**value):
        calls.append(value)
        assert value["spend"]["max_total_spend_usd"] == 0
        assert value["base_scene"] == ready_atlas["base_scene"]
        return {"status": "needs_input", "blockers": ["website_scene_execution_authority_required"],
                "digest": "sha256:" + "a" * 64, "thumbnail": {"path": "proposed-thumbnail.png"}}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "needs_input" and len(calls) == 1
    assert result["simulator_ready"] is False and result["visual_reconstruction_ready"] is True
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


def test_runtime_inputs_read_the_collider_compile_sealed(tmp_path, monkeypatch, ready_atlas):
    # Compile flattens placement surfaces into a new collider and rewrites the base scene
    # (website_collider_conditioning); native inputs must bind that one, not the Atlas copy.
    kwargs = inputs(tmp_path)
    kwargs["descriptor"]["metadata"]["site_task_context"]["context_digest"] = "sha256:" + "c" * 64
    sealed = {**ready_atlas["base_scene"], "collision_mesh_digest": "sha256:" + "d" * 64}

    def compile(**value):
        assert value["base_scene"] == ready_atlas["base_scene"]
        write_json(value["output_root"] / "base_scene.json", sealed)
        return {"status": "needs_input", "blockers": ["website_scene_execution_authority_required"],
                "digest": "sha256:" + "a" * 64, "thumbnail": {}, "collider_conditioning": {"status": "conditioned"}}

    bound = []

    def runtime_inputs(**value):
        bound.append(value["base_scene"])
        raise ValueError("stopped_after_binding")

    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile)
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", runtime_inputs)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert bound == [sealed]
    assert result["runtime_inputs"]["blockers"] == ["stopped_after_binding"]


@pytest.mark.parametrize("blockers", [["website_scene_execution_authority_required"],
                                      ["task_destination_surface_contact_required"]])
@pytest.mark.parametrize("sealed_by", [("destination_surface_estimate", "estimated"),
                                       ("collider_conditioning", "conditioned"), ("subject_excision", "excised")])
def test_every_package_reads_the_collider_holding_an_estimated_destination_surface(
        tmp_path, monkeypatch, ready_atlas, blockers, sealed_by):
    # Compile added a destination surface the scan missed (glass) to the collider, or flattened the surfaces objects
    # rest on, and sealed it in the base scene: native inputs bind that one, and so does the room package a job
    # ships when its object still does not fit, so its README is true of its collision file. A collider with the
    # job's own object cut out (geometry only) is never a room package's: that one keeps the objects in its scan.
    kwargs = inputs(tmp_path)
    kwargs["descriptor"]["metadata"]["site_task_context"]["context_digest"] = "sha256:" + "c" * 64
    sealed = {**ready_atlas["base_scene"], "collision_mesh_digest": "sha256:" + "e" * 64}

    def compile(**value):
        assert value["base_scene"] == ready_atlas["base_scene"]
        write_json(value["output_root"] / "base_scene.json", sealed)
        return {"status": "needs_input", "blockers": blockers, "digest": "sha256:" + "a" * 64, "thumbnail": {},
                sealed_by[0]: {"status": sealed_by[1]}}

    bound = []

    def runtime_inputs(**value):
        bound.append(value["base_scene"])
        raise ValueError("stopped_after_binding")

    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile)
    monkeypatch.setattr(handoff, "prepare_website_runtime_inputs", runtime_inputs)
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    if result["status"] == handoff.ROOM_PACKAGE:
        expected = ready_atlas["base_scene"] if sealed_by[1] == "excised" else sealed
        assert [call["base_scene"] for call in packaged] == [expected] and bound == []
    else:
        assert bound == [sealed] and packaged == []
    assert result["status"] == (handoff.ROOM_PACKAGE if "task_destination_surface_contact_required" in blockers
                                else "needs_input")


@pytest.mark.parametrize("deferred_masks", [False, True])
@pytest.mark.parametrize("atlas,status", [
    ({"status": "needs_input", "reason": "website_atlas_job_required"}, "needs_input"),
    ({"status": "needs_input", "reason": "website_atlas_background_unavailable"}, "needs_input"),
    ({"status": "pending", "reason": "website_atlas_job_processing", "stage": "pose"}, "awaiting_reconstruction"),
])
def test_preparation_waits_for_the_capture_atlas_job_with_no_other_scene(
        tmp_path, monkeypatch, deferred_masks, atlas, status):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    kwargs = inputs(tmp_path)
    kwargs["clean_plate"]["task_masks"] = {
        "deferred_target_ids": ["support"] if deferred_masks else []}
    def unexpected(**_):
        pytest.fail("a pending Atlas job must not invoke masks or the compiler")
    monkeypatch.setattr(ATLAS, lambda **_: atlas)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks", unexpected)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", unexpected)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == status and result["blockers"] == [atlas["reason"]]
    assert result["visual_reconstruction_ready"] is False and "base_scene_path" not in result
    assert result["provider_mutation_performed"] is False
    assert result["simulator_ready"] is False and result["geometry_backend"] is None
    assert result["atlas_scene_source"]["status"] == atlas["status"]
    assert result["native_geometry_requirements"] == {
        "source_aligned_depth_and_camera_poses": "required_for_mask_lifting_and_scene_registration",
        "measured_scale": "unknown", "atlas_required": True}
    assert not (tmp_path / "pipeline/website_scene_preparation/base_scene.json").exists()
    assert result["digest"] == canonical_digest(result, digest_field="digest")
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


@pytest.mark.parametrize("change,reason", [
    ("privacy", "website_scene_preparation_pending"),
    ("identity", "website_scene_task_identity_mismatch"),
])
def test_held_or_changed_inputs_do_not_enter_scene_construction(tmp_path, monkeypatch, change, reason):
    kwargs = inputs(tmp_path)
    if change == "privacy":
        kwargs["clean_plate"]["privacy_verified"] = False
    else:
        kwargs["descriptor"]["metadata"]["site_task_context"]["capture_id"] = "other"
    monkeypatch.setattr(ATLAS, lambda **_: pytest.fail("held inputs must not read the Atlas job"))
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: pytest.fail("must not compile"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == [reason]
    assert result["simulator_ready"] is False


def test_a_ready_atlas_job_is_the_scene_and_its_pose_the_geometry_without_marble(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    masks = {"targets": [{"target_id": "object"}], "digest": "sha256:" + "b" * 64}
    kwargs["clean_plate"].update(task_masks=masks)
    ready = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **kw: ready)
    bound = {**masks, "source_geometry_digest": "atlas-geometry"}
    seen = {}
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: seen.update(bind=kw) or bound)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(compile=value) or (_ for _ in ()).throw(ValueError("stop")))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    # An untyped "stop" is read as Blueprint's defect: the job waits under its code (``blocker_code``).
    assert result["blockers"] == [handoff.DEFECT + ":ValueError"] and result["defect"] == "ValueError: stop"
    assert result["geometry_backend"] == "world_labs_atlas"
    assert seen["bind"] == {"task_masks": masks, "source_geometry": ready["source_geometry"]}
    assert seen["compile"]["base_scene"] == ready["base_scene"]
    assert seen["compile"]["source_geometry"] == ready["source_geometry"] and seen["compile"]["task_masks"] == bound
    assert result["atlas_scene_source"]["alignment"] == "anchored"
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/base_scene.json").read_text()) == ready["base_scene"]
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/task_masks.geometry.json").read_text()) == bound


@pytest.mark.parametrize("support_ready", [False, True])
def test_full_masks_are_required_before_they_lift_onto_atlas_geometry(tmp_path, monkeypatch, support_ready):
    kwargs = inputs(tmp_path)
    plan = tmp_path / "pipeline/plan.json"
    write_json(plan, {"targets": [{"target_id": "support"}]})
    kwargs["clean_plate"].update(source_frames={"digest": "source"},
        removal_plan_path=str(plan), task_masks={"deferred_target_ids": ["support"]})
    monkeypatch.setattr(ATLAS, lambda **kw: atlas_ready(tmp_path))
    calls = []
    full_masks = {"targets": [{"target_id": "object"}, {"target_id": "support"}]}
    def resolve(**kw):
        calls.append("masks")
        assert kw["source_geometry"] == {"digest": "source"}
        if not support_ready:
            raise ValueError("support_unresolved")
        return full_masks
    def bind(**kw):
        calls.append("bind")
        assert kw["task_masks"] == full_masks
        raise ValueError("binding_pending")
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks", resolve)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", bind)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["visual_reconstruction_ready"] is True and result["simulator_ready"] is False
    assert result["blockers"] == ["binding_pending" if support_ready else "support_unresolved"]
    assert calls == (["masks", "bind"] if support_ready else ["masks"])


def test_a_destination_kept_as_one_frame_lifts_with_the_video_track_its_mask_matches(tmp_path, monkeypatch):
    """Job move-the-lamp-2 (24 Nov): the glass shelf was kept as one frame's mask and the estimate of a surface the
    scan missed refused one frame. The retained video track that mask matches is named from the completion's own
    retained inputs (``website_task_masks.surface_views``) and lifted with the masks, for compile to read."""
    kwargs = inputs(tmp_path)
    plan = tmp_path / "pipeline/plan.json"
    write_json(plan, {"targets": [{"target_id": "shelf"}]})
    first = {"deferred_target_ids": ["shelf"]}
    kwargs["clean_plate"].update(source_frames={"digest": "source"}, removal_plan_path=str(plan), task_masks=first)
    monkeypatch.setattr(ATLAS, lambda **kw: atlas_ready(tmp_path))
    completed = {"targets": [{"target_id": "object"}, {"target_id": "shelf"}]}
    named = {"shelf": {"basis": "retained_video_track_matching_the_one_frame_mask"}}
    calls = []

    def views(**kw):
        calls.append("views")
        assert kw == {"plan": {"targets": [{"target_id": "shelf"}]}, "source_geometry": {"digest": "source"},
                      "task_masks": first, "output_root": plan.parent / "task_masks",
                      "view_plan_root": plan.parent / "mask_view_plan", "completed": completed}
        return named

    def bind(**kw):
        calls.append("bind")
        assert kw["task_masks"] == completed and kw["surface_views"] is named
        raise ValueError("binding_pending")

    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks",
                        lambda **_: completed)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.surface_views", views)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", bind)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["binding_pending"]
    assert calls == ["views", "bind"]


@pytest.mark.parametrize("mode", ["default", "enabled", "missing", "outside", "symlink"])
def test_retained_static_recovery_requires_explicit_execution_and_capture_source(tmp_path, monkeypatch, mode):
    kwargs = inputs(tmp_path)
    plan = tmp_path / "pipeline/plan.json"
    write_json(plan, {"targets": [{"target_id": "support"}]})
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"original source")
    if mode == "outside":
        video = tmp_path.parent / "outside-capture.mov"
    elif mode == "symlink":
        link = tmp_path / "linked.mov"
        link.symlink_to(video)
        video = link
    kwargs["clean_plate"].update(source_frames={"digest": "source"}, removal_plan_path=str(plan),
                                task_masks={"deferred_target_ids": ["support"]})
    if mode != "missing":
        kwargs["clean_plate"]["input_video_path"] = str(video)
    if mode != "default":
        kwargs["allow_static_target_recovery"] = True
    ready = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: ready)
    calls = []

    def resolve(**options):
        calls.append(options)
        if mode == "enabled":
            assert options["task_context"] == kwargs["descriptor"]["metadata"]["site_task_context"]
            assert options["source_video"] == video
            assert options["recovery_geometry"] == ready["source_geometry"]
        else:
            assert not {"task_context", "source_video", "recovery_geometry"} & options.keys()
        raise ValueError("checked_static_recovery_boundary")

    monkeypatch.setattr("blueprint_pipeline.website_task_masks.complete_retained_static_task_masks", resolve)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["checked_static_recovery_boundary"]
    assert len(calls) == 1


def test_articulated_targets_gain_coverage_from_the_retained_video_before_compile(tmp_path, monkeypatch,
                                                                                   ready_atlas):
    kwargs = inputs(tmp_path)
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"source")
    masks = {"targets": [{"target_id": "dishwasher-1"}], "digest": "sha256:" + "b" * 64}
    kwargs["clean_plate"].update(task_masks=masks, input_video_path=str(video))
    seen, attached = {}, {**masks, "targets": [{"target_id": "dishwasher-1", "authoring_coverage": {"status": "complete"}}]}

    def attach(**value):
        seen["attach"] = value
        return attached
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", attach)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(compile=value) or (_ for _ in ()).throw(ValueError("stop")))
    handoff.prepare_website_scene_handoff(**kwargs)
    bound = {**masks, "source_geometry_digest": "atlas-geometry"}
    assert seen["attach"]["task_masks"] == bound and seen["attach"]["source_video"] == video
    assert seen["attach"]["source_geometry"] == ready_atlas["source_geometry"]
    assert seen["attach"]["task_context"] == kwargs["descriptor"]["metadata"]["site_task_context"]
    assert seen["attach"]["output_root"] == tmp_path / "pipeline/website_scene_preparation/assembly_coverage"
    assert seen["compile"]["task_masks"] == attached
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/task_masks.coverage.json").read_text()) == attached
    # A video outside the capture is never read.
    outside = tmp_path.parent / (tmp_path.name + "-outside.mov")
    outside.write_bytes(b"other")
    kwargs["clean_plate"]["input_video_path"] = str(outside)
    handoff.prepare_website_scene_handoff(**kwargs)
    assert seen["attach"]["source_video"] is None


def test_identified_targets_gain_published_specs_after_coverage_and_before_compile(tmp_path, monkeypatch,
                                                                                   ready_atlas):
    kwargs = inputs(tmp_path)
    masks = {"targets": [{"target_id": "dishwasher-1"}], "digest": "sha256:" + "b" * 64}
    kwargs["clean_plate"].update(task_masks=masks)
    covered = {**masks, "targets": [{"target_id": "dishwasher-1", "authoring_coverage": {"status": "complete"}}]}
    specified = {**covered, "targets": [{**covered["targets"][0], "object_spec": {"status": "researched"}}]}
    seen = {}
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", lambda **_: covered)

    def attach(**value):
        seen["attach"] = value
        return specified
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", attach)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **value: seen.update(compile=value) or (_ for _ in ()).throw(ValueError("stop")))
    handoff.prepare_website_scene_handoff(**kwargs)
    assert seen["attach"]["task_masks"] == covered
    assert seen["attach"]["task_context"] == kwargs["descriptor"]["metadata"]["site_task_context"]
    assert seen["attach"]["output_root"] == tmp_path / "pipeline/website_scene_preparation/object_spec"
    assert seen["compile"]["task_masks"] == specified
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/task_masks.coverage.json").read_text()) == specified


def test_records_a_base_scene_from_another_atlas_result_named_are_kept_once_before_they_are_replaced(tmp_path):
    preparation = tmp_path / "pipeline/website_scene_preparation"
    root = preparation / "revisions" / ("a" * 64)
    parent, rebuilt = "sha256:" + "1" * 64, "sha256:" + "2" * 64
    write_json(root / "base_scene.json", {"atlas_result_digest": parent})
    write_json(root / "native/runtime_inputs.json", {"bound": "old collider"})
    write_json(preparation / "handoff.json", {"status": "intake_ready"})
    (root / "level").mkdir()  # content-addressed outputs stay where they are; they are not copied
    old = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    # The same result: nothing to keep.
    assert handoff.retain_superseded_records(capture_root=tmp_path, root=root, atlas_result_digest=parent) is None
    archive = handoff.retain_superseded_records(capture_root=tmp_path, root=root, atlas_result_digest=rebuilt)
    assert archive == preparation / handoff.SUPERSEDED_DIRECTORY / parent[7:23]
    copied = {path.relative_to(archive).as_posix(): path.read_bytes() for path in archive.rglob("*") if path.is_file()}
    assert copied == {"handoff.json": old[preparation / "handoff.json"],
                      f"revisions/{'a' * 64}/base_scene.json": old[root / "base_scene.json"],
                      f"revisions/{'a' * 64}/native/runtime_inputs.json": old[root / "native/runtime_inputs.json"]}
    # Kept once: a later run (or one after a crash) never replaces what was kept, and nothing was moved.
    write_json(preparation / "handoff.json", {"status": "changed by a later run"})
    handoff.retain_superseded_records(capture_root=tmp_path, root=root, atlas_result_digest=rebuilt)
    assert (archive / "handoff.json").read_bytes() == old[preparation / "handoff.json"]
    assert all(path.is_file() for path in old)


@pytest.mark.parametrize("pending", [True, False])
def test_an_unresolved_reading_waits_only_while_a_clarification_can_still_resolve_it(tmp_path, monkeypatch,
                                                                                    ready_atlas, pending):
    """The default (observational) handoff buys no clarification: a reading one could still resolve keeps its
    hold for the capture stage's run, which buys it; one nothing can resolve ships the room package."""
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family="general_task", description="Restock the shelves.")
    ready_atlas["source_geometry"]["binding"] = {"source_video_digest": "sha256:" + "5" * 64}
    gap = {"code": "website_task_operation_coverage_unresolved", "task_family": "general_task"}
    monkeypatch.setattr("blueprint_pipeline.website_task_semantics.load_task_semantics", lambda **_: {"reading": 1})
    monkeypatch.setattr(handoff, "native_task_preparation_gap", lambda *_, **__: gap)
    monkeypatch.setattr("blueprint_pipeline.website_operation_coverage.clarification_request",
                        lambda *_, **__: {"scope": "operation_coverage"} if pending else None)
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    if pending:
        assert (result["status"], result["blockers"], packaged) == ("needs_input", [gap["code"]], [])
    else:
        assert result["status"] == "room_package_written" and len(packaged) == 1


class _Compiled(Exception):
    pass


@pytest.mark.parametrize("entries, profile, built", [
    ([{"target_id": "box", "task_effect": "manipulated"}], "rigid_pick_place", True),
    ([{"target_id": "box", "task_effect": "manipulated"}], "articulated_open_close", False),
    ([{"target_id": "box", "task_effect": "manipulated"}, {"target_id": "cup", "task_effect": "manipulated"}],
     "rigid_pick_place", False),
    ([], "rigid_pick_place", False),
])
def test_one_removed_object_of_a_job_no_profile_models_is_built_with_its_rule_withheld(tmp_path, monkeypatch,
                                                                                       ready_atlas, entries, profile,
                                                                                       built):
    """Objects step 1: the object is built on the rigid path, the rule withheld for a named, sealed reason; several
    objects, none, or a part that opens, still ship as the room package."""
    kwargs = inputs(tmp_path)
    write_json(tmp_path / "pipeline/removal.json", {"entries": entries})
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family="unimplemented_task_family", description="Do the thing with the box",
                   context_digest="sha256:" + "a" * 64)
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": row["target_id"]} for row in entries]}
    corrections = []
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: corrections.append(1) or {"status": "ready"})
    monkeypatch.setattr(handoff, "native_task_subject_profile", lambda *_, **__: (None, profile))
    monkeypatch.setattr(handoff, "subject_rigidity", lambda *_, **__: None)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **kw: kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs",
                        lambda **kw: kw["task_masks"])
    compiled = []
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **kw: compiled.append(kw) or (_ for _ in ()).throw(_Compiled()))
    packaged = room_recorder(monkeypatch)
    if not built:
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "room_package_written"
        assert not compiled
        assert "rule_withheld" not in result
        assert not corrections  # decided before anything is queued, as before
        return
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    with pytest.raises(_Compiled):  # the handoff again (a redelivery or a release): the same mark, nothing new
        handoff.prepare_website_scene_handoff(**kwargs)
    first, again = compiled
    assert first["rule_withheld"] == again["rule_withheld"] == {
        "code": "website_native_task_profile_preparation_required", "task_id": "website-" + "a" * 20}
    assert not packaged
    compiled.clear()
    # With no place on its support to run the internal rigid path toward, the object is not built: the room package.
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **kw: compiled.append(kw) or {"status": "needs_input", "blockers": ["task_destination_pose_required"]})
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written"
    assert len(packaged) == 1
    assert "rule_withheld" not in result
    assert len(compiled) == 1


def _through_compile(tmp_path, monkeypatch, *, family="unimplemented_task_family", place=True, refusal=None,
                     stop=True, spend=None, rights=None):
    """One removed rigid object through the handoff and the real compile (on that compile's own test geometry,
    ``tests/test_website_task_preparation``), with the provider choice the handoff hands compile recorded.
    ``refusal``: the registration refuses with that code. ``stop``: end the handoff once compile returns (what
    follows is the existing rigid path; ``"ready"``: only once compile returns an intake-ready preparation).
    ``spend``: the grant compile reads, kept by reference so a later pass can change it. ``rights``: the
    context's capture rights."""
    from blueprint_pipeline import website_task_preparation as preparation
    from blueprint_pipeline.website_registration_admissibility import RegistrationRefusal
    from tests.test_website_task_preparation import _arguments, _masks
    kwargs = inputs(tmp_path)
    write_json(tmp_path / "pipeline/removal.json", {"entries": [{"target_id": "box", "task_effect": "manipulated"}]})
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family=family, description="Do the thing with the box", context_digest="sha256:" + "a" * 64)
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": "box"}]}
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    monkeypatch.setattr(handoff, "native_task_subject_profile", lambda *_, **__: (None, "rigid_pick_place"))
    monkeypatch.setattr(handoff, "subject_rigidity", lambda *_, **__: None)
    for name in ("website_assembly_coverage.attach_assembly_coverage",
                 "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: kw["task_masks"])
    (tmp_path / "geometry").mkdir()
    fixture = _arguments(tmp_path / "geometry", lambda geometry: {"task_masks": _masks(geometry, destination=place)})
    fixture["task_context"].update(task_family=family, **({"capture_rights": rights} if rights is not None else {}))
    fixture["spend"] = fixture["spend"] if spend is None else spend
    fixture["task_context"]["context_digest"] = canonical_digest(fixture["task_context"], digest_field="context_digest")
    if not place:  # no named destination and no spot left on the start surface
        monkeypatch.setattr(preparation, "farthest_supported_destination", lambda *_, **__: None)
    if refusal is not None:
        def refuse(**_):
            raise RegistrationRefusal(refusal, {"best_score": 0.1})
        monkeypatch.setattr(preparation, "register_source_to_runtime", refuse)
    asked, compiled = [], []
    monkeypatch.setattr(handoff, "artisan_provider_choice_for", lambda **kw: asked.append(kw))
    real = handoff.compile_website_scene_preparation

    def compile_on_its_geometry(**kw):
        compiled.append(real(**{**fixture, "output_root": kw["output_root"]},
                             **{key: kw[key] for key in ("rule_withheld", "artisan_provider_choice") if key in kw}))
        if stop is True or (stop == "ready" and compiled[-1]["status"] == "intake_ready"):
            raise _Compiled
        return compiled[-1]
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile_on_its_geometry)
    return kwargs, asked, compiled, room_recorder(monkeypatch)


@pytest.mark.parametrize("place", [True, False], ids=["object_built", "no_place_room_package"])
def test_the_object_of_a_job_no_profile_models_chooses_its_provider_once_and_its_room_package_never(
        tmp_path, monkeypatch, ready_atlas, place):
    """Objects step 1: the one object of a job no task profile models is authored, so the provider choice the
    handoff hands compile is asked for exactly once, after the one-rigid-object decision and every gap. With no
    internal place the job ships as the room package: the choice is never asked for, so no choice file, ledger
    entry or model read exists."""
    kwargs, asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, place=place, stop=place)
    if place:
        with pytest.raises(_Compiled):
            handoff.prepare_website_scene_handoff(**kwargs)
        assert compiled[0]["intake_request"] is not None
        assert len(asked) == 1
        return
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert "task_destination_pose_required" in compiled[0]["blockers"]
    assert result["status"] == "room_package_written"
    assert [call["flags"] for call in packaged] == [compiled[0]["blockers"]]
    assert asked == []


@pytest.mark.parametrize("code", sorted(REGISTRATION_REFUSALS | {"website_front_normal_undetermined"}))
def test_an_object_that_does_not_register_leaves_a_withheld_job_its_room_package_and_never_a_hold(
        tmp_path, monkeypatch, ready_atlas, code):
    """The room package is the floor of a job whose rule is withheld: an object the scene cannot place is left
    out, the code flagged on the room record; no hold, no new video asked for, no provider choice. Over two
    passes: the held-handoff replay finds no recapture hold to look at again, and the next handoff ends the same."""
    from blueprint_pipeline.website_held_handoff_replay import registration_recapture_code
    kwargs, asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, refusal=code)
    for _ in range(2):
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "room_package_written"
        assert "recapture_required" not in result
        assert packaged[-1]["flags"] == [code]
        assert registration_recapture_code(tmp_path, result, kwargs["descriptor"]["metadata"]["site_task_context"]) is None
    assert compiled == [] and asked == [] and len(packaged) == 2
    assert list(tmp_path.rglob("registration_refusal.json"))  # its scores stay written, as today


@pytest.mark.parametrize("code", sorted(REGISTRATION_CODES | {"website_front_normal_undetermined"}))
def test_a_modelled_job_whose_object_does_not_register_ships_its_room_flagged_and_asks_no_new_video(
        tmp_path, monkeypatch, ready_atlas, code):
    """The room was built before the fit or facing check: the job ships it, flagged, and the site is not asked to
    film again (section 8 of the retry doc). Nothing is chosen or bought for the object."""
    kwargs, asked, _, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", refusal=code)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written"
    assert "recapture_required" not in result
    assert packaged[-1]["flags"] == [code] and asked == []


@pytest.mark.parametrize("code", ["website_capture_frames_unusable", "website_task_object_not_observed"])
def test_a_stop_found_before_any_room_still_asks_for_a_new_video(code):
    assert not handoff.goes_on([code], task_context={"context_digest": "sha256:" + "0" * 64}, now=0.0,
                               window_end=lambda: 0.0)


def test_development_test_mode_still_places_a_withheld_job_s_object_that_does_not_register(
        tmp_path, monkeypatch, ready_atlas):
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: True)
    kwargs, _, compiled, packaged = _through_compile(tmp_path, monkeypatch, refusal="website_registration_poor_fit",
                                                     stop=False)
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation",
                        lambda *_, **__: (_ for _ in ()).throw(_Compiled()))  # past the room-package floor
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert "website_registration_poor_fit" in compiled[0]["blockers"] and packaged == []


def _unmet(code, spend):
    """The grant or context with one authoring admission unmet (``authoring_admission_blockers``)."""
    if code == "website_scene_execution_authority_required":
        spend.pop("owner")
    elif code == "website_anthropic_provider_terms_authority_required":
        spend.update(authoring_provider="anthropic", replacement_authoring_backend="astra_cad_blender_v1")
    elif code == "website_agents_api_signed_authority_required":
        spend.update(authoring_agent_runtime="openai_agents_api")
    return {"derived_scene_generation_allowed": False} if code == "website_scene_processing_rights_required" else None


ADMISSIONS = ("website_scene_execution_authority_required", "website_scene_processing_rights_required",
              "website_agents_api_signed_authority_required", "website_anthropic_provider_terms_authority_required")


@pytest.mark.parametrize("code", ADMISSIONS)
def test_an_unmet_authoring_admission_leaves_a_withheld_job_its_room_package_and_spends_nothing(
        tmp_path, monkeypatch, ready_atlas, code):
    """Each admission the object's authoring needs stops the object, not the job: with one unmet the job ships
    the room package main ships it today, the object left out and the code a flag. Nothing is spent or sent for
    the object: no provider choice, no provider named in the withdrawal ledger, no rights admission, no intake.
    A later pass with the admission met builds the object: its choice made then, once, and no second package."""
    from tests.test_website_task_preparation import SPEND
    spend = json.loads(json.dumps(SPEND))
    rights = _unmet(code, spend)
    kwargs, asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, stop="ready", spend=spend,
                                                         rights=rights)
    named = []
    monkeypatch.setattr(handoff, "name_anthropic_for_site", lambda *a, **k: named.append(a) or True)
    monkeypatch.setattr("blueprint_pipeline.website_scene_dispatch.register_website_preparation",
                        lambda **_: pytest.fail("registered"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert code in compiled[0]["blockers"]
    assert result["status"] == "room_package_written"
    assert "website_intake_outbox" not in result
    assert [{key: call[key] for key in ("task_profile", "task_masks", "objects_in_scan", "flags")}
            for call in packaged] == [{"task_profile": None, "task_masks": None, "objects_in_scan": False,
                                       "flags": compiled[0]["blockers"]}]
    assert asked == []
    assert named == []
    assert not list(tmp_path.rglob("rights_admission.json"))
    if rights is not None:
        return  # the context's rights are the site's own; a new context is a new job
    spend.clear()
    spend.update(json.loads(json.dumps(SPEND)))  # the grant arrives: the next pass builds the object
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert compiled[-1]["intake_request"] is not None
    assert len(asked) == 1
    assert len(packaged) == 1


@pytest.mark.parametrize("code", ADMISSIONS)
def test_a_modelled_job_with_an_unmet_authoring_admission_waits_as_today(tmp_path, monkeypatch, ready_atlas, code):
    from tests.test_website_task_preparation import SPEND
    spend = json.loads(json.dumps(SPEND))
    rights = _unmet(code, spend)
    kwargs, _, compiled, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", stop=False,
                                                     spend=spend, rights=rights)
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation",
                        lambda *_, **__: (_ for _ in ()).throw(_Compiled()))  # held as before, never the room
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert code in compiled[0]["blockers"]
    assert packaged == []


@pytest.mark.parametrize("written", [True, False])
def test_an_intake_ready_object_job_keeps_its_room_floor_and_a_floor_that_fails_stops_nothing(
        tmp_path, monkeypatch, ready_atlas, written):
    """The package floor (``website_company_simulation.object_path_stopped``): written beside the object's route
    once the preparation is intake ready, flagged; a floor that cannot be written is recorded and never a stop."""
    kwargs, _, compiled, packaged = _through_compile(tmp_path, monkeypatch, stop=False)
    # The scene source's own flag (gap D): a floor says the same true things a room package would.
    monkeypatch.setattr(ATLAS, lambda **_: {**ready_atlas, "flags": ["scene_completion_not_admitted"]})
    if not written:
        monkeypatch.setattr(handoff, "write_room_package",
                            lambda **kw: packaged.append(kw) or (_ for _ in ()).throw(ValueError("website_base_scene_changed")))
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled",
                        lambda *_: (_ for _ in ()).throw(_Compiled()))  # past the floor: the object's own route
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert compiled[0]["status"] == "intake_ready"
    assert [(call["output_root"].name, call["flags"], call["task_profile"]) for call in packaged] == [
        (handoff.ROOM_FLOOR, [handoff.ROOM_FLOOR_FLAG, "scene_completion_not_admitted"], None)]


# --- 2026-11-17: two test jobs stopped in the preparation half; neither code holds a job any more -----------------

RECEIPTS = "pipeline/website_scene_preparation/assembly_coverage/drawer_cabinet_assembly/receipts"


class GeminiClientError(Exception):
    """``google.genai.errors.ClientError``'s shape: ``code``, ``status``, ``message``, ``details``, and its text."""

    def __init__(self, code, status, message):
        self.code, self.status, self.message = code, status, message
        self.details = {"error": {"code": code, "message": message, "status": status}}
        super().__init__(f"{code} {status}. {self.details}")


def depleted():
    """Gemini's answer to test job 3's three dispatches (20:12, 20:16 and 20:22Z, 17 Nov)."""
    return GeminiClientError(402, "RESOURCE_EXHAUSTED", "Your prepayment credits are depleted. Please go to AI "
                             "Studio at https://ai.studio/projects to manage your project and billing.")


def _drawer_job(tmp_path, monkeypatch):
    """Test job 3's handoff up to its assembly views: one articulated drawer unit, an object path to build."""
    kwargs = inputs(tmp_path)
    write_json(tmp_path / "pipeline/removal.json", {"entries": [
        {"target_id": "drawer_cabinet_assembly", "task_effect": "manipulated", "articulation_kind": "prismatic"}]})
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family="articulated_open_close", description="Open the top drawer",
                   context_digest="sha256:" + "c" * 64)
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": "drawer_cabinet_assembly"}]}
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    monkeypatch.setattr(handoff, "native_task_subject_profile", lambda *_, **__: ("prismatic", "articulated_open_close"))
    monkeypatch.setattr(handoff, "subject_rigidity", lambda *_, **__: None)
    return kwargs


def _sent_three_times(tmp_path, monkeypatch, answer, *, ledger=False):
    """One request of the assembly views sent three times and answered by ``answer()`` each time. ``ledger``: as
    the release before 2026-11-17 recorded it (a 402 kept no answer on the receipt, and the 4th run counted the
    third and sent nothing), the job ledger keeping each run's failure text. Returns the call arguments and the
    calls made since."""
    from datetime import datetime, timezone

    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission
    from tests.test_website_gemini_receipts import setup_call
    rows = []

    def now():
        return datetime.now(timezone.utc).isoformat()
    with pytest.MonkeyPatch.context() as release:
        args, calls = setup_call(tmp_path / RECEIPTS, release)
        reserve = receipts.reserve_website_preparation_spend

        def in_window(**kw):  # the job's window: a week after the handoff's ``now`` (``inputs``)
            admission, _grant = reserve(**kw)
            admission = {**admission, "expires_at_epoch": 1800000000 + 7 * 86400}
            return admission, require_paid_resource_admission(admission, resource_class=kw["resource_class"],
                                                              expected_schema_version=admission["schema_version"])
        release.setattr(receipts, "reserve_website_preparation_spend", in_window)
        if ledger:
            release.setattr(receipts, "credit_depleted", lambda _exc: False)  # the release that ran them

        def answering():
            calls.append(("invoke", None))
            raise answer()
        args["invoke"] = answering
        for attempt in range(10, 13):
            started = now()
            with pytest.raises(Exception) as raised:
                receipts.retained_gemini_call(**args)
            rows.append({"attempt_number": attempt, "status": "failed_retryable", "stage": "run_e2e",
                         "started_at": started, "failed_at": now(), "error_type": "PipelineError",
                         "error": str(raised.value)})
        if not ledger:
            return args, calls
        started = now()
        with pytest.raises(ValueError, match=receipts.EXHAUSTED):  # attempt 13 at 20:32:02Z, nothing sent
            receipts.retained_gemini_call(**args)
        rows.append({"attempt_number": 13, "status": "failed_retryable", "stage": "run_e2e", "started_at": started,
                     "failed_at": now(), "error_type": "PipelineError",
                     "error": "website_scene_preparation: " + receipts.EXHAUSTED})
    write_json(tmp_path / "pipeline_job_ledger.json", {"status": "failed_retryable", "attempt_history": rows})
    return setup_call(tmp_path / RECEIPTS, monkeypatch)  # the same call on the release with this change


def test_job3_a_genuine_non_answer_past_the_bound_ships_the_room_package_and_buys_nothing(
        tmp_path, monkeypatch, ready_atlas):
    """A request sent three times with no answer (the call timed out): the 4th run stops at
    ``website_gemini_unanswered_dispatches_exhausted``. The job goes on as the room package, flagged with that code,
    and nothing is reserved, sent, researched or compiled, on this run or the next."""
    from blueprint_pipeline import website_gemini_receipts as receipts
    args, calls = _sent_three_times(tmp_path, monkeypatch, lambda: TimeoutError("no answer"))
    assert [row[0] for row in calls] == ["reserve", "invoke"] * 3
    kwargs = _drawer_job(tmp_path, monkeypatch)
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", lambda **_: pytest.fail("reserved again"))
    args["invoke"] = lambda: pytest.fail("sent again")
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: receipts.retained_gemini_call(**args))  # the real chain, as the step reads it
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs",
                        lambda **_: pytest.fail("researched"))
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: pytest.fail("compiled"))
    packaged = room_recorder(monkeypatch)
    for run in (1, 2):
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "room_package_written" and result["blockers"] == []
        assert result["job_disposition"] == "accepted"
        assert len(packaged) == run and packaged[-1]["flags"] == [receipts.EXHAUSTED]
        assert packaged[-1]["task_profile"] is None
    assert len(calls) == 6
    assert len(list((tmp_path / RECEIPTS).glob("*.unanswered-*.json"))) == 3
    written = json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert written["digest"] == canonical_digest(written, digest_field="digest") and written == result


def test_job3_replay_its_402s_kept_as_no_answer_take_the_room_package_and_buy_nothing(
        tmp_path, monkeypatch, ready_atlas):
    """Test job 3 as the host holds it: three dispatches Gemini answered 402 "prepayment credits are depleted",
    which the release then kept as no answer and counted as spent, the ledger keeping each run's failure text but
    naming no dispatch. Nothing reads a dispatch as refused from the ledger, so the next run ends on the
    unanswered bound and ships the room package, flagged; nothing is reserved or sent."""
    from blueprint_pipeline import website_gemini_receipts as receipts
    args, calls = _sent_three_times(tmp_path, monkeypatch, depleted, ledger=True)
    assert len(list((tmp_path / RECEIPTS).glob("*.unanswered-*.json"))) == 3  # as on the host
    kwargs = _drawer_job(tmp_path, monkeypatch)
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", lambda **_: pytest.fail("reserved again"))
    args["invoke"] = lambda: pytest.fail("sent again")
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: receipts.retained_gemini_call(**args))
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written" and packaged[-1]["flags"] == [receipts.EXHAUSTED]
    assert calls == []


def _granted(tmp_path, kwargs, *, ends_in, extended_by=None):
    """The grant the job's delivery retained (``website_preparation_terminal.GRANT_FILE``), its window ending
    ``ends_in`` seconds after the handoff's ``now``; ``extended_by``: an owner-approved extension on its scene
    intent (``effective_execution_expiry``), in seconds past that."""
    from blueprint_pipeline.website_preparation_terminal import GRANT_FILE
    descriptor = kwargs["descriptor"]
    grant = {"schema_version": "website_scene_sponsorship.v1", "scene_id": descriptor["scene_id"],
             "capture_id": descriptor["capture_id"], "owner": {"user_id": "u", "organization_id": "o"},
             "expires_at_epoch": kwargs["now"] + ends_in}
    write_json(tmp_path / GRANT_FILE, {**grant, "authority_digest": canonical_digest(grant, digest_field="authority_digest")})
    return grant


def test_job3_going_forward_credits_out_is_a_money_wait_and_a_top_up_buys_the_drawer_under_the_same_hold(
        tmp_path, monkeypatch, ready_atlas):
    """A 402 credits-depleted answer kept on its receipt: while the credits are out the job is a money wait on the
    same hold, nothing counted or shipped; once Gemini answers, the drawer's views are bought under that hold and
    the job goes on to compile, never paying for the refused dispatches."""
    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline.hard_stops import is_wall
    from blueprint_pipeline.website_preparation_retry import classify
    from tests.test_website_gemini_receipts import setup_call
    args, calls = setup_call(tmp_path / RECEIPTS, monkeypatch)
    kwargs = _drawer_job(tmp_path, monkeypatch)
    _granted(tmp_path, kwargs, ends_in=5 * 86400)
    raised = []

    def coverage(**kw):
        try:
            result = receipts.retained_gemini_call(**args)
        except Exception as exc:
            raised.append(exc)
            raise
        return {**kw["task_masks"], "targets": [{"target_id": "drawer_cabinet_assembly",
                                                 "authoring_coverage": {"classified": result}}]}
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", coverage)
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs", lambda **kw: kw["task_masks"])
    compiled = []
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **kw: compiled.append(kw) or (_ for _ in ()).throw(_Compiled()))
    packaged = room_recorder(monkeypatch)
    args["invoke"] = lambda: calls.append(("invoke", None)) or (_ for _ in ()).throw(depleted())
    for _ in range(3):
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "awaiting_inputs" and result["blockers"] == [receipts.CREDIT_DEPLETED]
        assert is_wall(raised[-1]) and classify(f"website_scene_preparation: {result['blockers'][0]}") == "money"
    assert packaged == [] and compiled == [] and not list((tmp_path / RECEIPTS).glob("*.unanswered-*.json"))
    args["invoke"] = lambda: calls.append(("invoke", None)) or {"classification": "drawer"}  # the owner topped up
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert compiled[0]["task_masks"]["targets"][0]["authoring_coverage"] == {"classified": {"classification": "drawer"}}
    assert [row[0] for row in calls] == ["reserve", "invoke"] + ["source_read", "invoke"] * 3  # one hold
    with pytest.raises(_Compiled):  # and never bought again
        handoff.prepare_website_scene_handoff(**kwargs)
    assert len(calls) == 8 and packaged == []


def test_credits_out_ships_the_room_package_only_near_the_window_end_read_when_the_answer_comes(
        tmp_path, monkeypatch, ready_atlas):
    """The window end is read when the answer comes: the retained grant, moved later by an extension on its scene
    intent. Within two hours of it the room package ships, flagged; an extension keeps the job waiting; a grant
    that cannot be read keeps it waiting."""
    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline import website_preparation_terminal as terminal
    kwargs = _drawer_job(tmp_path, monkeypatch)
    packaged = room_recorder(monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(receipts.CreditDepleted()))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(tmp_path / "intents"))

    def handed():
        return handoff.prepare_website_scene_handoff(**kwargs)["status"]
    assert handed() == "awaiting_inputs"  # no grant retained: no window to read
    for ends_in, status in ((3 * 3600, "awaiting_inputs"), (2 * 3600, "room_package_written"),
                            (60, "room_package_written")):
        _granted(tmp_path, kwargs, ends_in=ends_in)
        assert handed() == status
    assert [call["flags"] for call in packaged] == [[receipts.CREDIT_DEPLETED]] * 2
    grant = _granted(tmp_path, kwargs, ends_in=60)
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_intent_contracts._read",
                        lambda path, field: {"request": {"submission_id": kwargs["descriptor"]["capture_id"]}})
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_execution_window_evidence.effective_execution_expiry",
                        lambda directory, intent: grant["expires_at_epoch"] + 2 * 86400)
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import canonical_digest as intent_digest
    identity = intent_digest({"owner": grant["owner"], "submission_id": grant["capture_id"]})
    intent = tmp_path / "intents" / ("scene-" + identity.removeprefix("sha256:")) / "intent.json"
    intent.parent.mkdir(parents=True)
    intent.write_text("{}")
    assert terminal.retained_window_end(tmp_path, scene_id=grant["scene_id"], capture_id=grant["capture_id"]) == (
        grant["expires_at_epoch"] + 2 * 86400)
    assert handed() == "awaiting_inputs" and len(packaged) == 2  # the owner extended it: still a money wait
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_intent_contracts._read",
                        lambda path, field: (_ for _ in ()).throw(ValueError("unreadable")))
    assert handed() == "awaiting_inputs" and len(packaged) == 2  # an intent that cannot be read: waits
    (tmp_path / terminal.GRANT_FILE).write_text("{not json")
    assert handed() == "awaiting_inputs" and len(packaged) == 2


def test_a_wall_or_another_code_at_the_same_step_is_never_the_room_package(tmp_path, monkeypatch, ready_atlas):
    """Only a provider's bound, credits out at the window's end, or the job's spending cap sends the job on: another
    money wall raised there, or any other code, keeps ``awaiting_inputs`` as before, and a room that cannot be
    packaged keeps the code. A code that is no wall ships the room floor meanwhile (owner, 2026-11-23); a money
    wall ships nothing."""
    from blueprint_pipeline.hard_stops import MoneyStop
    from blueprint_pipeline.website_gemini_receipts import EXHAUSTED
    kwargs = inputs(tmp_path)
    kwargs["descriptor"]["metadata"]["site_task_context"].update(context_digest="sha256:" + "c" * 64)
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    packaged = room_recorder(monkeypatch)
    for error in (MoneyStop(EXHAUSTED), MoneyStop("account_monthly_spend_limit_reached"),
                  ValueError("website_gemini_request_in_progress")):
        monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                            lambda error=error, **_: (_ for _ in ()).throw(error))
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "awaiting_inputs" and result["blockers"] == [str(error)]
        assert (handoff.ROOM_FLOOR in result) is (not isinstance(error, MoneyStop))
    assert [row["output_root"].name for row in packaged] == [handoff.ROOM_FLOOR]  # never the room package
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(ValueError(EXHAUSTED)))
    monkeypatch.setattr(handoff, "write_room_package",
                        lambda **_: (_ for _ in ()).throw(ValueError("website_base_scene_changed")))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "awaiting_inputs" and result["blockers"] == [EXHAUSTED]


def _sunk_into_its_table(monkeypatch, *, sink_m=0.23, residual_m=0.075):
    """Test job 4's shape on compile's own test geometry: the object's estimated box reaches ``sink_m`` below the
    table top (the lamp: bottom 0.54 m, table top about 0.77 m), past the search window the registration's
    residual near the task sets (7.5 cm)."""
    from blueprint_pipeline import website_task_preparation as preparation
    real_bounds, real_registration = preparation.estimate_target_bounds, preparation.register_source_to_runtime

    def sunk(*args, **kwargs):
        value = real_bounds(*args, **kwargs)
        shift = (1.75 - sink_m / 2.0) - value["minimum"][1]  # the table top is runtime y = 1.75; 2 m per unit
        value["minimum"][1] += shift
        value["maximum"][1] += shift
        return value

    def measured(**kwargs):
        value = real_registration(**kwargs)
        return {**value, "task_region": {**(value.get("task_region") or {}), "trimmed_rmse_m": residual_m}}
    monkeypatch.setattr(preparation, "estimate_target_bounds", sunk)
    monkeypatch.setattr(preparation, "register_source_to_runtime", measured)


def test_job4_an_object_with_no_surface_to_start_on_ships_the_room_package(tmp_path, monkeypatch, ready_atlas):
    """Test job 4 ("Move the lamp ... to the glass shelf"): compile found no surface under the lamp and the
    modelled job rested at ``needs_input`` on ``support_surface_not_found_under_subject``. Now the real compile's
    blockers go to the room package as flags; the preparation is not bound, registered or enqueued. Compile itself
    is main's (its retained, unbilled provider choice included). The next pass ends the same."""
    _sunk_into_its_table(monkeypatch)
    kwargs, asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", stop=False)
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation",
                        lambda *_, **__: pytest.fail("bound"))
    for run in (1, 2):
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert compiled[-1]["status"] == "needs_input"
        assert compiled[-1]["blockers"] == ["support_surface_not_found_under_subject"]
        assert result["status"] == "room_package_written" and result["blockers"] == []
        assert [call["flags"] for call in packaged] == [["support_surface_not_found_under_subject"]] * run
    assert len(asked) == 2 and "website_intake_outbox" not in result and "runtime_inputs" not in result


def test_job4_s_object_resting_within_the_window_still_builds_as_before(tmp_path, monkeypatch, ready_atlas):
    """The same job with its estimate 4 cm into the table: the surface is found and the object's route runs."""
    _sunk_into_its_table(monkeypatch, sink_m=0.04)
    kwargs, asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", stop="ready")
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert "support_surface_not_found_under_subject" not in compiled[0]["blockers"]
    assert not [call for call in packaged if call["output_root"].name != handoff.ROOM_FLOOR]


CONTEXT = {"context_digest": "sha256:" + "a" * 64}


@pytest.mark.parametrize("blockers, development, expected", [
    # Facts about the job's own scene or wording, as compile finds them, ship at once: one per hold it had.
    *[([code], False, True) for code in sorted(__import__(
        "blueprint_pipeline.website_task_preparation", fromlist=["SCENE_FACT_BLOCKERS"]).SCENE_FACT_BLOCKERS)],
    (["support_surface_not_found_under_subject", "website_assembly_whole_object_coverage_required"], False, True),
    (["website_assembly_reference_parts_uncovered:drawer-1"], False, True),  # a coverage blocker names its target
    # Any code that reads as privacy, rights or money keeps the preparation: the authoring admissions.
    *[(["support_surface_not_found_under_subject", code], False, False) for code in ADMISSIONS],
    # A development test places the object on its own authored surface.
    (["support_surface_not_found_under_subject"], True, False),
    # A bad request or record waits for its first retry cycle (none spent here); a pending privacy check is kept.
    (["scene_intake_owner_invalid"], False, False), (["website_scene_preparation_pending"], False, False),
    (["website_subject_origin_conflict"], False, False),
    # A stop only a new video clears keeps its own hold; a fit stop found once the room was built ships at once.
    (["website_capture_frames_unusable"], False, False), (["website_registration_poor_fit"], False, True),
    # A dollar budget reads as money and holds.
    (["website_native_construction_budget_exceeds_authority"], False, False),
    # The job's spending cap ships at once (owner, 2026-11-19), except a development test, which holds anything.
    (["website_scene_preparation_budget_exhausted"], False, True),
    (["website_scene_preparation_budget_exhausted"], True, False),
])
def test_one_rule_ships_every_blocker_that_is_no_wall(monkeypatch, blockers, development, expected):
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: development)
    assert handoff.goes_on(blockers, task_context=CONTEXT, now=0.0) is expected


@pytest.mark.parametrize("code", [
    # Anything that is no fact about the job's scene waits for the window's end, an unknown code included.
    "website_some_code_nobody_classified", "authoring_frames_coverage_missing:front_view",
    "website_assembly_builder_refused:the hinge axis is degenerate", "authoring_frames_token_budget_exceeded",
    "authoring_frame_bytes_exceed_budget:decoded-000000003.png", "website_assembly_front_plane_degenerate",
    "website_registration_scale_conflicts_declared", "website_assembly_coverage_key_missing",
    "website_task_grounding_key_missing", "website_sam31_profile_missing",
    # A published size that refutes our own size estimate is our estimate's fault, so it waits too.
    "website_object_spec_dimension_conflict"])
def test_every_other_blocker_waits_for_the_window_end_then_ships(monkeypatch, code):
    """A request-size "budget" is not money, so it waits rather than holds; a configuration fault waits too."""
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: False)
    for window, ships in ((None, False), (3 * 86400.0, False), (2 * 3600.0, True), (60.0, True)):
        assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, window_end=lambda window=window: window) is ships


@pytest.mark.parametrize("code", [
    "website_control_preparation-spend_http_409:website_scene_sponsorship_binding_invalid",
    "website_control_task-context_http_409:website_task_context_changed",
    "website_control_preparation-spend_http_409:website_capture_frames_unusable"])
def test_a_holding_code_after_the_website_s_operation_still_holds(monkeypatch, code):
    """A site that edits its job mid-run: the website's refusal carries the holding code after its operation."""
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: False)
    assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, error=ValueError(code),
                           window_end=lambda: 0.0) is False


def test_at_the_jobs_spending_cap_the_job_ships_what_it_has_flagged_and_waits_for_no_one(monkeypatch):
    """One cap at $40 per job (owner, 2026-11-19): the refusal stopped the purchase; nothing waits for a grant."""
    from blueprint_pipeline.hard_stops import MoneyStop, PrivacyRightsStop
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: False)
    code = "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"
    for window in (None, 3 * 86400.0):
        assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, error=MoneyStop(code),
                               window_end=lambda window=window: window) is True
    assert handoff.shipped_code(code) == "spending_cap_reached"
    assert handoff.shipped_code("website_scene_preparation_budget_exhausted") == "spending_cap_reached"
    assert handoff.shipped_code("support_surface_not_found_under_subject") == "support_surface_not_found_under_subject"
    # The account's monthly limit is still a wait, and a privacy wall beside the cap still holds.
    account = "account_monthly_spend_limit_reached"
    assert handoff.goes_on([account], task_context=CONTEXT, now=0.0, error=MoneyStop(account),
                           window_end=lambda: None) is False
    revoked = "website_control_preparation-spend_http_409:source_revoked"
    assert handoff.goes_on([code, revoked], task_context=CONTEXT, now=0.0, error=PrivacyRightsStop(revoked),
                           window_end=lambda: None) is False
    # Another operation's refusal at the cap (the scene intake) ships the same way.
    assert handoff.spending_cap_code("website_control_prepared-scene_http_409:website_scene_preparation_budget_exhausted")


@pytest.mark.parametrize("code", ["website_assembly_coverage_key_missing", "website_task_grounding_key_missing",
                                  "website_sam31_profile_missing"])
def test_a_missing_key_or_profile_pages_the_owner(code):
    from blueprint_pipeline import preparation_fault_alert as fault
    assert fault._fault({"status": "failed_retryable", "last_error": f"website_scene_preparation: {code}"}) == code


def test_an_unreadable_development_allowlist_keeps_the_preparation(monkeypatch):
    monkeypatch.setenv("BLUEPRINT_WEBSITE_DEVELOPMENT_TEST_TASK_DIGESTS", "not json")
    assert handoff.goes_on(["support_surface_not_found_under_subject"], task_context=CONTEXT, now=0.0) is False


@pytest.mark.parametrize("error, transient", [
    (TimeoutError("timed out"), True), (ValueError("website_control_preparation-spend_http_503:unavailable"), True),
    (ValueError("website_gemini_request_in_progress"), True), (OSError(28, "No space left on device"), True),
    # Blueprint's defects wait like an outage: a release may fix them, and the room package is final.
    (KeyError("frames"), True), (TypeError("'NoneType' object is not subscriptable"), True),
    (ValueError("operands could not be broadcast together"), True), (FileNotFoundError(2, "No such file"), True),
    # So do a step's typed refusal and a website refusal; only the end of a provider's bound ships at once.
    (ValueError("website_assembly_front_plane_degenerate"), True),
    (ValueError("website_control_task-context_http_422:website_task_context_unreadable"), True),
    (ValueError("website_gemini_unanswered_dispatches_exhausted"), False)])
def test_a_raised_failure_retried_until_the_deadline_waits_for_the_window_end_and_a_refusal_ships_at_once(
        monkeypatch, error, transient):
    from blueprint_pipeline.hard_stops import MoneyStop, PrivacyRightsStop
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: False)
    code = str(error)
    for window, ships in ((None, not transient), (3 * 3600.0, not transient), (2 * 3600.0, True), (60.0, True)):
        assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, error=error,
                               window_end=lambda window=window: window) is ships
    for wall in (MoneyStop(code), PrivacyRightsStop(code)):  # a wall waits whatever the window
        assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, error=wall, window_end=lambda: 0.0) is False


@pytest.mark.parametrize("code", ["website_gemini_unanswered_dispatches_exhausted",
                                  "website_gemini_provider_credit_depleted"])
def test_the_room_package_of_a_drawer_that_could_not_be_prepared_says_so(tmp_path, monkeypatch, code):
    """The real room package and its README, after the drawer's views could not be had (never answered, or credits
    out at the window's end): the room as reconstructed, no rule, the drawer taken out, and that it could not be
    prepared."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline import website_gemini_receipts as receipts
    from tests.test_website_scene_handoff_navigation import _published, _record, _said, room_job
    assert sim.OBJECT_NOT_PREPARED_FLAGS == {receipts.EXHAUSTED, receipts.CREDIT_DEPLETED}
    error = ValueError(code) if code == receipts.EXHAUSTED else receipts.CreditDepleted()
    monkeypatch.setattr("blueprint_pipeline.website_preparation_terminal.retained_window_end",
                        lambda *_a, **_k: 0.0)  # the window ends before the money wait can ask again
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(error))
    drawer = {"target_id": "cup-1", "task_effect": "manipulated", "articulation_kind": "prismatic",
              "articulated_part": "middle drawer", "semantic_label": "three-drawer cabinet"}
    job, _context, result = room_job(tmp_path, monkeypatch, family="articulated_open_close", removal=[drawer],
                                     all_targets=True, articulated=True, description="Open the middle drawer")
    assert (result["status"], result["blockers"]) == ("room_package_written", [])
    assert _record(result)["flags"][0] == code and _record(result)["success_rule_defined"] is False
    _status, members = _published(job)
    said = _said(members["README.txt"])
    assert said.endswith(f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} {sim.OBJECT_NOT_PREPARED} "
                         f"{sim.OBJECTS_TAKEN_OUT_ONE}")


def test_the_room_package_of_an_object_with_no_surface_to_start_on_says_so(tmp_path, monkeypatch):
    """Test job 4's package: the real room package and its README after compile found no surface under the object:
    the room as reconstructed, no rule, that the object was not placed and why, and that it was taken out."""
    from blueprint_pipeline import website_company_simulation as sim
    from tests.test_website_scene_handoff_navigation import _published, _record, _said, room_job
    assert sim.OBJECT_NOT_PLACED_FLAGS == {"support_surface_not_found_under_subject",
                                           "website_synthetic_subject_start_unplaceable"}
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    for name in ("website_assembly_coverage.attach_assembly_coverage",
                 "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: kw["task_masks"])
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: {
        "status": "needs_input", "blockers": ["support_surface_not_found_under_subject"]})
    lamp = {"target_id": "cup-1", "task_effect": "manipulated", "semantic_label": "table lamp"}
    job, _context, result = room_job(tmp_path, monkeypatch, family="rigid_pick_place", removal=[lamp],
                                     all_targets=True, description="Move the lamp to the shelf")
    assert (result["status"], result["blockers"]) == ("room_package_written", [])
    assert _record(result)["flags"][0] == "support_surface_not_found_under_subject"
    _status, members = _published(job)
    assert _said(members["README.txt"]).endswith(f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} {sim.OBJECT_NOT_PLACED} "
                                                 f"{sim.OBJECTS_TAKEN_OUT_ONE}")


# --- Owner, 2026-11-17: accept every job; one rule at the handoff (``goes_on``) ----------------------------------

def test_a_compile_blocker_ships_the_room_package_at_once_and_names_the_piece(tmp_path, monkeypatch, ready_atlas):
    """``website_task_preparation``: a destination compile cannot find ships the room package now, flagged, with
    the piece it leaves out recorded; nothing is bound, registered or chosen."""
    kwargs, asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", stop=False)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **kw: compiled.append(kw) or {
        "status": "needs_input", "blockers": ["task_destination_surface_contact_required"]})
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation",
                        lambda *_, **__: pytest.fail("bound"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written" and result["blockers"] == []
    assert packaged[-1]["flags"] == ["task_destination_surface_contact_required"]
    assert packaged[-1]["not_prepared"] == [{"piece": "destination", "code": "task_destination_surface_contact_required"}]


def test_a_raised_refusal_ships_at_once_and_a_transient_failure_waits_for_the_window_end(
        tmp_path, monkeypatch, ready_atlas):
    """``website_scene_handoff``: a step's refusal (no wall) ships the room package now; a timeout waits, retried,
    until the job's window, read when it fails, ends within two hours, then ships the same, its room floor shipping
    meanwhile (owner, 2026-11-23); a wall waits, with no floor."""
    from blueprint_pipeline.hard_stops import MoneyStop
    kwargs = _drawer_job(tmp_path, monkeypatch)
    recorded = room_recorder(monkeypatch)

    def rooms():  # the room packages written, never the room floors
        return [row for row in recorded if row["output_root"].name != handoff.ROOM_FLOOR]

    def failing(error):
        monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                            lambda **_: (_ for _ in ()).throw(error))
        return handoff.prepare_website_scene_handoff(**kwargs)
    result = failing(ValueError("website_gemini_unanswered_dispatches_exhausted"))
    assert result["status"] == "room_package_written"
    assert rooms()[-1]["not_prepared"] == [{"piece": "object", "code": "website_gemini_unanswered_dispatches_exhausted"}]
    shipped = len(rooms())
    waiting = failing(TimeoutError("timed out"))
    assert waiting["status"] == "awaiting_inputs"  # no grant: no window to read
    assert waiting[handoff.ROOM_FLOOR][handoff.ROOM_FLOOR_BLOCKERS] == ["timed out"]  # the room ships meanwhile
    _granted(tmp_path, kwargs, ends_in=3 * 86400)
    assert failing(TimeoutError("timed out"))["status"] == "awaiting_inputs"
    _granted(tmp_path, kwargs, ends_in=3600)
    assert failing(TimeoutError("timed out"))["status"] == "room_package_written"
    assert len(rooms()) == shipped + 1 and rooms()[-1]["flags"] == ["timed out"]
    # A wall waits: the account's monthly limit.
    floors = len(recorded)
    result = failing(MoneyStop("account_monthly_spend_limit_reached"))
    assert result["status"] == "awaiting_inputs" and len(rooms()) == shipped + 1
    assert handoff.ROOM_FLOOR not in result and len(recorded) == floors  # and no floor
    # The job's spending cap waits for no one (owner, 2026-11-19): the room package ships now, plainly flagged.
    result = failing(MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"))
    assert result["status"] == "room_package_written" and len(rooms()) == shipped + 2
    assert rooms()[-1]["flags"] == ["spending_cap_reached"]
    assert rooms()[-1]["not_prepared"] == [{"piece": "object", "code": "spending_cap_reached"}]


def test_a_development_test_still_takes_its_own_path(tmp_path, monkeypatch, ready_atlas):
    _sunk_into_its_table(monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: True)
    kwargs, _asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", stop=False)
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation",
                        lambda *_, **__: (_ for _ in ()).throw(_Compiled()))  # past the room package: its own path
    with pytest.raises(_Compiled):
        handoff.prepare_website_scene_handoff(**kwargs)
    assert compiled[0]["blockers"] == ["support_surface_not_found_under_subject"] and packaged == []


def test_a_room_record_of_a_job_with_nothing_left_out_is_unchanged(tmp_path, monkeypatch):
    """Jobs that prepare: a room record with nothing left out carries no new field, so its bytes and digest are
    main's (checked against main on 2026-11-18: this navigation job's handoff and record, and an intake-ready
    job's floor, byte for byte)."""
    from tests.test_website_scene_handoff_navigation import _record, room_job
    _job, _context, result = room_job(tmp_path, monkeypatch)
    assert "not_prepared" not in _record(result)


def test_the_room_package_names_each_piece_it_could_not_prepare(tmp_path, monkeypatch):
    """The README's one sentence per piece left out, with its codes and no reason of its own."""
    from blueprint_pipeline import website_company_simulation as sim
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    for name in ("website_assembly_coverage.attach_assembly_coverage",
                 "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: kw["task_masks"])
    codes = ["task_destination_surface_contact_required", "website_owner_success_criteria_translation_required"]
    monkeypatch.setattr(handoff, "compile_website_scene_preparation",
                        lambda **_: {"status": "needs_input", "blockers": codes})
    lamp = {"target_id": "cup-1", "task_effect": "manipulated", "semantic_label": "table lamp"}
    job, _context, result = room_job(tmp_path, monkeypatch, family="rigid_pick_place", removal=[lamp],
                                     all_targets=True, description="Move the lamp to the shelf")
    assert result["status"] == "room_package_written"
    _status, members = _published(job)
    assert _said(members["README.txt"]).endswith(
        f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} "
        "Blueprint could not prepare the job's destination. "
        "Blueprint could not prepare the job's success rule. "
        f"{sim.OBJECTS_TAKEN_OUT_ONE}")
    # The blocker codes stay in the record, never in the file a company reads.
    assert not any(code in members["README.txt"] for code in codes)


@pytest.mark.parametrize("cause", ["other_geometry_in_its_box_wherever_it_would_rest",
                                   "no_spot_with_the_disc_and_half_of_it_on_the_surface", None])
def test_a_room_package_says_why_no_spot_on_the_estimated_destination_held_the_object(tmp_path, monkeypatch, cause):
    """Job move-the-lamp-2 (24 Nov): the README said "no spot on it held the object's estimated footprint", but the
    lamp's 6.5 cm foot would have fitted on the 16 cm glass shelf: what no spot held was its whole box, shade and all,
    clear of the wall behind the shelf with its centre 5 cm inside the shelf's edges. Compile's search records why
    (``destination_unheld``), the room package's row carries the cause, and the README says it; without one, only
    that the object could not be placed there."""
    from pathlib import Path

    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline.atlas_scene_estimates import DESTINATION_SURFACE_ESTIMATED
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    for name in ("website_assembly_coverage.attach_assembly_coverage",
                 "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: kw["task_masks"])
    code = "task_destination_surface_contact_required"

    def compiled(**kw):
        row = {"code": DESTINATION_SURFACE_ESTIMATED, "source": "website_unscanned_surface", "evidence": {}}
        write_json(kw["output_root"] / "base_scene.json",
                   {**kw["base_scene"], "scene_estimates": [*kw["base_scene"].get("scene_estimates", []), row]})
        return {"status": "needs_input", "blockers": [code], "destination_surface_estimate": {"status": "estimated"},
                "destination_unheld": {"status": "no_spot_holds_it", **({"cause": cause} if cause else {})},
                "digest": "sha256:" + "e" * 64}

    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compiled)
    lamp = {"target_id": "cup-1", "task_effect": "manipulated", "semantic_label": "table lamp"}
    job, _context, result = room_job(tmp_path, monkeypatch, family="rigid_pick_place", removal=[lamp],
                                     all_targets=True, description="Move the lamp to the glass shelf")
    assert result["status"] == "room_package_written"
    record = json.loads(Path(result["room_package"]["path"]).read_text())
    assert record["not_prepared"] == [{"piece": "destination", "code": code, **({"cause": cause} if cause else {})}]
    assert record["preparation_digest"] == "sha256:" + "e" * 64  # the company's page reads the object's estimates there
    _status, members = _published(job)
    said = _said(members["README.txt"])
    sentence = {
        "other_geometry_in_its_box_wherever_it_would_rest": (
            "Blueprint could not place the job's object on that estimated surface: at every spot it tried with the "
            "object's centre at least 5 cm inside the surface's edges and at least half of the object over the "
            "surface, the object's estimated box, which takes in the whole object and not only its base, reached "
            "into another surface of the room."),
        "no_spot_with_the_disc_and_half_of_it_on_the_surface": (
            "Blueprint could not place the job's object on that estimated surface: no spot it tried had room for the "
            "object's centre at least 5 cm inside the surface's edges with at least half of the object's estimated "
            "box over the surface."),
        None: "Blueprint could not place the job's object on that estimated surface."}[cause]
    assert said.endswith(f"{sim.SURFACE_ESTIMATED_ROOM_NOTE} {sim.ROOM_NOT_RUN} {sim.NO_RULE} {sentence} "
                         f"{sim.OBJECTS_TAKEN_OUT_ONE}")
    assert "footprint" not in said and code not in members["README.txt"] and (cause or code) not in said


@pytest.mark.parametrize("estimated", [True, False])
def test_a_room_package_holding_an_estimated_destination_says_the_object_was_not_placed_on_it(tmp_path, monkeypatch,
                                                                                             estimated):
    """Job move-the-lamp-2 (24 Nov): the glass shelf's estimate holds, and is sealed in the base scene every package
    reads, but no spot on it holds the lamp, so compile stops on the destination and the room package ships. Its
    README says the shelf was estimated without saying an object is placed on it, and says the object could not be
    placed on that surface rather than that the destination could not be prepared."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline.atlas_scene_estimates import DESTINATION_SURFACE_ESTIMATED
    from tests.test_website_scene_handoff_navigation import _published, _said, room_job
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    for name in ("website_assembly_coverage.attach_assembly_coverage",
                 "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: kw["task_masks"])
    code = "task_destination_surface_contact_required"

    def compiled(**kw):
        value = {"status": "needs_input", "blockers": [code]}
        if estimated:  # compile seals the estimate in the base scene it writes (``website_unscanned_surface._seal``)
            row = {"code": DESTINATION_SURFACE_ESTIMATED, "source": "website_unscanned_surface", "evidence": {}}
            write_json(kw["output_root"] / "base_scene.json",
                       {**kw["base_scene"], "scene_estimates": [*kw["base_scene"].get("scene_estimates", []), row]})
            value["destination_surface_estimate"] = {"status": "estimated"}
        return value

    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compiled)
    lamp = {"target_id": "cup-1", "task_effect": "manipulated", "semantic_label": "table lamp"}
    job, _context, result = room_job(tmp_path, monkeypatch, family="rigid_pick_place", removal=[lamp],
                                     all_targets=True, description="Move the lamp to the glass shelf")
    assert result["status"] == "room_package_written"
    _status, members = _published(job)
    said = _said(members["README.txt"])
    if estimated:
        assert said.endswith(f"{sim.SURFACE_ESTIMATED_ROOM_NOTE} {sim.ROOM_NOT_RUN} {sim.NO_RULE} "
                             f"{sim.NOT_PLACED_ON_ESTIMATED_SURFACE} {sim.OBJECTS_TAKEN_OUT_ONE}")
        assert "is placed on" not in said and "could not prepare the job's destination" not in said
    else:
        assert said.endswith(f"{sim.ROOM_NOT_RUN} {sim.NO_RULE} Blueprint could not prepare the job's destination. "
                             f"{sim.OBJECTS_TAKEN_OUT_ONE}")
        assert "estimated from its visible edges" not in said
    assert code not in members["README.txt"]


def test_a_defect_waits_named_and_paged_and_ships_only_at_the_window_end(tmp_path, monkeypatch, ready_atlas):
    """A KeyError in a preparation step is Blueprint's bug: the job waits under ``website_preparation_defect:KeyError``
    (which pages the owner) with the message kept, so a fix release can still find it. Its room floor ships at once,
    flagged with the code (owner, 2026-11-23); only when the job's window ends within two hours does the room package
    end the job, the code named on its record."""
    from blueprint_pipeline import preparation_fault_alert as fault
    kwargs = _drawer_job(tmp_path, monkeypatch)
    packaged = room_recorder(monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: {}["frames"])
    code = "website_preparation_defect:KeyError"
    stopped = {"status": "failed_retryable", "last_error": f"website_scene_preparation: {code}"}
    assert fault._fault(stopped) == handoff.DEFECT  # the owner is paged, as for our own records
    for ends_in in (None, 3 * 86400):
        if ends_in is not None:
            _granted(tmp_path, kwargs, ends_in=ends_in)
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "awaiting_inputs" and result["blockers"] == [code]
        assert result["defect"] == "KeyError: 'frames'" and "room_package" not in result
        assert result[handoff.ROOM_FLOOR][handoff.ROOM_FLOOR_BLOCKERS] == [code]
        assert packaged[-1]["output_root"].name == handoff.ROOM_FLOOR
        assert packaged[-1]["flags"] == [handoff.ROOM_FLOOR_FLAG, code]
    _granted(tmp_path, kwargs, ends_in=3600)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "room_package_written"
    assert packaged[-1]["flags"] == [code]  # the type stays on the record's flags, for whoever fixes it
    assert packaged[-1]["not_prepared"] == [{"piece": "object", "code": handoff.DEFECT}]  # the README: code only


def test_a_waiting_preparation_runs_once_more_in_the_last_two_hours_of_its_window(tmp_path):
    """Retry lane: whatever its schedule (spent timed retries included), a waiting preparation runs once when its
    window, read live, ends within two hours, so the handoff can ship the room package before the window ends."""
    from datetime import datetime, timedelta, timezone
    from blueprint_pipeline.website_preparation_retry import LAST_CALL_SECONDS, last_call_due
    kwargs = inputs(tmp_path)
    now = datetime.fromtimestamp(kwargs["now"], timezone.utc)
    ledger = {"scene_id": kwargs["descriptor"]["scene_id"], "capture_id": kwargs["descriptor"]["capture_id"],
              "last_attempt_started_at": (now - timedelta(hours=5)).isoformat()}
    assert LAST_CALL_SECONDS == handoff.LAST_CALL_SECONDS
    assert last_call_due(tmp_path, ledger, now) is False  # no grant: no window to read
    _granted(tmp_path, kwargs, ends_in=3 * 3600)
    assert last_call_due(tmp_path, ledger, now) is False  # not yet
    _granted(tmp_path, kwargs, ends_in=3600)
    assert last_call_due(tmp_path, ledger, now) is True
    ran = {**ledger, "last_attempt_started_at": (now - timedelta(minutes=10)).isoformat()}
    assert last_call_due(tmp_path, ran, now) is False  # once: it already ran in that stretch
    _granted(tmp_path, kwargs, ends_in=-60)
    assert last_call_due(tmp_path, ledger, now) is False  # the window has ended


# --- Review 2026-11-18: never after the prepared scene is sent; our own faults wait; held codes by definition ------

def _to_the_outbox(tmp_path, monkeypatch, enqueue):
    """An intake-ready rigid job through the real compile to the prepared-scene write, which ``enqueue`` answers.
    Returns the handoff's arguments and the room packages written."""
    kwargs, _asked, compiled, packaged = _through_compile(tmp_path, monkeypatch, family="rigid_pick_place", stop=False)
    kwargs["descriptor"]["metadata"]["website_scene_execution_authority"] = {
        "schema_version": "website_scene_sponsorship.v1", "max_total_spend_usd": 20, "max_paid_attempts": 2,
        "expires_at_epoch": kwargs["now"] + 3600}
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation", lambda value, **_: value)
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: False)
    monkeypatch.setattr(handoff, "prepare_rigid_native_inputs", lambda **_: True)
    monkeypatch.setattr("blueprint_pipeline.website_scene_dispatch.binding_root", lambda: tmp_path / "bindings")
    monkeypatch.setattr("blueprint_pipeline.website_scene_dispatch.register_website_preparation",
                        lambda **_: {"status": "registered"})
    monkeypatch.setattr("blueprint_pipeline.website_task_context.enqueue_website_prepared_scene", enqueue)
    return kwargs, compiled, packaged


@pytest.mark.parametrize("error", [ValueError("website_scene_outbox_receipt_invalid"),
                                   ValueError("website_control_prepared-scene_http_409:idempotency_conflict"),
                                   TimeoutError("timed out after the website committed")])
def test_a_failed_prepared_scene_write_never_ships_the_room_package(tmp_path, monkeypatch, ready_atlas, error):
    """The prepared-scene write creates an outbox entry forwarded into paid preparation: once it is sent (a refusal,
    a conflict on a rerun, or a timeout after the website committed), the job keeps waiting as before, even at the
    window's end; and a later run that fails earlier never ships the room package either."""
    kwargs, compiled, packaged = _to_the_outbox(tmp_path, monkeypatch, lambda **_: (_ for _ in ()).throw(error))
    _granted(tmp_path, kwargs, ends_in=60)  # the window's end: an outage would ship now
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert compiled[0]["status"] == "intake_ready" and result["status"] == "intake_ready"
    assert result["runtime_inputs"] == {"status": "awaiting_inputs", "blockers": [str(error)]}
    assert result["source_registration"] == {"status": "registered"}  # it got as far as the write
    assert [call for call in packaged if call["output_root"].name != handoff.ROOM_FLOOR] == []
    assert handoff.prepared_scene_may_exist(tmp_path)
    exhausted = ValueError("website_gemini_unanswered_dispatches_exhausted")  # would ship at once otherwise
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(exhausted))
    for _ in range(2):  # reruns failing before compile, the second after the first rewrote the handoff
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "awaiting_inputs"
        assert [call for call in packaged if call["output_root"].name != handoff.ROOM_FLOOR] == []


@pytest.mark.parametrize("prior_state", [None, "sending", "sent"])
def test_prepared_scene_cap_refusal_ships_only_without_an_earlier_send(tmp_path, monkeypatch, ready_atlas, prior_state):
    def refuse(**_):
        raise ValueError("website_control_prepared-scene_http_409:website_scene_preparation_budget_exhausted")

    kwargs, _compiled, packaged = _to_the_outbox(tmp_path, monkeypatch, refuse)
    _granted(tmp_path, kwargs, ends_in=3 * 86400)
    path = tmp_path / "pipeline/website_scene_preparation" / handoff.OUTBOX_MARKER
    if prior_state is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"state": prior_state})
    result = handoff.prepare_website_scene_handoff(**kwargs)
    packages = [row for row in packaged if row["output_root"].name != handoff.ROOM_FLOOR]
    assert json.loads(path.read_text())["state"] == (prior_state or "refused")
    if prior_state is None:
        assert result["status"] == handoff.ROOM_PACKAGE
        assert "spending_cap_reached" in packages[0]["flags"]
    else:
        assert result["runtime_inputs"]["status"] == "awaiting_inputs"
        assert not packages


def test_capped_atlas_result_ships_the_retained_room_before_native_preparation(tmp_path, monkeypatch, ready_atlas):
    ready_atlas["flags"] = ["spending_cap_reached"]
    kwargs = _drawer_job(tmp_path, monkeypatch)
    _granted(tmp_path, kwargs, ends_in=3 * 86400)
    packaged = room_recorder(monkeypatch)

    def unexpected(**_):
        pytest.fail("The capped job tried to prepare a separate object")

    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage", unexpected)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == handoff.ROOM_PACKAGE
    [room] = packaged
    assert "spending_cap_reached" in room["flags"]
    assert "spending_cap_reached" in room["not_prepared"][0]["code"]
    assert "website_intake_outbox" not in result


def test_an_earlier_release_s_outbox_receipt_is_kept_across_held_runs(tmp_path, monkeypatch, ready_atlas):
    """A job that sent its prepared scene on an earlier release: its handoff's receipt is kept as the marker before
    a held run rewrites the handoff, so no later run takes the fallback either."""
    kwargs = _drawer_job(tmp_path, monkeypatch)
    packaged = room_recorder(monkeypatch)
    write_json(tmp_path / "pipeline/website_scene_preparation/handoff.json",
               {"status": "intake_ready", "website_intake_outbox": {"id": "scene-1", "state": "forward_pending",
                                                                   "request_digest": "sha256:" + "d" * 64}})
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(ValueError("website_gemini_unanswered_dispatches_exhausted")))
    for _ in range(3):
        result = handoff.prepare_website_scene_handoff(**kwargs)
        assert result["status"] == "awaiting_inputs" and packaged == []
        assert "website_intake_outbox" not in result  # the handoff no longer carries it; the marker does
    marker = json.loads((tmp_path / "pipeline/website_scene_preparation" / handoff.OUTBOX_MARKER).read_text())
    assert (marker["state"], marker["outbox_id"]) == ("sent", "scene-1")


def test_a_refused_retry_never_clears_an_earlier_send_that_may_have_committed(tmp_path, monkeypatch, ready_atlas):
    """Codex P1 on #3637: a send that timed out after the website may have committed leaves the marker ``sending``;
    a retry the website refuses with a typed answer proves only that the retry committed nothing, so the earlier
    evidence stands and no later run takes the room-package fallback."""
    answers = iter([ValueError("website_control_prepared-scene_timeout"),
                    ValueError("website_control_prepared-scene_http_401:website_control_unauthorized")])
    kwargs, _compiled, packaged = _to_the_outbox(tmp_path, monkeypatch, lambda **_: (_ for _ in ()).throw(next(answers)))
    path = tmp_path / "pipeline/website_scene_preparation" / handoff.OUTBOX_MARKER
    handoff.prepare_website_scene_handoff(**kwargs)
    assert json.loads(path.read_text())["state"] == "sending"
    handoff.prepare_website_scene_handoff(**kwargs)
    marker = json.loads(path.read_text())
    assert marker["state"] == "sending" and "http_401" in marker["later_refusal"]
    assert handoff.prepared_scene_may_exist(tmp_path, task_context_digest=marker["task_context_digest"])
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(ValueError("website_gemini_unanswered_dispatches_exhausted")))
    assert handoff.prepare_website_scene_handoff(**kwargs)["status"] == "awaiting_inputs"
    assert [call for call in packaged if call["output_root"].name != handoff.ROOM_FLOOR] == []


def test_a_marker_for_a_refused_send_or_another_task_never_holds_the_job(tmp_path, monkeypatch, ready_atlas):
    """A send the website refused with a typed answer that commits nothing settles the marker ``refused``; a marker
    written for another task context names another job: neither keeps this job from its room package."""
    error = ValueError("website_control_prepared-scene_http_422:website_scene_request_invalid")
    kwargs, _compiled, packaged = _to_the_outbox(tmp_path, monkeypatch, lambda **_: (_ for _ in ()).throw(error))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["runtime_inputs"]["blockers"] == [str(error)]  # the send's own failure holds, as before
    marker = json.loads((tmp_path / "pipeline/website_scene_preparation" / handoff.OUTBOX_MARKER).read_text())
    assert marker["state"] == "refused" and not handoff.prepared_scene_may_exist(tmp_path)
    write_json(tmp_path / "pipeline/website_scene_preparation" / handoff.OUTBOX_MARKER,
               {**marker, "state": "sending"})
    assert handoff.prepared_scene_may_exist(tmp_path, task_context_digest=marker["task_context_digest"])
    assert not handoff.prepared_scene_may_exist(tmp_path, task_context_digest="sha256:" + "e" * 64)
    assert packaged == [] or all(call["output_root"].name == handoff.ROOM_FLOOR for call in packaged)


@pytest.mark.parametrize("code", ["website_registration_scale_conflicts_declared",
                                  "website_registration_anchor_frame_invalid",
                                  "website_registration_anchor_frame_missing"])
def test_a_fault_of_our_own_waits_is_paged_and_ships_only_at_the_window_end(code):
    """``preparation_fault_alert`` pages these as ours to fix; each waits until the job's window, read live, ends
    within two hours, raised or a compile blocker alike, so a fix release can rescue it."""
    from blueprint_pipeline import preparation_fault_alert as fault
    assert code in fault.FAULT_CODES
    for error in (None, ValueError(code)):
        for window, ships in ((None, False), (3 * 86400.0, False), (3600.0, True)):
            assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, error=error,
                                   window_end=lambda window=window: window) is ships


def test_the_codes_the_room_package_would_be_wrong_for_are_read_from_their_definitions():
    """Every code raised or appended on the preparation path that protects privacy or money, found in the source by
    its own wording (a digest or binding that changed or does not match, a path outside the capture, a receipt
    invalid or missing or needing reconciliation, a step still pending), keeps the job waiting, even once its first
    retry cycle is spent. A bad record that protects neither ships then, flagged."""
    import re
    from pathlib import Path
    root = Path(handoff.__file__).parent
    codes = set()
    for name in ("website_scene_handoff", "website_task_preparation", "website_scene_runtime_inputs",
                 "website_assembly_coverage", "website_task_masks", "website_atlas_scene_source",
                 "website_preparation_revision", "task_family_preparation", "website_object_local_frame",
                 "website_gemini_receipts", "website_task_grounding", "website_object_spec_research"):
        source = (root / f"{name}.py").read_text()
        codes |= set(re.findall(r'(?:Error|Refusal|blockers\.append)\("([a-z][a-z0-9_]+)"', source))
    integrity = re.compile(r".*(_changed|_mismatch|_binding_invalid|_outside_capture|_receipt_invalid"
                           r"|_receipts_invalid|_receipt_missing|_result_missing|_requires_reconciliation|_pending)$")
    found = sorted(code for code in codes if integrity.match(code))
    assert {"website_assembly_coverage_binding_mismatch", "website_scene_preparation_pending",
            "website_scene_task_identity_mismatch", "website_scene_removal_manifest_outside_capture",
            "website_gemini_requires_reconciliation"} <= set(found)
    assert all(handoff.keeps_holding(code) for code in found)
    assert all(handoff.goes_on([code], task_context=CONTEXT, now=0.0, retries_spent=lambda: True) is False
               for code in found)
    for code in ("website_subject_origin_conflict", "website_retained_static_view_plan_missing",
                 "website_authoring_backend_invalid", "website_authoring_provider_invalid",
                 "website_task_context_family_invalid", "website_base_scene_scale_invalid",
                 "website_task_family_preparation_not_ready", "scene_intake_record_unreadable"):
        assert not handoff.keeps_holding(code)
    assert not handoff.keeps_holding("website_object_spec_dimension_conflict")  # a finding, not a bad record


def test_the_readme_names_the_right_piece_and_never_a_builder_s_or_an_exception_s_words():
    assert handoff.piece_of("website_articulated_task_has_no_destination") == "success rule"
    assert handoff.piece_of("task_destination_surface_contact_required") == "destination"
    assert handoff.piece_of("website_assembly_builder_refused:the hinge axis 'z' is degenerate") == "object"


@pytest.mark.parametrize("code", ["website_some_code_nobody_classified", "website_subject_origin_conflict",
                                  "website_preparation_defect:KeyError"])
def test_a_blocker_that_is_no_wall_ships_its_room_floor_at_once_and_the_room_package_ends_its_retry_cycle(
        monkeypatch, code):
    """Owner, 2026-11-23 (it replaces the 2026-11-22 timing): the room floor ships flagged at once, on the first
    run that stops on the blocker; the job keeps retrying, and only the last run of its first retry cycle ends it
    as the room package, flagged."""
    monkeypatch.setattr("blueprint_pipeline.website_development_test.enabled", lambda *_: False)
    assert handoff.floor_ships([code], task_context=CONTEXT)
    for spent, ships in ((False, False), (True, True)):
        assert handoff.goes_on([code], task_context=CONTEXT, now=0.0, window_end=lambda: 7 * 86400.0,
                               retries_spent=lambda spent=spent: spent) is ships
    # A wall ships neither, before or after the cycle.
    assert not handoff.floor_ships(["website_capture_withdrawn"], task_context=CONTEXT)
    assert handoff.goes_on(["website_capture_withdrawn"], task_context=CONTEXT, now=0.0,
                           retries_spent=lambda: True) is False


def test_the_first_retry_cycle_is_spent_after_six_failures_in_a_row_over_its_whole_schedule(tmp_path):
    from datetime import datetime, timedelta, timezone
    from blueprint_pipeline.handoff_job_state import JOB_LEDGER_FILENAME
    from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    schedule = sum(REFUSAL_RETRY_DELAYS_SECONDS)

    def ledger(failures):
        rows = [{"status": "completed"}] + [
            {"status": "failed_retryable", "failed_at": (start + timedelta(seconds=60 * n)).isoformat()}
            for n in range(failures)]
        write_json(tmp_path / JOB_LEDGER_FILENAME, {"status": "processing", "attempt_history": rows})

    assert handoff.first_retry_cycle_spent(tmp_path, start.timestamp() + 10 * schedule) is False  # no ledger
    ledger(len(REFUSAL_RETRY_DELAYS_SECONDS) - 1)
    assert handoff.first_retry_cycle_spent(tmp_path, start.timestamp() + 10 * schedule) is False
    ledger(len(REFUSAL_RETRY_DELAYS_SECONDS))
    assert handoff.first_retry_cycle_spent(tmp_path, start.timestamp() + schedule - 1) is False  # an outage, fast
    assert handoff.first_retry_cycle_spent(tmp_path, start.timestamp() + schedule) is True


# --- 2026-11-23: once the room is built, a blocker that is no wall ships the room floor at once ---------------------

NO_WALL = ["website_some_code_nobody_classified", "website_subject_origin_conflict",
           "website_retained_static_view_plan_missing"]


@pytest.mark.parametrize("code", NO_WALL)
def test_a_blocker_that_is_no_wall_writes_the_room_floor_on_its_first_run_and_the_job_keeps_retrying(
        tmp_path, monkeypatch, ready_atlas, code):
    """Case 1: the room is reconstructed and registered (``room_ready``), and a later step stops on a blocker that
    is no wall, far from the end of the first retry cycle. The handoff stays blocked (the job retries on its
    schedule) and carries the room floor, flagged with the blocker, for the company pass to ship now."""
    kwargs = _drawer_job(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(ValueError(code)))
    monkeypatch.setattr(handoff, "first_retry_cycle_spent", lambda *_a, **_k: False)
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "awaiting_inputs" and result["blockers"] == [code]
    assert "room_package" not in result
    [floor] = packaged
    assert floor["output_root"].name == handoff.ROOM_FLOOR and floor["task_profile"] is None
    assert floor["flags"] == [handoff.ROOM_FLOOR_FLAG, code]
    assert floor["not_prepared"] == [{"piece": "object", "code": code}]
    assert result[handoff.ROOM_FLOOR][handoff.ROOM_FLOOR_BLOCKERS] == [code]
    written = json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text())
    assert written == result and written["digest"] == canonical_digest(written, digest_field="digest")


def test_a_defect_writes_the_room_floor_under_its_code(tmp_path, monkeypatch, ready_atlas):
    kwargs = _drawer_job(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: {}["missing"])
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["website_preparation_defect:KeyError"] and "defect" in result
    assert [call["flags"] for call in packaged] == [[handoff.ROOM_FLOOR_FLAG, "website_preparation_defect:KeyError"]]


@pytest.mark.parametrize("code", ["website_assembly_coverage_binding_mismatch",  # privacy: another job's record
                                  "website_gemini_requires_reconciliation",  # money: spend not yet proven
                                  "website_capture_withdrawn",  # the site withdrew
                                  "authoring_admission_spend_limit_reached"])  # money: an admission refused
def test_a_privacy_rights_or_money_hold_writes_no_room_floor(tmp_path, monkeypatch, ready_atlas, code):
    """Case 3: nothing ships, as before."""
    kwargs = _drawer_job(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(ValueError(code)))
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == [code] and packaged == [] and handoff.ROOM_FLOOR not in result


def test_no_room_floor_before_the_room_is_built(tmp_path, monkeypatch):
    """Case 2 (the provider's room exists, but its source pose could not be established): the scene source is not
    ready, so no base scene with a scale and a levelled floor exists and no floor is written."""
    kwargs = inputs(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: {"status": "needs_input", "reason": "website_atlas_source_pose_unavailable",
                                            "job_reason": "website_atlas_pose_attempt_failed"})
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == ["website_atlas_source_pose_unavailable"]
    assert packaged == [] and handoff.ROOM_FLOOR not in result


def test_a_floor_that_cannot_be_written_stops_nothing_on_a_blocked_run(tmp_path, monkeypatch, ready_atlas):
    kwargs = _drawer_job(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: (_ for _ in ()).throw(ValueError(NO_WALL[0])))
    monkeypatch.setattr(handoff, "write_room_package",
                        lambda **_: (_ for _ in ()).throw(ValueError("website_base_scene_changed")))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["status"] == "awaiting_inputs" and result["blockers"] == [NO_WALL[0]]
    assert result[handoff.ROOM_FLOOR] == {"status": "not_written", "reason": "website_base_scene_changed"}


def _wiping_job(tmp_path, monkeypatch):
    """A wiping job through the family route (its runtime installed) and the real compile, its room built."""
    from tests.test_task_family_preparation import _arguments_for_family
    (tmp_path / "pipeline").mkdir(parents=True)
    install_family_runtimes(tmp_path, monkeypatch)
    args = _arguments_for_family(tmp_path, "wiping")
    args["source_geometry"]["geometry_backend"] = "fixture-native-depth"
    args["source_geometry"]["digest"] = canonical_digest(args["source_geometry"], digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = args["source_geometry"]["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    write_json(tmp_path / "pipeline/removal.json", args["removal_manifest"])
    monkeypatch.setattr(ATLAS, lambda **_: {"status": "ready", "base_scene": args["base_scene"],
        "source_geometry": args["source_geometry"], "source_pose_digest": "sha256:" + "b" * 64,
        "background_pose_digest": "sha256:" + "c" * 64, "alignment": "anchored"})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    return dict(descriptor={"scene_id": args["task_context"]["scene_id"], "capture_id": args["task_context"]["capture_id"],
                            "metadata": {"site_task_context": args["task_context"]}},
                clean_plate={"privacy_verified": True, "status": "objects_removed",
                             "removal_manifest_path": str(tmp_path / "pipeline/removal.json"),
                             "task_masks": args["task_masks"]}, capture_root=tmp_path, now=args["now"])


def test_a_family_preparation_blocked_on_no_wall_ships_its_room_floor_and_a_wall_ships_none(tmp_path, monkeypatch):
    """The family route returns on its own: its blocked preparation still carries the room floor (review of #3708),
    and a withdrawal among its blockers is a wall that writes none."""
    kwargs = _wiping_job(tmp_path / "open", monkeypatch)
    packaged = room_recorder(monkeypatch)
    result = handoff.prepare_website_scene_handoff(**kwargs)
    codes = ["website_task_family_success_contract_required", "website_task_family_native_runtime_inputs_required"]
    assert result["status"] == "needs_input" and result["blockers"] == codes
    assert "source_registration" not in result and "website_intake_outbox" not in result
    [floor] = packaged
    assert floor["output_root"].name == handoff.ROOM_FLOOR and floor["flags"] == [handoff.ROOM_FLOOR_FLAG, *codes]
    assert result[handoff.ROOM_FLOOR][handoff.ROOM_FLOOR_BLOCKERS] == codes
    written = json.loads((tmp_path / "open/pipeline/website_scene_preparation/handoff.json").read_text())
    assert written == result and written["digest"] == canonical_digest(written, digest_field="digest")
    kwargs, packaged[:] = _wiping_job(tmp_path / "wall", monkeypatch), []
    real = handoff.compile_website_scene_preparation
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **kw: (lambda prepared: {
        **prepared, "blockers": [*prepared["blockers"], "website_capture_withdrawn"]})(real(**kw)))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"][-1] == "website_capture_withdrawn"
    assert packaged == [] and handoff.ROOM_FLOOR not in result
