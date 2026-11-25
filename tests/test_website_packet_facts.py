"""What a deployment company reads about the job's object on its job page (preparation status v9).

The by-hand job packet export is retired (owner, 2026-11-19: one delivery path, the website). Its object, height
and uncertainty lines, and Blueprint's estimates of what the site did not say, reach the company's job page as one
small record the website renders; these are the export's checks, carried over.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_packet_facts.py
#   src/blueprint_pipeline/atlas_automation_plan.py
#   src/blueprint_pipeline/website_job_estimates.py
#   src/blueprint_pipeline/website_preparation_status.py

from __future__ import annotations

import json

from blueprint_pipeline import website_packet_facts as facts
from tests.test_website_robot_fit_projection import CONTEXT, capture_root, write_fit_table


def _texts(rows):
    return [row["text"] for row in rows]


def test_every_estimate_the_pipeline_records_has_a_plain_sentence() -> None:
    from blueprint_pipeline.atlas_scene_estimates import CODES, TASK_OBJECT_BOX_FIT_FLAGS
    # Or is one of Blueprint's own checks a company cannot act on, which its page leaves out.
    assert CODES | TASK_OBJECT_BOX_FIT_FLAGS <= set(facts.ESTIMATE_SENTENCES) | facts.UNSAID
    assert not facts.UNSAID & set(facts.ESTIMATE_SENTENCES)


def test_an_estimate_a_company_cannot_act_on_is_left_out_and_never_described_as_unknown() -> None:
    measured, estimated = facts.measured_and_estimated(
        LAMP, {"scene_estimates": [{"code": "scene_estimate:review_citations_dropped"}]}, None)
    texts = " ".join(_texts(estimated))
    assert "citation" not in texts and "not yet described" not in texts and "scene review" not in texts



def test_a_scene_whose_gap_views_were_not_admitted_says_they_were_not_filled_in_and_never_that_they_were_generated() -> None:
    """Live test job move-the-lamp-2 (2026-11-24): its generated gap views were refused, so nothing fills the room
    the video did not show. The company's page says what its package README says, and no line says generated."""
    from blueprint_pipeline.atlas_automation_plan import SCENE_COMPLETION_NOT_ADMITTED
    from blueprint_pipeline.website_company_simulation import NOT_FILLED_IN

    rows = [{"code": "scene_estimate:surroundings_not_connected"},
            {"code": "scene_estimate:surroundings_past_frame_budget"}]
    floored = {"scene_estimates": rows, "scene_completion": {
        "generated_geometry_rejected": SCENE_COMPLETION_NOT_ADMITTED,
        "not_admitted_reason": "atlas_completion_visual_review_required"}}
    estimated = _texts(facts.measured_and_estimated(LAMP, floored, None)[1])
    assert facts.ESTIMATE_SENTENCES[SCENE_COMPLETION_NOT_ADMITTED] == NOT_FILLED_IN
    assert NOT_FILLED_IN in estimated
    for code, sentence in facts.NOT_FILLED_IN_SENTENCES.items():
        assert sentence in estimated and facts.ESTIMATE_SENTENCES[code] not in estimated
    assert not any("generated" in text for text in estimated)
    # A scene whose completion was admitted keeps the generated wording and says nothing of gaps left unfilled.
    plain = _texts(facts.measured_and_estimated(LAMP, {"scene_estimates": rows}, None)[1])
    assert NOT_FILLED_IN not in plain
    assert all(facts.ESTIMATE_SENTENCES[row["code"]] in plain for row in rows)
    # Any other record under the same key says nothing.
    other = _texts(facts.measured_and_estimated(LAMP, {"scene_completion": {"status": "completed"}}, None)[1])
    assert NOT_FILLED_IN not in other

#: A preparation as test job 4's was (a table lamp to a glass wall shelf, 2026-11-18): every estimate it records.
LAMP = {"subject": {"description": "table lamp", "aabb_min_xyz": [0.078, -0.3697, 0.539],
                    "aabb_max_xyz": [0.476, 0.005, 1.208], "complete_object_dimensions": False},
        "destination": {"relation": "on", "visible_label": "glass wall shelf", "position_world_m": [0.470, -1.211, 1.308]},
        "registration": {"task_region": {"radius_m": 1.0, "trimmed_rmse_m": 0.075}},
        "coordinate_frame": {"scale_authority": "atlas_reference_estimate", "physical_scale_measured": False,
                             "placement_uncertainty_m": 0.075, "task_coordinates": "Z_up_estimated_meters"},
        "authoring_inputs": {"metric_envelope": {"maximum_dimension_relative_error": 0.25}},
        "physics": {"bounds": {"mass_kg": [0.999, 249.8]}},
        "video_box_fit_check": {"flags": ["task_object_box_fit_offset", "task_object_box_fit_size"],
                                "summary": {"median_size_ratio": {"x": 0.643, "y": 0.467}}},
        "support_height_agreement": {"flags": ["scene_estimate:support_height_differs_from_picture"],
                                     "start_task_plane_minus_picture_m": -0.0786,
                                     "goal_task_plane_minus_picture_m": -0.0828}}


def test_each_estimate_says_how_sure_it_is_and_every_estimate_the_preparation_records_is_named(tmp_path) -> None:
    """Job 4's sheet gave the lamp's size bare and left four of the preparation's own estimates out."""
    from blueprint_pipeline.atlas_scene_estimates import preparation_flags, preparation_record_flags
    measured, estimated = facts.measured_and_estimated(LAMP, {"scene_estimates": []}, None)
    assert _texts(measured) == ["How closely the video's geometry and the reconstruction line up near the job "
                                "(registration residual, trimmed RMS): 8 cm, in the scene's estimated metres."]
    estimated = _texts(estimated)
    assert estimated[0] == ("Scale: set by the 3D reconstruction's own estimate, not measured at the site. Sizes, "
                            "heights and distances may be 0.75 to 1.33 times those given here (the scale band).")
    assert ("Object size: 40 cm × 37 cm × 67 cm (30 cm × 28 cm × 50 cm to 53 cm × 50 cm × 89 cm across the scale "
            "band), from the video; in the frames its outline measures 64 % and 47 % of the box along its sides "
            "(median), so the box may be larger than the object.") in estimated
    assert ("The object's box is fitted to the parts of it the frames show; Blueprint did not check that every side "
            "of it was filmed.") in estimated
    # Only where it starts: the check's goal figure compares the start's seat with another surface.
    assert ("Surface height under the object: where it starts, the simulation seats it 8 cm below the surface the "
            "reconstruction's picture shows there; not corrected.") in estimated
    # Job 4 was prepared before Blueprint estimated weights: it says so, beside the simulation's wide prior.
    assert ("Weight: not estimated: this job was prepared before Blueprint estimated weights. The simulation uses a "
            "wide prior from its size alone: 1 to 250 kg.") in estimated
    flags = {"scene_estimate:placement_uncertainty", "scene_estimate:support_height_differs_from_picture",
             "task_object_box_fit_offset", "task_object_box_fit_size"}
    for code in flags:
        assert facts.ESTIMATE_SENTENCES[code] in estimated
    assert not any("not yet described" in line for line in estimated)
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(LAMP))
    assert preparation_record_flags(LAMP) == preparation_flags(path) == flags
    # Without a dimension prior or a fit table nothing is said about how far sizes may be off.
    bare = {key: value for key, value in LAMP.items() if key != "authoring_inputs"}
    lines = _texts(facts.measured_and_estimated(bare, None, None)[1])
    assert "How far sizes may be off: not recorded." in lines[0]
    assert next(line for line in lines if line.startswith("Object size")).startswith("Object size: 40 cm × 37 cm × 67 cm, ")


def test_where_the_object_starts_and_goes_how_high_and_how_much_room_there_is() -> None:
    """Job 4's sheet gave the shelf as a bare point and said nothing of where the lamp starts or how high either is."""
    handoff = {"task_object_rigidity": {"rigidity": "rigid", "source": "object_analysis", "reason": "PLANTREASON"}}
    rows = facts.object_lines(LAMP, {"ground_plane_offset_m": 0.0}, None, handoff)
    assert rows == [
        {"text": "Object: table lamp.", "answers": None},
        {"text": "Rigid or soft: rigid: it keeps its shape when handled (Blueprint's reading of the video; not "
                 "checked).", "answers": None},
        {"text": "Material, how fragile it is and how it should be held: not recorded.", "answers": None},
        {"text": "Sizes that matter for holding it, from the video: 37 cm across the narrower side of its footprint, "
                 "40 cm across the wider, 67 cm tall. Each may be 0.75 to 1.33 times that (the scale band). No "
                 "particular gripper or hand was checked against them.", "answers": None},
        # Blueprint's estimate of the site's size question names it: the website shows the site's answer instead.
        {"text": "Longest side: about 67 cm (Blueprint's estimate; the site did not say).",
         "answers": "item_size_longest_side"},
        {"text": "Starts: the middle of its base at (0.277, -0.182, 0.539) m, about 54 cm above the floor (40 cm to "
                 "72 cm across the scale band).", "answers": None},
        {"text": "Goes to: glass wall shelf; centre (0.470, -1.211, 1.308) m; about 131 cm above the floor (98 cm to "
                 "174 cm across the scale band).", "answers": None},
        {"text": "Start to destination: about 1.30 m in a straight line (0.97 m to 1.73 m across the scale band), "
                 "from the middle of its base to the place point.", "answers": None},
        {"text": "Positions are in the simulation's coordinates, in metres with z up; heights are above the scene's "
                 "floor, which Blueprint levelled to height 0. Each may also be off by about 8 cm, the registration "
                 "residual near the job.", "answers": None}]
    assert "PLANTREASON" not in json.dumps(rows)  # the model's own words about the object are not quoted
    # With the fit table: its floor, where a base stands, how wide; never whether one particular gripper fits.
    table = {"rows": [{"checks": [{"check": "stance", "values": {"station_footprint_half_extent_m": [0.15, 0.15]}}]}],
             "inputs": {"scale_band": [0.75, 1.3333], "point_rule": {"grasp": "object_top_plus_tool_length_top_down"},
                        "tool": {"tool_id": "robotiq_2f85"}, "straight_line_m": [0.9, 1.6],
                        "stations": {"positions_world_m": [[0.9, 0.1, 0.0], [0.9, -0.4, 0.0]]},
                        "floor_height": {"floor_z_m": 0.01},
                        "floor_search": {"result": {"floor_z_m": 0.01, "lower_half_m": 0.4, "upper_half_m": 0.45,
                                                    "upper_hole_possible": True,
                                                    "searches": [{"half_extent_m": 0.45, "clear": False}]}}}}
    lines = _texts(facts.object_lines(LAMP, {"ground_plane_offset_m": 0.0}, table, {}))
    assert not any("85 mm" in line or "Robotiq" in line or "two-finger" in line for line in lines)
    assert any(line.startswith("Starts: ") and "about 53 cm above the floor" in line for line in lines)
    assert not any(line.startswith("Start to destination") for line in lines)  # the table's carry line says it
    assert any("heights are above the floor next to the task, as Blueprint's robot fit check found it." in line
               for line in lines)
    assert "Room to stand: 2 clear spot(s) for a 30 cm square robot base were found next to the task." in lines
    assert ("Widest clear floor found next to the task: a square at least 60 cm wide; one wider than 120 cm was not "
            "clear (or the scan has a gap there).") in lines
    # A material the video analysis read is named, as its reading.
    read = _texts(facts.object_lines({**LAMP, "weight_estimate": {"material": "metal"}}, {}, None, None))
    assert read[1].startswith("Material: mostly metal, as Blueprint's video analysis read it (not checked);")
    # Without a known floor a height is a bare coordinate, never a guess.
    bare = _texts(facts.object_lines(LAMP, {}, None, None))
    assert any("at height 0.54 in the scene's coordinates (where the floor is: not recorded)" in line for line in bare)
    assert not any("above the floor" in line for line in bare)


def test_a_floor_search_the_time_budget_stopped_claims_nothing_about_wider_floor():
    """Review of #3650: an upper bound no search failed at is the search's limit, not a measured restriction."""
    table = {"inputs": {"scale_band": [0.75, 1.3333], "floor_height": {"floor_z_m": 0.01},
                        "floor_search": {"result": {"floor_z_m": 0.01, "lower_half_m": 0.4, "upper_half_m": 0.9,
                                                    "budget_exhausted": True, "searches": []}}}}
    lines = _texts(facts.object_lines(LAMP, {"ground_plane_offset_m": 0.0}, table, {}))
    # Without a stance check's recorded footprint the base's size is not given.
    table = {**table, "inputs": {**table["inputs"], "stations": {"positions_world_m": []}}}
    assert ("Room to stand: no clear spot for a robot base was found next to the task; the scan may have gaps there."
            in _texts(facts.object_lines(LAMP, {}, table, {})))
    assert [line for line in lines if line.startswith("Widest clear floor")] == [
        "Widest clear floor found next to the task: a square at least 60 cm wide; wider floor was not assessed."]


def _job(tmp_path, *, context=CONTEXT, preparation=LAMP):
    root = capture_root(tmp_path)
    base = root / "pipeline" / "website_scene_preparation" / "base_scene.json"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(json.dumps({"ground_plane_offset_m": 0.0,
                                "scene_estimates": [{"code": "scene_estimate:destination_guessed"}]}))
    write_fit_table(root, context=context, preparation=preparation,
                    handoff={"base_scene_path": str(base),
                             "task_object_rigidity": {"rigidity": "rigid", "source": "object_analysis"}})
    return root


def test_the_record_reads_what_the_handoff_names_bound_to_the_current_task(tmp_path) -> None:
    root = _job(tmp_path)
    value = facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT)
    assert value["schema_version"] == "website_packet_facts.v1" and value["truncated"] is False
    assert {"text": "Object: table lamp.", "answers": None} in value["object"]
    assert any(row["text"].startswith("Rigid or soft: rigid") for row in value["object"])
    assert facts.ESTIMATE_SENTENCES["scene_estimate:destination_guessed"] in _texts(value["estimated"])
    assert len(json.dumps(value, separators=(",", ":")).encode()) <= facts.MAX_BYTES
    # Never a path, a storage link or an id of the job.
    text = json.dumps(value)
    assert str(tmp_path) not in text and "walkthrough-req1" not in text and "site-req1" not in text
    # Another task context, no handoff yet, or a handoff whose seal is broken: nothing to show, never an error.
    assert facts.read_packet_facts(capture_root=root, task_context_digest="sha256:" + "d" * 64) is None
    assert facts.read_packet_facts(capture_root=capture_root(tmp_path, capture_id="walkthrough-req2"),
                                   task_context_digest=CONTEXT) is None
    handoff = root / "pipeline" / "website_scene_preparation" / "handoff.json"
    handoff.write_text(handoff.read_text().replace("table lamp", "other"))
    sealed = json.loads(handoff.read_text())
    sealed["digest"] = "sha256:" + "0" * 64
    handoff.write_text(json.dumps(sealed))
    assert facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT) is None


def test_the_record_stays_inside_its_bound(tmp_path) -> None:
    long = {**LAMP, "subject": {**LAMP["subject"], "description": "lamp " * 400}}
    flags = [{"code": f"scene_estimate:unknown_{index}"} for index in range(40)]
    root = _job(tmp_path, preparation=long)
    base = root / "pipeline" / "website_scene_preparation" / "base_scene.json"
    base.write_text(json.dumps({"ground_plane_offset_m": 0.0, "scene_estimates": flags}))
    value = facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT)
    assert len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) <= facts.MAX_BYTES
    assert all(len(row["text"]) <= facts.MAX_LINE for section in ("object", "measured", "estimated")
               for row in value[section])


def test_no_price_and_none_of_blueprints_spend_reaches_the_record(tmp_path) -> None:
    from tests.money_copy import price_findings
    root = _job(tmp_path, preparation={**LAMP, "reserved_cost_usd": 37.25, "budget_note": "Scene budget $40"})
    value = facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT)
    assert value is not None and not price_findings(json.dumps(value))


def _room_package_job(tmp_path, *, bound=True, cause="other_geometry_in_its_box_wherever_it_would_rest"):
    """A room package shipped on compile's own stop, as the scene handoff leaves it: its sealed record names the
    preparation compile wrote beside it by digest, and its handoff names no preparation."""
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root = _job(tmp_path, preparation={**LAMP, "blockers": ["task_destination_surface_contact_required"],
                                       "destination_unheld": {"status": "no_spot_holds_it", "cause": cause}})
    work = root / "pipeline" / "website_scene_preparation"
    prepared = json.loads((work / "preparation.json").read_text())
    record = {"schema_version": "blueprint.room_package_inputs.v1", "task_context_digest": CONTEXT,
              "not_prepared": [{"piece": "destination", "code": "task_destination_surface_contact_required"}],
              **({"preparation_digest": prepared["digest"]} if bound else {})}
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    write_json(work / "room_package" / "half" / "room_package_inputs.v1.json", record)
    handoff = {key: value for key, value in json.loads((work / "handoff.json").read_text()).items()
               if key not in {"preparation_path", "preparation_digest", "robot_fit_table", "digest"}}
    handoff.update(status="room_package_written", room_package={
        "path": str(work / "room_package" / "half" / "room_package_inputs.v1.json"), "inputs_digest": record["inputs_digest"]})
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(work / "handoff.json", handoff)
    return root


def test_a_room_package_in_a_corrections_workspace_reads_the_preparation_beside_it(tmp_path):
    """Codex, #3735: after a site's correction, compile writes the preparation and the room package under
    ``website_scene_preparation/revisions/<digest>/`` while ``handoff.json`` stays at the top. The preparation the
    room package names is read from the room package's own workspace, not from beside the handoff."""
    import shutil

    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root = _room_package_job(tmp_path)
    work = root / "pipeline" / "website_scene_preparation"
    revision = work / "revisions" / ("e" * 64)
    revision.mkdir(parents=True)
    shutil.move(str(work / "preparation.json"), revision / "preparation.json")
    shutil.move(str(work / "room_package"), revision / "room_package")
    # An older preparation left beside the handoff must not be the one read.
    stale = json.loads((revision / "preparation.json").read_text())
    stale["subject"]["description"] = "an older lamp"
    stale["digest"] = canonical_digest(stale, digest_field="digest")
    write_json(work / "preparation.json", stale)
    handoff = json.loads((work / "handoff.json").read_text())
    handoff["room_package"]["path"] = str(revision / "room_package" / "half" / "room_package_inputs.v1.json")
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(work / "handoff.json", handoff)
    value = facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT)
    assert value is not None and {"text": "Object: table lamp.", "answers": None} in value["object"]


def test_a_room_package_shipped_on_compiles_stop_shows_the_objects_estimates_and_why_it_was_not_placed(tmp_path):
    """Job move-the-lamp-2 (24 Nov): the room package's handoff names no preparation, so the company's page read
    "Nothing about the object is measured or estimated yet." under "add a stand-in of your own, sized from the facts
    about the object below", though compile had estimated the lamp. The preparation the room package's record names
    by digest is read instead, and says why the lamp was not placed."""
    from blueprint_pipeline import website_destination_unheld as unheld
    root = _room_package_job(tmp_path)
    value = facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT)
    assert value is not None and {"text": "Object: table lamp.", "answers": None} in value["object"]
    assert any(row["text"].startswith("Sizes that matter for holding it") for row in value["object"])
    assert unheld.sentence(unheld.IN_THE_WAY, "the surface it goes to") in _texts(value["estimated"])
    assert ("Blueprint could not place the job's object on the surface it goes to: at every spot it tried with the "
            "object's centre at least 5 cm inside the surface's edges and at least half of the object over the surface, "
            "the object's estimated box, which takes in the whole object and not only its base, reached into another "
            "surface of the room.") in _texts(value["estimated"])
    assert facts.read_packet_facts(capture_root=root, task_context_digest="sha256:" + "d" * 64) is None
    # A record that names no preparation, or another one: nothing is shown, as before.
    assert facts.read_packet_facts(capture_root=_room_package_job(tmp_path / "unbound", bound=False),
                                   task_context_digest=CONTEXT) is None
    work = root / "pipeline" / "website_scene_preparation"
    prepared = json.loads((work / "preparation.json").read_text())
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    prepared["subject"]["description"] = "another lamp"
    prepared["digest"] = canonical_digest(prepared, digest_field="digest")
    (work / "preparation.json").write_text(json.dumps(prepared))
    assert facts.read_packet_facts(capture_root=root, task_context_digest=CONTEXT) is None


def test_an_unknown_cause_says_nothing_about_why_and_the_words_keep_the_scorers_radius():
    from blueprint_pipeline import website_destination_unheld as unheld
    from blueprint_pipeline.website_surface_target_admission import WEBSITE_TARGET_TOLERANCE_M
    assert unheld.CENTRE_INSIDE_M == WEBSITE_TARGET_TOLERANCE_M
    assert all(unheld.sentence(cause, "x") for cause in unheld.CAUSES)
    assert unheld.sentence("a builder's own words", "x") is None and unheld.sentence(None, "x") is None
    texts = _texts(facts.measured_and_estimated({**LAMP, "destination_unheld": {"cause": "free text"}}, None, None)[1])
    assert not any("could not place" in text for text in texts)
