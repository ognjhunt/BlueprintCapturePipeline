"""What a deployment company reads about the job's object on its job page (preparation status v9).

The by-hand job packet export is retired (owner, 2026-11-19: one delivery path, the website). Its object, height
and uncertainty lines, and Blueprint's estimates of what the site did not say, reach the company's job page as one
small record the website renders; these are the export's checks, carried over.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_packet_facts.py
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
    assert CODES | TASK_OBJECT_BOX_FIT_FLAGS <= set(facts.ESTIMATE_SENTENCES)


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
    assert _texts(measured) == ["How closely the video's geometry and the reconstruction line up around the task "
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
        {"text": "Rigid or soft: rigid: it keeps its shape when handled (Blueprint's model reading of the video; not "
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
                 "residual near the task.", "answers": None}]
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
