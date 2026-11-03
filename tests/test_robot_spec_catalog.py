"""Every robot spec figure is traceable to its source, or the catalog is refused with every reason.

Fixture catalogs only: the shipped file is replaced by a researcher's, so no test reads its figures.
"""
from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.robot_spec_catalog import (
    CATALOG_PATH,
    RobotSpecCatalogError,
    load_robot_spec_catalog,
    validate_robot_spec_catalog,
)

RECIPES = {"arm_recipe", "mobile_recipe"}
SOURCE = "https://example.com/datasheet.pdf"


def _figure(value, unit, quote, conversion=None):
    return {"value": value, "unit": unit, "source_url": SOURCE, "quote": quote, "conversion": conversion}


def _model(model_id="arm_a", body_type="fixed_arm", *, recipe_id=None, **figures):
    """A fixed arm with every figure but a length-and-width footprint, unless overridden; each
    figure left null gets a reason, except a fixed arm's base speed."""
    row = {"max_reach_m": _figure(0.9, "m", "Reach 900 mm", "900 mm = 0.9 m"),
           "rated_payload_kg": _figure(5, "kg", "Payload 5 kg"),
           "max_tcp_speed_m_s": _figure(1.0, "m/s", "Maximum TCP speed 1 m/s"),
           "base_footprint_diameter_m": _figure(0.2, "m", "Footprint 200 mm", "200 mm = 0.2 m"),
           "base_footprint_l_w_m": None,
           "robot_mass_kg": _figure(20, "kg", "Weight 20 kg"),
           "repeatability_mm": _figure(0.05, "mm", "Repeatability 0.05 mm"),
           "degrees_of_freedom": _figure(6, "count", "Degrees of freedom 6"),
           "max_base_speed_m_s": None}
    row.update(figures)
    unknown = {name: "The sheet does not give it." for name, figure in row.items() if figure is None
               and not (name == "max_base_speed_m_s" and body_type == "fixed_arm")}
    return {"model_id": model_id, "recipe_id": recipe_id, "manufacturer": "Example Robotics",
            "name": model_id.replace("_", " ").title(), "body_type": body_type, "mounting": ["floor"],
            "figures": row, "unknown_figures": unknown, "notes": "fixture"}


def _catalog(*models, tool_length=0.15):
    return {"schema_version": "robot_spec_catalog.v1", "retrieved_on": "2026-11-02",
            "default_tool": {"tool_id": "example_gripper", "name": "Example gripper",
                             "figures": {"stroke_m": _figure(0.085, "m", "Stroke 85 mm", "85 mm = 0.085 m"),
                                         "mass_kg": _figure(0.9, "kg", "Weight 0.9 kg"),
                                         "length_m": None if tool_length is None else _figure(
                                             tool_length, "m", "Length 150 mm", "150 mm = 0.15 m")},
                             "unknown_figures": {} if tool_length is not None else {
                                 "length_m": "The sheet gives no length."}},
            "models": list(models) or [_model()]}


def sealed(*models, **options):
    """A fixture catalog as the loader returns it."""
    return validate_robot_spec_catalog(_catalog(*models, **options), recipe_ids=RECIPES)


def test_a_traceable_catalog_is_sealed_with_a_stable_digest():
    mobile = _model("mobile_b", "mobile_manipulator", recipe_id="mobile_recipe",
                    max_base_speed_m_s=_figure(1.5, "m/s", "Base speed 1.5 m/s"))
    value = _catalog(_model(recipe_id="arm_recipe"), mobile)
    catalog = validate_robot_spec_catalog(value, recipe_ids=RECIPES)
    assert catalog["catalog_digest"] == canonical_digest(catalog, digest_field="catalog_digest")
    # Key order does not matter, and re-validating a sealed catalog changes nothing.
    reordered = json.loads(json.dumps(value, sort_keys=True))
    assert validate_robot_spec_catalog(reordered, recipe_ids=RECIPES) == catalog
    assert validate_robot_spec_catalog(catalog, recipe_ids=RECIPES) == catalog
    changed = copy.deepcopy(value)
    changed["models"][0]["figures"]["rated_payload_kg"]["value"] = 6
    assert validate_robot_spec_catalog(changed, recipe_ids=RECIPES)["catalog_digest"] != catalog["catalog_digest"]


def _set(path, item):
    def apply(catalog):
        *parents, last = path
        target = catalog
        for key in parents:
            target = target[key]
        if item is KeyError:
            del target[last]
        else:
            target[last] = item
    return apply


REACH = ("models", 0, "figures", "max_reach_m")
INVALID = {
    "duplicate_model_id": (lambda c: c["models"].append(copy.deepcopy(c["models"][0])),
                           "robot_spec_catalog_duplicate_model_id:arm_a"),
    "unknown_recipe": (_set(("models", 0, "recipe_id"), "no_such_recipe"), "robot_spec_catalog_recipe_unknown:arm_a"),
    "missing_source": (_set((*REACH, "source_url"), KeyError), "robot_spec_catalog_source_url_missing:arm_a.max_reach_m"),
    "plain_http_source": (_set((*REACH, "source_url"), "http://example.com/datasheet.pdf"),
                          "robot_spec_catalog_source_url_missing:arm_a.max_reach_m"),
    "missing_quote": (_set((*REACH, "quote"), KeyError), "robot_spec_catalog_quote_missing:arm_a.max_reach_m"),
    "sixteen_word_quote": (_set((*REACH, "quote"), " ".join(["word"] * 15 + ["900"])),
                           "robot_spec_catalog_quote_too_long:arm_a.max_reach_m"),
    "quote_without_number": (_set(("models", 0, "figures", "rated_payload_kg", "quote"), "Payload five kilograms"),
                             "robot_spec_catalog_quote_without_number:arm_a.rated_payload_kg"),
    "unit_mismatch": (_set((*REACH, "unit"), "mm"), "robot_spec_catalog_unit_mismatch:arm_a.max_reach_m"),
    "null_without_reason": (_set(("models", 0, "figures", "robot_mass_kg"), None),
                            "robot_spec_catalog_null_figure_reason_missing:arm_a.robot_mass_kg"),
    "mobile_base_speed_without_reason": (_set(("models", 0, "body_type"), "mobile_manipulator"),
                                         "robot_spec_catalog_null_figure_reason_missing:arm_a.max_base_speed_m_s"),
    "tool_length_without_reason": (_set(("default_tool", "figures", "length_m"), None),
                                   "robot_spec_catalog_null_figure_reason_missing:default_tool.length_m"),
    "non_positive_value": (_set((*REACH, "value"), 0), "robot_spec_catalog_value_invalid:arm_a.max_reach_m"),
    "misspelled_figure": (_set(("models", 0, "figures", "max_reach"), None),
                          "robot_spec_catalog_figure_unrecognized:arm_a.max_reach"),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_each_untraceable_or_malformed_figure_is_refused_by_name(case):
    mutate, code = INVALID[case]
    value = _catalog()
    mutate(value)
    with pytest.raises(RobotSpecCatalogError) as refused:
        validate_robot_spec_catalog(value, recipe_ids=RECIPES)
    assert code in refused.value.errors


def test_every_reason_is_listed_at_once():
    value = _catalog()
    value["models"][0]["figures"]["max_reach_m"]["unit"] = "mm"
    value["models"][0]["figures"]["robot_mass_kg"] = None
    with pytest.raises(RobotSpecCatalogError) as refused:
        validate_robot_spec_catalog(value, recipe_ids=RECIPES)
    assert set(refused.value.errors) == {"robot_spec_catalog_unit_mismatch:arm_a.max_reach_m",
                                        "robot_spec_catalog_null_figure_reason_missing:arm_a.robot_mass_kg"}


def test_a_stated_conversion_or_reason_and_a_fixed_arm_base_speed_are_accepted():
    # Spelled-out numbers need the arithmetic; unknown figures need a reason; a fixed arm has no base.
    worded = _model(degrees_of_freedom=_figure(6, "count", "six rotating joints", "six = 6"), robot_mass_kg=None)
    catalog = sealed(worded)
    assert catalog["models"][0]["unknown_figures"] == {"base_footprint_l_w_m": "The sheet does not give it.",
                                                       "robot_mass_kg": "The sheet does not give it."}
    assert catalog["models"][0]["figures"]["max_base_speed_m_s"] is None


def test_recipe_ids_default_to_the_asset_recipe_catalog():
    validate_robot_spec_catalog(_catalog(_model(recipe_id="droid_franka_robotiq_2f85")))
    with pytest.raises(RobotSpecCatalogError) as refused:
        validate_robot_spec_catalog(_catalog(_model(recipe_id="no_such_recipe")))
    assert refused.value.errors == ("robot_spec_catalog_recipe_unknown:arm_a",)


def test_a_file_with_a_repeated_key_is_refused(tmp_path):
    text = json.dumps(_catalog())
    path = tmp_path / "catalog.json"
    path.write_text(text.replace('"schema_version": "robot_spec_catalog.v1"',
                                 '"schema_version": "robot_spec_catalog.v1", "schema_version": "robot_spec_catalog.v1"'))
    with pytest.raises(RobotSpecCatalogError) as refused:
        load_robot_spec_catalog(path, recipe_ids=RECIPES)
    assert refused.value.errors == ("robot_spec_catalog_duplicate_json_key:schema_version",)
    path.write_text(text)
    assert load_robot_spec_catalog(path, recipe_ids=RECIPES) == sealed()


def test_the_shipped_catalog_is_traceable():
    # Whatever figures it holds, the file dropped in at this path must pass its own validator.
    assert CATALOG_PATH.name == "robot_spec_catalog.v1.json"
    catalog = load_robot_spec_catalog()
    assert catalog["catalog_digest"].startswith("sha256:")
