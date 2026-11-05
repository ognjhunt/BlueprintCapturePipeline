"""Robot models' public spec-sheet figures, for the first and cheapest fit tier of a website job.

A posted job is matched to robot models in tiers. The first (``website_robot_fit``) runs on the CPU
before any scene-build spend: from the site video's geometry and these figures, whether each model
can stand, reach, carry and keep pace. Policy simulation of a shortlist is a later tier.

Every figure is traceable to its manufacturer: a value in the figure's one unit, the ``https`` page
it came from, and at most 15 words quoted from that page that contain the number, or an exact
``conversion`` when the page states it differently (``"1300 mm = 1.3 m"``). A figure the page does
not give is ``null`` with its reason in ``unknown_figures``; nothing is inferred. A fixed arm has no
base speed and may leave it ``null`` without a reason. The figures describe the product, not a unit
measured on a site, so a fit built on them is a bounded estimate.

Units: ``m`` (``max_reach_m``, ``base_footprint_diameter_m``, ``base_footprint_l_w_m`` as
``[length, width]``, and the tool's ``stroke_m`` and ``length_m``), ``kg`` (``rated_payload_kg``,
``robot_mass_kg``, the tool's ``mass_kg``), ``m/s`` (``max_tcp_speed_m_s``, ``max_base_speed_m_s``),
``mm`` (``repeatability_mm``) and ``count`` (``degrees_of_freedom``, a whole number).

The loader fails closed and lists every reason at once, so a whole catalog can be corrected in one
pass. It returns the catalog sealed with ``catalog_digest``.
"""
from __future__ import annotations

import datetime
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import DecisionEvidenceContractError, canonical_digest

SCHEMA_VERSION = "robot_spec_catalog.v1"
CATALOG_PATH = Path(__file__).parent / "_catalog_data" / "robot_spec_catalog.v1.json"
RECIPE_CATALOG_PATH = Path(__file__).parent / "_catalog_data" / "embodiment_asset_recipes.v1.json"
BODY_TYPES = ("fixed_arm", "mobile_manipulator", "humanoid")
MODEL_FIGURE_UNITS = {"max_reach_m": "m", "rated_payload_kg": "kg", "max_tcp_speed_m_s": "m/s",
                      "base_footprint_diameter_m": "m", "base_footprint_l_w_m": "m", "robot_mass_kg": "kg",
                      "repeatability_mm": "mm", "degrees_of_freedom": "count", "max_base_speed_m_s": "m/s"}
TOOL_FIGURE_UNITS = {"stroke_m": "m", "mass_kg": "kg", "length_m": "m"}
#: The default tool must state these; its length may be unknown, with a reason.
REQUIRED_TOOL_FIGURES = ("stroke_m", "mass_kg")
#: ``[length, width]``; every other figure is one number.
PAIR_FIGURES = frozenset({"base_footprint_l_w_m"})
MAXIMUM_QUOTE_WORDS = 15
_FIGURE_KEYS = frozenset({"value", "unit", "source_url", "quote", "conversion"})
_MODEL_REQUIRED = frozenset({"model_id", "recipe_id", "manufacturer", "name", "body_type", "mounting", "figures"})
_MODEL_OPTIONAL = frozenset({"unknown_figures", "notes"})
_DIGIT = re.compile(r"[0-9]")


class RobotSpecCatalogError(DecisionEvidenceContractError):
    """The catalog cannot be read safely or a figure cannot be traced to its source; ``errors`` lists why."""


def known_recipe_ids() -> frozenset[str]:
    """Every robot recipe (embodiment) id in the asset recipe catalog."""
    value = json.loads(RECIPE_CATALOG_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "embodiment_asset_recipes.v1" or not isinstance(value.get("recipes"), list):
        raise RobotSpecCatalogError(["robot_spec_catalog_recipe_catalog_invalid"])
    return frozenset(str(row["embodiment_id"]) for row in value["recipes"])


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value > 0)


def _figure_errors(owner: str, row: Mapping[str, Any], units: Mapping[str, str],
                   may_omit: Iterable[str] = ()) -> list[str]:
    figures, unknown = row.get("figures"), row.get("unknown_figures", {})
    if not isinstance(figures, Mapping):
        return [f"robot_spec_catalog_figures_invalid:{owner}"]
    if not isinstance(unknown, Mapping) or not all(_text(reason) for reason in unknown.values()):
        return [f"robot_spec_catalog_unknown_figures_invalid:{owner}"]
    errors = [f"robot_spec_catalog_figure_unrecognized:{owner}.{name}"
              for name in [*figures, *unknown] if name not in units]
    for name, unit in units.items():
        figure, where = figures.get(name), f"{owner}.{name}"
        if figure is None:
            if name not in unknown and name not in may_omit:
                errors.append(f"robot_spec_catalog_null_figure_reason_missing:{where}")
            continue
        if name in unknown:
            errors.append(f"robot_spec_catalog_figure_known_and_unknown:{where}")
        if not isinstance(figure, Mapping) or set(figure) - _FIGURE_KEYS:
            errors.append(f"robot_spec_catalog_figure_invalid:{where}")
            continue
        value = figure.get("value")
        if name in PAIR_FIGURES:
            valid = isinstance(value, list) and len(value) == 2 and all(_positive(v) for v in value)
        else:
            valid = _positive(value) and (unit != "count" or float(value).is_integer())
        if not valid:
            errors.append(f"robot_spec_catalog_value_invalid:{where}")
        if figure.get("unit") != unit:
            errors.append(f"robot_spec_catalog_unit_mismatch:{where}")
        url = figure.get("source_url")
        if not isinstance(url, str) or not re.fullmatch(r"https://[^\s/]+\.[^\s/]+(/\S*)?", url):
            errors.append(f"robot_spec_catalog_source_url_missing:{where}")
        quote, conversion = figure.get("quote"), figure.get("conversion")
        if conversion is not None and not _text(conversion):
            errors.append(f"robot_spec_catalog_conversion_invalid:{where}")
        if not _text(quote):
            errors.append(f"robot_spec_catalog_quote_missing:{where}")
            continue
        if len(quote.split()) > MAXIMUM_QUOTE_WORDS:
            errors.append(f"robot_spec_catalog_quote_too_long:{where}")
        if not _DIGIT.search(quote) and not _text(conversion):
            errors.append(f"robot_spec_catalog_quote_without_number:{where}")
    return errors


def validate_robot_spec_catalog(value: Mapping[str, Any], *,
                                recipe_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """The catalog, sealed with ``catalog_digest``; raise ``RobotSpecCatalogError`` with every reason.

    ``recipe_ids`` defaults to the asset recipe catalog's (``known_recipe_ids``).
    """
    try:
        catalog = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RobotSpecCatalogError(["robot_spec_catalog_not_json"]) from exc
    if not isinstance(catalog, dict):
        raise RobotSpecCatalogError(["robot_spec_catalog_not_json"])
    catalog.pop("catalog_digest", None)
    errors: list[str] = []
    if catalog.get("schema_version") != SCHEMA_VERSION:
        errors.append("robot_spec_catalog_schema_invalid")
    try:
        datetime.date.fromisoformat(catalog.get("retrieved_on"))
        dated = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", catalog["retrieved_on"]))
    except (TypeError, ValueError):
        dated = False
    if not dated:
        errors.append("robot_spec_catalog_retrieved_on_invalid")
    tool = catalog.get("default_tool")
    if not isinstance(tool, dict) or not _text(tool.get("tool_id")) or not _text(tool.get("name")):
        errors.append("robot_spec_catalog_default_tool_invalid")
    else:
        errors += _figure_errors("default_tool", tool, TOOL_FIGURE_UNITS)
        errors += [f"robot_spec_catalog_figure_required:default_tool.{name}" for name in REQUIRED_TOOL_FIGURES
                   if not isinstance(tool.get("figures"), Mapping) or tool["figures"].get(name) is None]
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        errors.append("robot_spec_catalog_models_missing")
        models = []
    recipes = known_recipe_ids() if recipe_ids is None else frozenset(recipe_ids)
    seen: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict) or not _text(model.get("model_id")):
            errors.append(f"robot_spec_catalog_model_invalid:{index}")
            continue
        key = model["model_id"]
        if key in seen:
            errors.append(f"robot_spec_catalog_duplicate_model_id:{key}")
        seen.add(key)
        if _MODEL_REQUIRED - set(model) or set(model) - _MODEL_REQUIRED - _MODEL_OPTIONAL:
            errors.append(f"robot_spec_catalog_model_fields_invalid:{key}")
        recipe = model.get("recipe_id")
        if recipe is not None and (not isinstance(recipe, str) or recipe not in recipes):
            errors.append(f"robot_spec_catalog_recipe_unknown:{key}")
        if not _text(model.get("manufacturer")) or not _text(model.get("name")):
            errors.append(f"robot_spec_catalog_model_name_missing:{key}")
        if model.get("body_type") not in BODY_TYPES:
            errors.append(f"robot_spec_catalog_body_type_invalid:{key}")
        if not isinstance(model.get("mounting"), list) or not all(_text(item) for item in model["mounting"]):
            errors.append(f"robot_spec_catalog_mounting_invalid:{key}")
        if "notes" in model and not isinstance(model["notes"], str):
            errors.append(f"robot_spec_catalog_notes_invalid:{key}")
        errors += _figure_errors(key, model, MODEL_FIGURE_UNITS,
                                 ("max_base_speed_m_s",) if model.get("body_type") == "fixed_arm" else ())
    if errors:
        raise RobotSpecCatalogError(errors)
    catalog["catalog_digest"] = canonical_digest(catalog, digest_field="catalog_digest")
    return catalog


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            # A repeated key would silently replace a figure.
            raise RobotSpecCatalogError([f"robot_spec_catalog_duplicate_json_key:{key}"])
        value[key] = item
    return value


def load_robot_spec_catalog(path: str | Path | None = None, *,
                            recipe_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Read and validate a catalog file (the shipped ``CATALOG_PATH`` by default)."""
    source = Path(path) if path is not None else CATALOG_PATH
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except RobotSpecCatalogError:
        raise
    except ValueError as exc:
        raise RobotSpecCatalogError(["robot_spec_catalog_not_json"]) from exc
    return validate_robot_spec_catalog(value, recipe_ids=recipe_ids)


def figure_value(row: Mapping[str, Any], name: str) -> Any:
    """A validated figure's value, or ``None`` when the catalog records it as unknown."""
    figure = (row.get("figures") or {}).get(name)
    return None if figure is None else figure["value"]


__all__ = ["CATALOG_PATH", "SCHEMA_VERSION", "RobotSpecCatalogError", "figure_value", "known_recipe_ids",
           "load_robot_spec_catalog", "validate_robot_spec_catalog"]
