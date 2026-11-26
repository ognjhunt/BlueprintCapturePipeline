"""Simulated variations of the job's objects in the company's package (backlog: "Variation within plausible ranges").

A deployment company tests the job in its own tools, and one object at one size, mass and friction says little about
how much the job depends on what the video could not pin down. So the package (``website_company_simulation``) also
carries ``variations.usda``: scene.usda as its one sublayer and, on each of the job's objects in it
(``/World/Object``, ``/World/Object_02``, ...), one variant set, ``simulated_variation``: ``nominal`` (the object as
scene.usda places it) and ``sample_01`` to ``sample_08``. A sample overrides only what the object's own file holds
(``_inspect``); scene.usda and every part's file are unchanged:

- its size: one uniform scale on the part's ``Asset`` prim about the bottom centre of its box, so it stays on the
  surface it starts on; not for an object with joints, whose joint ranges and rule are set for its size;
- its mass: ``physics:mass`` on its one rigid body (not for an object with joints, several bodies, or a mass or
  density set on another of its prims);
- its friction: ``physics:staticFriction`` and ``physics:dynamicFriction`` on its physics materials, when they agree
  (a sample's dynamic friction never above its static friction unless its own range is);
- its look: each authored ``primvars:displayColor`` of at most ``MAX_COLOUR_VALUES`` colours, scaled and clipped to
  0..1, and each authored, unconnected ``inputs:roughness`` of a ``UsdPreviewSurface``, shifted and clipped to 0..1.

``variations.json`` lists each object's ranges, what each rests on and every sample's values, and what was not varied
and why. The README says plainly that these are simulated variations for testing in the company's own tools, not
measurements of the site's objects (``readme_lines``).

Each range records its basis, the first of these that gives one:

- ``site_answer``: the site's quick answer of the item's weight (``operator_task_details.item_weight``, one weight,
  give or take ``OWNER_WEIGHT_RELATIVE_BAND``, as the robot fit reads it), or a size or mass the build plan quotes
  from the site's words (``website_build_plan``, ``basis: site_words``);
- ``video_estimate``: the build plan's estimated size or mass; Blueprint's weight estimate from the object's size and
  likely material in the video (``website_job_estimates``); the mass prior from its size in the video that the
  preparation's physics records (the range success_rule.json gives); for size, the scale band of the video's sizes,
  ``[1 - f, 1 / (1 - f)]``, as the company's job page gives it;
- ``default``: a narrow band around the value in the object's file (``DEFAULT_BANDS``). Friction and look are always
  this: nothing reads them from the video (the preparation's friction range is Blueprint's standard prior).

The evidence is read from the half's own preparation and the job's sealed build plan (``evidence``, with the half's
other disclosures, ``website_half_preparation``). It describes the one object that preparation authored, the plan's
first object part (``EVIDENCE_PART``); a further object gets default bands until its own evidence reaches the package.
Only numbers and fixed words ship: never the site's words (a build plan quote, an answer's text, the plan's names).

Samples are deterministic: each draw is a sha256 of the plan's digest, the object's part and file digest, the sample
and the quantity, so the same inputs give the same bytes, and the same zip. Neither digest ships.

Rights are a hard stop: an object in a package that ships a library model is varied only under a license that allows
adaptations (``ADAPTATION_ALLOWED``, of the catalog's own licenses), and the README says the variants change that
model (CC BY 4.0 3(a)(1)(B)); a library credit that cannot be read varies nothing. A package without the job's object
(a room package, a floor) has no variations. Both files go through the package's outbound scan with every other file.
Local files and ``pxr`` only: no model, provider or paid resource.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import website_library_notes as library_notes
from .website_job_estimates import OWNER_WEIGHT_RELATIVE_BAND, PHYSICS, owner_stated_weight_kg

logger = logging.getLogger(__name__)

GENERATOR = "website_object_variations.v1"
SCHEMA_VERSION = "simulated_variations.v1"
LAYER_FILE = library_notes.VARIATIONS_FILE  # named in a library model's credit when it ships
SIDECAR_FILE = "variations.json"
VARIANT_SET = "simulated_variation"
NOMINAL = "nominal"
SAMPLES = 8
SITE_ANSWER, VIDEO_ESTIMATE, DEFAULT = "site_answer", "video_estimate", "default"
#: The part the preparation's evidence describes: the plan's first object (one object per job today).
EVIDENCE_PART = "Object"
#: Where nothing the site said or the video showed gives a range: a narrow band around the value in the object's file,
#: as a factor (a shift for roughness), so a sample stays close to the object as built.
DEFAULT_BANDS = {"scale": (0.9, 1.1), "mass_kg": (0.8, 1.25), "static_friction": (0.8, 1.2),
                 "dynamic_friction": (0.8, 1.2), "display_color_factor": (0.85, 1.15),
                 "roughness_offset": (-0.15, 0.15)}
QUANTITIES = tuple(DEFAULT_BANDS)
UNITS = {"scale": "factor of the object's size in scene.usda, the same on every side",
         "mass_kg": "kilograms", "static_friction": "coefficient", "dynamic_friction": "coefficient",
         "display_color_factor": "factor of each colour in the object's file, each channel clipped to 0..1",
         "roughness_offset": "added to each roughness in the object's file, clipped to 0..1"}
BASES = {SITE_ANSWER: "the site's own answer",
         VIDEO_ESTIMATE: "Blueprint's estimate from the video",
         DEFAULT: "a narrow band around the value in the object's file, where neither gives one"}
#: The catalog's licenses (``website_library_catalog.ALLOWED_LICENSES``) that allow adaptations. A license added to the
#: catalog later varies nothing until it is named here.
ADAPTATION_ALLOWED = frozenset({"CC-BY-4.0", "CC0-1.0"})
LICENSE_STOP = "library_license_does_not_allow_changes"
CREDIT_STOP = "library_credit_unreadable"
MAX_COLOUR_VALUES = 64
#: The ops a size variant appends to the part's own (``xformOp:translate:<PIVOT>``, ``xformOp:scale:<SCALE_OP>``).
PIVOT, SCALE_OP = "variationPivot", "variation"

HEADING = "Simulated variations"
NOT_MEASUREMENTS = ("These are simulated variations for testing in your own tools, not measurements of the site's "
                    "objects.")
LAYER_LINE = ("variations.usda: scene.usda with one variant set, simulated_variation, on each object below: "
              "nominal (the object as in scene.usda) and sample_01 to sample_{last:02d}. Open it instead of scene.usda "
              "and pick a variant to try the object at another size, mass, friction or look; scene.usda and the parts' "
              "files are unchanged.")
SIDECAR_LINE = "variations.json: each object's ranges, what each range rests on (basis) and every sample's values."
BASIS_LINE = ("Each range rests on the site's own answer (site_answer), else Blueprint's estimate from the video "
              "(video_estimate), else a narrow band around the value in the object's file (default). Nothing reads "
              "friction, colour or roughness from the video, so theirs are default.")
SIZE_LINE = ("A size variant scales the object the same on every side about the bottom centre of its box, so it stays "
             "on the surface it starts on.")
RULE_LINE = "success_rule.json is the same for every variant: it was set for the object as in scene.usda."
OBJECT_LINE = "/World/{part}: {quantities}."
NOT_VARIED_LINE = "/World/{part} is not varied: {why}."
LIBRARY_LINE = ("The variants change the {quantities} of the library model credited under Library models; its own "
                "file is unchanged.")
NONE_LINE = "This package has no simulated variations of the job's object: {why}."
WHY = {LICENSE_STOP: "the license of the library model it is built from does not allow changes",
       CREDIT_STOP: "the credit of the library model it is built from cannot be read",
       "nothing_to_vary": "its file holds nothing Blueprint varies (see variations.json)"}
WORDS = {"scale": "size", "mass_kg": "mass", "static_friction": "static friction",
         "dynamic_friction": "dynamic friction", "display_color_factor": "colour", "roughness_offset": "roughness"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
        else None


def _pair(value: Any) -> list[float] | None:
    """Two finite numbers, the first above zero and at most the second; else None."""
    if isinstance(value, Mapping):
        value = [value.get("low"), value.get("high")]
    numbers = [_number(item) for item in value] if isinstance(value, (list, tuple)) and len(value) == 2 else []
    if len(numbers) != 2 or None in numbers or not 0 < numbers[0] <= numbers[1]:
        return None
    return [float(numbers[0]), float(numbers[1])]


def _row(basis: str, came_from: str, value: Any, key: str = "range") -> dict[str, Any] | None:
    found = _pair(value)
    return {key: found, "basis": basis, "from": came_from} if found is not None else None


def _plan_row(capture_root: Path, target: Any) -> Mapping[str, Any]:
    """The job's sealed build plan's row for the object the preparation authored (its ``source_object_identity``)."""
    from .website_build_plan_notes import NOT_IN_VIDEO, sealed_decision
    if not isinstance(target, str) or target == NOT_IN_VIDEO:
        return {}
    plan = _mapping(_mapping(_mapping(sealed_decision(capture_root)).get("build_plan")).get("plan"))
    return next((row for row in plan.get("objects") or [] if isinstance(row, Mapping)
                 and row.get("target_id") == target), {})


def _planned(row: Mapping[str, Any], key: str, said: bool) -> dict[str, Any] | None:
    """The build plan's size (its largest side) or mass, quoted from the site's words (``said``) or estimated."""
    value = _mapping(row.get(key))
    if value.get("basis") != ("site_words" if said else "estimated"):
        return None
    return _row(SITE_ANSWER if said else VIDEO_ESTIMATE, "build_plan_site_words" if said else "build_plan_estimate",
                value, key="largest_side_m" if key == "size_m" else "range")


def evidence(preparation: Mapping[str, Any], capture_root: Path) -> dict[str, Any]:
    """What the half's own ``preparation`` and the job's sealed build plan say of the preparation's object, for its
    variations: ``{"variation_evidence": {EVIDENCE_PART: {"scale" | "mass_kg": row}}}``, each row the first that gives
    one (module docstring), numbers and fixed words only; empty when none does."""
    inputs = _mapping(preparation.get("authoring_inputs"))
    configuration = _mapping(inputs.get("configuration"))
    stated = owner_stated_weight_kg(_mapping(_mapping(configuration.get("construction_constraints"))
                                             .get("operator_task_details")).get("item_weight"))
    row = _plan_row(Path(capture_root), configuration.get("source_object_identity"))
    weight, physics = _mapping(preparation.get("weight_estimate")), _mapping(preparation.get("physics"))
    error = _number(_mapping(inputs.get("metric_envelope")).get("maximum_dimension_relative_error"))
    band = OWNER_WEIGHT_RELATIVE_BAND
    mass = [_row(SITE_ANSWER, "site_answer_item_weight", [stated * (1 - band), stated * (1 + band)] if stated else None),
            _planned(row, "mass_kg", True), _planned(row, "mass_kg", False),
            _row(VIDEO_ESTIMATE, "video_size_and_material", weight.get("range_kg")
                 if weight.get("status") == "estimated" and weight.get("basis") == PHYSICS else None),
            _row(VIDEO_ESTIMATE, "video_size_prior", _mapping(physics.get("bounds")).get("mass_kg")
                 if physics.get("basis") == "estimated" else None)]
    size = [_planned(row, "size_m", True), _planned(row, "size_m", False),
            _row(VIDEO_ESTIMATE, "video_scale_band", [1 - error, 1 / (1 - error)]
                 if error is not None and 0 <= error < 1 and inputs.get("dimension_authority") == "estimated" else None)]
    found = {key: next(item for item in rows if item is not None)
             for key, rows in (("scale", size), ("mass_kg", mass)) if any(item is not None for item in rows)}
    return {"variation_evidence": {EVIDENCE_PART: found}} if found else {}


def rights_stop(source: Mapping[str, Any]) -> str | None:
    """Why no object of this package may be varied: a library model it ships under a license that does not allow
    adaptations, or a library credit that cannot be read; None otherwise."""
    from .website_library_catalog import ALLOWED_LICENSES
    if source.get("library_refusal"):
        return CREDIT_STOP
    for row in library_notes.shipped(source):
        spdx = next((key for key, value in ALLOWED_LICENSES.items()
                     if value["name"] == row.get("license") and value["url"] == row.get("license_url")), None)
        if spdx not in ADAPTATION_ALLOWED:
            return LICENSE_STOP
    return None


def _authored(prim: Any, name: str) -> bool:
    attribute = prim.GetAttribute(name)
    return bool(attribute) and attribute.HasAuthoredValue()


def _inspect(stage: Any, part: str, axis: str, plan_scale: Sequence[float] | None) -> dict[str, Any]:
    """What the composed part ``/World/<part>`` holds that a sample can override: each quantity's targets (paths
    relative to the part) and nominal, or why it cannot be varied."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    root, asset = stage.GetPrimAtPath(f"/World/{part}"), stage.GetPrimAtPath(f"/World/{part}/Asset")
    if not root.IsValid() or not asset.IsValid():
        return dict.fromkeys(QUANTITIES, "part_not_composed")
    prims = list(Usd.PrimRange(asset))

    def relative(prim: Any) -> str:
        return prim.GetPath().MakeRelativePath(root.GetPath()).pathString

    joints = any(prim.IsA(UsdPhysics.Joint) for prim in prims)
    found: dict[str, Any] = {}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                       UsdGeom.Tokens.proxy])
    world, local = (cache.ComputeWorldBound(asset).ComputeAlignedRange(),
                    cache.ComputeRelativeBound(asset, root).ComputeAlignedRange())
    # Its sides as scene.usda sizes it: its box in the part's frame, times the plan's scale (before its turn).
    sides = [] if local.IsEmpty() else [abs(local.GetSize()[i] * (plan_scale[i] if plan_scale else 1.0))
                                        for i in range(3)]
    if joints:
        found["scale"] = "has_joints"
    elif not asset.IsA(UsdGeom.Xformable):
        found["scale"] = "part_not_transformable"
    elif world.IsEmpty() or not sides or max(sides) <= 0:
        found["scale"] = "no_bounds"
    else:
        up = "XYZ".index(axis)
        point = [(world.GetMin()[i] + world.GetMax()[i]) / 2 if i != up else world.GetMin()[i] for i in range(3)]
        xformable = UsdGeom.Xformable(asset)
        pivot = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse().Transform(Gf.Vec3d(*point))
        found["scale"] = {"pivot": [round(float(value), 6) for value in pivot], "largest_side_m": max(sides),
                          "order": [str(op) for op in xformable.GetXformOpOrderAttr().Get() or []],
                          "bottom": float(world.GetMin()[up]), "prims": ["Asset"]}
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    massed = [prim for prim in prims if _authored(prim, "physics:mass") or _authored(prim, "physics:density")]
    if joints:
        found["mass_kg"] = "has_joints"
    elif len(bodies) != 1:
        found["mass_kg"] = "not_one_rigid_body"
    elif any(prim != bodies[0] for prim in massed):
        found["mass_kg"] = "mass_set_on_other_prims"
    else:
        nominal = _number(bodies[0].GetAttribute("physics:mass").Get()) if _authored(bodies[0], "physics:mass") else None
        found["mass_kg"] = {"prims": [relative(bodies[0])], "nominal": nominal if nominal and nominal > 0 else None,
                            "mass_api": bodies[0].HasAPI(UsdPhysics.MassAPI)}
    materials = [prim for prim in prims if prim.HasAPI(UsdPhysics.MaterialAPI)]
    for key, name in (("static_friction", "physics:staticFriction"), ("dynamic_friction", "physics:dynamicFriction")):
        values = {_number(prim.GetAttribute(name).Get()) for prim in materials if _authored(prim, name)}
        found[key] = ("no_physics_material" if not materials else "no_friction_in_file" if not values
                      else "materials_differ" if len(values) != 1 or None in values
                      else {"prims": [relative(prim) for prim in materials], "nominal": values.pop()})
    colours: dict[str, list[list[float]]] = {}
    per_point = False
    for prim in prims:
        attribute = UsdGeom.Gprim(prim).GetDisplayColorAttr() if prim.IsA(UsdGeom.Gprim) else None
        value = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
        if value is not None and 0 < len(value) <= MAX_COLOUR_VALUES:
            colours[relative(prim)] = [[float(channel) for channel in colour] for colour in value]
        per_point = per_point or (value is not None and len(value) > MAX_COLOUR_VALUES)
    found["display_color_factor"] = ({"prims": sorted(colours), "colours": colours} if colours
                                     else "display_color_per_point" if per_point else "no_display_color")
    rough: dict[str, float] = {}
    for prim in prims:
        shader = UsdShade.Shader(prim) if prim.IsA(UsdShade.Shader) else None
        put = shader.GetInput("roughness") if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface" else None
        value = _number(put.GetAttr().Get()) if put and put.GetAttr().HasAuthoredValue() \
            and not put.HasConnectedSource() else None
        if value is not None:
            rough[relative(prim)] = value
    found["roughness_offset"] = {"prims": sorted(rough), "values": rough} if rough else "no_roughness_value"
    return found


def _ranges(target: Mapping[str, Any], told: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Each quantity's range (``low``, ``high``, ``basis``, ``from``, ``unit`` and where it applies), or why not."""
    ranges: dict[str, Any] = {}
    not_varied: dict[str, str] = {}
    for key in QUANTITIES:
        found = target[key]
        if isinstance(found, str):
            not_varied[key] = found
            continue
        nominal, row = found.get("nominal"), told.get(key)
        if key == "scale":
            nominal = round(found["largest_side_m"], 6)
            if row is not None:
                pair = row.get("range") or [value / found["largest_side_m"] for value in row["largest_side_m"]]
                row = {**row, "range": pair}
        elif row is None and key in ("mass_kg", "static_friction", "dynamic_friction") and nominal is not None:
            row = {"range": [nominal * value for value in DEFAULT_BANDS[key]], "basis": DEFAULT,
                   "from": "object_file_value"}
        elif row is None and key in ("mass_kg", "static_friction", "dynamic_friction"):
            not_varied[key] = "no_mass_in_file" if key == "mass_kg" else "no_friction_in_file"
            continue
        row = row or {"range": list(DEFAULT_BANDS[key]), "basis": DEFAULT,
                      "from": "scene_usda_size" if key == "scale" else "object_file_value"}
        low, high = (round(float(value), 6) for value in row["range"])
        ranges[key] = {"low": low, "high": high, "basis": row["basis"], "from": row["from"], "unit": UNITS[key],
                       **({"nominal_largest_side_m": nominal, "about": "bottom_centre_of_box"} if key == "scale"
                          else {"nominal": round(nominal, 6)} if nominal is not None else {})}
    return ranges, not_varied


def _unit(seed: str, index: int, key: str) -> float:
    digest = hashlib.sha256(f"{GENERATOR}|{seed}|{index}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _samples(ranges: Mapping[str, Mapping[str, Any]], seed: str) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for index in range(1, SAMPLES + 1):
        values = {key: round(row["low"] + (row["high"] - row["low"]) * _unit(seed, index, key), 6)
                  for key, row in ranges.items()}
        if "static_friction" in values and "dynamic_friction" in values:
            values["dynamic_friction"] = min(values["dynamic_friction"],
                                             max(values["static_friction"], ranges["dynamic_friction"]["low"]))
        rows.append((f"sample_{index:02d}", values))
    return rows


def _n(value: float) -> str:
    return repr(round(float(value), 6) + 0.0)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _opinions(target: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, dict[str, list[str]]]:
    """Each prim's lines for one sample, by its path relative to the part: ``meta`` and ``attributes``."""
    found: dict[str, dict[str, list[str]]] = {}

    def at(path: str) -> dict[str, list[str]]:
        return found.setdefault(path, {"meta": [], "attributes": []})
    if "scale" in values:
        scale, size = target["scale"], _n(values["scale"])
        order = ", ".join(f'"{op}"' for op in [*scale["order"], f"xformOp:translate:{PIVOT}", f"xformOp:scale:{SCALE_OP}",
                                                f"!invert!xformOp:translate:{PIVOT}"])
        at("Asset")["attributes"] += [f"uniform token[] xformOpOrder = [{order}]",
                                      f"double3 xformOp:scale:{SCALE_OP} = ({size}, {size}, {size})",
                                      f"double3 xformOp:translate:{PIVOT} = ({', '.join(map(_n, scale['pivot']))})"]
    if "mass_kg" in values:
        body = at(target["mass_kg"]["prims"][0])
        body["meta"] += [] if target["mass_kg"]["mass_api"] else ['prepend apiSchemas = ["PhysicsMassAPI"]']
        body["attributes"].append(f"float physics:mass = {_n(values['mass_kg'])}")
    for key, name in (("static_friction", "physics:staticFriction"), ("dynamic_friction", "physics:dynamicFriction")):
        for path in target[key]["prims"] if key in values else []:
            at(path)["attributes"].append(f"float {name} = {_n(values[key])}")
    for path in target["display_color_factor"]["prims"] if "display_color_factor" in values else []:
        colours = ", ".join("(" + ", ".join(_n(_clip(channel * values["display_color_factor"])) for channel in colour)
                            + ")" for colour in target["display_color_factor"]["colours"][path])
        at(path)["attributes"].append(f"color3f[] primvars:displayColor = [{colours}]")
    for path in target["roughness_offset"]["prims"] if "roughness_offset" in values else []:
        value = _clip(target["roughness_offset"]["values"][path] + values["roughness_offset"])
        at(path)["attributes"].append(f"float inputs:roughness = {_n(value)}")
    return found


def _overs(opinions: Mapping[str, Mapping[str, list[str]]], indent: int) -> list[str]:
    """Nested ``over`` blocks holding every prim's lines, in path order."""
    tree: dict[str, Any] = {}
    for path in sorted(opinions):
        node = tree
        for name in path.split("/"):
            node = node.setdefault(name, {})
        node[""] = opinions[path]

    def block(name: str, node: Mapping[str, Any], depth: int) -> list[str]:
        pad, own = " " * depth, node.get("", {"meta": [], "attributes": []})
        head = [f'{pad}over "{name}" (', *(f"{pad}    {line}" for line in own["meta"]), f"{pad})"] if own["meta"] \
            else [f'{pad}over "{name}"']
        body = [f"{pad}    {line}" for line in own["attributes"]]
        for child in sorted(key for key in node if key):
            body += block(child, node[child], depth + 4)
        return [*head, f"{pad}{{", *body, f"{pad}}}"]
    return [line for name in sorted(tree) for line in block(name, tree[name], indent)]


def _layer(axis: str, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["#usda 1.0", "(", '    defaultPrim = "World"',
             '    doc = "Simulated variations of the job\'s objects for testing in your own tools; not measurements. '
             'See README.txt and variations.json."', "    metersPerUnit = 1", "    subLayers = [", "        @./scene.usda@",
             "    ]", f'    upAxis = "{axis}"', ")", "", 'over "World"', "{"]
    for row in rows:
        lines += [f'    over "{row["part"]}" (', f'        prepend variantSets = "{VARIANT_SET}"', "        variants = {",
                  f'            string {VARIANT_SET} = "{NOMINAL}"', "        }", "    )", "    {",
                  f'        variantSet "{VARIANT_SET}" = {{', f'            "{NOMINAL}" {{', "            }"]
        for name, values in row["samples"]:
            lines += [f'            "{name}" {{', *_overs(_opinions(row["target"], values), 16), "            }"]
        lines += ["        }", "    }"]
    return "\n".join([*lines, "}", ""])


def _check(staging: Path, rows: Sequence[Mapping[str, Any]], axis: str) -> str | None:
    """The layer opens on its own sublayer only, every variant composes its values, and a size variant leaves the
    object's bottom where it was; else why not."""
    from pxr import Sdf, Usd, UsdGeom
    path = staging / LAYER_FILE
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None or list(layer.subLayerPaths) != ["./scene.usda"] or set(layer.GetCompositionAssetDependencies()) \
            != {"./scene.usda"}:
        return "layer_unreadable"
    stage = Usd.Stage.OpenMasked(str(path), Usd.StagePopulationMask([f"/World/{row['part']}" for row in rows]))
    stage.SetEditTarget(stage.GetSessionLayer())  # selections are read here, never written to the shipped file
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                       UsdGeom.Tokens.proxy])
    names = {"mass_kg": "physics:mass", "static_friction": "physics:staticFriction",
             "dynamic_friction": "physics:dynamicFriction"}
    for row in rows:
        part = stage.GetPrimAtPath(f"/World/{row['part']}")
        chosen = part.GetVariantSets().GetVariantSet(VARIANT_SET)
        if chosen.GetVariantNames() != sorted([NOMINAL, *(name for name, _ in row["samples"])]):
            return "variants_missing"
        for name, values in row["samples"]:
            chosen.SetVariantSelection(name)
            for key, attribute in names.items():
                prim = stage.GetPrimAtPath(f"/World/{row['part']}/{row['target'][key]['prims'][0]}") \
                    if key in values else None
                if prim is not None and not math.isclose(prim.GetAttribute(attribute).Get(), values[key],
                                                         rel_tol=1e-5):
                    return "variant_value_not_composed"
            if "scale" in values:
                cache.Clear()
                bottom = cache.ComputeWorldBound(part.GetChild("Asset")).ComputeAlignedRange().GetMin()["XYZ".index(axis)]
                if not math.isclose(bottom, row["target"]["scale"]["bottom"], abs_tol=1e-5):
                    return "size_variant_moved_the_object"
        chosen.SetVariantSelection(NOMINAL)
    return None


def write(source: Mapping[str, Any], staging: Path, placed: Mapping[str, str]) -> dict[str, Any]:
    """Writes ``LAYER_FILE`` and ``SIDECAR_FILE`` into the package's ``staging`` (after scene.usda) for every object
    part it places, and returns ``{"variations": summary}`` for the README (``readme_lines``); empty for a package
    without the job's object. Never raises for a fault of its own: the package ships without them, logged."""
    files = source.get("files") or {}
    # The job's object and each further object of it (``job_object``: default bands, ``EVIDENCE_PART`` is the first's).
    parts = [prim for prim in placed if _mapping(files.get(prim)).get("role") in ("task_object", "job_object")]
    if not parts or source.get("room") is not None:
        return {}
    stop = rights_stop(source)
    if stop is not None:
        return {"variations": {"objects": [{"part": f"/World/{part}", "varied": False, "reason": stop}
                                           for part in parts]}}
    staging = Path(staging)
    try:
        from pxr import Usd
        stage = Usd.Stage.OpenMasked(str(staging / "scene.usda"),
                                     Usd.StagePopulationMask([f"/World/{part}" for part in parts]))
        told = _mapping(source.get("variation_evidence"))
        objects, rows = [], []
        for part in parts:
            target = _inspect(stage, part, source["axis"], files[part]["placement"].get("scale"))
            ranges, not_varied = _ranges(target, _mapping(told.get(part)))
            if not ranges:
                objects.append({"part": f"/World/{part}", "varied": False, "reason": "nothing_to_vary",
                                "not_varied": not_varied})
                continue
            # Seeded only by what the package revision already covers (the part, its file and its evidence), so one
            # revision always holds the same samples (a review of PR #3752: the plan's digest is not in it).
            evidence = json.dumps(_mapping(told.get(part)), sort_keys=True, separators=(",", ":"))
            samples = _samples(ranges, f"{part}|{files[part]['sha256']}|{evidence}")
            rows.append({"part": part, "target": target, "samples": samples})
            objects.append({"part": f"/World/{part}", "varied": True,
                            "ranges": {key: {**row, "prims": [f"/World/{part}/{path}" for path in target[key]["prims"]]}
                                       for key, row in ranges.items()},
                            "not_varied": not_varied,
                            "samples": [{"variant": name, "values": values} for name, values in samples]})
        del stage
        if not rows:
            return {"variations": {"objects": objects}}
        (staging / LAYER_FILE).parent.mkdir(parents=True, exist_ok=True)
        (staging / LAYER_FILE).write_text(_layer(source["axis"], rows), encoding="utf-8")
        (staging / SIDECAR_FILE).write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "what_these_are": NOT_MEASUREMENTS, "layer": LAYER_FILE,
            "variant_set": VARIANT_SET, "variants": [NOMINAL, *(name for name, _ in rows[0]["samples"])],
            "bases": BASES, "objects": objects}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        why = _check(staging, rows, source["axis"])
    except Exception as error:  # noqa: BLE001 - a fault here never holds the package
        why = "variations_failed:" + type(error).__name__
    if why is not None:
        logger.warning("website_company_simulation_variations_not_written reason=%s", why)
        for name in (LAYER_FILE, SIDECAR_FILE):
            (staging / name).unlink(missing_ok=True)
        return {"variations": {"not_written": why}}
    return {"variations": {"objects": objects, "rule": source.get("rule") is not None,
                           "library": bool(library_notes.shipped(source))}}


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent="- ", subsequent_indent="  ", break_long_words=False,
                         break_on_hyphens=False)


def _quantities(ranges: Mapping[str, Mapping[str, Any]]) -> str:
    """``size (video_estimate), mass (site_answer), friction (default)``: each quantity varied, with its basis."""
    shown = {WORDS[key]: row["basis"] for key, row in ranges.items()}
    if shown.get("static friction") is not None and shown.get("static friction") == shown.get("dynamic friction"):
        shown = {("friction" if word == "static friction" else word): basis for word, basis in shown.items()
                 if word != "dynamic friction"}
    words = [f"{word} ({basis})" for word, basis in shown.items()]
    return ", ".join(words[:-1]) + (" and " if len(words) > 1 else "") + words[-1]


def readme_lines(source: Mapping[str, Any], *, width: int = 100) -> list[str]:
    """The README's "Simulated variations" section, or nothing for a package with no object (or whose variations were
    not written)."""
    summary = _mapping(source.get("variations"))
    objects = [row for row in summary.get("objects") or [] if isinstance(row, Mapping)]
    if not objects:
        return []
    lines = ["", HEADING, ""]
    varied = [row for row in objects if row.get("varied")]
    if not varied:
        return [*lines, *_wrap(NONE_LINE.format(why=WHY.get(str(objects[0].get("reason")), WHY["nothing_to_vary"])),
                               width)]
    notes = [LAYER_LINE.format(last=SAMPLES), SIDECAR_LINE, NOT_MEASUREMENTS, BASIS_LINE,
             *([SIZE_LINE] if any("scale" in row["ranges"] for row in varied) else []),
             *([RULE_LINE] if summary.get("rule") else []),
             *(OBJECT_LINE.format(part=row["part"].removeprefix("/World/"), quantities=_quantities(row["ranges"]))
               if row.get("varied") else
               NOT_VARIED_LINE.format(part=row["part"].removeprefix("/World/"), why=WHY["nothing_to_vary"])
               for row in objects)]
    if summary.get("library"):
        changed = list(dict.fromkeys(WORDS[key].replace("static ", "").replace("dynamic ", "")
                                     for row in varied for key in row["ranges"]))
        notes.append(LIBRARY_LINE.format(quantities=", ".join(changed[:-1]) + (" and " if len(changed) > 1 else "")
                                         + changed[-1]))
    for note in notes:
        lines += _wrap(note, width)
    return lines


__all__ = ["ADAPTATION_ALLOWED", "BASES", "DEFAULT", "DEFAULT_BANDS", "EVIDENCE_PART", "GENERATOR", "HEADING",
           "LAYER_FILE", "NOMINAL", "NOT_MEASUREMENTS", "SAMPLES", "SIDECAR_FILE", "SITE_ANSWER", "VARIANT_SET",
           "VIDEO_ESTIMATE", "evidence", "readme_lines", "rights_stop", "write"]
