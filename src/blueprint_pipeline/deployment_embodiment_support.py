"""Prepare shortlisted robot/policy integrations without granting execution.

Beta job-to-evidence preparation / ADP-011 (partner day 7) and ADP-050 (day 28).
The catalog records exact bodies and evidence for interface routes. A shortlist
can be exported by the existing Website setup and replayed here. It is never a
policy pair, a robot model, a readiness receipt, or paid-run authority.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

CATALOG_SCHEMA = "deployment_embodiment_support.v1"
SHORTLIST_SCHEMA = "deployment_embodiment_shortlist.v1"
PLAN_SCHEMA = "deployment_embodiment_integration_plan.v1"
CATALOG_PATH = Path(__file__).parent / "_catalog_data" / f"{CATALOG_SCHEMA}.json"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_KINDS = {"hosted_api", "checkpoint", "customer_controller", "planning_endpoint", "skill_trace", "early_access"}
_EVIDENCE = {"publisher_demonstrated", "interface_candidate", "customer_supplied"}


class DeploymentSupportError(ValueError):
    """A catalog, shortlist, or exact-robot handoff does not satisfy its contract."""


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise DeploymentSupportError(code)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4000


def _https(value: Any) -> bool:
    if not _text(value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme == "https" and bool(parsed.hostname) and not (
            parsed.username or parsed.password or parsed.query or parsed.fragment
        )
    except ValueError:
        return False


def _ids(value: Any, *, nonempty: bool = True) -> bool:
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(isinstance(item, str) and _ID.fullmatch(item) for item in value)
            and len(set(value)) == len(value))


def validate_support_catalog(value: Any) -> dict[str, Any]:
    """Validate source metadata, without treating a URL as a verified asset."""

    _require(isinstance(value, Mapping), "support_catalog_invalid")
    catalog = deepcopy(dict(value))
    _require(set(catalog) == {"schema_version", "claim_ceiling", "policy_routes", "embodiments"}
             and catalog["schema_version"] == CATALOG_SCHEMA
             and catalog["claim_ceiling"] == "planning_only", "support_catalog_invalid")
    routes, bodies = catalog["policy_routes"], catalog["embodiments"]
    _require(isinstance(routes, list) and bool(routes) and isinstance(bodies, list) and bool(bodies),
             "support_catalog_entries_invalid")
    route_ids: set[str] = set()
    for route in routes:
        _require(isinstance(route, dict) and set(route) == {
            "route_id", "display_name", "kind", "model_id", "observation_schema_id",
            "action_schema_id", "source_url", "required_artifacts",
        }, "support_catalog_route_invalid")
        _require(_ids([route["route_id"]]) and route["route_id"] not in route_ids
                 and _text(route["display_name"]) and isinstance(route["kind"], str) and route["kind"] in _KINDS
                 and _https(route["source_url"]) and _ids(route["required_artifacts"])
                 and all(route[key] is None or _text(route[key])
                         for key in ("model_id", "observation_schema_id", "action_schema_id")),
                 "support_catalog_route_invalid")
        route_ids.add(route["route_id"])
    body_ids: set[str] = set()
    for body in bodies:
        _require(isinstance(body, dict) and set(body) == {
            "embodiment_id", "display_name", "manufacturer", "arm_count", "arm_dof", "base_mode",
            "tool_requirement", "related_teams", "asset_sources", "policy_routes", "required_artifacts",
        }, "support_catalog_embodiment_invalid")
        _require(_ids([body["embodiment_id"]]) and body["embodiment_id"] not in body_ids
                 and all(_text(body[key]) for key in ("display_name", "manufacturer", "tool_requirement"))
                 and type(body["arm_count"]) is int and body["arm_count"] in (1, 2)
                 and body["base_mode"] == "fixed"
                 and _ids(body["required_artifacts"]), "support_catalog_embodiment_invalid")
        dof = body["arm_dof"]
        _require(dof is None or (isinstance(dof, list) and len(dof) == body["arm_count"]
                 and all(type(item) is int and 1 <= item <= 12 for item in dof)),
                 "support_catalog_arm_dof_invalid")
        _require(isinstance(body["related_teams"], list)
                 and all(_text(item) for item in body["related_teams"]), "support_catalog_team_names_invalid")
        sources = body["asset_sources"]
        _require(isinstance(sources, list) and bool(sources), "support_catalog_asset_sources_invalid")
        source_ids: set[str] = set()
        for source in sources:
            _require(isinstance(source, dict) and set(source) == {
                "source_id", "format", "url", "relative_path", "revision", "license_note", "preparation",
            }, "support_catalog_asset_source_invalid")
            _require(_ids([source["source_id"]]) and source["source_id"] not in source_ids
                     and isinstance(source["format"], str)
                     and source["format"] in {"usd", "urdf", "xacro", "mjcf", "documentation"}
                     and _https(source["url"]) and _text(source["license_note"])
                     and _text(source["preparation"])
                     and (source["revision"] is None or _text(source["revision"])),
                     "support_catalog_asset_source_invalid")
            path = source["relative_path"]
            _require(path is None or (_text(path) and not Path(path).is_absolute()
                                     and ".." not in Path(path).parts), "support_catalog_asset_path_invalid")
            source_ids.add(source["source_id"])
        links = body["policy_routes"]
        _require(isinstance(links, list) and bool(links), "support_catalog_route_links_invalid")
        linked: set[str] = set()
        for link in links:
            _require(isinstance(link, dict) and set(link) == {"route_id", "evidence", "source_url"}
                     and isinstance(link["route_id"], str) and link["route_id"] in route_ids
                     and link["route_id"] not in linked and isinstance(link["evidence"], str)
                     and link["evidence"] in _EVIDENCE and _https(link["source_url"]),
                     "support_catalog_route_link_invalid")
            linked.add(link["route_id"])
        body_ids.add(body["embodiment_id"])
    return catalog


def support_catalog() -> dict[str, Any]:
    """A fresh catalog view, packaged with Pipeline and mirrored by the Website."""

    catalog = validate_support_catalog(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
    catalog["catalog_digest"] = cross_runtime_canonical_digest(catalog, digest_field="catalog_digest")
    return catalog


def make_shortlist(
    selections: Sequence[Mapping[str, Any]], *, source_launch_id: str | None = None,
) -> dict[str, Any]:
    """Build a canonical planning handoff; no new runtime identity is invented."""

    catalog = support_catalog()
    _require(isinstance(selections, (list, tuple)) and bool(selections), "support_shortlist_selections_invalid")
    selected: dict[str, dict[str, Any]] = {}
    order = [route["route_id"] for route in catalog["policy_routes"]]
    bodies = {body["embodiment_id"]: body for body in catalog["embodiments"]}
    for selection in selections:
        _require(isinstance(selection, Mapping)
                 and set(selection) == {"embodiment_id", "policy_route_ids"}, "support_shortlist_selection_invalid")
        identity, route_ids = selection["embodiment_id"], selection["policy_route_ids"]
        _require(isinstance(identity, str) and identity in bodies and identity not in selected,
                 "support_shortlist_embodiment_invalid")
        allowed = {route["route_id"] for route in bodies[identity]["policy_routes"]}
        _require(_ids(route_ids) and set(route_ids) <= allowed, "support_shortlist_policy_route_invalid")
        selected[identity] = {"embodiment_id": identity, "policy_route_ids": sorted(route_ids, key=order.index)}
    _require(source_launch_id is None or (isinstance(source_launch_id, str) and _ID.fullmatch(source_launch_id)),
             "support_shortlist_source_invalid")
    value = {"schema_version": SHORTLIST_SCHEMA, "claim_ceiling": "planning_only",
             "catalog_digest": catalog["catalog_digest"], "source_launch_id": source_launch_id,
             "selections": [selected[body["embodiment_id"]] for body in catalog["embodiments"]
                            if body["embodiment_id"] in selected]}
    value["shortlist_digest"] = cross_runtime_canonical_digest(value, digest_field="shortlist_digest")
    return value


def validate_shortlist(value: Any) -> dict[str, Any]:
    """Reject drift, variant substitution, duplicate entries and incompatible routes."""

    _require(isinstance(value, Mapping) and set(value) == {
        "schema_version", "claim_ceiling", "catalog_digest", "source_launch_id", "selections", "shortlist_digest",
    }, "support_shortlist_invalid")
    _require(value["schema_version"] == SHORTLIST_SCHEMA and value["claim_ceiling"] == "planning_only",
             "support_shortlist_invalid")
    expected = make_shortlist(value["selections"], source_launch_id=value["source_launch_id"])
    _require(value["catalog_digest"] == expected["catalog_digest"], "support_shortlist_catalog_changed")
    _require(dict(value) == expected, "support_shortlist_digest_or_order_invalid")
    return expected


def prepare_shortlist(value: Any) -> dict[str, Any]:
    """Resolve every selected integration into exact work and existing admission seams."""

    request = validate_shortlist(value)
    catalog = support_catalog()
    bodies = {body["embodiment_id"]: body for body in catalog["embodiments"]}
    routes = {route["route_id"]: route for route in catalog["policy_routes"]}
    integrations = []
    for selected in request["selections"]:
        body = bodies[selected["embodiment_id"]]
        selected_routes = []
        for route_id in selected["policy_route_ids"]:
            route = deepcopy(routes[route_id])
            evidence = next(link for link in body["policy_routes"] if link["route_id"] == route_id)
            route["compatibility_evidence"] = evidence["evidence"]
            route["compatibility_source_url"] = evidence["source_url"]
            route["blockers"] = list(route["required_artifacts"])
            selected_routes.append(route)
        blockers = list(body["required_artifacts"])
        integrations.append({
            "embodiment_id": body["embodiment_id"], "display_name": body["display_name"],
            "arm_count": body["arm_count"], "asset_sources": deepcopy(body["asset_sources"]),
            "policy_routes": selected_routes, "status": "requires_setup", "blockers": sorted(set(blockers)),
            "native_admission": {"manifest_schema": ("blueprint.robot_embodiment.v2" if body["arm_count"] > 1
                                                     else "blueprint.robot_embodiment.v1"),
                                 "module": "blueprint_pipeline.team_candidate_quick10_admission",
                                 "exact_model_and_scene_checks_required": True},
        })
    plan = {"schema_version": PLAN_SCHEMA, "claim_ceiling": "planning_only",
            "catalog_digest": request["catalog_digest"], "shortlist_digest": request["shortlist_digest"],
            "source_launch_id": request["source_launch_id"], "integrations": integrations,
            "execution_authorized": False, "provider_mutation_performed": False}
    plan["plan_digest"] = cross_runtime_canonical_digest(plan, digest_field="plan_digest")
    return plan


def prepare_native_binding(
    shortlist: Any, *, embodiment_id: str, manifest: Mapping[str, Any], model_path: str | Path,
    scene_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Check a shortlisted exact model through the existing native admission.

    Bimanual models require v2 independent tool channels; the native runtime
    executes and reads each tool separately. This validates only robot binding. Policy, rights, spend, media and runtime
    admission remain required at the existing Quick-10 entry point.
    """

    from .team_embodiment import (
        read_model_structure,
        team_embodiment_binding,
        validate_embodiment_manifest,
        verify_model_file,
    )

    request = validate_shortlist(shortlist)
    _require(embodiment_id in {row["embodiment_id"] for row in request["selections"]},
             "support_native_embodiment_not_shortlisted")
    body = next(row for row in support_catalog()["embodiments"] if row["embodiment_id"] == embodiment_id)
    _require(isinstance(scene_plan, Mapping)
             and isinstance(scene_plan.get("task_spec"), Mapping)
             and scene_plan["task_spec"].get("task_kind") == "rigid_pick_place",
             "team_embodiment_task_kind_unsupported")
    if "plan_digest" in scene_plan:
        _require(scene_plan["plan_digest"] == canonical_digest(dict(scene_plan), digest_field="plan_digest"),
                 "support_native_scene_plan_digest_mismatch")
    validated = validate_embodiment_manifest(manifest)
    _require(validated["embodiment_id"] == embodiment_id, "support_native_embodiment_identity_mismatch")
    expected_form = "bimanual_arm" if body["arm_count"] > 1 else "fixed_base_arm"
    _require(validated["form"] == expected_form and validated["base"]["mode"] == "fixed",
             "support_native_form_mismatch")
    if body["arm_count"] > 1:
        _require(validated["schema_version"] == "blueprint.robot_embodiment.v2"
                 and len(validated["grippers"]) == body["arm_count"],
                 "independent_bimanual_tool_channels_required")
    if body["arm_dof"] is not None:
        _require(len(validated["joints"]) == sum(body["arm_dof"]), "support_native_arm_joint_count_mismatch")
    model_record = verify_model_file(validated, model_path)
    binding = team_embodiment_binding(validated, model_record=model_record,
                                     structure=read_model_structure(model_path), scene_plan=scene_plan)
    result = {"schema_version": "deployment_embodiment_native_preflight.v1", "claim_ceiling": "planning_only",
              "embodiment_id": embodiment_id, "shortlist_digest": request["shortlist_digest"],
              "scene_plan_digest": canonical_digest(dict(scene_plan)),
              "task_spec_digest": canonical_digest(dict(scene_plan["task_spec"])),
              "robot_binding": binding, "execution_authorized": False,
              "model_file_digest_verified": True, "catalog_asset_identity_verified": False,
              "policy_handoff": {"robot_embodiment": validated,
                                 "policy_interface_id": "blueprint_policy_https.v1",
                                 "execution_profile": "team_evaluation_scored_v1"},
              "remaining_admission": sorted(set([
                  *body["required_artifacts"], "exact_catalog_asset_provenance_and_license",
                  "exact_policy_interface", "rights_and_provider_terms", "runtime_readiness_receipt",
                  "controlled_observations_and_media", "spend_watchdog_and_teardown",
              ]))}
    result["preflight_digest"] = cross_runtime_canonical_digest(result, digest_field="preflight_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--catalog", action="store_true")
    mode.add_argument("--shortlist", type=Path, help="Planning handoff exported from the existing Website setup")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embodiment-id")
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--native-model", type=Path)
    parser.add_argument("--scene-plan", type=Path)
    args = parser.parse_args(argv)
    native = [args.embodiment_id, args.native_manifest, args.native_model, args.scene_plan]
    if any(native) and (args.catalog or not all(native)):
        parser.error("Native preflight requires --shortlist, --embodiment-id, --native-manifest, --native-model and --scene-plan")
    try:
        if args.catalog:
            value = support_catalog()
        else:
            request = json.loads(args.shortlist.read_text(encoding="utf-8"))
            value = (prepare_native_binding(request, embodiment_id=args.embodiment_id,
                     manifest=json.loads(args.native_manifest.read_text(encoding="utf-8")),
                     model_path=args.native_model, scene_plan=json.loads(args.scene_plan.read_text(encoding="utf-8")))
                     if all(native) else prepare_shortlist(request))
        encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Support preparation refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
