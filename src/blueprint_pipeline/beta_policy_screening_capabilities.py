"""Derive beta planning inputs from the prepared scene and operator runtimes.

The default four-hour/$4 candidate ceiling is the existing diagnostic canary
envelope. It is a maximum reservation, not a prediction: the canonical allocator
also accounts for startup, transfer, storage and teardown before provider create.
Unknown hourly prices never produce quotes. The site's $15 coordinator remains
the authority over aggregate reservations.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_UP
from pathlib import Path
import subprocess
import re
from typing import Any

from .beta_policy_screening_launch import EXECUTION_SCHEMA, _bound_path, _read, _record, _write
from .decision_evidence_contracts import canonical_digest

DEFAULT_HARD_TTL_SECONDS = 14_400
DEFAULT_CANDIDATE_CAP_USD = Decimal("4.00")


def _interface_matches(configured: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    """Compare the complete catalog profile with the interface actually served.

    The local DROID interfaces have fixed robot/observation/action contracts.
    A robot with a different camera or control contract needs an admitted
    profile before it can supply a capability, even if its endpoint works.
    """
    from .beta_policy_candidate_catalog import load_catalog
    interface = spec["policy_interface_id"]
    if interface == "humanoidarena_semantic_v3":
        from .beta_policy_screening_g1 import interface_contract, runtime_contract, validate_execution_spec
        validate_execution_spec(spec)
        return (configured.get("interface") == interface_contract(spec["candidate_id"])
                and configured.get("runtime") == runtime_contract())
    from .policy_interface_binding import TEAM_ROBOT_INTERFACES
    if interface in TEAM_ROBOT_INTERFACES:
        from .beta_policy_screening_team import team_interface_matches
        return team_interface_matches(configured, spec)
    if interface == "molmoact2_droid_joint8.v1":
        from .molmoact2_droid_runtime import interface_contract, validate_execution_spec, MINIMUM_GPU_MEMORY_GB
        validate_execution_spec(spec)
        return (configured.get("interface") == interface_contract() and configured.get("runtime") == {
            "backend": "robolab.v1", "runtime_id": "native_task_arena_policy_canary.v1",
            "minimum_gpu_memory_gb": MINIMUM_GPU_MEMORY_GB})
    if interface in {"cosmos_edge_droid.v1", "cosmos_nano_droid.v1"}:
        from .cosmos_droid_bundle import profile
        selected = profile(spec["candidate_id"])
        selected.validate_execution_spec(spec)
        return (configured.get("interface") == selected.interface_contract() and configured.get("runtime") == {
            "backend": "robolab.v1", "runtime_id": "native_task_arena_policy_canary.v1",
            "minimum_gpu_memory_gb": selected.MINIMUM_GPU_MEMORY_GB})
    if interface not in {"openpi_droid.v1", "groot_n17_droid.v1", "openpi_droid_https.v1"}:
        return False
    family = "groot_n17_droid" if interface == "groot_n17_droid.v1" else "pi05_droid"
    if interface != "openpi_droid_https.v1" and spec.get("candidate_id") != family:
        return False  # The existing local provisioner admits exact pinned identities.
    reference = next(row for row in load_catalog(
        Path(__file__).resolve().parents[2] / "config" / "beta_policy_candidates.v1.json")["candidates"]
        if row["candidate_id"] == family)
    expected = dict(reference["interface"])
    expected["policy_interface_id"] = interface
    expected["execution_horizon"] = spec["open_loop_horizon"]
    if family == "pi05_droid":
        expected["action_horizon"] = spec["policy_spec"]["action_chunk_rows"]
    runtime = configured.get("runtime") or {}
    return (configured.get("interface") == expected
            and runtime.get("backend") == "robolab.v1"
            and runtime.get("runtime_id") == "native_task_arena_policy_canary.v1"
            and runtime.get("minimum_gpu_memory_gb") == reference["runtime"]["minimum_gpu_memory_gb"])


def _retain(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        if _read(path) != value:
            raise ValueError("beta_screening_materialized_input_changed")
        return path
    return _write(path, value)


def _native_templates(*, repo: Path, scene_plan: Mapping[str, Any], contract: Mapping[str, Any],
                      commit: str) -> dict[str, tuple[dict[str, Any], Path | None]]:
    from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
    from .task_evaluation_policy_canary_handoff import MANIFESTS
    from .task_evaluation_policy_canary_scene_setup import _candidate_spec

    readiness = _read(repo / MANIFESTS["historical_policy_readiness"])
    if readiness.get("readiness_digest") != canonical_digest(readiness, digest_field="readiness_digest"):
        raise ValueError("beta_screening_policy_readiness_digest_invalid")
    result = {}
    for candidate in readiness.get("candidates", []):
        key = candidate["candidate_id"]
        if key not in ("pi05_droid", "groot_n17_droid"):
            continue
        expected = EXPECTED_CANDIDATES[key]
        checkpoint = candidate.get("checkpoint") or {}
        source = candidate.get("source") or {}
        if (source.get("revision") != expected["source_revision"]
                or source.get("tree") != expected["source_tree"]
                or checkpoint.get("inventory_digest") != expected["checkpoint_inventory_digest"]
                or checkpoint.get("checkpoint_ready") is not True
                or checkpoint.get("missing_secrets_or_gated_access") != []
                or any(candidate.get(field) is not True for field in
                       ("rights_ready", "observation_adapter_ready", "action_adapter_ready"))):
            continue
        spec = _candidate_spec(candidate=candidate, source_commit=commit, scene_plan=scene_plan,
                               task_success_contract=contract)
        spec["evaluation_backend"] = "robolab.v1"
        spec["policy_interface_id"] = "openpi_droid.v1" if key == "pi05_droid" else "groot_n17_droid.v1"
        spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
        result[key] = (spec, repo / MANIFESTS["pi05_checkpoint_inventory"] if key == "pi05_droid" else None)
    return result


def build_planning_inputs(*, catalog: Mapping[str, Any], profile: Mapping[str, Any],
                          capture_root: Path, source: Mapping[str, Any], inputs_dir: Path) -> dict[str, Any]:
    """Materialize facts, validated capabilities and real hard-cap cost quotes.

    Additional runtimes are operator registry ``screening_candidates`` rows,
    each naming a byte-sealed native execution spec and optional inventory.
    No catalog declaration alone grants a runtime capability.
    """
    from .beta_policy_candidate_catalog import validate_catalog, load_catalog
    from .beta_policy_screening_plan import validate_execution_descriptor
    from .beta_robolab_policy_execution import validate_robolab_policy_execution_spec
    from .robot_free_probe_gate import require_robot_free_probe, retained_probe_receipt, task_success_contract_binding
    from .team_candidate_quick10_registry import scene_facts
    from .task_family_policy_execution import FAMILY_BACKEND, validate_task_family_policy_execution_spec

    catalog = validate_catalog(catalog)
    facts = scene_facts(profile)
    inputs = facts["inputs"]
    task = facts["scene"]["task_spec"]
    family = inputs.get("schema_version") == "blueprint.task_family_scene_inputs.v1"
    if family:
        from .beta_policy_screening_family import static_family_readiness
        static = static_family_readiness(facts)
        family_fields = {"evaluation_backend": FAMILY_BACKEND, "pre_policy_site_qualification_required": True,
                         "family_static_readiness_digest": static["readiness"]["readiness_digest"],
                         "family_static_readiness": static["readiness"], "scenario_scope": "native_frozen_reset_seeds",
                         "family_source_inputs": _record(Path(profile["family_scene_inputs_path"]))}
    else:
        receipt = retained_probe_receipt(Path(profile["robot_free_probe_root"]))
        probe = require_robot_free_probe(receipt, scene_revision_digest=inputs["scene_revision_digest"],
            task_kind=task["task_kind"], task_spec_digest=canonical_digest(task),
            task_success_contract_digest=task_success_contract_binding(task["task_kind"], inputs["task_success_contract"]))
    if (source.get("scene_revision_digest") != inputs["scene_revision_digest"]
            or source.get("task_id") != profile["task_id"]
            or Path(capture_root).resolve() != Path(profile["capture_root"]).resolve()):
        raise ValueError("beta_screening_source_scene_mismatch")
    root = Path(inputs_dir).resolve()
    normalized_inputs = ({**inputs, **family_fields} if family else
        {**inputs, "beta_robot_free_probe_receipt": receipt,
         "beta_probe_task_spec_digest": canonical_digest(task), "beta_task_kind": task["task_kind"]})
    normalized_inputs["screening_capture_root"] = str(Path(capture_root).resolve())
    if family:
        normalized_inputs.update(schema_version="blueprint.beta_family_screening_scene_inputs.v1",
                                 source_family_inputs_digest=inputs["inputs_digest"])
        normalized_inputs["inputs_digest"] = canonical_digest(normalized_inputs, digest_field="inputs_digest")
    normalized = _retain(root / (canonical_digest(normalized_inputs)[7:] + ".scene_inputs.json"), normalized_inputs)
    scene = {**dict(source), "site_id": str(profile["scene_id"]), "scene_id": str(profile["scene_id"]),
             "task_kind": task["task_kind"], "task_spec": task,
             "task_success_contract_digest": inputs["task_success_contract_digest"],
             **(family_fields if family else {"robot_free_probe_status": "passed", "robot_free_probe_digest": probe["receipt_digest"]}),
             "scene_inputs": _record(normalized)}
    if profile.get("nominal_dynamic_friction") is not None:
        scene["nominal_dynamic_friction"] = profile["nominal_dynamic_friction"]
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    scene_plan = _read(Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json")
    if scene_plan.get("plan_digest") != canonical_digest(scene_plan, digest_field="plan_digest"):
        raise ValueError("beta_screening_native_scene_plan_digest_invalid")
    scene["native_scene_plan"] = scene_plan
    template_failures = {}
    templates = {}
    if not family:
        try:
            templates = _native_templates(repo=repo, scene_plan={**scene_plan,
                "scene_id": scene_plan.get("scene_id") or profile["scene_id"]},
                contract=inputs["task_success_contract"], commit=commit)
        except (ValueError, OSError, KeyError, TypeError):
            template_failures.update({key: "beta_screening_baseline_runtime_template_unavailable"
                                      for key in ("pi05_droid", "groot_n17_droid")})
    from .cosmos_droid_bundle import CANDIDATES, profile as cosmos_profile
    for candidate in CANDIDATES:
        if family or not any(row["candidate_id"] == candidate for row in catalog["candidates"]):
            continue
        try:
            templates[candidate] = (cosmos_profile(candidate).candidate_execution_spec(scene_plan={**scene_plan,
                "scene_id": scene_plan.get("scene_id") or profile["scene_id"]},
                contract=inputs["task_success_contract"], source_commit=commit), None)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            code = str(exc).split(":", 1)[0]
            template_failures[candidate] = (code if re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code)
                                            else "cosmos_runtime_template_unavailable")
    if not family and any(row["candidate_id"] == "molmoact2_droid" for row in catalog["candidates"]):
        from .molmoact2_droid_runtime import candidate_execution_spec
        try:
            templates["molmoact2_droid"] = (candidate_execution_spec(scene_plan={**scene_plan,
                "scene_id": scene_plan.get("scene_id") or profile["scene_id"]},
                contract=inputs["task_success_contract"], source_commit=commit), None)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            code = str(exc).split(":", 1)[0]
            template_failures["molmoact2_droid"] = (code if re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code)
                                                   else "molmoact2_droid_runtime_template_unavailable")
    bindings, quotes, candidate_blockers = {}, {}, {}
    selection = scene_plan.get("policy_selection") if family else None
    if selection is not None:
        from .task_family_dispatch_contracts import validate_policy_selection
        selection = validate_policy_selection(selection)
        # Compilation's seed policy is provenance, not the candidate universe.
        # Each current candidate is independently matched to the frozen native
        # robot/interface below. Adding a catalog entry must not retire scenes.
        scene["policy_selection"] = selection
    credentials = []
    # Credential presence is intentionally not guessed from a public endpoint.
    # Hosted policies need a separately configured private delivery lane.
    for configured in catalog["candidates"]:
        key = configured["candidate_id"]
        try:
            from .beta_policy_screening_g1 import BACKEND as G1_BACKEND, CANDIDATES as G1_CANDIDATES
            g1 = key in G1_CANDIDATES
            override = (profile.get("screening_candidates") or {}).get(key)
            model = None
            if g1:
                from .beta_policy_screening_g1 import candidate_execution_spec, validate_execution_spec
                if family:
                    raise ValueError("g1_beta_native_rigid_task_required")
                if not override or not override.get("native_g1"):
                    raise ValueError("g1_beta_scene_packet_missing")
                spec = candidate_execution_spec(candidate_id=key, registration=override["native_g1"],
                    scene_plan=scene_plan, base_packet=inputs["base_native_packet"], source=source,
                    contract=inputs["task_success_contract"], site_id=scene["site_id"])
                inventory = repo / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
            elif override:
                if override.get("robot_policy") is not None:
                    from .beta_policy_screening_team import build_registered_team_spec
                    if family:
                        raise ValueError("beta_screening_team_robot_requires_rigid_scene")
                    spec, model = build_registered_team_spec(candidate_id=key, registration=override["robot_policy"],
                        scene_plan=scene_plan, contract=inputs["task_success_contract"], prompt=facts["prompt"],
                        scene_revision_digest=inputs["scene_revision_digest"])
                    inventory = None
                else:
                    spec = _read(_bound_path(override["policy_execution_spec"]))
                    inventory = _bound_path(override["checkpoint_inventory"]) if override.get("checkpoint_inventory") else None
                    model = override.get("team_robot_model")
            elif key in templates:
                spec, inventory = templates[key]
            else:
                raise ValueError(template_failures.get(key, "runtime_candidate_binding_missing"))
            spec = (validate_execution_spec(spec, candidate_id=key) if g1 else
                    validate_task_family_policy_execution_spec(spec, candidate_id=key, scene_plan=scene_plan) if family
                    else validate_robolab_policy_execution_spec(spec, candidate_id=key))
            if (key in templates and not override
                    and configured["policy"] != next(row["policy"] for row in
                        load_catalog(repo / "config" / "beta_policy_candidates.v1.json")["candidates"]
                        if row["candidate_id"] == key)):
                raise ValueError("beta_screening_policy_identity_changed")
            if family:
                from .beta_policy_screening_family import family_interface_matches
                interface_matches = family_interface_matches(configured, spec, static)
            else:
                interface_matches = _interface_matches(configured, spec)
            if (spec["checkpoint_digest"] != configured["policy"]["digest"]
                    or spec["task_success_contract_digest"] != inputs["task_success_contract_digest"]
                    or not interface_matches):
                raise ValueError("beta_screening_policy_interface_or_scene_mismatch")
            if spec.get("team_embodiment") is not None:
                from .team_embodiment import apply_team_embodiment
                apply_team_embodiment(scene_plan, spec["team_embodiment"])
                if model is None:
                    raise ValueError("beta_screening_team_robot_model_missing")
                _bound_path(model)
            # These values come from the registry, never the customer or policy.
            if profile.get("max_hourly_rate_usd") is None:
                raise ValueError("beta_screening_hourly_price_missing")
            rate = Decimal(str(profile["max_hourly_rate_usd"]))
            ttl = profile.get("screening_hard_ttl_seconds", DEFAULT_HARD_TTL_SECONDS)
            cap = Decimal(str(profile.get("screening_candidate_cap_usd", DEFAULT_CANDIDATE_CAP_USD)))
            from .policy_interface_binding import TEAM_HTTPS_INTERFACES
            hosted = spec["policy_interface_id"] in TEAM_HTTPS_INTERFACES
            hosted_settings = {}
            if hosted:
                from .beta_policy_screening_hosted import HOSTED_FIELDS, request_limit, validate_hosted_delivery
                if not override:
                    raise ValueError("beta_screening_hosted_delivery_missing")
                hosted_settings = {name: override[name] for name in HOSTED_FIELDS if name in override}
                hosted_settings["hosted_http_request_limit"] = request_limit(spec, 10)
                validate_hosted_delivery(spec, hosted_settings, episodes=10)
                cap = Decimal(hosted_settings["provider_hard_cap_usd"])

            if (not rate.is_finite() or rate <= 0 or type(ttl) is not int or not 1800 <= ttl <= DEFAULT_HARD_TTL_SECONDS
                    or not cap.is_finite() or not 0 < cap <= 15 or rate * Decimal(ttl) / Decimal(3600) > cap):
                raise ValueError("beta_screening_resource_bounds_invalid")
            path = _retain(root / (spec["execution_spec_digest"][7:] + ".execution_spec.json"), spec)
            descriptor = {"schema_version": EXECUTION_SCHEMA, "candidate_id": key,
                          "policy_execution_spec": _record(path), "checkpoint_inventory": _record(inventory) if inventory else None,
                          "configuration_digest": configured["configuration_digest"],
                          "runtime_digest": spec["runtime_identity_digest"], "backend": G1_BACKEND if g1 else FAMILY_BACKEND if family else "robolab.v1",
                          "task_success_contract_digest": spec["task_success_contract_digest"],
                          "max_hourly_rate_usd": str(rate.quantize(Decimal("0.01"), rounding=ROUND_UP)),
                          "hard_ttl_seconds": ttl, "implementation_commit": commit, **hosted_settings}
            if model is not None:
                descriptor["team_robot_model"] = model
            minimum_memory = configured["runtime"]["minimum_gpu_memory_gb"]
            if minimum_memory is not None:
                descriptor["minimum_gpu_memory_gb"] = minimum_memory
            if profile.get("machine_avoidlist_path"):
                descriptor["machine_avoidlist_path"] = profile["machine_avoidlist_path"]
            descriptor = validate_execution_descriptor(descriptor, candidate_id=key,
                                                        configuration_digest=configured["configuration_digest"])
            if hosted:
                credentials.extend(configured["eligibility"]["requires_credentials"])
            total_cap = cap + Decimal(hosted_settings.get("hosted_inference_maximum_usd", "0"))
            if total_cap > 15:
                raise ValueError("beta_screening_candidate_exceeds_site_budget")
            bindings[key] = {"configuration_digest": configured["configuration_digest"],
                             "runtime_digest": descriptor["runtime_digest"], "backend": descriptor["backend"],
                             "admission_digest": canonical_digest(descriptor), "execution_spec": descriptor}
            quote = {"schema_version": "beta_policy_screening_cost_quote.v1", "quote_id": "operator-cap-" + key,
                     "price_source": ("canonical_gpu_hard_cap_plus_bounded_hosted_http_charges" if hosted else
                                      "canonical_gpu_canary_hard_cap_including_startup_transfer_storage_teardown"),
                     "valid_until": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                     "configuration_digest": configured["configuration_digest"],
                     "scene_revision_digest": inputs["scene_revision_digest"], "episode_count": 10,
                     "maximum_cost_usd": str(total_cap.quantize(Decimal("0.01"), rounding=ROUND_UP)), "quote_digest": ""}
            quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
            quotes[key] = quote
        except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as exc:
            # An invalid robot/endpoint must not suppress every other candidate.
            # Keep stable codes only; arbitrary provider errors may carry URLs
            # or credentials and must never become public blocker text.
            code = str(exc).split(":", 1)[0]
            candidate_blockers[key] = [code if re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code)
                                       else "beta_screening_candidate_configuration_invalid"]
    return {"scene": scene,
            "capabilities": {"candidate_bindings": bindings, "credentials_available": credentials,
                             "candidate_blockers": candidate_blockers},
            "cost_quotes": quotes}
