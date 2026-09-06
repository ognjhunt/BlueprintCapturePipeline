"""Bind autonomous learned-policy execution to the owner's actual checkpoints.

An artifact digest here is the admitted checkpoint inventory digest. It is not
a model-name digest, source commit, setup digest, or evaluation result digest.
Legacy profiles without persistent owner intent remain unchanged.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_intake as intake

SCHEMA = "task_evaluation_scene_policy_binding.v1"
INTERPRETATION_DEFAULT = {
    "enabled": True, "external_disclosure_authorized": True,
    "provider_training_authorized": False, "public_redistribution_authorized": False,
    "maximum_cost_usd": 1.5,
}


class ScenePolicyBindingError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ScenePolicyBindingError("scene_policy_" + code)


def candidate_map(value: Any) -> dict[str, str]:
    _require(isinstance(value, list) and len(value) == 2, "two_frozen_candidates_required")
    result = {}
    for row in value:
        _require(isinstance(row, Mapping) and set(row) == {"id", "artifact_digest"}
                 and intake._identifier(row.get("id")) and
                 isinstance(row.get("artifact_digest"), str) and
                 intake._DIGEST.fullmatch(row["artifact_digest"]) is not None, "candidate_identity_invalid")
        _require(row["id"] not in result, "candidate_duplicate")
        result[row["id"]] = row["artifact_digest"]
    return result


def seal_binding(*, scene_intent_digest: str, attempt_id: str,
                 policy_candidates: Sequence[Mapping[str, Any]], runtime_digest: str,
                 input_digest: str) -> dict[str, Any]:
    value = {"schema_version": SCHEMA, "scene_intent_digest": scene_intent_digest,
             "attempt_id": attempt_id, "policy_candidates": [dict(row) for row in policy_candidates],
             "runtime_digest": runtime_digest, "input_digest": input_digest}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    return validate_binding(value)


def validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(value) == {"schema_version", "scene_intent_digest", "attempt_id",
                           "policy_candidates", "runtime_digest", "input_digest", "binding_digest"}
             and value.get("schema_version") == SCHEMA and
             isinstance(value.get("scene_intent_digest"), str) and
             intake._DIGEST.fullmatch(value["scene_intent_digest"]) is not None and
             intake._identifier(value.get("attempt_id")) and value.get("binding_digest") ==
             canonical_digest(value, digest_field="binding_digest"), "binding_invalid")
    candidate_map(value["policy_candidates"])
    _require(all(isinstance(value.get(k), str) and intake._DIGEST.fullmatch(value[k]) is not None
                 for k in ("runtime_digest", "input_digest")), "attempt_identity_invalid")
    return json.loads(json.dumps(value))


def scene_store() -> tuple[Path, set[str]]:
    root = os.getenv(intake.ROOT_ENV)
    clients = set(filter(None, os.getenv(intake.CLIENTS_ENV, "blueprint-webapp").split(",")))
    config_path = os.getenv("BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG")
    if config_path:
        config = json.loads(Path(config_path).read_text())
        root = root or config.get("scene_root")
        clients = set(config.get("trusted_clients", clients))
    _require(isinstance(root, str) and bool(root) and Path(root).is_absolute(), "owner_store_missing")
    return Path(root), clients


def owner_for_profile(profile: Mapping[str, Any], *, now: float | None = None) -> dict[str, Any] | None:
    digest = profile.get("scene_intent_digest")
    if "scene_intent_digest" not in profile:
        _require(not any(key in profile for key in ("scene_attempt_id", "scene_attempt_binding", "scene_policy_candidates")),
                 "owner_digest_missing")
        return None
    _require(isinstance(digest, str) and intake._DIGEST.fullmatch(digest) is not None,
             "owner_digest_invalid")
    root, clients = scene_store()
    _require(not any(p.is_symlink() for p in (root, *root.parents)), "owner_store_unsafe")
    for path in root.glob("scene-*/intent.json"):
        _require(not path.parent.is_symlink(), "owner_store_unsafe")
        intent = intake._read(path, "intent_digest")
        if intent["intent_digest"] != digest:
            continue
        _require(intent.get("schema_version") == intake.INTENT_SCHEMA and
                 intent.get("authenticated_issuer") in clients, "owner_issuer_invalid")
        owner = intake.validate_request(intent["request"], now=intent["accepted_at_epoch"])
        _require(not (path.parent / "revoked.json").exists(), "owner_revoked")
        moment = time.time() if now is None else now
        _require(moment < owner["execution"]["expires_at_epoch"], "owner_expired")
        if "scene_policy_candidates" in profile:
            _require(candidate_map(profile["scene_policy_candidates"]) ==
                     candidate_map(owner["execution"]["policy_candidates"]), "owner_pair_mismatch")
        task = profile.get("task_evaluation_run") or {}
        _require(task.get("task_id") == owner["task"]["task_id"], "owner_task_mismatch")
        return intent
    raise ScenePolicyBindingError("scene_policy_owner_missing")


def validate_setup_pair(setup: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    bound = validate_binding(binding)
    robots = setup.get("robot_presets")
    _require(isinstance(robots, list) and len(robots) == 1, "setup_robot_pair_invalid")
    candidates = robots[0].get("policy_candidates")
    _require(isinstance(candidates, list) and len(candidates) == 2, "setup_candidate_pair_invalid")
    actual = candidate_map([{"id": row.get("candidate_id"),
        "artifact_digest": (row.get("checkpoint") or {}).get("digest")} for row in candidates])
    _require(actual == candidate_map(bound["policy_candidates"]), "setup_checkpoint_mismatch")


def validate_execution_specs(specs: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]) -> None:
    bound = validate_binding(binding)
    actual = candidate_map([{"id": spec.get("candidate_id"),
        "artifact_digest": spec.get("checkpoint_digest")} for spec in specs])
    _require(actual == candidate_map(bound["policy_candidates"]), "execution_checkpoint_mismatch")
    _require(all((spec.get("runtime_identity") or {}).get("checkpoint_inventory_digest") ==
                 spec.get("checkpoint_digest") for spec in specs), "runtime_checkpoint_mismatch")


def validate_owner_binding(profile: Mapping[str, Any], binding: Mapping[str, Any],
                           *, source_commit: str) -> dict[str, Any]:
    owner = owner_for_profile(profile)
    _require(owner is not None, "owner_binding_without_owner")
    bound = validate_binding(binding)
    _require(bound["scene_intent_digest"] == owner["intent_digest"] and
             candidate_map(bound["policy_candidates"]) ==
             candidate_map(owner["request"]["execution"]["policy_candidates"]), "owner_binding_mismatch")
    root, _ = scene_store()
    attempt = intake._read(root / owner["intent_id"] / "attempts" /
                           (bound["attempt_id"] + ".json"), "attempt_digest")
    _require(attempt.get("intent_digest") == owner["intent_digest"] and
             attempt.get("attempt_id") == bound["attempt_id"] and
             attempt.get("source_commit") == source_commit and
             attempt.get("runtime_digest") == bound["runtime_digest"] and
             attempt.get("input_digest") == bound["input_digest"] and
             attempt.get("provider") == "vast", "reserved_attempt_mismatch")
    return bound


def profile_binding_blockers(profile: Mapping[str, Any]) -> list[str]:
    """Pure profile/plan/setup consistency check for the canonical dispatcher."""
    plan = profile.get("internal_policy_canary_execution_plan") or {}
    if not plan and "internal_policy_canary_setup" not in profile:
        return []
    binding = plan.get("scene_policy_binding")
    autonomous = "scene_intent_digest" in profile
    if binding is None and not autonomous:
        return []
    try:
        _require(isinstance(binding, Mapping) and autonomous, "profile_binding_missing")
        bound = validate_binding(binding)
        _require(bound["scene_intent_digest"] == profile.get("scene_intent_digest") and
                 bound["attempt_id"] == profile.get("scene_attempt_id") and
                 candidate_map(bound["policy_candidates"]) ==
                 candidate_map(profile.get("scene_policy_candidates")), "profile_binding_mismatch")
        validate_setup_pair(profile.get("internal_policy_canary_setup") or {}, bound)
        runtime, inputs = policy_attempt_identity(plan, bound["policy_candidates"])
        _require((runtime, inputs) == (bound["runtime_digest"], bound["input_digest"]),
                 "plan_attempt_identity_mismatch")
        attempt = profile.get("scene_attempt_binding") or {}
        _require(attempt.get("intent_digest") == bound["scene_intent_digest"] and
                 attempt.get("attempt_id") == bound["attempt_id"] and
                 attempt.get("source_commit") == plan.get("source_commit") and
                 attempt.get("runtime_digest") == runtime and attempt.get("input_digest") == inputs,
                 "profile_attempt_mismatch")
        _require(set((plan.get("legacy_policy_run_setup") or {}).get("candidate_ids") or []) ==
                 set(candidate_map(bound["policy_candidates"])), "legacy_candidate_mismatch")
        return []
    except (ValueError, KeyError, TypeError) as exc:
        return [str(exc)]


def policy_attempt_identity(plan: Mapping[str, Any], policies: Any) -> tuple[str, str]:
    candidate_map(policies)
    runtime = ((plan.get("preparation_template") or {}).get("execution_adapter") or {}).get(
        "runtime_source_bundle") or {}
    digest = runtime.get("digest")
    _require(isinstance(digest, str) and intake._DIGEST.fullmatch(digest) is not None,
             "runtime_digest_missing")
    inputs = canonical_digest({key: plan[key] for key in (
        "configured_source_launch_id", "scene_revision_digest", "public_setup_digest", "source_commit")}
        | {"policy_candidates": policies})
    return digest, inputs


def stable_presubmission_as_of(*, profile: Mapping[str, Any], source_commit: str,
                              profile_id: str, output_dir: str | Path, as_of: str) -> str:
    """A crash before the handoff checkpoint cannot mint a new setup/attempt."""
    owner = owner_for_profile(profile)
    if owner is None:
        return as_of
    root = Path(output_dir).expanduser()
    _require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
             "presubmission_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    identity = {"scene_intent_digest": owner["intent_digest"], "source_commit": source_commit,
                "profile_id": profile_id, "configured_profile_digest": profile["profile_digest"]}
    path = root / "scene-policy-presubmission-clock.json"
    with intake._lock(root):
        if path.exists():
            _require(not path.is_symlink(), "presubmission_identity_conflict")
            value = json.loads(path.read_text())
            _require(isinstance(value, Mapping) and
                     value.get("clock_digest") == canonical_digest(value, digest_field="clock_digest") and
                     all(value.get(k) == v for k, v in identity.items()) and
                     isinstance(value.get("as_of"), str), "presubmission_identity_conflict")
            return value["as_of"]
        value = {**identity, "as_of": as_of}
        value["clock_digest"] = canonical_digest(value, digest_field="clock_digest")
        intake.write_exclusive(path, value)
    return as_of


def bind_execution_plan(*, profile: Mapping[str, Any], plan: Mapping[str, Any],
                        setup: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reserve the learned run and bind the producer's actual setup before publication."""
    owner = owner_for_profile(profile)
    result = json.loads(json.dumps(plan))
    if owner is None:
        return result, None
    policies = owner["request"]["execution"]["policy_candidates"]
    runtime, inputs = policy_attempt_identity(plan, policies)
    attempt_id = "policy-" + inputs.removeprefix("sha256:")[:48]
    binding = seal_binding(scene_intent_digest=owner["intent_digest"], attempt_id=attempt_id,
        policy_candidates=policies, runtime_digest=runtime, input_digest=inputs)
    validate_setup_pair(setup, binding)
    root, _ = scene_store()
    resource = plan["resource_authority"]
    _require(resource.get("retry_cap") == 0 and resource.get("maximum_provider_allocations") == 1,
             "policy_resource_scope_invalid")
    attempt = intake.reserve_scene_attempt(queue_root=root, intent_id=owner["intent_id"],
        attempt_id=attempt_id, source_commit=plan["source_commit"], runtime_digest=runtime,
        input_digest=inputs, provider="vast", maximum_spend_usd=resource["hard_cap_usd"])
    result["scene_policy_binding"] = binding
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    attempt_binding = {"schema_version": "task_evaluation_scene_attempt_binding.v1",
        **{key: attempt[key] for key in ("intent_id", "intent_digest", "attempt_id", "source_commit",
                                      "runtime_digest", "input_digest")}}
    return result, attempt_binding


def interpretation_for_owner(*, profile: Mapping[str, Any], plan: Mapping[str, Any],
                             default: Mapping[str, Any]) -> dict[str, Any]:
    owner = owner_for_profile(profile)
    if owner is None:
        return dict(default)
    requested = owner["request"]["task"].get("episode_interpretation")
    if requested is not True and not (isinstance(requested, Mapping) and requested.get("enabled") is True):
        return {"enabled": False}
    _require("openai" in owner["request"]["execution"]["allowed_providers"], "interpretation_provider_not_authorized")
    bound = validate_binding(plan.get("scene_policy_binding") or {})
    root, _ = scene_store()
    intake.reserve_scene_attempt(queue_root=root, intent_id=owner["intent_id"],
        attempt_id=bound["attempt_id"] + "-interpretation", source_commit=plan["source_commit"],
        runtime_digest=canonical_digest({"provider": "openai", "purpose": "episode_interpretation",
                                        "source_commit": plan["source_commit"], "authority": default}),
        input_digest=bound["input_digest"], provider="openai", maximum_spend_usd=default["maximum_cost_usd"])
    return dict(default)


def validate_requested_interpretation(*, profile: Mapping[str, Any], plan: Mapping[str, Any],
                                     authority: Any) -> None:
    if "scene_intent_digest" not in profile or authority is None:
        return
    _require(isinstance(authority, Mapping) and intake._number(authority.get("maximum_cost_usd")) and
             0 < authority["maximum_cost_usd"] <= INTERPRETATION_DEFAULT["maximum_cost_usd"] and
             (authority.get("interpreter") or {}).get("provider_id") == "openai",
             "interpretation_authority_scope_invalid")
    admitted = interpretation_for_owner(profile=profile, plan=plan, default=INTERPRETATION_DEFAULT)
    _require(admitted.get("enabled") is True, "interpretation_not_requested")


def execution_setup_binding_blockers(setup: Mapping[str, Any],
                                     specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check retained native setup and real loaded specs immediately before reuse.

    The dispatcher separately reopens each spec's retained file digest and checks
    current owner consent plus the stored attempt through its lifetime guard.
    """
    fields = {"scene_policy_binding", "scene_intent_digest", "scene_attempt_id",
              "scene_attempt_binding", "scene_policy_candidates"}
    if not fields.intersection(setup):
        return []
    try:
        _require(fields.issubset(setup), "execution_owner_fields_missing")
        bound = validate_binding(setup["scene_policy_binding"])
        _require(setup["scene_intent_digest"] == bound["scene_intent_digest"] and
                 setup["scene_attempt_id"] == bound["attempt_id"] and
                 candidate_map(setup["scene_policy_candidates"]) ==
                 candidate_map(bound["policy_candidates"]) and
                 set(setup.get("candidate_ids") or []) == set(candidate_map(bound["policy_candidates"])),
                 "execution_owner_binding_mismatch")
        attempt = setup["scene_attempt_binding"]
        _require(attempt.get("schema_version") == "task_evaluation_scene_attempt_binding.v1" and
                 attempt.get("intent_digest") == bound["scene_intent_digest"] and
                 attempt.get("attempt_id") == bound["attempt_id"] and
                 attempt.get("source_commit") == setup.get("source_commit") and
                 attempt.get("runtime_digest") == bound["runtime_digest"] and
                 attempt.get("input_digest") == bound["input_digest"], "execution_attempt_mismatch")
        validate_execution_specs(specs, bound)
        _require(all((spec.get("runtime_identity") or {}).get("source_commit") ==
                     setup.get("source_commit") for spec in specs), "execution_source_commit_mismatch")
        return []
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return [str(exc)]
