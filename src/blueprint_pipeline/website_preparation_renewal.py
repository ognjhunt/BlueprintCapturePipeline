"""Re-issue a completed website preparation as a new job after its sponsor renewed the expired grant.

capture-447ae4f4 (2026-11-02): the preparation completed, its grant expired
and its intent ended. The sponsor renewed the grant
(``renewExpiredWebsiteSceneSponsorship``), but the identical request is
idempotent at both the WebApp and the Pipeline intake, so nothing can start.
A renewal is a new derivative identity (``website_preparation_revision_identity``):
the same capture, task, source and caps under the renewed consent window.

This buys, reserves and reconstructs nothing. It reopens the retained inputs,
recomputes on the CPU only the native inputs bound to the new preparation's
digest, registers that preparation and hands its request to the WebApp's normal
intake. The Pipeline intake then reopens the retained authority and charges every
sibling's holds against the capture's original limits; the WebApp's job ceiling
still applies, and its refusal is a recorded outcome. The handoff, the old
preparation, its registration and its intent are never changed.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .website_preparation_revision_identity import (
    PREFIX, RENEWAL_FIELD, RENEWAL_PREFIX, capture_id, renewal, revision)

FIELD = "website_preparation_renewal"
SCHEMA = "website_preparation_renewal.v1"
REISSUE_SCHEMA = "website_preparation_renewal_reissue.v1"
#: Every refusal this module raises; a detail, when present, is another typed code.
REFUSAL = re.compile(r"website_preparation_renewal_[a-z_]+(?::[a-z][a-z0-9_]{0,127})?")
_CAPS = ("max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("website_preparation_renewal_" + code)


def _detail(exc: BaseException) -> str:
    """The typed code inside a refusal, never its free text."""
    text = str(exc)
    for candidate in (text.rsplit(":", 1)[-1], text):
        if _CODE.fullmatch(candidate):
            return candidate
    return "unclassified"


def _refused(code: str, exc: BaseException) -> ValueError:
    return ValueError(f"website_preparation_renewal_{code}:{_detail(exc)}")


def renewal_request(prior: Mapping[str, Any], grant: Mapping[str, Any]) -> dict[str, Any]:
    """The prior request under the renewed grant's consent window, with a derived identity.

    Only the consent, the grant's caps and expiry, the renewal field and the
    submission id change. A correction revision stays a revision: re-deriving
    its own id already names the new window, so it gains no renewal field.
    """
    value = deepcopy(dict(prior))
    value["consent"] = deepcopy(dict(grant["consent"]))
    value["execution"].update({key: grant[key] for key in _CAPS})
    if revision(prior) is not None:
        prefix = PREFIX
    else:
        value["task"]["subject"][RENEWAL_FIELD] = {"capture_id": capture_id(prior),
                                                  "parent_submission_id": prior["submission_id"]}
        prefix = RENEWAL_PREFIX
    value["submission_id"] = prefix + cross_runtime_canonical_digest(
        {key: item for key, item in value.items() if key != "submission_id"})[7:]
    if capture_id(value) != capture_id(prior):
        raise ValueError("website_preparation_renewal_identity_invalid")
    return value


def _sealed(path: Path, field: str, code: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise ValueError("website_preparation_renewal_" + code) from None
    _require(isinstance(value, dict) and value.get(field) == canonical_digest(value, digest_field=field), code)
    return value


def _registration(root: Path, request: Mapping[str, Any], code: str) -> tuple[Path, dict[str, Any]]:
    """The checked registration of exactly this request; every referenced byte is reopened."""
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .website_scene_source_registry_paths import _index_path
    path = _index_path(root, request)
    try:
        value = read(path, digest_field="registration_digest")
        refs = value["references"]
        if (value.get("schema_version") != "website_scene_source_registration.v1"
                or value.get("request_digest") != cross_runtime_canonical_digest(request)
                or value.get("execution_authority_granted") is not False):
            raise ValueError
        for key in ("preparation", "runtime_inputs", "task_context"):
            checked_file(refs[key]["path"], refs[key])
    except (ValueError, KeyError, TypeError, OSError):
        raise ValueError("website_preparation_renewal_" + code) from None
    return path, value


def _intent_id(request: Mapping[str, Any]) -> str:
    from .task_evaluation_scene_intent_contracts import canonical_digest as intent_digest
    identity = intent_digest({"owner": request["owner"], "submission_id": request["submission_id"]})
    return "scene-" + identity.removeprefix("sha256:")


def _intent(directory: Path, code: str) -> dict[str, Any]:
    from .task_evaluation_scene_intent_contracts import _read
    try:
        if directory.is_symlink():
            raise ValueError
        return _read(directory / "intent.json", "intent_digest")
    except (ValueError, OSError):
        raise ValueError("website_preparation_renewal_" + code) from None


def _ended(directory: Path, intent: Mapping[str, Any], now: float) -> bool:
    """Revoked or past its effective window, and no attempt row can still run or reserve.

    A row ends only with a validated never-started cancellation or a terminal
    settlement, the evidence the intake already accepts for its sibling holds.
    """
    from .task_evaluation_retained_controls_evidence import validated_cancellation
    from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry
    from .task_evaluation_scene_intent_contracts import _read
    try:
        if not ((directory / "revoked.json").exists() or now >= effective_execution_expiry(directory, intent)):
            return False
        attempts = directory / "attempts"
        if attempts.is_symlink():
            return False
        for path in sorted(attempts.glob("*.json")):
            row = _read(path, "attempt_digest")
            if row.get("intent_digest") != intent["intent_digest"] or validated_cancellation(directory, row) is None:
                return False
    except (ValueError, OSError, KeyError, TypeError):
        return False
    return True


def _scope(grant: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a renewal may not change: caps, attempts, policy, task, owner, provider and rights."""
    value = {key: item for key, item in grant.items()
             if key not in {"authority_digest", "expires_at_epoch", "renewal"}
             and not key.endswith("_parent_authority_digest")}
    value["consent"] = {key: item for key, item in (grant.get("consent") or {}).items() if key != "accepted_at_epoch"}
    return value


def _retained_inputs(capture_root: Path, workspace: Path, binding: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """The base scene, source geometry and task masks the prior preparation bound, reopened unchanged.

    The capture's finished Atlas job is reopened, never advanced: no reconstruction or purchase.
    """
    from .local_reconstruction_adapters import _sha256_file
    from .website_atlas_scene_source import atlas_scene_source
    try:
        base = json.loads((workspace / "base_scene.json").read_text())
        _require(all(base[key + "_digest"] == binding[key + "_digest"]
                     and _sha256_file(Path(base[key + "_path"])) == binding[key + "_digest"]
                     for key in ("splat", "collision_mesh")), "inputs_changed")
        # Same output root as the handoff, so the reopened geometry keeps its bound digest.
        atlas = atlas_scene_source(capture_root=capture_root, output_root=workspace / "atlas")
        _require(atlas.get("status") == "ready", "inputs_unavailable")
        geometry = atlas["source_geometry"]
        _require(geometry.get("digest") == binding["source_geometry_digest"], "inputs_changed")
        for name in ("task_masks.coverage.json", "task_masks.geometry.json"):
            path = workspace / name
            if path.is_file() and not path.is_symlink():
                masks = json.loads(path.read_text())
                if (masks.get("digest") == binding["task_masks_digest"]
                        and masks["digest"] == canonical_digest(masks, digest_field="digest")):
                    return base, geometry, masks
        raise ValueError("website_preparation_renewal_inputs_changed")
    except ValueError as exc:
        if REFUSAL.fullmatch(str(exc)):
            raise
        raise _refused("inputs_changed", exc) from exc
    except (KeyError, TypeError, OSError) as exc:
        raise _refused("inputs_changed", exc) from exc


def _copy(source: Path, target: Path) -> None:
    """A byte copy that never replaces different retained bytes."""
    from .core.common import write_text
    raw = source.read_bytes()
    if target.exists() or target.is_symlink():
        _require(not target.is_symlink() and target.read_bytes() == raw, "workspace_conflict")
        return
    write_text(target, raw.decode("utf-8"))


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    from .common import write_json
    if path.exists() or path.is_symlink():
        _require(not path.is_symlink() and json.loads(path.read_text()) == value, "workspace_conflict")
        return
    write_json(path, value)


def reissue_renewed_preparation(*, capture_root: str | Path, queue_root: str | Path, now: float,
                                load_sponsorship: Callable[..., dict[str, Any]] | None = None,
                                load_context: Callable[..., dict[str, Any]] | None = None,
                                enqueue: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Register and enqueue the renewed preparation; idempotent, and a WebApp refusal is an outcome.

    The keyword callables exist for tests; production uses the signed WebApp reads.
    """
    from . import task_evaluation_scene_intake as intake
    from . import website_task_context as webapp
    from .website_preparation_contracts import validate_preparation_request
    from .website_preparation_revision import capture_root_for_preparation
    from .website_scene_source_registry_paths import binding_root
    load_sponsorship = load_sponsorship or webapp.load_website_scene_sponsorship
    load_context = load_context or webapp.load_current_website_task_context
    enqueue = enqueue or webapp.enqueue_website_prepared_scene
    capture_root, queue_root = Path(capture_root), Path(queue_root)
    from . import website_atlas_preview_store as captures
    _require(not captures.capture_withdrawn(capture_root), "capture_withdrawn")

    # 1. The completed handoff and the rigid preparation it registered and enqueued.
    handoff = _sealed(capture_root / "pipeline" / "website_scene_preparation" / "handoff.json", "digest",
                      "handoff_invalid")
    _require(handoff.get("status") == "intake_ready" and handoff.get("website_intake_outbox"),
             "handoff_not_completed")
    _require("development_test" not in handoff, "development_test_unsupported")
    prior_path = Path(str(handoff.get("preparation_path") or ""))
    _require(prior_path.is_absolute() and prior_path.parent.parent.name != "renewals"
             and (capture_root_for_preparation(prior_path) or Path("/")).resolve() == capture_root.resolve(),
             "handoff_invalid")
    prior = _sealed(prior_path, "digest", "prior_preparation_invalid")
    _require(prior["digest"] == handoff.get("preparation_digest"), "prior_preparation_changed")
    _require(prior.get("status") == "intake_ready" and prior.get("blockers") == []
             and isinstance(prior.get("intake_request"), dict), "prior_preparation_not_ready")
    from .website_task_family_packet import is_task_family_preparation
    _require(not is_task_family_preparation(prior), "task_family_unsupported")
    from .website_development_test import environment
    try:
        development = environment(prior)
    except ValueError:
        development = True
    _require(development is None, "development_test_unsupported")
    prior_request = prior["intake_request"]
    _require(renewal(prior_request) is None, "lineage_unproven")
    workspace = prior_path.parent

    # 2. Its registration, under the root the intake reopens it from.
    bindings = binding_root({"intent_root": str(queue_root)})
    _, registered = _registration(bindings, prior_request, "prior_registration_invalid")
    refs = registered["references"]
    _require(refs["preparation"]["path"] == str(prior_path), "prior_registration_invalid")
    context_path = Path(refs["task_context"]["path"])
    stored = _sealed(context_path, "context_digest", "prior_registration_invalid")
    _require(stored["context_digest"] == prior["binding"]["task_context_digest"]
             and stored.get("capture_id") == handoff.get("capture_id"), "prior_registration_invalid")

    # 3. The prior intent has ended, and no attempt of it can still run.
    prior_intent_id = _intent_id(prior_request)
    directory = queue_root / prior_intent_id
    _require(not directory.is_symlink() and (directory / "intent.json").is_file(), "prior_intent_missing")
    intent = _intent(directory, "prior_intent_invalid")
    _require(intent["request"] == prior_request, "prior_intent_mismatch")
    from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry
    try:
        live = not (directory / "revoked.json").exists() and now < effective_execution_expiry(directory, intent)
    except (ValueError, OSError, KeyError, TypeError):
        live = True
    _require(not live, "prior_intent_live")
    _require(_ended(directory, intent, now), "prior_attempt_unsettled")

    # 4. The current task and the renewed grant, through the signed reads.
    try:
        current = load_context(request_id=stored["request_id"], scene_id=stored["scene_id"],
                               capture_id=stored["capture_id"], purpose="scene_preparation")
    except ValueError as exc:
        raise _refused("task_context_unavailable", exc) from exc
    _require(current == stored, "task_context_changed")
    try:
        fresh = load_sponsorship(task_context=current, now=now, create=False)
    except ValueError as exc:
        raise _refused("grant_unavailable", exc) from exc

    # 5. The grant the prior request was issued under; equal consent pins it.
    old = prior.get("website_preparation_authority")
    if old is None:
        old = _sealed(capture_root / "pipeline" / "website_scene_sponsorship.json", "authority_digest",
                      "parent_authority_invalid")
    _require(isinstance(old, dict) and old.get("schema_version") == "website_scene_sponsorship.v1"
             and old.get("authority_digest") == canonical_digest(old, digest_field="authority_digest"),
             "parent_authority_invalid")
    execution = prior_request["execution"]
    _require(old.get("owner") == prior_request["owner"] and old.get("consent") == prior_request["consent"]
             and all(old.get(key) == execution[key] for key in _CAPS), "parent_authority_mismatch")

    # 6. The fresh grant is the sponsor's renewal of exactly that grant, with nothing else changed.
    _require("renewal" not in old, "lineage_unproven")
    block = fresh.get("renewal")
    _require(isinstance(block, dict), "grant_not_renewed")
    _require(isinstance(block.get("parent_authority_digest"), str) and _SHA.fullmatch(block["parent_authority_digest"])
             and type(block.get("parent_expires_at_epoch")) in {int, float}
             and isinstance(block.get("approved_by"), str)
             and isinstance(block.get("approval_reference"), str) and block["approval_reference"].strip(),
             "renewal_invalid")
    root = (old.get("assessment_parent_authority_digest") or old.get("continuation_parent_authority_digest")
            or old["authority_digest"])
    _require(block["parent_authority_digest"] == root
             and block["parent_expires_at_epoch"] == old["expires_at_epoch"], "parent_mismatch")
    _require(block["approved_by"] == (fresh.get("owner") or {}).get("user_id"), "approval_invalid")
    _require(_scope(fresh) == _scope(old), "scope_changed")

    # 7. The renewed request: a new identity of the same capture, checked like any intake.
    new_request = renewal_request(prior_request, fresh)
    providers = ["vast", "openai", "anthropic"] if fresh.get("authoring_provider") == "anthropic" else ["vast", "openai"]
    _require(new_request["execution"]["allowed_providers"] == providers, "providers_changed")
    try:
        intake.validate_request(new_request, now=now)
        validate_preparation_request(request=new_request, task_context=current, authority=fresh, now=now)
    except ValueError as exc:
        raise _refused("request_invalid", exc) from exc
    new_intent_id = _intent_id(new_request)
    key = (prior_request["owner"], capture_id(prior_request))
    for sibling in sorted(queue_root.glob("scene-*")):
        if sibling.name == prior_intent_id:
            continue
        other = _intent(sibling, "sibling_invalid")
        request = other["request"]
        if (request.get("execution") or {}).get("purpose") != "scene_preparation":
            continue
        try:
            same = (request["owner"], capture_id(request)) == key
        except (ValueError, KeyError, TypeError):
            raise ValueError("website_preparation_renewal_sibling_invalid") from None
        if not same or (sibling.name == new_intent_id and request == new_request):
            continue  # Another capture, or this same renewal already staged by an earlier run.
        _require(_ended(sibling, other, now), "live_sibling")

    # 8-9. The renewed preparation over the same retained inputs.
    base, geometry, masks = _retained_inputs(capture_root, workspace, prior["binding"])
    from .task_evaluation_public_scene_attempt_factory import record
    preparation = deepcopy(prior)
    preparation["intake_request"] = new_request
    preparation["website_preparation_authority"] = deepcopy(fresh)
    preparation[FIELD] = {
        "schema_version": SCHEMA, "parent_preparation": record(prior_path),
        "parent_preparation_digest": prior["digest"],
        "parent_request_digest": cross_runtime_canonical_digest(prior_request),
        "parent_intent_id": prior_intent_id, "parent_authority_digest": old["authority_digest"],
        "authority_digest": fresh["authority_digest"], "renewal": deepcopy(block),
        "provider_mutation_performed": False}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    target = workspace / "renewals" / fresh["authority_digest"][7:]
    target.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target / ".reissue.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("website_preparation_renewal_busy") from None
        return _register_and_enqueue(target=target, preparation=preparation, stored=current, context_path=context_path,
            workspace=workspace, bindings=bindings, inputs=(base, geometry, masks), now=now, enqueue=enqueue,
            summary={"capture_id": capture_id(prior_request), "parent_submission_id": prior_request["submission_id"],
                     "parent_intent_id": prior_intent_id, "submission_id": new_request["submission_id"],
                     "intent_id": new_intent_id, "request_digest": cross_runtime_canonical_digest(new_request),
                     "parent_authority_digest": old["authority_digest"], "authority_digest": fresh["authority_digest"]})
    finally:
        os.close(descriptor)


def _native(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in ("runtime_inputs", "capability_verdict", "robot_fit_precheck") if key in result}


def _outcome(target: Path, value: dict[str, Any]) -> dict[str, Any]:
    from .common import write_json
    value = {"schema_version": REISSUE_SCHEMA, **value, "provider_mutation_performed": False,
             "execution_authority_granted": False}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(target / "reissue.json", value)
    return value


def _register_and_enqueue(*, target, preparation, stored, context_path, workspace, bindings, inputs, now,
                          enqueue, summary) -> dict[str, Any]:
    from .task_evaluation_public_scene_attempt_factory import record
    from .task_evaluation_scene_configuration_submission_inputs import read
    from .website_scene_source_registry_paths import _index_path
    request = preparation["intake_request"]
    path = target / "preparation.json"
    native: dict[str, Any] = {}
    if _index_path(bindings, request).exists():
        # An earlier run registered it, and its intent may already be live: rewrite nothing.
        index, registered = _registration(bindings, request, "registration_conflict")
        refs = registered["references"]
        try:
            same = (refs["preparation"]["path"] == str(path) and read(path, digest_field="digest") == preparation
                    and read(refs["task_context"]["path"], digest_field="context_digest") == stored)
        except ValueError:
            same = False
        _require(same, "registration_conflict")
        registration = record(index)
    else:
        _write_once(path, preparation)
        _copy(context_path, target / "task_context.json")
        # Byte copies of the bound scene and masks, beside the preparation, for readers of its directory.
        for name in ("base_scene.json", "task_masks.geometry.json"):
            if (workspace / name).is_file():
                _copy(workspace / name, target / name)
        base, geometry, masks = inputs
        from .website_scene_handoff import prepare_rigid_native_inputs
        try:
            admissible = prepare_rigid_native_inputs(result=native, preparation=preparation, preparation_path=path,
                native_root=target / "native", task_context=stored, now=now, base_scene=base,
                source_geometry=geometry, task_masks=masks)
        except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
            refusal = _refused("native_inputs_failed", exc)
            _outcome(target, {**summary, **_native(native), "status": "held", "blockers": [str(refusal)]})
            raise refusal from exc
        if not admissible:
            # Nothing is registered or enqueued: the re-issue stops where the handoff would have.
            verdict = native.get("capability_verdict") or {}
            code = ("website_preparation_renewal_capability_abstained" if verdict.get("status") == "abstained"
                    else "website_preparation_renewal_robot_fit_none"
                    if (native.get("robot_fit_precheck") or {}).get("status") == "no_fit"
                    else "website_preparation_renewal_appearance_not_authored")
            _outcome(target, {**summary, **_native(native), "status": "held", "blockers": [code]})
            raise ValueError(code)
        from .website_scene_dispatch import register_website_preparation
        registration = register_website_preparation(preparation_path=path,
            runtime_inputs_path=target / "native" / "runtime_inputs.json",
            task_context_path=target / "task_context.json", root=bindings, now=now)
    result = {**summary, "preparation": record(path), "preparation_digest": preparation["digest"],
              "source_registration": registration, **_native(native)}
    try:
        receipt = enqueue(task_context=stored, request=request)
    except ValueError as exc:
        # The WebApp's typed refusal (its job ceiling, for one) is this re-issue's outcome.
        status = re.search(r"_http_([0-9]{3}):", str(exc))
        return _outcome(target, {**result, "status": "enqueue_refused", "webapp_refusal": {
            "code": _detail(exc), "http_status": int(status.group(1)) if status else None}})
    return _outcome(target, {**result, "status": "reissued", "website_intake_outbox": receipt})
