"""Derive controls from authenticated scene intent and retained preparation.

This worker publishes admitted metadata and installs the canonical controls
intent. It never allocates a provider. Its catalog is service-owned configuration,
never a filesystem-path field accepted from a customer.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_intake as intake
from . import task_evaluation_configured_controls_continuation_provisioning as producer
from .project_spend_reconciliation import validate_project_spend_reconciliation

LINK_SCHEMA = "task_evaluation_scene_preparation_link.v1"
CATALOG_SCHEMA = "task_evaluation_controls_robot_catalog.v1"
CONFIG_ENV = "BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("controls_autoprovision_" + code)


def _json(path: Path) -> dict[str, Any]:
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_refused")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _sealed(path: Path, field: str) -> dict[str, Any]:
    value = _json(path)
    _require(value.get(field) == canonical_digest(value, digest_field=field), "digest_invalid")
    return value


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_digest(result, digest_field=field)
    return result


def _scene_intent(path: Path) -> dict[str, Any]:
    # Scene intents cross the Website/Pipeline boundary; let their canonical
    # contract own its number encoding rather than using controls-only hashing.
    _json(path)
    return intake._read(path, "intent_digest")


def build_preparation_link(**fields: Any) -> dict[str, Any]:
    value = {"schema_version": LINK_SCHEMA, **fields}
    value["link_digest"] = canonical_digest(value, digest_field="link_digest")
    return validate_preparation_link(value)


def validate_preparation_link(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(value) == {"schema_version", "intent_id", "intent_digest", "preparation_id",
        "request_digest", "expected_production_commit", "team_namespace", "scene_id", "task_id",
        "result_filename", "link_digest"}, "link_fields_invalid")
    _require(value["schema_version"] == LINK_SCHEMA and value["link_digest"] ==
             canonical_digest(value, digest_field="link_digest"), "link_digest_invalid")
    for key in ("intent_id", "preparation_id", "team_namespace", "scene_id", "task_id"):
        _require(intake._identifier(value[key]), "link_identity_invalid")
    for key in ("intent_digest", "request_digest"):
        _require(isinstance(value[key], str) and intake._DIGEST.fullmatch(value[key]) is not None,
                 "link_identity_invalid")
    _require(isinstance(value["expected_production_commit"], str) and
             intake._COMMIT.fullmatch(value["expected_production_commit"]) is not None,
             "link_release_invalid")
    expected = value["preparation_id"] + "-" + value["request_digest"].removeprefix("sha256:") + ".json"
    _require(value["result_filename"] == expected, "link_filename_invalid")
    return dict(value)


def payload_digest(root: Path) -> str:
    _require(root.is_dir() and not root.is_symlink(), "runtime_missing")
    rows = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "runtime_symlink")
        if path.is_file():
            rows[path.relative_to(root).as_posix()] = producer._sha256(path)
    _require(bool(rows), "runtime_empty")
    return canonical_digest(rows)


def _asset(row: Mapping[str, Any]) -> Path:
    path = Path(row["path"])
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file() and producer._sha256(path) == row["digest"], "asset_invalid")
    return path


def provision_link(*, link_path: Path, scene_root: Path, preparation_queue_root: Path,
                   catalog: Mapping[str, Any], controls_root: Path, intent_root: Path,
                   profile_dir: Path, expected_production_commit: str,
                   trusted_clients: set[str], now: float | None = None,
                   provisioner: Callable[..., Mapping[str, Any]] | None = None,
                   installer: Callable[..., Mapping[str, Any]] | None = None,
                   service_group: str | None = "blueprint") -> dict[str, Any]:
    moment = time.time() if now is None else now
    link = validate_preparation_link(_sealed(link_path, "link_digest"))
    _require(link_path.parent == scene_root / link["intent_id"], "link_location_invalid")
    directory = scene_root / link["intent_id"]
    intent = _scene_intent(directory / "intent.json")
    _require(intent.get("schema_version") == intake.INTENT_SCHEMA and
             intent.get("intent_id") == link["intent_id"] and
             intent.get("intent_digest") == link["intent_digest"] and
             intent.get("authenticated_issuer") in trusted_clients, "owner_intent_invalid")
    request = intake.validate_request(intent["request"], now=intent["accepted_at_epoch"])
    _require(not (directory / "revoked.json").exists(), "authority_revoked")
    _require(moment < request["execution"]["expires_at_epoch"], "authority_expired")
    _require(link["expected_production_commit"] == expected_production_commit, "release_mismatch")
    _require(link["task_id"] == request["task"]["task_id"], "task_mismatch")
    _require(catalog.get("schema_version") == CATALOG_SCHEMA and catalog.get("catalog_digest") ==
             canonical_digest(catalog, digest_field="catalog_digest"), "catalog_invalid")
    binding = catalog["bindings"].get(request["task"].get("robot_binding_id"))
    _require(isinstance(binding, dict), "robot_binding_missing")
    _require(binding.get("expected_production_commit") == expected_production_commit, "runtime_release_mismatch")
    result_path = preparation_queue_root / "results" / link["result_filename"]
    if not result_path.exists():
        return {"status": "waiting_for_preparation_result", "intent_id": link["intent_id"]}
    result = _sealed(result_path, "result_digest")
    envelope = _sealed(preparation_queue_root / "materialized" / link["result_filename"], "envelope_digest")
    preparation = envelope["request"]
    _require(canonical_digest(preparation) == link["request_digest"] and
             envelope.get("request_digest") == link["request_digest"] and
             preparation.get("preparation_id") == link["preparation_id"] and
             preparation.get("team_namespace") == link["team_namespace"] and
             preparation["scene"]["identity"]["id"] == link["scene_id"] and
             preparation["task"]["identity"]["id"] == link["task_id"] and
             preparation.get("scene_intent_digest") == intent["intent_digest"], "preparation_identity_mismatch")
    _require(result.get("preparation_id") == link["preparation_id"] and
             result.get("source_commit") == expected_production_commit and
             result.get("team_namespace") == link["team_namespace"] and
             result.get("status") == "queued_for_production_scene_configuration", "preparation_result_invalid")
    # Reopen actual task/rights/camera references before reserving any exposure.
    producer._preparation_context(preparation_result_path=result_path,
        preparation_queue_root=preparation_queue_root, expected_production_commit=expected_production_commit)
    robot = _asset(binding["robot_asset_usd"])
    cameras = _asset(binding["embodiment_camera_template"])
    runtime = Path(binding["runtime_source_payload_dir"])
    _require(payload_digest(runtime) == binding["runtime_digest"], "runtime_digest_mismatch")
    cap = binding.get("phase_hard_cap_usd", producer.DEFAULT_PHASE_HARD_CAP_USD)
    _require(intake._number(cap) and 0 < cap <= 50, "phase_cap_invalid")
    inference_cap = producer.DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD
    key = link["request_digest"].removeprefix("sha256:")
    root = controls_root / link["intent_id"] / key
    _require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)), "controls_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    # Serialize all retries, including recovery between publication and install.
    with intake._lock(root):
        identity = {"link_digest": link["link_digest"], "catalog_binding_digest": canonical_digest({
            k: v for k, v in binding.items() if k not in {
                "project_spend_reconciliation", "project_spend_observed_at_epoch"}})}
        retained_path = root / "autoprovision-inputs.json"
        if retained_path.exists():
            retained = _sealed(retained_path, "receipt_digest")
            _require(all(retained.get(k) == v for k, v in identity.items()), "retained_identity_conflict")
        else:
            if binding.get("project_spend_current_path"):
                current = _sealed(Path(binding["project_spend_current_path"]), "receipt_digest")
                _require(current.get("schema_version") == "task_evaluation_project_spend_current.v1", "project_spend_pointer_invalid")
                spend_path = _asset(current)
                observed = current["observed_at_epoch"]
            else:
                spend_path = _asset(binding["project_spend_reconciliation"])
                observed = binding["project_spend_observed_at_epoch"]
            spend = _sealed(spend_path, "receipt_digest")
            _require(spend.get("schema_version") == "adp_project_spend_reconciliation.v1" and
                     intake._number(observed) and 0 <= moment - observed <= 900, "project_spend_stale")
            validate_project_spend_reconciliation(spend_path)
            snapshot_path = root / "project-spend-snapshot.json"
            if snapshot_path.exists():
                _require(_sealed(snapshot_path, "receipt_digest") == spend, "spend_snapshot_conflict")
            else:
                intake.write_exclusive(snapshot_path, spend)
            retained = _seal({**identity, "issued_at_epoch": int(moment),
                "spend_path": str(snapshot_path), "spend_digest": producer._sha256(snapshot_path)}, "receipt_digest")
            intake.write_exclusive(retained_path, retained)
        _require(producer._sha256(Path(retained["spend_path"])) == retained["spend_digest"], "spend_changed")
        # A construction allocation and controls allocation are TWO holds. The
        # OpenAI placement call is a third hold, never hidden outside the cap.
        for phase, provider, amount in [(p, "vast", cap) for p in producer.PHASES] + [
                ("placement", "openai", inference_cap)]:
            intake.reserve_scene_attempt(queue_root=scene_root, intent_id=link["intent_id"],
                attempt_id="controls-" + key[:40] + "-" + phase,
                source_commit=expected_production_commit, runtime_digest=binding["runtime_digest"],
                input_digest=link["request_digest"], provider=provider, maximum_spend_usd=amount, now=moment)
        issued = retained["issued_at_epoch"]
        # Derive text authority from the persisted authenticated record only.
        authority = "scene-intent:" + intent["intent_digest"]
        result = dict((provisioner or producer.provision_configured_controls_continuation)(
            expected_production_commit=expected_production_commit, preparation_result_path=result_path,
            preparation_queue_root=preparation_queue_root, robot_asset_usd_path=robot,
            runtime_source_payload_dir=runtime, embodiment_camera_template_path=cameras,
            project_spend_reconciliation_path=Path(retained["spend_path"]), controls_root=root,
            profile_dir=profile_dir, authorization_reference=authority,
            authorized_by=request["owner"]["user_id"], release_reference=authority,
            openai_project_id=binding["openai_project_id"], openai_api_key_id=binding["openai_api_key_id"],
            phase_hard_cap_usd=cap, phase_ttl_seconds=min(producer.DEFAULT_PHASE_TTL_SECONDS,
                int(cap * 3600 / producer.DEFAULT_HOURLY_RATE_USD)),
            authority_valid_seconds=int(request["execution"]["expires_at_epoch"] - issued),
            now=datetime.fromtimestamp(issued, timezone.utc),
            external_layer_bucket=binding.get("external_layer_bucket")))
        _require(result.get("provider_mutation_performed") is False and
                 result.get("status") == "configured_controls_continuation_provisioned", "producer_result_invalid")
        # Check live authority again immediately before registry installation.
        _require(not (directory / "revoked.json").exists(), "authority_revoked")
        _require((time.time() if now is None else moment) < request["execution"]["expires_at_epoch"], "authority_expired")
        installed = dict((installer or producer.install_intent_into_registry)(
            intent_path=result["intent_path"], intent_root=intent_root,
            expected_production_commit=expected_production_commit, service_group=service_group))
        _require(installed.get("status") == "installed" and installed.get("intent_digest") ==
                 result["intent_digest"], "registry_readback_invalid")
        receipt = _seal({"schema_version": "task_evaluation_controls_autoprovision_receipt.v1",
            "status": "installed", **identity, "intent_id": intent["intent_id"],
            "owner_intent_digest": intent["intent_digest"], "policy_candidates": request["execution"]["policy_candidates"],
            "provisioning": result, "installation": installed,
            "provider_mutation_performed": False}, "receipt_digest")
        path = root / "autoprovision-receipt.json"
        if path.exists():
            _require(_sealed(path, "receipt_digest") == receipt, "receipt_conflict")
        else:
            intake.write_exclusive(path, receipt)
        return receipt


def process_config(config_path: str | Path, *, expected_production_commit: str) -> list[dict[str, Any]]:
    config = _json(Path(config_path))
    catalog = _sealed(Path(config["robot_catalog_path"]), "catalog_digest")
    scene_root = Path(config["scene_root"])
    rows = []
    for link_path in sorted(scene_root.glob("scene-*/preparation-link.json")):
        try:
            rows.append(provision_link(link_path=link_path, scene_root=scene_root, catalog=catalog,
                preparation_queue_root=Path(config["preparation_queue_root"]),
                controls_root=Path(config["controls_root"]), intent_root=Path(config["intent_root"]),
                profile_dir=Path(config["profile_dir"]), expected_production_commit=expected_production_commit,
                trusted_clients=set(config["trusted_clients"]), service_group=config.get("service_group", "blueprint")))
        except (ValueError, OSError, KeyError, TypeError, producer.ConfiguredControlsProvisioningError) as exc:
            row = {"status": "controls_autoprovision_refused", "intent_id": link_path.parent.name,
                   "blocker": str(exc)}
            try:
                link = validate_preparation_link(_sealed(link_path, "link_digest"))
                row["blocked_scene_key"] = [link[k] for k in ("team_namespace", "scene_id", "task_id")]
            except (ValueError, OSError, KeyError, TypeError):
                # Corruption with no recoverable scene scope cannot authorize
                # an older installed intent; only this unresolved case stops all.
                row["scope_unresolved"] = True
            rows.append(row)
    return rows


def owner_authority_blocker(config_path: str | Path, *, scene_intent_digest: str,
                            now: float | None = None) -> str | None:
    """Reopen owner consent at queued dispatch; None means owner consent is live.

    This supplements, never replaces, exact attempt/cap/provider admission.
    Callers must require a digest on autonomous profiles, not downgrade missing
    bindings to legacy authority.
    """
    try:
        _require(isinstance(scene_intent_digest, str) and
                 intake._DIGEST.fullmatch(scene_intent_digest) is not None, "owner_digest_missing")
        config = _json(Path(config_path))
        moment = time.time() if now is None else now
        for path in Path(config["scene_root"]).glob("scene-*/intent.json"):
            intent = _scene_intent(path)
            if intent["intent_digest"] != scene_intent_digest:
                continue
            _require(intent.get("authenticated_issuer") in config["trusted_clients"], "owner_intent_invalid")
            request = intake.validate_request(intent["request"], now=intent["accepted_at_epoch"])
            _require(not (path.parent / "revoked.json").exists(), "authority_revoked")
            _require(moment < request["execution"]["expires_at_epoch"], "authority_expired")
            return None
        return "controls_autoprovision_owner_intent_missing"
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.getenv(CONFIG_ENV), required=not os.getenv(CONFIG_ENV))
    parser.add_argument("--expected-production-commit", required=True)
    args = parser.parse_args()
    rows = process_config(args.config, expected_production_commit=args.expected_production_commit)
    print(json.dumps({"rows": rows, "provider_mutation_performed": False}, sort_keys=True))
    return int(any(row["status"] == "controls_autoprovision_refused" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
