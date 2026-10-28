"""Evidence-consuming, operator-authorized qualification for beta site-task simulation.

No network or provider execution occurs here. The existing operator registry pins
the request, frozen tolerances, independent measurements, simulator measurements
and human review. Raw source bytes are rechecked on every admission and closeout.
No output constitutes physical performance or ADP qualified evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest as digest
from .policy_interface_binding import INTERFACES

SCHEMA = "blueprint.site_task_simulation_qualification.v1"
DIMENSIONS = {"scale", "registration", "collision", "task_state", "observation", "task_dynamics"}
ROLES = {"capture", "testbed", "scene", "task_spec", "success_contract", "robot_model", "measurement", "simulator_measurement"}
BINDING_FIELDS = {"scene_request_id", "site_id", "task_id", "capture_id", "source_capture_digest",
                  "testbed_digest_sha256", "scene_revision_digest", "task_spec_digest", "success_contract_digest",
                  "robot_model_digest", "policy_interface_id"}
PROJECTION_FIELDS = {"evidence_scope", "capture_id", "scene_revision_digest", "registration_receipt_digest",
                     "fidelity_review_receipt_digest"}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class SiteTaskQualificationError(ValueError):
    """A qualification is absent, stale, unauthorized or no longer bound."""


def _require(ok: Any, code: str) -> None:
    if not ok:
        raise SiteTaskQualificationError("site_task_qualification_" + code)


def _object(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "record_invalid")
    return dict(value)


def _time(value: Any) -> datetime:
    try:
        _require(isinstance(value, str) and value.endswith("Z"), "time_invalid")
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require(result.tzinfo is not None, "time_invalid")
        return result
    except (TypeError, ValueError) as exc:
        raise SiteTaskQualificationError("site_task_qualification_time_invalid") from exc


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _read(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "source_file_invalid")
    return _object(json.loads(path.read_text(encoding="utf-8")))


def _file_digest(path: Path) -> str:
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(chunk)
    return "sha256:" + actual.hexdigest()


def _seal(value: dict[str, Any], field: str = "receipt_digest") -> dict[str, Any]:
    return {**value, field: digest(value, digest_field=field)}


def validate_projection(value: Any) -> dict[str, Any]:
    row = _object(value)
    _require(set(row) == PROJECTION_FIELDS and row.get("evidence_scope") == "simulation"
             and isinstance(row.get("capture_id"), str) and bool(row["capture_id"])
             and all(_DIGEST.fullmatch(str(row.get(key))) for key in PROJECTION_FIELDS
                     - {"evidence_scope", "capture_id"}), "projection_invalid")
    return row


def validate_receipt(value: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """Validate portable structure and cross-record bindings; provenance needs registry recomputation."""
    row = _object(value)
    fields = {"schema_version", "evidence_scope", "binding", "qualification", "registration_receipt",
              "fidelity_review_receipt", "source_artifacts", "request_digest", "profile_digest", "review_digest",
              "authority_digest", "reviewed_by", "reviewed_at_iso", "expires_at_iso", "physical_performance_claimed",
              "adp_qualified_evidence_eligible", "receipt_digest"}
    _require(set(row) == fields and row.get("schema_version") == SCHEMA
             and row.get("evidence_scope") == "simulation" and row.get("physical_performance_claimed") is False
             and row.get("adp_qualified_evidence_eligible") is False
             and row.get("receipt_digest") == digest(row, digest_field="receipt_digest"), "receipt_invalid")
    binding = _object(row["binding"])
    _require(set(binding) == BINDING_FIELDS and all(isinstance(v, str) and v for v in binding.values())
             and binding["policy_interface_id"] in INTERFACES
             and all(_DIGEST.fullmatch(binding[key]) for key in BINDING_FIELDS if "digest" in key), "binding_invalid")
    _require(all(_DIGEST.fullmatch(str(row[key])) for key in
                 ("request_digest", "profile_digest", "review_digest", "authority_digest")), "digest_invalid")
    reviewed, expiry = _time(row["reviewed_at_iso"]), _time(row["expires_at_iso"])
    moment = now or datetime.now(timezone.utc)
    _require(reviewed <= moment < expiry and isinstance(row["reviewed_by"], str) and row["reviewed_by"], "review_expired")
    sources = row["source_artifacts"]
    _require(isinstance(sources, list) and len(sources) == len(ROLES), "sources_invalid")
    source_map = {}
    for source in sources:
        source = _object(source)
        _require(set(source) == {"role", "uri", "sha256", "size_bytes"} and source["role"] in ROLES
                 and source["role"] not in source_map and _DIGEST.fullmatch(str(source["sha256"]))
                 and type(source["size_bytes"]) is int and source["size_bytes"] > 0
                 and isinstance(source["uri"], str) and source["uri"].startswith(("gs://", "s3://", "https://")),
                 "sources_invalid")
        source_map[source["role"]] = source
    _require(source_map["robot_model"]["sha256"] == binding["robot_model_digest"], "robot_model_mismatch")
    projection = validate_projection(row["qualification"])
    _require(projection["capture_id"] == binding["capture_id"]
             and projection["scene_revision_digest"] == binding["scene_revision_digest"], "projection_mismatch")
    check_ids = set()
    for field, schema, dimensions in (
        ("registration_receipt", "blueprint.site_task_registration.v1", {"scale", "registration", "collision"}),
        ("fidelity_review_receipt", "blueprint.site_task_fidelity_review.v1", {"task_state", "observation", "task_dynamics"}),
    ):
        sub = _object(row[field])
        _require(set(sub) == {"schema_version", "binding", "request_digest", "profile_digest", "source_measurement_digest",
                              "simulator_measurement_digest", "checks", "receipt_digest"}
                 and sub["schema_version"] == schema and sub["binding"] == binding
                 and all(sub[key] == row[key] for key in ("request_digest", "profile_digest"))
                 and sub["source_measurement_digest"] == source_map["measurement"]["sha256"]
                 and sub["simulator_measurement_digest"] == source_map["simulator_measurement"]["sha256"]
                 and sub["receipt_digest"] == digest(sub, digest_field="receipt_digest")
                 and projection[field + "_digest"] == sub["receipt_digest"], "subreceipt_mismatch")
        _require(isinstance(sub["checks"], list) and bool(sub["checks"]), "checks_invalid")
        seen = set()
        for check in sub["checks"]:
            _require(isinstance(check, dict) and set(check) == {"id", "dimension", "unit", "sample_count", "max_abs_error",
                         "rms_error", "maximum_absolute_error", "maximum_rms_error"}
                     and _LABEL.fullmatch(str(check["id"])) and _LABEL.fullmatch(str(check["unit"]))
                     and check["id"] not in check_ids and check["dimension"] in dimensions
                     and type(check["sample_count"]) is int and check["sample_count"] > 0
                     and all(_number(check[key]) and check[key] >= 0 for key in
                             ("max_abs_error", "rms_error", "maximum_absolute_error", "maximum_rms_error"))
                     and check["max_abs_error"] <= check["maximum_absolute_error"]
                     and check["rms_error"] <= check["maximum_rms_error"], "checks_invalid")
            check_ids.add(check["id"])
            seen.add(check["dimension"])
        _require(seen == dimensions, "coverage_incomplete")
    return row


def _pinned_document(authority: Mapping[str, Any], role: str) -> dict[str, Any]:
    value = _read(Path(str(authority.get(role + "_path") or "")))
    _require(digest(value) == authority.get(role + "_digest"), role + "_digest_mismatch")
    return value


def qualify_from_profile(operator_profile: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Consume the exact request/profile/review authorized by the existing operator registry."""
    moment = now or datetime.now(timezone.utc)
    authority = _object(operator_profile.get("site_task_qualification"))
    request, profile, review = (_pinned_document(authority, key) for key in ("request", "profile", "review"))
    binding = _object(request.get("binding"))
    _require(request.get("schema_version") == "blueprint.site_task_qualification_request.v1"
             and profile.get("schema_version") == "blueprint.site_task_fidelity_profile.v1"
             and review.get("schema_version") == "blueprint.site_task_qualification_review.v1", "input_schema_invalid")
    _require(review.get("decision") == "accepted_for_bounded_simulation"
             and review.get("request_digest") == authority["request_digest"]
             and review.get("profile_digest") == authority["profile_digest"]
             and review.get("reviewed_by") in authority.get("allowed_reviewers", [])
             and _time(review.get("reviewed_at_iso")) <= moment < _time(review.get("expires_at_iso")), "review_unauthorized")
    artifacts = request.get("source_artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == len(ROLES), "sources_invalid")
    portable, documents, role_paths = [], {}, {}
    root = Path(str(authority["request_path"])).resolve().parent
    prefixes = authority.get("canonical_uri_prefixes")
    _require(isinstance(prefixes, list) and prefixes and all(isinstance(p, str) and p.endswith("/")
             and p.startswith(("gs://", "s3://", "https://")) for p in prefixes), "canonical_storage_unconfigured")
    for item in artifacts:
        item = _object(item)
        _require(set(item) == {"role", "path", "uri", "sha256", "size_bytes"} and item["role"] in ROLES
                 and item["role"] not in role_paths and isinstance(item["uri"], str)
                 and any(item["uri"].startswith(prefix) for prefix in prefixes), "source_authority_invalid")
        local = Path(str(item["path"]))
        _require(not local.is_absolute() and ".." not in local.parts, "source_path_invalid")
        local = root / local
        _require(local.resolve().is_relative_to(root) and not local.is_symlink() and local.is_file(), "source_path_invalid")
        structured = item["role"] in {"capture", "testbed", "task_spec", "success_contract", "measurement", "simulator_measurement"}
        if structured:
            # Authenticate and parse one byte buffer. A second open after hashing
            # could otherwise parse a different, unreviewed measurement document.
            raw = local.read_bytes()
            _require(len(raw) == item["size_bytes"] and len(raw) > 0
                     and "sha256:" + hashlib.sha256(raw).hexdigest() == item["sha256"], "source_bytes_changed")
            documents[item["role"]] = _object(json.loads(raw))
        else:
            _require(local.stat().st_size == item["size_bytes"] and item["size_bytes"] > 0
                     and _file_digest(local) == item["sha256"], "source_bytes_changed")
        role_paths[item["role"]] = local
        portable.append({key: item[key] for key in ("role", "uri", "sha256", "size_bytes")})
    from .decision_evidence_contracts import MaintainedSiteTaskTestbed
    from .capture_intake import validate_capture_intake_envelope
    capture = validate_capture_intake_envelope(documents["capture"])
    testbed = MaintainedSiteTaskTestbed.from_mapping(documents["testbed"]).to_mapping()
    _require(testbed["testbed_digest"] == binding.get("testbed_digest_sha256")
             and testbed.get("approved_task_definition", {}).get("approved_task_id") == binding.get("task_id")
             and binding.get("task_id") in testbed.get("task_distribution", {}).get("tasks", [])
             and capture["scene_id"] == binding.get("site_id")
             and len([source for source in testbed["source_capture_bundles"]
                      if source.get("digest") == binding.get("source_capture_digest")
                      and source.get("bundle_id") == capture["intake_id"]
                      and source.get("envelope_digest") == capture["envelope_digest"]
                      and source.get("version") == capture["capture_authority_profile"]]) == 1, "capture_testbed_mismatch")
    capture_root = role_paths["capture"].parent.resolve()
    for original in capture["original_files"]:
        path = capture_root / original["relative_path"]
        _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(capture_root),
                 "capture_original_missing")
        _require(path.stat().st_size == original["size_bytes"] and _file_digest(path) == original["sha256"],
                 "capture_original_changed")
    from .adp_articulated_task_success_contract import validate_task_success_contract, task_kind_of_contract
    contract = validate_task_success_contract(documents["success_contract"],
                                             task_kind=task_kind_of_contract(documents["success_contract"]))
    _require(canonical_digest(documents["task_spec"]) == binding.get("task_spec_digest")
             and contract["contract_digest"] == binding.get("success_contract_digest"), "task_binding_mismatch")
    source_map = {item["role"]: item for item in portable}
    if (operator_profile.get("default_robot_model") is not None
            and binding.get("policy_interface_id") in {"openpi_droid.v1", "groot_n17_droid.v1", "openpi_droid_https.v1"}):
        from .blueprint_robot_model_delivery import require_default_robot_model
        require_default_robot_model(profile=operator_profile, model_path=role_paths["robot_model"],
            expected_model_sha256=binding["robot_model_digest"],
            expected_closure_digest=authority.get("default_robot_model_closure_digest"))
    measured, simulated = documents["measurement"], documents["simulator_measurement"]
    if simulated.get("extraction") is not None:
        # A generated numeric document is not a substitute for its retained
        # engine/render sources. Recompute those bytes at admission/closeout.
        from .site_task_simulator_measurements import produce_from_operator_profile
        _require(produce_from_operator_profile(operator_profile) == simulated, "simulator_extraction_changed")
    for doc, role in ((measured, "measurement"), (simulated, "simulator")):
        _require(doc.get("schema_version") == "blueprint.site_task_measurement_samples.v1"
                 and doc.get("binding") == binding and doc.get("producer_id") in authority.get("allowed_" + role + "_producers", [])
                 and _time(doc.get("observed_at_iso")) <= _time(review["reviewed_at_iso"]), "measurement_provenance_invalid")
    _require(measured["producer_id"] != simulated["producer_id"]
             and review["reviewed_by"] not in {measured["producer_id"], simulated["producer_id"]}, "independent_review_required")
    _require(isinstance(profile.get("checks"), list) and profile["checks"], "profile_invalid")
    results = []
    for criterion in profile["checks"]:
        criterion = _object(criterion)
        _require(set(criterion) == {"id", "dimension", "unit", "minimum_sample_count", "maximum_absolute_error", "maximum_rms_error"}
                 and criterion["dimension"] in DIMENSIONS and type(criterion["minimum_sample_count"]) is int
                 and criterion["minimum_sample_count"] > 0 and all(_number(criterion[key]) and criterion[key] >= 0
                     for key in ("maximum_absolute_error", "maximum_rms_error")), "profile_invalid")
        pairs = []
        for doc in (measured, simulated):
            checks = doc.get("checks")
            _require(isinstance(checks, list), "samples_invalid")
            rows = [row for row in checks if isinstance(row, dict) and row.get("id") == criterion["id"]]
            _require(len(rows) == 1 and rows[0].get("unit") == criterion["unit"]
                     and rows[0].get("dimension") == criterion["dimension"], "samples_binding_mismatch")
            samples = rows[0].get("samples")
            _require(isinstance(samples, list) and len(samples) >= criterion["minimum_sample_count"], "samples_insufficient")
            by_id = {}
            for sample in samples:
                _require(isinstance(sample, dict) and set(sample) == {"sample_id", "value"}
                         and _LABEL.fullmatch(str(sample["sample_id"])) and sample["sample_id"] not in by_id
                         and _number(sample["value"]), "samples_invalid")
                by_id[sample["sample_id"]] = sample["value"]
            pairs.append(by_id)
        _require(set(pairs[0]) == set(pairs[1]), "sample_pairs_mismatch")
        errors = [abs(pairs[0][key] - pairs[1][key]) for key in pairs[0]]
        max_error, rms = max(errors), math.sqrt(sum(error * error for error in errors) / len(errors))
        _require(max_error <= criterion["maximum_absolute_error"] and rms <= criterion["maximum_rms_error"],
                 "fidelity_failed:" + str(criterion["id"]))
        results.append({key: criterion[key] for key in ("id", "dimension", "unit", "maximum_absolute_error", "maximum_rms_error")} |
                       {"sample_count": len(errors), "max_abs_error": max_error, "rms_error": rms})
    common = {"binding": binding, "request_digest": authority["request_digest"], "profile_digest": authority["profile_digest"],
              "source_measurement_digest": source_map["measurement"]["sha256"],
              "simulator_measurement_digest": source_map["simulator_measurement"]["sha256"]}
    registration = _seal({"schema_version": "blueprint.site_task_registration.v1", **common,
                          "checks": [row for row in results if row["dimension"] in {"scale", "registration", "collision"}]})
    fidelity = _seal({"schema_version": "blueprint.site_task_fidelity_review.v1", **common,
                      "checks": [row for row in results if row["dimension"] not in {"scale", "registration", "collision"}]})
    receipt = _seal({"schema_version": SCHEMA, "evidence_scope": "simulation", "binding": binding,
        "qualification": {"evidence_scope": "simulation", "capture_id": binding["capture_id"],
            "scene_revision_digest": binding["scene_revision_digest"], "registration_receipt_digest": registration["receipt_digest"],
            "fidelity_review_receipt_digest": fidelity["receipt_digest"]},
        "registration_receipt": registration, "fidelity_review_receipt": fidelity, "source_artifacts": portable,
        "request_digest": authority["request_digest"], "profile_digest": authority["profile_digest"], "review_digest": authority["review_digest"],
        # Local locators are deployment settings, not portable business identity.
        # The operator registry still protects them; content/review pins remain
        # part of this authority digest when retained evidence moves computers.
        "authority_digest": digest({key: value for key, value in authority.items()
                                     if key not in {"receipt_digest", "request_path", "profile_path", "review_path"}}),
        **{key: review[key] for key in ("reviewed_by", "reviewed_at_iso", "expires_at_iso")},
        "physical_performance_claimed": False, "adp_qualified_evidence_eligible": False})
    return validate_receipt(receipt, now=moment)


def require_current_qualification(*, receipt: Any, operator_profile: Mapping[str, Any],
                                  request: Mapping[str, Any], facts: Mapping[str, Any],
                                  robot_model_path: Path, now: datetime | None = None) -> dict[str, Any]:
    """Recompute from operator-pinned raw evidence and match the actual scene, task and robot bytes."""
    frozen = validate_receipt(receipt, now=now)
    authority = _object(operator_profile.get("site_task_qualification"))
    _require(authority.get("receipt_digest") == frozen["receipt_digest"], "operator_receipt_pin_missing")
    current = qualify_from_profile(operator_profile, now=now)
    _require(current == frozen, "receipt_changed")
    binding = frozen["binding"]
    scene = _object(facts.get("scene"))
    package = _object(request.get("policy_package"))
    blueprint_owned = request.get("execution_operator") == {"kind": "blueprint"}
    endpoint = package if blueprint_owned else _object(package.get("policy_api_endpoint"))
    _require(binding["scene_revision_digest"] == scene.get("scene_revision_digest")
             and binding["task_spec_digest"] == canonical_digest(_object(scene.get("task_spec")))
             and binding["success_contract_digest"] == _object(scene.get("task_success_contract")).get("contract_digest")
             and binding["policy_interface_id"] == endpoint.get("policy_interface_id")
             and binding["capture_id"] == _object(request.get("site_package")).get("capture_id"), "runtime_binding_mismatch")
    # Team declarations retain their original path. Blueprint defaults must be
    # selected by the operator and match the full closure pinned by this review.
    from .team_candidate_quick10_launch import request_embodiment
    manifest = request_embodiment(request)
    if manifest is None:
        _require(operator_profile.get("default_robot_model") is not None, "default_robot_model_unbound")
        from .blueprint_robot_model_delivery import require_default_robot_model
        require_default_robot_model(profile=operator_profile, model_path=robot_model_path,
            expected_model_sha256=binding["robot_model_digest"],
            expected_closure_digest=authority.get("default_robot_model_closure_digest"))
    else:
        _require(manifest["model"]["sha256"] == binding["robot_model_digest"]
                 and robot_model_path.is_file() and not robot_model_path.is_symlink()
                 and _file_digest(robot_model_path) == binding["robot_model_digest"],
                 "robot_model_mismatch")
    sources = {item["role"]: item for item in frozen["source_artifacts"]}
    scene_path = Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json"
    _require(_file_digest(scene_path) == sources["scene"]["sha256"], "scene_bytes_changed")
    return frozen


def publish_receipt(receipt: Mapping[str, Any], *, client: Any) -> dict[str, Any]:
    """Explicit signed publication through the existing Pipeline/WebApp HMAC transport."""
    value = validate_receipt(receipt)
    result = client._json("/api/internal/pipeline/site-task-qualifications", method="POST", payload=value)
    _require(result.get("receiptDigest") == value["receipt_digest"] and type(result.get("created")) is bool,
             "publication_acknowledgement_mismatch")
    return result


def qualification_for_execution(*, row: Mapping[str, Any], request: Mapping[str, Any],
                                operator_profile: Mapping[str, Any], facts: Mapping[str, Any],
                                robot_model_path: Path, now: datetime | None = None) -> dict[str, Any] | None:
    envelope = _object(row.get("execution_admission"))
    receipt = envelope.get("site_task_qualification_receipt")
    projection = envelope.get("site_task_qualification")
    if envelope.get("evidence_scope") != "simulation":
        _require(receipt is None and projection is None, "diagnostic_cannot_promote")
        return None
    frozen = require_current_qualification(receipt=receipt, operator_profile=operator_profile,
        request=request, facts=facts, robot_model_path=robot_model_path, now=now)
    _require(projection == frozen["qualification"], "projection_mismatch")
    envelope_binding = _object(envelope.get("binding"))
    for key in ("scene_request_id", "site_id", "task_id", "capture_id", "testbed_digest_sha256"):
        _require(envelope_binding.get(key) == frozen["binding"][key], "execution_binding_mismatch:" + key)
    _require(envelope_binding.get("capture_digest_sha256") == frozen["binding"]["source_capture_digest"],
             "execution_binding_mismatch:capture_digest_sha256")
    return frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="Publish the retained receipt via existing signed WebApp transport.")
    parser.add_argument("--webapp-url")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--prepare-simulator-measurements", action="store_true",
                        help="Extract operator-pinned runtime readings only; no qualification or publication.")
    args = parser.parse_args(argv)
    if args.publish and not (args.webapp_url and args.token_file):
        parser.error("--publish requires --webapp-url and --token-file")
    if not args.publish and (args.webapp_url or args.token_file):
        parser.error("--webapp-url and --token-file require explicit --publish")
    if args.prepare_simulator_measurements and args.publish:
        parser.error("--prepare-simulator-measurements cannot publish a qualification")
    from .team_candidate_quick10_registry import profile_for_capture
    profile = profile_for_capture(args.capture_root)
    _require(profile is not None, "operator_profile_missing")
    if args.prepare_simulator_measurements:
        from .site_task_simulator_measurements import produce_from_operator_profile, retain_measurements
        document = produce_from_operator_profile(profile)
        retain_measurements(args.out, document)
        print(json.dumps({"measurement_digest": digest(document), "path": str(args.out.resolve()),
                          "evidence_scope": "simulator_readings_not_site_qualification"}))
        return 0
    receipt = qualify_from_profile(profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or args.out.is_symlink():
        _require(_read(args.out) == receipt, "output_conflict")
    else:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"receipt_digest": receipt["receipt_digest"], "path": str(args.out.resolve())}))
    if args.publish:
        from .agent_run_executor import AgentRunWebAppClient
        from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
        client = AgentRunWebAppClient(base_url=args.webapp_url,
            token=load_pipeline_sync_token(token_file_path=args.token_file, require_file=True))
        result = publish_receipt(receipt, client=client)
        print(json.dumps({"receipt_digest": receipt["receipt_digest"], "created": result["created"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
