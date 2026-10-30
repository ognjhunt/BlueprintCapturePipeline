"""Prepare fixed ten-body script registrations as blueprint; no registry write or compute."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from .deployment_embodiment_preparation_contract import write_once
from .openwam_piper_registration import REPO_ROOT, PROFILE_ROOT, _contract as common

_spec = importlib.util.spec_from_file_location("_scripted_registration_contract",
    REPO_ROOT / "deploy/operator-door/operator_door/scripted_policy_registration_contract.py")
assert _spec and _spec.loader
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def prepare(request: dict, *, state_root: Path = common.STATE_ROOT, model_root: Path = common.MODEL_ROOT,
            profile_root: Path = PROFILE_ROOT, release_commit: str | None = None) -> dict:
    from .deployment_embodiment_runtime_configuration import bind_prepared_configuration
    from .scripted_robot_baseline_runtime import build_policy_spec
    from .scripted_robot_registration import registration_from_prepared_inputs
    request = common.validate_request(request, operation="install-scripted-robot-baselines")
    actual = release_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    common.require(actual == request["release_commit"], "installed_release_mismatch")
    profile = contract.load_profile(profile_root, source_root=Path(__file__).parent, license_path=REPO_ROOT / "LICENSE")
    published = {row["embodiment_id"]: row for row in contract.publication(request, model_root)["results"]}
    destination = common.safe_path(state_root / request["installation_id"])
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_once(destination / "request.json", request)
    # Retry an already sealed proposal without replacing its recorded per-body
    # outcome. A corrected import batch gets a distinct reviewed installation.
    if (destination / "proposal.json").exists():
        proposal = common.read(destination / "proposal.json")
        common.require(proposal["request_digest"] == request["request_digest"], "audit_identity_conflict")
        return _receipt(request, proposal)
    registrations, results = {}, []
    for body in common.SCRIPTED_BODIES:
        if published.get(body, {}).get("status") != "retained":
            results.append(contract.result_row(body, "unavailable", "prepared_body_unavailable"))
            continue
        reason = "prepared_body_verification_failed"
        try:
            directory, original, model = common.retained_body(request, model_root=model_root, embodiment_id=body)
            reason = "source_or_import_verification_failed"
            native = original["robot_embodiment"]
            prepared = bind_prepared_configuration(directory / "configuration", directory / "prepared",
                native["model"]["url"], runtime_settings={key: native[key] for key in ("control_hz", "action_chunk_rows")})
            common.require(prepared["robot_embodiment"] == native, "native_binding_changed")
            reason = "tool_profile_unavailable"
            policy = build_policy_spec(native)
            contract.validated_policy(native, policy, profile)
            reason = "source_or_import_verification_failed"
            target = common.safe_path(destination / body)
            target.mkdir(mode=0o700, exist_ok=True)
            registration = registration_from_prepared_inputs(prepared=prepared, model_path=model["path"], destination=target,
                team_id=common.SCRIPTED_TEAM_ID, checkpoint_id=contract.CHECKPOINT_ID,
                rights=contract.rights_record(request, profile, body))
            registrations["blueprint_scripted_" + body] = registration
            results.append(contract.result_row(body, "registered"))
        except (ValueError, OSError, KeyError, TypeError):
            # An unavailable body does not suppress other candidates. Only a
            # bounded stage reason is retained, never paths or exception text.
            results.append(contract.result_row(body, "unavailable", reason))
    proposal = {"schema_version": "scripted_robot_baseline_install_proposal.v1", "request_digest": request["request_digest"],
        "registrations": registrations, "results": results, "candidate_policy_queried": False,
        "scripted_controller_invoked": False, "provider_allocations_performed": 0}
    write_once(destination / "proposal.json", proposal)
    return _receipt(request, proposal)


def _receipt(request: dict, proposal: dict) -> dict:
    return {"status": "prepared", "installation_id": request["installation_id"], "request_digest": request["request_digest"],
        "registered_count": len(proposal["registrations"]), "results": proposal["results"], "candidate_policy_queried": False,
        "scripted_controller_invoked": False, "provider_allocations_performed": 0}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(common.decode(args.request_json)), sort_keys=True))


if __name__ == "__main__":
    main()
