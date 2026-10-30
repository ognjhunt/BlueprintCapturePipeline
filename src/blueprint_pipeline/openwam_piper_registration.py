"""Prepare one fixed public PiPER registration as the unprivileged service user.

No registry write, credential discovery, download, simulator or allocation occurs.
The independently installed operator door validates and installs the proposal.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess

from .deployment_embodiment_preparation_contract import write_once

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = REPO_ROOT / "config/policy_runtimes"
# This operational-only module shares the reviewed stdlib contract with the
# independently installed door. The reverse import never occurs under root.
_spec = importlib.util.spec_from_file_location("_public_policy_install_contract",
    REPO_ROOT / "deploy/operator-door/operator_door/public_policy_registration_contract.py")
assert _spec is not None and _spec.loader is not None
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)
CANDIDATE_ID, CHECKPOINT_ID, TEAM_ID = _contract.CANDIDATE_ID, _contract.CHECKPOINT_ID, _contract.TEAM_ID
MODEL_ROOT, STATE_ROOT = _contract.MODEL_ROOT, _contract.STATE_ROOT
decode, expected_manifest, expected_policy = _contract.decode, _contract.expected_manifest, _contract.expected_policy
file_record, read, require = _contract.file_record, _contract.read, _contract.require
retained_body, rights_record = _contract.retained_body, _contract.rights_record
safe_path, validate_request = _contract.safe_path, _contract.validate_request


def prepare(request: dict, *, state_root: Path = STATE_ROOT, model_root: Path = MODEL_ROOT,
            profile_root: Path = PROFILE_ROOT, release_commit: str | None = None) -> dict:
    from .beta_policy_screening_team import registration_from_prepared_inputs
    from .deployment_embodiment_runtime_configuration import bind_prepared_configuration
    from .openwam_piper_runtime import build_policy_spec

    request = validate_request(request)
    actual = release_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    require(actual == request["release_commit"], "installed_release_mismatch")
    profile = read(profile_root / "openwam_piper_registration_profile.json")
    require(file_record(profile_root / "openwam_piper_model_card.md")["sha256"] == profile["model_card_sha256"],
            "model_license_evidence_changed")
    body, original, model = retained_body(request, model_root=model_root)
    prepared = bind_prepared_configuration(body / "configuration", body / "prepared",
        original["robot_embodiment"]["model"]["url"], runtime_settings={"control_hz": 60, "action_chunk_rows": 1})
    require(prepared["robot_embodiment"] == expected_manifest(original["robot_embodiment"]), "native_binding_changed")
    policy = build_policy_spec(prepared["robot_embodiment"])
    require(policy == expected_policy(prepared["robot_embodiment"], profile), "fixed_policy_profile_changed")
    destination = safe_path(state_root / request["installation_id"])
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_once(destination / "request.json", request)
    registration = registration_from_prepared_inputs(prepared=prepared, model_path=model["path"],
        destination=destination, candidate_id=CANDIDATE_ID, team_id=TEAM_ID, checkpoint_id=CHECKPOINT_ID,
        policy_spec=policy, rights=rights_record(request, profile), display_name=profile["display_name"])
    registration.update(minimum_gpu_memory_gb=profile["minimum_gpu_memory_gb"], deployment=profile["deployment"])
    proposal = {"schema_version": "openwam_piper_policy_install_proposal.v1", "request_digest": request["request_digest"],
                "registration": registration, "candidate_policy_queried": False, "provider_allocations_performed": 0}
    write_once(destination / "proposal.json", proposal)
    return {"status": "prepared", "installation_id": request["installation_id"],
            "request_digest": request["request_digest"], "proposal_path": str(destination / "proposal.json"),
            "candidate_policy_queried": False, "provider_allocations_performed": 0}


def main(argv=None) -> None:
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(decode(args.request_json)), sort_keys=True))


if __name__ == "__main__":
    main()
