"""Install one exact SAM profile for both production preparation workers.

Environment files contain paths and explicit enablement only. The intake worker
receives no inference-key binding. No command here starts jobs or spends money.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_sam31_preparation_profile import (
    PROFILE_SCHEMA, _file_record, _git, _read_json, _safe_path, _secret_path,
    materialize_sam31_preparation_profile,
)
from .task_evaluation_scene_configuration_sam31_plan import PROFILE_ENV

SCHEMA = "task_evaluation_sam31_service_provisioning.v1"
INTAKE_SERVICE = "blueprint-task-evaluation-launch-preparation.service"
SAM_SERVICE = "blueprint-task-evaluation-sam31-preparation-execution.service"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("sam31_service_provisioning_" + code)


def _path(value: str | Path) -> Path:
    path = Path(value)
    _require(path.is_absolute() and re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)) is not None
             and not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    return path


def _atomic_write(path: Path, content: str, *, immutable: bool = False) -> None:
    _path(path)
    _require(not path.exists() or path.is_file(), "output_not_regular")
    if immutable and path.exists():
        _require(path.read_text() == content, "immutable_binding_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".sam31-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)  # Non-secret configuration, readable by both services.
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validated_profile(path: Path, commit: str) -> dict[str, Any]:
    profile = _read_json(_safe_path(path, kind="file", code="profile_invalid"),
                         code="profile_invalid")
    _require(profile.get("schema_version") == PROFILE_SCHEMA
             and profile.get("source_commit") == commit
             and profile.get("profile_digest") == canonical_digest(profile, digest_field="profile_digest"),
             "profile_identity_invalid")
    repo = Path(profile["repo_root"])
    _require(_git(repo, "rev-parse", "HEAD") == commit and not _git(repo, "status", "--short"),
             "source_checkout_mismatch")
    review, dependencies = profile["sam31_visual_review"], profile["released_dependencies"]
    reopened = materialize_sam31_preparation_profile(
        source_commit=commit, repo_root=repo, server_data_root=profile["server_data_root"],
        runtime_root=profile["runtime_root"],
        sam31_provider_profile_path=profile["artifact_references"]["sam31_provider_profile"]["path"],
        sam31_review_rights_attestation_path=review["rights_attestation"]["path"],
        sam31_review_cost_scope_attestation_path=review["openai_cost_scope_attestation"]["path"],
        sam31_hf_token_file=profile["paid_stages"]["sam31_tracking"]["hf_token_file"],
        openai_admin_api_key_file=review["openai_admin_api_key_file"],
        openai_project_id=review["openai_project_id"], openai_api_key_id=review["openai_api_key_id"],
        flashsplat_root=dependencies["flashsplat_root"],
        dependency_wheelhouse_path=dependencies["dependency_wheelhouse_path"],
        dependency_manifest_path=dependencies["dependency_manifest"]["path"],
        approved_roots=profile["approved_paid_input_roots"],
        ffmpeg_executable=profile["ffmpeg_executable"],
    )
    _require(reopened == profile, "profile_evidence_changed")
    return profile


def provision_sam31_service_environment(
    *, profile_path: str | Path, expected_source_commit: str,
    openai_api_key_file: str | Path, openai_api_key_id: str,
    allow_live_agents_sdk: bool = False,
    environment_root: str | Path = "/etc/blueprint",
    systemd_unit_root: str | Path = "/etc/systemd/system",
    reload_systemd: bool = False,
) -> dict[str, Any]:
    """Reopen the profile, install both bindings, optionally reload unit definitions."""
    _require(type(allow_live_agents_sdk) is bool and type(reload_systemd) is bool,
             "enablement_not_boolean")
    profile_source, env_root, units = map(_path, (profile_path, environment_root, systemd_unit_root))
    profile = _validated_profile(profile_source, expected_source_commit)
    key = _secret_path(_path(openai_api_key_file), group_read_allowed=False, code="inference_key_invalid")
    review = profile["sam31_visual_review"]
    _require(openai_api_key_id == review["openai_api_key_id"], "inference_key_id_mismatch")
    binding = {
        "profile": _file_record(profile_source), "source_commit": expected_source_commit,
        "openai_api_key_file": str(key), "openai_api_key_id": openai_api_key_id,
        "allow_live_agents_sdk": allow_live_agents_sdk,
    }
    binding_digest = canonical_digest(binding)
    retained = env_root / "sam31-service-bindings" / binding_digest.removeprefix("sha256:")
    intake_env, sam_env = retained / "intake.env", retained / "execution.env"
    profile_line = f"{PROFILE_ENV}={profile_source}\n"
    contents = {
        intake_env: profile_line,
        sam_env: (profile_line + f"BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS={int(allow_live_agents_sdk)}\n"
                  + f"OPENAI_API_KEY_FILE={key}\nOPENAI_API_KEY=\n"),
    }
    drop_ins = {
        units / (INTAKE_SERVICE + ".d") / "90-sam-profile-environment.conf": intake_env,
        units / (SAM_SERVICE + ".d") / "90-dedicated-sam-environment.conf": sam_env,
    }
    # Validate every destination before the first mutation. Retain each immutable
    # env before changing either pointer; daemon-reload happens only after both.
    for path in (*contents, *drop_ins):
        _path(path)
        _require(not path.exists() or path.is_file(), "output_not_regular")
    for path, content in contents.items():
        _atomic_write(path, content, immutable=True)
    for path, env in drop_ins.items():
        _atomic_write(path, "[Service]\nEnvironmentFile=" + str(env) + "\n")
    for path, content in contents.items():
        _require(path.read_text() == content, "environment_readback_failed")
    for path, env in drop_ins.items():
        _require(path.read_text() == "[Service]\nEnvironmentFile=" + str(env) + "\n",
                 "drop_in_readback_failed")
    if reload_systemd:
        result = subprocess.run(["systemctl", "daemon-reload"], check=False,
                                capture_output=True, text=True, timeout=30)
        _require(result.returncode == 0, "daemon_reload_failed")
    receipt = {
        "schema_version": SCHEMA, "status": "installed", **binding,
        "binding_digest": binding_digest,
        "environment_files": [_file_record(path) for path in contents],
        "drop_ins": [_file_record(path) for path in drop_ins],
        "systemd_reloaded": reload_systemd, "services_started": False,
        "inference_key_values_read": False, "inference_key_identity_source": "operator_supplied_key_file_and_id",
        "inference_key_bound_to_intake": False, "provider_mutations_performed": 0,
        "submission_performed": False, "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--openai-api-key-file", required=True)
    parser.add_argument("--openai-api-key-id", required=True)
    parser.add_argument("--allow-live-agents-sdk", action="store_true")
    parser.add_argument("--environment-root", default="/etc/blueprint")
    parser.add_argument("--systemd-unit-root", default="/etc/systemd/system")
    parser.add_argument("--reload-systemd", action="store_true")
    parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args(argv)
    receipt_path = _path(args.receipt_out)
    _require(not receipt_path.exists(), "receipt_exists")
    result = provision_sam31_service_environment(
        profile_path=args.profile, expected_source_commit=args.expected_source_commit,
        openai_api_key_file=args.openai_api_key_file, openai_api_key_id=args.openai_api_key_id,
        allow_live_agents_sdk=args.allow_live_agents_sdk, environment_root=args.environment_root,
        systemd_unit_root=args.systemd_unit_root, reload_systemd=args.reload_systemd,
    )
    _atomic_write(receipt_path, canonical_json(result) + "\n", immutable=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
