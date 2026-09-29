"""ADP-050/day 28: seal the observation-only native policy worker."""
from __future__ import annotations
import json
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, Mapping

from .controlled_policy_configuration import validate_native_configuration
from .native_task_arena_bundle import build_native_task_arena_bundle
from .native_task_arena_construction_bundle import load_verified_native_task_arena_construction_bundle
from .native_task_arena_execution_contract import CONTROLLED_POLICY_RUNTIME_MODULE_NAMES

PROBE_KIND = "native-task-arena-controlled-policy"
RESULT_FILENAME = "controlled_native_policy_result.v1.json"
RUNTIME_MODULE_NAMES = CONTROLLED_POLICY_RUNTIME_MODULE_NAMES


def build_controlled_native_policy_bundle(*, job_dir: Path, packet_dir: Path,
        runtime_source_packet_receipt: Path, implementation_commit: str,
        configuration: Mapping[str, Any], job_request: Mapping[str, Any],
        observations: list[Mapping[str, Any]], policy_credential: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    config = validate_native_configuration(configuration)
    plan = json.loads((packet_dir / "native_task_arena_scene_plan.v1.json").read_text())
    if config["scene_plan_digest"] != plan["plan_digest"]:
        raise ValueError("controlled_native_bundle_scene_binding_mismatch")
    package = Path(__file__).parent
    from .provider_runtime_import_closure import assert_provider_runtime_import_closure
    assert_provider_runtime_import_closure(package_source_dir=package,
        shipped_module_names=(*RUNTIME_MODULE_NAMES, "controlled_native_policy_worker.py"),
        shipped_package_files=("core/__init__.py", "core/security_controls.py"),
        code="controlled_native_provider_import_closure_incomplete")
    dependency = distribution("rfc8785")
    if dependency.version != "0.1.4":
        raise ValueError("controlled_native_canonicalizer_version_mismatch")
    extras = {"blueprint_pipeline/core/__init__.py": package / "core/__init__.py",
              "blueprint_pipeline/core/security_controls.py": package / "core/security_controls.py"}
    extras.update({relative: Path(dependency.locate_file(relative)) for relative in (
        "rfc8785/__init__.py", "rfc8785/_impl.py", "rfc8785/py.typed", "rfc8785-0.1.4.dist-info/LICENSE")})
    with tempfile.TemporaryDirectory(prefix="blueprint-controlled-native-") as raw:
        inputs = {}
        for name, value in (("controlled_native_configuration.json", config),
                            ("controlled_job_request.json", job_request), ("controlled_observations.json", observations)):
            path = Path(raw) / name
            path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
            inputs[name] = path
        if policy_credential is not None:
            if (policy_credential.get("job_id") != job_request["job_id"]
                    or policy_credential.get("endpoint_url") != job_request["policy_package"]["policy_api_endpoint"]["endpoint_url"]
                    or not isinstance(policy_credential.get("bearer_token"), str)
                    or not policy_credential["bearer_token"] or any(c in policy_credential["bearer_token"] for c in "\r\n")):
                raise ValueError("controlled_native_policy_credential_binding_invalid")
            credential_path = Path(raw) / "policy_credential.json"
            credential_path.write_text(json.dumps(policy_credential))
            credential_path.chmod(0o600)
            inputs["policy_credential.json"] = credential_path
        return build_native_task_arena_bundle(job_dir=job_dir, packet_dir=packet_dir,
            worker_source=package / "controlled_native_policy_worker.py",
            runtime_module_sources=[package / name for name in RUNTIME_MODULE_NAMES],
            runtime_source_packet_receipt=runtime_source_packet_receipt, implementation_commit=implementation_commit,
            execution_mode="controlled_policy", expected_output_filename=RESULT_FILENAME,
            bound_runtime_inputs=inputs, runtime_extra_files=extras, **kwargs)


def load_verified_controlled_native_policy_bundle(receipt_path, **kwargs):
    return load_verified_native_task_arena_construction_bundle(receipt_path,
        expected_execution_mode="controlled_policy", **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Rebuild operator-owned inputs at an exact release; never allocate."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job-dir", "packet-dir", "runtime-source-packet-receipt",
                 "configuration", "job-request", "observations"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--policy-credential", type=Path)
    args = parser.parse_args(argv)
    try:
        credential = None
        if args.policy_credential:
            if args.policy_credential.is_symlink() or args.policy_credential.stat().st_mode & 0o077:
                raise ValueError("controlled_native_policy_credential_not_private")
            credential = json.loads(args.policy_credential.read_text())
        receipt = build_controlled_native_policy_bundle(
            job_dir=args.job_dir, packet_dir=args.packet_dir,
            runtime_source_packet_receipt=args.runtime_source_packet_receipt,
            implementation_commit=args.implementation_commit,
            configuration=json.loads(args.configuration.read_text()),
            job_request=json.loads(args.job_request.read_text()),
            observations=json.loads(args.observations.read_text()), policy_credential=credential)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error_class": type(exc).__name__,
                          "provider_mutation_performed": False}))
        return 2
    print(json.dumps({key: receipt.get(key) for key in
                     ("status", "bundle_path", "bundle_sha256", "implementation_commit")}))
    return 0 if receipt.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
