"""Execute and bind the bounded ADP-009A NVIDIA Content Agents preflight.

This seam performs real CLI dry-runs for Material, Texture, and Physics Agent,
and a real rules-only Validation Agent check. It deliberately cannot promote a
dry-run, source checkout, or static validation into a generated asset or a
dynamic physics result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009a_usd_content_agents_preflight_receipt.v1"
REQUEST_SCHEMA_VERSION = "adp009a_usd_content_agents_preflight_request.v1"
PROGRAM_ID = "arm-decision-proof-v1"
ADP_ITEM = "ADP-009A"
CLAIM_CEILING = "development_only"


class ContentAgentsPreflightError(ValueError):
    """The observed preflight evidence did not satisfy its frozen checks."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_under(path: Path, roots: Sequence[Path]) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ContentAgentsPreflightError(f"path_outside_approved_roots:{resolved}")
    return resolved


def _rooted(root: Path, value: str) -> Path:
    if not value or Path(value).is_absolute():
        raise ContentAgentsPreflightError("paths_must_be_nonempty_and_relative")
    return _require_under(root / value, (root,))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContentAgentsPreflightError(f"not_json_object:{path.name}")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _record(path: Path, *, root: Path, role: str) -> dict[str, Any]:
    _require_under(path, (root,))
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContentAgentsPreflightError(f"missing_or_empty_artifact:{role}")
    return {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _container_path(relative: str) -> str:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ContentAgentsPreflightError("container_path_invalid")
    return "/adp/" + Path(relative).as_posix()


def _execute_and_log(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = _run(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise ContentAgentsPreflightError(
            f"content_agent_command_failed:{Path(command[command.index('--entrypoint') + 1]).name}"
        )
    return result


def _docker_command(
    *, docker: str, image: str, data_root: Path, entrypoint: str, arguments: Sequence[str]
) -> list[str]:
    return [
        docker,
        "run",
        "--rm",
        "--platform",
        "linux/arm64",
        "-v",
        f"{data_root}:/adp",
        "--entrypoint",
        entrypoint,
        image,
        *arguments,
    ]


def _verify_validation_result(result: Mapping[str, Any]) -> dict[str, Any]:
    templates = result.get("template_results")
    if result.get("verdict") != "pass" or not isinstance(templates, list):
        raise ContentAgentsPreflightError("validation_agent_static_check_failed")
    physics = next(
        (row for row in templates if isinstance(row, Mapping) and row.get("template_name") == "physics_sane"),
        None,
    )
    if not isinstance(physics, Mapping) or physics.get("status") != "passed":
        raise ContentAgentsPreflightError("validation_agent_physics_sane_missing")
    metrics = physics.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ContentAgentsPreflightError("validation_agent_metrics_missing")
    required_counts = {
        "physics_scene_count": 1,
        "rigid_body_count": 1,
        "collider_count": 1,
        "mass_api_count": 1,
        "material_api_count": 1,
    }
    for key, minimum in required_counts.items():
        if int(metrics.get(key, 0)) < minimum:
            raise ContentAgentsPreflightError(f"validation_agent_metric_missing:{key}")
    metadata = result.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("dry_run") is not False:
        raise ContentAgentsPreflightError("validation_agent_execution_not_observed")
    return {key: int(metrics[key]) for key in required_counts}


def materialize_content_agents_preflight(
    *,
    request_path: Path,
    repo_root: Path,
    data_root: Path,
    content_agents_root: Path,
    receipt_output: Path,
    docker: str = "docker",
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    content_agents_root = content_agents_root.resolve()
    request_path = _require_under(request_path, (repo_root,))
    receipt_output = _require_under(receipt_output, (repo_root,))
    request = _read_json(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ContentAgentsPreflightError("request_schema_invalid")
    forbidden = {"status", "admitted", "qualified", "passed"}.intersection(request)
    if forbidden:
        raise ContentAgentsPreflightError("caller_asserted_status_forbidden")

    source = request["source"]
    expected_commit = str(source["commit"])
    expected_tree = str(source["tree"])
    head = _git(content_agents_root, "rev-parse", "HEAD")
    tree = _git(content_agents_root, "rev-parse", "HEAD^{tree}")
    dirty = bool(_git(content_agents_root, "status", "--porcelain"))
    if head != expected_commit or tree != expected_tree or dirty:
        raise ContentAgentsPreflightError("content_agents_source_identity_mismatch")

    image = request["image"]
    image_ref = str(image["reference"])
    inspect_result = _run([docker, "image", "inspect", image_ref])
    if inspect_result.returncode != 0:
        raise ContentAgentsPreflightError("content_agents_container_image_missing")
    inspected = json.loads(inspect_result.stdout)
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise ContentAgentsPreflightError("content_agents_container_inspect_invalid")
    image_data = inspected[0]
    repo_digests = image_data.get("RepoDigests") or []
    expected_digest = str(image["digest"])
    if not any(str(value).endswith("@" + expected_digest) for value in repo_digests):
        raise ContentAgentsPreflightError("content_agents_container_digest_mismatch")
    observed_platform = f"{image_data.get('Os')}/{image_data.get('Architecture')}"
    if observed_platform != str(image["platform"]):
        raise ContentAgentsPreflightError("content_agents_container_platform_mismatch")

    native_log = _rooted(data_root, str(request["native_install_failure_log"]))
    native_text = native_log.read_text(encoding="utf-8")
    if "usd-exchange==2.3.0" not in native_text or "macosx" not in native_text:
        raise ContentAgentsPreflightError("native_platform_failure_receipt_invalid")

    artifacts: list[dict[str, Any]] = [
        _record(
            _rooted(repo_root, str(request["dockerfile_path"])),
            root=repo_root,
            role="linux_arm64_container_recipe",
        ),
        _record(native_log, root=data_root, role="native_macos_install_failure"),
        _record(
            _rooted(data_root, str(request["installed_packages_path"])),
            root=data_root,
            role="installed_dependency_freeze",
        ),
        _record(
            _rooted(content_agents_root, str(source["license_path"])),
            root=content_agents_root,
            role="source_license",
        ),
    ]
    for skill_path in source["skill_paths"]:
        artifacts.append(
            _record(
                _rooted(content_agents_root, str(skill_path)),
                root=content_agents_root,
                role="agent_skill_contract",
            )
        )

    versions: dict[str, str] = {}
    for name in ("material-agent", "texture-agent", "physics-agent", "validation-agent"):
        result = _run(
            _docker_command(
                docker=docker,
                image=image_ref,
                data_root=data_root,
                entrypoint=name,
                arguments=("--version",),
            )
        )
        if result.returncode != 0 or str(source["version"]) not in result.stdout:
            raise ContentAgentsPreflightError(f"content_agent_cli_version_mismatch:{name}")
        versions[name] = result.stdout.strip()

    agents: dict[str, Any] = {}
    markers = {
        "material": "Dry run complete",
        "texture": "Dry run -- execution plan",
        "physics": "Dry run complete",
    }
    entrypoints = {
        "material": "material-agent",
        "texture": "texture-agent",
        "physics": "physics-agent",
    }
    for agent_name in ("material", "texture", "physics"):
        spec = request["agents"][agent_name]
        config_path = _rooted(data_root, str(spec["config_path"]))
        log_path = _rooted(data_root, str(spec["dry_run_log_path"]))
        command = _docker_command(
            docker=docker,
            image=image_ref,
            data_root=data_root,
            entrypoint=entrypoints[agent_name],
            arguments=("run", _container_path(str(spec["config_path"])), "--dry-run"),
        )
        result = _execute_and_log(command, log_path)
        combined = result.stdout + result.stderr
        if markers[agent_name] not in combined:
            raise ContentAgentsPreflightError(f"content_agent_dry_run_marker_missing:{agent_name}")
        artifacts.extend(
            (
                _record(config_path, root=data_root, role=f"{agent_name}_agent_config"),
                _record(log_path, root=data_root, role=f"{agent_name}_agent_dry_run_log"),
            )
        )
        agents[agent_name] = {
            "cli": entrypoints[agent_name],
            "version": versions[entrypoints[agent_name]],
            "dry_run_executed": True,
            "full_agent_executed": False,
            "command": command,
            "smallest_blocker": str(spec["smallest_blocker"]),
        }

    validation = request["agents"]["validation"]
    stage_path = _rooted(data_root, str(validation["stage_path"]))
    stage_receipt_path = _rooted(data_root, str(validation["stage_receipt_path"]))
    stage_receipt = _read_json(stage_receipt_path)
    if stage_receipt.get("receipt_digest") != canonical_digest(
        stage_receipt, digest_field="receipt_digest"
    ):
        raise ContentAgentsPreflightError("validation_stage_receipt_digest_mismatch")
    output_dir = _rooted(data_root, str(validation["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_log = _rooted(data_root, str(validation["log_path"]))
    validation_command = _docker_command(
        docker=docker,
        image=image_ref,
        data_root=data_root,
        entrypoint="validation-agent",
        arguments=(
            "validate",
            _container_path(str(validation["stage_path"])),
            "--task",
            str(validation["task"]),
            "--template",
            "physics_sane",
            "--output-dir",
            _container_path(str(validation["output_dir"])),
            "--format",
            "json",
        ),
    )
    _execute_and_log(validation_command, validation_log)
    result_path = output_dir / "validation_result.json"
    metrics = _verify_validation_result(_read_json(result_path))
    artifacts.extend(
        (
            _record(stage_path, root=data_root, role="validation_context_stage"),
            _record(stage_receipt_path, root=data_root, role="validation_context_receipt"),
            _record(validation_log, root=data_root, role="validation_agent_execution_log"),
            _record(output_dir / "validation_request.json", root=data_root, role="validation_request"),
            _record(output_dir / "validation_plan.json", root=data_root, role="validation_plan"),
            _record(result_path, root=data_root, role="validation_result"),
        )
    )
    agents["validation"] = {
        "cli": "validation-agent",
        "version": versions["validation-agent"],
        "executed": True,
        "dry_run": False,
        "template": "physics_sane",
        "verdict": "pass",
        "metrics": metrics,
        "command": validation_command,
    }
    agents["joint"] = {
        "applicable": False,
        "reason": "selected_target_is_one_rigid_body_with_zero_articulated_joints",
        "executed": False,
    }

    blockers = sorted(
        {
            str(request["agents"][name]["smallest_blocker"])
            for name in ("material", "texture", "physics")
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "source": {
            "repository": str(source["repository"]),
            "commit": head,
            "tree": tree,
            "clean": not dirty,
            "version": str(source["version"]),
            "license": str(source["license"]),
        },
        "runtime": {
            "image_reference": image_ref,
            "image_digest": expected_digest,
            "image_id": str(image_data.get("Id")),
            "platform": observed_platform,
            "paid_resource_allocated": False,
            "model_or_remote_renderer_called": False,
        },
        "agents": agents,
        "artifacts": artifacts,
        "status": "prepared_static_validation_passed",
        "blockers": blockers,
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": {
            "dry_runs_are_not_agent_execution": True,
            "static_validation_is_not_dynamic_simulation": True,
            "static_validation_is_not_material_or_texture_prediction": True,
            "no_inpainting_result": True,
            "no_physical_evidence": True,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--content-agents-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_content_agents_preflight(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        content_agents_root=args.content_agents_root,
        receipt_output=args.receipt_output,
    )
    print(json.dumps({"receipt_digest": receipt["receipt_digest"], "status": receipt["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
