#!/usr/bin/env python3
"""Build the released Content Agents adapter used by every configured scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from blueprint_pipeline.adp_content_agents_vast import (
    CONTENT_IMAGE_MODEL,
    CONTENT_LLM_MODEL,
    SOURCE_COMMIT,
    SOURCE_TREE,
    SOURCE_VERSION,
)
from blueprint_pipeline.content_agents_model_compatibility import (
    materialize_content_agents_model_compatibility_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.production_cad_skill_sources import SOURCE_SPECS
from scripts.build_task_evaluation_scene_configuration_component_package import (
    build_scene_configuration_component_package,
)


_EXECUTABLE_SOURCES = {
    "scripts/run_adp_content_agents_provider_runtime.sh",
    "scripts/run_task_evaluation_scene_configuration_content_agents_component.sh",
}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("scene_configuration_content_agents_source_invalid") from exc


def build_content_agents_scene_configuration_component(
    *,
    repository_root: str | Path,
    expected_blueprint_commit: str,
    content_agents_root: str | Path,
    text_to_cad_root: str | Path,
    multi_agent_cad_root: str | Path,
    output_root: str | Path,
) -> dict:
    """Seal exact released source plus the existing Blueprint runtime adapter."""

    repository = Path(repository_root).expanduser().resolve()
    upstream = Path(content_agents_root).expanduser().resolve()
    text_to_cad = Path(text_to_cad_root).expanduser().resolve()
    multi_agent_cad = Path(multi_agent_cad_root).expanduser().resolve()
    if (
        _git(repository, "rev-parse", "HEAD") != expected_blueprint_commit
        or _git(repository, "status", "--porcelain=v1")
    ):
        raise ValueError("scene_configuration_content_agents_blueprint_source_invalid")
    if (
        _git(upstream, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(upstream, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(upstream, "status", "--porcelain=v1")
    ):
        raise ValueError("scene_configuration_content_agents_source_invalid")
    cad_specs = {str(spec["id"]): spec for spec in SOURCE_SPECS}
    for source, source_id in (
        (text_to_cad, "text-to-cad"),
        (multi_agent_cad, "multi-agent-cad"),
    ):
        spec = cad_specs[source_id]
        if (
            _git(source, "rev-parse", "HEAD") != spec["commit"]
            or _git(source, "rev-parse", "HEAD^{tree}") != spec["tree"]
            or _git(source, "status", "--porcelain=v1")
            or _sha256(source / "LICENSE") != spec["license_sha256"]
        ):
            raise ValueError("scene_configuration_cad_skill_source_invalid")
    staging = Path(tempfile.mkdtemp(prefix="content-agents-scene-configuration-"))
    try:
        archive = staging / "content_agents_source.zip"
        _git(
            upstream,
            "archive",
            "--format=zip",
            f"--output={archive}",
            "HEAD",
        )
        text_to_cad_archive = staging / "text_to_cad_skills_source.zip"
        _git(
            text_to_cad,
            "archive",
            "--format=zip",
            f"--output={text_to_cad_archive}",
            "HEAD",
            "LICENSE",
            *[f"skills/{name}" for name in cad_specs["text-to-cad"]["skills"]],
            "packages/cadpy",
            "packages/cadpy_metadata",
        )
        multi_agent_cad_archive = staging / "multi_agent_cad_source.zip"
        _git(
            multi_agent_cad,
            "archive",
            "--format=zip",
            f"--output={multi_agent_cad_archive}",
            "HEAD",
        )
        cad_source_receipt = {
            "schema_version": "task_evaluation_cad_skill_component_source.v1",
            "status": "pinned_sources_packaged",
            "sources": [
                {
                    "id": source_id,
                    "repository": cad_specs[source_id]["repository"],
                    "commit": cad_specs[source_id]["commit"],
                    "tree": cad_specs[source_id]["tree"],
                    "license": cad_specs[source_id]["license"],
                    "license_sha256": cad_specs[source_id]["license_sha256"],
                    "skills": list(cad_specs[source_id]["skills"]),
                    "archive_sha256": _sha256(
                        text_to_cad_archive
                        if source_id == "text-to-cad"
                        else multi_agent_cad_archive
                    ),
                }
                for source_id in ("text-to-cad", "multi-agent-cad")
            ],
            "skill_count": 10,
            "scene_specific_source": False,
            "receipt_digest": "",
        }
        cad_source_receipt["receipt_digest"] = canonical_digest(
            cad_source_receipt, digest_field="receipt_digest"
        )
        (staging / "cad_skill_source_receipt.json").write_text(
            json.dumps(cad_source_receipt, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        source_receipt = {
            "schema_version": "task_evaluation_content_agents_component_source.v1",
            "repository": (
                "https://github.com/NVIDIA-Omniverse/usd-content-agents"
            ),
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
            "version": SOURCE_VERSION,
            "license": "Apache-2.0",
            "archive_sha256": _sha256(archive),
            "receipt_digest": "",
        }
        source_receipt["receipt_digest"] = canonical_digest(
            source_receipt, digest_field="receipt_digest"
        )
        (staging / "content_agents_source_receipt.json").write_text(
            json.dumps(source_receipt, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        copies = {
            "scripts/run_task_evaluation_scene_configuration_content_agents_component.sh": "run",
            "scripts/run_adp_content_agents_provider_runtime.sh": (
                "run_adp_content_agents_provider_runtime.sh"
            ),
            "scripts/adp_content_agents_provider_runner.py": (
                "adp_content_agents_provider_runner.py"
            ),
            "src/blueprint_pipeline/provider_archive.py": "provider_archive.py",
            "src/blueprint_pipeline/content_agents_model_compatibility.py": (
                "content_agents_model_compatibility.py"
            ),
            "src/blueprint_pipeline/production_cad_skill_sources.py": (
                "production_cad_skill_sources.py"
            ),
            "skillpacks/cad_authoring/skills/multi-agent-cad/SKILL.md": (
                "multi_agent_cad_skill.md"
            ),
            "docs/arm_decision_proof_v1/assets/adp009a_content_agents_material.vast.yaml": (
                "material_agent.yaml"
            ),
            "docs/arm_decision_proof_v1/assets/adp009a_content_agents_texture.vast.yaml": (
                "texture_agent.yaml"
            ),
            "docs/arm_decision_proof_v1/assets/adp009a_content_agents_physics.vast.yaml": (
                "physics_agent.yaml"
            ),
        }
        for source_name, destination_name in copies.items():
            source = repository / source_name
            destination = staging / destination_name
            if source.is_symlink() or not source.is_file():
                raise ValueError(
                    "scene_configuration_content_agents_blueprint_source_invalid"
                )
            shutil.copyfile(source, destination)
            executable = source_name in _EXECUTABLE_SOURCES
            destination.chmod(0o755 if executable else 0o644)
            if bool(destination.stat().st_mode & 0o111) is not executable:
                raise ValueError(
                    "scene_configuration_content_agents_blueprint_source_invalid"
                )
        materialize_content_agents_model_compatibility_plan(
            model_ids=(CONTENT_LLM_MODEL, CONTENT_IMAGE_MODEL),
            destination=staging
            / "content_agents_model_compatibility_plan.json",
        )
        return build_scene_configuration_component_package(
            adapter_id="content_agents_rigid_replacement",
            source_root=staging,
            driver_entrypoint="run",
            source_repository="https://github.com/ognjhunt/BlueprintCapturePipeline",
            source_commit=expected_blueprint_commit,
            source_license=(
                "Blueprint adapter; bundled upstream Apache-2.0 and MIT CAD skills"
            ),
            output_root=output_root,
        )
    finally:
        shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-blueprint-commit", required=True)
    parser.add_argument("--content-agents-root", required=True)
    parser.add_argument("--text-to-cad-root", required=True)
    parser.add_argument("--multi-agent-cad-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    value = build_content_agents_scene_configuration_component(
        repository_root=args.repository_root,
        expected_blueprint_commit=args.expected_blueprint_commit,
        content_agents_root=args.content_agents_root,
        text_to_cad_root=args.text_to_cad_root,
        multi_agent_cad_root=args.multi_agent_cad_root,
        output_root=args.output_root,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
