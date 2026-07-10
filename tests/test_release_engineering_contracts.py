from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_warehouse_fixture_inputs_are_not_ignored_and_validate() -> None:
    for relative in (
        "tests/fixtures/warehouse_task_min/pipeline/evaluation_prep/task_anchor_manifest.json",
        "tests/fixtures/warehouse_task_min/pipeline/geometry/camera/intrinsics.json",
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative],
            cwd=ROOT,
            check=False,
        )
        assert ignored.returncode == 1, f"release fixture is ignored: {relative}"

    completed = _run_script("verify_clean_checkout_inputs.py", "--root", str(ROOT))
    assert completed.returncode == 0, completed.stderr
    assert "[clean-checkout-inputs] ok" in completed.stdout


def test_uv_lock_and_frozen_exports_are_release_contracts() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "uv.lock"],
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 1

    completed = _run_script("verify_dependency_exports.py")
    assert completed.returncode == 0, completed.stderr
    assert "[dependency-exports] ok" in completed.stdout

    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        if "uv sync" in text:
            assert "uv lock --check" in text, workflow
            assert "uv sync --frozen" in text, workflow

    for workflow_name in ("sim-only-local-gate.yml", "python-compatibility.yml"):
        workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        assert '"uv.lock"' in workflow

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependency_gate = (ROOT / "scripts" / "run_dependency_security_gate.py").read_text(
        encoding="utf-8"
    )
    assert pyproject.count('"pip-audit==2.10.1"') == 2
    assert '"uv", "run", "--frozen", "pip-audit"' in dependency_gate
    assert '"uvx"' not in dependency_gate


def test_full_lane_has_no_free_form_test_reduction_input() -> None:
    workflow = (ROOT / ".github" / "workflows" / "full-test-lane.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "pytest_args" not in workflow
    assert "inputs.pytest" not in workflow
    assert "extra_args" not in workflow
    assert "uv run scripts/pytest_full.sh" in workflow
    assert "--junitxml=output/ci/full-test-lane-junit.xml" in workflow
    assert "blueprint_pipeline.pytest_full_lane_evidence" in workflow
    assert "scripts/verify_full_lane_collection.py" in workflow
    assert "full-test-lane-planned.json" in workflow
    assert "full-test-lane-executed.json" in workflow


def test_full_lane_collection_verifier_requires_exact_nodeids(tmp_path: Path) -> None:
    nodeids = ["tests/test_one.py::test_a", "tests/test_two.py::test_b"]

    def write_manifest(path: Path, *, phase: str, values: list[str]) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "blueprint_full_lane_collection.v1",
                    "phase": phase,
                    "test_count": len(values),
                    "nodeids_sha256": hashlib.sha256("\n".join(values).encode()).hexdigest(),
                    "nodeids": values,
                }
            ),
            encoding="utf-8",
        )

    planned = tmp_path / "planned.json"
    executed = tmp_path / "executed.json"
    write_manifest(planned, phase="planned", values=nodeids)
    write_manifest(executed, phase="executed", values=nodeids)

    passed = _run_script(
        "verify_full_lane_collection.py",
        "--planned",
        str(planned),
        "--executed",
        str(executed),
    )
    assert passed.returncode == 0, passed.stderr

    write_manifest(executed, phase="executed", values=nodeids[:-1])
    failed = _run_script(
        "verify_full_lane_collection.py",
        "--planned",
        str(planned),
        "--executed",
        str(executed),
    )
    assert failed.returncode == 1
    assert "planned_executed_nodeids" in failed.stderr


def test_package_policy_and_spdx_metadata_are_complete() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert "License :: OSI Approved :: MIT License" not in project["classifiers"]
    assert set(project["urls"]) >= {
        "Homepage",
        "Repository",
        "Issues",
        "Support",
        "Documentation",
        "Security",
        "Changelog",
    }
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert (ROOT / "SECURITY.md").is_file()
    assert (ROOT / ".github" / "CODEOWNERS").is_file()

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv build --out-dir output/ci/dist" in workflow
    assert "scripts/verify_distribution_metadata.py" in workflow


def test_github_actions_are_pinned_to_full_commit_shas() -> None:
    uses_pattern = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)
    sha_pattern = re.compile(r"^[0-9a-f]{40}$")
    discovered: list[str] = []
    for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = workflow_path.read_text(encoding="utf-8")
        for action in uses_pattern.findall(workflow):
            discovered.append(action)
            assert "@" in action, (workflow_path, action)
            _repository, revision = action.rsplit("@", 1)
            assert sha_pattern.fullmatch(revision), (workflow_path, action)
    assert discovered


def test_primary_container_and_compose_contracts_are_hardened() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "python:3.12-slim@sha256:" in dockerfile
    assert "FROM python:3.11" not in dockerfile
    assert "COPY pyproject.toml uv.lock README.md LICENSE" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "AS development" in dockerfile
    assert "USER blueprint:blueprint" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "DINOV3_MODEL_REVISION=ea8dc2863c51be0a264bab82070e3e8836b02d51" in dockerfile
    assert "revision=revision" in dockerfile

    assert "target: base" not in compose
    assert "target: development" in compose
    assert "target: production" in compose
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose
    assert "jupyter" not in compose.lower()
    assert "--allow-root" not in compose
    assert "0.0.0.0" not in compose

    completed = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_every_container_base_and_remote_source_is_immutable() -> None:
    dockerfiles = sorted((ROOT / "deploy" / "docker").rglob("Dockerfile"))
    dockerfiles.append(ROOT / "Dockerfile")
    digest_pattern = re.compile(r"@sha256:[0-9a-f]{64}$")

    for dockerfile_path in dockerfiles:
        text = dockerfile_path.read_text(encoding="utf-8")
        defaults = dict(
            re.findall(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)$", text, re.MULTILINE)
        )
        from_lines = re.findall(r"^FROM\s+(.+)$", text, re.MULTILINE)
        stage_names = {
            match.group(1)
            for from_line in from_lines
            if (match := re.search(r"\s+AS\s+([A-Za-z0-9_.-]+)$", from_line, re.IGNORECASE))
        }
        for from_line in from_lines:
            image = next(
                token for token in from_line.split() if not token.startswith("--platform=")
            )
            if image in stage_names:
                continue
            variable = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", image)
            resolved = defaults.get(variable.group(1), "") if variable else image
            assert digest_pattern.search(resolved), (dockerfile_path, from_line, resolved)

    deepprivacy = (ROOT / "deploy/docker/deepprivacy2/Dockerfile").read_text(
        encoding="utf-8"
    )
    video_to_world = (ROOT / "deploy/docker/video_to_world/Dockerfile").read_text(
        encoding="utf-8"
    )
    groot_oscar = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")
    groot_wam = (
        ROOT / "deploy/docker/robot_eval_worker/unitree_groot_sonic_wam/Dockerfile"
    ).read_text(encoding="utf-8")

    assert "DEEPPRIVACY2_SOURCE_REF=" in deepprivacy
    assert 'checkout --detach FETCH_HEAD' in deepprivacy
    assert "git clone --branch main" not in video_to_world
    for source_ref in (
        "VIDEO_TO_WORLD_SOURCE_REF",
        "DEPTH_ANYTHING_3_SOURCE_REF",
        "ROMAV2_SOURCE_REF",
    ):
        assert re.search(rf"ARG {source_ref}=[0-9a-f]{{40}}", video_to_world)
        assert f'origin "${{{source_ref}}}"' in video_to_world
    assert re.search(r"ARG WBC_SOURCE_REF=[0-9a-f]{40}", groot_oscar)
    assert 'revision=os.environ["SONIC_CHECKPOINT_REVISION"]' in groot_oscar
    assert 'revision=os.environ["GROOT_CHECKPOINT_REVISION"]' in groot_wam


def test_dependabot_covers_every_dockerfile_directory() -> None:
    config = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    dockerfiles = [ROOT / "Dockerfile"]
    dockerfiles.extend(sorted((ROOT / "deploy" / "docker").rglob("Dockerfile")))
    dockerfile_directories = {
        f"/{path.parent.relative_to(ROOT).as_posix()}"
        if path.parent != ROOT
        else "/"
        for path in dockerfiles
    }

    for directory in sorted(dockerfile_directories):
        assert re.search(
            rf"^\s*(?:directory:\s*{re.escape(directory)}|-\s*{re.escape(directory)})\s*$",
            config,
            re.MULTILINE,
        ), directory


def test_package_import_is_lazy_and_module_entrypoints_are_not_preloaded() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys,blueprint_pipeline; "
                "targets=['lightwheel_kitchen_isaac_scenarios',"
                "'oscar_wam_command_adapter','oscar_cosmos_wam_command_adapter',"
                "'robot_eval_job_orchestrator']; "
                "print(json.dumps([name for name in targets if "
                "'blueprint_pipeline.'+name in sys.modules]))"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONWARNINGS": "error::RuntimeWarning"},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
