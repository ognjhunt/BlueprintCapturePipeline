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
        workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
        assert '"uv.lock"' in workflow

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependency_gate = (ROOT / "scripts" / "run_dependency_security_gate.py").read_text(
        encoding="utf-8"
    )
    assert pyproject.count('"pip-audit==2.10.1"') == 2
    assert '"uv", "run", "--frozen", "pip-audit"' in dependency_gate
    assert '"uvx"' not in dependency_gate


def test_full_lane_has_no_free_form_test_reduction_input() -> None:
    workflow = (ROOT / ".github" / "workflows" / "full-test-lane.yml").read_text(encoding="utf-8")
    event_block = workflow.split("jobs:", 1)[0]

    assert "workflow_dispatch:" in workflow
    assert "workflow_call:" in event_block
    assert "pull_request:" not in event_block
    assert "push:" not in event_block
    assert 'cron: "17 8 * * *"' in event_block
    assert "production_deployment_promotion" in event_block
    assert "cross_cutting_diagnostic" in event_block
    assert "cancel-in-progress: true" in workflow
    assert "pytest_args" not in workflow
    assert "inputs.pytest" not in workflow
    assert "extra_args" not in workflow
    assert "uv run scripts/pytest_full.sh" in workflow
    assert '--junitxml="${{ runner.temp }}/blueprint-ci/full-test-lane-junit.xml"' in workflow
    assert "blueprint_pipeline.pytest_full_lane_evidence" in workflow
    assert "scripts/verify_full_lane_collection.py" in workflow
    assert "full-test-lane-planned.json" in workflow
    assert "full-test-lane-executed.json" in workflow


def test_risk_based_verification_workflows_are_bounded() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "name: Determine changed test surface" in workflow
    assert "name: Impacted tests and sentinels" in workflow
    assert "name: Impacted test and sentinel gate" in workflow
    assert "blueprint_pipeline.impacted_test_selection" in workflow
    assert "--timeout-seconds 480" in workflow
    assert "timeout-minutes: 10" in workflow
    assert "needs.cross-cutting-full-suite.result" in workflow
    assert "uses: ./.github/workflows/full-test-lane.yml" in workflow
    assert "needs.impact.outputs.requires_full_suite == 'true'" in workflow
    assert "uv run pytest -q" not in workflow
    assert "--cov-fail-under" not in workflow

    for job in (
        "lint",
        "foundation-prerequisites",
        "typecheck",
        "sast",
        "source-governance",
        "supply-chain",
        "dependency-security",
        "container-contract",
    ):
        match = re.search(rf"(?ms)^  {re.escape(job)}:\n(.*?)(?=^  \S|\Z)", workflow)
        assert match is not None, job
        job_block = match.group(1)
        assert "if: github.event_name == 'push'" in job_block, job


def test_core_workflows_bind_runner_temp_only_after_job_start() -> None:
    for workflow_name in ("ci.yml", "sim-only-local-gate.yml", "full-test-lane.yml"):
        workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        before_first_step = workflow.split("steps:", 1)[0]
        assert "${{ runner.temp }}" not in before_first_step, workflow_name
        assert (
            'BLUEPRINT_ARTIFACT_CACHE_ROOT=${RUNNER_TEMP}/blueprint-artifact-cache'
            in workflow
        ), workflow_name
        assert 'BLUEPRINT_EVIDENCE_ROOT=${RUNNER_TEMP}/blueprint-evidence' in workflow, workflow_name
        assert '>> "${GITHUB_ENV}"' in workflow, workflow_name


def test_only_bounded_ci_workflow_runs_for_ordinary_pull_requests() -> None:
    for workflow_name in ("ci.yml",):
        workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        event_block = workflow.split("jobs:", 1)[0]
        assert "pull_request:" in event_block, workflow_name
        assert 'branches: ["main"]' in event_block, workflow_name
        assert 'branches: ["**"]' not in event_block, workflow_name
        assert "cancel-in-progress: true" in event_block, workflow_name

    for workflow_name in (
        "codeql.yml",
        "full-test-lane.yml",
        "python-compatibility.yml",
        "sim-only-local-gate.yml",
    ):
        workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        event_block = workflow.split("jobs:", 1)[0]
        assert "pull_request:" not in event_block, workflow_name


def test_groot_oscar_disk_admission_measures_every_write_filesystem() -> None:
    script = (ROOT / "scripts" / "build_push_groot_oscar_closed_loop_image.sh").read_text(
        encoding="utf-8"
    )

    assert "docker info --format '{{.DockerRootDir}}'" in script
    assert 'build_temp_root="${TMPDIR:-/tmp}"' in script
    assert '("source_and_evidence", "docker_buildkit", "build_and_scan_temp")' in script
    assert "shutil.disk_usage(path).free" in script
    assert 'evidence["limiting_storage_path"]' in script


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
    assert 'uv build --out-dir "${{ runner.temp }}/blueprint-ci/dist"' in workflow
    assert "scripts/verify_distribution_metadata.py" in workflow


def test_github_actions_are_pinned_to_full_commit_shas() -> None:
    uses_pattern = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)
    sha_pattern = re.compile(r"^[0-9a-f]{40}$")
    discovered: list[str] = []
    for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = workflow_path.read_text(encoding="utf-8")
        for action in uses_pattern.findall(workflow):
            discovered.append(action)
            if action.startswith("./"):
                continue
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
    assert "ARG BLUEPRINT_SOURCE_COMMIT" in dockerfile
    assert 'BLUEPRINT_SOURCE_COMMIT="${BLUEPRINT_SOURCE_COMMIT}"' in dockerfile
    assert 'org.opencontainers.image.revision="${BLUEPRINT_SOURCE_COMMIT}"' in dockerfile
    assert "DINOV3_MODEL_REVISION" not in dockerfile
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" not in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "revision=revision" in dockerfile

    deploy_script = (ROOT / "deploy" / "scripts" / "deploy.sh").read_text(encoding="utf-8")
    assert '--build-arg "BLUEPRINT_SOURCE_COMMIT=${GIT_SHA}"' in deploy_script

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
        defaults = dict(re.findall(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)$", text, re.MULTILINE))
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

    deepprivacy = (ROOT / "deploy/docker/deepprivacy2/Dockerfile").read_text(encoding="utf-8")
    video_to_world = (ROOT / "deploy/docker/video_to_world/Dockerfile").read_text(encoding="utf-8")
    groot_oscar = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")
    groot_wam = (
        ROOT / "deploy/docker/robot_eval_worker/unitree_groot_sonic_wam/Dockerfile"
    ).read_text(encoding="utf-8")

    assert "DEEPPRIVACY2_SOURCE_REF=" in deepprivacy
    assert "checkout --detach FETCH_HEAD" in deepprivacy
    assert "git clone --branch main" not in video_to_world
    for source_ref in (
        "VIDEO_TO_WORLD_SOURCE_REF",
        "DEPTH_ANYTHING_3_SOURCE_REF",
        "ROMAV2_SOURCE_REF",
    ):
        assert re.search(rf"ARG {source_ref}=[0-9a-f]{{40}}", video_to_world)
        assert f'origin "${{{source_ref}}}"' in video_to_world
    assert re.search(r"ARG WBC_SOURCE_REF=[0-9a-f]{40}", groot_oscar)
    assert "nvcr.io/nvidia/isaac-sim:6.0.0@sha256:" in groot_oscar
    assert "BLUEPRINT_WORKER_IMAGE_FAMILY=isaac-eval-worker" in groot_oscar
    assert "/isaac-sim/python.sh -m pip install" in groot_oscar
    assert "uv venv /opt/oscar-venv --python 3.10" in groot_oscar
    assert "git -C /opt/wbc lfs pull" in groot_oscar
    assert "TENSORRT_VERSION=10.4.0.26-1+cuda12.6" in groot_oscar
    assert "TensorRT_ROOT=/usr" in groot_oscar
    assert "libnvinfer-dev=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvinfer-headers-dev=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvinfer10=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvinfer-plugin-dev=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvinfer-plugin10=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvonnxparsers-dev=${TENSORRT_VERSION}" in groot_oscar
    assert "libnvonnxparsers10=${TENSORRT_VERSION}" in groot_oscar
    assert "nvinfer nvinfer_plugin nvonnxparser nvparsers" in groot_oscar
    assert "nvinfer nvinfer_plugin nvonnxparser/" in groot_oscar
    assert 'revision=os.environ["SONIC_CHECKPOINT_REVISION"]' in groot_oscar
    assert 'revision=os.environ["GROOT_CHECKPOINT_REVISION"]' in groot_wam


def test_groot_oscar_foundation_enables_and_pins_tensorrt_repository() -> None:
    foundation = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile"
    ).read_text(encoding="utf-8")
    assert "FROM ${ISAAC_SIM_BASE_IMAGE} AS tensorrt-base" in foundation
    assert (
        "ADD --checksum=sha256:d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba"
        in foundation
    )
    assert (
        "developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
        in foundation
    )
    assert "FROM tensorrt-base AS wbc-builder" in foundation
    wbc_builder = foundation.split("FROM tensorrt-base AS wbc-builder", 1)[1].split(
        "FROM tensorrt-base", 1
    )[0]
    assert "ca-certificates sudo" in wbc_builder
    assert "sha256:a1bc93654f31669fd964ea3011a5e5e9676b9b6f8adcd762606e5140632ea72d" in wbc_builder
    assert "sha256:b072f989d6315ac0e22dcb4771b083c5156d974a3496ac3504c77f4062eb248e" in wbc_builder
    assert "cppzmq-dev" in wbc_builder
    assert "test ! -d third_party/cppzmq/.git" in wbc_builder
    assert foundation.count("apt-cache madison libnvinfer10") == 2
    assert foundation.count("'$3 == version { found=1 } END { exit !found }'") == 3
    assert foundation.count("libnvinfer10=${TENSORRT_VERSION}") == 2
    assert "ARG CUDA_CUDART_VERSION=12.6.77-1" in foundation
    assert "apt-cache madison cuda-cudart-12-6" in foundation
    assert "cuda-cudart-12-6=${CUDA_CUDART_VERSION}" in foundation
    assert "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8" in foundation
    assert "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.6" not in foundation
    assert "libnvinfer-plugin10=${TENSORRT_VERSION}" in foundation
    assert "libnvonnxparsers10=${TENSORRT_VERSION}" in foundation
    assert "uv venv /opt/oscar-venv --python 3.10 --seed" in foundation
    assert "uv venv /opt/gr00t-venv --python 3.10 --seed" in foundation
    assert "requirements_uv_bootstrap.txt" in foundation
    assert "--require-hashes -r /tmp/requirements_uv_bootstrap.txt" in foundation
    assert "/tmp/oscar/requirements_minimal.txt" in foundation
    assert "requirements_oscar_foundation.lock" in foundation
    assert "uv pip install --require-hashes" in foundation
    assert "uv sync --project /tmp/gr00t --active --no-dev --frozen" in foundation
    assert "Tag: cp36-cp36m-manylinux2010_x86_64" in foundation
    assert "Tag: py3-none-manylinux2010_x86_64" in foundation
    assert "ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python" in foundation
    assert (
        "COPY --from=robot-env-builder --chown=blueprint:blueprint /opt/uv-python /opt/uv-python"
    ) in foundation
    assert "/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu" in foundation
    assert "tee /tmp/g1_deploy_onnx_ref.ldd" in foundation
    assert foundation.count("/opt/oscar-venv/bin/python -m pip check") == 2
    assert foundation.count("/opt/gr00t-venv/bin/python -m pip check") == 2
    assert "/opt/oscar-venv/bin/python /opt/blueprint/fetch_pinned_isaac_assets.py" in foundation
    assert "cp -a build target g1 scripts reference" not in foundation
    assert "COPY --from=wbc-builder /opt/onnxruntime-runtime /opt/onnxruntime" in foundation
    assert "test ! -d /opt/wbc/gear_sonic_deploy/build" in foundation


def test_groot_oscar_release_restates_required_cuda_metadata() -> None:
    release = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile"
    ).read_text(encoding="utf-8")

    assert "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.8" in release
    assert "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.6" not in release


def test_groot_oscar_small_carrier_matches_foundation_runtime_link_surface() -> None:
    carrier = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Carrier.Dockerfile"
    ).read_text(encoding="utf-8")
    assert "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" in carrier
    assert "ARG TENSORRT_VERSION=10.4.0.26-1+cuda12.6" in carrier
    assert "ARG CUDA_CUDART_VERSION=12.6.77-1" in carrier
    assert "apt-cache madison libnvinfer10" in carrier
    assert "apt-cache madison cuda-cudart-12-6" in carrier
    assert "libnvinfer10=${TENSORRT_VERSION}" in carrier
    assert "libnvinfer-plugin10=${TENSORRT_VERSION}" in carrier
    assert "libnvonnxparsers10=${TENSORRT_VERSION}" in carrier
    assert "libosmesa6" in carrier
    assert "libnghttp2-14" in carrier
    assert "libyaml-cpp0.8" in carrier
    assert "libzmq5" in carrier
    assert "NVIDIA_DRIVER_CAPABILITIES=all" in carrier
    assert "VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json" in carrier
    assert "/usr/share/glvnd/egl_vendor.d/10_nvidia.json" in carrier
    assert "/etc/vulkan/icd.d/nvidia_icd.json" in carrier
    assert "PYTHONPATH=/opt/wbc:/opt/OSCAR" in carrier
    assert "/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu" in carrier


def test_oscar_foundation_lock_is_exact_and_hash_checked() -> None:
    lock = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_oscar_foundation.lock"
    ).read_text(encoding="utf-8")
    requirement_lines = [
        line for line in lock.splitlines() if re.match(r"^[a-zA-Z0-9_.-]+==", line)
    ]
    assert len(requirement_lines) == 121
    assert lock.count("--hash=sha256:") >= len(requirement_lines)
    assert "torch==2.10.0+cu128" in lock
    assert "torchvision==0.25.0+cu128" in lock
    assert "pytest==9.1.1" in lock
    assert "mujoco==" in lock
    assert "msgpack-numpy==0.4.8" in lock
    assert (
        "# blueprint-input-sha256 oscar-requirements-minimal "
        "6002a7c982b96435f995d765306785f4835e404ea41308d4864f59bc8e34d117" in lock
    )
    runtime_requirements = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_robot_runtime.txt"
    ).read_bytes()
    runtime_requirements_text = runtime_requirements.decode("utf-8")
    assert "torch==2.10.0+cu128" in runtime_requirements_text
    assert "torchvision==0.25.0+cu128" in runtime_requirements_text
    assert "pytest>=8.0.0" in runtime_requirements_text
    assert (
        "# blueprint-input-sha256 requirements-robot-runtime "
        + hashlib.sha256(runtime_requirements).hexdigest()
        in lock
    )
    runbook = (ROOT / "docs/runbooks/groot-oscar-thin-release.md").read_text(encoding="utf-8")
    assert (
        "cp deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_oscar_foundation.lock" in runbook
    )
    assert "--constraints /tmp/requirements_oscar_foundation.previous.lock" in runbook
    assert "--index-url https://download.pytorch.org/whl/cu128" in runbook
    assert "--extra-index-url https://pypi.org/simple" in runbook


def test_groot_oscar_checkpoint_ownership_is_established_in_producing_layer() -> None:
    """The ~8.7GB checkpoint layer must not be duplicated by a later chown.

    OCI layers are copy-on-write: a recursive chown of /opt/blueprint/ckpts in
    a RUN after the checkpoint download rewrites every checkpoint byte into a
    second ~8.7GB layer. The runtime user must exist before the checkpoint
    layer, ownership must be set inside the layer that produces the files, and
    no later layer may recursively chown the checkpoint tree.
    """
    dockerfile = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")

    useradd_at = dockerfile.index("useradd")
    checkpoint_download_at = dockerfile.index("snapshot_download")
    assert useradd_at < checkpoint_download_at, (
        "runtime user must be created before the checkpoint layer so checkpoint "
        "ownership can be established without a later duplicate copy-up layer"
    )

    checkpoint_layer_start = dockerfile.rindex("RUN", 0, checkpoint_download_at)
    checkpoint_layer_end = dockerfile.index("\nBASH\n", checkpoint_download_at)
    checkpoint_layer = dockerfile[checkpoint_layer_start:checkpoint_layer_end]
    assert re.search(
        r"chown\s+(-R\s+)?blueprint:blueprint\s+/opt/blueprint/ckpts", checkpoint_layer
    ), "checkpoint ownership must be set inside the layer that produces the files"

    after_checkpoint_layer = dockerfile[checkpoint_layer_end:]
    assert not re.search(r"chown[^\n]*ckpts", after_checkpoint_layer), (
        "no layer after the checkpoint download may chown /opt/blueprint/ckpts; "
        "that duplicates ~8.7GB of checkpoint bytes in registry layer history"
    )

    # The runtime identity itself is preserved.
    assert "ARG APP_UID=10001" in dockerfile
    assert "ARG APP_GID=10001" in dockerfile
    assert re.search(r'groupadd --gid "\$\{APP_GID\}" blueprint', dockerfile)
    assert re.search(
        r'useradd --uid "\$\{APP_UID\}" --gid "\$\{APP_GID\}" --create-home '
        r"--shell /usr/sbin/nologin blueprint",
        dockerfile,
    )
    assert re.search(r"^USER blueprint$", dockerfile, re.MULTILINE), (
        "USER must be the name-only form: an explicit :group makes the "
        "runtime skip supplementary groups, dropping isaac-sim access"
    )


def test_groot_oscar_runtime_user_can_execute_worker_interpreters() -> None:
    """The runtime user must be able to execute every worker interpreter.

    The 2026-07-12 GPU canary on the sealed image failed every gate with
    rc=126: ``uv venv`` placed the managed CPython under ``/root/.local``
    (0700 ``/root``), and the Isaac base ships ``/isaac-sim`` as
    ``drwxr-x--- isaac-sim:isaac-sim`` — both unreachable for
    ``USER blueprint`` (uid 10001). A root-only build-time healthcheck cannot
    see either failure, so the Dockerfile must (a) pin the uv interpreter
    outside ``/root``, (b) grant the runtime user the ``isaac-sim`` group, and
    (c) run the build-time healthcheck as the runtime user.
    """
    dockerfile = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")

    uv_install_dir = re.search(r"UV_PYTHON_INSTALL_DIR=(\S+)", dockerfile)
    assert uv_install_dir, (
        "UV_PYTHON_INSTALL_DIR must be pinned so the uv-managed CPython the "
        "venvs symlink to is not created under /root (untraversable by the "
        "runtime user)"
    )
    target = uv_install_dir.group(1).rstrip("\\").strip()
    assert not target.startswith("/root"), target
    env_block_at = dockerfile.index("UV_PYTHON_INSTALL_DIR=")
    first_venv_at = dockerfile.index("uv venv ")
    assert env_block_at < first_venv_at, (
        "UV_PYTHON_INSTALL_DIR must be set before any `uv venv` layer"
    )

    assert re.search(r"usermod\s+-aG\s+isaac-sim\s+blueprint", dockerfile), (
        "the runtime user needs the isaac-sim group to traverse the "
        "group-restricted /isaac-sim tree (drwxr-x--- isaac-sim:isaac-sim)"
    )

    assert re.search(
        r"RUN\s+runuser\s+-u\s+blueprint\s+--\s+\S*python\S*\s+\S*"
        r"groot_oscar_closed_loop_image_healthcheck\.py\s+--build-time",
        dockerfile,
    ), (
        "the build-time healthcheck must run AS the runtime user; a root "
        "healthcheck cannot observe runtime-user exec/traversal failures"
    )


def test_groot_oscar_runtime_user_can_write_isaac_kit_runtime_dirs() -> None:
    """IMGFIX-004: the review render lane needs writable kit cache/data/logs.

    The 2026-07-12 live A40 canary on image c107af2a wedged inside
    ``isaac_review_renderer_canary``: the Isaac base ships
    ``/isaac-sim/kit/cache`` and ``/isaac-sim/kit/data`` group-readable but not
    group-writable, so ``USER blueprint`` (uid 10001, supplementary group
    isaac-sim) could not create ``kit/cache/DerivedDataCache`` or
    ``kit/data/documents/...`` and HydraEngine rtx failed creating the scene
    renderer. The fix must be SURGICAL: a non-recursive chown/chmod of exactly
    those directories (created if absent) in a root layer before
    ``USER blueprint``. A recursive chown/chmod of /isaac-sim would copy-up
    the multi-GB tree into a duplicate registry layer (the
    checkpoint_ownership_copyup discipline applies to the Isaac tree too).
    """
    dockerfile = (
        ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")

    kit_dirs = (
        "/isaac-sim/kit/cache",
        "/isaac-sim/kit/data",
        "/isaac-sim/kit/logs",
    )
    for kit_dir in kit_dirs:
        pattern = re.escape(kit_dir)
        assert re.search(rf"mkdir\s+-p[^\n]*{pattern}", dockerfile), (
            f"{kit_dir} must be created if absent so the write grant below "
            "cannot silently no-op on a base-image change"
        )
        assert re.search(rf"chown\s+blueprint:isaac-sim[^\n]*{pattern}", dockerfile), (
            f"{kit_dir} must be chowned (non-recursively) to the runtime user; "
            "the Isaac 6 kit creates DerivedDataCache/documents directly under "
            "these roots at renderer startup"
        )
        assert re.search(rf"chmod\s+0775[^\n]*{pattern}", dockerfile), (
            f"{kit_dir} must stay group-writable for the isaac-sim group"
        )

    grant_at = dockerfile.index("chown blueprint:isaac-sim")
    user_at = dockerfile.rindex("\nUSER blueprint")
    assert grant_at < user_at, (
        "the kit-dir write grant must happen in a root layer before USER blueprint"
    )
    healthcheck_at = dockerfile.index("groot_oscar_closed_loop_image_healthcheck.py --build-time")
    assert grant_at < healthcheck_at, (
        "the write grant must precede the runtime-user build-time healthcheck "
        "so the healthcheck can verify writability"
    )

    assert not re.search(r"ch(?:own|mod)\s+-[a-zA-Z]*R[a-zA-Z]*\s+[^\n]*/isaac-sim", dockerfile), (
        "never recursively chown/chmod under /isaac-sim: registry copy-up "
        "would duplicate the multi-GB Isaac tree in layer history"
    )

    healthcheck = (
        ROOT
        / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
        / "groot_oscar_closed_loop_image_healthcheck.py"
    ).read_text(encoding="utf-8")
    for kit_dir in kit_dirs:
        assert kit_dir in healthcheck, (
            f"the runtime-user healthcheck must probe writability of {kit_dir}; "
            "a wedge-at-runtime regression must fail the build instead"
        )
    assert "_dir_writable_by_current_user" in healthcheck
    assert "isaac_kit_dir_not_writable" in healthcheck


def test_groot_oscar_healthcheck_kit_dir_writability_probe() -> None:
    """The healthcheck writability probe must be a real create/delete probe."""
    import importlib.util

    path = (
        ROOT
        / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
        / "groot_oscar_closed_loop_image_healthcheck.py"
    )
    spec = importlib.util.spec_from_file_location(
        "groot_oscar_closed_loop_image_healthcheck_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        writable_dir = Path(tmp) / "writable"
        writable_dir.mkdir()
        assert module._dir_writable_by_current_user(writable_dir) is True
        assert not list(writable_dir.iterdir()), "probe must clean up after itself"

        missing_dir = Path(tmp) / "missing"
        assert module._dir_writable_by_current_user(missing_dir) is False

        readonly_dir = Path(tmp) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)
        try:
            if os.access(readonly_dir, os.W_OK):  # root can write anywhere
                import pytest

                pytest.skip("running as a user that bypasses directory modes")
            assert module._dir_writable_by_current_user(readonly_dir) is False
        finally:
            readonly_dir.chmod(0o755)


def test_groot_oscar_release_acceptance_requires_real_oci_runtime_smoke() -> None:
    """Release acceptance requires exercising the finished immutable image.

    ``runuser`` during a Docker build resolves supplementary groups differently
    from an OCI runtime when Dockerfile ``USER`` includes an explicit group.
    The release path therefore loads and smokes the finished local closure
    before its release tag can be pushed. The attested registry export must
    then bind its runnable config digest back to that smoke-tested image ID.
    """
    script = (ROOT / "scripts/build_push_groot_oscar_closed_loop_image.sh").read_text(
        encoding="utf-8"
    )

    build_at = script.index('"${local_build_args[@]}"')
    smoke_at = script.index('docker run --rm --entrypoint /bin/bash "$runtime_image_ref"')
    publish_at = script.index('"${publish_build_args[@]}"')
    digest_at = script.index('runtime_image_ref="${runtime_image_ref%:*}@${build_digest}"')
    binding_at = script.index('published_config_digest="$(python3')
    binding_check_at = script.index(
        'if [[ "$published_config_digest" != "$smoked_local_image_id" ]]'
    )
    registry_scan_at = script.index('syft "registry:${exact_digest_ref}"')
    provenance_gate_at = script.index("validate_buildkit_provenance_binding(")
    identity_recheck_at = script.index('source_identity_after_json="$(')
    promotion_at = script.index('docker buildx imagetools create --tag "$image_ref"')
    assert (
        build_at
        < smoke_at
        < publish_at
        < digest_at
        < binding_at
        < binding_check_at
        < registry_scan_at
        < provenance_gate_at
        < identity_recheck_at
        < promotion_at
    )
    assert "--load" in script[build_at - 500 : smoke_at]
    assert "--push" in script[publish_at - 500 : digest_at]
    assert '-t "$publish_staging_ref"' in script[publish_at - 500 : publish_at]
    assert '-t "$image_ref"' not in script[publish_at - 500 : publish_at]
    assert 'publish_staging_ref="${image_ref}-candidate-${source_commit:0:12}"' in script
    assert '--image "$runtime_image_ref" --output "$registry_manifest_output"' in script
    assert 'if [[ "$promoted_digest" != "$build_digest" ]]' in script
    assert '"$source_identity_after_json" == "$source_identity_json"' in script
    assert 'test "$(id -un)" = blueprint' in script
    assert "grep -Fx isaac-sim" in script
    assert "/isaac-sim/python.sh" in script
    assert "/opt/oscar-venv/bin/python" in script
    assert "/opt/gr00t-venv/bin/python" in script
    assert "groot_oscar_closed_loop_oci_runtime_smoke_failed" in script
    assert '"oci_runtime_smoke": {' in script
    assert 'runtime_smoke.get("status")' in script
    smoke_failure_at = script.index('if [[ "$runtime_smoke_exit" -ne 0 ]]')
    smoke_failure_exit_at = script.index("exit 2", smoke_failure_at)
    assert smoke_failure_at < smoke_failure_exit_at < publish_at
    assert "published_runtime_identity_matches_smoked_local_image" in script


def test_isaac_worker_final_runtime_user_can_discover_and_write_isaac_tree() -> None:
    dockerfile = (ROOT / "deploy/docker/robot_eval_worker/isaac/Dockerfile").read_text(
        encoding="utf-8"
    )

    assert re.search(r"usermod\s+-aG\s+isaac-sim\s+blueprint", dockerfile)
    assert re.search(r"^USER blueprint$", dockerfile, re.MULTILINE)
    assert "USER blueprint:blueprint" not in dockerfile
    assert "chmod o+x /isaac-sim" not in dockerfile
    assert "runuser -u blueprint -- test -r /isaac-sim" in dockerfile
    for kit_dir in (
        "/isaac-sim/kit/cache",
        "/isaac-sim/kit/data",
        "/isaac-sim/kit/logs",
    ):
        assert re.search(rf"mkdir\s+-p[^\n]*{re.escape(kit_dir)}", dockerfile)
        assert re.search(rf"chown\s+blueprint:isaac-sim[^\n]*{re.escape(kit_dir)}", dockerfile)
        assert re.search(rf"chmod\s+0775[^\n]*{re.escape(kit_dir)}", dockerfile)
        assert f"runuser -u blueprint -- test -w {kit_dir}" in dockerfile

    runtime_gate = re.search(
        r"&&\s+runuser\s+-u\s+blueprint\s+--\s+env\s+PYTHONPATH=/app/src\s+"
        r"/isaac-sim/python\.sh\s+-c[^\n]*\n"
        r"\s*\"from blueprint_pipeline\.nvidia_warehouse_native_camera_canary "
        r"import import_simulation_app; assert callable\(import_simulation_app\(\)\)\"",
        dockerfile,
    )
    assert runtime_gate, "the launcher import gate must run as the final non-root runtime user"
    assert not re.search(r"ch(?:own|mod)\s+-[a-zA-Z]*R[a-zA-Z]*\s+[^\n]*/isaac-sim", dockerfile)


def test_groot_oscar_release_ref_requires_tag_on_final_path_component(
    tmp_path: Path,
) -> None:
    script_path = ROOT / "scripts/build_push_groot_oscar_closed_loop_image.sh"
    script = script_path.read_text(encoding="utf-8")

    assert 'image_name="${image_ref##*/}"' in script
    assert 'if [[ "$image_ref" != *@sha256:* ]]; then' in script
    assert 'if [[ "$image_name" != *:* || -z "${image_name##*:}" ]]; then' in script
    assert 'case "$image_name" in' in script
    assert 'if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]' not in script

    manifest_path = tmp_path / "build-manifest.json"
    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF": (
                "registry.example:5000/blueprint/groot-oscar"
            ),
            "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_MANIFEST_OUTPUT": str(manifest_path),
        },
    )
    assert completed.returncode == 2
    assert "legacy build path disabled" in completed.stderr
    assert not manifest_path.exists()


def test_groot_oscar_release_push_requires_digest_pinned_base_image(
    tmp_path: Path,
) -> None:
    script_path = ROOT / "scripts/build_push_groot_oscar_closed_loop_image.sh"
    script = script_path.read_text(encoding="utf-8")

    assert '"$base_image" =~ @sha256:[0-9a-f]{64}$' in script
    manifest_path = tmp_path / "build-manifest.json"
    mutable_base = "nvcr.io/nvidia/isaac-sim:6.0.0"
    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF": (
                "registry.example/blueprint/groot-oscar:20260711"
            ),
            "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BASE_IMAGE": mutable_base,
            "BLUEPRINT_ALLOW_GROOT_OSCAR_CLOSED_LOOP_IMAGE_PUSH": "true",
            "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_MANIFEST_OUTPUT": str(manifest_path),
        },
    )

    assert completed.returncode == 2
    assert "legacy build path disabled" in completed.stderr
    assert not manifest_path.exists()


def test_dependabot_covers_every_dockerfile_directory() -> None:
    config = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    dockerfiles = [ROOT / "Dockerfile"]
    dockerfiles.extend(sorted((ROOT / "deploy" / "docker").rglob("Dockerfile")))
    dockerfile_directories = {
        f"/{path.parent.relative_to(ROOT).as_posix()}" if path.parent != ROOT else "/"
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
