from pathlib import Path
import json
import subprocess

import pytest

from scripts import provision_website_geometry_runtime as subject
from blueprint_pipeline.website_geometry_dispatch import load_profile


@pytest.fixture
def setup_runtime(tmp_path, monkeypatch):
    files = []
    for name in ("mapanything-1.1.4-py3-none-any.whl", "blueprint_contracts-0.1.0-py3-none-any.whl", "requirements.txt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append({"path": str(path), "digest": subject._digest(path)})
    value = dict(schema_version="website_mapanything_deployment.v1", runtime_files=files,
                 worker_image_digest="pytorch/pytorch@sha256:" + "b" * 64,
                 maximum_cost_usd=2, max_hourly_rate_usd=1.5,
                 hard_ttl_seconds=1800, minimum_gpu_ram_mb=80000)
    template = tmp_path / "template.json"
    template.write_text(json.dumps(value))
    builds = []

    def build(repository, commit, destination):
        builds.append(commit)
        path = destination / "blueprint_capture_pipeline-2.0.0-py3-none-any.whl"
        path.write_bytes(commit.encode())
        return path

    monkeypatch.setattr(subject, "_build_wheel", build)
    kwargs = dict(repository_root=tmp_path, source_commit="a" * 40,
                  runtime_root=tmp_path / "runtimes", template_path=template,
                  readback=lambda path: path.read_bytes())
    return kwargs, builds, value


def test_provisions_once_reopens_and_rebuilds_for_next_release(setup_runtime):
    kwargs, builds, _ = setup_runtime
    profile = subject.provision_website_geometry_runtime(**kwargs)
    value = load_profile(source_commit=kwargs["source_commit"], profile_path=profile)
    assert len(value["runtime_files"]) == 4
    assert all(Path(row["path"]).stat().st_mode & 0o777 == 0o444 for row in value["runtime_files"])
    assert subject.provision_website_geometry_runtime(**kwargs) == profile
    next_profile = subject.provision_website_geometry_runtime(**{**kwargs, "source_commit": "b" * 40})
    assert next_profile != profile
    assert builds == ["a" * 40, "b" * 40]


def test_refuses_changed_dependency_before_build(setup_runtime):
    kwargs, builds, value = setup_runtime
    Path(value["runtime_files"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="deployment_input_changed"):
        subject.provision_website_geometry_runtime(**kwargs)
    assert builds == []


def test_refuses_changed_published_runtime_without_rebuilding(setup_runtime):
    kwargs, builds, _ = setup_runtime
    profile = subject.provision_website_geometry_runtime(**kwargs)
    wheel = Path(json.loads(profile.read_text())["runtime_files"][0]["path"])
    wheel.chmod(0o644)
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="runtime_file_changed"):
        subject.provision_website_geometry_runtime(**kwargs)
    assert len(builds) == 1


def test_refuses_new_budget_for_existing_release(setup_runtime):
    kwargs, _, value = setup_runtime
    subject.provision_website_geometry_runtime(**kwargs)
    value["maximum_cost_usd"] = 3
    kwargs["template_path"].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="configuration_changed"):
        subject.provision_website_geometry_runtime(**kwargs)


def test_requires_service_readback(setup_runtime):
    kwargs, _, _ = setup_runtime
    with pytest.raises(ValueError, match="service_readback_failed"):
        subject.provision_website_geometry_runtime(**{**kwargs, "readback": lambda _: b"wrong"})


def test_invalid_budget_never_publishes_profile(setup_runtime):
    kwargs, _, value = setup_runtime
    value["maximum_cost_usd"] = 0.1
    kwargs["template_path"].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="budget_below_ttl"):
        subject.provision_website_geometry_runtime(**kwargs)
    assert not (kwargs["runtime_root"] / "website-mapanything" / kwargs["source_commit"]).exists()


def test_wheel_build_uses_committed_package_and_ignores_unrelated_symlinks(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True, text=True)
    git("init")
    (repository / "src").mkdir()
    (repository / "src/module.py").write_text("committed")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (repository / name).write_text(name)
    (repository / "unrelated-link").symlink_to("/missing/global/skill")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD").stdout.strip()
    (repository / "src/module.py").write_text("uncommitted")
    execute = subprocess.run
    def run(command, **kwargs):
        if "wheel" not in command:
            return execute(command, **kwargs)
        source = Path(command[-1])
        assert (source / "src/module.py").read_text() == "committed"
        assert (source / "LICENSE").is_file()
        assert not (source / "unrelated-link").is_symlink()
        assert "--no-index" in command and "--no-deps" in command
        wheel = Path(command[command.index("--wheel-dir") + 1]) / "blueprint_capture_pipeline-2.0.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
    monkeypatch.setattr(subject.subprocess, "run", run)
    destination = tmp_path / "wheel"
    destination.mkdir()
    assert subject._build_wheel(repository, commit, destination).read_bytes() == b"wheel"
