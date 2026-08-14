from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_push_sam31_source_tracks_image.py"
SPEC = importlib.util.spec_from_file_location("build_push_sam31_source_tracks_image", SCRIPT)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)

COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
IMAGE = "ghcr.io/ognjhunt/blueprint-sam31-source-tracks:20260813-c091426f400f"
DOCKER_HUB_IMAGE = "docker.io/nijelhunt/blueprint-sam31-source-tracks:20260813-c091426f400f"


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        "pyproject.toml": b"[project]\nname='fixture'\nversion='1'\n",
        "README.md": b"fixture\n",
        "LICENSE": b"fixture\n",
        "deploy/docker/sam31_source_tracks/Dockerfile": b"FROM scratch\nCOPY src/ src/\n",
        "src/fixture.py": b"VALUE = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


class FakeRunner:
    def __init__(self, *, dirty: bool = False, inspect_digest: str = DIGEST) -> None:
        self.dirty = dirty
        self.inspect_digest = inspect_digest
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        command = list(argv)
        self.calls.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return CompletedProcess(command, 0, COMMIT + "\n", "")
        if command[:3] == ["git", "status", "--porcelain=v1"]:
            return CompletedProcess(command, 0, "?? changed\n" if self.dirty else "", "")
        if command[:2] == ["git", "ls-files"]:
            files = [
                "LICENSE",
                "README.md",
                "deploy/docker/sam31_source_tracks/Dockerfile",
                "pyproject.toml",
                "src/fixture.py",
            ]
            return CompletedProcess(command, 0, "\0".join(files) + "\0", "")
        if command[:3] == ["docker", "buildx", "build"]:
            metadata_path = Path(command[command.index("--metadata-file") + 1])
            metadata_path.write_text(
                json.dumps({"containerimage.digest": DIGEST}) + "\n", encoding="utf-8"
            )
            context = Path(command[-1])
            assert sorted(
                path.relative_to(context).as_posix()
                for path in context.rglob("*")
                if path.is_file()
            ) == [
                "LICENSE",
                "README.md",
                "deploy/docker/sam31_source_tracks/Dockerfile",
                "pyproject.toml",
                "src/fixture.py",
            ]
            return CompletedProcess(command, 0, "build complete\n", "")
        if command[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            payload = {"manifest": {"digest": self.inspect_digest}}
            return CompletedProcess(command, 0, json.dumps(payload), "")
        return CompletedProcess(command, 0, "ok\n", "")


def test_publishes_exact_context_and_retains_immutable_digest(tmp_path: Path) -> None:
    runner = FakeRunner()
    output = tmp_path / "evidence"

    receipt = publisher.publish_sam31_source_tracks_image(
        repo_root=_repo(tmp_path),
        source_commit=COMMIT,
        image_ref=IMAGE,
        output_dir=output,
        runner=runner,
    )

    build = next(call for call in runner.calls if call[:3] == ["docker", "buildx", "build"])
    assert build[build.index("--platform") + 1] == "linux/amd64"
    assert [build[index + 1] for index, value in enumerate(build) if value == "--attest"] == [
        "type=sbom",
        "type=provenance,mode=max",
    ]
    assert "--push" in build
    assert receipt["resolved_digest_ref"] == IMAGE.rsplit(":", 1)[0] + "@" + DIGEST
    assert receipt["schema_version"] == "semantic_sam31_runtime_image_build_receipt.v1"
    assert receipt["runtime_image_identity"] == receipt["resolved_digest_ref"]
    assert receipt["runtime_digest"] == DIGEST
    assert receipt["registry_api_digest_verified"] is True
    assert receipt["official_code_revision"] == publisher.OFFICIAL_CODE_REVISION
    assert receipt["dockerfile_sha256"].startswith("sha256:")
    assert receipt["source_tree_digest"].startswith("sha256:")
    assert receipt["build_provenance_digest"].startswith("sha256:")
    assert receipt["source_commit_sha"] == COMMIT
    assert receipt["registry_credentials_read_by_publisher"] is False
    assert receipt["raw_secret_values_recorded"] is False
    assert (output / "buildx.log").read_text(encoding="utf-8") == "ok\nok\nbuild complete\n"
    retained = json.loads((output / "publication_receipt.json").read_text(encoding="utf-8"))
    assert retained == receipt
    assert retained["receipt_digest"].startswith("sha256:")


@pytest.mark.parametrize(
    ("image_ref", "error"),
    [
        (
            "ghcr.io/ognjhunt/blueprint-sam31-source-tracks:latest",
            "sam31_image_ref_unstable_tag_forbidden",
        ),
        (
            "ghcr.io/ognjhunt/blueprint-sam31-source-tracks@sha256:" + "c" * 64,
            "sam31_image_ref_not_versioned_registry_tag",
        ),
        ("quay.io/example/sam31:v1", "sam31_image_ref_not_versioned_registry_tag"),
    ],
)
def test_rejects_nonversioned_or_unapproved_registry_target_before_commands(
    tmp_path: Path, image_ref: str, error: str
) -> None:
    runner = FakeRunner()
    with pytest.raises(publisher.Sam31ImagePublicationError, match=error):
        publisher.publish_sam31_source_tracks_image(
            repo_root=_repo(tmp_path),
            source_commit=COMMIT,
            image_ref=image_ref,
            output_dir=tmp_path / "evidence",
            runner=runner,
        )
    assert runner.calls == []


def test_accepts_versioned_docker_hub_target(tmp_path: Path) -> None:
    receipt = publisher.publish_sam31_source_tracks_image(
        repo_root=_repo(tmp_path),
        source_commit=COMMIT,
        image_ref=DOCKER_HUB_IMAGE,
        output_dir=tmp_path / "evidence",
        runner=FakeRunner(),
    )
    assert receipt["runtime_image_identity"] == (
        DOCKER_HUB_IMAGE.rsplit(":", 1)[0] + "@" + DIGEST
    )


def test_rejects_dirty_checkout_before_docker_or_output(tmp_path: Path) -> None:
    runner = FakeRunner(dirty=True)
    output = tmp_path / "evidence"
    with pytest.raises(
        publisher.Sam31ImagePublicationError, match="sam31_source_checkout_not_clean"
    ):
        publisher.publish_sam31_source_tracks_image(
            repo_root=_repo(tmp_path),
            source_commit=COMMIT,
            image_ref=IMAGE,
            output_dir=output,
            runner=runner,
        )
    assert not output.exists()
    assert all(call[0] != "docker" for call in runner.calls)


def test_rejects_registry_digest_different_from_build_metadata(tmp_path: Path) -> None:
    runner = FakeRunner(inspect_digest="sha256:" + "c" * 64)
    with pytest.raises(
        publisher.Sam31ImagePublicationError, match="sam31_registry_digest_mismatch"
    ):
        publisher.publish_sam31_source_tracks_image(
            repo_root=_repo(tmp_path),
            source_commit=COMMIT,
            image_ref=IMAGE,
            output_dir=tmp_path / "evidence",
            runner=runner,
        )


def test_publisher_never_reads_registry_secret_sources() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "REGISTRY_PASSWORD" not in source
    assert "DOCKER_PASSWORD" not in source
    assert "docker login" not in source
