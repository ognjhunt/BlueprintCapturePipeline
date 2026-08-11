from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline.public_scene_cad_inspection_capture import (
    EARTHTOJAKE_TEXT_TO_CAD_REPOSITORY,
    PublicSceneCadInspectionCaptureError,
    capture_cad_inspection,
)


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    skill = tmp_path / "skill"
    step = repo / "assets" / "control.step"
    launcher = skill / "skills" / "cad" / "scripts" / "inspect" / "__main__.py"
    step.parent.mkdir(parents=True)
    evidence.mkdir()
    launcher.parent.mkdir(parents=True)
    step.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
    launcher.write_text(
        """
import hashlib
import json
from pathlib import Path
import sys

step = Path(sys.argv[2])
digest = hashlib.sha256(step.read_bytes()).hexdigest()
print(json.dumps({
    "ok": True,
    "tokens": [{
        "stepHash": digest,
        "summary": {"kind": "part", "shapeCount": 1},
    }],
    "errors": [],
}))
""".lstrip(),
        encoding="utf-8",
    )
    _git(skill, "init", "-q")
    _git(skill, "config", "user.email", "test@example.com")
    _git(skill, "config", "user.name", "Test")
    _git(skill, "add", ".")
    _git(skill, "commit", "-qm", "fixture")
    _git(skill, "remote", "add", "origin", EARTHTOJAKE_TEXT_TO_CAD_REPOSITORY)
    return {
        "repo": repo,
        "evidence": evidence,
        "skill": skill,
        "step": step,
        "output": evidence / "inspection.json",
        "commit": _git(skill, "rev-parse", "HEAD"),
        "tree": _git(skill, "rev-parse", "HEAD^{tree}"),
    }


def test_capture_binds_pinned_skill_command_and_step_digest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = capture_cad_inspection(
        repo_root=paths["repo"],
        evidence_root=paths["evidence"],
        cad_skill_root=paths["skill"],
        cad_python=sys.executable,
        expected_commit=str(paths["commit"]),
        expected_tree=str(paths["tree"]),
        step_path=paths["step"],
        output_path=paths["output"],
    )

    assert receipt["ok"] is True
    assert receipt["capture_provenance"]["cad_skill_commit"] == paths["commit"]
    assert receipt["capture_provenance"]["step_sha256"].startswith("sha256:")
    assert json.loads(Path(paths["output"]).read_text(encoding="utf-8"))["ok"] is True


def test_capture_rejects_unpinned_skill_revision(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(PublicSceneCadInspectionCaptureError, match="cad_skill_revision_mismatch"):
        capture_cad_inspection(
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            cad_skill_root=paths["skill"],
            cad_python=sys.executable,
            expected_commit="0" * 40,
            expected_tree=str(paths["tree"]),
            step_path=paths["step"],
            output_path=paths["output"],
        )


def test_capture_rejects_a_checkout_without_the_approved_earthtojake_origin(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _git(Path(paths["skill"]), "remote", "set-url", "origin", "https://example.test/not-earthtojake")

    with pytest.raises(PublicSceneCadInspectionCaptureError, match="cad_skill_repository_mismatch"):
        capture_cad_inspection(
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            cad_skill_root=paths["skill"],
            cad_python=sys.executable,
            expected_commit=str(paths["commit"]),
            expected_tree=str(paths["tree"]),
            step_path=paths["step"],
            output_path=paths["output"],
        )


def test_capture_rejects_an_unbounded_timeout(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(PublicSceneCadInspectionCaptureError, match="cad_inspection_timeout_invalid"):
        capture_cad_inspection(
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            cad_skill_root=paths["skill"],
            cad_python=sys.executable,
            expected_commit=str(paths["commit"]),
            expected_tree=str(paths["tree"]),
            step_path=paths["step"],
            output_path=paths["output"],
            timeout_seconds=0,
        )
