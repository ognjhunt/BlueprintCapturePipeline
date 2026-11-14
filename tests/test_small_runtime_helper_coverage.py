from __future__ import annotations

import io
import json
import runpy
import sys
from pathlib import Path
from urllib import error as urllib_error

import pytest

from blueprint_pipeline.agent_runtime import skill_sync
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.local_capture import resolve_local_capture_context
from blueprint_pipeline.video_to_world_client import run_video_to_world_provider


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) if not isinstance(row, str) else row for row in rows) + "\n\n",
        encoding="utf-8",
    )


def _capture_context(tmp_path: Path):
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "site-1" / "captures" / "cap-1"
    capture_root.mkdir(parents=True)
    return resolve_local_capture_context(capture_root)


class _Response:
    def __init__(self, body: object) -> None:
        self._body = body if isinstance(body, bytes) else str(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body


def test_video_to_world_client_payload_success_and_runner_helpers(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.headers)
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Response(json.dumps({"status": "succeeded", "artifact_uri": "gs://bucket/out"}))

    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example/root/")
    monkeypatch.delenv("VIDEO_TO_WORLD_RUNNER_TOKEN", raising=False)
    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "privacy-token")
    monkeypatch.setenv("VIDEO_TO_WORLD_TIMEOUT_SECONDS", "bad-int")
    monkeypatch.setattr("blueprint_pipeline.video_to_world_client.urllib_request.urlopen", fake_urlopen)

    payload = run_video_to_world_provider(
        video_path=tmp_path / "raw.mov",
        video_uri="gs://bucket/raw.mov",
        geometry_root=tmp_path / "geometry",
        dynamic_mask_manifest_path=tmp_path / "masks" / "manifest.json",
        dynamic_mask_manifest_uri="gs://bucket/capture/masks/manifest.json",
        provider="da3",
        model="depth-anything-3",
        execution_mode="remote",
        video_probe={"fps": 24},
    )

    assert payload["artifact_uri"] == "gs://bucket/out"
    assert captured["url"] == "https://runner.example/root/run"
    assert captured["timeout"] == 7200
    assert captured["headers"]["Authorization"] == "Bearer privacy-token"
    assert captured["payload"] == {
        "input_video_path": str(tmp_path / "raw.mov"),
        "input_video_uri": "gs://bucket/raw.mov",
        "geometry_root_path": str(tmp_path / "geometry"),
        "geometry_root_uri": "gs://bucket/capture",
        "dynamic_mask_manifest_path": str(tmp_path / "masks" / "manifest.json"),
        "dynamic_mask_manifest_uri": "gs://bucket/capture/masks/manifest.json",
        "provider": "da3",
        "model": "depth-anything-3",
        "execution_mode": "remote",
        "video_probe": {"fps": 24},
    }


def test_video_to_world_client_adds_cloud_run_iam_header(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        del timeout
        captured["headers"] = dict(req.headers)
        return _Response(json.dumps({"status": "succeeded"}))

    def add_cloud_run_header(headers: dict[str, str], *, url: str) -> dict[str, str]:
        assert url == "https://runner.example"
        merged = dict(headers)
        merged["X-Serverless-Authorization"] = "Bearer google-id-token"
        return merged

    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "video-token")
    monkeypatch.setattr(
        "blueprint_pipeline.video_to_world_client.cloud_run_id_token_headers",
        add_cloud_run_header,
    )
    monkeypatch.setattr("blueprint_pipeline.video_to_world_client.urllib_request.urlopen", fake_urlopen)

    payload = run_video_to_world_provider(
        video_path=tmp_path / "raw.mov",
        video_uri="gs://bucket/raw.mov",
        geometry_root=tmp_path / "geometry",
        dynamic_mask_manifest_path=tmp_path / "masks" / "manifest.json",
        dynamic_mask_manifest_uri="gs://bucket/capture/masks/manifest.json",
        provider="da3",
        model="depth-anything-3",
        execution_mode="remote",
        video_probe={},
    )

    assert payload["status"] == "succeeded"
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer video-token"  # type: ignore[index]
    assert headers["X-serverless-authorization"] == "Bearer google-id-token"  # type: ignore[index]


@pytest.mark.parametrize(
    ("body", "error_match"),
    [
        ("not-json", "video_to_world_invalid_json"),
        (json.dumps(["bad"]), "video_to_world_invalid_payload"),
        (json.dumps({"status": "failed", "reason": "model_crashed"}), "model_crashed"),
        (json.dumps({}), "video_to_world_failed"),
    ],
)
def test_video_to_world_client_response_failures(monkeypatch, tmp_path: Path, body: str, error_match: str) -> None:
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example")
    monkeypatch.setenv("VIDEO_TO_WORLD_TIMEOUT_SECONDS", "4")
    monkeypatch.setattr(
        "blueprint_pipeline.video_to_world_client.urllib_request.urlopen",
        lambda *_args, **_kwargs: _Response(body),
    )

    with pytest.raises(RuntimeError, match=error_match):
        run_video_to_world_provider(
            video_path=tmp_path / "raw.mov",
            video_uri="gs://bucket/raw.mov",
            geometry_root=tmp_path / "geometry",
            dynamic_mask_manifest_path=tmp_path / "masks.json",
            dynamic_mask_manifest_uri="gs://bucket/masks/manifest.json",
            provider="provider",
            model="model",
            execution_mode="remote",
            video_probe={},
        )


@pytest.mark.parametrize(
    "urlopen_error, error_match",
    [
        (
            urllib_error.HTTPError(
                "https://runner.example/run",
                503,
                "unavailable",
                {},
                io.BytesIO(b"service unavailable with extra diagnostic text"),
            ),
            "video_to_world_http_503:service unavailable",
        ),
        (urllib_error.URLError("network down"), "video_to_world_unreachable:network down"),
    ],
)
def test_video_to_world_client_transport_failures(monkeypatch, tmp_path: Path, urlopen_error: Exception, error_match: str) -> None:
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example")

    def raise_error(*_args, **_kwargs):
        raise urlopen_error

    monkeypatch.setattr("blueprint_pipeline.video_to_world_client.urllib_request.urlopen", raise_error)

    with pytest.raises(RuntimeError, match=error_match):
        run_video_to_world_provider(
            video_path=tmp_path / "raw.mov",
            video_uri="gs://bucket/raw.mov",
            geometry_root=tmp_path / "geometry",
            dynamic_mask_manifest_path=tmp_path / "masks.json",
            dynamic_mask_manifest_uri="gs://bucket/masks/manifest.json",
            provider="provider",
            model="model",
            execution_mode="remote",
            video_probe={},
        )


def test_video_to_world_client_blocks_without_runner_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIDEO_TO_WORLD_URL", raising=False)

    with pytest.raises(RuntimeError, match="video_to_world_runner_not_configured"):
        run_video_to_world_provider(
            video_path=tmp_path / "raw.mov",
            video_uri="gs://bucket/raw.mov",
            geometry_root=tmp_path / "geometry",
            dynamic_mask_manifest_path=tmp_path / "masks.json",
            dynamic_mask_manifest_uri="gs://bucket/masks/manifest.json",
            provider="provider",
            model="model",
            execution_mode="remote",
            video_probe={},
        )


def _build_skillpack_repo(root: Path) -> None:
    _write_json(
        root / "skillpacks" / "ops" / "skillpack_manifest.json",
        {"name": "ops", "source_root": "skill_src", "skills": ["drive", "", "dock"]},
    )
    for skill in ["drive", "dock"]:
        skill_dir = root / "skill_src" / skill
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill}\n", encoding="utf-8")


def test_skill_sync_copies_skillpacks_and_main_reports_success(tmp_path: Path, capsys) -> None:
    _build_skillpack_repo(tmp_path)

    assert (skill_sync._repo_root() / "pyproject.toml").is_file()
    result = skill_sync.sync_skill_pack(tmp_path)

    assert result["schema_version"] == "v1"
    assert result["skill_count"] == 2
    assert result["skillpacks"] == ["ops"]
    assert result["skills"] == ["drive", "dock"]
    for target in [tmp_path / ".claude" / "skills", tmp_path / ".agents" / "skills"]:
        assert (target / "drive" / "SKILL.md").read_text(encoding="utf-8") == "# drive\n"
        assert (target / "dock" / "SKILL.md").read_text(encoding="utf-8") == "# dock\n"

    assert skill_sync.main(["--repo-root", str(tmp_path)]) == 0
    assert "[skill-sync] synced 2 skills" in capsys.readouterr().out


def test_production_cad_skillpack_declares_access_to_all_pinned_skills() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifests = skill_sync.load_skillpack_manifests(repository)
    skill_sync._validate_unique_skills(manifests)
    cad_manifest = next(row for row in manifests if row.get("name") == "cad_authoring")
    expected = {
        "cad",
        "cad-viewer",
        "urdf",
        "sdf",
        "srdf",
        "step-parts",
        "implicit-cad",
        "dxf",
        "gcode",
        "multi-agent-cad",
    }
    assert set(cad_manifest["skills"]) == expected
    source_root = repository / cad_manifest["source_root"]
    for skill in expected:
        assert (source_root / skill / "SKILL.md").is_file()


def test_skill_sync_validates_missing_and_duplicate_skillpacks(tmp_path: Path, capsys) -> None:
    with pytest.raises(PipelineError, match="No skill pack manifests"):
        skill_sync.load_skillpack_manifests(tmp_path)

    _write_json(
        tmp_path / "skillpacks" / "a" / "skillpack_manifest.json",
        {"name": "a", "source_root": "a_src", "skills": ["drive"]},
    )
    _write_json(
        tmp_path / "skillpacks" / "b" / "skillpack_manifest.json",
        {"name": "b", "source_root": "b_src", "skills": ["drive"]},
    )
    with pytest.raises(PipelineError, match="Duplicate skill 'drive'"):
        skill_sync.sync_skill_pack(tmp_path)

    (tmp_path / "skillpacks" / "b" / "skillpack_manifest.json").unlink()
    assert skill_sync.main(["--repo-root", str(tmp_path)]) == 1
    assert "[skill-sync] FAILED: Skill source root is missing" in capsys.readouterr().out


def test_skill_sync_validates_missing_declared_skill_and_main_guard(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "skillpacks" / "ops" / "skillpack_manifest.json",
        {"name": "ops", "source_root": "skill_src", "skills": ["missing"]},
    )
    (tmp_path / "skill_src").mkdir()

    with pytest.raises(PipelineError, match="Skill source is missing"):
        skill_sync.sync_skill_pack(tmp_path)

    monkeypatch.setattr(sys, "argv", ["skill_sync", "--repo-root", str(tmp_path)])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.agent_runtime.skill_sync", run_name="__main__")
    assert excinfo.value.code == 1


