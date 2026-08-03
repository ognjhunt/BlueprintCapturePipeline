from pathlib import Path
from types import SimpleNamespace

import blueprint_pipeline.splat_transform_collision as adapter


def _install_cli(tmp_path: Path) -> None:
    cli = tmp_path / "tools/splat_render/node_modules/@playcanvas/splat-transform/bin/cli.mjs"
    cli.parent.mkdir(parents=True)
    cli.write_text("// test fixture\n", encoding="utf-8")


def test_generates_unqualified_upstream_collision_candidate(tmp_path, monkeypatch) -> None:
    _install_cli(tmp_path)
    source = tmp_path / "scene.spz"
    output = tmp_path / "result" / "scene.voxel.json"
    source.write_bytes(b"source")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="splat-transform v3.2.0\n", stderr="")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"json")
        output.with_name("scene.voxel.bin").write_bytes(b"voxel")
        output.with_name("scene.collision.glb").write_bytes(b"glb")
        return SimpleNamespace(
            returncode=0,
            stdout="gaussians: 100\ngaussians: 72\n",
            stderr="",
        )

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    result = adapter.generate_splat_transform_collision_candidate(
        source,
        output,
        repo_root=tmp_path,
        robust_bounds=[-3, -1, -4, 6, 2, 5],
    )

    assert result["status"] == "candidate_generated"
    assert result["claim_ceiling"] == "splat_derived_collision_candidate"
    assert result["collision_validated"] is False
    assert result["evaluation_authorized"] is False
    assert result["actions"]["global_decimation_applied"] is False
    assert result["retained_splat_count"] == 72
    assert result["retained_splat_fraction"] == 0.72
    assert result["artifacts"]["collision_glb"]["bytes"] == 3
    command = calls[-1]
    assert "--filter-value=opacity,lt,0.999999" in command
    assert command.count("--stats") == 2
    assert "--collision-mesh=faces" in command
    assert any(token.startswith("--filter-box=") for token in command)


def test_rejects_invalid_output_without_running(tmp_path, monkeypatch) -> None:
    source = tmp_path / "scene.spz"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    result = adapter.generate_splat_transform_collision_candidate(source, tmp_path / "scene.glb")
    assert result["status"] == "blocked"
    assert "splat_collision_output_must_end_voxel_json" in result["blockers"]
