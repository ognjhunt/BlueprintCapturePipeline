"""The installer's Caddyfile patch: one route, anchored, idempotent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.caddy import CaddyPatchError, patch_caddyfile  # noqa: E402

LIVE = """# Blueprint Pipeline control-plane edge.
paperclip.tryblueprint.io, 174-138-76-111.sslip.io {
\thandle /api/live-pipeline/* {
\t\treverse_proxy 127.0.0.1:8765
\t}
\thandle {
\t\trespond "blueprint-pipeline-control-plane" 200
\t}
}
"""


def test_inserts_the_operator_route_before_the_intake_route() -> None:
    patched = patch_caddyfile(LIVE)
    assert patched is not None
    lines = patched.splitlines()
    operator = lines.index("\thandle /api/live-pipeline/operator/* {")
    assert lines[operator + 1] == "\t\treverse_proxy 127.0.0.1:8767"
    assert lines[operator + 2] == "\t}"
    assert lines[operator + 3] == "\thandle /api/live-pipeline/* {"
    assert patched.count("reverse_proxy 127.0.0.1:8765") == 1


def test_is_idempotent() -> None:
    patched = patch_caddyfile(LIVE)
    assert patched is not None and patch_caddyfile(patched) is None


def test_matches_space_indented_files() -> None:
    spaced = LIVE.replace("\t", "    ")
    patched = patch_caddyfile(spaced)
    assert patched is not None
    assert "    handle /api/live-pipeline/operator/* {\n        reverse_proxy 127.0.0.1:8767\n    }\n" in patched


def test_refuses_a_file_without_the_intake_route() -> None:
    with pytest.raises(CaddyPatchError, match="caddy_anchor_missing"):
        patch_caddyfile("example.com {\n\trespond ok\n}\n")


def test_repo_caddyfile_already_carries_the_route() -> None:
    repo = Path(__file__).resolve().parents[1] / "deploy" / "caddy" / "Caddyfile"
    assert patch_caddyfile(repo.read_text(encoding="utf-8")) is None
