from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.bootstrap_task_evaluation_scene_configuration_sources import (
    ensure_pinned_source_mirror,
)


def test_source_mirror_is_exact_detached_clean_and_idempotent(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (upstream / "driver.py").write_text("print('released')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Blueprint Tests",
            "-c",
            "user.email=tests@blueprint.invalid",
            "commit",
            "-qm",
            "released",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    destination = tmp_path / "sources/released"

    created = ensure_pinned_source_mirror(
        repository=str(upstream),
        commit=commit,
        tree=tree,
        destination=destination,
    )
    reopened = ensure_pinned_source_mirror(
        repository=str(upstream),
        commit=commit,
        tree=tree,
        destination=destination,
    )

    assert created["created"] is True
    assert reopened["created"] is False
    assert created["commit"] == commit
    assert created["tree"] == tree
    assert created["clean"] is True

