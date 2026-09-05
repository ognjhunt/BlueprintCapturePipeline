"""Every production root a unit names has a storage class, and reclaim tools honour it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_storage_roots import (
    STORAGE_CLASSES,
    STORAGE_ROOTS,
    classify_path,
    require_storage_class,
    roots_of_class,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
_HOST_PATH = re.compile(r"(/var/lib/blueprint[A-Za-z0-9_./-]*|/opt/blueprint[A-Za-z0-9_./-]*)")
# Paths that are not storage: the interpreter and file arguments inside the venv.
_NOT_STORAGE = ("/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python",)


def test_every_root_named_by_a_production_unit_is_classified() -> None:
    unclassified: set[str] = set()
    for unit in sorted(SYSTEMD_DIR.glob("blueprint-*")):
        for match in _HOST_PATH.findall(unit.read_text(encoding="utf-8")):
            path = match.rstrip("/")
            if path in _NOT_STORAGE:
                continue
            root = classify_path(path)
            if root is None or root.storage_class == "container" and path not in {
                r.path for r in STORAGE_ROOTS
            }:
                # A path inside a container root without a more specific class
                # is exactly the unclassified growth this table exists to stop.
                unclassified.add(path)
    assert unclassified == set(), sorted(unclassified)


def test_storage_table_is_well_formed() -> None:
    seen: set[str] = set()
    for root in STORAGE_ROOTS:
        assert root.storage_class in STORAGE_CLASSES, root
        assert root.owner in {"blueprint", "root"}, root
        assert root.path.startswith(("/var/lib/blueprint", "/opt/blueprint")), root
        assert root.path not in seen, root
        seen.add(root.path)
    # The spend guard is hot evidence and never a reclaim target of any class.
    guard = classify_path("/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/billing-audit")
    assert guard is not None and guard.storage_class == "evidence_hot"
    assert "/var/lib/blueprint/task-evaluation-inputs/prepared-references" in roots_of_class("cache")
    assert "/var/lib/blueprint/pipeline-control-plane/profile-install-staging" in roots_of_class("cache")
    assert "/var/lib/blueprint/pipeline-control-plane/policy-canary-presubmission" in roots_of_class(
        "cache"
    )
    assert "/var/lib/blueprint/pipeline-control-plane/episode-interpretation-rights" in roots_of_class(
        "evidence_hot"
    )
    assert "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs" in roots_of_class(
        "evidence_cold"
    )
    assert "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions" in roots_of_class(
        "work"
    )
    assert "/var/lib/blueprint/task-evaluation-inputs/sam31-preparations" in roots_of_class(
        "cache"
    )


def test_most_specific_root_wins_and_tools_refuse_wrong_classes() -> None:
    nested = classify_path(
        "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes/content-addressed/"
        "adapter-members/sha256/abc"
    )
    assert nested is not None and nested.storage_class == "cache"
    assert classify_path("/var/lib/blueprint/pipeline-control-plane").storage_class == "container"
    assert classify_path("/srv/elsewhere") is None

    root = require_storage_class(
        "/var/lib/blueprint/task-evaluation-inputs/prepared-references",
        expected="cache",
        code="fixture_code",
    )
    assert root.storage_class == "cache"
    with pytest.raises(ValueError, match="^fixture_code:evidence_hot$"):
        require_storage_class(
            "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard",
            expected="cache",
            code="fixture_code",
        )
    with pytest.raises(ValueError, match="^fixture_code:unclassified$"):
        require_storage_class("/srv/elsewhere", expected="cache", code="fixture_code")
    with pytest.raises(ValueError, match="control_plane_storage_class_invalid"):
        roots_of_class("bogus")
