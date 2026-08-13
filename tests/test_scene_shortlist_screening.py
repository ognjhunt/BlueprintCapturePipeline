"""Run the preregistered scene shortlist screening as code, not by hand.

`adp009a_scene_shortlist_extension_preregistration.v1.json` freezes both the
ordering rule and the screening sequence, and records
`method_outcomes_may_not_influence_selection`. Until now nothing executed it:
the sequence lived only as five strings in a manifest, so each scene was
screened by an operator and the result was a one-off. That is why 840313 is the
only scene with a sealed bundle, and why an agent picking up the work later
anchored on it rather than on the rank the frozen rule actually points to.

Screening decides which scene a paid run will use, so it has to be replayable
from the manifest and the retained bytes alone. These tests pin the parts that
protect that: the frozen order is followed rather than re-derived, a scene is
skipped only for a recorded reason, and a scene missing bytes can never be
silently promoted over one that is complete.
"""

import json
from pathlib import Path

import pytest

from blueprint_pipeline.scene_shortlist_screening import (
    SceneShortlistScreeningError,
    load_preregistered_shortlist,
    screen_shortlist,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests"
    / "adp009a_scene_shortlist_extension_preregistration.v1.json"
)

# Ranks as frozen in the preregistration; 840313 is rank 9, not rank 1.
EXPECTED_ORDER = ["840874", "840076", "841193", "840313", "840796"]


def _scene(root: Path, scene_id: str, folder: str, *, files: tuple[str, ...]) -> None:
    directory = root / "InteriorGS" / folder
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        (directory / name).write_bytes(f"{scene_id}:{name}".encode())


def _complete(root: Path, scene_id: str, folder: str) -> None:
    _scene(root, scene_id, folder,
           files=("3dgs_compressed.ply", "labels.json", "structure.json"))
    usdz = root / "SAGE-3D_InteriorGS_usdz" / "InteriorGS_usdz"
    usdz.mkdir(parents=True, exist_ok=True)
    (usdz / f"{scene_id}.usdz").write_bytes(b"usdz")


def test_reads_the_frozen_order_from_the_preregistration() -> None:
    shortlist = load_preregistered_shortlist(PREREGISTRATION)
    assert [entry["scene_id"] for entry in shortlist] == EXPECTED_ORDER


def test_refuses_a_preregistration_whose_digest_does_not_bind(tmp_path: Path) -> None:
    """A shortlist that can be edited freely is not a preregistration."""
    payload = json.loads(PREREGISTRATION.read_text())
    payload["shortlist"][0]["scene_id"] = "999999"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(SceneShortlistScreeningError, match="record_digest"):
        load_preregistered_shortlist(tampered)


def test_selects_the_first_complete_scene_in_frozen_order(tmp_path: Path) -> None:
    _complete(tmp_path, "841193", "0619_841193")
    _complete(tmp_path, "840313", "0442_840313")

    result = screen_shortlist(PREREGISTRATION, source_root=tmp_path)

    assert result["selected"]["scene_id"] == "841193", (
        "841193 is rank 8 and 840313 rank 9; the frozen order decides, not availability"
    )
    assert result["status"] == "selected"


def test_records_a_reason_for_every_scene_it_passes_over(tmp_path: Path) -> None:
    _complete(tmp_path, "841193", "0619_841193")

    result = screen_shortlist(PREREGISTRATION, source_root=tmp_path)

    skipped = {row["scene_id"]: row for row in result["screened"] if row["eligible"] is False}
    assert set(skipped) == {"840874", "840076"}
    for row in skipped.values():
        assert row["blockers"], "a scene may only be passed over for a recorded reason"


def test_never_promotes_an_incomplete_scene_over_a_complete_one(tmp_path: Path) -> None:
    # 840874 is rank 6 but missing its collision source entirely.
    _scene(tmp_path, "840874", "0239_840874",
           files=("3dgs_compressed.ply", "labels.json", "structure.json"))
    _complete(tmp_path, "841193", "0619_841193")

    result = screen_shortlist(PREREGISTRATION, source_root=tmp_path)

    assert result["selected"]["scene_id"] == "841193"
    row = next(r for r in result["screened"] if r["scene_id"] == "840874")
    assert "scene_shortlist_screening_missing:sage_usdz" in row["blockers"]


def test_blocks_rather_than_guessing_when_no_scene_is_complete(tmp_path: Path) -> None:
    result = screen_shortlist(PREREGISTRATION, source_root=tmp_path)
    assert result["status"] == "blocked"
    assert result["selected"] is None


def test_retains_a_digest_for_every_byte_it_screened(tmp_path: Path) -> None:
    """The selection has to be replayable from the retained bytes alone."""
    _complete(tmp_path, "841193", "0619_841193")

    result = screen_shortlist(PREREGISTRATION, source_root=tmp_path)

    inputs = result["selected"]["inputs"]
    assert set(inputs) == {"appearance_3dgs", "semantic_metadata", "scene_structure", "sage_usdz"}
    for entry in inputs.values():
        assert entry["digest"].startswith("sha256:")
        assert entry["size_bytes"] > 0


def test_is_deterministic_for_the_same_retained_bytes(tmp_path: Path) -> None:
    _complete(tmp_path, "841193", "0619_841193")
    first = screen_shortlist(PREREGISTRATION, source_root=tmp_path)
    second = screen_shortlist(PREREGISTRATION, source_root=tmp_path)
    assert first["screening_digest"] == second["screening_digest"]


def test_screening_digest_changes_when_a_screened_byte_changes(tmp_path: Path) -> None:
    _complete(tmp_path, "841193", "0619_841193")
    before = screen_shortlist(PREREGISTRATION, source_root=tmp_path)["screening_digest"]
    (tmp_path / "InteriorGS" / "0619_841193" / "labels.json").write_bytes(b"different")
    after = screen_shortlist(PREREGISTRATION, source_root=tmp_path)["screening_digest"]
    assert before != after


def test_reports_no_provider_mutation() -> None:
    """Screening reads retained bytes; it must never look like paid work."""
    result = screen_shortlist(PREREGISTRATION, source_root=REPO_ROOT / "does-not-exist")
    assert result["provider_mutation_performed"] is False
