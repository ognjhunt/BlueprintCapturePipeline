# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_queue_auxiliary_observation.py
"""Typed retained receipt locations, without consumer or cleanup authority."""
import hashlib
import json
import os

import pytest

from blueprint_pipeline import control_plane_queue_auxiliary_observation as auxiliary

HEX = "a" * 64
OTHER = "b" * 64
STEM = "room.part-with-hyphen-" + HEX
CHILD = "sam31-" + HEX
CONTAINERS = {
    "preparation": ("identities", "results/conflicts", "source-progress",
                    "source-resume-pending", "source-resume-blocked", "source-resume-completed"),
    "sam": ("results", "started", "progress", "wake-pending", "wake-completed"),
    "activation": ("identities", "results/conflicts"),
}
LAYOUTS = (
    ("activation", "identities/activation-1.json", "identity"),
    ("activation", f"results/activation-1-{HEX}.json", "result"),
    ("activation", f"results/conflicts/activation-1-{HEX}-{OTHER}.json", "result_conflict"),
    ("preparation", "identities/room.part-with-hyphen.json", "identity"),
    ("preparation", f"results/{STEM}.json", "result"),
    ("preparation", f"results/conflicts/{STEM}-{OTHER}.json", "result_conflict"),
    ("preparation", f"source-progress/{STEM}/000001-{OTHER}.json", "source_progress"),
    ("preparation", f"source-progress/{STEM}/1000000-{HEX}.json", "source_progress"),
    ("preparation", f"source-resume-pending/{HEX}.json", "resume_pending"),
    ("preparation", f"source-resume-blocked/{HEX}.json", "resume_blocked"),
    ("preparation", f"source-resume-blocked/{HEX}.failure.json", "resume_failure"),
    ("preparation", f"source-resume-completed/{STEM}/{OTHER}.json", "resume_completed"),
    ("sam", f"results/{CHILD}.json", "result"),
    ("sam", f"results/{CHILD}.conflict-{OTHER}.json", "result_conflict"),
    ("sam", f"started/{CHILD}.json", "started"),
    ("sam", f"progress/{CHILD}/000001.json", "progress"),
    ("sam", f"progress/{CHILD}/1000000.json", "progress"),
    ("sam", f"wake-pending/{CHILD}.json", "wake_pending"),
    ("sam", f"wake-completed/{CHILD}.json", "wake_completed"),
)


def root_for(tmp_path, family, name=None):
    root = tmp_path / (name or family)
    for directory in CONTAINERS[family]:
        (root / directory).mkdir(parents=True)
    return root


def write(root, relative, text='{"retained":true}'):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def observe(*pairs, **kwargs):
    return auxiliary.observe_preparation_sam_auxiliaries(
        [auxiliary.AuxiliaryQueueContract(family, str(root)) for family, root in pairs],
        observed_at_epoch=123.0, **kwargs)


@pytest.mark.parametrize("family,relative,role", LAYOUTS)
def test_every_retained_variant_has_only_layout_authority(tmp_path, family, relative, role):
    root = root_for(tmp_path, family)
    path = write(root, relative)
    result = observe((family, root))
    assert result.complete and not result.blockers
    assert len(result.rows) == 1
    row = result.rows[0]
    assert (row.family, row.layout_role, row.row_path) == (family, role, str(path))
    assert row.raw_text == path.read_text()
    assert row.raw_sha256 == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert row.raw_size_bytes == path.stat().st_size
    assert result.scope == "preparation_sam_auxiliary_layouts_only"
    for flag in ("execution_authorized", "producer_seals_verified", "consumer_bindings_verified",
                 "history_chain_complete", "general_reference_inventory_complete", "consumer_fence_checked"):
        assert getattr(result, flag) is False
    assert result.mutations == 0


@pytest.mark.parametrize("family", CONTAINERS)
def test_stable_empty_roles_have_narrow_complete_evidence(tmp_path, family):
    root = root_for(tmp_path, family)
    result = observe((family, root))
    assert result.complete and result.rows == ()
    assert result.roots[0].family == family
    assert result.roots[0].root_identity is not None
    assert all(directory.status == "observed" for directory in result.roots[0].role_directories)


@pytest.mark.parametrize("relative", CONTAINERS["preparation"])
def test_missing_required_container_is_unknown(tmp_path, relative):
    root = root_for(tmp_path, "preparation")
    (root / relative).rmdir()
    result = observe(("preparation", root))
    assert not result.complete and result.blockers
    assert any(row.relative_path == relative and row.status == "missing_unproven"
               for row in result.roots[0].role_directories)


def test_missing_first_root_keeps_scope_and_later_positive(tmp_path):
    missing = tmp_path / "a-missing"
    root = root_for(tmp_path, "sam", "z-present")
    write(root, f"started/{CHILD}.json")
    result = observe(("preparation", missing), ("sam", root))
    assert not result.complete and len(result.rows) == 1
    assert result.roots[0].root_path == str(missing)
    assert result.roots[0].root_identity is None and result.roots[0].attempted_roles


@pytest.mark.parametrize("name", ["malformed-job.json", CHILD + ".conflict-" + OTHER.upper() + ".json"])
def test_unknown_regular_json_retains_raw_positive_without_binding(tmp_path, name):
    root = root_for(tmp_path, "sam")
    write(root, "results/" + name)
    result = observe(("sam", root))
    assert not result.complete and len(result.rows) == 1
    assert result.rows[0].layout_role == "unrecognized_row"
    assert result.rows[0].expected_container_role == "result"


@pytest.mark.parametrize("relative", ["progress/unknown/group.json", f"progress/{CHILD}/deeper/file.json",
                                      "results/staging.tmp"])
def test_unknown_group_deeper_or_non_json_is_not_traversed(tmp_path, relative):
    root = root_for(tmp_path, "sam")
    write(root, relative)
    result = observe(("sam", root))
    assert not result.complete and result.rows == ()


def test_same_bytes_and_hardlinks_remain_distinct_versions(tmp_path):
    root = root_for(tmp_path, "sam")
    first = write(root, f"results/{CHILD}.json")
    os.link(first, root / "results" / f"{CHILD}.conflict-{OTHER}.json")
    result = observe(("sam", root))
    assert result.complete and len(result.rows) == 2
    assert result.rows[0].raw_sha256 == result.rows[1].raw_sha256
    assert result.rows[0].row_path != result.rows[1].row_path


@pytest.mark.parametrize("text", ['[]', '{"a":1,"a":2}', '{"a":NaN}', '{"a":"\\ud800"}', 'invalid'])
def test_invalid_row_retains_other_positive(tmp_path, text):
    root = root_for(tmp_path, "sam")
    write(root, f"results/{CHILD}.json", text)
    write(root, f"started/{CHILD}.json", json.dumps({"digest": HEX}))
    result = observe(("sam", root))
    assert not result.complete and len(result.rows) == 1
    assert result.rows[0].layout_role == "started"
