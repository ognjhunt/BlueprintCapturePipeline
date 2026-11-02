"""The provider output ceiling counts what a run can return, not the toolchain it was sent."""
from __future__ import annotations

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_provider_contracts import (
    PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER,
    _provider_transfer_byte_budget,
)
from blueprint_pipeline.task_evaluation_scene_configuration_transfer_budget import (
    SCENE_OUTPUT_ARCHIVE_FLOOR_BYTES,
)

BUNDLE = 6_115_738_290  # capture-447ae4f4's activation bundle, 2026-11-01


def test_a_receipt_without_its_toolchain_size_keeps_the_whole_bundle_ceiling():
    _, upload = _provider_transfer_byte_budget({"bundle_size_bytes": BUNDLE})
    assert upload == PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER * BUNDLE  # 12.2 GB, as before


def test_a_bundle_that_states_its_toolchain_reserves_only_returnable_bytes():
    receipt = {"bundle_size_bytes": BUNDLE,
               "toolchain_archive": {"fragment_size_bytes": BUNDLE - 120_000_000}}
    download, upload = _provider_transfer_byte_budget(receipt)
    assert upload == SCENE_OUTPUT_ARCHIVE_FLOOR_BYTES  # 2 x 0.12 GB is under the floor
    assert download > BUNDLE  # the download still carries the whole bundle
    big_scene = {"bundle_size_bytes": BUNDLE + 9_000_000_000,
                 "toolchain_archive": {"fragment_size_bytes": BUNDLE}}
    assert _provider_transfer_byte_budget(big_scene)[1] == PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER * 9_000_000_000


@pytest.mark.parametrize("fragment", [True, 0, -1, BUNDLE, BUNDLE + 1, "6000000000", None])
def test_an_implausible_toolchain_size_falls_back_to_the_whole_bundle(fragment):
    receipt = {"bundle_size_bytes": BUNDLE, "toolchain_archive": {"fragment_size_bytes": fragment}}
    assert _provider_transfer_byte_budget(receipt)[1] == PROVIDER_OUTPUT_UPLOAD_BUNDLE_MULTIPLIER * BUNDLE


def test_the_toolchain_rule_never_raises_a_small_bundles_ceiling():
    small = {"bundle_size_bytes": 50_000_000, "toolchain_archive": {"fragment_size_bytes": 40_000_000}}
    whole = _provider_transfer_byte_budget({"bundle_size_bytes": 50_000_000})[1]
    assert _provider_transfer_byte_budget(small)[1] == whole  # 1 GB minimum, not the 8 GB floor
