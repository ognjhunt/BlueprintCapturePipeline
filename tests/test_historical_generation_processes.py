"""Bounded process-channel refusals; kernel visibility is a separate native gate."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_processes.py
import os

import pytest


def test_truncated_or_ambiguous_mapping_is_not_absence():
    from blueprint_pipeline.control_plane_lane_historical_processes import _mapping_inodes
    with pytest.raises(ValueError, match='process_unknown'):
        _mapping_inodes(b'not a kernel mapping record\n')


def test_mapping_identity_detects_alias_without_selected_path():
    from blueprint_pipeline.control_plane_lane_historical_processes import _mapping_inodes
    assert _mapping_inodes(b'1000-2000 rw-s 00000000 08:01 42 /an/alias (deleted)\n') == {(os.makedev(8, 1), 42)}


def test_invalid_start_identity_is_not_a_stable_process():
    from blueprint_pipeline.control_plane_lane_historical_processes import _process_start
    with pytest.raises(ValueError, match='process_unknown'):
        _process_start(b'123 (unknown) S 1\n', '123')


def test_ordinary_uid_cannot_certify_foreign_process_absence():
    from blueprint_pipeline.control_plane_lane_historical_processes import refuse_historical_process_references
    with pytest.raises(ValueError, match='native_unavailable'):
        refuse_historical_process_references({'members': []}, tick=lambda: None)
