"""Installed source identities for the finite registered local G1 producer.

Only the fixed native participant set is eligible. The root bootstrap publisher
independently reads these same installed files before publishing any authority.
"""
from __future__ import annotations

from pathlib import Path

from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_reference_budget import ReferenceCollectionBudget

SOURCE_MODULES = frozenset({
    "control_plane_lane_experiment_consumer", "control_plane_lane_experiment_completion",
    "native_g1_development_pair", "native_g1_development_worker", "native_g1_registered_containment",
    "native_g1_runtime_assembly", "native_g1_policy_server_supervisor", "native_g1_shared_scene_episode",
    "control_plane_scratch_lifetime", "control_plane_g1_lifetime_adapter",
})


def producer_source_identities():
    """Hash actual protected installed files under one bounded admission."""
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        result = {}
        for name in sorted(SOURCE_MODULES):
            raw, record = files.read(Path(__file__).parent / (name + ".py"), cap=1024 * 1024, protected=True)
            result[name] = retained._digest(raw, _work_budget=files.budget)
            files.verify_record(record)
        files.verify()
        return result
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
