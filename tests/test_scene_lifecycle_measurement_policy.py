# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_measurement.py
"""Storage law and dependency proof never imply exclusive scene ownership."""
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def test_only_documented_bind_aliases_receive_storage_policy():
    from blueprint_pipeline.task_evaluation_scene_lifecycle_measurement import classification
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    known = classification('/mnt/blueprint-work/task-evaluation-inputs/completed-scene-preparation-inputs/prep-1', budget)
    assert known['storage_class'] == 'work'
    assert known['canonical_policy_path'] == '/var/lib/blueprint/task-evaluation-inputs/completed-scene-preparation-inputs/prep-1'
    assert classification('/mnt/blueprint-work/unknown-copy/prep-1', budget)['storage_class'] == 'unclassified'
    sam = classification('/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/parent/child', budget)
    assert sam['storage_class'] == 'cache'
    assert sam['storage_policy_allows_cleanup'] is False


def test_original_sam_dependency_requires_verified_adoption_and_exact_receipt_hierarchy():
    from blueprint_pipeline.task_evaluation_scene_lifecycle_measurement import members
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from tests.test_scene_source_family_adoption import fixture
    from tests.test_scene_source_family_website import api
    args = fixture()
    source = api().join_retained_scene_source_family_inventory(**args)
    historical = {'declared_lexical_members': [], 'source_family_inventory': source}
    budget = ReferenceCollectionBudget(monotonic=lambda: 0)
    sink = RetainedEmissionBudget(max_bytes=16*1024*1024, max_rows=10000, max_references=10000, work_budget=budget)
    rows = members(historical, budget, sink, roots=args['roots'])
    original = [row for group in rows.values() for row in group if row['kind'] == 'sam_original_execution_dependency']
    assert original and all(row['binding_strength'] == 'verified_original_prefix_receipt_dependency' for row in original)
    assert all(row['original_owner_transfer_authorized'] is False for row in original)
    source['adoption_observations'][0]['prefix_binding_verified'] = False
    fresh = ReferenceCollectionBudget(monotonic=lambda: 0)
    sink = RetainedEmissionBudget(max_bytes=16*1024*1024, max_rows=10000, max_references=10000, work_budget=fresh)
    assert not any(row['kind'] == 'sam_original_execution_dependency'
                   for group in members(historical, fresh, sink, roots=args['roots']).values() for row in group)
