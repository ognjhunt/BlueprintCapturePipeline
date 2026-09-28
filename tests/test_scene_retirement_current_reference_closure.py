"""Current action admission keeps raw identity distinct from lineage metadata.

These isolated tests prove owner-selector admission only. They stub independent
member and consumer admission and do not claim a finished action or clearance.
"""
import pytest

from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance


@pytest.mark.parametrize('changed',[None,'path','sha256','size_bytes'])
def test_native_intent_provenance_preserves_exact_raw_owner_tuple(monkeypatch,changed):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from tests.test_scene_inventory_history import fixture
    args=fixture()
    context={}
    from blueprint_pipeline.task_evaluation_scene_preparation_lineage import _record
    _,proof=_record(args['records']['intent'],'intent',set())
    reference={key:proof[key] for key in ('path','sha256','size_bytes')}
    if changed=='path':
        reference['path']+='.foreign'
    elif changed=='sha256':
        reference['sha256']='sha256:'+'f'*64
    elif changed=='size_bytes':
        reference['size_bytes']+=1
    fresh=dict(schema_version='task_evaluation_scene_lifecycle_plan.v1',
        finished_observation={'status':'completed'},historical_lineage={},selected_intent_provenance=proof,
        reference_observation={'blockers':[],'child_scopes':[{'child':name,'complete':True} for name in ('pins','primary_queues','auxiliary_queues')],
                               'record_dispositions':[],'protections':[]},
        reference_keeps=[],other_owner_capture_keeps=[])
    monkeypatch.setattr(engine,'build_scene_lifecycle_plan',lambda **kwargs:fresh)
    monkeypatch.setattr(engine,'_plan_members',lambda *args:None)
    monkeypatch.setattr(engine,'_installed_cohort',lambda *args:None)
    monkeypatch.setattr(engine,'_current_readers',lambda *args:None,raising=False)
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    consent={'intent_id':args['intent_id'],'intent_raw_ref':reference}
    retained={'schema_version':'task_evaluation_scene_lifecycle_plan.v1',
              'intent_id':args['intent_id'],'planner_context':context}
    if changed:
        with pytest.raises(ValueError,match='scene_retirement_owner_changed'):
            engine._current_plan({'reference_context':context},consent,retained,allowance,lambda:200,lambda:0)
    else:
        assert engine._current_plan({'reference_context':context},consent,retained,allowance,lambda:200,lambda:0)==fresh


def test_genuine_hashed_participant_subset_is_not_a_complete_consumer_cohort(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from tests.test_scene_retirement_connected_acceptance import _installed_cohort
    rows=_installed_cohort()
    assert len(rows)>1
    entered=[]
    monkeypatch.setattr(engine.importlib,'import_module',lambda name:entered.append(name))
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    with pytest.raises(ValueError,match='scene_retirement_cohort_unproven'):
        engine._installed_cohort({'consumer_cohort':[rows[0]]},allowance)
    assert entered==[], 'missing fixed consumer coverage must refuse before a target import'


def test_fixed_cohort_catalogue_covers_every_actual_participating_entrypoint():
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from tests.test_scene_retirement_connected_acceptance import _installed_cohort
    assert engine._REQUIRED_COHORT==frozenset(row['entrypoint'] for row in _installed_cohort())


def test_native_measured_rows_are_accepted_without_accepting_foreign_list_callbacks(tmp_path):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    root=str(tmp_path.resolve()/'member')
    row={'path':root,'status':'observed_scoped_metadata','keeps':[]}
    consent={'members':[{'canonical_path':root,'class':'host'}],'private_archive_classes':[]}
    native=RetainedEmissionBudget(max_bytes=4096,max_rows=10,max_references=10).rows([row])
    engine._plan_members({'measured_members':native},consent)
    entered=[]
    class Foreign(list):
        def __iter__(self):
            entered.append('iteration')
            return super().__iter__()
    with pytest.raises(ValueError,match='scene_retirement_members_unproven'):
        engine._plan_members({'measured_members':Foreign([row])},consent)
    assert entered==[]


def test_empty_complete_current_records_do_not_clear_unknown_processes(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline import task_evaluation_scene_retirement_supervisor as supervisor
    from tests.test_scene_inventory_history import fixture
    from blueprint_pipeline.task_evaluation_scene_preparation_lineage import _record
    args=fixture()
    _,proof=_record(args['records']['intent'],'intent',set())
    context={}
    fresh=dict(schema_version='task_evaluation_scene_lifecycle_plan.v1',
        finished_observation={'status':'completed'},historical_lineage={},selected_intent_provenance=proof,
        reference_observation={'blockers':[],'child_scopes':[{'child':name,'complete':True}
            for name in ('pins','primary_queues','auxiliary_queues')],
            'record_dispositions':[],'protections':[]},
        measured_members=[],reference_keeps=[],other_owner_capture_keeps=[])
    monkeypatch.setattr(engine,'build_scene_lifecycle_plan',lambda **kwargs:fresh)
    monkeypatch.setattr(engine,'_plan_members',lambda *args:None)
    monkeypatch.setattr(engine,'_installed_cohort',lambda *args:None)
    # No loaded native authority exists. A complete finite declaration cannot
    # authorize the unknown HTTP/manual/old-code consumer cohort.
    monkeypatch.setattr(supervisor,'require_current_reader_closure',None,raising=False)
    reference={key:proof[key] for key in ('path','sha256','size_bytes')}
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    with pytest.raises(ValueError,match='scene_retirement_reader_closure_unproven'):
        engine._current_plan({'reference_context':context},{'intent_id':args['intent_id'],
            'intent_raw_ref':reference},{'schema_version':'task_evaluation_scene_lifecycle_plan.v1',
            'intent_id':args['intent_id'],'planner_context':context},allowance,lambda:200,lambda:0)


def test_native_measurement_keep_rows_preserve_exact_protection_semantics(tmp_path):
    from blueprint_pipeline import task_evaluation_scene_retirement as engine
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    sink=RetainedEmissionBudget(max_bytes=4096,max_rows=10,max_references=10)
    path=str(tmp_path.resolve())
    member={'canonical_path':path,'class':'host'}
    row={'path':path,'status':'observed_scoped_metadata','storage_class':'host','keeps':sink.rows()}
    engine._plan_members({'measured_members':sink.rows([row])},
        {'members':[member],'private_archive_classes':['host']})
    row['keeps'].append('external_hardlink_dependency')
    with pytest.raises(ValueError,match='scene_retirement_shared_or_unresolved_member'):
        engine._plan_members({'measured_members':[row]},
            {'members':[member],'private_archive_classes':['host']})
