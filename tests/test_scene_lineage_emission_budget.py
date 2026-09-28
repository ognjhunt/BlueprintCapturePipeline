# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lineage_budget.py
"""ADP-009D/day28: every retained child emission shares one bounded sink."""
import pytest


def budget_api():
    from blueprint_pipeline import task_evaluation_scene_lineage_budget
    return task_evaluation_scene_lineage_budget


def test_shared_sink_refuses_before_append_or_partial_counter_commit():
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=32, max_rows=2, max_references=2)
    rows = budget.rows()
    rows.append({'a': 'tiny'})
    before = dict(budget.used)
    with pytest.raises(c.RetainedEmissionBudgetError):
        rows.append({'a': 'x'*32})
    assert rows == [{'a': 'tiny'}] and budget.used == before


def test_nested_native_scope_and_shared_cap_both_apply_without_refunds():
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=128, max_rows=8, max_references=8)
    child = budget.scope(max_bytes=32, max_rows=1, max_references=8)
    child.rows().append({'a': 1})
    with pytest.raises(c.RetainedEmissionBudgetError):
        child.rows().append({'a': 2})
    assert budget.used['rows'] == 1
    second = budget.scope(max_bytes=128, max_rows=8, max_references=8)
    second.rows(reference=True).append({'ref': 1})
    assert budget.used['rows'] == 2 and budget.used['references'] == 1


def test_framing_measure_refuses_without_encoding_or_charging_document_again(monkeypatch):
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=12, max_rows=8, max_references=8)
    rows = budget.rows([{'a': 1}])
    before = dict(budget.used)
    with pytest.raises(c.RetainedEmissionBudgetError):
        budget.check_document({'rows': rows})
    assert budget.used == before


def test_provenance_occurrences_charge_before_consuming_next_generator_item():
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=128, max_rows=1, max_references=8)
    seen = []
    def values():
        for i in range(3):
            seen.append(i)
            yield {'proof': i}
    with pytest.raises(c.RetainedEmissionBudgetError):
        budget.reserve_provenance(values())
    assert seen == [0, 1] and budget.used['rows'] == 1


@pytest.mark.parametrize('child', ['preparation', 'source', 'seed', 'downstream'])
def test_private_child_path_charges_emissions_before_bulk_output_encoding(child, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as preparation
    from blueprint_pipeline import task_evaluation_scene_source_attempt_lineage as source
    from blueprint_pipeline import task_evaluation_scene_inventory_seed as seed
    from blueprint_pipeline import task_evaluation_scene_downstream_inventory as downstream
    from tests.test_scene_preparation_lineage import fixture as preparation_fixture
    from tests.test_scene_source_attempt_lineage import fixture as source_fixture
    from tests.test_scene_inventory_preparations import fixture as seed_fixture
    from tests.test_scene_downstream_execution import fixture as downstream_fixture
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=1024*1024, max_rows=0, max_references=100)
    if child == 'preparation':
        a = preparation_fixture()
        def call():
            return preparation._join(a['intent_id'], a['intent_record'], a['preparation_links'],
                a['preparation_envelopes'], a['configuration_attempt_records'], a['roots'], emission_budget=budget)
    elif child == 'source':
        a = source_fixture()
        def call():
            return source._join(a['intent_id'], a['intent_record'], a['attempt_records'], a['snapshot_records'],
                a['factory_records'], a['submission_records'], a['roots'], emission_budget=budget)
    elif child == 'seed':
        a = seed_fixture()
        def call():
            return seed._join(a['intent_id'], a['records'], a['roots'], emission_budget=budget)
    else:
        a = downstream_fixture()
        def call():
            return downstream._join(a['intent_id'], a['seed_records'], a['downstream_records'], a['roots'], emission_budget=budget)
    original = preparation._encoded
    def encoded(value):
        assert not (isinstance(value, dict) and value.get('schema_version') in {
            'task_evaluation_scene_preparation_lineage.v1', 'task_evaluation_scene_source_attempt_lineage.v1',
            'task_evaluation_scene_inventory_seed.v1', 'task_evaluation_scene_downstream_inventory.v1'})
        return original(value)
    monkeypatch.setattr(preparation, '_encoded', encoded)
    with pytest.raises(c.RetainedEmissionBudgetError):
        call()


@pytest.mark.parametrize('kind', ['rows', 'references'])
def test_exhausted_shared_occurrence_refuses_before_native_measurement(monkeypatch, kind):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as downstream
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=128, max_rows=0 if kind == 'rows' else 4,
                                    max_references=0 if kind == 'references' else 4)
    rows = downstream.OutputRows({'rows': 0, 'bytes': 0}, {'MAX_ROWS': 4, 'MAX_OUTPUT_BYTES': 128},
                                 emission_budget=budget, reference=kind == 'references')
    monkeypatch.setattr(downstream, 'bounded_size', lambda *a: pytest.fail('native traversal before exhausted shared cap'))
    with pytest.raises(c.RetainedEmissionBudgetError):
        rows.append({'a': 1})


def test_native_measurement_receives_smaller_shared_remaining_allowance(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as downstream
    c = budget_api()
    budget = c.RetainedEmissionBudget(max_bytes=16, max_rows=4, max_references=4)
    original = downstream.bounded_size
    seen = []
    def measure(value, limit):
        seen.append(limit)
        assert limit == 16
        return original(value, limit)
    monkeypatch.setattr(downstream, 'bounded_size', measure)
    rows = downstream.OutputRows({'rows': 0, 'bytes': 0}, {'MAX_ROWS': 4, 'MAX_OUTPUT_BYTES': 128}, emission_budget=budget)
    rows.append({'a': 1})
    assert seen == [16]


def sealed_row():
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as downstream
    value = {'schema_version': 'fixture.v1'}
    value['result_digest'] = downstream.canonical_digest(value, digest_field='result_digest')
    proof = {'role': 'fixture', 'path': '/retained/result.json', 'sha256': 'sha256:'+'a'*64,
             'size_bytes': 1, 'seal_field': None, 'seal_digest': None}
    return value, proof


def test_proof_future_seal_growth_refuses_before_next_emission_or_child():
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as downstream
    c = budget_api()
    row = sealed_row()
    initial = downstream.bounded_size(row[1], 4096)
    budget = c.RetainedEmissionBudget(max_bytes=initial, max_rows=4, max_references=4)
    continued = []
    with pytest.raises(c.RetainedEmissionBudgetError):
        downstream.Context({'fixture': [row]}, {}, {'MAX_ROWS': 4, 'MAX_OUTPUT_BYTES': 4096},
                           'intent', emission_budget=budget)
        downstream.seal(row, 'result_digest')
        continued.append('next child')
    assert continued == [] and row[1]['seal_field'] is None


def test_each_emitted_proof_alias_reserves_future_seal_bytes_without_changing_output():
    from blueprint_pipeline import task_evaluation_scene_downstream_contracts as downstream
    c = budget_api()
    row = sealed_row()
    budget = c.RetainedEmissionBudget(max_bytes=4096, max_rows=8, max_references=8)
    context = downstream.Context({'fixture': [row]}, {}, {'MAX_ROWS': 8, 'MAX_OUTPUT_BYTES': 4096, 'MAX_REFERENCES': 8},
                                 'intent', emission_budget=budget)
    context.raw_ref({k: row[1][k] for k in ('path', 'sha256', 'size_bytes')}, row[1])
    reserved = budget.used['bytes']
    downstream.seal(row, 'result_digest')
    actual = sum(downstream.bounded_size(value, 4096) for value in [*context.raw, *context.obligations])
    assert reserved >= actual and budget.used['bytes'] == reserved
    assert context.obligations[0]['source_provenance'][0] is row[1]
    assert context.obligations[0]['matched_provenance'] is row[1]
    assert row[1]['seal_field'] == 'result_digest'
