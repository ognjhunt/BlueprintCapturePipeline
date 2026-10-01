"""Actual fixture assertion projections; never native or owner authority.

An expired unconsumed approval does not become the selected journal intent.
A consumed pending approval remains selected through a fresh DELETE resume.
"""
# Covers: tests/historical_generation_native_acceptance.py
import ast
from copy import deepcopy
from pathlib import Path

import pytest


def _assert_selected_row(values):
    source = Path(__file__).with_name('historical_generation_native_acceptance.py')
    tree = ast.parse(source.read_text())
    # Execute the actual terminal assertion block, rather than a copied
    # expectation. The rest of the native fixture cannot run on macOS.
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'planned_id'
                                  for target in child.targets) for child in node.body)]
    assert len(candidates) == 1
    module = ast.Module(body=deepcopy(candidates[0].body[:6]), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), values)
    return values['reconciled']


@pytest.mark.parametrize('scenario', ['fresh_unconsumed', 'pending_resume', 'ordinary'])
@pytest.mark.parametrize('wrong_selected_row', [False, True])
def test_terminal_assertion_selects_actual_consumed_approval(scenario, wrong_selected_row):
    old = dict(decision_id='old expired decision')
    current = dict(decision_id='fresh current decision', packet=dict(
        resume_from={'event_digest': 'actual prior intent'} if scenario == 'pending_resume' else None))
    expected = old if scenario == 'pending_resume' else current
    row = dict(kind='restore_intent', body=dict(phase='reconciled',
        decision_id=expected['decision_id'], uncertain=True, credited_removed_allocated_bytes=0))
    if wrong_selected_row:
        row['body']['decision_id'] = 'unselected approval'
    values = dict(restore_events=[row], additional_reconciliations=[],
                  initial_discard=old, reconciliation=current, delete_expiry=scenario != 'ordinary')
    if wrong_selected_row:
        with pytest.raises(AssertionError):
            _assert_selected_row(values)
    else:
        assert _assert_selected_row(values) == [row]
