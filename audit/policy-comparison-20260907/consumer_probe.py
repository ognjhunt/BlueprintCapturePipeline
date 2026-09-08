"""Synthetic producer cohort packet. Never a policy evaluation or real media."""
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_result_delivery import materialize_policy_canary_result_delivery
from tests.test_task_evaluation_policy_canary_result_delivery import _result, _closure

with TemporaryDirectory(prefix='policy-comparison-cohort-') as scratch:
    root = Path(scratch)
    evidence = root / 'evidence'
    evidence.mkdir()
    result = _result(evidence)
    template = result['episodes'][0]
    candidates = ['pi05_droid', 'groot_n17_droid']
    result['episodes'] = []
    for cell in range(10):
        for index, candidate in enumerate(candidates):
            row = deepcopy(template)
            valid = cell < 6 if index == 0 else cell < 2 or cell >= 6
            success = cell < 2 if index == 0 else cell >= 6
            row.update(candidate_id=candidate, cell_id=f'cell-{cell}', seed=100 + cell,
                       policy_outcome_interpretable=valid)
            row['episode']['episode_id'] = f'{candidate}-cell-{cell}'
            row['episode']['score'].update(status='scored' if valid else 'undetermined',
                                          task_succeeded=success if valid else None)
            result['episodes'].append(row)
    result['result_digest'] = canonical_digest(result, digest_field='result_digest')
    closure = {name: _closure(root / (name + '.json'), flag=flag) for name, flag in (
        ('billing', 'official_billing_sealed'), ('teardown', 'teardown_completed'),
        ('provider_zero', 'provider_zero_verified'))}
    delivery = materialize_policy_canary_result_delivery(run_root=root, run_id='offline-cohort-probe',
        result_status='completed_unqualified', session_result=result, evidence_root=evidence,
        closure_records=closure)
    packet = {'fixture_only': True, 'claim_ceiling': 'offline_contract_test',
        'synthetic_media_and_closure_records': True,
        'website_commit': '5b07fe48e93cabf2a758b7740aab3d6594cbdd10',
        'producer_candidate_results': delivery['candidate_results'],
        'episode_cells': [{'candidate_id': row['candidate_id'], 'cell_id': row['cell_id'],
            'seed': row['seed'], 'interpretable': row['policy_outcome_interpretable'],
            'success': row['episode']['score']['task_succeeded']} for row in result['episodes']],
        'independent_paired_cohort': {'cells': ['cell-0', 'cell-1'], 'pairs': 2,
            'pi05_only_wins': 2, 'groot_only_wins': 0, 'two_sided_sign_test_p': .5},
        'integration_instruction': 'Keep marginal summaries separate from the paired headline; no winner or qualification.'}
    print(json.dumps(packet, indent=2))
