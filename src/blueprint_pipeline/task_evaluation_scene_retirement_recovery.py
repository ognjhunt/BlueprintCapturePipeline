"""Resume only an exact protected action; no fresh origin or orphan adoption."""
from pathlib import Path

from .task_evaluation_scene_retirement_access import _require
from .task_evaluation_scene_retirement_authority import selected_document, load_document
from .task_evaluation_scene_retirement_intent_receipt import _projection, _resume


def select_retirement(policy, authority, allowance):
    consent = authority['consent']
    context = policy.get('reference_context')
    if not isinstance(context, dict):
        return None
    path = Path(context['roots']['intent_root']) / consent['intent_id'] / 'scene-retired.v1.json'
    try:
        _, reference = load_document(path, maximum=65536)
    except FileNotFoundError:
        return None
    _, _, _, _, pending, _ = _projection(policy, consent, reference, allowance)
    journal = _resume(policy, pending, allowance)
    initial = selected_document(journal.initial_ref, maximum=16 * 1024 * 1024, protected=True)
    _require(initial.get('schema_version') == 'scene_retirement_journal.v1'
             and initial.get('status') == 'pending'
             and initial.get('intent_id') == consent['intent_id']
             and initial.get('intent_raw_ref') == consent['intent_raw_ref']
             and initial.get('plan_raw_ref') == consent['plan_raw_ref']
             and initial.get('consent_raw_ref') == authority['consent_raw_ref']
             and initial.get('policy_sha256') == consent['policy_sha256']
             and initial.get('cohort_sha256') == consent['cohort_sha256']
             and initial.get('members') == consent['members'], 'scene_retirement_resume_unproven')
    return journal, reference, initial


def bind_original_allowance(journal, initial, allowance):
    selected = initial.get('action_allowance')
    for event in journal.events:
        if event['event'] == 'allowance_reserved':
            _require(event['member_key'] == 'action'
                     and event['evidence'].get('phase') in {'remove', 'resume-readback'},
                     'scene_retirement_resume_allowance_unproven')
            selected = event['evidence'].get('action_allowance')
    allowance.bind_resume(selected)


def reserve_phase(journal, preserved, *, readback=False):
    """Persist an upper bound before physical work, including a crash prefix.

    Actual reads still charge natively. This conservative reservation is never
    refunded; a retry pays its next phase against the original remaining cap.
    """
    allowance = journal.allowance
    if readback:
        allowance.charge('remote_bytes', preserved['archive']['size_bytes'])
    else:
        allowance.charge('local_bytes', 2 * sum(row['size_bytes'] for row in preserved['files']))
    evidence = dict(phase='resume-readback' if readback else 'remove',
                    action_allowance=allowance.checkpoint())
    journal.preflight([('allowance_reserved', 'action', evidence)])
    journal.append('allowance_reserved', member_key='action', evidence=evidence)


def resumed_generations(engine, policy, consent, journal, initial):
    result = []
    _require(len(initial['generations']) == len(consent['members']), 'scene_retirement_resume_unproven')
    for index, member in enumerate(consent['members']):
        current, _ = engine._generation(policy, member, expected_states={'active', 'restored-active', 'retiring', 'retired'})
        original = initial['generations'][index]
        if current['state'] in {'active', 'restored-active'}:
            _require(current == original, 'scene_retirement_generation_changed')
        else:
            _require(current.get('retirement_token') == journal.token
                     and current.get('inventory_sha256') == member['inventory_sha256'],
                     'scene_retirement_generation_changed')
            events = [event for event in journal.events if event['member_key'] == str(index)
                      and event['raw_ref']['sha256'] == current.get('journal_sha256')]
            _require(len(events) == 1 and events[0]['event'] == (
                'retiring' if current['state'] == 'retiring' else 'member_removed'),
                'scene_retirement_generation_changed')
            proof = events[0]['evidence']
            _require(proof.get('generation_id') == current['generation_id']
                     and proof.get('inventory_sha256') == member['inventory_sha256'],
                     'scene_retirement_generation_changed')
        result.append(current)
    return result


def removed_inode_counts(journal):
    """Only completed native unlink events establish prior union link changes."""
    plans = {event['raw_ref']['sha256']: event for event in journal.events
             if event['event'] == 'leaf_unlink_planned'}
    result, seen = {}, set()
    for event in journal.events:
        if event['event'] != 'leaf_unlinked':
            continue
        selected = event['evidence']['leaf_plan_raw_ref']
        _require(selected['sha256'] in plans, 'scene_retirement_journal_chain_unproven')
        plan = plans[selected['sha256']]
        _require(plan['raw_ref'] == selected and plan['member_key'] == event['member_key'],
                 'scene_retirement_journal_chain_unproven')
        if selected['sha256'] not in seen:
            seen.add(selected['sha256'])
            inode = tuple(plan['evidence']['physical_identity'][:2])
            result[inode] = result.get(inode, 0) + 1
    return result
