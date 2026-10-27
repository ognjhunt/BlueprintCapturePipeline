"""Adopt registered website family packets into the existing beta screening queue.

The source profile supplies scene facts. Current consent, canonical intent,
operator candidate catalog, quotes, publication and allocator gates still own
whether any policy can run.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .decision_evidence_contracts import canonical_digest


def _profiles(max_sources):
    from .team_candidate_quick10_registry import load_registry, profile_for_capture
    registry = load_registry()
    if registry is None:
        return []
    captures = {str(row.get('capture_root')) for row in registry['profiles']
                if isinstance(row, dict) and row.get('family_scene_inputs_path')}
    settings = registry.get('derived_profiles') or {}
    root = Path(str(settings.get('root') or ''))
    if root.is_absolute() and root.is_dir() and not root.is_symlink():
        for path in sorted(root.glob('*.json')):
            if path.is_symlink():
                continue
            try:
                row = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and row.get('family_scene_inputs_path') and row.get('capture_root'):
                captures.add(row['capture_root'])
    profiles = []
    for capture in sorted(captures):
        profile = profile_for_capture(capture)
        if profile and profile.get('family_scene_inputs_path'):
            profiles.append(profile)
        if len(profiles) >= max_sources:
            break
    return profiles


def enqueue_registered_family_scenes(*, queue_root, max_sources=32):
    """Freeze queue work from exact registered facts; never allocate a resource."""
    from .beta_policy_candidate_catalog import load_default_catalog
    from .beta_policy_screening_capabilities import build_planning_inputs
    from .beta_policy_screening_coordinator import _lock, _read, _root, _write, enqueue_screening_campaign
    from .beta_policy_screening_plan import build_screening_plan
    from .beta_policy_screening_worker import require_current_source
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    from .task_evaluation_scene_configuration_submission_inputs import read
    from .website_task_family_scene_inputs import read_family_scene_inputs

    root = _root(queue_root)
    catalog = load_default_catalog()
    outcomes = []
    with _lock(root, '.family-source.lock'):
        states = root / 'sources'
        states.mkdir(exist_ok=True, mode=0o700)
        for profile in _profiles(max_sources):
            key = 'family-' + canonical_digest({'capture_root': profile['capture_root']})[7:39]
            outcome = {'request_key': key, 'status': 'blocked'}
            try:
                retained = read(profile['family_scene_inputs_path'], digest_field='inputs_digest')
                intent = reopen_scene_intent(retained['references']['scene_intent'])
                inputs = read_family_scene_inputs(profile['family_scene_inputs_path'])
                source = inputs['source']
                if (intent['intent_id'] != source['scene_intent_id']
                        or intent['intent_digest'] != source['scene_intent_digest']
                        or intent['request']['task']['task_id'] != source['task_id']
                        or intent['request']['task']['task_success_contract_digest'] != inputs['task_success_contract_digest']):
                    raise ValueError('beta_screening_family_source_intent_mismatch')
                context = require_current_source(source, site_id=profile['scene_id'])
                if context['scene_id'] != profile['scene_id']:
                    raise ValueError('beta_screening_family_source_scene_mismatch')
                campaign_id = 'screening-' + canonical_digest({'source': source,
                    'catalog_digest': catalog['catalog_digest'], 'protocol': 'beta_policy_screening.v1'})[7:39]
                existing = root / 'plans' / (campaign_id + '.json')
                if existing.exists():
                    plan = _read(existing)
                else:
                    planning = build_planning_inputs(catalog=catalog, profile=profile,
                        capture_root=Path(profile['capture_root']), source=source,
                        inputs_dir=root / 'inputs' / campaign_id)
                    plan = build_screening_plan(catalog, planning['scene'], campaign_id=campaign_id,
                        capabilities=planning['capabilities'], cost_quotes=planning['cost_quotes'],
                        candidate_ids=planning.get('candidate_ids'))
                outcome.update(enqueue_screening_campaign(plan, queue_root=root))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else ''
                outcome['blockers'] = [code if re.fullmatch(r'[a-z][a-z0-9_:,-]{0,199}', code)
                                       else 'beta_screening_family_source_unavailable']
            _write(states / (key + '.json'), outcome)
            outcomes.append(outcome)
    return {'sources': outcomes}
