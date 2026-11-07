"""Adopt registered website family packets into the existing beta screening queue.

The source profile supplies scene facts. Current consent, canonical intent,
operator candidate catalog, quotes, publication and allocator gates still own
whether any policy can run.

A navigation job's registered scene inputs (``navigation_scene_inputs_path``,
``website_navigation_scene_inputs``) are picked up the same way. Its current-source
identity is the one its website intake receipt binds (``website_navigation_intake``),
bound here once the Pipeline has staged that intake's intent; while the record has no
receipt the source holds by name and the next source goes on.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .decision_evidence_contracts import canonical_digest


def _registered_source(row):
    return bool(row.get('family_scene_inputs_path') or row.get('navigation_scene_inputs_path'))


def _profile_captures():
    from .team_candidate_quick10_registry import load_registry
    registry = load_registry()
    if registry is None:
        return []
    captures = {row['capture_root'] for row in registry['profiles']
                if isinstance(row, dict) and _registered_source(row)
                and isinstance(row.get('capture_root'), str) and row['capture_root']}
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
            if (isinstance(row, dict) and _registered_source(row)
                    and isinstance(row.get('capture_root'), str) and row['capture_root']):
                captures.add(row['capture_root'])
    return sorted(captures)


def enqueue_registered_family_scenes(*, queue_root, max_sources=32, packet_queue_root=None):
    """Freeze queue work from exact registered facts; never allocate a resource.

    ``packet_queue_root`` is the team scene packet queue the G1 site registration reads
    (``native_g1_site_registration.ensure_site_registration``), as the packet pass passes it.
    """
    from .beta_policy_candidate_catalog import load_default_catalog
    from .beta_policy_screening_capabilities import build_planning_inputs
    from .beta_policy_screening_coordinator import (
        _lock, _read, _root, _write, enqueue_screening_campaign, latest_campaign, replan_scope, supersede_campaign,
    )
    from .beta_policy_screening_job_ceiling import plan_within_job_ceiling
    from .beta_policy_screening_team import catalog_for_profile
    from .beta_policy_screening_worker import implementation_commit, require_current_source
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    from .task_evaluation_scene_configuration_submission_inputs import read
    from .website_task_family_scene_inputs import read_family_scene_inputs
    from .team_candidate_quick10_registry import profile_for_capture
    from .website_candidate_ranking import retained_priority
    from .website_candidate_selection import retain_campaign_catalog

    if type(max_sources) is not int or not 1 <= max_sources <= 100:
        raise ValueError('beta_screening_family_source_limit_invalid')
    root = _root(queue_root)
    base_catalog = load_default_catalog()
    outcomes = []
    with _lock(root, '.family-source.lock'):
        states = root / 'sources'
        states.mkdir(exist_ok=True, mode=0o700)
        cursor_path = root / 'family_source_cursor.json'
        last = _read(cursor_path).get('last', '') if cursor_path.exists() else ''
        if not isinstance(last, str):
            raise ValueError('beta_screening_family_source_cursor_invalid')
        captures = _profile_captures()
        captures = [capture for capture in captures if capture > last] + [capture for capture in captures if capture <= last]
        for capture in captures[:max_sources]:
            key = 'family-' + canonical_digest({'capture_root': capture})[7:39]
            outcome = {'request_key': key, 'status': 'blocked'}
            try:
                profile = profile_for_capture(capture)
                navigation = bool(profile and not profile.get('family_scene_inputs_path')
                                  and profile.get('navigation_scene_inputs_path'))
                if not profile or not (profile.get('family_scene_inputs_path') or navigation):
                    raise ValueError('beta_screening_family_profile_unavailable')
                if navigation:
                    # A scene kept as filmed gets its scene intent from the job's website navigation
                    # intake; its intake receipt binds the source (website_navigation_intake), bound here
                    # once the intent is staged. With no receipt this refuses by name before the record
                    # and its packet are read (and hashed) again on every pass.
                    from . import website_navigation_scene_inputs as navigation_inputs
                    from .website_navigation_intake import bind_navigation_intake_receipt, receipt_exists
                    record_path = profile['navigation_scene_inputs_path']
                    if not receipt_exists(record_path):
                        bind_navigation_intake_receipt(record_path=record_path, capture_root=profile['capture_root'])
                    if not receipt_exists(record_path):
                        raise navigation_inputs.WebsiteNavigationSceneInputsError(navigation_inputs.SOURCE_UNBOUND)
                    source = navigation_inputs.screening_source(
                        navigation_inputs.read_navigation_scene_inputs(record_path))
                    # The owner's standing approval (2026-11-06): the service registers the site's G1
                    # candidates itself, before the catalog is read, as the packet pass does. A new row
                    # changes the catalog's revision, so the site plans again. A G1 problem is recorded,
                    # never a stop for this site.
                    try:
                        from .native_g1_site_registration import ensure_site_registration
                        from .team_scene_packet_worker import current_release_commit

                        if packet_queue_root is None:
                            raise ValueError('beta_screening_team_scene_packet_queue_root_not_configured')
                        registered = ensure_site_registration(site_id=str(profile['scene_id']), source=source,
                            source_queue=packet_queue_root, release=current_release_commit())
                        if registered.get('status') == 'registered':
                            outcome['g1_registration'] = {'status': 'registered'}
                    except Exception as exc:  # noqa: BLE001 - only privacy, rights and money stop a site
                        from .bounded_error import error_description

                        outcome['g1_registration'] = {'status': 'failed', 'error': error_description(exc)}
                else:
                    retained = read(profile['family_scene_inputs_path'], digest_field='inputs_digest')
                    intent = reopen_scene_intent(retained['references']['scene_intent'])
                    inputs = read_family_scene_inputs(profile['family_scene_inputs_path'])
                    source = inputs['source']
                    if (intent['intent_id'] != source['scene_intent_id']
                            or intent['intent_digest'] != source['scene_intent_digest']
                            or intent['request']['task']['task_id'] != source['task_id']
                            or intent['request']['task']['task_success_contract_digest']
                            != inputs['task_success_contract_digest']):
                        raise ValueError('beta_screening_family_source_intent_mismatch')
                catalog = catalog_for_profile(profile, base_catalog)
                context = require_current_source(source, site_id=profile['scene_id'])
                if context['scene_id'] != profile['scene_id']:
                    raise ValueError('beta_screening_family_source_scene_mismatch')
                campaign_id = 'screening-' + canonical_digest({'source': source,
                    'catalog_digest': catalog['catalog_digest'], 'protocol': 'beta_policy_screening.v1'})[7:39]
                existing = root / 'plans' / (campaign_id + '.json')
                replan = superseded = None
                if existing.exists():
                    # Unprepared candidates bound to a commit replaced by a deploy are planned again.
                    current = latest_campaign(campaign_id, queue_root=root)
                    commit = implementation_commit()
                    replan = replan_scope(current, implementation_commit=commit, queue_root=root)
                    if replan is not None:
                        superseded = current
                        campaign_id = 'screening-' + canonical_digest({'source': source,
                            'catalog_digest': catalog['catalog_digest'], 'protocol': 'beta_policy_screening.v1',
                            'supersedes': current, 'implementation_commit': commit})[7:39]
                if existing.exists() and replan is None:
                    plan = _read(root / 'plans' / (current + '.json'))
                else:
                    priority = retained_priority(Path(profile['capture_root']) / 'pipeline',
                                                 task_context_digest=source['task_context_digest'])
                    planning = build_planning_inputs(catalog=catalog, profile=profile,
                        capture_root=Path(profile['capture_root']), source=source,
                        inputs_dir=root / 'inputs' / campaign_id, priority=priority)
                    candidate_ids = planning.get('candidate_ids')
                    if replan is not None:
                        candidate_ids = [key for key in (candidate_ids or replan) if key in set(replan)]
                    plan = plan_within_job_ceiling(catalog, planning['scene'], campaign_id=campaign_id,
                        queue_root=root, context=context, capture_root=Path(profile['capture_root']),
                        superseding=superseded,
                        capabilities=planning['capabilities'], cost_quotes=planning['cost_quotes'],
                        candidate_ids=candidate_ids, priority=planning.get('candidate_priority'),
                        intake_ranking=bool(priority))
                    retain_campaign_catalog(queue_root=root, campaign_id=campaign_id, catalog=catalog)
                outcome.update(enqueue_screening_campaign(plan, queue_root=root) if superseded is None else
                               supersede_campaign(superseded, plan, implementation_commit=commit, queue_root=root))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else ''
                outcome['blockers'] = [code if re.fullmatch(r'[a-z][a-z0-9_:,-]{0,199}', code)
                                       else 'beta_screening_family_source_unavailable']
            _write(states / (key + '.json'), outcome)
            # Advance after every retained outcome, including authority holds.
            # A crash before this write only replays idempotent queue adoption.
            _write(cursor_path, {'last': capture})
            outcomes.append(outcome)
    return {'sources': outcomes}
