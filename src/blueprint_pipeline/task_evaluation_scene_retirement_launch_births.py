"""The real launch writer's original authenticated owner-to-output boundary."""
from __future__ import annotations

import os
from pathlib import Path

from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _require
from .task_evaluation_scene_retirement_authority import load_document, selected_document
from .task_evaluation_scene_retirement_producer_births import current_directory_generation, existing_scene_directory


def enroll_launch_output(target, *, request, profile, now=None):
    if access._policy() is None:
        return None
    from .task_evaluation_scene_attempt_binding import OWNER_FIELDS, POLICY_FIELDS
    if not (OWNER_FIELDS | POLICY_FIELDS).intersection(profile):
        return None  # Existing legacy profiles are not adopted or made eligible.
    from .task_evaluation_launch_dispatcher import validate_launch_profile, validate_launch_request
    from .task_evaluation_scene_execution_authority import require_scene_execution_authority
    code = 'scene_retirement_launch_storage_authority_unproven'
    target = _canonical(str(target))
    with access.scene_access(target):
        _require(not validate_launch_request(request) and not validate_launch_profile(profile), code)
        _require(target.name == request['launch_id']
            and request['launch_profile_id'] == profile['profile_id']
            and request['launch_profile_digest'] == profile['profile_digest']
            and request.get('source_commit') == profile.get('source_commit')
            and request['source_bundle'] == profile['source_bundle']
            and request['evaluation_run_spec'] == profile['evaluation_run_spec']
            and request['claim_ceiling'] == profile['claim_ceiling'], code)
        require_scene_execution_authority(profile, source_commit=profile['source_commit'],
            maximum_spend_usd=profile['allocator']['max_spend_usd'], now=now)
        # Only the explicitly configured store and the native validated binding
        # select these records. No newest record, directory scan or owner guess.
        root = _canonical(os.getenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', ''))
        binding = profile['scene_attempt_binding']
        intent, owner_ref = load_document(root / binding['intent_id'] / 'intent.json', maximum=65536)
        attempt, birth_ref = load_document(
            root / binding['intent_id'] / 'attempts' / (binding['attempt_id'] + '.json'), maximum=65536)
        _require(intent['intent_id'] == binding['intent_id']
            and intent['intent_digest'] == binding['intent_digest']
            and all(attempt.get(key) == binding[key] for key in (
                'intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest')), code)
        return access.birth_scene_member(target, owner_intent_id=intent['intent_id'],
            owner_raw_ref=owner_ref, birth_request_raw_ref=birth_ref, now=now)


def create_launch_output(target, *, request, profile):
    if access._policy() is None:
        Path(target).mkdir(parents=True, exist_ok=True)
        return
    with access.scene_access(target):
        if existing_scene_directory(target):
            _existing_launch_identity(target, request=request, profile=profile)
            return  # Retained metadata/teardown is not another paid launch.
        if enroll_launch_output(target, request=request, profile=profile) is None:
            Path(target).mkdir(parents=True, exist_ok=True)


def _existing_launch_identity(target, *, request, profile):
    generation = current_directory_generation(target)
    if generation is None:
        return
    from .task_evaluation_launch_dispatcher import validate_launch_profile_structure, validate_launch_request
    from .task_evaluation_scene_attempt_binding import scene_execution_binding_blockers
    code = 'scene_retirement_launch_storage_authority_unproven'
    _require(not validate_launch_request(request)
        and not validate_launch_profile_structure(profile)
        and not scene_execution_binding_blockers(profile)
        and type(profile.get('scene_attempt_binding')) is dict, code)
    binding = profile['scene_attempt_binding']
    intent = selected_document(generation['owner_raw_ref'], maximum=65536)
    attempt = selected_document(generation['birth_request_raw_ref'], maximum=65536)
    root = _canonical(os.getenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', ''))
    _require(generation.get('owner_intent_id') == binding['intent_id'] == intent['intent_id']
        and generation['owner_raw_ref']['path'] == str(root / binding['intent_id'] / 'intent.json')
        and generation['birth_request_raw_ref']['path'] == str(root / binding['intent_id'] / 'attempts' / (binding['attempt_id'] + '.json'))
        and intent['intent_digest'] == binding['intent_digest']
        and all(attempt.get(key) == binding[key] for key in (
            'intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest'))
        and Path(target).name == request['launch_id']
        and request['launch_profile_id'] == profile['profile_id']
        and request['launch_profile_digest'] == profile['profile_digest']
        and request.get('source_commit') == profile.get('source_commit')
        and request['source_bundle'] == profile['source_bundle']
        and request['evaluation_run_spec'] == profile['evaluation_run_spec'], code)
