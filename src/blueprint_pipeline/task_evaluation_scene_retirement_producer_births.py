"""Connect real downstream creation to one exact authenticated preparation.

The caller first validates its native request and materialized references. An
unregistered preparation remains compatible, without granting retirement proof.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _require
from .task_evaluation_scene_retirement_authority import load_document, selected_document


def enroll_preparation_child(target, *, preparation_root, request, verified_paths=(), now=None):
    policy = access._policy()
    if policy is None:
        return None
    parent = _canonical(str(preparation_root))
    _require(parent.name == request.get('preparation_id'),
             'scene_retirement_storage_authority_unproven')
    key = hashlib.sha256(str(parent).encode()).hexdigest() + '.json'
    with access.scene_access(parent):
        try:
            generation, _ = load_document(Path(policy['generation_store']) / key, maximum=65536)
        except FileNotFoundError:
            return None
        _require(generation.get('canonical_path') == str(parent)
                 and generation.get('schema_version') == 'scene_member_generation.v1',
                 'scene_retirement_storage_authority_unproven')
        reference = generation.get('source_storage_authority_raw_ref')
        if reference is None:
            return None
        authority = selected_document(reference, maximum=65536)
        from .task_evaluation_scene_retirement_cache import _validate
        intent, _ = _validate(authority, request, now=time.time() if now is None else now)
        _require(generation.get('owner_intent_id') == intent['intent_id']
                 and generation.get('owner_raw_ref') == authority['intent_raw_ref']
                 and generation.get('birth_request_raw_ref') == authority['attempt_raw_ref'],
                 'scene_retirement_storage_authority_unproven')
        # No source-parent/name search: these are the exact native verifier's
        # positively selected projection paths, never global CAS ownership.
        _require(len(verified_paths) <= 10000
                 and all(_canonical(str(path)).is_relative_to(parent) for path in verified_paths),
                 'scene_retirement_storage_authority_unproven')
        return access.birth_scene_member(target, owner_intent_id=intent['intent_id'],
            owner_raw_ref=authority['intent_raw_ref'],
            birth_request_raw_ref=authority['attempt_raw_ref'], now=now)
