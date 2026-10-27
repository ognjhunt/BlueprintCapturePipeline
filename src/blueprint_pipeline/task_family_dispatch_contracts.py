"""Pure family dispatch identities; these records grant no execution authority."""
from __future__ import annotations

import re
from collections.abc import Mapping

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def validate_runtime_selection(value):
    if (not isinstance(value, Mapping) or set(value) != {"runtime_id", "runtime_catalog_digest"}
            or not isinstance(value.get("runtime_id"), str) or not _ID.fullmatch(value["runtime_id"])
            or not isinstance(value.get("runtime_catalog_digest"), str)
            or not _DIGEST.fullmatch(value["runtime_catalog_digest"])):
        raise ValueError("website_task_family_qualified_runtime_selection_required")
    return dict(value)


def validate_policy_selection(value):
    if (not isinstance(value, Mapping) or set(value) != {"candidate_id", "catalog_digest", "configuration_digest"}
            or not isinstance(value.get("candidate_id"), str) or not _ID.fullmatch(value["candidate_id"])
            or any(not isinstance(value.get(key), str) or not _DIGEST.fullmatch(value[key])
                   for key in ("catalog_digest", "configuration_digest"))):
        raise ValueError("website_task_family_policy_selection_invalid")
    return dict(value)


def validate_family_intake_task(task):
    # Python-owned scientific contracts stay in exact registered source bytes;
    # the WebApp forwards their immutable identity without rewriting numbers.
    allowed = {"task_id", "strategy", "task_success_contract_digest", "native_runtime_contract_digest",
               "task_family_runtime_selection", "policy_selection"}
    if set(task) - allowed or not allowed.difference({"policy_selection"}) <= set(task):
        raise ValueError("website_task_family_intake_fields_invalid")
    for key in ("task_success_contract_digest", "native_runtime_contract_digest"):
        if not isinstance(task[key], str) or not _DIGEST.fullmatch(task[key]):
            raise ValueError("website_task_family_intake_binding_invalid")
    validate_runtime_selection(task["task_family_runtime_selection"])
    if "policy_selection" in task:
        validate_policy_selection(task["policy_selection"])


def validate_family_preparation_result_reference(value, *, request, status, phase):
    """Validate CPU preparation completion without asserting native application."""
    from pathlib import PurePosixPath
    from .task_family_scoring import canonical_task_family
    expected = {'schema_version', 'path', 'sha256', 'size_bytes', 'digest', 'task_family',
                'task_success_contract_digest', 'native_runtime_contract_digest',
                'task_family_runtime_selection', 'native_application_claimed',
                'policy_episode_claimed', 'pre_policy_site_qualification_required'}
    task = request.get('task') or {}
    family = canonical_task_family(task.get('strategy'))
    if (not isinstance(value, Mapping) or set(value) != expected
            or status != 'completed' or phase != 'family_native_prepared' or family is None
            or (request.get('execution') or {}).get('purpose') != 'scene_preparation'
            or value.get('schema_version') != 'website_task_family_preparation_result.v1'
            or value.get('task_family') != family
            or any(value.get(key) != task.get(key) for key in (
                'task_success_contract_digest', 'native_runtime_contract_digest', 'task_family_runtime_selection'))
            or value.get('native_application_claimed') is not False
            or value.get('policy_episode_claimed') is not False
            or value.get('pre_policy_site_qualification_required') is not True
            or type(value.get('size_bytes')) is not int or value['size_bytes'] <= 0
            or any(not isinstance(value.get(key), str) or not _DIGEST.fullmatch(value[key])
                   for key in ('sha256', 'digest', 'task_success_contract_digest', 'native_runtime_contract_digest'))):
        raise ValueError('website_task_family_preparation_result_reference_invalid')
    path = PurePosixPath(str(value.get('path') or ''))
    if (not path.is_absolute() or '..' in path.parts
            or path.name != 'family_native_preparation_result.v1.json'):
        raise ValueError('website_task_family_preparation_result_reference_invalid')
    validate_runtime_selection(value['task_family_runtime_selection'])
    return dict(value)
