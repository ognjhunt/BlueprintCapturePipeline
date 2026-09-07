"""Closed historical administrative contracts; no live policy or code loading.

The caller must still verify the immutable parent/child graph and current reuse
rights. These contracts interpret bytes only and cannot authorize execution.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest

SCHEMA_PATH = (Path(__file__).resolve().parents[2] / 'docs/schemas/'
               'task_evaluation_retained_scene_preparation.v1.schema.json')


@dataclass(frozen=True)
class ScenePreparationBudget:
    contract_id: str
    parent_ttl_seconds: int
    semantic_teacher_minimum: float
    visual_review_minimum: float
    content_agents_minimum: float
    external_maximum: float
    attempt_maximum: float
    source_revision: str


# Pinned source revisions establish these administrative contracts. No code from
# a historical checkout is imported or executed. New variants require review.
LEGACY_SINGLE_PASS = ScenePreparationBudget(
    'scene_preparation_25200_single_pass.v1', 25200, 2.4, .32, .2, 3., 10.,
    '1df1785e48220633e90b507ef68b04929ef74103')
LEGACY_REPAIR = ScenePreparationBudget(
    'scene_preparation_25200_repair.v1', 25200, 4.8, .64, .2, 6., 12.,
    '6bae36660c460c3115ce17fe6352145395e50e8f')
RETAINED_REPAIR = ScenePreparationBudget(
    'scene_preparation_27000_repair.v1', 27000, 4.8, .64, .2, 6., 12.,
    'ac689e03ab6a7c6fb598f855d4f5bc37b4f87d43')


def retained_schema():
    """Read the frozen scene-only schema rather than today's intake schema."""
    from .task_evaluation_launch_preparation_contract import TaskEvaluationLaunchPreparationContractError
    try:
        schema = json.loads(SCHEMA_PATH.read_text())
    except (OSError, ValueError) as exc:
        raise TaskEvaluationLaunchPreparationContractError(
            'launch_preparation_retained_schema_unavailable') from exc
    if not isinstance(schema, dict):
        raise TaskEvaluationLaunchPreparationContractError(
            'launch_preparation_retained_schema_invalid')
    return schema


def retained_budget(value):
    """Select only enumerated administrative shapes after structural validation."""
    spend = value['spend']
    ttl = spend['hard_ttl_seconds']
    if ttl == 27000:
        return RETAINED_REPAIR
    if ttl == 25200:
        if spend['external_service_caps']['openai']['maximum_cost_usd'] <= 3:
            return LEGACY_SINGLE_PASS
        return LEGACY_REPAIR
    # Match the existing typed refusal; an arbitrary historical TTL is not safe.
    from .task_evaluation_launch_preparation_contract import TaskEvaluationLaunchPreparationContractError
    raise TaskEvaluationLaunchPreparationContractError(
        'launch_preparation_scene_configuration_parent_runtime_budget_invalid')


def retained_contract_identity(value):
    """Bind the recognized contract in new adoption records without changing parents."""
    budget = retained_budget(value)
    identity = {'schema_version': 'task_evaluation_retained_preparation_contract_identity.v1',
        'request_schema_version': value['schema_version'],
        'request_source_commit': value['expected_production_commit'],
        'contract_id': budget.contract_id,
        'schema_sha256': 'sha256:' + hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        'policy_source_revision': budget.source_revision,
        'policy_source_path': 'src/blueprint_pipeline/task_evaluation_scene_configuration_runtime_budget.py'}
    identity['contract_digest'] = canonical_digest(identity, digest_field='contract_digest')
    return identity
