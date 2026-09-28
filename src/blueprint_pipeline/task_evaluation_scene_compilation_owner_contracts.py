"""Pure retained compilation metadata primitives; ADP-009D/day28."""
from __future__ import annotations

from . import task_evaluation_scene_source_family_contracts as retained
from .task_evaluation_scene_lineage_budget import RetainedEmissionBudgetError

path, child, under = retained.path, retained.child, retained.under
matches, canonical_digest = retained.matches, retained.canonical_digest
ID, COMMIT = retained.ID, retained.COMMIT
OWNER_ID = retained.c.ID
ROLES = {'native_preparation_envelopes', 'native_preparation_results', 'native_activation_envelopes',
    'native_activation_results', 'configured_revisions', 'compilation_intake_receipts', 'compiler_outputs',
    'compilation_adapter_results', 'native_owner_records'}
SCHEMAS = {
    'native_preparation_envelopes': ('task_evaluation_launch_preparation_envelope.v1', 'envelope_digest'),
    'native_preparation_results': ('task_evaluation_launch_preparation_result.v1', 'result_digest'),
    'native_activation_envelopes': ('task_evaluation_launch_activation_envelope.v1', 'envelope_digest'),
    'native_activation_results': ('task_evaluation_launch_activation_result.v1', 'result_digest'),
    'configured_revisions': ('task_evaluation_configured_scene_revision.v1', 'revision_digest'),
    'compilation_intake_receipts': ('task_evaluation_episode_compilation_intake_receipt.v1', 'receipt_digest'),
    'compiler_outputs': ('task_evaluation_episode_compiler_output.v1', 'compiler_output_digest'),
    'compilation_adapter_results': ('task_evaluation_native_arena_adapter_result.v1', 'result_digest'),
    'native_owner_records': ('task_evaluation_scene_owner_attempt.v1', 'owner_attempt_digest'),
}
seal, observation = retained.c.seal, retained.observation


class SceneCompilationOwnerInventoryError(ValueError):
    """Fixed refusal without private supplied text."""


def require(condition, code):
    if not condition:
        raise SceneCompilationOwnerInventoryError('scene_compilation_owner_' + code)


def text(value, maximum=192):
    require(isinstance(value, str) and 0 < len(value) <= maximum and len(value.encode('utf-8')) <= maximum, 'text_invalid')
    return value


class Context(retained.Context):
    def known(self, role, schema=None, seal_field=None):
        if schema is None:
            schema, seal_field = SCHEMAS[role]
        return super().known(role, schema, seal_field)

    def selected(self, reference, source, roles=None, *, positive=True):
        row = super().selected(reference, source, roles, positive=positive)
        if row and row[1]['role'] in SCHEMAS and row[0].get('schema_version') != SCHEMAS[row[1]['role']][0]:
            self.missing(row[1]['role'], 'unsupported_retained_schema', [row[1]])
            return None
        return row

    def metadata(self, row):
        require(any(under(row[1]['path'], root) for root in self.metadata_roots), 'metadata_path_invalid')

    def canonical(self, role, digest, source, expected=None):
        self.missing(role, 'canonical_selector_bytes_unavailable', [source], expected, {'canonical_digest': digest})


def translate(exc):
    if isinstance(exc, RetainedEmissionBudgetError):
        return SceneCompilationOwnerInventoryError('scene_compilation_owner_output_limit')
    if isinstance(exc, (retained.SceneSourceFamilyInventoryError, retained.c.SceneDownstreamInventoryError)):
        code = str(exc).removeprefix('scene_source_family_').removeprefix('scene_downstream_')
        return SceneCompilationOwnerInventoryError('scene_compilation_owner_' + code)
    return SceneCompilationOwnerInventoryError('scene_compilation_owner_input_invalid')
