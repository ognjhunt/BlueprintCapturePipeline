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
# Finite producer OUTER wrappers only. Embedded request/source/appearance
# dictionaries retain their existing source-defined open contracts.
PREPARATION_BASE_FIELDS = {'schema_version', 'status', 'preparation_id', 'run_id', 'team_namespace', 'source_commit',
    'reference_count', 'unique_object_count', 'content_addressed_reuse_count', 'references',
    'full_byte_service_account_readback_passed', 'service_account', 'service_account_uid', 'provider_mutation_performed',
    'catalog_mutation_performed', 'paid_execution_requested', 'observed_at_iso', 'result_digest'}
PREPARATION_HANDOFF_FIELDS = {'run_mode', 'configured_scene_revision_digest', 'configured_scene_bundle_digest',
    'episode_compilation_id', 'episode_compilation_queue_envelope_digest', 'episode_compilation_queue_receipt_digest',
    'customer_supplied_prebuilt_episode_packet', 'construction_packet_materialized', 'automatic_progression_required'}
NATIVE_PREPARATION_REQUEST_FIELDS = {'construction', 'controller', 'execution_adapter', 'expected_production_commit',
    'policy_canary_activation', 'policy_run_configuration', 'policy_run_selection', 'policy_run_setup', 'preparation_id',
    'publication', 'replacement_authoring_agent_runtime', 'replacement_authoring_backend', 'replacement_authoring_model',
    'replacement_authoring_model_provider', 'robot', 'run_id', 'run_mode', 'runtime', 'scene', 'scene_intent_digest',
    'schema_version', 'sensors', 'spend', 'task', 'team_namespace'}
NATIVE_ACTIVATION_REQUEST_FIELDS = {'activation_id', 'authorization', 'capture_session_id', 'episode_interpretation_authority',
    'episode_interpretation_source_rights_admission', 'expected_production_commit', 'intake_id', 'lane', 'lineage',
    'preparation', 'release_window', 'requested_mutations', 'run_kind', 'schema_version', 'team_namespace'}
FIELD_SETS = {
    # Shared live-profile skeleton/control surface plus native/scene lane
    # owner fields and dispatcher-defined optional wrappers. No nested policy
    # or runtime validation is introduced by this retained metadata bridge.
    'launch_profiles': {'schema_version', 'profile_id', 'profile_digest', 'program_id', 'source_commit', 'claim_ceiling',
        'allocator', 'execution_admission', 'evaluation_run_spec', 'source_bundle', 'immutable_inputs', 'runtime_environment',
        'reconciliation', 'required_controls', 'terminal_contract', 'webapp_sync', 'standing_launch_authorization',
        'manifest_publication', 'scene_intent_digest', 'scene_attempt_id', 'scene_attempt_binding', 'native_policy_binding',
        'task_evaluation_run', 'policy_run_setup', 'internal_policy_canary_setup', 'internal_policy_canary_execution_plan',
        'prelaunch_skill_plan', 'same_goal_spend_lineage'},
    'native_preparation_envelopes': {'schema_version', 'request_digest', 'request', 'submitted_by', 'submitted_at_iso',
        'provider_mutation_performed_inside_intake', 'catalog_mutation_performed_inside_intake', 'envelope_digest'},
    'native_activation_envelopes': {'schema_version', 'request_digest', 'request', 'submitted_by', 'submitted_at_iso',
        'provider_mutation_performed_inside_intake', 'catalog_mutation_performed_inside_intake',
        'standing_authorization_published_inside_intake', 'paid_execution_requested', 'envelope_digest'},
    'native_activation_results': {'schema_version', 'status', 'activation_id', 'preparation_id', 'team_namespace', 'lane',
        'source_commit', 'preparation_result_digest', 'release_window_digest', 'profile_id', 'profile_digest',
        'profile_publication_receipt_digest', 'standing_authorization_digest', 'full_byte_activation_reference_readback_passed',
        'profile_publication_performed', 'catalog_mutation_performed', 'standing_authorization_published',
        'provider_mutation_performed', 'paid_execution_requested', 'blockers', 'observed_at_iso', 'result_digest'},
    'configured_revisions': {'appearance', 'configuration_run_id', 'configured_scene_bundle', 'evaluation_admission',
        'geometry', 'presentation', 'publication', 'registration', 'replacement', 'revision_digest', 'robot_team_interface',
        'scene_identity', 'schema_version', 'source', 'source_commit', 'status', 'task_template', 'team_namespace'},
    'compilation_intake_receipts': {'schema_version', 'status', 'compilation_id', 'run_id', 'configured_scene_revision_digest',
        'envelope_digest', 'queue_path', 'created', 'automatic_progression_required', 'provider_mutation_performed',
        'paid_execution_requested', 'receipt_digest'},
    'compilation_envelopes': {'schema_version', 'compilation_id', 'preparation_id', 'run_id', 'team_namespace',
        'expected_production_commit', 'configured_scene_revision_digest', 'configured_scene_bundle', 'materialized_references',
        'request', 'preparation_result_digest', 'automatic_progression_required', 'robot_specific_episode_packet_compiled_in_production',
        'customer_supplied_prebuilt_episode_packet', 'production_compiler_owns_episode_packet', 'provider_mutation_performed',
        'paid_execution_requested', 'envelope_digest'},
    'compilation_results': {'schema_version', 'status', 'compilation_id', 'run_id', 'team_namespace', 'source_commit',
        'configured_scene_revision_digest', 'compiled_episode_packet_digest', 'compiled_episode_packet_size_bytes',
        'compiled_episode_packet_path', 'adapter_result_path', 'adapter_result_digest', 'compiler_output_digest',
        'destination_native_probe_request_path', 'destination_native_probe_request_digest', 'destination_native_probe_request_document_digest',
        'customer_supplied_prebuilt_episode_packet', 'compiled_by_production', 'provider_mutation_performed', 'paid_execution_requested',
        'automatic_progression_required', 'blockers', 'result_digest'},
    'native_owner_records': {'scene_intent_digest', 'scene_attempt_id', 'scene_attempt_binding', 'schema_version', 'phase',
        'team_namespace', 'scene_id', 'task_id', 'runtime_source_bundle_digest', 'owner_attempt_digest'},
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
        rows = super().known(role, schema, seal_field)
        for row in rows:
            if role == 'native_preparation_results' and row[0].get('status') in {
                    'inputs_materialized_awaiting_construction_adapter', 'queued_for_production_episode_compilation'}:
                allowed = PREPARATION_BASE_FIELDS | {'policy_run_plan'}
                if row[0]['status'] == 'queued_for_production_episode_compilation':
                    allowed |= PREPARATION_HANDOFF_FIELDS
                self.fields(row, allowed)
            elif role in FIELD_SETS and (role not in {'native_activation_results', 'compilation_results'}
                    or row[0].get('status') in {'profile_authority_materialized_no_execution', 'compiled_for_production_launch'}):
                self.fields(row, FIELD_SETS[role])
        return rows

    def fields(self, row, allowed):
        if not hasattr(self, 'unsupported_records'):
            self.unsupported_records = set()
        if set(row[0]) - allowed:
            key = tuple(row[1][k] for k in ('role', 'path', 'sha256', 'size_bytes'))
            if key not in self.unsupported_records:
                self.unsupported_records.add(key)
                self.missing(row[1]['role'], 'unsupported_retained_field_set', [row[1]])
        return self.supported(row)

    def supported(self, row):
        key = tuple(row[1][k] for k in ('role', 'path', 'sha256', 'size_bytes'))
        known = SCHEMAS.get(row[1]['role'])
        return (known is None or 'json_pointer' in row[1] or row[0].get('schema_version') == known[0]) and key not in getattr(self, 'unsupported_records', ())

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
