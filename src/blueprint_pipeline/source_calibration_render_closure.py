"""Retain and revalidate private source-render disclosure and terminal spend evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json
from .source_calibration_render_return import record, require, verify_source_calibration_return
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read

CLOSURE_ROLES = frozenset({'source_disclosure', 'private_store', 'provider_execution',
                          'official_billing', 'teardown', 'provider_zero'})


def _validate_closure(prepared: Mapping, value: Mapping) -> None:
    from .sam31_contribution_disclosure import PROOF_SCHEMA, AUTHORITY_SCHEMA
    from .vast_independent_watchdog_control import _inventory_is_confirmed_zero
    from .vast_official_billing_extractor import extract_vast_official_instance_charge
    closure = value.get('execution_closure', {})
    require(set(closure) == CLOSURE_ROLES, 'execution_closure_missing')
    paths = {name: checked_file(row['path'], row) for name, row in closure.items()}
    proof = read(paths['source_disclosure'], digest_field='proof_digest')
    authority_ref = proof.get('disclosure_authority', {})
    authority = read(checked_file(authority_ref.get('path', ''), authority_ref), digest_field='authorization_digest')
    conversion_ref = proof.get('conversion_receipt', {})
    conversion = read(checked_file(conversion_ref.get('path', ''), conversion_ref), digest_field='receipt_digest')
    binding = proof.get('source_binding', {})
    layer = prepared['layers']['images']
    require(proof.get('schema_version') == PROOF_SCHEMA
            and proof.get('status') == 'explicit_full_source_disclosure_verified'
            and proof.get('purpose') == 'exact_source_calibration_gpu_render'
            and conversion.get('receipt_digest') == proof.get('conversion_receipt_digest')
            and conversion.get('rights') == proof.get('conversion_rights')
            and authority.get('schema_version') == AUTHORITY_SCHEMA
            and authority.get('status') == 'authorized'
            and authority.get('provider_id') == 'vast'
            and authority.get('purpose') == proof.get('purpose')
            and authority.get('authority_kind') == 'explicit_human_full_source_provider_processing'
            and authority.get('agent_accepted_terms') is False
            and authority.get('publisher_rights_permit_private_full_source_processing') is True
            and authority.get('provider_retention_terms_accepted') is True
            and authority.get('provider_training_terms_accepted') is True
            and authority.get('publisher_rights_basis') == proof.get('publisher_rights_basis')
            and authority.get('source_commit') == prepared['repository']['commit']
            and authority.get('authorization_digest') == proof.get('disclosure_authority_digest')
            and authority.get('source_binding') == binding
            and authority.get('full_source_scene_content_upload_authorized') is True
            and authority.get('private_provider_processing_authorized') is True
            and authority.get('public_redistribution_authorized') is False
            and authority.get('provider_training_authorized') is False
            and binding.get('standard_splat_sha256') == layer['sha256']
            and binding.get('standard_splat_size_bytes') == layer['size_bytes']
            and binding.get('retained_gaussian_count') == layer['retained_gaussian_count'],
            'closed_disclosure_invalid')
    for ref in proof.get('publisher_rights_basis', {}).values():
        if isinstance(ref, Mapping) and 'path' in ref:
            checked_file(ref['path'], dict(ref))
    private = read(paths['private_store'], digest_field='readback_digest')
    buckets = private.get('bucket_response', {}).get('buckets', [])
    require(private.get('schema_version') == 'source_calibration_private_store_readback.v1'
            and private.get('status') == 'verified_private'
            and private.get('provider') == 'backblaze_b2'
            and private.get('authenticated_native_readback') is True
            and private.get('credential_values_recorded') is False
            and private.get('s3_endpoint') == f"https://s3.{private.get('region')}.backblazeb2.com"
            and private.get('bucket_type') == 'allPrivate'
            and any(row.get('bucketName') == private.get('bucket')
                    and row.get('bucketId') == private.get('bucket_id')
                    and row.get('bucketType') == 'allPrivate' for row in buckets), 'closed_private_store_invalid')
    execution = read(paths['provider_execution'], digest_field='receipt_digest')
    ids = execution.get('vast_instance_ids', [])
    require(execution.get('status') == 'completed' and execution.get('render_scope') == 'source_calibration'
            and len(ids) == 1 and type(ids[0]) is int and ids[0] > 0
            and execution.get('continuing_spend_from_this_run') is False
            and execution.get('all_staged_objects_absent') is True
            and execution.get('private_source_store_readback') == private
            and record(Path(execution['execution_result_path'])) == value['provider_result'],
            'closed_provider_execution_invalid')
    teardown = read(paths['teardown'])
    require(str(paths['teardown']) == execution.get('teardown_manifest_path')
            and teardown.get('continuing_spend_from_this_run') is False
            and teardown.get('runner_gpu_teardown_completed') is True
            and teardown.get('vast_instance_ids') == ids, 'closed_teardown_invalid')
    zero = read(paths['provider_zero'])
    recorded = zero.get('recorded_vast_instance_teardown', {})
    attempts = recorded.get('inspect_attempts', [])
    require(zero.get('status') == 'provider_terminal' and zero.get('provider') == 'vast'
            and zero.get('provider_absence_confirmed') is True
            and str(recorded.get('instance_id')) == str(ids[0])
            and recorded.get('status') == 'absent' and recorded.get('provider_absence_confirmed') is True
            and len(attempts) >= 2
            and all(row.get('api_confirmed') is True and row.get('provider_absence_confirmed') is True
                    and row.get('status') == 'absent' and str(row.get('instance_id')) == str(ids[0]) for row in attempts)
            and all(_inventory_is_confirmed_zero(zero.get(name, {}), name_prefix='')
                    for name in ('initial_global_inventory', 'final_global_inventory')),
            'closed_provider_zero_invalid')
    charge = read(paths['official_billing'], digest_field='charge_digest')
    billing_ref = charge.get('provider_billing_source_receipt', {})
    actual = extract_vast_official_instance_charge(
        provider_billing_source_receipt_path=checked_file(billing_ref.get('path', ''), billing_ref),
        instance_id=ids[0], launch_label=charge.get('launch_label', ''))
    require(actual == charge and 0 <= charge['official_charge_usd'] <= 1.0, 'closed_official_billing_invalid')
    require(value.get('full_source_scene_content_transferred') is True
            and value.get('original_downloaded_file_uploaded') is False
            and value.get('private_only') is True, 'closed_transfer_scope_invalid')


def require_source_calibration_closure(prepared: Mapping, returned_group_path: str | Path) -> dict:
    verify_source_calibration_return(prepared, returned_group_path)
    value = read(returned_group_path, digest_field='return_digest')
    _validate_closure(prepared, value)
    return value


def materialize_source_calibration_closed_return(*, prepared_inputs: Mapping,
        returned_group_path: str | Path, execution_closure: Mapping, output_path: str | Path) -> dict:
    verify_source_calibration_return(prepared_inputs, returned_group_path)
    value = read(returned_group_path, digest_field='return_digest')
    value.update(execution_closure=dict(execution_closure), full_source_scene_content_transferred=True,
                 original_downloaded_file_uploaded=False, private_only=True)
    value['return_digest'] = canonical_digest(value, digest_field='return_digest')
    _validate_closure(prepared_inputs, value)
    with Path(output_path).open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(value)+'\n')
    return value
