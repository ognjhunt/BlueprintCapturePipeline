"""Native listener bodies behind real scene lifetimes.

All former module globals are resolved through the live listener namespace,
preserving existing injected providers, constants and monkeypatched callbacks.
No additional authority or provider call is introduced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

def _stage_handoff_capture_body(_listener, /, handoff, *, storage_root, storage_client):
    if handoff.source_finalize is not None:
        from .capture_delivery_staging import stage_selected_capture
        return stage_selected_capture(_listener, handoff, storage_root=storage_root,
                                      storage_client=storage_client)
    client = storage_client or _listener.storage.Client()
    resolved_storage_root = storage_root.resolve()
    bucket_root = _listener.contained_path(resolved_storage_root, handoff.bucket, field='Pub/Sub staging bucket path')
    capture_root = _listener.contained_path(bucket_root, 'scenes', handoff.scene_id, 'captures', handoff.capture_id, field='Pub/Sub capture staging path')
    capture_root.mkdir(parents=True, exist_ok=True)
    expected_prefix = f'{handoff.capture_prefix}/'
    blobs = list(client.list_blobs(handoff.bucket, prefix=expected_prefix))
    if not blobs:
        raise _listener.PipelineError(f'No objects found for handoff prefix: {handoff.capture_prefix}/')
    prefix_depth = len(_listener.PurePosixPath(handoff.capture_prefix).parts)
    previously_staged = _listener._previous_staging_rows(capture_root, handoff=handoff)
    manifest_rows: list[dict[str, Any]] = []
    downloads: list[tuple[Any, Path]] = []
    for blob in blobs:
        blob_name = str(blob.name or '')
        if not blob_name.startswith(expected_prefix):
            raise _listener.PipelineError('Pub/Sub blob escaped the declared capture prefix')
        blob_path = _listener.PurePosixPath(blob_name)
        if blob_path.is_absolute() or any((part in {'', '.', '..'} for part in blob_path.parts)) or '\\' in blob_name or ('\x00' in blob_name):
            raise _listener.PipelineError('Pub/Sub blob name contains an unsafe path')
        if blob_name.endswith('/'):
            continue
        try:
            destination = _listener.contained_path(bucket_root, *blob_path.parts, field='Pub/Sub blob destination')
            _listener.prove_path_contained(capture_root, destination, field='Pub/Sub blob capture destination')
        except _listener.SecurityValidationError as exc:
            raise _listener.PipelineError(str(exc)) from exc
        row = _listener._staging_manifest_row(blob, name=blob_name, relative_path=_listener.PurePosixPath(*blob_path.parts[prefix_depth:]).as_posix())
        manifest_rows.append(row)
        previous = previously_staged.get(blob_name)
        if previous is not None and _listener._staged_copy_is_current(previous, row, destination):
            continue
        downloads.append((blob, destination))
    _listener.download_with_reservation(downloads=downloads, manifest_rows=manifest_rows, storage_root=storage_root, capture_root=capture_root)
    _listener.write_json(capture_root / _listener.STAGING_MANIFEST_FILENAME, {'schema_version': _listener.STAGING_MANIFEST_SCHEMA_VERSION, 'bucket': handoff.bucket, 'prefix': expected_prefix, 'staged_at': _listener.utc_now_iso(), 'objects': manifest_rows})
    if not (capture_root / 'raw' / 'capture_upload_complete.json').is_file():
        raise _listener.PipelineError(f'Staged handoff capture is missing raw/capture_upload_complete.json; capture_root={capture_root}')
    _listener._preserve_local_website_derivatives(capture_root, {str(blob.name) for blob in blobs}, handoff.capture_prefix)
    if not (capture_root / 'pipeline_handoff.json').is_file():
        _listener._synthesize_pipeline_handoff(handoff, capture_root=capture_root)
    return capture_root


def __synthesize_pipeline_handoff_body(_listener, /, handoff, *, capture_root):
    """Materialize pipeline_handoff.json from raw sidecars when the iOS bundle omits it.

    We never invent provenance: values come only from raw/manifest.json and
    raw/capture_context.json, both of which the iOS app already writes.
    """
    raw_root = capture_root / 'raw'
    manifest = _listener._read_optional_json_object(raw_root / 'manifest.json')
    context = _listener._read_optional_json_object(raw_root / 'capture_context.json')
    site_submission_id = _listener._first_non_empty(manifest, context, keys=('site_submission_id', 'siteSubmissionId'))
    buyer_request_id = _listener._first_non_empty(manifest, context, keys=('buyer_request_id', 'buyerRequestId'))
    capture_job_id = _listener._first_non_empty(manifest, context, keys=('capture_job_id', 'captureJobId'))
    request_id = buyer_request_id or capture_job_id
    requested_outputs: list[str] = []
    for source in (manifest, context):
        for key in ('requested_outputs', 'requestedOutputs', 'requested_lanes', 'requestedLanes'):
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    text = str(item).strip()
                    if text and text not in requested_outputs:
                        requested_outputs.append(text)
    payload: dict[str, Any] = {'schema_version': 'pipeline_handoff.v1', 'synthesized': True, 'synthesized_from': ['raw/manifest.json', 'raw/capture_context.json'], 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'bucket': handoff.bucket, 'raw_prefix_uri': handoff.raw_prefix_uri, 'site_submission_id': site_submission_id, 'buyer_request_id': buyer_request_id, 'capture_job_id': capture_job_id, 'owner_system': {'owner_system': 'blueprint_capture', 'request_id': request_id, 'site_submission_id': site_submission_id, 'buyer_request_id': buyer_request_id, 'capture_job_id': capture_job_id}}
    if requested_outputs:
        payload['requested_outputs'] = requested_outputs
    destination = capture_root / 'pipeline_handoff.json'
    _listener.write_json(destination, payload)
    _listener.logger.info('pubsub_handoff.synthesized_pipeline_handoff', extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_job_id': capture_job_id})
    return destination


def _process_handoff_payload_body(_listener, /, payload, *, storage_root, provider, run_e2e, storage_client, run_evaluation_prep, run_e2e_enabled, stage_control_plane, control_plane_manifest_path, control_plane_work_dir, control_plane_staged_inputs_path, overwrite_control_plane_input, lease_owner, lease_seconds, payload_digest, expected_assessment_resume=None):
    handoff = _listener.parse_handoff_payload(payload)
    digest = payload_digest or _listener.payload_sha256(payload)
    capture_root = _listener._handoff_capture_root(handoff, storage_root=storage_root)
    selected_staged = None
    producer_delivery_key = None
    prior_retired = None
    if handoff.source_finalize is not None:
        from .capture_original_owner_observer import load_original_owner_observation

        try:
            observation = load_original_owner_observation(
                bucket=handoff.bucket, scene_id=handoff.scene_id,
                capture_id=handoff.capture_id,
                marker_generation=handoff.source_finalize['generation'],
            )
        except Exception:
            _listener.logger.warning('pubsub_handoff.capture_owner_observation_unavailable',
                                     extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id})
            return {'schema_version': 'v1', 'status': 'capture_owner_observation_unavailable_retryable',
                    'queue_disposition': 'retryable', 'bucket': handoff.bucket,
                    'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                    'capture_root': str(capture_root),
                    'blockers': ['capture_owner_observation_unavailable']}
        _listener.logger.info('pubsub_handoff.capture_owner_observed',
                              extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                                     'observation_digest': observation['observation_digest']})
        if handoff.source_membership_selector is not None:
            producer_delivery_key = observation['producer_delivery']['delivery_key']
            try:
                from .website_scene_workspace_retention import retired_capture_status
                retired = retired_capture_status(
                    storage_root=storage_root, bucket=handoff.bucket,
                    scene_id=handoff.scene_id, capture_id=handoff.capture_id)
            except Exception:
                return {'schema_version': 'v1', 'status': 'retirement_lookup_failed_retryable',
                        'queue_disposition': 'retryable', 'bucket': handoff.bucket,
                        'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                        'capture_root': str(capture_root),
                        'blockers': ['retirement_lookup_failed']}
            if retired is not None:
                prior_retired = retired
                ended_keys = retired['producer_delivery_keys']
                if not ended_keys:
                    return {'schema_version': 'v1',
                            'status': 'retired_delivery_identity_unproven_retryable',
                            'queue_disposition': 'retryable', 'bucket': handoff.bucket,
                            'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                            'capture_root': str(capture_root),
                            'blockers': ['retired_delivery_identity_unproven']}
                if producer_delivery_key in ended_keys:
                    return {'schema_version': 'v1', 'status': 'skipped_retired_terminal',
                            'queue_disposition': retired['queue_disposition'],
                            'bucket': handoff.bucket, 'scene_id': handoff.scene_id,
                            'capture_id': handoff.capture_id, 'capture_root': str(capture_root),
                            'retirement_receipt': retired['receipt']}
            try:
                selected_staged = _listener.stage_handoff_capture(
                    handoff, storage_root=storage_root, storage_client=storage_client)
            except Exception:
                _listener.logger.warning('pubsub_handoff.capture_source_membership_unavailable',
                                         extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id})
                return {'schema_version': 'v1',
                        'status': 'capture_source_membership_unavailable_retryable',
                        'queue_disposition': 'retryable', 'bucket': handoff.bucket,
                        'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                        'capture_root': str(capture_root),
                        'blockers': ['capture_source_membership_unavailable']}
        else:
            return {'schema_version': 'v1', 'status': 'capture_original_birth_unavailable_retryable',
                    'queue_disposition': 'retryable', 'bucket': handoff.bucket,
                    'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                    'capture_root': str(capture_root),
                    'blockers': ['capture_original_birth_unavailable']}

    def retired_terminal() -> dict[str, Any] | None:
        nonlocal prior_retired
        try:
            from .website_scene_workspace_retention import retired_capture_status
            retired = retired_capture_status(storage_root=storage_root, bucket=handoff.bucket, scene_id=handoff.scene_id, capture_id=handoff.capture_id)
            prior_retired = retired
        except Exception:
            _listener.logger.exception('pubsub_handoff.retirement_lookup_failed')
            return {'schema_version': 'v1', 'status': 'retirement_lookup_failed_retryable', 'queue_disposition': 'retryable', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'blockers': ['retirement_lookup_failed'], 'alerts': ['retirement_lookup_failed']}
        if retired is None or not (retired.get('covers_every_payload') or digest in retired['payload_sha256s']):
            return None
        _listener.logger.info('pubsub_handoff.skipped_retired_terminal', extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id})
        return {'schema_version': 'v1', 'status': 'skipped_retired_terminal', 'queue_disposition': retired['queue_disposition'], 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'retirement_receipt': retired['receipt']}
    capture_present = capture_root.exists()
    if selected_staged is None and not capture_present and (skipped := retired_terminal()) is not None:
        return skipped
    owner = lease_owner or _listener._lease_owner()
    try:
        claim_status, ledger = _listener._claim_job_lease(
            capture_root, scene_id=handoff.scene_id, capture_id=handoff.capture_id,
            owner=owner, lease_seconds=lease_seconds, payload_sha256=digest,
            producer_delivery_key=producer_delivery_key,
            create_capture_root=not capture_present and selected_staged is None,
            retired_ended_payload_sha256s=(prior_retired['payload_sha256s']
                if producer_delivery_key is None and prior_retired is not None
                and prior_retired['status'] == _listener.TERMINAL_AUTHORITY_STATUS else ()),
            retired_ended_producer_delivery_keys=(prior_retired['producer_delivery_keys']
                if producer_delivery_key is not None and prior_retired is not None else ()),
            **({'expected_assessment_resume': expected_assessment_resume} if expected_assessment_resume is not None else {}))
    except _listener.HandoffCaptureRetired:
        return retired_terminal() or {'schema_version': 'v1', 'status': 'capture_retired_retryable', 'queue_disposition': 'retryable', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'blockers': ['handoff_capture_retired_while_claiming']}
    if claim_status == 'assessment_resume_changed':
        return {'schema_version': 'v1', 'status': 'assessment_resume_changed', 'queue_disposition': 'retryable',
                'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                'blockers': ['website_assessment_resume_changed']}
    if claim_status == 'terminal':
        _listener._repair_terminal_receipt(capture_root, handoff=handoff)
        _listener.logger.info('pubsub_handoff.skipped_terminal_authority_ended', extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'terminal_code': ledger.get('terminal_code')})
        return _listener._terminal_authority_result(handoff, capture_root=capture_root, ledger=ledger, status='skipped_terminal_authority_ended')
    if claim_status == 'completed':
        commit = _listener._output_commit(capture_root, scene_id=handoff.scene_id, capture_id=handoff.capture_id)
        if not commit:
            return {'schema_version': 'v1', 'status': 'completed_output_commit_missing_retryable', 'queue_disposition': 'retryable', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'job_ledger': ledger, 'blockers': ['completed_handoff_output_commit_missing_or_invalid']}
        _listener.logger.info('pubsub_handoff.skipped_already_processed', extra={'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id})
        return {'schema_version': 'v1', 'status': 'skipped_already_processed', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'queue_disposition': 'terminal_success', 'output_commit': commit, 'job_ledger': ledger}
    if claim_status == 'active':
        return {'schema_version': 'v1', 'status': 'lease_active_retryable', 'queue_disposition': 'retryable', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'job_ledger': ledger, 'blockers': ['handoff_job_active_lease']}
    if claim_status == 'corrupt':
        return {'schema_version': 'v1', 'status': 'job_ledger_corrupt_retryable', 'queue_disposition': 'retryable', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'job_ledger': ledger, 'blockers': ['handoff_job_ledger_corrupt']}
    if claim_status == 'source_conflict':
        return {'schema_version': 'v1', 'status': 'capture_delivery_conflict_retryable',
                'queue_disposition': 'retryable', 'bucket': handoff.bucket,
                'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id,
                'capture_root': str(capture_root), 'job_ledger': ledger,
                'blockers': ['capture_delivery_conflict']}
    attempt_count = int(ledger.get('attempt_count') or 0)
    previous_history = _listener._attempt_history(ledger)
    job_started_at = _listener._string(ledger.get('started_at')) or _listener.utc_now_iso()
    attempt_started_at = _listener._string(ledger.get('last_attempt_started_at')) or _listener.utc_now_iso()
    token = _listener._string(ledger.get('lease_token'))
    recovered_commit = _listener._output_commit(capture_root, scene_id=handoff.scene_id, capture_id=handoff.capture_id)
    if ledger.get('recovered_expired_lease') is True and recovered_commit:
        recovered_at = _listener.utc_now_iso()
        recovered_record = {'attempt_number': attempt_count, 'status': 'completed_from_output_commit', 'stage': 'output_commit_recovery', 'started_at': attempt_started_at, 'completed_at': recovered_at, 'result_sha256': recovered_commit.get('result_sha256')}
        completed_ledger = _listener._finish_job_lease(capture_root, owner=owner, token=token, update={'status': 'completed', 'updated_at': recovered_at, 'completed_at': recovered_at, 'output_commit_status': 'committed', 'output_commit_path': _listener.JOB_OUTPUT_COMMIT_FILENAME, 'attempt_history': [*previous_history, recovered_record]})
        return {'schema_version': 'v1', 'status': 'skipped_committed_output_recovered', 'queue_disposition': 'terminal_success', 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'output_commit': recovered_commit, 'job_ledger': completed_ledger}
    control_plane_staging: dict[str, Any] | None = None
    reconstruction_enqueue: dict[str, Any] | None = None
    failure_stage = 'stage_handoff_capture'
    try:
        with _listener._JobLeaseHeartbeat(capture_root=capture_root, owner=owner, token=token, lease_seconds=lease_seconds):
            staged_capture_root = (selected_staged if selected_staged is not None else
                _listener.stage_handoff_capture(handoff=handoff, storage_root=storage_root,
                                                storage_client=storage_client))
            raw_manifest = _listener._read_optional_json_object(staged_capture_root / 'raw' / 'manifest.json')
            website_capture = _listener.is_website_capture_manifest(raw_manifest)
            run_kwargs: dict[str, Any] = {'capture_root': str(staged_capture_root), 'provider': provider, 'run_evaluation_prep': run_evaluation_prep, 'resume_completed_stages': True}
            if website_capture:
                run_kwargs.update(pipeline_lane='qualification', run_evaluation_prep=False)
            if producer_delivery_key is not None:
                from .website_assessment_resume import admit_browser_preparation, safe_to_arm
                failure_stage = 'website_assessment_preparation'
                admit_browser_preparation(_listener, payload=payload, handoff=handoff,
                    capture_root=capture_root, observation=observation,
                    producer_delivery_key=producer_delivery_key, payload_digest=digest,
                    allow_resume=safe_to_arm(attempt_count, previous_history, ledger.get('recovered_expired_lease')))
            robot_eval_job_request = _listener._resolve_staged_handoff_path(handoff.robot_eval_job_request_uri, handoff=handoff, capture_root=staged_capture_root, storage_root=storage_root, expect_directory=False)
            robot_eval_request_inbox = _listener._resolve_staged_handoff_path(handoff.robot_eval_request_inbox_uri, handoff=handoff, capture_root=staged_capture_root, storage_root=storage_root, expect_directory=True)
            if robot_eval_job_request is not None:
                run_kwargs['robot_eval_job_request'] = str(robot_eval_job_request)
            if robot_eval_request_inbox is not None:
                run_kwargs['robot_eval_request_inbox'] = str(robot_eval_request_inbox)
            if robot_eval_job_request is not None or robot_eval_request_inbox is not None:
                run_kwargs.update({'robot_eval_job_id': handoff.robot_eval_job_id, 'robot_eval_provisioner': handoff.robot_eval_provisioner or 'fixture_local', 'robot_eval_simulator': handoff.robot_eval_simulator or 'fixture', 'robot_eval_evaluation_substrate': handoff.robot_eval_evaluation_substrate, 'robot_eval_budget_usd': handoff.robot_eval_budget_usd, 'allow_robot_eval_gpu_provisioning': False, 'allow_robot_eval_simulator_execution': False})
            if stage_control_plane and (not website_capture):
                failure_stage = 'control_plane_staging'
                if control_plane_manifest_path is None:
                    raise _listener.PipelineError('Pub/Sub handoff control-plane staging requires a manifest path.')
                control_plane_staging = _listener._stage_control_plane_input(handoff=handoff, capture_root=staged_capture_root, manifest_path=control_plane_manifest_path, work_dir=control_plane_work_dir, staged_inputs_path=control_plane_staged_inputs_path, overwrite=overwrite_control_plane_input)
                failure_stage = 'reconstruction_launch_enqueue'
                reconstruction_enqueue = _listener._enqueue_capture_reconstruction_if_configured(handoff=handoff, capture_root=staged_capture_root)
            failure_stage = 'run_e2e'
            result = run_e2e(**run_kwargs) if run_e2e_enabled or website_capture else {'status': 'skipped', 'reason': 'run_e2e_disabled_after_control_plane_staging'}
    except Exception as exc:
        if isinstance(exc, _listener.HandoffStagingCapacityError):
            return _listener.finish_staging_capacity_blocked(capture_root=capture_root, handoff=handoff, owner=owner, token=token, attempt_count=attempt_count, attempt_started_at=attempt_started_at, previous_history=previous_history, failure_stage=failure_stage, finish_job_lease=_listener._finish_job_lease)
        ending = _listener.authority_ending(exc)
        if ending is not None:
            operation, code = ending
            return _listener._finish_terminal_authority_ending(capture_root, handoff=handoff, owner=owner, token=token, operation=operation, code=code, error=exc, stage=failure_stage, attempt_count=attempt_count, attempt_started_at=attempt_started_at, previous_history=previous_history, payload_digest=digest, producer_delivery_key=producer_delivery_key)
        failed_at = _listener.utc_now_iso()
        failure_record = {'attempt_number': attempt_count, 'status': 'failed_retryable', 'stage': failure_stage, 'started_at': attempt_started_at, 'failed_at': failed_at, 'error_type': type(exc).__name__, 'error': str(exc)}
        from .website_assessment_resume import AssessmentPreparationPending
        resume_record = exc.resume_record if isinstance(exc, AssessmentPreparationPending) else None
        if resume_record is not None:
            failure_record['assessment_resume'] = resume_record
        _listener._finish_job_lease(capture_root, owner=owner, token=token, update={'status': 'failed_retryable', 'updated_at': failed_at, 'last_failed_at': failed_at, 'last_error_type': type(exc).__name__, 'last_error': str(exc), 'attempt_history': [*previous_history, failure_record], **({'assessment_resume': resume_record} if resume_record else {})})
        raise
    completed_at = _listener.utc_now_iso()
    control_plane_staging_status = str(control_plane_staging.get('status') or '') or None if control_plane_staging else None
    control_plane_staging_path = str((control_plane_staging.get('webapp_staging') or {}).get('target_path') or '') or None if control_plane_staging else None
    disposition, result_blockers = _listener._handoff_result_disposition(result)
    terminal_success = disposition == 'terminal_success'
    output_commit = _listener._write_output_commit(capture_root, scene_id=handoff.scene_id, capture_id=handoff.capture_id, attempt_count=attempt_count, result=result) if terminal_success else None
    completion_record = {'attempt_number': attempt_count, 'status': 'completed' if terminal_success else 'retryable_blocked', 'stage': 'run_e2e', 'started_at': attempt_started_at, 'completed_at': completed_at, 'run_e2e_status': str(result.get('status') or '') or None, 'queue_disposition': disposition, 'output_commit_status': 'committed' if output_commit else None, 'output_commit_path': _listener.JOB_OUTPUT_COMMIT_FILENAME if output_commit else None, 'blockers': result_blockers}
    if control_plane_staging:
        completion_record.update({'control_plane_staging_status': control_plane_staging_status, 'control_plane_staging_path': control_plane_staging_path})
    ledger_update = {'schema_version': _listener.JOB_LEDGER_SCHEMA_VERSION, 'status': 'completed' if terminal_success else 'retryable_blocked', 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'attempt_count': attempt_count, 'started_at': job_started_at, 'updated_at': completed_at, 'last_attempt_started_at': attempt_started_at, 'completed_at': completed_at, 'run_e2e_status': str(result.get('status') or '') or None, 'last_error_type': None if terminal_success else 'RetryableBlockedResult', 'last_error': None if terminal_success else ','.join(result_blockers), 'retry_blockers': result_blockers, 'queue_disposition': disposition, 'output_commit_status': 'committed' if output_commit else None, 'output_commit_path': _listener.JOB_OUTPUT_COMMIT_FILENAME if output_commit else None, 'output_result_sha256': output_commit.get('result_sha256') if output_commit else None, 'attempt_history': [*previous_history, completion_record]}
    if control_plane_staging:
        ledger_update.update({'control_plane_staging_status': control_plane_staging_status, 'control_plane_staging_path': control_plane_staging_path})
    _listener._finish_job_lease(capture_root, owner=owner, token=token, update=ledger_update)
    return {'schema_version': 'v1', 'status': 'processed' if terminal_success else 'retryable_blocked', 'queue_disposition': 'terminal_success' if terminal_success else 'retryable', 'blockers': result_blockers, 'bucket': handoff.bucket, 'scene_id': handoff.scene_id, 'capture_id': handoff.capture_id, 'capture_root': str(capture_root), 'run_e2e': result, 'control_plane_staging': control_plane_staging, 'reconstruction_enqueue': reconstruction_enqueue, 'output_commit': output_commit}


def _pull_and_process_body(_listener, /, *, subscription, storage_root, provider, max_messages, run_evaluation_prep, run_e2e_enabled, stage_control_plane, control_plane_manifest_path, control_plane_work_dir, control_plane_staged_inputs_path, overwrite_control_plane_input, ack_deadline_seconds, max_delivery_attempts):
    # This existing scheduled drain is also the durable delivery retry owner.
    # It reads committed ledgers; no provider processing is reopened by delivery.
    from .website_preparation_status import reconcile_preparation_wakeups
    try:
        reconcile_preparation_wakeups(storage_root, limit=1)
    except Exception:
        _listener.logger.debug("pubsub_handoff.preparation_delivery_retry_unavailable")
    from .website_assessment_resume import reconcile_waiting_assessments
    reconcile_waiting_assessments(_listener, storage_root=storage_root, limit=1, process_args={
        'provider': provider, 'run_evaluation_prep': run_evaluation_prep, 'run_e2e_enabled': run_e2e_enabled,
        'stage_control_plane': stage_control_plane, 'control_plane_manifest_path': control_plane_manifest_path,
        'control_plane_work_dir': control_plane_work_dir, 'control_plane_staged_inputs_path': control_plane_staged_inputs_path,
        'overwrite_control_plane_input': overwrite_control_plane_input})
    from google.cloud import pubsub_v1
    subscriber = pubsub_v1.SubscriberClient()
    subscription_resource = _listener._canonical_subscription_resource(subscription)
    acknowledged = 0

    def pulled_one_at_a_time() -> Iterator[Any]:
        for _ in range(max(1, max_messages)):
            response = subscriber.pull(request={'subscription': subscription_resource, 'max_messages': 1}, timeout=30)
            received_messages = list(response.received_messages)
            if not received_messages:
                return
            yield from received_messages

    def acknowledge(ack_id: str) -> None:
        subscriber.acknowledge(request={'subscription': subscription_resource, 'ack_ids': [ack_id]})
    for received in pulled_one_at_a_time():
        message = received.message
        _listener.logger.info('pubsub_handoff.received', extra={'message_id': message.message_id, 'attributes': dict(message.attributes)})
        try:
            _listener.parse_handoff_payload(message.data)
        except _listener.PipelineError as exc:
            evidence_path = _listener._write_delivery_evidence(storage_root=storage_root, message=message, received=received, disposition='permanent_invalid', blockers=[str(exc)])
            _listener.logger.error('pubsub_handoff.permanent_invalid', extra={'message_id': message.message_id, 'error_type': type(exc).__name__, 'error': str(exc), 'queue_disposition': 'permanent_invalid_ack', 'failure_evidence_path': str(evidence_path)})
            acknowledge(received.ack_id)
            acknowledged += 1
            continue
        digest = _listener.payload_sha256(message.data)
        heartbeat = _listener._AckDeadlineHeartbeat(subscriber=subscriber, subscription=subscription_resource, ack_id=received.ack_id, ack_deadline_seconds=ack_deadline_seconds)
        try:
            with heartbeat:
                result = _listener.process_handoff_payload(message.data, storage_root=storage_root, provider=provider, run_evaluation_prep=run_evaluation_prep, run_e2e_enabled=run_e2e_enabled, stage_control_plane=stage_control_plane, control_plane_manifest_path=control_plane_manifest_path, control_plane_work_dir=control_plane_work_dir, control_plane_staged_inputs_path=control_plane_staged_inputs_path, overwrite_control_plane_input=overwrite_control_plane_input, payload_digest=digest)
        except Exception:
            delivery_attempt = getattr(received, 'delivery_attempt', None)
            if isinstance(delivery_attempt, int) and delivery_attempt >= max_delivery_attempts:
                _listener._write_delivery_evidence(storage_root=storage_root, message=message, received=received, disposition='retry_exhausted_pending_pubsub_dlq', blockers=['handoff_processing_exception'])
            _listener.logger.exception('pubsub_handoff.processing_failed', extra={'message_id': message.message_id, 'queue_disposition': 'retryable_deferred', 'retry_defer_seconds': _listener.RETRY_DEFER_SECONDS, 'delivery_attempt': getattr(received, 'delivery_attempt', None), 'max_delivery_attempts': max_delivery_attempts})
            heartbeat.defer_retry()
            continue
        if result.get('queue_disposition') == 'retryable' or result.get('status') in _listener._JOB_RETRYABLE_STATUSES:
            delivery_attempt = getattr(received, 'delivery_attempt', None)
            if isinstance(delivery_attempt, int) and delivery_attempt >= max_delivery_attempts:
                _listener._write_delivery_evidence(storage_root=storage_root, message=message, received=received, disposition='retry_exhausted_pending_pubsub_dlq', blockers=[str(item) for item in result.get('blockers') or []])
            _listener.logger.warning('pubsub_handoff.retryable_result', extra={'message_id': message.message_id, 'status': result.get('status'), 'blockers': result.get('blockers') or [], 'delivery_attempt': getattr(received, 'delivery_attempt', None), 'max_delivery_attempts': max_delivery_attempts, 'dead_letter_policy_owns_exhausted_delivery': True, 'retry_defer_seconds': _listener.RETRY_DEFER_SECONDS})
            heartbeat.defer_retry()
            continue
        acknowledge(received.ack_id)
        acknowledged += 1
        _listener._record_acknowledgement(result, subscription=subscription_resource, message=message, received=received, payload_digest=digest)
    return acknowledged
