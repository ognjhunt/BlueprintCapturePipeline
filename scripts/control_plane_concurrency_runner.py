"""One real, isolated developmental scene chain for Plan 13c.

Only external rendering, configuration, policy and object transport are fixtures.
Admission, queue transitions, compilation, activation and delivery are original.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from blueprint_pipeline.task_evaluation_episode_compilation_worker import process_episode_compilation_queue
from scripts.control_plane_concurrency_load_test import AllocationSampler
from scripts.control_plane_concurrency_scene import (
    advance_fixture_intake, advance_fixture_preparation, advance_fixture_configuration, fixture_environment,
)
from scripts.control_plane_concurrency_activation import (
    advance_fixture_configuration_activation, advance_fixture_native_activation,
)
from scripts.control_plane_concurrency_readiness import materialize_fixture_placement, stage_fixture_episode_preparation
from scripts.control_plane_concurrency_policy import prepare_policy_fixture, ingest_policy_fixture, deliver_policy_fixture
from scripts.control_plane_concurrency_provider import fixture_publisher


def _expect(value, expected, boundary):
    if value != expected:
        raise ValueError('harness_stage_incomplete:' + boundary + ':' + str(value))


def run_scene(*, scene_key: str, control_root: Path, object_root: Path, worker_root: Path,
              source_commit: str, runtime_bundle: dict, release_binding: dict | None,
              heavy_slot, on_reserved=None, stage_recorder=None) -> dict:
    """Advance eight joined stages; the parent retires only after children join."""
    host = control_root / scene_key
    worker = worker_root / scene_key
    worker.mkdir(mode=0o700)
    reservations, pins = control_root / 'reservations', control_root / 'pins'
    rows = [] if stage_recorder is None else stage_recorder
    def measured(stage, function):
        wall, cpu = time.monotonic(), time.process_time()
        with AllocationSampler(control_root) as allocation:
            value = function()
        rows.append({'stage':stage,'status':'completed','wall_seconds':time.monotonic()-wall,
            'cpu_seconds':time.process_time()-cpu,'peak_allocated_bytes':allocation.peak_bytes,
            'allocation_measurement':'continuous_shared_tree_samples',
            'allocation_sample_count':allocation.sample_count,
            'allocation_incomplete_scan_count':allocation.incomplete_scan_count,
            'allocation_interval_seconds':allocation.interval_seconds,
            'allocated_delta_bytes':allocation.final_bytes-allocation.initial_bytes})
        return value
    def intake():
        value = advance_fixture_intake(host_root=host,object_root=object_root,
            source_commit=source_commit,scene_key=scene_key,release_binding=release_binding)
        _expect(value['progression']['results'][0]['status'],'running','intake')
        _expect(value['publication']['status'],'published_and_read_back','intake_publication')
        return value
    first=measured('scene_intake',intake)
    def prepare():
        with heavy_slot:
            value=advance_fixture_preparation(intake=first,object_root=object_root,
                reservation_root=reservations,pins_root=pins)
        _expect(value['run']['results'][0]['status'],'queued_for_production_scene_configuration','scene_preparation')
        return value
    preparation=measured('scene_preparation',prepare)
    activation=None
    def configure():
        nonlocal activation
        with heavy_slot:
            activation=advance_fixture_configuration_activation(intake=first,preparation=preparation,
                object_root=object_root,output_root=host/'configuration-activation',reservation_root=reservations)
            _expect(activation['worker']['results'][0]['status'],'profile_authority_materialized_no_execution','configuration_activation')
            _expect(activation['preparer']['status'],'prepared','configuration_graph')
            value=advance_fixture_configuration(preparation=preparation,object_root=object_root,
                output_root=host/'configured',provider_output_root=worker/'scene-provider')
        _expect(value['publication']['status'],'configured_scene_published','configuration_publication')
        _expect(value['terminal']['scene_construction_queue_finalization']['queue_state'],'completed','construction_queue')
        return value
    configured=measured('scene_configuration',configure)
    episode=None
    def launch_prepare():
        nonlocal episode
        with heavy_slot:
            placement=materialize_fixture_placement(configured=configured,object_root=object_root,output_root=host/'placement')
            from tests.test_task_evaluation_configured_controls_progression import _runtime
            runtime=_runtime()
            runtime['spend']['hard_cap_usd']=2.0
            runtime['execution_adapter']['runtime_source_bundle']=runtime_bundle['reference']
            runtime['runtime']['mounts'][0]['source']=configured['revision']['configured_scene_bundle']
            health=worker/'health.json';health.write_text('{"fixture_runtime":true}')
            health_reference=fixture_publisher(object_root)(path=health,object_name='health.json')
            runtime['runtime']['health_protocol']={key:health_reference[key] for key in ('uri','digest','size_bytes')}
            with fixture_environment(first['environment']):
                episode=stage_fixture_episode_preparation(configured=configured,placement=placement,
                    object_root=object_root,runtime_binding=runtime,output_root=host/'readiness',
                    queue_root=first['preparation_queue'],source_commit=source_commit)
            _expect(episode['status'],'episode_preparation_queued','readiness_queue')
            value=advance_fixture_preparation(intake=first,object_root=object_root,
                reservation_root=reservations,pins_root=pins)
        _expect(value['run']['results'][0]['status'],'queued_for_production_episode_compilation','episode_preparation')
        return value
    ready=measured('launch_preparation',launch_prepare)
    def compile_episode():
        with heavy_slot,fixture_environment(first['environment']):
            value=process_episode_compilation_queue(queue_root=ready['compilation_queue'],
                input_root=control_root/'prepared-references',output_root=control_root/'compiled',
                source_commit=source_commit,disk_reservation_root=reservations,storage_pins_root=pins)
        _expect(value['results'][0]['status'],'compiled_for_production_launch','compilation')
        return value['results'][0]
    compiled=measured('episode_compilation',compile_episode)
    def activate_native():
        with heavy_slot:
            value=advance_fixture_native_activation(intake=first,episode=episode,preparation=ready,
                compiled=compiled,configuration_activation=activation,object_root=object_root,
                output_root=host/'native-activation',reservation_root=reservations)
        _expect(value['worker']['results'][0]['status'],'profile_authority_materialized_no_execution','native_activation')
        _expect(value['preparer']['status'],'prepared','native_graph')
        return value
    native=measured('launch_activation',activate_native)
    provider=prepare_policy_fixture(compiled=compiled,preparation_request=ready['compilation_envelope']['request'],
        object_root=object_root,worker_root=worker/'policy')
    def ingest():
        value=ingest_policy_fixture(provider=provider,object_root=object_root,output_root=host/'policy-result',
            reservation_root=reservations,on_reserved=on_reserved)
        _expect(value['ingestion']['status'],'materialized','selected_member_ingestion')
        return value
    collected=measured('policy_output_ingestion',ingest)
    delivered=measured('result_delivery',lambda:deliver_policy_fixture(collected=collected))
    return {'scene_id':configured['revision']['scene_identity']['id'],'scene_key':scene_key,'stages':rows,
        'source_commit':source_commit,'claim_ceiling':'development_only','actual_provider_calls':0,
        'intake':first,'preparation':preparation,'episode_preparation':ready,'configured':configured,
        'compiled':compiled,'configuration_activation':activation,'native_activation':native,
        'collected':collected,'delivery':delivered,
        'provider_archive_bytes':provider['archive']['size_bytes'],
        'host_selected_transport_bytes':collected['host_transport_bytes'],
        'runtime_source_kind':runtime_bundle['source_kind']}
