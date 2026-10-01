"""Plan 13c joined developmental benchmark; no paid-provider execution authority."""
from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_host_input_intake import _verified_checkout_head
from scripts.control_plane_concurrency_load_test import (
    AllocationSampler, REQUIRED_STAGES, allocated_tree_bytes, argument_parser,
    build_summary, child_environment, create_run_roots,
)
from scripts.control_plane_concurrency_protocol import observe_overlap, seal_document


def require_kernel_confinement(*, process_root=Path('/proc')):
    """Require the real Linux sandbox before reading source inputs or writing roots."""
    try:
        status=(process_root/'self/status').read_text()
        isolated=os.readlink(process_root/'self/ns/net')!=os.readlink(process_root/'1/ns/net')
        protected=bool(os.statvfs('/').f_flag & os.ST_RDONLY)
        inaccessible=stat.S_IMODE(Path('/etc/blueprint').stat().st_mode)==0
        if 'NoNewPrivs:\t1' not in status or not isolated or not protected or not inaccessible:
            raise ValueError('harness_kernel_confinement_required')
        try:
            connection=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        except OSError:
            return {'private_network':True,'internet_address_family_denied':True,
                    'root_read_only':True,'live_configuration_inaccessible':True,'no_new_privileges':True}
        connection.close()
    except (OSError,ValueError):
        pass
    raise ValueError('harness_kernel_confinement_required')


def child_failure(children,*,now,child_timeout,global_deadline):
    if now>global_deadline:return 'global_timeout'
    for row in children:
        code=row['process'].poll()
        if code is not None and code!=0:return 'child_failed:'+row['scene_key']+':'+str(code)
        if code is None and now-row['started']>child_timeout:return 'child_timeout:'+row['scene_key']
    return None


def _wait_children(children,predicate,*,child_timeout,global_deadline):
    while not predicate():
        failure=child_failure(children,now=time.monotonic(),child_timeout=child_timeout,
                              global_deadline=global_deadline)
        if failure:raise ValueError(failure)
        time.sleep(0.05)


def _stop_owned_children(children):
    for row in children:
        if row['process'].poll() is None:row['process'].terminate()
    for row in children:
        try:row['process'].wait(timeout=10)
        except subprocess.TimeoutExpired:
            row['process'].kill();row['process'].wait(timeout=10)
        row['log'].close()


def _runtime_bundle(source_root,source_commit,roots):
    from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
        build_task_evaluation_runtime_source_bundle,external_layer_uri_prefix_for_bucket)
    from scripts.control_plane_concurrency_provider import fixture_publisher
    from tests.test_task_evaluation_configured_controls_progression import _runtime
    source_root=Path(source_root)
    packet=source_root/'native_task_runtime_source_packet.v1.json'
    if source_root.is_symlink() or not packet.is_file() or packet.is_symlink():
        raise ValueError('genuine_runtime_source_packet_required')
    wrapper=roots['workers']/'runtime-source.zip'
    prefix=external_layer_uri_prefix_for_bucket('blueprint')
    layer_root=roots['objects']/'blueprint'/prefix.split('s3://blueprint/',1)[1]
    built=build_task_evaluation_runtime_source_bundle(source_root=source_root,output_path=wrapper,
        expected_production_commit=source_commit,runtime_identity=_runtime()['runtime']['identity'],
        external_layer_store_root=layer_root,external_layer_bucket='blueprint')
    published=fixture_publisher(roots['objects'])(path=wrapper,object_name='runtime-source.zip')
    return {'reference':{key:published[key] for key in ('uri','digest','size_bytes')},
            'source_kind':'genuine_owner_source_packet','build_receipt':built,
            'original_source_packet_digest':'sha256:'+__import__('hashlib').sha256(packet.read_bytes()).hexdigest()}


def run_benchmark(args):
    confinement=require_kernel_confinement()
    repo=Path(__file__).resolve().parents[1]
    source=_verified_checkout_head()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']):
        raise ValueError('harness_checkout_dirty')
    report=Path(args.report)
    roots_paths=[Path(args.control_plane_root),Path(args.object_store_root),Path(args.worker_root)]
    if (report.exists() or report.is_symlink() or not report.is_absolute() or '..' in report.parts
            or any(p.parent!=report.parent for p in roots_paths)
            or any(p.is_symlink() for p in (report.parent,*report.parent.parents))
            or report.parent.stat().st_uid!=os.geteuid()
            or stat.S_IMODE(report.parent.stat().st_mode)!=0o700):
        raise ValueError('harness_report_parent_not_owned')
    release=json.loads(Path(args.release_binding).read_text())
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import checked_file
    if (release.get('source_commit')!=source or
            release.get('release_digest')!=canonical_digest(release,digest_field='release_digest')):
        raise ValueError('harness_release_binding_mismatch')
    for key in ('deploy_receipt','release_provenance','release_environment'):
        checked_file(release[key]['path'],release[key])
    roots=create_run_roots(*roots_paths)
    baseline=allocated_tree_bytes(roots['control_plane'])
    children=[];scenes=[];peak=0;held=0.0;blockers=[];retirement=None
    started=time.monotonic();deadline=started+args.global_timeout_seconds
    sampler=AllocationSampler(roots['control_plane'])
    sampler.__enter__()
    try:
        runtime=_runtime_bundle(args.runtime_source_root,source,roots)
        barriers=roots['workers']/'barriers';barriers.mkdir(mode=0o700)
        ready=barriers/'ready';ready.mkdir(mode=0o700)
        startup=barriers/'startup';startup.mkdir(mode=0o700)
        heavy=roots['workers']/'heavy.lock';heavy.touch(mode=0o600)
        for index in range(2*args.expected_beta_concurrency):
            key='scene-'+str(index+1)
            temporary=roots['workers']/('tmp-'+key);temporary.mkdir(mode=0o700)
            result=roots['workers']/(key+'-result.json')
            spec=roots['workers']/(key+'-spec.json')
            seal_document(spec,{'scene_key':key,'source_commit':source,'runtime_bundle':runtime,
                'release_binding':release,'control_root':roots['control_plane'],'object_root':roots['objects'],
                'worker_root':roots['workers'],'temporary_root':temporary,'startup_ready_root':startup,
                'start_barrier':barriers/'start.json','release_barrier':barriers/'release.json',
                'ready_root':ready,'heavy_lock':heavy,'result_path':result,
                'child_timeout_seconds':args.child_timeout_seconds})
            log=(roots['workers']/(key+'.log')).open('xb')
            process=subprocess.Popen([sys.executable,'-m','scripts.control_plane_concurrency_protocol',
                '--child-spec',str(spec)],cwd=repo,env=child_environment({**os.environ,'PYTHONPATH':str(repo/'src')+':'+str(repo)}),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children.append({'scene_key':key,'process':process,'log':log,'result_path':result,
                             'started':time.monotonic()})
        registry=seal_document(roots['control_plane']/'producer-registry.json',{
            'schema_version':'concurrency_producer_registry.v1','source_commit':source,
            'producers':[{'scene_key':row['scene_key'],'pid':row['process'].pid} for row in children]})
        _wait_children(children,lambda:all((startup/(row['scene_key']+'.json')).is_file() for row in children),
                       child_timeout=args.child_timeout_seconds,global_deadline=deadline)
        seal_document(barriers/'start.json',{'status':'start','registry_digest':registry['receipt_digest']})
        def observed():
            nonlocal peak
            count=observe_overlap(children=[{'scene_key':row['scene_key'],'pid':row['process'].pid,
                'running':row['process'].poll() is None} for row in children],
                ready_root=ready,reservation_root=roots['control_plane']/'reservations')
            peak=max(peak,count)
            return count==len(children)
        _wait_children(children,observed,child_timeout=args.child_timeout_seconds,global_deadline=deadline)
        hold_started=time.monotonic()
        while time.monotonic()-hold_started<2:
            if not observed():raise ValueError('overlap_milestone_lost')
            failure=child_failure(children,now=time.monotonic(),child_timeout=args.child_timeout_seconds,global_deadline=deadline)
            if failure:raise ValueError(failure)
            time.sleep(0.05)
        held=time.monotonic()-hold_started
        seal_document(barriers/'release.json',{'status':'release','observed_concurrency':peak,'hold_seconds':held})
        _wait_children(children,lambda:all(row['process'].poll() is not None for row in children),
                       child_timeout=args.child_timeout_seconds,global_deadline=deadline)
        joins=[]
        for row in children:
            code=row['process'].wait()
            if code!=0:raise ValueError('child_failed:'+row['scene_key']+':'+str(code))
            scene=json.loads(row['result_path'].read_text())
            if scene.get('receipt_digest')!=canonical_digest(scene,digest_field='receipt_digest'):
                raise ValueError('child_receipt_changed')
            scenes.append(scene)
            joins.append({'scene_key':row['scene_key'],'pid':row['process'].pid,'exit_code':code,
                          'scene_result_digest':canonical_digest(scene)})
        seal_document(roots['control_plane']/'producer-joins.json',{'schema_version':'concurrency_producer_joins.v1',
            'source_commit':source,'registry_digest':registry['receipt_digest'],'joins':joins})
        from scripts.control_plane_concurrency_retirement import retire_fixture_chains
        wall,cpu=time.monotonic(),time.process_time()
        with AllocationSampler(roots['control_plane']) as allocation:
            retirement=retire_fixture_chains(control_root=roots['control_plane'],scenes=scenes,
                source_commit=source,producer_pids=[row['process'].pid for row in children])
        seal_document(roots['workers']/'retirement.json',retirement)
        for scene in scenes:
            scene['stages'].append({'stage':'retirement','status':'completed','wall_seconds':time.monotonic()-wall,
                'cpu_seconds':time.process_time()-cpu,'peak_allocated_bytes':allocation.peak_bytes,
                'allocation_measurement':'continuous_shared_tree_samples','allocation_sample_count':allocation.sample_count})
            scene.update(residual_pins=retirement['residual_pins'],residual_leases=retirement['residual_leases'])
    except Exception as exc:
        blockers=getattr(exc,'errors',None) or [type(exc).__name__+':'+str(exc)]
    finally:
        _stop_owned_children(children)
        try:sampler.__exit__(None,None,None)
        except Exception as exc:blockers.append(type(exc).__name__+':'+str(exc))
    if blockers:
        scenes=[json.loads(row['result_path'].read_text()) for row in children if row['result_path'].is_file()]
    final=allocated_tree_bytes(roots['control_plane'])
    summary=build_summary(source_commit=source,expected_beta_concurrency=args.expected_beta_concurrency,
        owner_confirmed=args.owner_confirmed_concurrency,scenes=scenes,measured_peak_concurrency=peak,
        concurrency_hold_seconds=held,baseline_allocated_bytes=baseline,final_allocated_bytes=final,
        maximum_retained_bytes=int(args.maximum_retained_gib*1024**3),
        maximum_delta_bytes=int(args.maximum_delta_gib*1024**3),external_calls=sum(s.get('actual_provider_calls',0) for s in scenes))
    summary.update(kernel_confinement=confinement,retirement=retirement,peak_allocated_bytes=sampler.peak_bytes,
        allocation_sample_count=sampler.sample_count,allocation_incomplete_scan_count=sampler.incomplete_scan_count,
        filesystem_and_allocation_samples=sampler.samples,allocation_interval_seconds=sampler.interval_seconds,
        fixture_object_allocated_bytes=allocated_tree_bytes(roots['objects']),worker_allocated_bytes=allocated_tree_bytes(roots['workers']),
        maximum_heavy_stage_parallelism_configured=1,concurrency_boundary='positive policy-output storage reservations',
        automatic_terminal_reconciliation_proven=False,normal_six_hour_retention_proven=False)
    summary['blockers']=sorted(set(summary['blockers']+blockers))
    summary['status']='failed' if summary['blockers'] else 'passed'
    summary['owner_sized_acceptance_complete']=args.owner_confirmed_concurrency and summary['status']=='passed'
    seal_document(report,summary)
    return 0 if summary['status']=='passed' else 1


def main(argv=None):
    parser=argument_parser()
    parser.add_argument('--runtime-source-root',type=Path,required=True)
    parser.add_argument('--release-binding',type=Path,required=True)
    return run_benchmark(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
