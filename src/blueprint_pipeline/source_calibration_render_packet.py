"""Exact full-source calibrated-view variant of the existing GPU render probe."""
from __future__ import annotations

import json
import shutil
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .adp_retained_scene_render_packet import (
    DEFAULT_IMAGE, ENTRYPOINT, PROBE_KIND, _VENDOR_PACKAGES, _copy_tree,
    _git_identity, _link_or_copy, _record, _write_deterministic_zip,
)
from .decision_evidence_contracts import canonical_digest, canonical_json, cross_runtime_canonical_digest
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint, provider_bundle_rehearsal_blockers
from .sam31_contribution_disclosure import validate_full_source_disclosure
from .source_calibration_render_return import ROLES, require, validate_prepared_render_inputs
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha

BUNDLE_SCHEMA = "adp009d_source_calibration_gpu_render_bundle.v1"
RUNTIME_SCHEMA = "adp009d_source_calibration_gpu_renderer_runtime_request.v1"
PURPOSE = "exact_source_calibration_gpu_render"
MANIFEST_NAME = "adp_retained_scene_gpu_render_manifest.json"
RECEIPT_NAME = "adp_retained_scene_gpu_render_bundle_receipt.json"
MAX_SPEND_USD = 1.0
TTL_SECONDS = 1800
COUNTRIES = ("US",)
WESTERN_US_REGEX = "california|oregon|washington|nevada|arizona|utah|idaho|montana|wyoming|colorado|new mexico"


def _source_context(context: dict, commit: str, roots: Sequence[Path]) -> tuple[dict, dict]:
    paths = {}
    for name in ("prepared_inputs", "task_request", "conversion_receipt", "original_source"):
        row = context[name]
        path = Path(row["path"])
        require(path.is_absolute() and any(path.resolve().is_relative_to(root.resolve()) for root in roots), "input_outside_approved_roots")
        paths[name] = checked_file(path, row)
    from .public_scene_inpainting_preparation import validate_prepared_inputs
    prepared = validate_prepared_render_inputs(validate_prepared_inputs(paths["prepared_inputs"]))
    require(prepared["repository"]["commit"] == commit, "prepared_commit_mismatch")
    task = read(paths["task_request"])
    proof = validate_full_source_disclosure(
        task_authority=task.get("human_authority", {}), conversion_path=paths["conversion_receipt"],
        standard_splat_path=Path(prepared["layers"]["images"]["path"]),
        original_source_path=paths["original_source"], expected_source_commit=commit,
        publisher_scene_id=str(task.get("publisher_scene_id") or ""), approved_roots=roots, purpose=PURPOSE,
    )
    require(all(layer['sha256'] != sha(paths['original_source']) for layer in prepared['layers'].values()),
            'original_downloaded_bytes_must_remain_local')
    require(str(task.get("publisher_scene_id")) == str(prepared["context"]["source_identity"]["scene_id"]),
            "publisher_scene_mismatch")
    return prepared, proof


def build_source_calibration_gpu_render_bundle(
    *, prepared_inputs_path: str | Path, repo_root: str | Path, renderer_vendor_root: str | Path,
    task_request_path: str | Path, conversion_receipt_path: str | Path, original_source_path: str | Path,
    job_dir: str | Path, expected_source_commit: str, approved_roots: Sequence[Path],
) -> dict[str, Any]:
    context = {name: _record(Path(path)) for name, path in (
        ("prepared_inputs", prepared_inputs_path), ("task_request", task_request_path),
        ("conversion_receipt", conversion_receipt_path), ("original_source", original_source_path),
    )}
    # Disclosure is proven before creating a bundle directory, not only before allocation.
    prepared, proof = _source_context(context, expected_source_commit, approved_roots)
    task = read(task_request_path)
    owner = task.get("human_authority", {})
    require(owner.get("source_calibration_gpu_render_authorized") is True, "paid_render_authority_required")
    options = prepared['render_options']
    require(options.get('graphics_backend') == 'egl'
            and options.get('supersampling', 1) == 1
            and options.get('color_space', 'srgb') == 'srgb'
            and options.get('alpha_mode', 'opaque_rgb') == 'opaque_rgb'
            and options.get('background_rgb', 0) == 0
            and options.get('exposure_mode', 'renderer_default_unmodified') == 'renderer_default_unmodified',
            'unsupported_frozen_render_options')
    repo = Path(repo_root).resolve()
    identity = _git_identity(repo)
    require(identity["commit"] == expected_source_commit, "repository_commit_mismatch")
    vendor = Path(renderer_vendor_root).resolve()
    require(vendor.is_dir() and any(vendor.is_relative_to(root.resolve()) for root in approved_roots), "vendor_root_invalid")
    job = Path(job_dir)
    require(job.is_absolute() and not job.exists() and not any(p.is_symlink() for p in (job,*job.parents)), "job_not_fresh")
    runtime = job/'provider_runtime'
    runtime.mkdir(parents=True)
    layers = {}
    for role in ROLES:
        layer = prepared['layers'][role]
        target = runtime/'input'/f'{role}.ply'
        _link_or_copy(Path(layer['path']), target)
        layers[role] = {**_record(target, root=runtime), "gaussian_count": layer['retained_gaussian_count'],
                       **{key:layer[key] for key in ('camera_set_label','purpose','provider_splat_import_receipt_digest','alignment_digest')}}
    cameras = runtime/'input/cameras.v1.json'
    _link_or_copy(Path(prepared['camera_file']['path']), cameras)
    renderer = runtime/'renderer'
    for relative in ('render_splat.mjs','harness.html','package.json','package-lock.json','src/render_entry.mjs'):
        source = repo/'tools/splat_render'/relative
        require(source.is_file() and not source.is_symlink(), 'renderer_source_missing')
        _link_or_copy(source, renderer/relative)
    vendor_files = {package:_copy_tree(vendor/package, renderer/'node_modules'/package) for package in _VENDOR_PACKAGES}
    for name in (Path(ENTRYPOINT).name,'adp_retained_scene_render_provider_runner.mjs','adp_retained_scene_render_rehearsal.py'):
        target = runtime/name
        shutil.copy2(repo/'scripts'/name, target)
        if name.endswith('.sh'):
            target.chmod(target.stat().st_mode|stat.S_IXUSR)
    renderer_identity = {'repository':identity,'harness_sha256':sha(renderer/'render_splat.mjs'),
        'render_entry_sha256':sha(renderer/'src/render_entry.mjs'),'package_manifest_sha256':sha(renderer/'package.json'),
        'package_lock_sha256':sha(renderer/'package-lock.json'),'graphics_backend':'egl',
        'vendor_packages':{name:len(rows) for name,rows in vendor_files.items()}}
    options = prepared['render_options']
    require(options.get('graphics_backend') == 'egl', 'hardware_backend_not_frozen')
    execution_authority = {'schema_version':'source_calibration_gpu_execution_authority.v1',
        'authority_kind':'explicit_user_direction_in_current_goal','purpose':PURPOSE,
        'authorized_by':owner['accepted_by'],'authorized_on':owner['accepted_on'],
        'authority_reference':owner['authority_reference'],'blueprint_commit':expected_source_commit,
        'source_disclosure_proof_digest':proof['proof_digest'],
        'paid_compute':{'provider':'vast','external_instance_allowlist':[],'zero_retry':True,
                        'maximum_resource_count':1,'hard_total_spend_cap_usd':1.0,'hard_ttl_seconds':1800},
        'authority_digest':''}
    execution_authority['authority_digest']=canonical_digest(execution_authority,digest_field='authority_digest')
    authority_path=runtime/'source_calibration_execution_authority.json'
    authority_path.write_text(canonical_json(execution_authority)+'\n')
    request = {'schema_version':RUNTIME_SCHEMA,'render_scope':'source_calibration',
        'preparation_digest':prepared['preparation_digest'],'blueprint_commit':expected_source_commit,
        'layers':layers,'camera_contract':_record(cameras,root=runtime),'camera_count':16,
        'dimensions':{'width':1280,'height':1280},'render_options':options,'renderer_identity':renderer_identity,
        'source_disclosure_proof_digest':proof['proof_digest'],'expected_png_count':48,
        'candidate_policy_queried':False,'paid_inference_performed':False,'digest_canonicalization':'rfc8785','runtime_request_digest':''}
    from .source_calibration_camera_resolution import validate_recovery_contract
    recovery = validate_recovery_contract(prepared)
    if recovery is not None:
        replacement_path = runtime / 'input/replacement_cameras.json'
        _link_or_copy(Path(recovery['replacement_camera_file']['path']), replacement_path)
        shutil.copy2(repo / 'scripts/source_calibration_camera_recovery.mjs',
                     runtime / 'source_calibration_camera_recovery.mjs')
        request['camera_recovery'] = {
            'schema_version': recovery['schema_version'], 'maximum_rounds': 1,
            'replacement_camera_contract': _record(replacement_path, root=runtime),
            'visibility_gate': recovery['visibility_gate']}
    request['runtime_request_digest']=cross_runtime_canonical_digest(request,digest_field='runtime_request_digest')
    (runtime/'render_request.json').write_text(canonical_json(request)+'\n')
    # Exact allowlist inventory includes only renderer/code, the three appearance
    # PLYs and cameras. Original compressed PLY, labels, collision and terms stay local.
    inventory=[_record(path,root=runtime) for path in sorted(runtime.rglob('*')) if path.is_file()]
    manifest={'schema_version':BUNDLE_SCHEMA,'status':'ready','program_id':'arm-decision-proof-v1','adp_item':'ADP-009D',
        'probe_kind':PROBE_KIND,'container_image':DEFAULT_IMAGE,'blueprint_commit':expected_source_commit,
        'render_scope':'source_calibration','preparation_digest':prepared['preparation_digest'],
        'runtime_request_digest':request['runtime_request_digest'],'inventory':inventory,
        'renderer_identity':renderer_identity,'layers':layers,'camera_contract':request['camera_contract'],
        'expected_png_count':48,'hard_total_spend_cap_usd':MAX_SPEND_USD,'hard_ttl_seconds':TTL_SECONDS,
        'retry_cap':0,'maximum_resource_count':1,'allowed_geolocation_country_codes':list(COUNTRIES),
        'preferred_geolocation_regex':WESTERN_US_REGEX,'source_payload_kind':'full_source_scene_reencoded_standard_splat',
        'full_source_scene_content_upload_authorized':True,'provider_training_authorized':False,
        'public_redistribution_authorized':False,'provider_network_dependency_install_required':False,
        'automatic_paid_retry_allowed':False,'provider_zero_required_after_return':True,
        'manifest_digest':''}
    manifest['manifest_digest']=canonical_digest(manifest,digest_field='manifest_digest')
    (runtime/MANIFEST_NAME).write_text(canonical_json(manifest)+'\n')
    bundle=job/'adp_source_calibration_gpu_render_bundle.zip'
    _write_deterministic_zip(job,bundle)
    rehearsal=rehearse_provider_bundle_entrypoint(bundle_path=bundle,entrypoint_relative_path=ENTRYPOINT,
        evidence_path=job/'source_calibration_exact_bundle_rehearsal.json')
    receipt={**manifest,'bundle_path':str(bundle),'bundle_relative_path':bundle.name,'bundle_sha256':sha(bundle),
        'bundle_size_bytes':bundle.stat().st_size,'source_context':context,'source_disclosure':proof,
        'approved_roots':[str(root) for root in approved_roots],
        'execution_authority':{**_record(authority_path),'relative_path':authority_path.relative_to(job).as_posix(),
                               'authority_digest':execution_authority['authority_digest']},
        'request':context['prepared_inputs'],'exact_bundle_entrypoint_rehearsal':rehearsal}
    (job/RECEIPT_NAME).write_text(canonical_json(receipt)+'\n')
    return receipt


def validate_source_calibration_bundle(value: dict, *, expected_commit: str | None = None) -> dict:
    require(value.get('schema_version')==BUNDLE_SCHEMA and value.get('status')=='ready'
            and value.get('render_scope')=='source_calibration', 'bundle_schema_invalid')
    commit=str(value.get('blueprint_commit') or '')
    require(expected_commit is None or commit==expected_commit,'bundle_commit_mismatch')
    prepared,proof=_source_context(value['source_context'],commit,tuple(Path(p) for p in value['approved_roots']))
    owner=read(value['source_context']['task_request']['path']).get('human_authority', {})
    require(owner.get('source_calibration_gpu_render_authorized') is True, 'paid_render_authority_required')
    execution=read(checked_file(value['execution_authority']['path'], value['execution_authority']),digest_field='authority_digest')
    require(execution.get('schema_version')=='source_calibration_gpu_execution_authority.v1'
            and execution.get('source_disclosure_proof_digest')==proof['proof_digest']
            and execution.get('authority_digest')==value['execution_authority']['authority_digest'], 'execution_authority_changed')
    require(proof==value['source_disclosure'] and value['preparation_digest']==prepared['preparation_digest'], 'bundle_disclosure_changed')
    path=checked_file(value['bundle_path'],{'sha256':value['bundle_sha256'],'size_bytes':value['bundle_size_bytes']})
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('provider_runtime/'+MANIFEST_NAME))
        require(manifest.get('manifest_digest')==canonical_digest(manifest,digest_field='manifest_digest')
                and all(value.get(k)==v for k,v in manifest.items()),'bundle_manifest_changed')
        names={'provider_runtime/'+MANIFEST_NAME}
        for row in manifest['inventory']:
            name='provider_runtime/'+row['relative_path']
            names.add(name)
            data=archive.read(name)
            import hashlib
            require(len(data)==row['size_bytes'] and 'sha256:'+hashlib.sha256(data).hexdigest()==row['sha256'], 'bundle_inventory_changed')
        require(set(archive.namelist())==names,'bundle_extra_or_missing_file')
    require(all(value['layers'][role]['sha256']==prepared['layers'][role]['sha256']
                and value['layers'][role]['gaussian_count']==prepared['layers'][role]['retained_gaussian_count']
                for role in ROLES) and value['camera_contract']['sha256']==prepared['camera_file']['sha256'],
            'bundle_prepared_bytes_mismatch')
    require(value.get('container_image')==DEFAULT_IMAGE and value.get('probe_kind')==PROBE_KIND
            and value.get('hard_total_spend_cap_usd')==1.0 and value.get('hard_ttl_seconds')==1800
            and value.get('maximum_resource_count')==1 and value.get('retry_cap')==0
            and value.get('allowed_geolocation_country_codes')==['US']
            and value.get('preferred_geolocation_regex')==WESTERN_US_REGEX,'bundle_bounds_invalid')
    require(not provider_bundle_rehearsal_blockers(value.get('exact_bundle_entrypoint_rehearsal'),
        bundle_sha256=value['bundle_sha256'],entrypoint_relative_path=ENTRYPOINT),'bundle_rehearsal_invalid')
    return value
