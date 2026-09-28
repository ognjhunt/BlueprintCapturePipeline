"""Policy-free native placement feasibility for explicitly adopted rigid assets.

Run a fixed hold sequence, retain initial/final frames and per-step native state,
then restore the same seed's frozen spawn. A passing probe does not claim that
policy episodes start settled, qualify controls, or establish physical truth.
The caller must attach a real physics-error observer before building the arena.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest

SPEC_SCHEMA = 'task_object_native_settle_spec.v1'
RESULT_SCHEMA = 'task_object_native_settle_gate.v1'
RESULT_FILENAME = RESULT_SCHEMA + '.json'
_ROLES = ('task_object', 'task_support')
_DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
_CONTACT = {'task_object': 'task_initial_support_contact_peak_force_n',
            'task_support': 'destination_scene_support_contact_peak_force_n'}
_FORBIDDEN = {'task_object': 'task_scene_collision_peak_force_n',
              'task_support': 'destination_scene_forbidden_contact_peak_force_n'}
_LIMITS = ('maximum_penetration_m', 'minimum_support_contact_force_n',
           'maximum_forbidden_contact_force_n', 'settle_translation_tolerance_m',
           'settle_rotation_tolerance_rad', 'reset_translation_tolerance_m', 'reset_rotation_tolerance_rad')


class NativeSettleGateError(ValueError):
    pass


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NativeSettleGateError('native_settle_nonfinite_or_missing_number')
    return float(value)


def _vector(value: Any, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise NativeSettleGateError('native_settle_vector_invalid')
    return [_number(v) for v in value]


def validate_native_settle_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed two-object probe contract without importing Isaac."""
    value = dict(spec)
    if (value.get('schema_version') != SPEC_SCHEMA or value.get('required') is not True
            or value.get('spec_digest') != canonical_digest(value, digest_field='spec_digest')
            or any(not _DIGEST.fullmatch(str(value.get(key, ''))) for key in
                   ('source_packet_digest', 'scene_collision_digest', 'support_measurement_digest'))):
        raise NativeSettleGateError('native_settle_spec_invalid')
    seconds, hz, physics = [_number(value.get(key)) for key in ('gravity_settle_seconds', 'control_hz', 'physics_hz')]
    window = value.get('settle_window_samples')
    steps = math.ceil(seconds * hz)
    if (seconds <= 0 or hz <= 0 or physics < hz or abs(physics/hz-round(physics/hz)) > 1e-9
            or isinstance(window, bool) or not isinstance(window, int) or not 2 <= window <= steps <= 900):
        raise NativeSettleGateError('native_settle_cadence_invalid')
    limits = value.get('qualification_limits', {})
    if not isinstance(limits, Mapping) or any(_number(limits.get(key)) <= 0 for key in _LIMITS):
        raise NativeSettleGateError('native_settle_limits_invalid')
    if 'support_height_tolerance_m' in limits and _number(limits['support_height_tolerance_m']) <= 0:
        raise NativeSettleGateError('native_settle_limits_invalid')
    rows = value.get('objects', [])
    if not isinstance(rows, list) or len(rows) != 2 or {row.get('role') for row in rows} != set(_ROLES):
        raise NativeSettleGateError('native_settle_exact_two_assets_required')
    for row in rows:
        if (not row.get('asset_id') or not row.get('asset_version')
                or not _DIGEST.fullmatch(str(row.get('asset_sha256', '')))):
            raise NativeSettleGateError('native_settle_asset_identity_invalid')
        bounds = row.get('collision_bounds_body_frame_m', {})
        low, high = _vector(bounds.get('minimum'), 3), _vector(bounds.get('maximum'), 3)
        if any(a >= b for a, b in zip(low, high, strict=True)):
            raise NativeSettleGateError('native_settle_collision_bounds_invalid')
        _number(row.get('support_top_z_m'))
        source, native = row.get('support_source_prim_paths'), row.get('support_native_prim_paths')
        if (not isinstance(source, list) or not source or not isinstance(native, list) or len(source) != len(native)
                or len(set(native)) != len(native) or any(not str(path).startswith('/Root/') for path in source)
                or native != ['{ENV_REGEX_NS}/scene_collision' + path.removeprefix('/Root') for path in source]
                or any(any(token in path for token in ('*', '[', ']', '..')) for path in native)):
            raise NativeSettleGateError('native_settle_exact_support_paths_required')
    return value


@dataclass(frozen=True)
class NativeSettleAdapter:
    reset: Callable[[int], Any]
    hold_step: Callable[[], Any]
    read_sample: Callable[[], Mapping[str, Any]]
    capture_frames: Callable[[str, Path], Mapping[str, Any]]
    read_cooking_errors: Callable[[], Mapping[str, Any]]


def make_native_settle_adapter(*, built: Any, capture_frames: Callable,
                               read_cooking_errors: Callable) -> NativeSettleAdapter:
    """Bind existing Arena rigid readback and the destination probe's hold action."""
    from .native_task_arena_readback import NativeRigidTaskArenaReadback, _first_environment
    native = getattr(built.env, 'unwrapped', built.env)
    scene, reader = native.scene, NativeRigidTaskArenaReadback(built)
    native_reset = built.env.reset
    def first(value):
        return _first_environment(getattr(value, 'torch', value), error='native_settle_native_state_missing')
    def normalize(path):
        for prefix in [getattr(scene, 'env_regex_ns', ''), *getattr(scene, 'env_prim_paths', [])]:
            if prefix and (path == prefix or path.startswith(prefix + '/')):
                return '{ENV_REGEX_NS}' + path[len(prefix):]
        return path
    plan_rows = {row['semantic_role']: row for row in built.plan['objects'] if row.get('semantic_role') in _ROLES}
    sensor_rows = {row['sensor_instance_id']: row for row in built.plan['articulation']['contact_sensors']}
    def sample():
        value = dict(reader.read_task_sample())
        value.update(asset_bindings={}, root_linear_velocity_m_s={}, root_angular_velocity_rad_s={}, contact_bindings={})
        collision_rows = [row for row in built.plan['objects'] if row.get('semantic_role') == 'scene_collision']
        if len(collision_rows) != 1:
            raise NativeSettleGateError('native_settle_scene_collision_binding_missing')
        value['scene_collision_digest'] = collision_rows[0]['sha256']
        for role, logical in [('task_object', 'task_initial_support_contact'), ('task_support', 'destination_scene_support_contact')]:
            asset, bound = scene[built.scene_asset_names[role]], plan_rows[role]
            if bound['object_type'] != 'RIGID':
                raise NativeSettleGateError('native_settle_dynamic_rigid_asset_required')
            value['asset_bindings'][role] = {'asset_id': bound['asset_id'], 'asset_sha256': bound['sha256']}
            value['root_linear_velocity_m_s'][role] = first(asset.data.root_lin_vel_w)
            value['root_angular_velocity_rad_s'][role] = first(asset.data.root_ang_vel_w)
            paths = set()
            names = built.contact_sensor_names.get(logical)
            if not names or isinstance(names, str):
                raise NativeSettleGateError('native_settle_support_sensor_missing:' + role)
            for name in names:
                cfg, expected = scene[name].cfg, sensor_rows[name]
                actual_paths = [normalize(str(path)) for path in cfg.filter_prim_paths_expr]
                if (normalize(str(cfg.prim_path)) != expected['prim_path']
                        or actual_paths != expected['filter_prim_paths_expr']):
                    raise NativeSettleGateError('native_settle_sensor_body_binding_mismatch:' + role)
                paths.update(actual_paths)
            value['contact_bindings'][role] = sorted(paths)
        value['step_dt_s'], value['physics_dt_s'] = float(native.step_dt), float(native.physics_dt)
        return value
    def hold():
        import torch
        # Identical to the existing policy-free destination qualification worker.
        current = first(scene['robot'].data.joint_pos)[:7]
        built.env.step(torch.tensor([[*[float(v) for v in current], 0.0]], device=native.device, dtype=torch.float32))
    return NativeSettleAdapter(lambda seed: native_reset(seed=seed), hold, sample, capture_frames, read_cooking_errors)


def _pose(raw: Any) -> list[float]:
    pose = _vector(raw, 7)
    norm = math.sqrt(sum(v*v for v in pose[3:]))
    if norm <= 0:
        raise NativeSettleGateError('native_settle_quaternion_invalid')
    return [*pose[:3], *[v/norm for v in pose[3:]]]


def _pose_distance(a, b):
    dot = abs(sum(x*y for x, y in zip(a[3:], b[3:], strict=True)))
    return math.dist(a[:3], b[:3]), 2 * math.acos(min(1., dot))


def _read_sample(adapter, spec):
    from .native_task_arena_readback import _quaternion_rotate_xyzw
    raw = adapter.read_sample()
    if raw.get('scene_collision_digest') != spec['scene_collision_digest']:
        raise NativeSettleGateError('native_settle_scene_collision_binding_mismatch')
    if raw.get('measurement_authority') != 'native_rigid_root_pose_and_filtered_contact_sensors':
        raise NativeSettleGateError('native_settle_measurement_authority_invalid')
    if (not math.isclose(_number(raw.get('step_dt_s')), 1/spec['control_hz'], rel_tol=1e-6)
            or not math.isclose(_number(raw.get('physics_dt_s')), 1/spec['physics_hz'], rel_tol=1e-6)):
        raise NativeSettleGateError('native_settle_observed_cadence_mismatch')
    rows = {}
    for body in spec['objects']:
        role = body['role']
        if (raw.get('asset_bindings', {}).get(role) != {key: body[key] for key in ('asset_id', 'asset_sha256')}
                or raw.get('contact_bindings', {}).get(role) != sorted(body['support_native_prim_paths'])):
            raise NativeSettleGateError('native_settle_asset_or_support_binding_mismatch:' + role)
        pose = _pose(raw.get('asset_root_pose_world' if role == 'task_object' else 'destination_pose_world'))
        bounds = body['collision_bounds_body_frame_m']
        corners = [_quaternion_rotate_xyzw(pose[3:], [x,y,z]) for x in (bounds['minimum'][0],bounds['maximum'][0])
                   for y in (bounds['minimum'][1],bounds['maximum'][1]) for z in (bounds['minimum'][2],bounds['maximum'][2])]
        gap = min(point[2] + pose[2] for point in corners) - body['support_top_z_m']
        contact, forbidden = _number(raw.get(_CONTACT[role])), _number(raw.get(_FORBIDDEN[role]))
        if contact < 0 or forbidden < 0:
            raise NativeSettleGateError('native_settle_contact_force_invalid')
        rows[role] = {'pose_world': pose, 'maximum_penetration_m': max(0., -gap), 'support_gap_m': gap,
                      'support_contact_force_n': contact, 'forbidden_contact_force_n': forbidden,
                      'linear_speed_m_s': math.sqrt(sum(v*v for v in _vector(raw.get('root_linear_velocity_m_s', {}).get(role), 3))),
                      'angular_speed_rad_s': math.sqrt(sum(v*v for v in _vector(raw.get('root_angular_velocity_rad_s', {}).get(role), 3)))}
    return rows


def _capture(adapter, label, root):
    destination = root / label
    error, snapshot = None, None
    try:
        destination.mkdir()
        snapshot = dict(adapter.capture_frames(label, destination))
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
    frames = []
    for path in sorted(destination.rglob('*.png')) if destination.exists() else []:
        if path.is_symlink() or not path.is_file():
            continue
        body = path.read_bytes()
        if body.startswith(b'\x89PNG\r\n\x1a\n'):
            frames.append({'path': str(path.relative_to(root)), 'sha256': 'sha256:' + hashlib.sha256(body).hexdigest(), 'size_bytes': len(body)})
    return {'status': 'captured' if frames and error is None else 'media_gap', 'frames': frames,
            'snapshot': snapshot, 'error': error}


def run_native_settle_gate(*, spec: Mapping[str, Any], adapter: NativeSettleAdapter, output_root: Path,
                           seed: int, scene_plan_digest: str, adoption_digest: str,
                           reset: bool = True, reference_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Probe fixed settling, capture failures, then verify the original spawn is restored."""
    spec = validate_native_settle_spec(spec)
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not _DIGEST.fullmatch(scene_plan_digest) or not _DIGEST.fullmatch(adoption_digest)):
        raise NativeSettleGateError('native_settle_runtime_binding_invalid')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / RESULT_FILENAME).exists():
        raise NativeSettleGateError('native_settle_receipt_already_exists')
    limits = spec['qualification_limits']
    steps = math.ceil(spec['gravity_settle_seconds'] * spec['control_hz'])
    duration = (spec['settle_window_samples'] - 1) / spec['control_hz']
    linear_limit = limits['settle_translation_tolerance_m'] / duration
    angular_limit = limits['settle_rotation_tolerance_rad'] / duration
    blockers, samples, frames, initial, restored = [], [], {}, None, None
    observed_contacts = {role: False for role in _ROLES}
    monitor_observation = {'monitoring_active': False, 'errors': []}
    restore_noop = False
    def cooking():
        state = adapter.read_cooking_errors()
        if not isinstance(state, Mapping) or state.get('monitoring_active') is not True or not isinstance(state.get('errors'), list):
            raise NativeSettleGateError('native_settle_physics_error_monitor_missing')
        monitor_observation.update(monitoring_active=True, errors=[str(error)[:1200] for error in state['errors']])
        if state['errors']:
            raise NativeSettleGateError('native_settle_collider_or_physics_error:' + str(state['errors'])[:1200])
    def parity(reference, observed):
        return all(_pose_distance(reference[role]['pose_world'], observed[role]['pose_world'])[0] <= limits['reset_translation_tolerance_m']
                   and _pose_distance(reference[role]['pose_world'], observed[role]['pose_world'])[1] <= limits['reset_rotation_tolerance_rad']
                   for role in _ROLES)
    try:
        if reset:
            adapter.reset(seed)
        frames['before'] = _capture(adapter, 'before', root)
        cooking()
        initial = _read_sample(adapter, spec)
        if reference_receipt is not None:
            if (reference_receipt.get('gate_digest') != canonical_digest(reference_receipt, digest_field='gate_digest')
                    or reference_receipt.get('seed') != seed or reference_receipt.get('spec_digest') != spec['spec_digest']
                    or not parity(reference_receipt['initial_state'], initial)):
                raise NativeSettleGateError('native_settle_candidate_reset_parity_mismatch')
        cooking()
        for index in range(steps):
            adapter.hold_step()
            state = _read_sample(adapter, spec)
            samples.append({'step_index': index + 1, 'objects': state})
            cooking()
            if any(row['maximum_penetration_m'] > limits['maximum_penetration_m']
                   or row['forbidden_contact_force_n'] > limits['maximum_forbidden_contact_force_n'] for row in state.values()):
                raise NativeSettleGateError('native_settle_penetration_tunneling_or_forbidden_contact')
        tail = samples[-spec['settle_window_samples']:]
        for role in _ROLES:
            rows = [sample['objects'][role] for sample in tail]
            gap_limit = limits.get('support_height_tolerance_m', limits['settle_translation_tolerance_m'])
            # Sleeping PhysX bodies can stop publishing forces. Require a real
            # near-plane filtered contact event, plus a stable near-plane tail.
            observed_contacts[role] = any(
                sample['objects'][role]['support_contact_force_n'] >= limits['minimum_support_contact_force_n']
                and abs(sample['objects'][role]['support_gap_m']) <= gap_limit for sample in samples)
            if not observed_contacts[role] or any(row['support_gap_m'] > gap_limit for row in rows):
                blockers.append('native_settle_support_contact_missing:' + role)
            if any(row['linear_speed_m_s'] > linear_limit or row['angular_speed_rad_s'] > angular_limit for row in rows):
                blockers.append('native_settle_velocity_not_stable:' + role)
            if any(_pose_distance(a['pose_world'], b['pose_world'])[0] > limits['settle_translation_tolerance_m']
                   or _pose_distance(a['pose_world'], b['pose_world'])[1] > limits['settle_rotation_tolerance_rad'] for a in rows for b in rows):
                blockers.append('native_settle_pose_not_stable:' + role)
    except Exception as exc:
        blockers.append(str(exc)[:2000] if isinstance(exc, NativeSettleGateError) else 'native_settle_runtime_exception:' + type(exc).__name__ + ':' + str(exc)[:1000])
    finally:
        if 'before' not in frames:
            frames['before'] = _capture(adapter, 'before', root)
        frames['after'] = _capture(adapter, 'after', root)
        try:
            adapter.reset(seed)
            restored = _read_sample(adapter, spec)
            if initial is not None and samples:
                restore_noop = any(initial[role]['pose_world'] != samples[-1]['objects'][role]['pose_world']
                    and restored[role]['pose_world'] == samples[-1]['objects'][role]['pose_world'] for role in _ROLES)
                if restore_noop:
                    blockers.append('native_settle_original_spawn_restore_noop')
            if initial is None or not parity(initial, restored):
                blockers.append('native_settle_original_spawn_restore_mismatch')
            cooking()
        except Exception as exc:
            blockers.append('native_settle_original_spawn_restore_failed:' + type(exc).__name__ + ':' + str(exc)[:1000])
    if any(frame['status'] != 'captured' for frame in frames.values()):
        blockers.append('native_settle_required_media_gap')
    result = {'schema_version': RESULT_SCHEMA, 'status': 'blocked' if blockers else 'passed',
              'spec_digest': spec['spec_digest'], 'scene_plan_digest': scene_plan_digest, 'adoption_digest': adoption_digest,
              'reference_gate_digest': reference_receipt.get('gate_digest') if reference_receipt else None,
              'candidate_reset_parity_checked': reference_receipt is not None,
              'seed': seed, 'fixed_hold_steps': steps, 'completed_hold_steps': len(samples), 'samples': samples,
              'initial_state': initial, 'restored_state': restored,
              'reset_state_digest': canonical_digest({'seed': seed, 'state': initial}),
              'frames': frames, 'qualification_limits': limits, 'physics_error_observer': monitor_observation,
              'observed_support_contacts': observed_contacts, 'continuous_support_contact_proven': False,
              'velocity_limits': {'linear_m_s': linear_limit, 'angular_rad_s': angular_limit,
                                  'basis': 'retained_settle_tolerances_over_observation_window_duration'},
              'original_spawn_restored': initial is not None and restored is not None and parity(initial, restored) and not restore_noop,
              'policy_start_already_settled': False, 'probe_only': True,
              'candidate_policy_loaded': False, 'candidate_policy_queried': False, 'controls_qualified': False,
              'physical_truth_claimed': False, 'continuous_substep_tunneling_qualified': False,
              'blockers': sorted(set(blockers))}
    result['gate_digest'] = canonical_digest(result, digest_field='gate_digest')
    (root / RESULT_FILENAME).write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return result
